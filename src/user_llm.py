
from __future__ import annotations

from collections import Counter
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
import pandas as pd

from src.config import Settings, normalize_llm_response_language
from src.game_card_builder import GameCard
from src.llm_provider import (
    build_gigachat_token_preflight_report,
    check_provider_readiness,
    generate_llm_json_response,
    get_effective_llm_model,
    get_effective_llm_provider,
    provider_credentials_configured,
)
from src.user_experiments import get_active_user_splits_path
from src.utils import get_logger, read_jsonl, truncate_text, write_jsonl


PROMPT_TOO_LONG_THRESHOLD = 12_000


class UserLLMRecommendationItem(BaseModel):

    rank: int
    game_id: str
    game_title: str
    relevance_score: float = 0.0
    explanation: str = ""
    matched_preferences: list[str] = Field(default_factory=list)
    possible_risks: list[str] = Field(default_factory=list)


class UserLLMResponse(BaseModel):

    recommendations: list[UserLLMRecommendationItem] = Field(default_factory=list)


def normalize_llm_mode(value: object) -> str:

    normalized = str(value or "real").strip().lower() or "real"
    if normalized not in {"real", "mock"}:
        raise ValueError("LLM_MODE must be either 'real' or 'mock'.")
    return normalized


def infer_token_obtained_from_error(
    *,
    provider: str,
    error_message: str,
    response_received: bool,
    http_status: int | None = None,
) -> bool:

    message = str(error_message or "").lower()
    token_error_markers = {
        "token request failed",
        "missing credentials",
        "missing auth",
        "authorization",
        "auth",
        "ssl",
        "certificate",
        "ca bundle",
    }
    if any(marker in message for marker in token_error_markers):
        return False
    if response_received:
        return True
    if http_status is not None and int(http_status) > 0:
        return True
    return provider != "gigachat"


def get_llm_response_language(settings: Settings) -> str:

    return normalize_llm_response_language(getattr(settings, "llm_response_language", "ru"))


def get_llm_provider_name(settings: Settings) -> str:

    return get_effective_llm_provider(settings)


def deduplicate_preserve_order(values: list[str]) -> list[str]:

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def run_user_llm_pilot(settings: Settings) -> list[dict[str, object]]:

    llm_mode = normalize_llm_mode(getattr(settings, "llm_mode", "real"))
    llm_provider = get_llm_provider_name(settings)
    llm_model = get_effective_llm_model(settings)
    if llm_mode == "mock":
        return run_user_llm_mock_pilot(settings)

    logger = get_logger()
    if not settings.active_processed_reviews_path.exists():
        logger.warning(
            "Active processed reviews file not found at %s.",
            settings.active_processed_reviews_path,
        )
        return []

    context = load_user_llm_context(settings)
    selected_users = select_pilot_users(settings, context["splits"])
    if not selected_users:
        logger.warning("No eligible users were found for the controlled LLM pilot.")
        write_jsonl(settings.user_llm_results_path, [])
        write_json_report(
            settings.user_llm_reranking_summary_path,
            {
                "status": "no_users_available",
                "llm_requested": True,
                "llm_ran": False,
                "selected_users": 0,
                "reranked_users": 0,
                "skipped_users": 0,
                "failed_users": 0,
                "max_llm_users": settings.max_llm_users,
                "max_llm_candidates": settings.max_llm_candidates,
                "llm_credentials_configured": provider_credentials_configured(settings),
                "warnings": ["No eligible users were available for the pilot."],
            },
        )
        return []

    llm_credentials_configured = provider_credentials_configured(settings)
    if not llm_credentials_configured and not _allow_llm_skip(settings):
        message = (
            f"LLM reranking was requested but credentials for provider '{llm_provider}' are missing. "
            "Set allow_llm_skip=true to skip cleanly, or configure credentials."
        )
        logger.warning("%s", message)
        raise RuntimeError(message)

    if not llm_credentials_configured:
        rows = build_skipped_user_llm_rows(selected_users, reason="skipped_no_credentials")
        write_jsonl(settings.user_llm_results_path, rows)
        summary = build_user_llm_summary(
            settings=settings,
            selected_users=selected_users,
            llm_rows=rows,
            status="skipped_no_credentials",
            llm_ran=False,
            llm_credentials_configured=False,
            llm_mode="real",
            llm_provider=llm_provider,
            mock_llm_ran=False,
            real_api_calls=0,
            not_for_scientific_metrics=False,
        )
        write_json_report(settings.user_llm_reranking_summary_path, summary)
        logger.info("Saved skipped user-LLM pilot summary to %s", settings.user_llm_reranking_summary_path)
        return rows

    provider_preflight_report: dict[str, object] = {"provider_preflight_ok": True, "provider_preflight_status": "ok"}
    completion_requests_attempted = 0
    token_requests_attempted = 0
    if llm_provider == "gigachat":
        provider_preflight_report = build_gigachat_token_preflight_report(settings, use_cache=True)
        token_requests_attempted = int(provider_preflight_report.get("token_requests_attempted", 0))
        if not bool(provider_preflight_report.get("provider_preflight_ok", False)):
            provider_failure_message = str(provider_preflight_report.get("provider_preflight_error_message_short", "") or "GigaChat provider preflight failed.")
            provider_failure_rows: list[dict[str, object]] = []
            for user_bundle in selected_users:
                baseline_rows = [
                    row
                    for row in user_bundle.get("baseline_rows", [])
                    if row.get("rank") is not None and row.get("game_id")
                ]
                provider_failure_rows.extend(
                    build_user_llm_records(
                        user_bundle=user_bundle,
                        items=baseline_rows[: settings.top_k],
                        id_to_card=context["game_cards_by_id"],
                        candidate_size=len(baseline_rows),
                        status="fallback_to_baseline",
                        notes=f"Fallback to baseline because provider preflight failed. Error: {provider_failure_message}",
                    )
                )
            llm_rows = provider_failure_rows
            validation_stats = {
                "users_requested": len(selected_users),
                "users_completed": 0,
                "users_failed": 0,
                "provider_failed_users": len(selected_users),
                "failed_user_count": 0,
                "failed_record_count": 0,
                "fallback_user_count": len(selected_users),
                "fallback_record_count": len(provider_failure_rows),
                "invalid_json_count": 0,
                "invalid_game_id_count": 0,
                "fallback_count": len(selected_users),
                "empty_response_count": 0,
                "schema_validation_error_count": 0,
                "candidate_pool_error_count": 0,
                "response_preview_saved_count": 0,
                "completion_requests_attempted": 0,
                "token_requests_attempted": token_requests_attempted,
                "real_api_calls_total": token_requests_attempted,
                "all_game_ids_inside_candidate_pool": True,
                "provider_preflight_ok": False,
                "provider_preflight_status": str(provider_preflight_report.get("provider_preflight_status", "token_error")),
                "provider_preflight_error_type": str(provider_preflight_report.get("provider_preflight_error_type", "token_error")),
                "provider_preflight_error_message_short": str(provider_preflight_report.get("provider_preflight_error_message_short", "")),
            }
            write_jsonl(settings.user_llm_results_path, llm_rows)
            write_user_llm_prompt_preview(settings, [])
            if settings.max_llm_users <= 3 and settings.max_users_for_experiment <= 3:
                write_user_llm_prompt_preview_variant(settings, [], variant_suffix="_tiny")
            summary = build_user_llm_summary(
                settings=settings,
                selected_users=selected_users,
                llm_rows=llm_rows,
                status="provider_preflight_failed",
                llm_ran=False,
                llm_credentials_configured=True,
                llm_mode="real",
                llm_provider=llm_provider,
                mock_llm_ran=False,
                real_api_calls=token_requests_attempted,
                not_for_scientific_metrics=False,
                validation_summary=validation_stats,
            )
            summary.update(validation_stats)
            write_json_report(settings.user_llm_reranking_summary_path, summary)
            write_json_report(settings.user_llm_validation_summary_path, validation_stats)
            write_user_llm_failure_report(
                settings,
                [],
                summary,
                validation_stats,
                provider_preflight_report=provider_preflight_report,
            )
            logger.info("Saved provider-preflight fallback rows to %s", settings.user_llm_results_path)
            return llm_rows

    system_prompt = build_user_llm_system_prompt(settings)
    llm_rows: list[dict[str, object]] = []
    prompt_preview_records: list[dict[str, object]] = []
    validation_stats = {
        "users_requested": len(selected_users),
        "users_completed": 0,
        "users_failed": 0,
        "provider_failed_users": 0,
        "failed_user_count": 0,
        "failed_record_count": 0,
        "fallback_user_count": 0,
        "fallback_record_count": 0,
        "invalid_json_count": 0,
        "invalid_game_id_count": 0,
        "fallback_count": 0,
        "empty_response_count": 0,
        "schema_validation_error_count": 0,
        "candidate_pool_error_count": 0,
        "response_preview_saved_count": 0,
        "completion_requests_attempted": 0,
        "token_requests_attempted": token_requests_attempted,
        "real_api_calls_total": token_requests_attempted,
        "provider_preflight_ok": True,
        "provider_preflight_status": str(provider_preflight_report.get("provider_preflight_status", "ok")),
        "provider_preflight_error_type": "",
        "provider_preflight_error_message_short": "",
        "all_game_ids_inside_candidate_pool": True,
    }
    failure_details: list[dict[str, object]] = []
    for index, user_bundle in enumerate(selected_users, start=1):
        completion_requests_attempted += 1
        candidate_rows = user_bundle["baseline_rows"][: settings.max_llm_candidates]
        prompt = build_user_llm_prompt(user_bundle, candidate_rows, context["game_cards_by_id"], settings)
        if index <= 3:
            prompt_preview_records.append(
                {
                    "masked_user_id": user_bundle["masked_user_id"],
                    "provider": llm_provider,
                    "model": llm_model,
                    "candidate_count": len(candidate_rows),
                    "candidate_game_titles": [
                        context["game_cards_by_id"][str(row.get("game_id", ""))].game_title
                        for row in candidate_rows
                        if str(row.get("game_id", "")) in context["game_cards_by_id"]
                    ],
                    "prompt_characters": len(prompt),
                    "prompt_too_long": len(prompt) > PROMPT_TOO_LONG_THRESHOLD,
                    "prompt": prompt,
                }
            )
        user_rows, user_stats, failure_detail = rerank_single_user(
                user_bundle=user_bundle,
                candidate_rows=candidate_rows,
                id_to_card=context["game_cards_by_id"],
                provider=llm_provider,
                model=llm_model,
                system_prompt=system_prompt,
                settings=settings,
        )
        llm_rows.extend(user_rows)
        validation_stats["users_completed"] += int(user_stats.get("users_completed", 0))
        validation_stats["users_failed"] += int(user_stats.get("users_failed", 0))
        validation_stats["failed_user_count"] += int(user_stats.get("users_failed", 0))
        validation_stats["failed_record_count"] += int(user_stats.get("failed_records", 0) if "failed_records" in user_stats else int(user_stats.get("users_failed", 0)))
        validation_stats["fallback_user_count"] += int(user_stats.get("fallback_count", 0))
        validation_stats["fallback_record_count"] += int(
            sum(1 for row in user_rows if str(row.get("status", "")) == "fallback_to_baseline")
        )
        validation_stats["invalid_json_count"] += int(user_stats.get("invalid_json_count", 0))
        validation_stats["invalid_game_id_count"] += int(user_stats.get("invalid_game_id_count", 0))
        validation_stats["fallback_count"] += int(user_stats.get("fallback_count", 0))
        validation_stats["empty_response_count"] += int(user_stats.get("empty_response_count", 0))
        validation_stats["schema_validation_error_count"] += int(user_stats.get("schema_validation_error_count", 0))
        validation_stats["candidate_pool_error_count"] += int(user_stats.get("candidate_pool_error_count", 0))
        validation_stats["response_preview_saved_count"] += int(user_stats.get("response_preview_saved_count", 0))
        validation_stats["all_game_ids_inside_candidate_pool"] = bool(
            validation_stats["all_game_ids_inside_candidate_pool"]
            and bool(user_stats.get("all_game_ids_inside_candidate_pool", True))
        )
        if failure_detail:
            failure_details.append(failure_detail)

    rows_by_user = group_rows_by_user(llm_rows)
    user_status_counts = Counter(first_status(rows) for rows in rows_by_user.values() if rows)
    record_status_counts = Counter(str(row.get("status", "")) for row in llm_rows)
    validation_stats.update(
        {
            "users_completed": int(len(rows_by_user) - int(user_status_counts.get("failed", 0))),
            "users_failed": int(user_status_counts.get("failed", 0)),
            "ok_users": int(user_status_counts.get("ok", 0)),
            "partial_fallback_users": int(user_status_counts.get("partial_fallback_to_baseline", 0)),
            "failed_users": int(user_status_counts.get("failed", 0)),
            "failed_user_count": int(user_status_counts.get("failed", 0)),
            "ok_records": int(record_status_counts.get("ok", 0)),
            "partial_fallback_records": int(record_status_counts.get("partial_fallback_to_baseline", 0)),
            "failed_records": int(record_status_counts.get("failed", 0)),
            "failed_record_count": int(record_status_counts.get("failed", 0)),
            "fallback_records": int(record_status_counts.get("fallback_to_baseline", 0)),
            "fallback_user_count": int(user_status_counts.get("fallback_to_baseline", 0)),
            "fallback_record_count": int(record_status_counts.get("fallback_to_baseline", 0)),
            "provider_failed_users": int(validation_stats.get("provider_failed_users", 0)),
            "completion_requests_attempted": int(completion_requests_attempted),
            "token_requests_attempted": int(token_requests_attempted),
            "real_api_calls_total": int(token_requests_attempted + completion_requests_attempted),
        }
    )
    validation_stats["schema_error_user_count"] = int(
        len(
            {
                str(detail.get("masked_user_id", ""))
                for detail in failure_details
                if str(detail.get("schema_validation_error_short", "")).strip()
            }
        )
    )

    write_jsonl(settings.user_llm_results_path, llm_rows)
    write_user_llm_prompt_preview(settings, prompt_preview_records)
    if settings.max_llm_users <= 3 and settings.max_users_for_experiment <= 3:
        write_user_llm_prompt_preview_variant(settings, prompt_preview_records, variant_suffix="_tiny")
    summary = build_user_llm_summary(
        settings=settings,
        selected_users=selected_users,
        llm_rows=llm_rows,
        status="completed",
        llm_ran=any(
            str(row.get("status", "")) in {"ok", "partial_fallback_to_baseline"}
            and row.get("rank") is not None
            and row.get("game_id")
            for row in llm_rows
        ),
        llm_credentials_configured=True,
        llm_mode="real",
        llm_provider=llm_provider,
        mock_llm_ran=False,
        real_api_calls=token_requests_attempted + completion_requests_attempted,
        not_for_scientific_metrics=False,
        validation_summary=validation_stats,
    )
    summary["validation_summary"] = validation_stats
    write_json_report(settings.user_llm_reranking_summary_path, summary)
    write_json_report(settings.user_llm_validation_summary_path, validation_stats)
    if validation_stats.get("schema_validation_error_count", 0):
        write_user_llm_schema_error_report(settings, failure_details, validation_stats)
    if failure_details or validation_stats["users_failed"] > 0:
        write_user_llm_failure_report(
            settings,
            failure_details,
            summary,
            validation_stats,
        )
    logger.info("Saved user-LLM recommendations to %s", settings.user_llm_results_path)
    return llm_rows


def run_user_llm_mock_pilot(settings: Settings) -> list[dict[str, object]]:

    logger = get_logger()
    context = load_user_llm_context(settings)
    selected_users = select_pilot_users(settings, context["splits"])
    if not selected_users:
        logger.warning("No eligible users were found for the mock LLM pilot.")
        empty_rows: list[dict[str, object]] = []
        write_jsonl(settings.user_llm_results_path, empty_rows)
        validation_summary = build_mock_validation_summary(
            users_requested=0,
            users_completed=0,
            users_failed=0,
            invalid_json_count=0,
            invalid_game_id_count=0,
            fallback_count=0,
            all_game_ids_inside_candidate_pool=True,
        )
        summary = build_user_llm_summary(
            settings=settings,
            selected_users=[],
            llm_rows=empty_rows,
            status="completed_mock_validation",
            llm_ran=False,
            llm_credentials_configured=False,
            llm_mode="mock",
            llm_provider="mock",
            mock_llm_ran=True,
            real_api_calls=0,
            not_for_scientific_metrics=True,
            validation_summary=validation_summary,
        )
        write_json_report(settings.user_llm_reranking_summary_path, summary)
        write_json_report(settings.user_llm_validation_summary_path, validation_summary)
        write_json_report(settings.user_llm_mock_validation_summary_path, validation_summary)
        save_user_llm_mock_validation_report(settings, summary, validation_summary, [])
        return empty_rows

    llm_rows: list[dict[str, object]] = []
    prompt_preview_records: list[dict[str, object]] = []
    validation_stats = {
        "users_requested": len(selected_users),
        "users_completed": 0,
        "users_failed": 0,
        "invalid_json_count": 0,
        "invalid_game_id_count": 0,
        "fallback_count": 0,
        "all_game_ids_inside_candidate_pool": True,
        "mock_mode": True,
        "real_api_calls": 0,
        "not_for_scientific_metrics": True,
    }
    for index, user_bundle in enumerate(selected_users, start=1):
        candidate_rows = user_bundle["baseline_rows"][: settings.max_llm_candidates]
        mock_rows = build_mock_user_llm_records(
            user_bundle=user_bundle,
            candidate_rows=candidate_rows,
            id_to_card=context["game_cards_by_id"],
            settings=settings,
        )
        llm_rows.extend(mock_rows)
        validation_stats["users_completed"] += 1
        if index <= 3:
            prompt = build_user_llm_prompt(user_bundle, candidate_rows, context["game_cards_by_id"], settings)
            prompt_preview_records.append(
                {
                    "masked_user_id": user_bundle["masked_user_id"],
                    "candidate_count": len(candidate_rows),
                    "candidate_game_titles": [
                        context["game_cards_by_id"][str(row.get("game_id", ""))].game_title
                        for row in candidate_rows
                        if str(row.get("game_id", "")) in context["game_cards_by_id"]
                    ],
                    "prompt_characters": len(prompt),
                    "prompt_too_long": len(prompt) > PROMPT_TOO_LONG_THRESHOLD,
                    "prompt": prompt,
                }
            )

    write_jsonl(settings.user_llm_results_path, llm_rows)
    write_user_llm_prompt_preview(settings, prompt_preview_records)
    if settings.max_llm_users <= 3 and settings.max_users_for_experiment <= 3:
        write_user_llm_prompt_preview_variant(settings, prompt_preview_records, variant_suffix="_tiny")
    summary = build_user_llm_summary(
        settings=settings,
        selected_users=selected_users,
        llm_rows=llm_rows,
        status="completed_mock_validation",
        llm_ran=False,
        llm_credentials_configured=False,
        llm_mode="mock",
        llm_provider="mock",
        mock_llm_ran=True,
        real_api_calls=0,
        not_for_scientific_metrics=True,
        validation_summary=validation_stats,
    )
    write_json_report(settings.user_llm_reranking_summary_path, summary)
    write_json_report(settings.user_llm_validation_summary_path, validation_stats)
    write_json_report(settings.user_llm_mock_validation_summary_path, validation_stats)
    save_user_llm_mock_validation_report(settings, summary, validation_stats, llm_rows)
    logger.info("Saved mock user-LLM recommendations to %s", settings.user_llm_results_path)
    return llm_rows


def run_user_llm_dry_run(settings: Settings) -> dict[str, object]:

    context = load_user_llm_context(settings)
    selected_users = select_pilot_users(settings, context["splits"])
    preview_limit = min(max(int(getattr(settings, "max_llm_users", 3) or 3), 1), 10)
    preview_users = selected_users[:preview_limit]
    provider = get_llm_provider_name(settings)
    model = get_effective_llm_model(settings)
    response_language = get_llm_response_language(settings)
    preview_records: list[dict[str, object]] = []
    for user_bundle in preview_users:
        candidate_rows = user_bundle["baseline_rows"][: settings.max_llm_candidates]
        prompt = build_user_llm_prompt(user_bundle, candidate_rows, context["game_cards_by_id"], settings)
        preview_records.append(
            {
                "masked_user_id": user_bundle["masked_user_id"],
                "provider": provider,
                "model": model,
                "top_k": settings.top_k,
                "candidate_count": len(candidate_rows),
                "response_language": response_language,
                "candidate_game_titles": [
                    context["game_cards_by_id"][str(row.get("game_id", ""))].game_title
                    for row in candidate_rows
                    if str(row.get("game_id", "")) in context["game_cards_by_id"]
                ],
                "prompt_characters": len(prompt),
                "prompt_too_long": len(prompt) > PROMPT_TOO_LONG_THRESHOLD,
                "prompt": prompt,
            }
        )

    write_user_llm_prompt_preview(settings, preview_records)
    if settings.max_llm_users <= 3 and settings.max_users_for_experiment <= 3:
        write_user_llm_prompt_preview_variant(settings, preview_records, variant_suffix="_tiny")
    if settings.max_llm_users >= 10 and str(getattr(settings, "llm_provider", "")).strip().lower() == "gigachat":
        write_user_llm_prompt_preview_variant(settings, preview_records, variant_suffix="_10_gigachat")
    prompt_lengths = [int(record["prompt_characters"]) for record in preview_records]
    report = {
        "selected_users": len(preview_records),
        "candidates_per_user": settings.max_llm_candidates,
        "top_k": settings.top_k,
        "provider": provider,
        "model": model,
        "response_language": response_language,
        "average_prompt_characters": round(sum(prompt_lengths) / len(prompt_lengths), 2) if prompt_lengths else 0,
        "max_prompt_characters": max(prompt_lengths, default=0),
        "prompt_too_long": any(record["prompt_too_long"] for record in preview_records),
        "preview_records": preview_records,
    }
    write_json_report(settings.user_llm_prompt_preview_json_path, report)
    if settings.max_llm_users <= 3 and settings.max_users_for_experiment <= 3:
        write_json_report(settings.user_llm_prompt_preview_tiny_json_path, report)
    if settings.max_llm_users >= 10 and str(getattr(settings, "llm_provider", "")).strip().lower() == "gigachat":
        write_json_report(settings.user_llm_prompt_preview_10_gigachat_json_path, report)
    return report


def run_llm_check(settings: Settings) -> dict[str, object]:

    logger = get_logger()
    provider_report = check_provider_readiness(settings)
    llm_mode = normalize_llm_mode(getattr(settings, "llm_mode", "real"))
    provider = get_effective_llm_provider(settings)
    report = {
        **provider_report,
        "llm_mode": llm_mode,
        "provider_credentials_present": provider_credentials_configured(settings),
        "openai_api_key_present": bool(str(settings.openai_api_key or "").strip()),
        "openai_model_present": bool(str(settings.llm_model or "").strip()),
    }
    if provider == "gigachat":
        report["gigachat_auth_key_present"] = bool(str(getattr(settings, "gigachat_auth_key", "") or "").strip())
        report["gigachat_scope"] = str(getattr(settings, "gigachat_scope", "") or "")
        report["gigachat_verify_ssl"] = bool(getattr(settings, "gigachat_verify_ssl", True))
        report["gigachat_ca_bundle"] = str(getattr(settings, "gigachat_ca_bundle", "") or "")
        report["gigachat_ca_bundle_exists"] = bool(
            report["gigachat_ca_bundle"] and Path(str(report["gigachat_ca_bundle"])).exists()
        )
        report["gigachat_ca_bundle_certificate_count"] = 0
        if report["gigachat_ca_bundle_exists"]:
            try:
                bundle_text = Path(str(report["gigachat_ca_bundle"])).read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                bundle_text = ""
            report["gigachat_ca_bundle_certificate_count"] = bundle_text.count("BEGIN CERTIFICATE")
        report["token_requests_attempted"] = int(provider_report.get("token_requests_attempted", 0))
        report["token_request_retries"] = int(provider_report.get("token_request_retries", 0))
        report["final_token_status"] = str(provider_report.get("final_token_status", provider_report.get("token_status", "")))
    write_json_report(settings.llm_check_json_path, report)
    settings.llm_check_markdown_path.write_text(
        "\n".join(
            [
                "# LLM Check",
                "",
                f"- Status: {report['status']}",
                f"- Selected provider: `{report['selected_provider']}`",
                f"- Selected model: `{report['selected_model'] or 'none'}`",
                f"- Response language: `{report['response_language']}`",
                f"- Provider credentials present: {report['provider_credentials_present']}",
                f"- Real LLM calls allowed: {report['real_llm_calls_allowed']}",
                f"- LLM mode: `{llm_mode}`",
                f"- Client initialized: {report['client_initialized']}",
                f"- Token status: `{report.get('token_status', '')}`",
                f"- Provider readiness: `{report['status']}`",
                *(
                    [
                        f"- GigaChat scope: `{report.get('gigachat_scope', '')}`",
                        f"- GigaChat verify SSL: {report.get('gigachat_verify_ssl', True)}",
                        f"- GigaChat auth key present: {report.get('gigachat_auth_key_present', False)}",
                        f"- GigaChat CA bundle: `{report.get('gigachat_ca_bundle', '') or 'none'}`",
                        f"- GigaChat CA bundle exists: {report.get('gigachat_ca_bundle_exists', False)}",
                        f"- GigaChat CA bundle certificate count: {report.get('gigachat_ca_bundle_certificate_count', 0)}",
                    ]
                    if report.get("selected_provider") == "gigachat"
                    else []
                ),
                "",
                "This check does not send a completion request.",
                "",
            ]
            + ([f"- Error: {report['error']}"] if report.get("error") else [])
        )
        + "\n",
        encoding="utf-8",
    )
    if report["status"] == "missing_credentials":
        logger.warning("LLM check: missing credentials.")
    else:
        logger.info(
            "LLM check: provider %s, model %s.",
            report["selected_provider"],
            report["selected_model"],
        )
    return report


def evaluate_user_llm(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger = get_logger()
    if not settings.user_llm_results_path.exists():
        logger.warning("User LLM recommendations are missing. Run `--step user_llm` first.")
        empty = pd.DataFrame()
        return empty, empty, empty

    context = load_user_llm_context(settings)
    selected_users = select_pilot_users(settings, context["splits"])
    selected_user_ids = {bundle["user_id"] for bundle in selected_users}
    if not selected_user_ids:
        empty = pd.DataFrame()
        return empty, empty, empty

    baseline_rows = [
        row
        for row in read_jsonl(settings.user_baseline_results_path)
        if str(row.get("user_id", "")) in selected_user_ids
    ]
    llm_rows = [
        row
        for row in read_jsonl(settings.user_llm_results_path)
        if str(row.get("user_id", "")) in selected_user_ids
    ]
    per_profile_df = build_user_llm_per_profile_frame(
        selected_users=selected_users,
        baseline_rows=baseline_rows,
        llm_rows=llm_rows,
    )
    per_profile_df.to_csv(settings.user_llm_per_profile_results_path, index=False)

    metrics_summary_df = build_user_llm_metrics_summary(per_profile_df)
    metrics_summary_df.to_csv(settings.user_llm_metrics_summary_path, index=False)
    if settings.user_metrics_summary_path.exists():
        try:
            shutil.copy2(settings.user_metrics_summary_path, settings.user_llm_metrics_summary_all_pilot_path)
        except OSError:
            pass

    rank_comparison_df = build_user_rank_comparison_frame(per_profile_df)
    rank_comparison_df.to_csv(settings.user_rank_comparison_path, index=False)
    save_user_rank_comparison_markdown(settings, rank_comparison_df)

    explanation_checks_df = build_user_llm_explanation_checks_frame(
        llm_rows=llm_rows,
        selected_users=selected_users,
        game_cards_by_id=context["game_cards_by_id"],
        top_k=settings.top_k,
    )
    explanation_checks_df.to_csv(settings.user_llm_explanation_checks_csv_path, index=False)
    save_user_llm_explanation_outputs(settings, explanation_checks_df)
    save_user_llm_examples_markdown(settings, llm_rows, selected_users, context["game_cards_by_id"])
    validation_summary = load_json_if_exists(settings.user_llm_validation_summary_path)

    summary = build_user_llm_pilot_summary(
        settings=settings,
        selected_users=selected_users,
        per_profile_df=per_profile_df,
        metrics_summary_df=metrics_summary_df,
        rank_comparison_df=rank_comparison_df,
        explanation_checks_df=explanation_checks_df,
        llm_rows=llm_rows,
        validation_summary=validation_summary,
    )
    write_json_report(settings.user_llm_reranking_summary_path, summary)
    save_user_llm_metrics_table(settings, metrics_summary_df)
    save_user_llm_pilot_summary_markdown(settings, summary)
    logger.info("Saved user LLM evaluation outputs to %s", settings.user_llm_metrics_summary_path)
    return per_profile_df, metrics_summary_df, rank_comparison_df


def load_user_llm_context(settings: Settings) -> dict[str, object]:

    profiles = read_jsonl(settings.user_profiles_path)
    split_path = get_active_user_splits_path(settings)
    splits = read_jsonl(split_path)
    baseline_rows = read_jsonl(settings.user_baseline_results_path)
    game_cards = [GameCard(**record) for record in read_jsonl(settings.game_cards_path)]
    game_cards_by_id = {card.game_id: card for card in game_cards}
    baseline_by_user = group_rows_by_user(baseline_rows)
    profile_by_user = {str(profile.get("user_id", "")): profile for profile in profiles}
    split_by_user = {str(split.get("user_id", "")): split for split in splits}
    return {
        "profiles": profiles,
        "splits": splits,
        "baseline_rows": baseline_rows,
        "game_cards": game_cards,
        "game_cards_by_id": game_cards_by_id,
        "baseline_by_user": baseline_by_user,
        "profile_by_user": profile_by_user,
        "split_by_user": split_by_user,
    }


def build_llm_pilot_candidate_user_report_frame(
    settings: Settings,
    splits: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
) -> pd.DataFrame:

    baseline_by_user = group_rows_by_user(baseline_rows)
    rows: list[dict[str, object]] = []
    for split in splits:
        user_id = str(split.get("user_id", ""))
        if not user_id:
            continue
        candidate_game_ids = {str(game_id) for game_id in split.get("candidate_game_ids", []) if str(game_id)}
        ground_truth_ids = {str(game_id) for game_id in split.get("ground_truth_game_ids", []) if str(game_id)}
        user_baseline_rows = sorted(
            [
                row
                for row in baseline_by_user.get(user_id, [])
                if row.get("rank") is not None and row.get("game_id")
            ],
            key=lambda row: int(row["rank"]),
        )
        candidate_pool_size = len(candidate_game_ids) if candidate_game_ids else min(len(user_baseline_rows), settings.max_llm_candidates)
        holdout_count = len(ground_truth_ids)
        baseline_rank = best_holdout_rank(user_baseline_rows[: settings.max_llm_candidates], ground_truth_ids)
        holdout_in_candidate_pool = bool(candidate_game_ids & ground_truth_ids) or baseline_rank is not None
        eligible = bool(
            holdout_count > 0
            and candidate_pool_size > 0
            and holdout_in_candidate_pool
            and baseline_rank is not None
            and baseline_rank > 1
        )
        rows.append(
            {
                "user_id": user_id,
                "masked_user_id": str(split.get("masked_user_id", "")) or mask_user_id(user_id),
                "holdout_count": int(holdout_count),
                "candidate_pool_size": int(candidate_pool_size),
                "holdout_in_candidate_pool": bool(holdout_in_candidate_pool),
                "baseline_best_holdout_rank": format_rank_value(baseline_rank),
                "baseline_hit_at_10": int(is_hit_at_k(baseline_rank, 10)),
                "eligible_for_meaningful_llm_pilot": bool(eligible),
            }
        )
    return pd.DataFrame(rows)


def save_llm_pilot_candidate_user_report(settings: Settings, frame: pd.DataFrame) -> None:

    frame.to_csv(settings.llm_pilot_candidate_user_report_csv_path, index=False)
    lines = [
        "# LLM Pilot Candidate User Report",
        "",
        "This report is for pilot-case selection only and does not use holdout labels as model input.",
        "",
    ]
    if frame.empty:
        lines.append("_No candidate users available._")
    else:
        display_frame = frame.drop(columns=["user_id"], errors="ignore")
        lines.append(dataframe_to_markdown(display_frame))
        eligible_count = int(
            display_frame["eligible_for_meaningful_llm_pilot"].fillna(False).astype(bool).sum()
        ) if "eligible_for_meaningful_llm_pilot" in display_frame.columns else 0
        lines.extend(
            [
                "",
                f"- Total users inspected: {len(frame)}",
                f"- Eligible for meaningful pilot: {eligible_count}",
                "- A user is eligible when the holdout is inside the candidate pool and the baseline rank is not already 1.",
            ]
        )
    settings.llm_pilot_candidate_user_report_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_pilot_users(settings: Settings, splits: list[dict[str, object]]) -> list[dict[str, object]]:

    baseline_rows = read_jsonl(settings.user_baseline_results_path)
    baseline_by_user = group_rows_by_user(baseline_rows)
    candidate_report = build_llm_pilot_candidate_user_report_frame(settings, splits, baseline_rows)
    save_llm_pilot_candidate_user_report(settings, candidate_report)
    if settings.llm_select_meaningful_users and not candidate_report.empty:
        eligible_ids = candidate_report[
            candidate_report["eligible_for_meaningful_llm_pilot"].fillna(False).astype(bool)
        ]["user_id"].astype(str).tolist()
        fallback_ids = [
            str(split.get("user_id", ""))
            for split in splits
            if str(split.get("user_id", ""))
        ]
        ordered_user_ids = eligible_ids + [user_id for user_id in fallback_ids if user_id not in eligible_ids]
        if len(eligible_ids) < settings.max_llm_users:
            get_logger().warning(
                "Only %s meaningful LLM pilot candidates were found; falling back to additional users to reach the pilot limit.",
                len(eligible_ids),
            )
    else:
        ordered_user_ids = [
            str(split.get("user_id", ""))
            for split in splits
            if str(split.get("user_id", ""))
        ]
    selected: list[dict[str, object]] = []
    split_by_user = {str(split.get("user_id", "")): split for split in splits}
    for user_id in ordered_user_ids:
        split = split_by_user.get(user_id, {})
        if not user_id:
            continue
        user_baseline_rows = sorted(
            [
                row
                for row in baseline_by_user.get(user_id, [])
                if row.get("rank") is not None and row.get("game_id")
            ],
            key=lambda row: int(row["rank"]),
        )
        if not user_baseline_rows:
            continue
        selected.append(
            {
                "user_id": user_id,
                "masked_user_id": str(split.get("masked_user_id", "")) or mask_user_id(user_id),
                "train_game_ids": [str(game_id) for game_id in split.get("train_game_ids", [])],
                "ground_truth_game_ids": [str(game_id) for game_id in split.get("ground_truth_game_ids", [])],
                "candidate_game_ids": [str(game_id) for game_id in split.get("candidate_game_ids", [])],
                "candidate_pool_size": int(len(split.get("candidate_game_ids", []) or [])),
                "baseline_rows": user_baseline_rows,
                "profile": None,
            }
        )
        if len(selected) >= settings.max_llm_users:
            break
    profile_by_user = {str(profile.get("user_id", "")): profile for profile in read_jsonl(settings.user_profiles_path)}
    for bundle in selected:
        bundle["profile"] = profile_by_user.get(bundle["user_id"], {})
    return selected


def build_user_llm_system_prompt(settings: Settings) -> str:

    response_language = get_llm_response_language(settings)
    if response_language == "ru":
        return (
            "You are a careful game recommender. Rank only the provided candidate games. "
            "Do not invent games or use external knowledge. Return valid JSON only. "
            "Keep JSON keys in English. Write explanation, matched_preferences, and possible_risks in Russian. "
            "Do not translate game titles. Use only the provided user profile and candidate game cards. "
            "Do not recommend games outside the candidate list. "
            "Return exactly the requested number of recommendations. Every rank from 1 to top_k must be present. "
            "Use each candidate at most once. "
            "Do not wrap the answer in markdown. Do not use ```json code fences. "
            "The first character of the response must be { and the last character must be }."
        )
    return (
        "You are a careful game recommender. Rank only the provided candidate games. "
        "Do not invent games or use external knowledge. Return valid JSON only. "
        "Keep JSON keys in English. Write explanation, matched_preferences, and possible_risks in English. "
        "Do not translate game titles. Use only the provided user profile and candidate game cards. "
        "Do not recommend games outside the candidate list. "
        "Return exactly the requested number of recommendations. Every rank from 1 to top_k must be present. "
        "Use each candidate at most once. "
        "Do not wrap the answer in markdown. Do not use ```json code fences. "
        "The first character of the response must be { and the last character must be }."
    )


def build_user_llm_prompt(
    user_bundle: dict[str, object],
    candidate_rows: list[dict[str, object]],
    game_cards_by_id: dict[str, GameCard],
    settings: Settings,
) -> str:

    profile = user_bundle.get("profile", {}) or {}
    masked_user_id = str(user_bundle.get("masked_user_id", ""))
    response_language = get_llm_response_language(settings)
    profile_text = build_profile_text(profile, game_cards_by_id)
    candidate_text = "\n".join(
        build_candidate_line(row, game_cards_by_id)
        for row in candidate_rows
        if str(row.get("game_id", "")) in game_cards_by_id
    )
    if response_language == "ru":
        prompt_lines = [
            f"Маскированный пользователь: {masked_user_id}",
            "Задача: ранжируйте только предоставленные игры-кандидаты для этого пользователя.",
            "Используйте только профиль пользователя, понравившиеся и не понравившиеся игры.",
            "Return valid JSON only.",
            "Keep JSON keys in English.",
            "Write explanation, matched_preferences, and possible_risks in Russian.",
            "Do not translate game titles.",
            "Use only the provided user profile and candidate game cards.",
            "Do not recommend games outside the candidate list.",
            f"Return exactly {settings.top_k} recommendations.",
            "Use each candidate at most once.",
            "Every recommendation must include rank, game_id, game_title, relevance_score, explanation, matched_preferences, and possible_risks.",
            f"All ranks from 1 to {settings.top_k} must be present.",
            "If unsure, still choose only from the provided candidate list.",
            "Do not omit lower-ranked candidates.",
            "Не используйте markdown code fences.",
            "Первая буква ответа должна быть {, а последняя буква — }.",
            "Схема ответа:",
            '{"recommendations":[{"rank":1,"game_id":"string","game_title":"string","relevance_score":0.0,"explanation":"string","matched_preferences":["string"],"possible_risks":["string"]}]}',
            "",
            "Профиль пользователя:",
            profile_text,
            "",
            "Игры-кандидаты:",
            candidate_text,
        ]
    else:
        prompt_lines = [
            f"Masked user: {masked_user_id}",
            "Task: Rank only the provided candidate games for this user.",
            "Use the profile, liked games, and disliked games only.",
            "Return valid JSON only.",
            "Keep JSON keys in English.",
            "Write explanation, matched_preferences, and possible_risks in English.",
            "Do not translate game titles.",
            "Use only the provided user profile and candidate game cards.",
            "Do not recommend games outside the candidate list.",
            f"Return exactly {settings.top_k} recommendations.",
            "Use each candidate at most once.",
            "Every recommendation must include rank, game_id, game_title, relevance_score, explanation, matched_preferences, and possible_risks.",
            f"All ranks from 1 to {settings.top_k} must be present.",
            "If unsure, still choose only from the provided candidate list.",
            "Do not omit lower-ranked candidates.",
            "Do not use markdown code fences.",
            "The first character of the response must be { and the last character must be }.",
            "Return JSON with this schema:",
            '{"recommendations":[{"rank":1,"game_id":"string","game_title":"string","relevance_score":0.0,"explanation":"string","matched_preferences":["string"],"possible_risks":["string"]}]}',
            "",
            "User profile:",
            profile_text,
            "",
            "Candidate games:",
            candidate_text,
        ]
    return "\n".join(prompt_lines).strip()


def build_profile_text(profile: dict[str, object], game_cards_by_id: dict[str, GameCard]) -> str:

    positive_game_ids = [str(game_id) for game_id in profile.get("positive_game_ids", [])]
    negative_game_ids = [str(game_id) for game_id in profile.get("negative_game_ids", [])]
    positive_titles = [game_cards_by_id[game_id].game_title for game_id in positive_game_ids if game_id in game_cards_by_id]
    negative_titles = [game_cards_by_id[game_id].game_title for game_id in negative_game_ids if game_id in game_cards_by_id]
    positive_text = truncate_text(", ".join(positive_titles) or "none", 220)
    negative_text = truncate_text(", ".join(negative_titles) or "none", 220)
    review_text = truncate_text(str(profile.get("positive_review_text", "")), 350)
    return "\n".join(
        [
            f"Profile id: {profile.get('masked_user_id', '')}",
            f"Review count: {profile.get('review_count', 0)}",
            f"Positive review count: {profile.get('positive_review_count', 0)}",
            f"Liked games: {positive_text}",
            f"Disliked games: {negative_text}",
            f"Positive review text: {review_text}",
        ]
    )


def build_candidate_line(row: dict[str, object], game_cards_by_id: dict[str, GameCard]) -> str:

    game_id = str(row.get("game_id", ""))
    card = game_cards_by_id[game_id]
    return (
        f"- game_id: {card.game_id}; game_title: {card.game_title}; "
        f"positive_ratio: {card.positive_ratio}; "
        f"positive_keywords: {', '.join(card.positive_keywords[:5]) or 'none'}; "
        f"negative_keywords: {', '.join(card.negative_keywords[:5]) or 'none'}; "
        f"game_card_text: {truncate_text(card.game_card_text, 240)}"
    )


def rerank_single_user(
    user_bundle: dict[str, object],
    candidate_rows: list[dict[str, object]],
    id_to_card: dict[str, GameCard],
    provider: str,
    model: str,
    system_prompt: str,
    settings: Settings,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object] | None]:

    baseline_ids = [str(row.get("game_id", "")) for row in candidate_rows if row.get("game_id")]
    baseline_ids = [game_id for game_id in baseline_ids if game_id in id_to_card]
    baseline_ids = deduplicate_preserve_order(baseline_ids)
    candidate_size = len(baseline_ids)
    allow_fallback = bool(getattr(settings, "allow_llm_fallback", True))
    save_response_preview = bool(getattr(settings, "llm_save_response_preview", True))

    def make_fallback_rows(message: str) -> list[dict[str, object]]:
        return [
            {
                "rank": index,
                "game_id": game_id,
                "game_title": id_to_card[game_id].game_title,
                "relevance_score": 0.0,
                "explanation": message,
                "matched_preferences": [],
                "possible_risks": [],
            }
            for index, game_id in enumerate(baseline_ids[: settings.top_k], start=1)
        ]

    def base_stats(**kwargs: object) -> dict[str, object]:
        stats = {
            "users_requested": 1,
            "users_completed": 0,
            "users_failed": 0,
            "invalid_json_count": 0,
            "invalid_game_id_count": 0,
            "fallback_count": 0,
            "empty_response_count": 0,
            "schema_validation_error_count": 0,
            "candidate_pool_error_count": 0,
            "response_preview_saved_count": 0,
            "all_game_ids_inside_candidate_pool": True,
            "error_type": "",
            "error_message_short": "",
            "token_obtained": False,
            "api_call_attempted": False,
            "http_status": None,
            "response_empty": False,
            "response_received": False,
            "response_preview_safe": "",
            "json_parse_attempted": False,
            "json_parse_error_short": "",
            "schema_validation_error_short": "",
            "candidate_pool_validation_error_short": "",
            "fallback_applied": False,
            "json_parsing_failed": False,
        }
        stats.update(kwargs)
        return stats

    def failure_detail_from_stats(stats: dict[str, object]) -> dict[str, object]:
        return {
            "masked_user_id": user_bundle["masked_user_id"],
            "provider": provider,
            "model": model,
            "llm_mode": "real",
            "api_call_attempted": bool(stats.get("api_call_attempted", False)),
            "token_obtained": bool(stats.get("token_obtained", False)),
            "http_status": stats.get("http_status"),
            "error_type": str(stats.get("error_type", "")),
            "error_message_short": str(stats.get("error_message_short", "")),
            "response_received": bool(stats.get("response_received", False)),
            "response_empty": bool(stats.get("response_empty", False)),
            "response_preview_safe": str(stats.get("response_preview_safe", "")),
            "json_parse_attempted": bool(stats.get("json_parse_attempted", False)),
            "json_parse_error_short": str(stats.get("json_parse_error_short", "")),
            "schema_validation_error_short": str(stats.get("schema_validation_error_short", "")),
            "candidate_pool_validation_error_short": str(stats.get("candidate_pool_validation_error_short", "")),
            "fallback_applied": bool(stats.get("fallback_applied", False)),
        }

    if not baseline_ids:
        stats = base_stats(
            users_failed=1,
            error_type="no_baseline_candidates",
            error_message_short="No valid baseline candidates remained for this user.",
        )
        rows = [{
            "user_id": user_bundle["user_id"],
            "masked_user_id": user_bundle["masked_user_id"],
            "method": "user_llm_reranker",
            "llm_mode": "real",
            "status": "failed",
            "rank": None,
            "game_id": None,
            "game_title": None,
            "score": None,
            "candidate_size": 0,
            "notes": "No valid baseline candidates remained for this user.",
            "explanation": "",
            "matched_preferences": [],
            "possible_risks": [],
        }]
        return rows, stats, failure_detail_from_stats(stats)

    prompt = build_user_llm_prompt(user_bundle, candidate_rows, id_to_card, settings)
    try:
        content = generate_llm_json_response(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.0,
            settings=settings,
        )
        parsed, diag = parse_llm_response_with_diagnostics(content, save_preview=save_response_preview)
        response_preview_safe = str(diag.get("response_preview_safe", "") or "")
        response_received = bool(diag.get("response_received", False))
        response_empty = bool(diag.get("response_empty", False))
        json_parse_attempted = bool(diag.get("json_parse_attempted", False))
        json_parse_error_short = str(diag.get("json_parse_error_short", "") or "")
        schema_validation_error_short = str(diag.get("schema_validation_error_short", "") or "")

        if parsed is None:
            error_type = "invalid_json" if diag.get("invalid_json") else "schema_validation_error"
            error_message_short = json_parse_error_short or schema_validation_error_short or "invalid_llm_response"
            if allow_fallback:
                rows = build_user_llm_records(
                    user_bundle=user_bundle,
                    items=make_fallback_rows("Fallback to baseline because LLM response failed validation."),
                    id_to_card=id_to_card,
                    candidate_size=candidate_size,
                    status="fallback_to_baseline",
                    notes="Fallback to baseline because LLM response failed validation.",
                )
                stats = base_stats(
                    users_completed=1,
                    fallback_count=1,
                    invalid_json_count=int(bool(diag.get("invalid_json"))),
                    empty_response_count=int(response_empty),
                    schema_validation_error_count=int(bool(schema_validation_error_short)),
                    response_preview_saved_count=int(bool(response_preview_safe)),
                    error_type=error_type,
                    error_message_short=error_message_short,
                    token_obtained=True,
                    api_call_attempted=True,
                    response_empty=response_empty,
                    response_received=response_received,
                    response_preview_safe=response_preview_safe,
                    json_parse_attempted=json_parse_attempted,
                    json_parse_error_short=json_parse_error_short,
                    schema_validation_error_short=schema_validation_error_short,
                    candidate_pool_validation_error_short="",
                    fallback_applied=True,
                    json_parsing_failed=bool(diag.get("invalid_json")),
                )
                return rows, stats, failure_detail_from_stats(stats)

            rows = [{
                "user_id": user_bundle["user_id"],
                "masked_user_id": user_bundle["masked_user_id"],
                "method": "user_llm_reranker",
                "llm_mode": "real",
                "status": "failed",
                "rank": None,
                "game_id": None,
                "game_title": None,
                "score": None,
                "candidate_size": candidate_size,
                "notes": f"LLM response failed validation: {error_message_short}",
                "explanation": "",
                "matched_preferences": [],
                "possible_risks": [],
            }]
            stats = base_stats(
                users_failed=1,
                invalid_json_count=int(bool(diag.get("invalid_json"))),
                empty_response_count=int(response_empty),
                schema_validation_error_count=int(bool(schema_validation_error_short)),
                response_preview_saved_count=int(bool(response_preview_safe)),
                error_type=error_type,
                error_message_short=error_message_short,
                token_obtained=True,
                api_call_attempted=True,
                response_empty=response_empty,
                response_received=response_received,
                response_preview_safe=response_preview_safe,
                json_parse_attempted=json_parse_attempted,
                json_parse_error_short=json_parse_error_short,
                schema_validation_error_short=schema_validation_error_short,
                candidate_pool_validation_error_short="",
                fallback_applied=False,
                json_parsing_failed=bool(diag.get("invalid_json")),
            )
            return rows, stats, failure_detail_from_stats(stats)

        allowed = set(baseline_ids)
        seen: set[str] = set()
        valid_items: list[dict[str, object]] = []
        invalid_game_id_count = 0
        for item in parsed.recommendations:
            game_id = str(item.game_id)
            if game_id not in allowed:
                invalid_game_id_count += 1
                continue
            if game_id in seen:
                continue
            seen.add(game_id)
            valid_items.append({
                "rank": item.rank,
                "game_id": game_id,
                "game_title": item.game_title,
                "relevance_score": item.relevance_score,
                "explanation": item.explanation,
                "matched_preferences": item.matched_preferences,
                "possible_risks": item.possible_risks,
            })

        valid_items = sorted(valid_items, key=lambda item: int(item.get("rank", 0) or 0))
        valid_ids = [str(item["game_id"]) for item in valid_items]
        filler_ids = [game_id for game_id in baseline_ids if game_id not in valid_ids]
        if len(valid_items) == 0:
            status = "fallback_to_baseline"
        elif len(valid_items) < min(settings.top_k, len(baseline_ids)):
            status = "partial_fallback_to_baseline"
        else:
            status = "ok"

        if status == "fallback_to_baseline" and allow_fallback:
            rows = build_user_llm_records(
                user_bundle=user_bundle,
                items=make_fallback_rows("Fallback to baseline because the LLM returned no valid ranked items."),
                id_to_card=id_to_card,
                candidate_size=candidate_size,
                status="fallback_to_baseline",
                notes="Fallback to baseline because the LLM returned no valid ranked items.",
            )
        elif status == "fallback_to_baseline":
            stats = base_stats(
                users_failed=1,
                invalid_game_id_count=int(invalid_game_id_count),
                candidate_pool_error_count=int(invalid_game_id_count),
                response_preview_saved_count=int(bool(response_preview_safe)),
                all_game_ids_inside_candidate_pool=invalid_game_id_count == 0,
                error_type="invalid_llm_response",
                error_message_short="LLM returned no valid ranked items.",
                token_obtained=True,
                api_call_attempted=True,
                response_empty=response_empty,
                response_received=response_received,
                response_preview_safe=response_preview_safe,
                json_parse_attempted=json_parse_attempted,
                json_parse_error_short=json_parse_error_short,
                schema_validation_error_short=schema_validation_error_short,
                candidate_pool_validation_error_short=(
                    f"Returned {invalid_game_id_count} game IDs outside the candidate pool." if invalid_game_id_count > 0 else ""
                ),
                fallback_applied=False,
                json_parsing_failed=bool(diag.get("invalid_json")),
            )
            rows = [{
                "user_id": user_bundle["user_id"],
                "masked_user_id": user_bundle["masked_user_id"],
                "method": "user_llm_reranker",
                "llm_mode": "real",
                "status": "failed",
                "rank": None,
                "game_id": None,
                "game_title": None,
                "score": None,
                "candidate_size": candidate_size,
                "notes": "LLM returned no valid ranked items.",
                "explanation": "",
                "matched_preferences": [],
                "possible_risks": [],
            }]
            return rows, stats, failure_detail_from_stats(stats)
        else:
            final_items = valid_items + [
                {
                    "rank": index,
                    "game_id": game_id,
                    "game_title": id_to_card[game_id].game_title,
                    "relevance_score": 0.0,
                    "explanation": "Fallback to baseline because the LLM output did not fill all ranks.",
                    "matched_preferences": [],
                    "possible_risks": [],
                }
                for index, game_id in enumerate(filler_ids, start=len(valid_items) + 1)
            ]
            final_items = final_items[: min(settings.top_k, len(baseline_ids))]
            rows = build_user_llm_records(
                user_bundle=user_bundle,
                items=final_items,
                id_to_card=id_to_card,
                candidate_size=candidate_size,
                status=status,
                notes="User LLM reranking completed." if status == "ok" else "User LLM reranking completed with baseline fallback.",
            )

        stats = base_stats(
            users_completed=1,
            invalid_json_count=int(bool(diag.get("invalid_json"))),
            invalid_game_id_count=int(invalid_game_id_count),
            fallback_count=int(status != "ok"),
            empty_response_count=int(response_empty),
            schema_validation_error_count=int(bool(schema_validation_error_short)),
            candidate_pool_error_count=int(invalid_game_id_count),
            response_preview_saved_count=int(bool(response_preview_safe)),
            all_game_ids_inside_candidate_pool=invalid_game_id_count == 0,
            response_empty=response_empty,
            response_received=response_received,
            response_preview_safe=response_preview_safe,
            json_parse_attempted=json_parse_attempted,
            json_parse_error_short=json_parse_error_short,
            schema_validation_error_short=schema_validation_error_short,
            candidate_pool_validation_error_short=(
                f"Returned {invalid_game_id_count} game IDs outside the candidate pool." if invalid_game_id_count > 0 else ""
            ),
            fallback_applied=bool(status != "ok"),
            json_parsing_failed=bool(diag.get("invalid_json")),
        )
        failure_detail = failure_detail_from_stats(stats) if status != "ok" or invalid_game_id_count > 0 else None
        return rows, stats, failure_detail
    except Exception as exc:  # noqa: BLE001
        error_type = type(exc).__name__
        error_message_short = truncate_text(str(exc), 180)
        http_status = getattr(getattr(exc, "response", None), "status_code", None)
        response_obj = getattr(exc, "response", None)
        response_text = str(getattr(response_obj, "text", "") or "")
        response_received = bool(response_text.strip())
        response_empty = not response_received
        response_preview_safe = truncate_text(response_text, 500) if response_received and save_response_preview else ""
        token_obtained = infer_token_obtained_from_error(
            provider=provider,
            error_message=error_message_short,
            response_received=response_received,
            http_status=http_status,
        )

        if allow_fallback:
            rows = build_user_llm_records(
                user_bundle=user_bundle,
                items=make_fallback_rows("Fallback to baseline because the LLM request failed."),
                id_to_card=id_to_card,
                candidate_size=candidate_size,
                status="fallback_to_baseline",
                notes=f"LLM reranking failed. Baseline order used instead. Error: {error_message_short}",
            )
            stats = base_stats(
                users_completed=1,
                fallback_count=1,
                empty_response_count=int(response_empty),
                response_preview_saved_count=int(bool(response_preview_safe)),
                error_type=error_type,
                error_message_short=error_message_short,
                token_obtained=token_obtained,
                api_call_attempted=True,
                http_status=http_status,
                response_empty=response_empty,
                response_received=response_received,
                response_preview_safe=response_preview_safe,
                json_parse_attempted=False,
                json_parse_error_short="",
                schema_validation_error_short="",
                candidate_pool_validation_error_short="",
                fallback_applied=True,
            )
            return rows, stats, failure_detail_from_stats(stats)

        rows = [{
            "user_id": user_bundle["user_id"],
            "masked_user_id": user_bundle["masked_user_id"],
            "method": "user_llm_reranker",
            "llm_mode": "real",
            "status": "failed",
            "rank": None,
            "game_id": None,
            "game_title": None,
            "score": None,
            "candidate_size": candidate_size,
            "notes": f"LLM reranking failed: {error_message_short}",
            "explanation": "",
            "matched_preferences": [],
            "possible_risks": [],
        }]
        stats = base_stats(
            users_failed=1,
            empty_response_count=int(response_empty),
            response_preview_saved_count=int(bool(response_preview_safe)),
            error_type=error_type,
            error_message_short=error_message_short,
            token_obtained=token_obtained,
            api_call_attempted=True,
            http_status=http_status,
            response_empty=response_empty,
            response_received=response_received,
            response_preview_safe=response_preview_safe,
            json_parse_attempted=False,
            json_parse_error_short="",
            schema_validation_error_short="",
            candidate_pool_validation_error_short="",
            fallback_applied=False,
        )
        return rows, stats, failure_detail_from_stats(stats)

def build_user_llm_records(
    user_bundle: dict[str, object],
    items: list[dict[str, object]],
    id_to_card: dict[str, GameCard],
    candidate_size: int,
    status: str,
    notes: str,
    llm_mode: str = "real",
) -> list[dict[str, object]]:

    rows: list[dict[str, object]] = []
    for index, item in enumerate(items, start=1):
        game_id = str(item.get("game_id", ""))
        if game_id not in id_to_card:
            continue
        rows.append(
            {
                "user_id": user_bundle["user_id"],
                "masked_user_id": user_bundle["masked_user_id"],
                "method": "user_llm_reranker",
                "llm_mode": llm_mode,
                "status": status,
                "rank": index,
                "game_id": game_id,
                "game_title": id_to_card[game_id].game_title,
                "score": round(float(item.get("relevance_score", 0.0)), 6),
                "candidate_size": candidate_size,
                "notes": notes,
                "explanation": item.get("explanation", ""),
                "matched_preferences": item.get("matched_preferences", []),
                "possible_risks": item.get("possible_risks", []),
            }
        )
    return rows


def build_mock_user_llm_records(
    user_bundle: dict[str, object],
    candidate_rows: list[dict[str, object]],
    id_to_card: dict[str, GameCard],
    settings: Settings,
) -> list[dict[str, object]]:

    baseline_ids = [str(row.get("game_id", "")) for row in candidate_rows if row.get("game_id")]
    baseline_ids = [game_id for game_id in baseline_ids if game_id in id_to_card]
    baseline_ids = deduplicate_preserve_order(baseline_ids)
    if not baseline_ids:
        return [
            {
                "user_id": user_bundle["user_id"],
                "masked_user_id": user_bundle["masked_user_id"],
                "method": "user_llm_reranker",
                "llm_mode": "mock",
                "status": "mock_validation_only",
                "rank": None,
                "game_id": None,
                "game_title": None,
                "score": None,
                "candidate_size": 0,
                "notes": "Mock validation mode had no valid baseline candidates.",
                "explanation": "Mock explanation: no valid candidates were available for pipeline validation only.",
                "matched_preferences": [],
                "possible_risks": [],
            }
        ]

    profile = user_bundle.get("profile", {}) or {}
    profile_tokens = extract_content_tokens(build_profile_text(profile, id_to_card))
    ranked_items: list[dict[str, object]] = []
    for index, game_id in enumerate(baseline_ids[: settings.top_k], start=1):
        card = id_to_card[game_id]
        card_tokens = extract_content_tokens(card.game_card_text)
        overlapping = sorted(
            token
            for token in (profile_tokens & card_tokens)
            if token not in GENERIC_TOKENS
        )
        explanation = "Mock explanation: kept from baseline ranking for pipeline validation only."
        if overlapping:
            explanation += " Matched tokens: " + ", ".join(overlapping[:3]) + "."
        risks = [card.negative_keywords[0]] if card.negative_keywords else []
        ranked_items.append(
            {
                "rank": index,
                "game_id": game_id,
                "game_title": card.game_title,
                "relevance_score": round(1.0 / index, 6),
                "explanation": explanation,
                "matched_preferences": overlapping[:5],
                "possible_risks": risks,
            }
        )
    return build_user_llm_records(
        user_bundle=user_bundle,
        items=ranked_items,
        id_to_card=id_to_card,
        candidate_size=len(baseline_ids),
        status="mock_validation_only",
        notes="Mock LLM reranking completed for pipeline validation only.",
        llm_mode="mock",
    )


def build_skipped_user_llm_rows(
    selected_users: list[dict[str, object]],
    reason: str,
    llm_mode: str = "real",
) -> list[dict[str, object]]:

    rows: list[dict[str, object]] = []
    for user_bundle in selected_users:
        rows.append(
            {
                "user_id": user_bundle["user_id"],
                "masked_user_id": user_bundle["masked_user_id"],
                "method": "user_llm_reranker",
                "llm_mode": llm_mode,
                "status": reason,
                "rank": None,
                "game_id": None,
                "game_title": None,
                "score": None,
                "candidate_size": len(user_bundle.get("baseline_rows", [])),
                "notes": "User LLM reranking skipped because credentials are missing.",
                "explanation": "",
                "matched_preferences": [],
                "possible_risks": [],
            }
        )
    return rows


def build_user_llm_per_profile_frame(
    selected_users: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
    llm_rows: list[dict[str, object]],
) -> pd.DataFrame:

    baseline_by_user = group_rows_by_user(baseline_rows)
    llm_by_user = group_rows_by_user(llm_rows)
    rows: list[dict[str, object]] = []
    for user_bundle in selected_users:
        user_id = str(user_bundle["user_id"])
        ground_truth_ids = {str(game_id) for game_id in user_bundle.get("ground_truth_game_ids", [])}
        candidate_ids = {str(game_id) for game_id in user_bundle.get("candidate_game_ids", [])}
        baseline_user_rows = sorted(
            [
                row
                for row in baseline_by_user.get(user_id, [])
                if row.get("rank") is not None and row.get("game_id")
            ],
            key=lambda row: int(row["rank"]),
        )
        llm_user_rows = sorted(
            [
                row
                for row in llm_by_user.get(user_id, [])
                if row.get("rank") is not None and row.get("game_id")
            ],
            key=lambda row: int(row["rank"]),
        )
        baseline_rank = best_holdout_rank(baseline_user_rows, ground_truth_ids)
        llm_rank = best_holdout_rank(llm_user_rows, ground_truth_ids)
        llm_mode = first_llm_mode(llm_user_rows, default="unavailable")
        candidate_pool_size = int(
            user_bundle.get(
                "candidate_pool_size",
                len(candidate_ids) if candidate_ids else len(baseline_user_rows),
            )
        )
        baseline_top_k_size = len(baseline_user_rows)
        llm_top_k_size = len(llm_user_rows)
        holdout_in_candidate_pool = bool(candidate_ids & ground_truth_ids) or baseline_rank is not None or llm_rank is not None
        rows.append(
            {
                "masked_user_id": user_bundle["masked_user_id"],
                "baseline_best_holdout_rank": format_rank_value(baseline_rank),
                "llm_best_holdout_rank": format_rank_value(llm_rank),
                "llm_mode": llm_mode,
                "holdout_count": len(ground_truth_ids),
                "candidate_pool_size": int(candidate_pool_size),
                "baseline_top_k_size": int(baseline_top_k_size),
                "llm_top_k_size": int(llm_top_k_size),
                "holdout_in_candidate_pool": bool(holdout_in_candidate_pool),
                "holdout_in_baseline_top_k": bool(baseline_rank is not None),
                "holdout_in_llm_top_k": bool(llm_rank is not None),
                "baseline_hit_rate_at_5": int(is_hit_at_k(baseline_rank, 5)),
                "llm_hit_rate_at_5": int(is_hit_at_k(llm_rank, 5)),
                "baseline_hit_rate_at_10": int(is_hit_at_k(baseline_rank, 10)),
                "llm_hit_rate_at_10": int(is_hit_at_k(llm_rank, 10)),
                "baseline_mrr": format_metric_value(reciprocal_rank(baseline_rank)),
                "llm_mrr": format_metric_value(reciprocal_rank(llm_rank)),
                "baseline_ndcg_at_10": format_metric_value(ndcg_from_rank(baseline_rank, 10)),
                "llm_ndcg_at_10": format_metric_value(ndcg_from_rank(llm_rank, 10)),
                "ground_truth_count": len(ground_truth_ids),
                "llm_status": first_status(llm_by_user.get(user_id, [])),
            }
        )
    return pd.DataFrame(rows)


def build_user_llm_metrics_summary(per_profile_df: pd.DataFrame) -> pd.DataFrame:

    if per_profile_df.empty:
        return pd.DataFrame(
            [
                {
                    "method": "user_tfidf_baseline",
                    "llm_mode": "baseline",
                    "evaluated_profiles": 0,
                    "skipped_profiles": 0,
                    "mean_hit_rate_at_5": 0.0,
                    "mean_hit_rate_at_10": 0.0,
                    "mean_mrr": 0.0,
                    "mean_ndcg_at_10": 0.0,
                },
                {
                    "method": "user_llm_reranker",
                    "llm_mode": "real",
                    "evaluated_profiles": 0,
                    "skipped_profiles": 0,
                    "mean_hit_rate_at_5": 0.0,
                    "mean_hit_rate_at_10": 0.0,
                    "mean_mrr": 0.0,
                    "mean_ndcg_at_10": 0.0,
                },
            ]
        )

    rows: list[dict[str, object]] = []
    baseline_df = per_profile_df.copy()
    rows.append(
        {
            "method": "user_tfidf_baseline",
            "llm_mode": "baseline",
            "evaluated_profiles": int(len(baseline_df)),
            "skipped_profiles": 0,
            "mean_hit_rate_at_5": round(float(baseline_df["baseline_hit_rate_at_5"].mean()), 6),
            "mean_hit_rate_at_10": round(float(baseline_df["baseline_hit_rate_at_10"].mean()), 6),
            "mean_mrr": round(float(baseline_df["baseline_mrr"].mean()), 6),
            "mean_ndcg_at_10": round(float(baseline_df["baseline_ndcg_at_10"].mean()), 6),
        }
    )

    successful_llm_statuses = {"ok", "partial_fallback_to_baseline"}
    llm_df = per_profile_df[
        per_profile_df["llm_status"].astype(str).isin(successful_llm_statuses)
    ].copy()
    if llm_df.empty:
        rows.append(
            {
                "method": "user_llm_reranker",
                "llm_mode": "real",
                "evaluated_profiles": 0,
                "skipped_profiles": int(len(per_profile_df)),
                "mean_hit_rate_at_5": 0.0,
                "mean_hit_rate_at_10": 0.0,
                "mean_mrr": 0.0,
                "mean_ndcg_at_10": 0.0,
            }
        )
    else:
        llm_mode = first_llm_mode(llm_df["llm_mode"].astype(str).tolist(), default="real") if "llm_mode" in llm_df.columns else "real"
        rows.append(
            {
                "method": "user_llm_reranker",
                "llm_mode": llm_mode,
                "evaluated_profiles": int(len(llm_df)),
                "skipped_profiles": int(len(per_profile_df) - len(llm_df)),
                "mean_hit_rate_at_5": round(float(llm_df["llm_hit_rate_at_5"].mean()), 6),
                "mean_hit_rate_at_10": round(float(llm_df["llm_hit_rate_at_10"].mean()), 6),
                "mean_mrr": round(float(llm_df["llm_mrr"].mean()), 6),
                "mean_ndcg_at_10": round(float(llm_df["llm_ndcg_at_10"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def build_user_rank_comparison_frame(per_profile_df: pd.DataFrame) -> pd.DataFrame:

    if per_profile_df.empty:
        return pd.DataFrame(
            columns=[
                "masked_user_id",
                "baseline_best_holdout_rank",
                "llm_best_holdout_rank",
                "llm_mode",
                "rank_delta",
                "holdout_in_candidate_pool",
                "holdout_in_baseline_top_k",
                "holdout_in_llm_top_k",
                "candidate_pool_size",
                "baseline_top_k_size",
                "llm_top_k_size",
                "baseline_hit_at_5",
                "llm_hit_at_5",
                "baseline_hit_at_10",
                "llm_hit_at_10",
                "llm_status",
                "interpretation",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in per_profile_df.iterrows():
        baseline_rank = row["baseline_best_holdout_rank"]
        llm_rank = row["llm_best_holdout_rank"]
        rank_delta = ""
        if baseline_rank != "not_found" and llm_rank != "not_found":
            rank_delta = int(baseline_rank) - int(llm_rank)
        llm_mode = str(row.get("llm_mode", "")).strip() or "unavailable"
        rows.append(
            {
                "masked_user_id": row["masked_user_id"],
                "baseline_best_holdout_rank": baseline_rank,
                "llm_best_holdout_rank": llm_rank,
                "llm_mode": llm_mode,
                "rank_delta": rank_delta,
                "holdout_in_candidate_pool": row.get("holdout_in_candidate_pool", False),
                "holdout_in_baseline_top_k": row.get("holdout_in_baseline_top_k", False),
                "holdout_in_llm_top_k": row.get("holdout_in_llm_top_k", False),
                "candidate_pool_size": row.get("candidate_pool_size", 0),
                "baseline_top_k_size": row.get("baseline_top_k_size", 0),
                "llm_top_k_size": row.get("llm_top_k_size", 0),
                "baseline_hit_at_5": row["baseline_hit_rate_at_5"],
                "llm_hit_at_5": row["llm_hit_rate_at_5"],
                "baseline_hit_at_10": row["baseline_hit_rate_at_10"],
                "llm_hit_at_10": row["llm_hit_rate_at_10"],
                "llm_status": row["llm_status"],
                "interpretation": build_rank_interpretation(
                    baseline_rank,
                    llm_rank,
                    row["llm_status"],
                    llm_mode=llm_mode,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_user_llm_explanation_checks_frame(
    llm_rows: list[dict[str, object]],
    selected_users: list[dict[str, object]],
    game_cards_by_id: dict[str, GameCard],
    top_k: int,
) -> pd.DataFrame:

    profile_by_user = {str(user_bundle["user_id"]): user_bundle for user_bundle in selected_users}
    rows: list[dict[str, object]] = []
    for row in llm_rows:
        user_id = str(row.get("user_id", ""))
        if row.get("rank") is None or not row.get("game_id"):
            continue
        explanation = str(row.get("explanation", "") or "").strip()
        game_id = str(row.get("game_id", ""))
        card = game_cards_by_id.get(game_id)
        profile = profile_by_user.get(user_id, {})
        user_profile_text = build_profile_text(profile.get("profile", {}), game_cards_by_id)
        user_tokens = extract_content_tokens(user_profile_text)
        game_tokens = extract_content_tokens(card.game_card_text if card else "")
        explanation_tokens = extract_content_tokens(explanation)
        ground_truth_ids = {str(game_id) for game_id in profile.get("ground_truth_game_ids", [])}
        rows.append(
            {
                "masked_user_id": row.get("masked_user_id", ""),
                "game_id": game_id,
                "game_title": row.get("game_title", ""),
                "rank": row.get("rank", ""),
                "llm_mode": row.get("llm_mode", ""),
                "has_explanation": bool(explanation),
                "explanation_length": len(explanation.split()),
                "mentions_user_preference_keyword": bool(explanation_tokens & user_tokens),
                "mentions_game_keyword": bool(explanation_tokens & game_tokens),
                "possible_hallucination_flag": len(
                    [
                        token
                        for token in explanation_tokens
                        if token not in user_tokens and token not in game_tokens and token not in GENERIC_TOKENS
                    ]
                ) >= 3,
                "status": row.get("status", ""),
                "profile_summary": summarize_user_profile(profile.get("profile", {}), game_cards_by_id),
                "recommended_game": row.get("game_title", ""),
                "holdout_in_topk": bool(game_id in ground_truth_ids and int(row.get("rank", 0) or 0) <= top_k),
            }
        )
    return pd.DataFrame(rows)


def save_user_llm_explanation_outputs(settings: Settings, frame: pd.DataFrame) -> None:

    mock_mode = bool(not frame.empty and "llm_mode" in frame.columns and (frame["llm_mode"].astype(str) == "mock").any())
    lines = [
        "# User LLM Explanation Checks",
        "",
        "These checks are heuristic and do not prove factual correctness.",
        "Mock LLM results are validation-only and must not be interpreted as scientific performance."
        if mock_mode
        else "This report reflects a controlled pilot subset."
        if not frame.empty
        else "",
        "",
    ]
    if frame.empty:
        lines.append("_No explanation rows available._")
    else:
        lines.append(dataframe_to_markdown(frame.head(10)))
    settings.user_llm_explanation_checks_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_user_llm_examples_markdown(
    settings: Settings,
    llm_rows: list[dict[str, object]],
    selected_users: list[dict[str, object]],
    game_cards_by_id: dict[str, GameCard],
) -> None:

    profile_by_user = {str(user_bundle["user_id"]): user_bundle for user_bundle in selected_users}
    mock_mode = any(str(row.get("llm_mode", "")).strip() == "mock" for row in llm_rows)
    lines = [
        "# User LLM Explanation Examples",
        "",
        "These examples are for qualitative inspection and are not a factuality audit.",
        "Mock LLM results are validation-only and must not be interpreted as scientific performance."
        if mock_mode
        else "These examples come from a controlled pilot subset."
        if llm_rows
        else "",
        "",
    ]
    shown = 0
    for row in llm_rows:
        if shown >= 10:
            break
        if row.get("rank") is None or not row.get("game_id"):
            continue
        user_bundle = profile_by_user.get(str(row.get("user_id", "")), {})
        profile_summary = summarize_user_profile(user_bundle.get("profile", {}), game_cards_by_id)
        lines.extend(
            [
                f"## {row.get('masked_user_id', '')} - {row.get('game_title', '')}",
                "",
                f"- Profile summary: {profile_summary}",
                f"- Recommended game: {row.get('game_title', '')}",
                f"- LLM explanation: {row.get('explanation', '')}",
                f"- Matched preferences: {format_list(row.get('matched_preferences', []))}",
                f"- Possible risks: {format_list(row.get('possible_risks', []))}",
                f"- Holdout in top-k: {row.get('game_id') in set(user_bundle.get('ground_truth_game_ids', []))}",
                "",
            ]
        )
        shown += 1
    if shown == 0:
        lines.append("_No LLM examples available._")
    settings.user_llm_explanation_examples_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_user_llm_metrics_table(settings: Settings, metrics_df: pd.DataFrame) -> None:

    mock_mode = bool(
        not metrics_df.empty
        and "llm_mode" in metrics_df.columns
        and (metrics_df["llm_mode"].astype(str) == "mock").any()
    )
    lines = ["# User LLM Metrics Table", ""]
    lines.append(
        "Mock LLM results are validation-only and must not be interpreted as scientific performance."
        if mock_mode
        else "LLM reranking was only evaluated on a limited pilot subset."
    )
    lines.append("")
    if metrics_df.empty:
        lines.append("_No LLM metrics available._")
    else:
        llm_mode_values = metrics_df["llm_mode"] if "llm_mode" in metrics_df.columns else pd.Series([""] * len(metrics_df))
        table = pd.DataFrame(
            {
                "Method": metrics_df["method"],
                "LLM mode": llm_mode_values,
                "HitRate@5": metrics_df["mean_hit_rate_at_5"].map(format_metric),
                "HitRate@10": metrics_df["mean_hit_rate_at_10"].map(format_metric),
                "MRR": metrics_df["mean_mrr"].map(format_metric),
                "NDCG@10": metrics_df["mean_ndcg_at_10"].map(format_metric),
                "Evaluated users": metrics_df["evaluated_profiles"],
            }
        )
        lines.append(dataframe_to_markdown(table))
    settings.user_llm_metrics_table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_user_llm_pilot_summary_markdown(settings: Settings, summary: dict[str, object]) -> None:

    mock_mode = bool(summary.get("mock_mode", False))
    lines = [
        "# User LLM Pilot Summary",
        "",
        "Mock LLM results are validation-only and must not be interpreted as scientific performance."
        if mock_mode
        else "This is a controlled pilot on a small subset of user-based evaluation splits.",
        "",
        f"- Status: {summary.get('status', 'unknown')}",
        f"- Selected users: {summary.get('selected_users', 0)}",
        f"- Candidates per user: {summary.get('candidates_per_user', 0)}",
        f"- Baseline evaluated users: {summary.get('baseline_evaluated_users', 0)}",
        f"- LLM evaluated users: {summary.get('llm_evaluated_users', 0)}",
        f"- LLM improved users: {summary.get('improved_users', 0)}",
        f"- LLM worsened users: {summary.get('worsened_users', 0)}",
        f"- LLM unchanged users: {summary.get('unchanged_users', 0)}",
        f"- LLM mode: {summary.get('llm_mode', 'real')}",
        f"- LLM provider: {summary.get('llm_provider', 'openai')}",
        f"- Response language: {summary.get('response_language', 'ru')}",
        f"- Mock mode: {mock_mode}",
        f"- Real API calls: {summary.get('real_api_calls', 0)}",
        f"- Real API calls total: {summary.get('real_api_calls_total', summary.get('real_api_calls', 0))}",
        f"- Not for scientific metrics: {summary.get('not_for_scientific_metrics', False)}",
        f"- Holdout in candidate pool users: {summary.get('holdout_in_candidate_pool_users', 0)}",
        f"- Holdout in baseline top-k users: {summary.get('holdout_in_baseline_top_k_users', 0)}",
        f"- Holdout in LLM top-k users: {summary.get('holdout_in_llm_top_k_users', 0)}",
        f"- LLM valid records: {summary.get('llm_valid_records', 0)}",
        f"- Users with any valid LLM records: {summary.get('users_with_any_valid_llm_records', 0)}",
        f"- Users with only fallback: {summary.get('users_with_only_fallback', 0)}",
        f"- Users with partial fallback: {summary.get('users_with_partial_fallback', 0)}",
        f"- Users with schema errors: {summary.get('users_with_schema_errors', 0)}",
        f"- Provider preflight OK: {summary.get('provider_preflight_ok', True)}",
        f"- Provider preflight status: `{summary.get('provider_preflight_status', 'ok') or 'ok'}`",
        f"- Token requests attempted: {summary.get('token_requests_attempted', 0)}",
        f"- Completion requests attempted: {summary.get('completion_requests_attempted', 0)}",
        f"- Provider failed users: {summary.get('provider_failed_users', 0)}",
        "",
        "## Failure diagnostics",
        f"- Failed users: {summary.get('failed_user_count', summary.get('failed_users', 0))}",
        f"- Failed records: {summary.get('failed_record_count', summary.get('failed_records', 0))}",
        f"- Fallback users: {summary.get('fallback_user_count', summary.get('fallback_users', 0))}",
        f"- Fallback records: {summary.get('fallback_record_count', summary.get('fallback_records', 0))}",
        f"- Valid LLM records: {summary.get('llm_valid_records', 0)}",
        f"- Provider failed users: {summary.get('provider_failed_users', 0)}",
        f"- Invalid JSON count: {summary.get('invalid_json_count', 0)}",
        f"- Empty response count: {summary.get('empty_response_count', 0)}",
        f"- Schema validation error count: {summary.get('schema_validation_error_count', 0)}",
        f"- Users with schema errors: {summary.get('users_with_schema_errors', 0)}",
        f"- Candidate pool error count: {summary.get('candidate_pool_error_count', 0)}",
        f"- Response preview saved count: {summary.get('response_preview_saved_count', 0)}",
        f"- LLM responses received: {max(0, int(summary.get('selected_users', 0)) - int(summary.get('empty_response_count', 0)))}",
        f"- LLM extraction failures: {summary.get('invalid_json_count', 0) + summary.get('schema_validation_error_count', 0) + summary.get('candidate_pool_error_count', 0)}",
        "",
        "## Metrics",
    ]
    baseline_metrics = summary.get("baseline_metrics", {}) or {}
    llm_metrics = summary.get("llm_metrics", {}) or {}
    if baseline_metrics:
        lines.extend(
            [
                "- Baseline HitRate@5: " + format_metric(baseline_metrics.get("mean_hit_rate_at_5", "")),
                "- Baseline HitRate@10: " + format_metric(baseline_metrics.get("mean_hit_rate_at_10", "")),
                "- Baseline MRR: " + format_metric(baseline_metrics.get("mean_mrr", "")),
                "- Baseline NDCG@10: " + format_metric(baseline_metrics.get("mean_ndcg_at_10", "")),
            ]
        )
    if llm_metrics:
        lines.extend(
            [
                "- LLM HitRate@5: " + format_metric(llm_metrics.get("mean_hit_rate_at_5", "")),
                "- LLM HitRate@10: " + format_metric(llm_metrics.get("mean_hit_rate_at_10", "")),
                "- LLM MRR: " + format_metric(llm_metrics.get("mean_mrr", "")),
                "- LLM NDCG@10: " + format_metric(llm_metrics.get("mean_ndcg_at_10", "")),
            ]
        )
    lines.extend(
        [
            "",
        "## Limitations",
        "This is a controlled pilot on a small subset of user-based evaluation splits. "
        "The results should not be presented as a full-experiment conclusion."
        + (" Mock metrics are validation-only." if mock_mode else ""),
        "",
        "## Candidate Selection",
        f"- Selected meaningful pilot users: {summary.get('selected_meaningful_users', 0)}",
        f"- Failed LLM users: {summary.get('failed_users', 0)}",
        f"- Partial fallback users: {summary.get('partial_fallback_users', 0)}",
        "",
        ]
    )
    settings.user_llm_pilot_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_user_llm_mock_validation_report(
    settings: Settings,
    summary: dict[str, object],
    validation_summary: dict[str, object],
    llm_rows: list[dict[str, object]],
) -> None:

    mock_mode = True
    generated_files = [
        settings.user_llm_prompt_preview_markdown_path,
        settings.user_llm_prompt_preview_json_path,
        settings.user_llm_prompt_preview_tiny_markdown_path,
        settings.user_llm_prompt_preview_tiny_json_path,
        settings.user_llm_results_path,
        settings.user_llm_reranking_summary_path,
        settings.user_llm_validation_summary_path,
        settings.user_llm_mock_validation_summary_path,
        settings.user_llm_per_profile_results_path,
        settings.user_llm_metrics_summary_path,
        settings.user_rank_comparison_path,
        settings.user_rank_comparison_markdown_path,
        settings.user_llm_explanation_checks_csv_path,
        settings.user_llm_explanation_checks_markdown_path,
        settings.user_llm_explanation_examples_markdown_path,
        settings.llm_pilot_candidate_user_report_csv_path,
        settings.llm_pilot_candidate_user_report_markdown_path,
        settings.user_llm_metrics_table_path,
        settings.user_llm_pilot_summary_path,
    ]
    lines = [
        "# Mock LLM Validation Report",
        "",
        "Mock LLM results are validation-only and must not be interpreted as scientific performance.",
        "",
        f"- Number of users processed: {validation_summary.get('users_completed', 0)}",
        f"- Number of recommendations generated: {len([row for row in llm_rows if row.get('rank') is not None and row.get('game_id')])}",
        f"- API calls made: {summary.get('real_api_calls', 0)}",
        f"- LLM mode: {summary.get('llm_mode', 'mock')}",
        f"- All game IDs inside candidate pools: {validation_summary.get('all_game_ids_inside_candidate_pool', True)}",
        f"- Mock mode: {mock_mode}",
        f"- Not for scientific metrics: {summary.get('not_for_scientific_metrics', True)}",
        "",
        "## Generated Files",
        *[f"- `{path.relative_to(settings.project_root)}`" for path in generated_files if path.exists()],
        "",
        "## Warnings",
        "- Mock outputs are for pipeline validation only.",
        "- Mock metrics must not be used as evidence of LLM effectiveness.",
        "",
    ]
    settings.user_llm_mock_validation_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_user_llm_failure_report(
    settings: Settings,
    failure_details: list[dict[str, object]],
    summary: dict[str, object],
    validation_summary: dict[str, object],
    provider_preflight_report: dict[str, object] | None = None,
) -> None:

    provider_preflight_report = provider_preflight_report or {}
    payload = {
        "status": "failed"
        if failure_details or not bool(provider_preflight_report.get("provider_preflight_ok", True))
        else "ok",
        "provider": summary.get("llm_provider", ""),
        "model": summary.get("llm_model", getattr(settings, "llm_model", "")),
        "llm_mode": summary.get("llm_mode", "real"),
        "masked_failures": failure_details,
        "dominant_error_type": (
            Counter(str(item.get("error_type", "")) for item in failure_details).most_common(1)[0][0]
            if failure_details
            else str(provider_preflight_report.get("provider_preflight_error_type", "") or validation_summary.get("provider_preflight_error_type", ""))
        ),
        "users_requested": int(validation_summary.get("users_requested", 0)),
        "users_completed": int(validation_summary.get("users_completed", 0)),
        "users_failed": int(validation_summary.get("users_failed", 0)),
        "invalid_json_count": int(validation_summary.get("invalid_json_count", 0)),
        "invalid_game_id_count": int(validation_summary.get("invalid_game_id_count", 0)),
        "fallback_count": int(validation_summary.get("fallback_count", 0)),
        "empty_response_count": int(validation_summary.get("empty_response_count", 0)),
        "schema_validation_error_count": int(validation_summary.get("schema_validation_error_count", 0)),
        "candidate_pool_error_count": int(validation_summary.get("candidate_pool_error_count", 0)),
        "response_preview_saved_count": int(validation_summary.get("response_preview_saved_count", 0)),
        "all_game_ids_inside_candidate_pool": bool(
            validation_summary.get("all_game_ids_inside_candidate_pool", True)
        ),
        "provider_failed_users": int(validation_summary.get("provider_failed_users", 0)),
        "provider_preflight_ok": bool(provider_preflight_report.get("provider_preflight_ok", True)) if provider_preflight_report else bool(validation_summary.get("provider_preflight_ok", True)),
        "provider_preflight_status": str(provider_preflight_report.get("provider_preflight_status", validation_summary.get("provider_preflight_status", ""))),
        "provider_preflight_error_type": str(provider_preflight_report.get("provider_preflight_error_type", validation_summary.get("provider_preflight_error_type", ""))),
        "provider_preflight_error_message_short": str(provider_preflight_report.get("provider_preflight_error_message_short", validation_summary.get("provider_preflight_error_message_short", ""))),
        "token_requests_attempted": int(validation_summary.get("token_requests_attempted", 0)),
        "completion_requests_attempted": int(validation_summary.get("completion_requests_attempted", 0)),
        "real_api_calls_total": int(validation_summary.get("real_api_calls_total", 0)),
    }
    settings.user_llm_failure_report_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# User LLM Failure Report",
        "",
        "This report is safe for debugging and does not expose API keys, tokens, or raw prompts.",
        "",
        f"- Status: {payload['status']}",
        f"- Provider: `{payload['provider']}`",
        f"- Model: `{payload['model']}`",
        f"- LLM mode: `{payload['llm_mode']}`",
        f"- Users requested: {payload['users_requested']}",
        f"- Users completed: {payload['users_completed']}",
        f"- Users failed: {payload['users_failed']}",
        f"- Dominant error type: `{payload['dominant_error_type'] or 'none'}`",
        f"- Invalid JSON count: {payload['invalid_json_count']}",
        f"- Invalid game ID count: {payload['invalid_game_id_count']}",
        f"- Fallback count: {payload['fallback_count']}",
        f"- Empty response count: {payload['empty_response_count']}",
        f"- Schema validation error count: {payload['schema_validation_error_count']}",
        f"- Candidate pool error count: {payload['candidate_pool_error_count']}",
        f"- Response preview saved count: {payload['response_preview_saved_count']}",
        f"- All game IDs inside candidate pool: {payload['all_game_ids_inside_candidate_pool']}",
        f"- Provider failed users: {payload['provider_failed_users']}",
        f"- Provider preflight OK: {payload['provider_preflight_ok']}",
        f"- Provider preflight status: `{payload['provider_preflight_status'] or 'unknown'}`",
        f"- Provider preflight error type: `{payload['provider_preflight_error_type'] or 'none'}`",
        f"- Provider preflight error message: `{payload['provider_preflight_error_message_short'] or 'none'}`",
        f"- Token requests attempted: {payload['token_requests_attempted']}",
        f"- Completion requests attempted: {payload['completion_requests_attempted']}",
        f"- Real API calls total: {payload['real_api_calls_total']}",
        "",
        "## Failure details",
    ]
    if provider_preflight_report and not provider_preflight_report.get("provider_preflight_ok", True):
        lines.extend(
            [
                "## Provider Preflight Failure",
                "",
                f"- Provider preflight status: `{provider_preflight_report.get('provider_preflight_status', 'token_error')}`",
                f"- Provider preflight error type: `{provider_preflight_report.get('provider_preflight_error_type', 'token_error')}`",
                f"- Provider preflight error message: `{provider_preflight_report.get('provider_preflight_error_message_short', 'none')}`",
                f"- Token requests attempted: {provider_preflight_report.get('token_requests_attempted', 0)}",
                f"- Completion requests attempted: 0",
                "",
            ]
        )
    if failure_details:
        lines.append(
            dataframe_to_markdown(
                pd.DataFrame(
                    failure_details,
                    columns=[
                        "masked_user_id",
                        "provider",
                        "model",
                        "llm_mode",
                        "api_call_attempted",
                        "token_obtained",
                        "http_status",
                        "error_type",
                        "error_message_short",
                        "response_received",
                        "response_empty",
                        "response_preview_safe",
                        "json_parse_attempted",
                        "json_parse_error_short",
                        "schema_validation_error_short",
                        "candidate_pool_validation_error_short",
                        "fallback_applied",
                    ],
                )
            )
        )
    else:
        lines.append("_No failures were recorded._")
    settings.user_llm_failure_report_markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_user_llm_schema_error_report(
    settings: Settings,
    failure_details: list[dict[str, object]],
    validation_summary: dict[str, object],
) -> None:

    schema_rows = [
        detail
        for detail in failure_details
        if str(detail.get("schema_validation_error_short", "")).strip()
    ]
    if not schema_rows and int(validation_summary.get("schema_validation_error_count", 0)) <= 0:
        return

    def extract_missing_fields_from_error(message: str) -> list[str]:
        text = str(message or "")
        known_fields = {
            "recommendations",
            "rank",
            "game_id",
            "game_title",
            "relevance_score",
            "explanation",
            "matched_preferences",
            "possible_risks",
        }
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        fields = sorted(token for token in tokens if token in known_fields)
        if not fields and text:
            fields = ["unknown"]
        return fields

    payload = {
        "status": "schema_error",
        "schema_validation_error_count": int(validation_summary.get("schema_validation_error_count", len(schema_rows))),
        "records": [
            {
                "masked_user_id": detail.get("masked_user_id", ""),
                "missing_fields": extract_missing_fields_from_error(detail.get("schema_validation_error_short", "")),
                "malformed_records_count": 1,
                "response_preview_safe": str(detail.get("response_preview_safe", "")),
                "fallback_applied": bool(detail.get("fallback_applied", False)),
            }
            for detail in schema_rows
        ],
    }
    settings.user_llm_schema_error_report_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# User LLM Schema Error Report",
        "",
        "This report is safe for debugging malformed LLM output.",
        "",
        f"- Schema validation error count: {payload['schema_validation_error_count']}",
        "",
    ]
    if payload["records"]:
        lines.append(
            dataframe_to_markdown(
                pd.DataFrame(
                    payload["records"],
                    columns=[
                        "masked_user_id",
                        "missing_fields",
                        "malformed_records_count",
                        "response_preview_safe",
                        "fallback_applied",
                    ],
                )
            )
        )
    else:
        lines.append("_No schema errors were recorded._")
    settings.user_llm_schema_error_report_markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_user_llm_prompt_preview(settings: Settings, preview_records: list[dict[str, object]]) -> None:

    write_user_llm_prompt_preview_variant(settings, preview_records, variant_suffix="")


def write_user_llm_prompt_preview_variant(
    settings: Settings,
    preview_records: list[dict[str, object]],
    variant_suffix: str,
) -> None:

    json_path = settings.user_llm_prompt_preview_json_path
    markdown_path = settings.user_llm_prompt_preview_markdown_path
    if variant_suffix == "_tiny":
        json_path = settings.user_llm_prompt_preview_tiny_json_path
        markdown_path = settings.user_llm_prompt_preview_tiny_markdown_path
    elif variant_suffix == "_10_gigachat":
        json_path = settings.user_llm_prompt_preview_10_gigachat_json_path
        markdown_path = settings.user_llm_prompt_preview_10_gigachat_markdown_path

    json_path.write_text(
        json.dumps(preview_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# User LLM Prompt Preview",
        "",
        "This preview does not call the API.",
        "",
    ]
    for index, record in enumerate(preview_records, start=1):
        lines.extend(
            [
                f"## Prompt {index}",
                "",
                f"- Masked user id: {record['masked_user_id']}",
                f"- Provider: {record.get('provider', 'openai')}",
                f"- Model: {record.get('model', '')}",
                f"- Response language: {record.get('response_language', 'ru')}",
                f"- Top-k: {record.get('top_k', 10)}",
                f"- Candidate count: {record['candidate_count']}",
                f"- Prompt characters: {record['prompt_characters']}",
                f"- Prompt too long: {record['prompt_too_long']}",
                "",
                "```text",
                str(record["prompt"]),
                "```",
                "",
            ]
        )
    if not preview_records:
        lines.append("_No prompt previews available._")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_user_llm_summary(
    settings: Settings,
    selected_users: list[dict[str, object]],
    llm_rows: list[dict[str, object]],
    status: str,
    llm_ran: bool,
    llm_credentials_configured: bool,
    llm_mode: str,
    llm_provider: str,
    mock_llm_ran: bool,
    real_api_calls: int,
    not_for_scientific_metrics: bool,
    validation_summary: dict[str, object] | None = None,
) -> dict[str, object]:

    validation_summary = validation_summary or {}
    status_counts = Counter(str(row.get("status", "")) for row in llm_rows)
    rows_by_user = group_rows_by_user(llm_rows)
    llm_rows_by_user = rows_by_user
    user_status_counts = Counter(first_status(rows) for rows in rows_by_user.values() if rows)
    valid_record_statuses = {"ok", "partial_fallback_to_baseline"}
    valid_record_count = int(
        sum(1 for row in llm_rows if str(row.get("status", "")) in valid_record_statuses and row.get("rank") is not None and row.get("game_id"))
    )
    reranked_users = int(
        sum(
            1
            for rows in rows_by_user.values()
            if first_status(rows) in {"ok", "partial_fallback_to_baseline"}
            and any(row.get("rank") is not None and row.get("game_id") for row in rows)
        )
    )
    skipped_users = int(user_status_counts.get("skipped_no_credentials", 0))
    failed_users = int(user_status_counts.get("failed", 0))
    ok_users = int(user_status_counts.get("ok", 0))
    partial_fallback_users = int(user_status_counts.get("partial_fallback_to_baseline", 0))
    fallback_users = int(user_status_counts.get("fallback_to_baseline", 0))
    provider_failed_users = int(validation_summary.get("provider_failed_users", 0))
    ok_records = int(status_counts.get("ok", 0))
    partial_fallback_records = int(status_counts.get("partial_fallback_to_baseline", 0))
    failed_records = int(status_counts.get("failed", 0))
    fallback_records = int(status_counts.get("fallback_to_baseline", 0))
    schema_error_user_count = int(validation_summary.get("schema_error_user_count", 0))
    users_with_any_valid_llm_records = int(
        sum(
            1
            for rows in rows_by_user.values()
            if any(str(row.get("status", "")) in valid_record_statuses for row in rows)
        )
    )
    users_with_only_fallback = int(
        sum(
            1
            for rows in rows_by_user.values()
            if rows
            and all(str(row.get("status", "")) == "fallback_to_baseline" for row in rows)
        )
    )
    users_with_partial_fallback = partial_fallback_users
    users_with_schema_errors = schema_error_user_count
    holdout_in_candidate_pool_users = 0
    holdout_in_baseline_top_k_users = 0
    holdout_in_llm_top_k_users = 0
    selected_meaningful_users = 0
    for user_bundle in selected_users:
        user_id = str(user_bundle.get("user_id", ""))
        ground_truth_ids = {str(game_id) for game_id in user_bundle.get("ground_truth_game_ids", [])}
        candidate_ids = {str(game_id) for game_id in user_bundle.get("candidate_game_ids", [])}
        baseline_rows_for_user = [
            row
            for row in user_bundle.get("baseline_rows", [])
            if row.get("rank") is not None and row.get("game_id")
        ]
        baseline_rank = best_holdout_rank(baseline_rows_for_user, ground_truth_ids)
        llm_rows_for_user = [
            row
            for row in llm_rows_by_user.get(user_id, [])
            if row.get("rank") is not None and row.get("game_id")
        ]
        llm_rank = best_holdout_rank(llm_rows_for_user, ground_truth_ids)
        if (ground_truth_ids & candidate_ids) or baseline_rank is not None or llm_rank is not None:
            holdout_in_candidate_pool_users += 1
        if baseline_rank is not None:
            holdout_in_baseline_top_k_users += 1
        if llm_rank is not None:
            holdout_in_llm_top_k_users += 1
        if (ground_truth_ids & candidate_ids) and baseline_rank is not None and baseline_rank > 1:
            selected_meaningful_users += 1
    return {
        "experiment_name": "steam_reviews_balanced_subset_llm_pilot",
        "status": status,
        "llm_requested": True,
        "llm_ran": llm_ran,
        "llm_mode": llm_mode,
        "llm_provider": llm_provider,
        "mock_mode": llm_mode == "mock",
        "mock_llm_ran": mock_llm_ran,
        "real_api_calls": int(real_api_calls),
        "real_api_calls_total": int(validation_summary.get("real_api_calls_total", real_api_calls)),
        "llm_credentials_configured": llm_credentials_configured,
        "users_requested": len(selected_users),
        "users_completed": int(validation_summary.get("users_completed", len(rows_by_user) - failed_users)),
        "users_failed": int(validation_summary.get("users_failed", failed_users)),
        "failed_user_count": failed_users,
        "failed_record_count": failed_records,
        "fallback_user_count": fallback_users,
        "fallback_record_count": fallback_records,
        "invalid_json_count": int(validation_summary.get("invalid_json_count", 0)),
        "empty_response_count": int(validation_summary.get("empty_response_count", 0)),
        "schema_validation_error_count": int(validation_summary.get("schema_validation_error_count", 0)),
        "candidate_pool_error_count": int(validation_summary.get("candidate_pool_error_count", 0)),
        "response_preview_saved_count": int(validation_summary.get("response_preview_saved_count", 0)),
        "selected_users": len(selected_users),
        "reranked_users": reranked_users,
        "evaluated_llm_users": reranked_users,
        "skipped_users": skipped_users,
        "failed_users": failed_users,
        "ok_users": ok_users,
        "partial_fallback_users": partial_fallback_users,
        "fallback_users": fallback_users,
        "ok_records": ok_records,
        "llm_valid_records": valid_record_count,
        "partial_fallback_records": partial_fallback_records,
        "failed_records": failed_records,
        "fallback_records": fallback_records,
        "users_with_any_valid_llm_records": users_with_any_valid_llm_records,
        "users_with_only_fallback": users_with_only_fallback,
        "users_with_partial_fallback": users_with_partial_fallback,
        "users_with_schema_errors": users_with_schema_errors,
        "holdout_in_candidate_pool_users": int(holdout_in_candidate_pool_users),
        "holdout_in_baseline_top_k_users": int(holdout_in_baseline_top_k_users),
        "holdout_in_llm_top_k_users": int(holdout_in_llm_top_k_users),
        "selected_meaningful_users": int(selected_meaningful_users),
        "max_llm_users": settings.max_llm_users,
        "max_llm_candidates": settings.max_llm_candidates,
        "candidate_pool_size": settings.user_candidate_pool_size,
        "status_counts": dict(status_counts),
        "not_for_scientific_metrics": not_for_scientific_metrics,
        "validation_summary": validation_summary or {},
        "warnings": [],
    }


def build_mock_validation_summary(
    *,
    users_requested: int,
    users_completed: int,
    users_failed: int,
    invalid_json_count: int,
    invalid_game_id_count: int,
    fallback_count: int,
    all_game_ids_inside_candidate_pool: bool,
) -> dict[str, object]:

    return {
        "users_requested": users_requested,
        "users_completed": users_completed,
        "users_failed": users_failed,
        "invalid_json_count": invalid_json_count,
        "invalid_game_id_count": invalid_game_id_count,
        "fallback_count": fallback_count,
        "all_game_ids_inside_candidate_pool": all_game_ids_inside_candidate_pool,
        "mock_mode": True,
        "real_api_calls": 0,
        "not_for_scientific_metrics": True,
    }


def write_json_report(path: Path, payload: dict[str, object]) -> None:

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_json_fenced_blocks(content: str) -> list[str]:

    blocks: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE):
        candidate = match.group(1).strip()
        if candidate:
            blocks.append(candidate)
    return blocks


def extract_balanced_json_object(content: str) -> str | None:

    text = content or ""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : index + 1].strip()
                        if '"recommendations"' in candidate:
                            return candidate
                        break
        start = text.find("{", start + 1)
    return None


def parse_llm_response_with_diagnostics(
    content: str,
    *,
    save_preview: bool = True,
) -> tuple[UserLLMResponse | None, dict[str, object]]:

    text = str(content or "")
    stripped = text.strip()
    diagnostics: dict[str, object] = {
        "response_received": bool(stripped),
        "response_empty": not bool(stripped),
        "response_preview_safe": truncate_text(stripped, 500) if save_preview and stripped else "",
        "json_parse_attempted": False,
        "json_parse_error_short": "",
        "schema_validation_error_short": "",
        "candidate_pool_validation_error_short": "",
        "invalid_json": False,
        "parse_source": "",
    }
    if not stripped:
        diagnostics["json_parse_attempted"] = True
        diagnostics["json_parse_error_short"] = "empty_response"
        diagnostics["invalid_json"] = True
        return None, diagnostics

    candidate_payloads: list[tuple[str, str]] = [("direct", stripped)]
    for block in extract_json_fenced_blocks(stripped):
        if block not in {payload for _, payload in candidate_payloads}:
            candidate_payloads.append(("fenced", block))
    balanced = extract_balanced_json_object(stripped)
    if balanced and balanced not in {payload for _, payload in candidate_payloads}:
        candidate_payloads.append(("balanced", balanced))

    payload: object | None = None
    parse_error: Exception | None = None
    for source, candidate in candidate_payloads:
        diagnostics["json_parse_attempted"] = True
        try:
            payload = json.loads(candidate)
            diagnostics["parse_source"] = source
            break
        except json.JSONDecodeError as exc:
            parse_error = exc
            continue

    if payload is None:
        diagnostics["json_parse_error_short"] = truncate_text(str(parse_error or "invalid_json"), 200)
        diagnostics["invalid_json"] = True
        return None, diagnostics

    try:
        try:
            parsed = UserLLMResponse.model_validate(payload)
        except AttributeError:
            parsed = UserLLMResponse.parse_obj(payload)
        return parsed, diagnostics
    except ValidationError as exc:
        diagnostics["schema_validation_error_short"] = truncate_text(str(exc), 220)
        return None, diagnostics


def parse_llm_response(content: str) -> UserLLMResponse:

    parsed, diagnostics = parse_llm_response_with_diagnostics(content, save_preview=False)
    if parsed is not None:
        return parsed
    if diagnostics.get("schema_validation_error_short"):
        raise ValueError(f"Invalid LLM response schema: {diagnostics['schema_validation_error_short']}")
    raise ValueError(
        f"LLM response did not contain valid JSON: {diagnostics.get('json_parse_error_short', 'invalid_json')}"
    )


def keep_valid_reranked_items(
    items: list[UserLLMRecommendationItem],
    baseline_ids: list[str],
) -> list[dict[str, object]]:

    allowed = set(baseline_ids)
    seen: set[str] = set()
    valid_items: list[dict[str, object]] = []
    for item in items:
        if item.game_id not in allowed or item.game_id in seen:
            continue
        seen.add(item.game_id)
        valid_items.append(
            {
                "rank": item.rank,
                "game_id": item.game_id,
                "game_title": item.game_title,
                "relevance_score": item.relevance_score,
                "explanation": item.explanation,
                "matched_preferences": item.matched_preferences,
                "possible_risks": item.possible_risks,
            }
        )
    return valid_items


def group_rows_by_user(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        user_id = str(row.get("user_id", ""))
        grouped.setdefault(user_id, []).append(row)
    return grouped


def best_holdout_rank(
    rows: list[dict[str, object]],
    ground_truth_ids: set[str],
) -> int | None:

    ranked_rows = sorted(
        [row for row in rows if row.get("rank") is not None and row.get("game_id")],
        key=lambda row: int(row["rank"]),
    )
    for row in ranked_rows:
        if str(row.get("game_id", "")) in ground_truth_ids:
            return int(row["rank"])
    return None


def reciprocal_rank(rank: int | None) -> float:

    return 0.0 if rank is None else 1.0 / rank


def ndcg_from_rank(rank: int | None, k: int) -> float:

    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def is_hit_at_k(rank: int | None, k: int) -> bool:

    return bool(rank is not None and rank <= k)


def format_rank_value(rank: int | None) -> int | str:

    return "not_found" if rank is None else int(rank)


def format_metric_value(value: float) -> float:

    return round(float(value), 6)


def first_status(rows: list[dict[str, object]]) -> str:

    if not rows:
        return "missing_results"
    return str(rows[0].get("status", "missing_results"))


def first_llm_mode(rows: list[object], default: str = "real") -> str:

    for row in rows:
        if isinstance(row, dict):
            llm_mode = str(row.get("llm_mode", "")).strip()
        else:
            llm_mode = str(row).strip()
        if llm_mode:
            return llm_mode
    return default


def summarize_user_profile(profile: dict[str, object], game_cards_by_id: dict[str, GameCard]) -> str:

    if not profile:
        return "No profile available."
    positive_game_ids = [str(game_id) for game_id in profile.get("positive_game_ids", [])]
    negative_game_ids = [str(game_id) for game_id in profile.get("negative_game_ids", [])]
    positive_titles = [game_cards_by_id[game_id].game_title for game_id in positive_game_ids if game_id in game_cards_by_id]
    negative_titles = [game_cards_by_id[game_id].game_title for game_id in negative_game_ids if game_id in game_cards_by_id]
    return (
        f"{profile.get('review_count', 0)} reviews, "
        f"{profile.get('positive_review_count', 0)} positive; "
        f"liked: {truncate_text(', '.join(positive_titles) or 'none', 120)}; "
        f"disliked: {truncate_text(', '.join(negative_titles) or 'none', 120)}"
    )


def build_rank_interpretation(
    baseline_rank: int | str,
    llm_rank: int | str,
    llm_status: str,
    *,
    llm_mode: str = "real",
) -> str:

    if llm_mode == "mock":
        return "Mock LLM validation mode preserved the controlled ranking for pipeline testing."
    if llm_status == "skipped_no_credentials":
        return "LLM reranking was skipped because credentials were missing."
    if baseline_rank == "not_found" and llm_rank == "not_found":
        return "No holdout game appeared in either ranking."
    if baseline_rank == "not_found":
        return "LLM found a holdout game that the baseline did not rank in top-k."
    if llm_rank == "not_found":
        return "The baseline ranked a holdout game, but the LLM did not."
    delta = int(baseline_rank) - int(llm_rank)
    if delta > 0:
        return f"LLM improved the best holdout rank by {delta} positions."
    if delta < 0:
        return f"LLM worsened the best holdout rank by {abs(delta)} positions."
    return "LLM preserved the best holdout rank."


def format_list(values: list[str] | object) -> str:

    if not isinstance(values, list) or not values:
        return "none"
    return ", ".join(str(value) for value in values)


def extract_content_tokens(text: str) -> set[str]:

    return {
        token
        for token in pd.Series([text]).astype(str).str.lower().str.findall(r"[a-zа-яё]{3,}").iloc[0]
    }


GENERIC_TOKENS = {
    "game",
    "games",
    "user",
    "users",
    "recommend",
    "recommended",
    "play",
    "like",
    "likes",
    "good",
    "bad",
    "best",
    "fun",
    "time",
    "experience",
    "story",
    "combat",
    "mechanics",
}


def _allow_llm_skip(settings: Settings) -> bool:
    return bool(getattr(settings, "allow_llm_skip", True))


def load_json_if_exists(path: Path) -> dict[str, object]:

    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_user_llm_pilot_summary(
    settings: Settings,
    selected_users: list[dict[str, object]],
    per_profile_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
    rank_comparison_df: pd.DataFrame,
    explanation_checks_df: pd.DataFrame,
    llm_rows: list[dict[str, object]],
    validation_summary: dict[str, object] | None = None,
) -> dict[str, object]:

    validation_summary = validation_summary or {}
    llm_status_counts = Counter(str(row.get("status", "")) for row in llm_rows)
    llm_rows_by_user = group_rows_by_user(llm_rows)
    user_status_counts = Counter(
        first_status(rows) for rows in llm_rows_by_user.values() if rows
    )
    llm_mode = first_llm_mode(llm_rows, default="real")
    llm_provider = str(getattr(settings, "llm_provider", "openai")).strip().lower() or "openai"
    response_language = normalize_llm_response_language(getattr(settings, "llm_response_language", "ru"))
    mock_mode = llm_mode == "mock"
    improved_users = 0
    worsened_users = 0
    unchanged_users = 0
    if not rank_comparison_df.empty and "rank_delta" in rank_comparison_df.columns:
        for value in rank_comparison_df["rank_delta"]:
            try:
                delta = int(value)
            except Exception:
                continue
            if delta > 0:
                improved_users += 1
            elif delta < 0:
                worsened_users += 1
            else:
                unchanged_users += 1

    baseline_metrics = _metrics_frame_to_dict(metrics_summary_df, "user_tfidf_baseline")
    llm_metrics = _metrics_frame_to_dict(metrics_summary_df, "user_llm_reranker")
    if rank_comparison_df.empty:
        llm_evaluated_users = 0
    elif mock_mode:
        llm_evaluated_users = int(
            sum(1 for _, row in rank_comparison_df.iterrows() if str(row.get("llm_mode", "")) == "mock")
        )
    else:
        llm_evaluated_users = int(
            sum(
                1
                for _, row in rank_comparison_df.iterrows()
                if str(row.get("llm_status", "")) in {"ok", "partial_fallback_to_baseline"}
            )
        )
    selected_count = len(selected_users)
    ok_users = int(user_status_counts.get("ok", 0))
    partial_fallback_users = int(user_status_counts.get("partial_fallback_to_baseline", 0))
    failed_users = int(user_status_counts.get("failed", 0))
    fallback_users = int(user_status_counts.get("fallback_to_baseline", 0))
    ok_records = int(llm_status_counts.get("ok", 0))
    partial_fallback_records = int(llm_status_counts.get("partial_fallback_to_baseline", 0))
    failed_records = int(llm_status_counts.get("failed", 0))
    fallback_records = int(llm_status_counts.get("fallback_to_baseline", 0))
    llm_valid_records = int(ok_records + partial_fallback_records)
    provider_failed_users = int(validation_summary.get("provider_failed_users", 0))
    holdout_in_candidate_pool_users = int(
        0 if rank_comparison_df.empty or "holdout_in_candidate_pool" not in rank_comparison_df.columns else rank_comparison_df["holdout_in_candidate_pool"].fillna(False).astype(bool).sum()
    )
    holdout_in_baseline_top_k_users = int(
        0 if rank_comparison_df.empty or "holdout_in_baseline_top_k" not in rank_comparison_df.columns else rank_comparison_df["holdout_in_baseline_top_k"].fillna(False).astype(bool).sum()
    )
    holdout_in_llm_top_k_users = int(
        0 if rank_comparison_df.empty or "holdout_in_llm_top_k" not in rank_comparison_df.columns else rank_comparison_df["holdout_in_llm_top_k"].fillna(False).astype(bool).sum()
    )
    selected_meaningful_users = int(
        0
        if rank_comparison_df.empty
        else sum(
            1
            for _, row in rank_comparison_df.iterrows()
            if bool(row.get("holdout_in_candidate_pool", False))
            and str(row.get("baseline_best_holdout_rank", "")) not in {"", "not_found", "1"}
        )
    )
    status = "skipped_no_credentials"
    warning_statuses = {"failed", "fallback_to_baseline", "partial_fallback_to_baseline"}
    if llm_metrics:
        status = "completed_with_warnings" if any(key in llm_status_counts for key in warning_statuses) else "completed"
    elif llm_status_counts and any(key in llm_status_counts for key in {"ok"} | warning_statuses):
        status = "completed_with_warnings"
    if mock_mode:
        status = "completed_mock_validation"
    warnings: list[str] = []
    if holdout_in_baseline_top_k_users == 0 and holdout_in_llm_top_k_users == 0:
        warnings.append("No holdout games appeared in either baseline or LLM top-k rankings.")
    if selected_meaningful_users == 0:
        warnings.append("No clearly meaningful candidate users were identified for a rank-improving LLM pilot.")

    return {
        "experiment_name": "steam_reviews_balanced_subset_llm_pilot",
        "status": status,
        "llm_requested": True,
        "llm_ran": bool(
            not mock_mode
            and llm_rows
            and any(str(row.get("status", "")) in {"ok", "partial_fallback_to_baseline"} for row in llm_rows)
        ),
        "llm_mode": llm_mode,
        "llm_provider": llm_provider,
        "llm_model": get_effective_llm_model(settings),
        "response_language": response_language,
        "mock_mode": mock_mode,
        "mock_llm_ran": mock_mode,
        "real_api_calls": 0 if mock_mode else len(selected_users),
        "llm_credentials_configured": provider_credentials_configured(settings),
        "not_for_scientific_metrics": mock_mode,
        "selected_users": selected_count,
        "candidates_per_user": settings.max_llm_candidates,
        "baseline_evaluated_users": int(len(per_profile_df)),
        "llm_evaluated_users": llm_evaluated_users,
        "users_requested": int(validation_summary.get("users_requested", selected_count)),
        "users_completed": int(validation_summary.get("users_completed", len(per_profile_df) - failed_users)),
        "failed_user_count": int(validation_summary.get("failed_user_count", failed_users)),
        "failed_record_count": int(validation_summary.get("failed_record_count", failed_records)),
        "fallback_user_count": int(validation_summary.get("fallback_user_count", fallback_users)),
        "fallback_record_count": int(validation_summary.get("fallback_record_count", fallback_records)),
        "provider_failed_users": provider_failed_users,
        "invalid_json_count": int(validation_summary.get("invalid_json_count", 0)),
        "empty_response_count": int(validation_summary.get("empty_response_count", 0)),
        "schema_validation_error_count": int(validation_summary.get("schema_validation_error_count", 0)),
        "candidate_pool_error_count": int(validation_summary.get("candidate_pool_error_count", 0)),
        "response_preview_saved_count": int(validation_summary.get("response_preview_saved_count", 0)),
        "provider_preflight_ok": bool(validation_summary.get("provider_preflight_ok", True)),
        "provider_preflight_status": str(validation_summary.get("provider_preflight_status", "")),
        "provider_preflight_error_type": str(validation_summary.get("provider_preflight_error_type", "")),
        "provider_preflight_error_message_short": str(validation_summary.get("provider_preflight_error_message_short", "")),
        "token_requests_attempted": int(validation_summary.get("token_requests_attempted", 0)),
        "completion_requests_attempted": int(validation_summary.get("completion_requests_attempted", 0)),
        "improved_users": int(improved_users),
        "worsened_users": int(worsened_users),
        "unchanged_users": int(unchanged_users),
        "skipped_users": int(user_status_counts.get("skipped_no_credentials", 0)),
        "failed_users": failed_users,
        "partial_fallback_users": partial_fallback_users,
        "fallback_users": fallback_users,
        "ok_users": ok_users,
        "ok_records": ok_records,
        "llm_valid_records": llm_valid_records,
        "partial_fallback_records": partial_fallback_records,
        "failed_records": failed_records,
        "fallback_records": fallback_records,
        "users_with_any_valid_llm_records": int(
            sum(
                1
                for rows in llm_rows_by_user.values()
                if any(str(row.get("status", "")) in {"ok", "partial_fallback_to_baseline"} for row in rows)
            )
        ),
        "users_with_only_fallback": int(
            sum(
                1
                for rows in llm_rows_by_user.values()
                if rows and all(str(row.get("status", "")) == "fallback_to_baseline" for row in rows)
            )
        ),
        "users_with_partial_fallback": int(partial_fallback_users),
        "users_with_schema_errors": int(validation_summary.get("schema_error_user_count", 0)),
        "holdout_in_candidate_pool_users": holdout_in_candidate_pool_users,
        "holdout_in_baseline_top_k_users": holdout_in_baseline_top_k_users,
        "holdout_in_llm_top_k_users": holdout_in_llm_top_k_users,
        "selected_meaningful_users": selected_meaningful_users,
        "baseline_metrics": baseline_metrics,
        "llm_metrics": llm_metrics,
        "explanation_check_count": int(len(explanation_checks_df)),
        "status_counts": dict(llm_status_counts),
        "pilot_limitations": [
            "This is a controlled pilot on a small subset of user-based evaluation splits.",
            "The results should not be presented as a full-experiment conclusion.",
            "Mock results are validation-only and must not be interpreted as scientific performance."
            if mock_mode
            else "This is a real LLM pilot, but still limited to a small controlled subset.",
        ],
        "warnings": warnings,
    }


def save_user_rank_comparison_markdown(settings: Settings, frame: pd.DataFrame) -> None:

    mock_mode = bool(
        not frame.empty
        and "llm_mode" in frame.columns
        and (frame["llm_mode"].astype(str) == "mock").any()
    )
    lines = [
        "# User Rank Comparison",
        "",
        "Positive `rank_delta` means the LLM improved the ranking of the best holdout game.",
        "Negative `rank_delta` means the LLM worsened the ranking.",
        "If holdout games are not found, the table uses `not_found`.",
        "The diagnostic columns show whether the holdout was present in the candidate pool and in the top-k lists.",
        "Mock LLM results are validation-only and must not be interpreted as scientific performance."
        if mock_mode
        else "This table reflects a controlled LLM pilot subset."
        if not frame.empty
        else "",
        "",
    ]
    if frame.empty:
        lines.append("_No rank comparison rows available._")
    else:
        lines.append(dataframe_to_markdown(frame))
    settings.user_rank_comparison_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:

    if frame.empty:
        return "_No rows available._"

    headers = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(rows)


def format_metric(value: object) -> str:

    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def mask_user_id(user_id: str) -> str:

    cleaned = str(user_id).strip()
    if not cleaned:
        return "user_unknown"
    if len(cleaned) <= 4:
        return "user_****"
    return f"{cleaned[:4]}********{cleaned[-4:]}"


def _metrics_frame_to_dict(metrics_df: pd.DataFrame, method_name: str) -> dict[str, object]:

    if metrics_df.empty or "method" not in metrics_df.columns:
        return {}
    matches = metrics_df[metrics_df["method"].astype(str) == method_name]
    if matches.empty:
        return {}
    row = matches.iloc[0].to_dict()
    return {
        "method": str(row.get("method", method_name)),
        "evaluated_profiles": int(row.get("evaluated_profiles", 0)),
        "mean_hit_rate_at_5": float(row.get("mean_hit_rate_at_5", 0.0)),
        "mean_hit_rate_at_10": float(row.get("mean_hit_rate_at_10", 0.0)),
        "mean_mrr": float(row.get("mean_mrr", 0.0)),
        "mean_ndcg_at_10": float(row.get("mean_ndcg_at_10", 0.0)),
    }
