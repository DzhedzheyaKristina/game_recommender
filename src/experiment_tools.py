"""Artifact-based helpers for diagnostics, scenario authoring, and readiness."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import os
import random
import re
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline_tfidf import run_baseline
from src.config import Settings, normalize_llm_response_language
from src.data_loader import SAMPLE_WARNING_MESSAGE, load_external_scenarios, load_reviews_csv
from src.environment_tools import run_environment_check
from src.evaluation import evaluate_recommendations
from src.game_card_builder import GameCard
from src.game_card_builder import build_game_cards
from src.llm_reranker import run_llm_reranker
from src.llm_provider import get_effective_llm_model, get_effective_llm_provider, provider_credentials_configured
from src.preprocessing import NORMALIZED_COLUMN_ORDER, preprocess_reviews
from src.scenario_builder import build_scenarios, normalize_id_list, normalize_predefined_scenarios
from src.user_experiments import (
    build_user_evaluation_splits,
    build_user_split_diagnostics_frame,
    build_user_splits_pilot,
    build_user_profiles,
    evaluate_user_baseline,
    get_active_user_splits_path,
    get_active_user_split_mode,
    load_clean_reviews_for_user_mode,
    run_user_baseline,
    summarize_user_eligibility,
)
from src.user_llm import (
    evaluate_user_llm,
    run_llm_check,
    run_user_llm_pilot,
    save_user_llm_mock_validation_report,
)
from src.utils import get_logger, model_to_dict, read_jsonl, write_jsonl


def run_data_diagnostics(settings: Settings) -> dict[str, object] | None:
    """Create dataset diagnostics and charts from cleaned reviews."""

    logger = get_logger()
    cleaned_reviews_path = settings.active_processed_reviews_path
    if not cleaned_reviews_artifact_is_valid(cleaned_reviews_path):
        subset_candidates = [
            settings.reviews_clean_subset_path,
            settings.reviews_clean_balanced_subset_path,
        ]
        if any(cleaned_reviews_artifact_is_valid(path) for path in subset_candidates):
            logger.warning(
                "Active processed reviews file %s is missing or invalid. Set ACTIVE_PROCESSED_REVIEWS to a valid subset file such as %s.",
                cleaned_reviews_path.relative_to(settings.project_root),
                settings.reviews_clean_balanced_subset_path.relative_to(settings.project_root),
            )
        else:
            logger.warning(
                "Cleaned reviews not found at %s. Run `python main.py --step preprocess` first.",
                cleaned_reviews_path.relative_to(settings.project_root),
            )
        return None

    reviews_df = pd.read_csv(cleaned_reviews_path)
    if reviews_df.empty:
        logger.warning("Cleaned reviews file is empty. Diagnostics were not generated.")
        return None

    review_text_column = "review_clean" if "review_clean" in reviews_df.columns else "review_text"
    review_lengths = reviews_df[review_text_column].fillna("").astype(str).str.len()
    reviews_df["recommended"] = reviews_df["recommended"].fillna(False).astype(bool)
    game_summary = summarize_games(reviews_df)
    user_summary = summarize_users(reviews_df)
    eligibility = summarize_user_eligibility(settings, reviews_df)

    language_distribution = (
        reviews_df["language"].fillna("unknown").astype(str).value_counts().to_dict()
    )
    recommendation_distribution = {
        "positive": int(reviews_df["recommended"].fillna(False).sum()),
        "negative": int((~reviews_df["recommended"].fillna(False)).sum()),
    }
    top_games_by_review_count = records_for_report(
        game_summary.sort_values(["review_count", "game_title"], ascending=[False, True]).head(20),
        ["game_id", "game_title", "review_count", "positive_ratio"],
    )
    top_games_by_positive_ratio = records_for_report(
        game_summary[game_summary["review_count"] >= settings.min_reviews_per_game]
        .sort_values(["positive_ratio", "review_count", "game_title"], ascending=[False, False, True])
        .head(20),
        ["game_id", "game_title", "review_count", "positive_ratio"],
    )
    top_games_by_helpful_votes = records_for_report(
        game_summary.sort_values(["votes_helpful_sum", "game_title"], ascending=[False, True]).head(20),
        ["game_id", "game_title", "votes_helpful_sum", "review_count"],
    )

    eligible_for_card_generation = int(
        (game_summary["review_count"] >= settings.min_reviews_per_game).sum()
    )
    schema_report = load_json_if_exists(settings.schema_validation_report_json_path)
    preprocessing_summary = load_json_if_exists(get_active_preprocessing_summary_path(settings))
    raw_inspection = load_json_if_exists(settings.raw_dataset_inspection_json_path)
    artifacts_match_current_dataset = True
    if (
        settings.active_processed_reviews_path == settings.reviews_clean_path
        and raw_inspection
        and preprocessing_summary
    ):
        artifacts_match_current_dataset = int(
            raw_inspection.get("raw_row_count", 0)
        ) == int(preprocessing_summary.get("raw_row_count", 0))
    likely_sample_or_incomplete = bool(
        len(reviews_df) <= 100
        or int(game_summary["game_id"].nunique()) <= 10
        or len(schema_report.get("missing_expected_columns", [])) >= 5
        or preprocessing_summary.get("user_id_mapping_status") == "mapping_failed"
        or (
            preprocessing_summary.get("raw_author_steamid_present")
            and int(preprocessing_summary.get("raw_author_steamid_non_empty_count", 0)) == 0
        )
    )
    warnings = [SAMPLE_WARNING_MESSAGE] if likely_sample_or_incomplete else []
    if not artifacts_match_current_dataset:
        warnings.append(
            "Processed artifacts do not match the currently selected raw dataset. "
            "Re-run preprocessing on the current CSV before trusting downstream reports."
        )
    diagnostics = {
        "active_processed_dataset_path": str(cleaned_reviews_path),
        "active_processed_dataset_label": str(cleaned_reviews_path.relative_to(settings.project_root)),
        "dataset_size": int(len(reviews_df)),
        "total_cleaned_reviews": int(len(reviews_df)),
        "unique_games": int(game_summary["game_id"].nunique()),
        "unique_users": int(eligibility["unique_users"]),
        "unique_languages": int(reviews_df["language"].astype(str).nunique()),
        "unique_review_ids": int(reviews_df["review_id"].astype(str).nunique()),
        "duplicate_review_count_after_preprocessing": int(
            reviews_df["review_id"].astype(str).duplicated().sum()
        ),
        "empty_review_count_before_cleaning": load_json_if_exists(get_active_preprocessing_summary_path(settings)).get("raw_empty_review_count", 0),
        "language_distribution": language_distribution,
        "recommendation_distribution": recommendation_distribution,
        "positive_review_ratio": round(float(reviews_df["recommended"].fillna(False).mean()), 4),
        "review_length_statistics": {
            "mean": round(float(review_lengths.mean()), 2),
            "median": round(float(review_lengths.median()), 2),
            "min": int(review_lengths.min()),
            "max": int(review_lengths.max()),
            "percentile_25": round(float(review_lengths.quantile(0.25)), 2),
            "percentile_75": round(float(review_lengths.quantile(0.75)), 2),
        },
        "weighted_vote_score_statistics": numeric_statistics(reviews_df["weighted_vote_score"]),
        "votes_helpful_statistics": numeric_statistics(reviews_df["votes_helpful"]),
        "votes_funny_statistics": numeric_statistics(reviews_df["votes_funny"]),
        "comment_count_statistics": numeric_statistics(reviews_df["comment_count"]),
        "reviews_per_user_statistics": numeric_statistics(user_summary.get("review_count", pd.Series(dtype=float))),
        "positive_reviews_per_user_statistics": numeric_statistics(user_summary.get("positive_review_count", pd.Series(dtype=float))),
        "eligible_users_for_user_based_evaluation": int(eligibility["eligible_user_count"]),
        "author_num_games_owned_statistics": numeric_statistics(reviews_df["author_num_games_owned"]),
        "author_num_reviews_statistics": numeric_statistics(reviews_df["author_num_reviews"]),
        "playtime_forever_statistics": numeric_statistics(reviews_df["playtime_forever"]),
        "playtime_at_review_statistics": numeric_statistics(reviews_df["playtime_at_review"]),
        "playtime_last_two_weeks_statistics": numeric_statistics(reviews_df["playtime_last_two_weeks"]),
        "percentage_reviews_with_playtime_gt_zero": round(float(reviews_df["playtime_forever"].fillna(0).gt(0).mean()), 4),
        "steam_purchase_ratio": round(float(reviews_df["steam_purchase"].fillna(False).astype(bool).mean()), 4),
        "received_for_free_ratio": round(float(reviews_df["received_for_free"].fillna(False).astype(bool).mean()), 4),
        "written_during_early_access_ratio": round(float(reviews_df["written_during_early_access"].fillna(False).astype(bool).mean()), 4),
        "reviews_per_game_statistics": numeric_statistics(game_summary["review_count"]),
        "positive_ratio_per_game_statistics": numeric_statistics(game_summary["positive_ratio"]),
        "top_20_games_by_review_count": top_games_by_review_count,
        "top_20_games_by_positive_ratio_with_enough_reviews": top_games_by_positive_ratio,
        "top_20_games_by_helpful_votes": top_games_by_helpful_votes,
        "games_below_min_reviews_per_game": int(
            (game_summary["review_count"] < settings.min_reviews_per_game).sum()
        ),
        "games_eligible_for_card_generation": eligible_for_card_generation,
        "has_user_id": bool(eligibility["has_user_id"]),
        "user_based_mode_available": bool(eligibility["user_based_mode_available"]),
        "likely_sample_or_incomplete": likely_sample_or_incomplete,
        "warnings": warnings,
        "artifacts_match_current_dataset": artifacts_match_current_dataset,
    }

    save_bar_chart(
        values=language_distribution,
        title="Language Distribution",
        xlabel="Language",
        ylabel="Reviews",
        output_path=settings.review_language_distribution_plot_path,
    )
    save_bar_chart(
        values=recommendation_distribution,
        title="Recommendation Distribution",
        xlabel="Recommendation",
        ylabel="Reviews",
        output_path=settings.review_recommendation_distribution_plot_path,
    )
    save_histogram(
        values=user_summary.get("review_count", pd.Series(dtype=float)),
        title="Reviews per User Distribution",
        xlabel="Reviews per user",
        output_path=settings.reviews_per_user_distribution_plot_path,
    )
    save_histogram(
        values=game_summary["review_count"],
        title="Reviews per Game Distribution",
        xlabel="Reviews per game",
        output_path=settings.reviews_per_game_distribution_plot_path,
    )
    save_histogram(
        values=reviews_df["playtime_forever"],
        title="Playtime Forever Distribution",
        xlabel="Playtime forever",
        output_path=settings.playtime_forever_distribution_plot_path,
    )
    save_top_games_chart(
        game_summary=game_summary,
        output_path=settings.top_games_by_review_count_plot_path,
    )

    write_json_report(settings.data_diagnostics_json_path, diagnostics)
    settings.data_diagnostics_markdown_path.write_text(
        build_data_diagnostics_markdown(diagnostics),
        encoding="utf-8",
    )
    logger.info(
        "Saved data diagnostics to %s and %s",
        settings.data_diagnostics_json_path.relative_to(settings.project_root),
        settings.data_diagnostics_markdown_path.relative_to(settings.project_root),
    )
    return diagnostics


def export_available_games(settings: Settings) -> pd.DataFrame:
    """Export compact game lists to help manual scenario design."""

    game_cards = load_game_cards_from_artifact(settings.game_cards_path)
    rows = [
        {
            "game_id": card.game_id,
            "game_title": card.game_title,
            "review_count": card.review_count,
            "positive_review_count": card.positive_review_count,
            "negative_review_count": card.negative_review_count,
            "positive_ratio": card.positive_ratio,
            "positive_keywords": ", ".join(card.positive_keywords),
            "negative_keywords": ", ".join(card.negative_keywords),
            "representative_positive_review": first_or_empty(card.representative_positive_reviews),
            "representative_negative_review": first_or_empty(card.representative_negative_reviews),
            "scenario_hint": build_scenario_hint(card),
        }
        for card in sorted(
            game_cards,
            key=lambda card: (card.review_count, card.positive_ratio, card.game_title),
            reverse=True,
        )
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(settings.available_games_path, index=False)

    markdown_frame = frame.head(settings.available_games_markdown_limit)
    settings.available_games_markdown_path.write_text(
        "# Available Games\n\n"
        + dataframe_to_markdown(markdown_frame),
        encoding="utf-8",
    )
    get_logger().info(
        "Saved available games list to %s",
        settings.available_games_path.relative_to(settings.project_root),
    )
    return frame


def generate_draft_scenarios(settings: Settings) -> list[dict[str, object]]:
    """Generate draft manual scenarios from game-card similarity."""

    logger = get_logger()
    game_cards = load_game_cards_from_artifact(settings.game_cards_path)
    if not game_cards:
        logger.warning("No game cards are available for draft scenario generation.")
        return []

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    matrix = vectorizer.fit_transform([card.game_card_text for card in game_cards])
    id_to_index = {card.game_id: index for index, card in enumerate(game_cards)}
    rng = random.Random(settings.random_seed)

    seed_candidates = [
        card
        for card in sorted(
            game_cards,
            key=lambda card: (card.review_count, card.positive_ratio),
            reverse=True,
        )
        if card.review_count >= settings.min_reviews_for_draft_scenario_seed
        and len(card.positive_keywords) >= 3
        and len(card.negative_keywords) >= 2
    ]
    if not seed_candidates:
        seed_candidates = sorted(game_cards, key=lambda card: card.review_count, reverse=True)

    scenarios: list[dict[str, object]] = []
    for index, seed_card in enumerate(seed_candidates[: settings.draft_scenario_count], start=1):
        similar_cards = rank_similar_cards(seed_card, game_cards, matrix, id_to_index)
        if not similar_cards:
            continue

        ground_truth_cards = similar_cards[: settings.draft_scenario_ground_truth_count]
        similar_pool = [card.game_id for card in similar_cards[: settings.draft_scenario_candidate_count]]
        distractor_pool = [
            card.game_id
            for card in game_cards
            if card.game_id not in {seed_card.game_id, *similar_pool}
        ]
        rng.shuffle(distractor_pool)
        candidate_game_ids = deduplicate_preserve_order(
            [card.game_id for card in ground_truth_cards]
            + similar_pool
            + distractor_pool
        )[: settings.draft_scenario_candidate_count]

        scenarios.append(
            {
                "scenario_id": f"draft_{index:03d}",
                "scenario_type": "manual_draft",
                "preference_text": build_draft_preference_text(seed_card),
                "seed_game_ids": [seed_card.game_id],
                "excluded_game_ids": [seed_card.game_id],
                "ground_truth_game_ids": [card.game_id for card in ground_truth_cards],
                "candidate_game_ids": candidate_game_ids,
                "notes": (
                    "Draft scenario generated from dataset keywords. Review and edit manually "
                    "before using in thesis evaluation."
                ),
            }
        )

    write_jsonl(settings.draft_scenarios_jsonl_path, scenarios)
    draft_frame = pd.DataFrame([scenario_record_to_csv_row(record) for record in scenarios])
    draft_frame.to_csv(settings.draft_scenarios_csv_path, index=False)
    logger.info(
        "Saved %s draft scenarios to %s and %s",
        len(scenarios),
        settings.draft_scenarios_jsonl_path.relative_to(settings.project_root),
        settings.draft_scenarios_csv_path.relative_to(settings.project_root),
    )
    return scenarios


def normalize_scenarios_file(settings: Settings) -> pd.DataFrame | None:
    """Normalize the configured external scenario file to canonical JSONL and validate it."""

    logger = get_logger()
    if settings.scenarios_file is None:
        logger.warning(
            "SCENARIOS_FILE is not configured. Set it in `.env` before running `python main.py --step normalize_scenarios`."
        )
        return None
    if not settings.scenarios_file.exists():
        logger.warning(
            "Configured SCENARIOS_FILE %s does not exist.",
            settings.scenarios_file,
        )
        return None

    game_cards = load_game_cards_from_artifact(settings.game_cards_path)
    external_records = load_external_scenarios(settings.scenarios_file)
    normalized_scenarios = normalize_predefined_scenarios(
        external_records=external_records,
        id_to_card={card.game_id: card for card in game_cards},
    )
    normalized_records = [scenario.model_dump() if hasattr(scenario, "model_dump") else scenario.dict() for scenario in normalized_scenarios]
    write_jsonl(settings.scenarios_output_path, normalized_records)
    validation_frame = validate_scenario_records(
        settings=settings,
        scenario_records=normalized_records,
        source_label=str(settings.scenarios_output_path),
    )
    logger.info(
        "Normalized %s scenarios from %s to %s",
        len(normalized_records),
        settings.scenarios_file,
        settings.scenarios_output_path.relative_to(settings.project_root),
    )
    return validation_frame


def validate_scenarios_from_artifacts(settings: Settings) -> pd.DataFrame:
    """Validate scenario definitions against the currently available game cards."""

    scenario_records, scenario_source = load_scenario_records_for_validation(settings)
    return validate_scenario_records(
        settings=settings,
        scenario_records=scenario_records,
        source_label=scenario_source,
    )


def validate_scenario_records(
    settings: Settings,
    scenario_records: list[dict[str, object]],
    source_label: str,
) -> pd.DataFrame:
    """Validate raw scenario records and save a report."""

    logger = get_logger()
    game_cards = load_game_cards_from_artifact(settings.game_cards_path)
    game_ids = {card.game_id for card in game_cards}

    rows = [
        validate_single_scenario_record(
            record=record,
            scenario_index=index,
            valid_game_ids=game_ids,
            top_k=settings.top_k,
        )
        for index, record in enumerate(scenario_records, start=1)
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "scenario_id",
            "status",
            "missing_fields",
            "unknown_game_ids",
            "candidate_count",
            "ground_truth_count",
            "warnings",
        ],
    )
    frame.to_csv(settings.scenario_validation_report_path, index=False)

    status_counts = Counter(frame["status"]) if not frame.empty else Counter()
    logger.info(
        "Validated %s scenarios: %s ok, %s warnings, %s invalid.",
        len(frame),
        int(status_counts.get("ok", 0)),
        int(status_counts.get("warning", 0)),
        int(status_counts.get("invalid", 0)),
    )
    logger.info(
        "Saved scenario validation report to %s",
        settings.scenario_validation_report_path.relative_to(settings.project_root),
    )
    logger.info("Scenario validation source: %s", source_label)
    return frame


def build_experiment_readiness_report(settings: Settings) -> dict[str, object]:
    """Inspect existing artifacts and summarize thesis-experiment readiness."""

    logger = get_logger()
    active_context = build_active_experiment_context(settings)
    active_processed_reviews_path = settings.active_processed_reviews_path
    cleaned_reviews_exists = cleaned_reviews_artifact_is_valid(active_processed_reviews_path)
    cleaned_review_count = 0
    unique_games = 0
    user_eligibility = {
        "has_user_id": False,
        "user_based_mode_available": False,
        "unique_users": 0,
        "eligible_user_count": 0,
        "min_user_reviews": settings.min_user_reviews,
        "min_user_positive_reviews": settings.min_user_positive_reviews,
        "can_run_user_experiment": False,
    }
    if cleaned_reviews_exists:
        reviews_df = pd.read_csv(active_processed_reviews_path)
        cleaned_review_count = int(len(reviews_df))
        unique_games = int(reviews_df["game_id"].astype(str).nunique()) if "game_id" in reviews_df else 0
        user_eligibility = summarize_user_eligibility(settings, reviews_df)

    game_cards = load_optional_game_cards(settings.game_cards_path)
    scenarios = load_optional_records(settings.scenarios_output_path)
    scenario_type_distribution = dict(
        Counter(str(record.get("scenario_type", "")).strip() for record in scenarios)
    )
    game_card_count = len(game_cards)
    scenario_count = len(scenarios)
    manual_scenario_count = sum(
        1
        for record in scenarios
        if str(record.get("scenario_type", "")).strip() in {"manual", "seed_games"}
    )
    synthetic_scenario_count = sum(
        1
        for record in scenarios
        if str(record.get("scenario_type", "")).strip() == "synthetic_demo"
    )
    scenario_mode = determine_scenario_mode(scenarios)
    scenario_metrics_exist = settings.metrics_summary_path.exists()
    user_metrics_exist = settings.user_metrics_summary_path.exists()
    llm_credentials_configured = provider_credentials_configured(settings)
    has_active_user_artifacts = (
        bool(active_context["profiles_exist"])
        and bool(active_context["splits_exist"])
        and bool(active_context["metrics_exist"])
    )

    full_dataset_ready = bool(active_context["full_dataset_ready"])
    balanced_subset_ready = bool(active_context["balanced_subset_user_based_ready"])
    active_dataset_mode = str(active_context["active_dataset_mode"])
    current_recommended_workflow = str(active_context["current_recommended_workflow"])

    full_user_based_ready = bool(active_context["full_user_based_ready"])
    balanced_user_based_ready = balanced_subset_ready and has_active_user_artifacts
    active_user_splits_exist = bool(active_context.get("active_user_splits_exist", False))
    active_user_split_mode = str(active_context.get("active_user_split_mode", "main"))
    active_user_split_count = int(active_context.get("active_user_split_count", 0))

    dataset_too_small = (
        cleaned_review_count < settings.min_reviews_for_real_experiment
        or game_card_count < settings.min_game_cards_for_real_experiment
    )

    warnings: list[str] = []
    if dataset_too_small and active_dataset_mode != "balanced_subset":
        warnings.append(
            "Dataset appears too small for meaningful thesis conclusions under the current thresholds."
        )
    if active_dataset_mode == "balanced_subset" and not balanced_subset_ready:
        warnings.append("Balanced subset is selected, but it is missing required user-based artifacts.")
    if active_dataset_mode == "full" and not full_dataset_ready:
        warnings.append("Full dataset preprocessing has not completed successfully.")
    if scenario_mode == "synthetic_demo_only":
        warnings.append("Only synthetic scenarios are available or scenario coverage is mixed.")
    if scenario_mode != "manual_or_predefined" and active_dataset_mode not in {"full", "balanced_subset"}:
        warnings.append("Manual/predefined scenarios are not ready yet.")
    if not user_eligibility["has_user_id"]:
        warnings.append("User-based mode is unavailable because no author.steamid data is present.")
    elif not user_eligibility["can_run_user_experiment"]:
        warnings.append("User-based mode is present but too few eligible users are available for evaluation.")
    if active_user_split_mode == "pilot" and not active_user_splits_exist:
        warnings.append("Pilot splits are configured but the pilot split file is missing.")

    readiness_status = "not_ready"
    readiness_message = "The project is not ready for the current requested workflow."
    if active_dataset_mode == "balanced_subset" and balanced_user_based_ready:
        readiness_status = (
            "ready_for_balanced_subset_user_based_llm_experiment"
            if llm_credentials_configured
            else "ready_for_balanced_subset_user_based_baseline_experiment"
        )
        readiness_message = "Balanced subset user-based baseline evaluation is ready."
    elif active_dataset_mode == "full" and full_user_based_ready:
        readiness_status = (
            "ready_for_full_user_based_llm_experiment"
            if llm_credentials_configured
            else "ready_for_full_user_based_baseline_experiment"
        )
        readiness_message = "Full-dataset user-based baseline evaluation is ready."
    elif dataset_too_small or scenario_mode == "synthetic_demo_only":
        readiness_status = "technical_validation_only"
        readiness_message = "This looks like a technical validation run, not a real thesis experiment."

    report = {
        "cleaned_reviews_exist": cleaned_reviews_exists,
        "cleaned_review_count": cleaned_review_count,
        "unique_games": unique_games,
        "unique_users": int(user_eligibility["unique_users"]),
        "game_card_count": game_card_count,
        "scenario_count": scenario_count,
        "manual_scenario_count": manual_scenario_count,
        "synthetic_scenario_count": synthetic_scenario_count,
        "scenario_type_distribution": scenario_type_distribution,
        "scenario_mode": scenario_mode,
        "active_processed_reviews": str(active_processed_reviews_path),
        "active_dataset_mode": active_dataset_mode,
        "current_recommended_workflow": current_recommended_workflow,
        "full_dataset_ready": full_dataset_ready,
        "balanced_subset_ready": balanced_subset_ready,
        "balanced_subset_user_based_ready": balanced_user_based_ready,
        "full_user_based_ready": full_user_based_ready,
        "active_user_split_mode": active_user_split_mode,
        "active_user_splits_exist": active_user_splits_exist,
        "active_user_split_count": active_user_split_count,
        "llm_credentials_configured": llm_credentials_configured,
        "scenario_metrics_exist": scenario_metrics_exist,
        "user_metrics_exist": user_metrics_exist,
        "metrics_exist": scenario_metrics_exist or user_metrics_exist,
        "dataset_too_small": dataset_too_small,
        "readiness_status": readiness_status,
        **user_eligibility,
        "thresholds": {
            "min_reviews_for_real_experiment": settings.min_reviews_for_real_experiment,
            "min_game_cards_for_real_experiment": settings.min_game_cards_for_real_experiment,
            "min_manual_scenarios_for_real_experiment": settings.min_manual_scenarios_for_real_experiment,
        },
        "warnings": warnings,
        "readiness_message": readiness_message,
    }

    write_json_report(settings.experiment_readiness_path, report)
    logger.info("Readiness summary:")
    logger.info("  Cleaned reviews: %s", cleaned_review_count)
    logger.info("  Unique games: %s", unique_games)
    logger.info("  Unique users: %s", user_eligibility["unique_users"])
    logger.info("  Game cards: %s", game_card_count)
    logger.info("  Scenarios: %s", scenario_count)
    logger.info("  Active dataset mode: %s", active_dataset_mode)
    logger.info("  Scenario mode: %s", scenario_mode)
    logger.info("  User-based mode available: %s", user_eligibility["user_based_mode_available"])
    logger.info("  Eligible users: %s", user_eligibility["eligible_user_count"])
    logger.info("  Active user split mode: %s", active_user_split_mode)
    logger.info("  Active user splits exist: %s", active_user_splits_exist)
    logger.info("  Active user split count: %s", active_user_split_count)
    logger.info("  LLM credentials configured: %s", llm_credentials_configured)
    logger.info("  Metrics exist: %s", report["metrics_exist"])
    logger.info("%s", readiness_message)
    logger.info(
        "Saved experiment readiness report to %s",
        settings.experiment_readiness_path.relative_to(settings.project_root),
    )
    return report


def build_llm_pilot_readiness_report(settings: Settings) -> dict[str, object]:
    """Check readiness for a controlled LLM pilot without making API calls."""

    logger = get_logger()
    response_language = normalize_llm_response_language(getattr(settings, "llm_response_language", "ru"))
    llm_mode = str(getattr(settings, "llm_mode", "real")).strip().lower() or "real"
    active_dataset_mode = get_active_dataset_mode(settings)
    balanced_subset_active = settings.active_processed_reviews_path == settings.reviews_clean_balanced_subset_path
    balanced_subset_exists = cleaned_reviews_artifact_is_valid(settings.reviews_clean_balanced_subset_path)
    user_profiles_exist = settings.user_profiles_path.exists()
    active_split_path = get_active_user_splits_path(settings)
    user_splits_exist = active_split_path.exists()
    active_split_mode = get_active_user_split_mode(settings)
    active_split_count = 0
    holdout_in_candidate_pool_splits = 0
    if user_splits_exist:
        try:
            split_df = build_user_split_diagnostics_frame(settings, read_jsonl(active_split_path))
            active_split_count = int(len(split_df))
            if not split_df.empty and "holdout_in_candidate_pool" in split_df.columns:
                holdout_in_candidate_pool_splits = int(split_df["holdout_in_candidate_pool"].fillna(False).astype(bool).sum())
        except Exception:
            active_split_count = 0
            holdout_in_candidate_pool_splits = 0
    user_baseline_exists = settings.user_baseline_results_path.exists()
    prompt_preview_exists = any(
        path.exists()
        for path in [
            settings.user_llm_prompt_preview_markdown_path,
            settings.user_llm_prompt_preview_json_path,
            settings.user_llm_prompt_preview_tiny_markdown_path,
            settings.user_llm_prompt_preview_tiny_json_path,
            settings.user_llm_prompt_preview_10_gigachat_markdown_path,
            settings.user_llm_prompt_preview_10_gigachat_json_path,
        ]
    )
    meaningful_candidate_report_exists = settings.llm_pilot_candidate_user_report_csv_path.exists()
    meaningful_candidate_users = 0
    if meaningful_candidate_report_exists:
        try:
            candidate_df = pd.read_csv(settings.llm_pilot_candidate_user_report_csv_path)
            if not candidate_df.empty and "eligible_for_meaningful_llm_pilot" in candidate_df.columns:
                meaningful_candidate_users = int(
                    candidate_df["eligible_for_meaningful_llm_pilot"].fillna(False).astype(bool).sum()
                )
        except Exception:
            meaningful_candidate_users = 0
    llm_credentials_configured = provider_credentials_configured(settings)
    llm_check_status = "configured" if llm_credentials_configured else "missing_credentials"
    selected_provider = get_effective_llm_provider(settings)
    selected_model = get_effective_llm_model(settings)
    real_llm_calls_allowed = bool(llm_credentials_configured)
    configured_max_llm_users = int(getattr(settings, "max_llm_users", 0) or 0)
    configured_max_llm_candidates = int(getattr(settings, "max_llm_candidates", 0) or 0)
    max_llm_users_ok = configured_max_llm_users <= 10
    max_llm_candidates_ok = configured_max_llm_candidates <= 10
    provider_display_name = "GigaChat" if selected_provider == "gigachat" else selected_provider.capitalize()

    prerequisites_missing: list[str] = []
    unsafe_config: list[str] = []
    if not balanced_subset_active:
        prerequisites_missing.append("ACTIVE_PROCESSED_REVIEWS does not point to the balanced subset.")
    if not balanced_subset_exists:
        prerequisites_missing.append("Balanced subset cleaned dataset is missing.")
    if not user_profiles_exist:
        prerequisites_missing.append("User profiles are missing.")
    if not user_splits_exist:
        prerequisites_missing.append("User evaluation splits are missing.")
    if not user_baseline_exists:
        prerequisites_missing.append("User baseline recommendations are missing.")
    if not prompt_preview_exists:
        prerequisites_missing.append("Dry-run prompt preview is missing.")

    if llm_mode != "real":
        unsafe_config.append("LLM_MODE must be real for the pilot.")
    if response_language != "ru":
        unsafe_config.append("LLM_RESPONSE_LANGUAGE must be ru for the Russian-language pilot.")
    if not max_llm_users_ok:
        unsafe_config.append("max_llm_users must be 10 or less for this prepared pilot.")
    if not max_llm_candidates_ok:
        unsafe_config.append("max_llm_candidates must be 10 or less.")
    if not real_llm_calls_allowed:
        unsafe_config.append("LLM credentials are missing, so a real run cannot start yet.")

    warnings: list[str] = []
    if balanced_subset_exists and not settings.reviews_clean_path.exists():
        warnings.append(
            "Full dataset preprocessing is still incomplete. This pilot is intentionally limited to the balanced subset."
        )
    if llm_check_status == "missing_credentials":
        warnings.append(f"{provider_display_name} credentials are missing, so the real pilot cannot run yet.")
    if active_dataset_mode != "balanced_subset":
        warnings.append(
            f"Active dataset mode is {active_dataset_mode!r}; the tiny real pilot expects the balanced subset."
        )
    if meaningful_candidate_users > 0:
        warnings.append(
            f"{meaningful_candidate_users} candidate users appear suitable for a more informative tiny pilot."
        )
    if active_split_mode == "pilot" and holdout_in_candidate_pool_splits > 0:
        warnings.append(
            f"{holdout_in_candidate_pool_splits} pilot split(s) include the holdout in the candidate pool."
        )

    if prerequisites_missing:
        status = "missing_prerequisites"
    elif unsafe_config:
        status = "unsafe_config"
    elif llm_check_status == "missing_credentials":
        status = "missing_credentials"
    else:
        status = "ready_for_real_llm_pilot"

    report = {
        "status": status,
        "active_processed_reviews": str(settings.active_processed_reviews_path),
        "active_dataset_mode": active_dataset_mode,
        "balanced_subset_active": balanced_subset_active,
        "balanced_subset_exists": balanced_subset_exists,
        "user_profiles_exist": user_profiles_exist,
        "user_evaluation_splits_exist": user_splits_exist,
        "active_user_splits_path": str(active_split_path),
        "active_split_mode": active_split_mode,
        "active_split_count": int(active_split_count),
        "holdout_in_candidate_pool_splits": int(holdout_in_candidate_pool_splits),
        "user_baseline_recommendations_exist": user_baseline_exists,
        "dry_run_prompt_preview_exists": prompt_preview_exists,
        "meaningful_candidate_report_exists": meaningful_candidate_report_exists,
        "meaningful_candidate_users": meaningful_candidate_users,
        "llm_check_status": llm_check_status,
        "selected_provider": selected_provider,
        "selected_model": selected_model,
        "response_language": response_language,
        "llm_mode": llm_mode,
        "real_llm_calls_allowed": real_llm_calls_allowed,
        "max_llm_users": configured_max_llm_users,
        "max_llm_candidates": configured_max_llm_candidates,
        "credentials_configured": llm_credentials_configured,
        "prerequisites_missing": prerequisites_missing,
        "unsafe_config": unsafe_config,
        "warnings": warnings,
    }
    write_json_report(settings.llm_pilot_readiness_json_path, report)
    missing_prereq_lines = [f"- {item}" for item in prerequisites_missing] if prerequisites_missing else ["- none"]
    unsafe_config_lines = [f"- {item}" for item in unsafe_config] if unsafe_config else ["- none"]
    warning_lines = [f"- {item}" for item in warnings] if warnings else ["- none"]

    lines = [
        "# LLM Pilot Readiness",
        "",
        f"- Status: {status}",
        f"- Active processed reviews: `{report['active_processed_reviews']}`",
        f"- Active dataset mode: `{active_dataset_mode}`",
        f"- Balanced subset active: {balanced_subset_active}",
        f"- Balanced subset cleaned dataset exists: {balanced_subset_exists}",
        f"- User profiles exist: {user_profiles_exist}",
        f"- User evaluation splits exist: {user_splits_exist}",
        f"- Active user splits path: `{active_split_path}`",
        f"- Active split mode: `{active_split_mode}`",
        f"- Active split count: {active_split_count}",
        f"- Holdout in candidate pool splits: {holdout_in_candidate_pool_splits}",
        f"- User baseline recommendations exist: {user_baseline_exists}",
        f"- Tiny dry-run prompt preview exists: {prompt_preview_exists}",
        f"- Meaningful candidate report exists: {meaningful_candidate_report_exists}",
        f"- Meaningful candidate users: {meaningful_candidate_users}",
        f"- LLM check status: `{llm_check_status}`",
        f"- Selected provider: `{selected_provider}`",
        f"- Selected model: `{selected_model or 'none'}`",
        f"- Response language: `{response_language}`",
        f"- LLM mode: `{llm_mode}`",
        f"- Real LLM calls allowed: {real_llm_calls_allowed}",
        f"- Credentials configured: {llm_credentials_configured}",
        f"- max_llm_users: {configured_max_llm_users}",
        f"- max_llm_candidates: {configured_max_llm_candidates}",
        "",
        "## Missing prerequisites",
        *missing_prereq_lines,
        "",
        "## Unsafe config",
        *unsafe_config_lines,
        "",
        "## Warnings",
        *warning_lines,
        "",
    ]
    settings.llm_pilot_readiness_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved LLM pilot readiness report to %s", settings.llm_pilot_readiness_markdown_path.relative_to(settings.project_root))
    return report


def summarize_games(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review-level data to per-game diagnostics."""

    summary = (
        reviews_df.groupby(["game_id", "game_title"], as_index=False)
        .agg(
            review_count=("review_id", "count"),
            positive_reviews=("recommended", lambda values: int(values.fillna(False).sum())),
            votes_helpful_sum=("votes_helpful", "sum"),
        )
        .sort_values(["review_count", "game_title"], ascending=[False, True])
    )
    summary["negative_reviews"] = summary["review_count"] - summary["positive_reviews"]
    summary["positive_ratio"] = (
        summary["positive_reviews"] / summary["review_count"]
    ).round(4)
    return summary


def summarize_users(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review-level data to per-user diagnostics."""

    if "user_id" not in reviews_df.columns:
        return pd.DataFrame(columns=["user_id", "review_count", "positive_review_count"])
    user_reviews = reviews_df[reviews_df["user_id"].fillna("").astype(str).str.len() > 0].copy()
    if user_reviews.empty:
        return pd.DataFrame(columns=["user_id", "review_count", "positive_review_count"])
    return user_reviews.groupby("user_id", as_index=False).agg(
        review_count=("review_id", "count"),
        positive_review_count=("recommended", lambda values: int(values.fillna(False).sum())),
    )


def records_for_report(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    """Convert a small frame slice to JSON-friendly row records."""

    if frame.empty:
        return []
    records = frame.loc[:, columns].copy()
    return json.loads(records.to_json(orient="records", force_ascii=False))


def get_active_preprocessing_summary_path(settings: Settings) -> Path:
    """Return the summary path that corresponds to the active cleaned reviews file."""

    if settings.active_processed_reviews_path == settings.reviews_clean_subset_path:
        return settings.preprocessing_subset_summary_path
    if settings.active_processed_reviews_path == settings.reviews_clean_debug_path:
        return settings.preprocessing_debug_summary_path
    if settings.active_processed_reviews_path == settings.reviews_clean_balanced_subset_path:
        return settings.preprocessing_balanced_subset_summary_path
    return settings.preprocessing_summary_path


def get_active_dataset_mode(settings: Settings) -> str:
    """Classify the currently active processed dataset path."""

    path = settings.active_processed_reviews_path
    if path == settings.reviews_clean_balanced_subset_path:
        return "balanced_subset"
    if path == settings.reviews_clean_subset_path:
        return "subset"
    if path == settings.reviews_clean_debug_path:
        return "debug"
    if path == settings.reviews_clean_path:
        return "full"
    return "custom"


def get_active_processed_reviews_summary_path(settings: Settings) -> Path:
    """Return the preprocessing summary for the active processed dataset."""

    return get_active_preprocessing_summary_path(settings)


def load_user_based_artifact_summary(settings: Settings) -> dict[str, object]:
    """Load the user-based artifact summaries when they exist."""

    active_split_path = get_active_user_splits_path(settings)
    active_split_mode = get_active_user_split_mode(settings)
    split_summary_path = (
        settings.user_evaluation_split_pilot_summary_path
        if active_split_mode == "pilot"
        else settings.user_split_summary_path
    )
    profile_summary = load_json_if_exists(settings.user_profile_summary_path)
    split_summary = load_json_if_exists(split_summary_path)
    metrics_summary = load_csv_summary_if_exists(settings.user_metrics_summary_path)
    profiles_exist = settings.user_profiles_path.exists()
    splits_exist = active_split_path.exists()
    active_split_count = 0
    if splits_exist:
        try:
            active_split_count = len(read_jsonl(active_split_path))
        except Exception:
            active_split_count = 0
    warnings: list[str] = []
    if active_split_mode == "pilot" and not splits_exist:
        warnings.append("Pilot splits are configured but the pilot split file is missing.")
    metrics_exist = settings.user_metrics_summary_path.exists()
    evaluated_users = 0
    if metrics_exist and settings.user_metrics_summary_path.exists():
        try:
            metrics_df = pd.read_csv(settings.user_metrics_summary_path)
            if not metrics_df.empty and "evaluated_profiles" in metrics_df:
                evaluated_users = int(metrics_df["evaluated_profiles"].max())
        except Exception:
            evaluated_users = 0

    return {
        "profile_summary": profile_summary,
        "split_summary": split_summary,
        "metrics_summary": metrics_summary,
        "profiles_exist": profiles_exist,
        "splits_exist": splits_exist,
        "active_split_path": str(active_split_path),
        "active_split_mode": active_split_mode,
        "active_split_exists": splits_exist,
        "active_split_count": int(active_split_count),
        "warnings": warnings,
        "metrics_exist": metrics_exist,
        "profile_count": int(profile_summary.get("profile_count", 0)),
        "split_count": int(split_summary.get("split_count", 0)),
        "evaluated_users": int(evaluated_users),
    }


def build_active_experiment_context(settings: Settings) -> dict[str, object]:
    """Summarize the currently active experiment path and readiness."""

    active_mode = get_active_dataset_mode(settings)
    active_processed_reviews = str(settings.active_processed_reviews_path)
    full_preprocessing_completed = cleaned_reviews_artifact_is_valid(settings.reviews_clean_path)
    balanced_subset_ready = cleaned_reviews_artifact_is_valid(settings.reviews_clean_balanced_subset_path)
    user_artifacts = load_user_based_artifact_summary(settings)
    active_split_path = get_active_user_splits_path(settings)
    active_split_mode = get_active_user_split_mode(settings)
    balanced_subset_user_based_ready = (
        active_mode == "balanced_subset"
        and balanced_subset_ready
        and int(user_artifacts["profile_count"]) > 0
        and int(user_artifacts["active_split_count"]) > 0
        and int(user_artifacts["evaluated_users"]) > 0
    )
    full_user_based_ready = (
        active_mode == "full"
        and full_preprocessing_completed
        and int(user_artifacts["profile_count"]) > 0
        and int(user_artifacts["active_split_count"]) > 0
        and int(user_artifacts["evaluated_users"]) > 0
    )
    current_workflow = "not_ready"
    if balanced_subset_user_based_ready:
        current_workflow = "balanced_subset_user_based_baseline"
    elif full_user_based_ready:
        current_workflow = "full_user_based_baseline"
    elif active_mode == "balanced_subset":
        current_workflow = (
            "balanced_subset_user_based_baseline"
            if int(user_artifacts["active_split_count"]) > 0
            else "balanced_subset_user_based_baseline_incomplete"
        )
    elif active_mode == "full":
        current_workflow = (
            "full_user_based_baseline"
            if int(user_artifacts["active_split_count"]) > 0
            else "full_user_based_baseline_incomplete"
        )
    elif active_mode in {"subset", "debug"}:
        current_workflow = "technical_validation_only"

    return {
        "active_processed_reviews": active_processed_reviews,
        "active_dataset_mode": active_mode,
        "active_user_splits_path": str(active_split_path),
        "active_user_split_mode": active_split_mode,
        "active_user_splits_exist": bool(user_artifacts["active_split_exists"]),
        "active_user_split_count": int(user_artifacts["active_split_count"]),
        "full_dataset_ready": full_preprocessing_completed,
        "balanced_subset_ready": balanced_subset_ready,
        "balanced_subset_user_based_ready": balanced_subset_user_based_ready,
        "full_user_based_ready": full_user_based_ready,
        "current_recommended_workflow": current_workflow,
        **user_artifacts,
    }


def cleaned_reviews_artifact_is_valid(path: Path) -> bool:
    """Check whether a cleaned reviews CSV has the expected normalized header."""

    if not path.exists():
        return False
    try:
        columns = [str(column) for column in pd.read_csv(path, nrows=0).columns.tolist()]
    except Exception:
        return False
    return columns == list(NORMALIZED_COLUMN_ORDER)


def save_bar_chart(
    values: dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a simple bar chart."""

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(values.keys())
    counts = list(values.values())
    ax.bar(labels, counts, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_top_games_chart(game_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot the top games by review count."""

    top_games = game_summary.head(20).sort_values("review_count", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_games["game_title"], top_games["review_count"], color="#59A14F")
    ax.set_title("Top Games by Review Count")
    ax.set_xlabel("Review Count")
    ax.set_ylabel("Game")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_histogram(values, title: str, xlabel: str, output_path: Path) -> None:
    """Save a compact histogram when numeric values are available."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if numeric.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.hist(numeric, bins=min(30, max(5, numeric.nunique())), color="#E15759", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def numeric_statistics(values) -> dict[str, float | int]:
    """Compute compact numeric summary statistics."""

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(numeric.mean()), 4),
        "median": round(float(numeric.median()), 4),
        "min": round(float(numeric.min()), 4),
        "max": round(float(numeric.max()), 4),
    }


def build_data_diagnostics_markdown(diagnostics: dict[str, object]) -> str:
    """Render data diagnostics as a thesis-friendly markdown report."""

    length_stats = diagnostics["review_length_statistics"]
    sections = [
        "# Data Diagnostics",
        "",
        "## Overview",
        f"- Active processed dataset: `{diagnostics['active_processed_dataset_path']}`",
        f"- Dataset size: {diagnostics['dataset_size']}",
        f"- Total cleaned reviews: {diagnostics['total_cleaned_reviews']}",
        f"- Unique games: {diagnostics['unique_games']}",
        f"- Unique users: {diagnostics['unique_users']}",
        f"- Unique languages: {diagnostics['unique_languages']}",
        f"- Unique review IDs: {diagnostics['unique_review_ids']}",
        f"- Empty review count before cleaning: {diagnostics['empty_review_count_before_cleaning']}",
        f"- Duplicate review count after preprocessing: {diagnostics['duplicate_review_count_after_preprocessing']}",
        f"- Positive review ratio: {diagnostics['positive_review_ratio']}",
        f"- Games below min_reviews_per_game: {diagnostics['games_below_min_reviews_per_game']}",
        f"- Games eligible for card generation: {diagnostics['games_eligible_for_card_generation']}",
        f"- User-based mode available: {diagnostics['user_based_mode_available']}",
        f"- Eligible users for user-based evaluation: {diagnostics['eligible_users_for_user_based_evaluation']}",
        "",
        "## Warnings",
        *([f"- {warning}" for warning in diagnostics.get("warnings", [])] or ["- none"]),
        "",
        "## Review Length Statistics",
        f"- Mean: {length_stats['mean']}",
        f"- Median: {length_stats['median']}",
        f"- Min: {length_stats['min']}",
        f"- Max: {length_stats['max']}",
        f"- 25th percentile: {length_stats['percentile_25']}",
        f"- 75th percentile: {length_stats['percentile_75']}",
        "",
        "## Language Distribution",
        dict_to_bullets(diagnostics["language_distribution"]),
        "",
        "## Recommendation Distribution",
        dict_to_bullets(diagnostics["recommendation_distribution"]),
        "",
        "## Review Quality",
        f"- weighted_vote_score mean: {diagnostics['weighted_vote_score_statistics']['mean']}",
        f"- votes_helpful mean: {diagnostics['votes_helpful_statistics']['mean']}",
        f"- votes_funny mean: {diagnostics['votes_funny_statistics']['mean']}",
        f"- comment_count mean: {diagnostics['comment_count_statistics']['mean']}",
        "",
        "## User Activity",
        f"- reviews per user mean: {diagnostics['reviews_per_user_statistics']['mean']}",
        f"- positive reviews per user mean: {diagnostics['positive_reviews_per_user_statistics']['mean']}",
        f"- author_num_games_owned mean: {diagnostics['author_num_games_owned_statistics']['mean']}",
        f"- author_num_reviews mean: {diagnostics['author_num_reviews_statistics']['mean']}",
        "",
        "## Playtime",
        f"- playtime_forever mean: {diagnostics['playtime_forever_statistics']['mean']}",
        f"- playtime_at_review mean: {diagnostics['playtime_at_review_statistics']['mean']}",
        f"- playtime_last_two_weeks mean: {diagnostics['playtime_last_two_weeks_statistics']['mean']}",
        f"- percentage of reviews with playtime > 0: {diagnostics['percentage_reviews_with_playtime_gt_zero']}",
        "",
        "## Review Context",
        f"- steam_purchase ratio: {diagnostics['steam_purchase_ratio']}",
        f"- received_for_free ratio: {diagnostics['received_for_free_ratio']}",
        f"- written_during_early_access ratio: {diagnostics['written_during_early_access_ratio']}",
        "",
        "## Top Games by Review Count",
        dataframe_to_markdown(pd.DataFrame(diagnostics["top_20_games_by_review_count"])),
        "",
        "## Top Games by Positive Ratio",
        dataframe_to_markdown(
            pd.DataFrame(diagnostics["top_20_games_by_positive_ratio_with_enough_reviews"])
        ),
        "",
        "## Top Games by Helpful Votes",
        dataframe_to_markdown(pd.DataFrame(diagnostics["top_20_games_by_helpful_votes"])),
    ]
    return "\n".join(sections)


def dict_to_bullets(values: dict[str, object]) -> str:
    """Render a small dictionary as markdown bullets."""

    if not values:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in values.items())


def load_game_cards_from_artifact(path: Path) -> list[GameCard]:
    """Load generated game cards from a JSONL artifact."""

    if not path.exists():
        raise FileNotFoundError(
            f"Game cards not found at '{path}'. Run `python main.py --step build_cards` first."
        )
    return [GameCard(**record) for record in read_jsonl(path)]


def load_optional_game_cards(path: Path) -> list[GameCard]:
    """Load game cards when present, otherwise return an empty list."""

    if not path.exists():
        return []
    return [GameCard(**record) for record in read_jsonl(path)]


def load_optional_records(path: Path) -> list[dict[str, object]]:
    """Load JSONL records when present, otherwise return an empty list."""

    if not path.exists():
        return []
    return read_jsonl(path)


def load_scenario_records_for_validation(
    settings: Settings,
) -> tuple[list[dict[str, object]], str]:
    """Load scenarios from the configured source or the processed artifact."""

    logger = get_logger()
    if settings.scenarios_file is not None:
        if settings.scenarios_file.exists():
            return load_external_scenarios(settings.scenarios_file), str(settings.scenarios_file)
        logger.warning(
            "Configured SCENARIOS_FILE %s does not exist. Falling back to processed scenarios.",
            settings.scenarios_file,
        )

    if settings.scenarios_output_path.exists():
        return read_jsonl(settings.scenarios_output_path), str(settings.scenarios_output_path)
    return [], "no scenario source found"


def validate_single_scenario_record(
    record: dict[str, object],
    scenario_index: int,
    valid_game_ids: set[str],
    top_k: int,
) -> dict[str, object]:
    """Validate one raw scenario record and produce a report row."""

    raw_scenario_id = str(record.get("scenario_id", "")).strip()
    scenario_id = raw_scenario_id or f"<missing_scenario_id_{scenario_index}>"
    scenario_type = str(record.get("scenario_type", "")).strip()
    preference_text = str(record.get("preference_text", "")).strip()
    seed_game_ids = normalize_id_list(record.get("seed_game_ids"))
    excluded_game_ids = normalize_id_list(record.get("excluded_game_ids"))
    ground_truth_game_ids = normalize_id_list(record.get("ground_truth_game_ids"))
    candidate_game_ids = normalize_id_list(record.get("candidate_game_ids"))

    missing_fields: list[str] = []
    if not raw_scenario_id:
        missing_fields.append("scenario_id")
    if not scenario_type:
        missing_fields.append("scenario_type")
    if not preference_text:
        missing_fields.append("preference_text")
    if not ground_truth_game_ids:
        missing_fields.append("ground_truth_game_ids")

    unknown_by_field = {
        "seed_game_ids": [game_id for game_id in seed_game_ids if game_id not in valid_game_ids],
        "excluded_game_ids": [game_id for game_id in excluded_game_ids if game_id not in valid_game_ids],
        "ground_truth_game_ids": [game_id for game_id in ground_truth_game_ids if game_id not in valid_game_ids],
        "candidate_game_ids": [game_id for game_id in candidate_game_ids if game_id not in valid_game_ids],
    }
    known_seed_ids = [game_id for game_id in seed_game_ids if game_id in valid_game_ids]
    known_excluded_ids = [game_id for game_id in excluded_game_ids if game_id in valid_game_ids]
    known_ground_truth_ids = [game_id for game_id in ground_truth_game_ids if game_id in valid_game_ids]
    known_candidate_ids = [game_id for game_id in candidate_game_ids if game_id in valid_game_ids]

    warnings: list[str] = []
    invalid_reasons: list[str] = []

    candidate_overlap = sorted(set(known_candidate_ids) & set(known_excluded_ids + known_seed_ids))
    if candidate_overlap:
        warnings.append(
            "candidate_game_ids contains seed or excluded ids: " + "|".join(candidate_overlap)
        )

    excluded_ground_truth = sorted(set(known_ground_truth_ids) & set(known_excluded_ids))
    if excluded_ground_truth:
        invalid_reasons.append(
            "ground_truth_game_ids overlap with excluded_game_ids: " + "|".join(excluded_ground_truth)
        )

    seed_ground_truth = sorted(set(known_ground_truth_ids) & set(known_seed_ids))
    if seed_ground_truth:
        warnings.append(
            "ground_truth_game_ids overlap with seed_game_ids: " + "|".join(seed_ground_truth)
        )

    candidate_pool = compute_candidate_pool(
        valid_game_ids=valid_game_ids,
        known_seed_ids=known_seed_ids,
        known_excluded_ids=known_excluded_ids,
        known_candidate_ids=known_candidate_ids,
    )
    candidate_count = len(candidate_pool)
    if candidate_count < top_k:
        warnings.append(
            f"Only {candidate_count} valid candidates are available for top-{top_k} evaluation."
        )
    if candidate_count == 0:
        invalid_reasons.append("No valid candidate games remain after filtering.")
    if ground_truth_game_ids and not known_ground_truth_ids:
        invalid_reasons.append(
            "ground_truth_game_ids were provided but none exist in the current game cards."
        )

    if known_candidate_ids:
        missing_ground_truth_from_candidates = [
            game_id for game_id in known_ground_truth_ids if game_id not in candidate_pool
        ]
        if missing_ground_truth_from_candidates:
            invalid_reasons.append(
                "ground_truth_game_ids are missing from the candidate pool: "
                + "|".join(missing_ground_truth_from_candidates)
            )

    if any(unknown_by_field.values()):
        warnings.append("Some referenced game ids are not present in the current game cards.")

    status = "ok"
    if missing_fields or invalid_reasons:
        status = "invalid"
    elif warnings:
        status = "warning"

    return {
        "scenario_id": scenario_id,
        "status": status,
        "missing_fields": "|".join(missing_fields),
        "unknown_game_ids": format_unknown_game_ids(unknown_by_field),
        "candidate_count": candidate_count,
        "ground_truth_count": len(known_ground_truth_ids),
        "warnings": " ; ".join([*invalid_reasons, *warnings]),
    }


def compute_candidate_pool(
    valid_game_ids: set[str],
    known_seed_ids: list[str],
    known_excluded_ids: list[str],
    known_candidate_ids: list[str],
) -> list[str]:
    """Compute the valid candidate pool after scenario filtering."""

    blocked_ids = set(known_seed_ids) | set(known_excluded_ids)
    if known_candidate_ids:
        return [game_id for game_id in known_candidate_ids if game_id not in blocked_ids]
    return sorted(game_id for game_id in valid_game_ids if game_id not in blocked_ids)


def format_unknown_game_ids(unknown_by_field: dict[str, list[str]]) -> str:
    """Format missing id references into a compact report string."""

    parts = []
    for field_name, ids in unknown_by_field.items():
        if ids:
            parts.append(f"{field_name}:{'|'.join(ids)}")
    return "; ".join(parts)


def determine_scenario_mode(scenarios: list[dict[str, object]]) -> str:
    """Describe whether the current scenarios are synthetic or manual/predefined."""

    if not scenarios:
        return "none"

    types = [str(record.get("scenario_type", "")).strip() for record in scenarios]
    if all(scenario_type == "synthetic_demo" for scenario_type in types):
        return "synthetic_demo_only"
    if all(scenario_type in {"manual", "seed_games"} for scenario_type in types):
        return "manual_or_predefined"
    return "mixed"


def rank_similar_cards(
    seed_card: GameCard,
    game_cards: list[GameCard],
    matrix,
    id_to_index: dict[str, int],
) -> list[GameCard]:
    """Rank other game cards by TF-IDF similarity to a seed card."""

    seed_index = id_to_index[seed_card.game_id]
    similarities = cosine_similarity(matrix[seed_index], matrix).flatten()
    ranked_indices = sorted(
        range(len(game_cards)),
        key=lambda index: similarities[index],
        reverse=True,
    )
    return [
        game_cards[index]
        for index in ranked_indices
        if game_cards[index].game_id != seed_card.game_id
    ]


def build_draft_preference_text(seed_card: GameCard) -> str:
    """Build a readable manual-draft preference description from one seed card."""

    positive_part = ", ".join(seed_card.positive_keywords[:4] or ["none"])
    negative_part = ", ".join(seed_card.negative_keywords[:3] or ["none"])
    return (
        f"Player is looking for games similar to {seed_card.game_title}. "
        f"Desired qualities include {positive_part}. "
        f"The player would prefer to avoid {negative_part}."
    )


def build_scenario_hint(card: GameCard) -> str:
    """Generate a short scenario-design hint from existing card fields only."""

    positive_part = ", ".join(card.positive_keywords[:3] or ["positive reception"])
    negative_part = ", ".join(card.negative_keywords[:2] or ["few repeated complaints"])
    return (
        f"Good candidate for scenarios about {positive_part}, while accounting for concerns about {negative_part}."
    )


def first_or_empty(values: list[str]) -> str:
    """Return the first list item or an empty string."""

    return values[0] if values else ""


def scenario_record_to_csv_row(record: dict[str, object]) -> dict[str, object]:
    """Convert a scenario record to a CSV-friendly row."""

    list_fields = {
        "seed_game_ids",
        "excluded_game_ids",
        "ground_truth_game_ids",
        "candidate_game_ids",
    }
    row: dict[str, object] = {}
    for key, value in record.items():
        if key in list_fields:
            row[key] = "|".join(str(item) for item in value)
        else:
            row[key] = value
    return row


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    """Deduplicate strings while preserving the first occurrence order."""

    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)
    return deduplicated


def write_json_report(path: Path, payload: dict[str, object]) -> None:
    """Write a small JSON report."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small markdown table without optional dependencies."""

    if frame.empty:
        return "_No rows available._"

    headers = [str(column) for column in frame.columns]
    separator = ["---"] * len(headers)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(str(row[column]) for column in frame.columns)
            + " |"
        )
    return "\n".join(rows)


def generate_case_studies(settings: Settings) -> dict[str, object]:
    """Generate compact case-study artifacts from recommendation outputs."""

    logger = get_logger()
    artifacts = load_analysis_artifacts(settings)
    llm_available = artifacts["llm_has_ranked_rows"]

    rank_frame = build_rank_comparison_frame(artifacts)
    save_rank_comparison_outputs(settings, rank_frame, artifacts)

    explanation_frame, explanation_note = build_llm_explanation_checks_frame(artifacts)
    save_llm_explanation_outputs(settings, explanation_frame, explanation_note)

    case_study_records = select_case_study_records(artifacts, rank_frame)
    payload = {
        "note": build_case_study_note(artifacts, llm_available, len(case_study_records)),
        "llm_available": llm_available,
        "case_studies": case_study_records,
    }
    write_json_report(settings.case_studies_json_path, payload)
    settings.case_studies_markdown_path.write_text(
        build_case_studies_markdown(payload),
        encoding="utf-8",
    )
    logger.info(
        "Saved case studies to %s and %s",
        settings.case_studies_json_path.relative_to(settings.project_root),
        settings.case_studies_markdown_path.relative_to(settings.project_root),
    )
    return payload


def generate_recommendation_examples(settings: Settings) -> str:
    """Generate a report-ready markdown file with recommendation examples."""

    artifacts = load_analysis_artifacts(settings)
    scenario_records = artifacts["scenario_records"]
    note_lines = build_analysis_warning_lines(artifacts)
    lines = ["# Recommendation Examples", ""]
    if note_lines:
        lines.extend(note_lines)
        lines.append("")

    if not scenario_records:
        lines.append("_No scenarios available._")
    else:
        selected_scenarios = scenario_records[: min(10, len(scenario_records))]
        for scenario in selected_scenarios:
            scenario_id = str(scenario["scenario_id"])
            lines.extend(
                build_recommendation_example_section(
                    scenario=scenario,
                    artifacts=artifacts,
                )
            )
            lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    settings.recommendation_examples_markdown_path.write_text(content, encoding="utf-8")
    get_logger().info(
        "Saved recommendation examples to %s",
        settings.recommendation_examples_markdown_path.relative_to(settings.project_root),
    )
    return content


def generate_thesis_tables(settings: Settings) -> dict[str, str]:
    """Generate markdown-ready tables for the thesis experimental chapter."""

    metrics_content = build_thesis_metrics_table(settings)
    user_metrics_content = build_user_thesis_metrics_table(settings)
    dataset_content = build_thesis_dataset_table(settings)
    balanced_subset_content = build_thesis_balanced_subset_dataset_table(settings)
    methodology_note_content = build_balanced_subset_methodology_note(settings)
    scenario_content = build_thesis_scenario_table(settings)

    settings.thesis_metrics_table_path.write_text(metrics_content, encoding="utf-8")
    settings.reports_dir.joinpath("user_thesis_metrics_table.md").write_text(
        user_metrics_content,
        encoding="utf-8",
    )
    settings.thesis_dataset_table_path.write_text(dataset_content, encoding="utf-8")
    settings.reports_dir.joinpath("thesis_balanced_subset_dataset_table.md").write_text(
        balanced_subset_content,
        encoding="utf-8",
    )
    settings.reports_dir.joinpath("balanced_subset_methodology_note.md").write_text(
        methodology_note_content,
        encoding="utf-8",
    )
    settings.thesis_scenario_table_path.write_text(scenario_content, encoding="utf-8")

    get_logger().info(
        "Saved thesis tables to %s, %s, and %s",
        settings.thesis_metrics_table_path.relative_to(settings.project_root),
        settings.thesis_dataset_table_path.relative_to(settings.project_root),
        settings.thesis_scenario_table_path.relative_to(settings.project_root),
    )
    return {
        "metrics": metrics_content,
        "user_metrics": user_metrics_content,
        "dataset": dataset_content,
        "balanced_subset_dataset": balanced_subset_content,
        "methodology_note": methodology_note_content,
        "scenario": scenario_content,
    }


def run_analysis_suite(settings: Settings) -> None:
    """Run the local analysis/report helpers without triggering LLM calls."""

    run_data_diagnostics(settings)
    export_available_games(settings)
    validate_scenarios_from_artifacts(settings)
    build_experiment_readiness_report(settings)
    generate_case_studies(settings)
    generate_recommendation_examples(settings)
    generate_thesis_tables(settings)


def build_preflight_report(settings: Settings) -> dict[str, object]:
    """Inspect environment and artifacts before a real thesis-scale experiment."""

    logger = get_logger()
    environment_report = run_environment_check(
        settings.project_root,
        dataset_path_override=settings.reviews_csv_path,
    )
    preprocessing_summary = load_json_if_exists(get_active_preprocessing_summary_path(settings))
    subset_preprocessing_summary = load_json_if_exists(settings.preprocessing_subset_summary_path)
    balanced_subset_preprocessing_summary = load_json_if_exists(
        settings.preprocessing_balanced_subset_summary_path
    )
    game_card_summary = load_json_if_exists(settings.game_card_summary_path)
    raw_inspection = load_json_if_exists(settings.raw_dataset_inspection_json_path)
    preprocessing_progress = load_json_if_exists(settings.preprocessing_progress_json_path)
    full_clean_exists = cleaned_reviews_artifact_is_valid(settings.reviews_clean_path)
    subset_clean_exists = cleaned_reviews_artifact_is_valid(settings.reviews_clean_subset_path)
    balanced_subset_clean_exists = cleaned_reviews_artifact_is_valid(
        settings.reviews_clean_balanced_subset_path
    )
    readiness_report = (
        build_experiment_readiness_report(settings)
        if settings.active_processed_reviews_path.exists()
        else load_json_if_exists(settings.experiment_readiness_path)
    )
    active_dataset_mode = str(readiness_report.get("active_dataset_mode", get_active_dataset_mode(settings)))
    active_processed_reviews = str(
        readiness_report.get("active_processed_reviews", str(settings.active_processed_reviews_path))
    )
    full_dataset_ready = bool(readiness_report.get("full_dataset_ready", full_clean_exists))
    balanced_subset_ready = bool(
        readiness_report.get("balanced_subset_ready", balanced_subset_clean_exists)
    )
    current_recommended_workflow = str(
        readiness_report.get("current_recommended_workflow", "not_ready")
    )

    scenario_validation_exists = settings.scenario_validation_report_path.exists()
    validation_summary = {"ok_count": 0, "warning_count": 0, "invalid_count": 0}
    if scenario_validation_exists:
        validation_df = pd.read_csv(settings.scenario_validation_report_path)
        validation_summary = {
            "ok_count": int((validation_df["status"] == "ok").sum()),
            "warning_count": int((validation_df["status"] == "warning").sum()),
            "invalid_count": int((validation_df["status"] == "invalid").sum()),
        }

    missing_items: list[str] = []
    ready_items: list[str] = []

    if environment_report["summary"]["errors"] == 0:
        ready_items.append("Local Python environment and required packages look usable.")
    else:
        missing_items.append("Environment check reported missing dependencies or other critical errors.")

    if Path(environment_report["configured_dataset_path"]).exists():
        ready_items.append("Configured Steam Reviews dataset path exists.")
    else:
        missing_items.append("Configured raw dataset path does not exist.")

    required_artifacts = [
        ("preprocessing summary", get_active_preprocessing_summary_path(settings)),
        ("game card summary", settings.game_card_summary_path),
        ("experiment readiness report", settings.experiment_readiness_path),
    ]
    for label, path in required_artifacts:
        if path.exists():
            ready_items.append(f"{label.capitalize()} is available.")
        else:
            missing_items.append(f"{label.capitalize()} is missing.")

    if settings.preprocessing_progress_json_path.exists():
        ready_items.append("Preprocessing progress file is available.")
    else:
        missing_items.append("Preprocessing progress file is missing.")

    llm_credentials_configured = provider_credentials_configured(settings)
    provider_display_name = "GigaChat" if get_effective_llm_provider(settings) == "gigachat" else get_effective_llm_provider(settings).capitalize()
    if llm_credentials_configured:
        ready_items.append(f"{provider_display_name} credentials are configured for optional LLM reranking.")
    else:
        missing_items.append(f"{provider_display_name} credentials are not configured, so LLM reranking cannot run.")

    scenario_mode = str(readiness_report.get("scenario_mode", "none"))
    dataset_too_small = bool(readiness_report.get("dataset_too_small", True))
    manual_scenario_count = int(readiness_report.get("manual_scenario_count", 0))
    invalid_scenarios = int(validation_summary["invalid_count"])
    has_user_id = bool(readiness_report.get("has_user_id", False))
    user_based_mode_available = bool(readiness_report.get("user_based_mode_available", False))
    unique_users = int(readiness_report.get("unique_users", 0))
    eligible_user_count = int(readiness_report.get("eligible_user_count", 0))
    can_run_user_experiment = bool(readiness_report.get("can_run_user_experiment", False))
    balanced_subset_user_ready = bool(readiness_report.get("balanced_subset_user_based_ready", False))
    full_user_ready = bool(readiness_report.get("full_user_based_ready", False))
    preprocessing_completed = preprocessing_progress.get("status") == "completed"
    full_preprocessing_completed = (
        preprocessing_progress.get("status") == "completed"
        and preprocessing_progress.get("mode") == "full"
        and full_clean_exists
        and bool(preprocessing_summary)
    )
    raw_inspection = load_json_if_exists(settings.raw_dataset_inspection_json_path)
    looks_like_sample_or_incomplete = bool(
        raw_inspection.get("likely_sample_or_incomplete", False)
        or (dataset_too_small and active_dataset_mode != "balanced_subset")
    )
    artifacts_match_current_dataset = True
    if raw_inspection and preprocessing_summary:
        artifacts_match_current_dataset = int(
            raw_inspection.get("raw_row_count", 0)
        ) == int(preprocessing_summary.get("raw_row_count", 0))
    balanced_subset_recommended = bool(
        not full_preprocessing_completed
        and balanced_subset_clean_exists
        and int(balanced_subset_preprocessing_summary.get("unique_games", 0)) >= 50
        and int(balanced_subset_preprocessing_summary.get("unique_users", 0)) >= 1000
        and int(balanced_subset_preprocessing_summary.get("reviews_with_user_id", 0)) > 0
    )
    warnings: list[str] = []

    if dataset_too_small and active_dataset_mode != "balanced_subset":
        missing_items.append("Dataset is still below the configured thesis-scale thresholds.")
    if (
        scenario_mode != "manual_or_predefined"
        and active_dataset_mode not in {"balanced_subset", "full"}
        and not balanced_subset_user_ready
        and not full_user_ready
    ):
        missing_items.append("Manual/predefined scenarios are not ready yet.")
    elif (
        active_dataset_mode not in {"balanced_subset", "full"}
        and manual_scenario_count < settings.min_manual_scenarios_for_real_experiment
    ):
        missing_items.append(
            "Manual/predefined scenario count is below the recommended thesis threshold."
        )

    if invalid_scenarios == 0 and scenario_validation_exists:
        ready_items.append("Scenario validation does not report invalid scenarios.")
    elif scenario_validation_exists:
        missing_items.append(f"Scenario validation reports {invalid_scenarios} invalid scenarios.")
    if not artifacts_match_current_dataset:
        missing_items.append(
            "Processed artifacts are out of sync with the currently selected raw dataset."
        )
    if active_dataset_mode == "full" and not full_preprocessing_completed:
        missing_items.append("Preprocessing has not completed successfully for the full dataset.")
    if active_dataset_mode == "balanced_subset" and not balanced_subset_ready:
        missing_items.append("Balanced subset preprocessing has not completed successfully.")
    if active_dataset_mode == "full" and not full_clean_exists:
        missing_items.append("Full cleaned reviews file is missing.")
    if active_dataset_mode == "balanced_subset" and not full_clean_exists:
        warnings.append("full_dataset_preprocessing_incomplete")
    if subset_clean_exists and not full_clean_exists and active_dataset_mode == "subset":
        missing_items.append(
            "Only the subset cleaned file is available. Final full-dataset preprocessing has not completed."
        )
    if balanced_subset_clean_exists and not full_clean_exists:
        ready_items.append("Balanced subset cleaned reviews file exists.")
    if balanced_subset_clean_exists and balanced_subset_ready:
        ready_items.append(
            "Balanced subset looks suitable for development or a limited thesis experiment, but it must be documented as a subset-based experiment."
        )

    active_user_artifacts_missing = active_dataset_mode in {"balanced_subset", "full"} and not (
        settings.user_profiles_path.exists()
        and get_active_user_splits_path(settings).exists()
        and settings.user_metrics_summary_path.exists()
    )
    processed_artifacts_missing = any(not path.exists() for _, path in required_artifacts) or active_user_artifacts_missing
    manual_scenarios_ready = (
        scenario_mode == "manual_or_predefined"
        and manual_scenario_count >= settings.min_manual_scenarios_for_real_experiment
        and invalid_scenarios == 0
    )

    scenario_validation_blocks = invalid_scenarios > 0 and active_dataset_mode not in {"balanced_subset", "full"}

    if environment_report["summary"]["errors"] > 0 or processed_artifacts_missing or scenario_validation_blocks:
        status = "not_ready"
    elif active_dataset_mode == "balanced_subset" and balanced_subset_user_ready and has_user_id:
        status = (
            "ready_for_balanced_subset_user_based_llm_experiment"
            if llm_credentials_configured
            else "ready_for_balanced_subset_user_based_baseline_experiment"
        )
    elif active_dataset_mode == "full" and full_user_ready and has_user_id:
        status = (
            "ready_for_full_user_based_llm_experiment"
            if llm_credentials_configured
            else "ready_for_full_user_based_baseline_experiment"
        )
    elif dataset_too_small or scenario_mode == "synthetic_demo_only":
        status = "technical_validation_only"
    else:
        status = "not_ready"

    recommended_next_steps = build_preflight_next_steps(
        status=status,
        dataset_too_small=dataset_too_small,
        scenario_mode=scenario_mode,
        llm_credentials_configured=llm_credentials_configured,
        scenario_validation_exists=scenario_validation_exists,
        invalid_scenarios=invalid_scenarios,
        preprocessing_completed=preprocessing_completed,
        subset_cleaned_reviews_exist=subset_clean_exists,
        full_cleaned_reviews_exist=full_clean_exists,
        active_dataset_mode=active_dataset_mode,
        balanced_subset_recommended=balanced_subset_clean_exists and balanced_subset_ready,
    )
    commands_to_run_next = build_preflight_commands(
        status=status,
        scenario_validation_exists=scenario_validation_exists,
    )
    blocking_issues: list[str] = []
    if environment_report["summary"]["errors"] > 0:
        blocking_issues.append("environment_errors")
    if processed_artifacts_missing:
        blocking_issues.append("missing_required_artifacts")
    if invalid_scenarios > 0 and active_dataset_mode not in {"balanced_subset", "full"}:
        blocking_issues.append("invalid_scenarios")
    if dataset_too_small:
        warnings.append("tiny_dataset")
    if looks_like_sample_or_incomplete:
        warnings.append("sample_or_incomplete_dataset")
    if scenario_mode == "synthetic_demo_only":
        warnings.append("synthetic_scenarios_only")
    if not llm_credentials_configured:
        warnings.append("llm_credentials_missing")
    if not full_preprocessing_completed and active_dataset_mode not in {"balanced_subset", "full"}:
        warnings.append("preprocessing_incomplete")
    if not artifacts_match_current_dataset:
        warnings.append("processed_artifacts_out_of_sync")
    if not has_user_id:
        warnings.append("user_id_missing")
    elif not can_run_user_experiment:
        warnings.append("not_enough_eligible_users")
    if balanced_subset_recommended:
        warnings.append("balanced_subset_recommended")

    report = {
        "status": status,
        "is_tiny_dataset": dataset_too_small,
        "uses_synthetic_scenarios": scenario_mode == "synthetic_demo_only",
        "has_manual_scenarios": manual_scenario_count > 0,
        "looks_like_sample_or_incomplete": looks_like_sample_or_incomplete,
        "artifacts_match_current_dataset": artifacts_match_current_dataset,
        "full_dataset_ready": full_dataset_ready,
        "balanced_subset_ready": balanced_subset_ready,
        "active_processed_reviews": active_processed_reviews,
        "active_dataset_mode": active_dataset_mode,
        "current_recommended_workflow": current_recommended_workflow,
        "has_user_id": has_user_id,
        "user_based_mode_available": user_based_mode_available,
        "unique_users": unique_users,
        "eligible_user_count": eligible_user_count,
        "min_user_reviews": settings.min_user_reviews,
        "min_user_positive_reviews": settings.min_user_positive_reviews,
        "can_run_user_experiment": can_run_user_experiment,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "environment_summary": environment_report["summary"],
        "configured_dataset_path": environment_report["configured_dataset_path"],
        "llm_credentials_configured": llm_credentials_configured,
        "scenario_mode": scenario_mode,
        "dataset_too_small": dataset_too_small,
        "manual_scenario_count": manual_scenario_count,
        "validation_summary": validation_summary,
        "preprocessing_progress_available": settings.preprocessing_progress_json_path.exists(),
        "preprocessing_progress_status": preprocessing_progress.get("status", "missing"),
        "preprocessing_progress_mode": preprocessing_progress.get("mode", "unknown"),
        "full_preprocessing_completed": full_preprocessing_completed,
        "full_cleaned_reviews_exist": full_clean_exists,
        "subset_cleaned_reviews_exist": subset_clean_exists,
        "balanced_subset_exists": balanced_subset_clean_exists,
        "balanced_subset_cleaned_reviews": int(
            balanced_subset_preprocessing_summary.get("final_row_count", 0)
        ),
        "balanced_subset_unique_games": int(
            balanced_subset_preprocessing_summary.get("unique_games", 0)
        ),
        "balanced_subset_unique_users": int(
            balanced_subset_preprocessing_summary.get("unique_users", 0)
        ),
        "balanced_subset_recommended_for_experiment": balanced_subset_recommended,
        "preprocessing_summary_available": bool(preprocessing_summary),
        "game_card_summary_available": bool(game_card_summary),
        "scenario_validation_report_available": scenario_validation_exists,
        "experiment_readiness_report_available": bool(readiness_report),
        "balanced_subset_user_based_ready": balanced_subset_user_ready,
        "full_user_based_ready": full_user_ready,
        "what_is_ready": deduplicate_preserve_order(ready_items),
        "what_is_missing": deduplicate_preserve_order(missing_items),
        "recommended_next_steps": recommended_next_steps,
        "commands_to_run_next": commands_to_run_next,
    }

    write_json_report(settings.preflight_report_json_path, report)
    settings.preflight_report_markdown_path.write_text(
        build_preflight_markdown(report),
        encoding="utf-8",
    )
    logger.info(
        "Saved preflight report to %s and %s",
        settings.preflight_report_json_path.relative_to(settings.project_root),
        settings.preflight_report_markdown_path.relative_to(settings.project_root),
    )
    logger.info("Preflight status: %s", status)
    return report


def run_smoke_test(settings: Settings) -> dict[str, object]:
    """Run a small baseline-only validation sequence on the current dataset."""

    logger = get_logger()
    report: dict[str, object] = {
        "success": False,
        "llm_called": False,
        "steps": [],
    }
    try:
        environment_report = run_environment_check(
            settings.project_root,
            dataset_path_override=settings.reviews_csv_path,
        )
        report["environment_summary"] = environment_report["summary"]
        if environment_report["summary"]["errors"] > 0:
            raise RuntimeError("environment check reported critical errors")
        report["steps"].append({"step": "check_env", "status": "ok"})

        reviews_df = load_reviews_csv(settings)
        cleaned_reviews_df = preprocess_reviews(reviews_df, settings)
        report["steps"].append(
            {"step": "preprocess", "status": "ok", "row_count": int(len(cleaned_reviews_df))}
        )

        game_cards = build_game_cards(cleaned_reviews_df, settings)
        report["steps"].append(
            {"step": "build_cards", "status": "ok", "game_card_count": int(len(game_cards))}
        )

        scenarios = build_scenarios(game_cards, settings)
        report["steps"].append(
            {"step": "build_scenarios", "status": "ok", "scenario_count": int(len(scenarios))}
        )

        validate_scenarios_from_artifacts(settings)
        report["steps"].append({"step": "validate_scenarios", "status": "ok"})

        baseline_results = run_baseline(scenarios, game_cards, settings)
        report["steps"].append(
            {
                "step": "baseline",
                "status": "ok",
                "recommendation_rows": int(len(baseline_results)),
            }
        )

        evaluate_recommendations(
            scenarios=scenarios,
            recommendation_sets=[baseline_results, []],
            settings=settings,
            reviews_clean_count=len(cleaned_reviews_df),
            game_cards=game_cards,
        )
        report["steps"].append({"step": "evaluate", "status": "ok"})

        generate_thesis_tables(settings)
        report["steps"].append({"step": "thesis_tables", "status": "ok"})

        report["success"] = True
        report["cleaned_review_count"] = int(len(cleaned_reviews_df))
        report["game_card_count"] = int(len(game_cards))
        report["scenario_count"] = int(len(scenarios))
        write_json_report(settings.smoke_test_report_path, report)
        logger.info(
            "Saved smoke test report to %s",
            settings.smoke_test_report_path.relative_to(settings.project_root),
        )
        print("Smoke test passed.")
        return report
    except Exception as exc:
        report["error"] = str(exc)
        report["steps"].append({"step": "failed", "status": "error", "detail": str(exc)})
        write_json_report(settings.smoke_test_report_path, report)
        logger.error(
            "Saved smoke test failure report to %s",
            settings.smoke_test_report_path.relative_to(settings.project_root),
        )
        print(f"Smoke test failed: {exc}")
        return report


def load_experiment_config(
    project_root: Path,
    config_path: Path | None = None,
) -> tuple[dict[str, object], Path | None]:
    """Load an optional experiment configuration JSON file."""

    resolved_path = config_path
    if resolved_path is None:
        default_path = project_root / "configs" / "experiment_config.json"
        if default_path.exists():
            resolved_path = default_path

    if resolved_path is None:
        return {}, None

    if not resolved_path.is_absolute():
        resolved_path = project_root / resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Experiment config not found at '{resolved_path}'.")

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must contain a JSON object.")
    return payload, resolved_path


def apply_experiment_settings_overrides(
    settings: Settings,
    config: dict[str, object],
) -> None:
    """Apply supported experiment-config values onto runtime settings."""

    if "dataset_path" in config:
        settings.reviews_csv_path = resolve_local_path(
            settings.project_root,
            str(config["dataset_path"]),
        )
    if "scenarios_file" in config:
        raw_value = str(config["scenarios_file"]).strip()
        settings.scenarios_file = (
            resolve_local_path(settings.project_root, raw_value) if raw_value else None
        )
    if "active_processed_reviews" in config:
        raw_value = str(config["active_processed_reviews"]).strip()
        if raw_value:
            settings.active_processed_reviews_path = resolve_local_path(settings.project_root, raw_value)
    if "pilot_splits_path" in config:
        raw_value = str(config["pilot_splits_path"]).strip()
        if raw_value:
            settings.pilot_splits_path = resolve_local_path(settings.project_root, raw_value)
    if "language_filter" in config:
        raw_language_filter = config["language_filter"]
        if isinstance(raw_language_filter, list):
            cleaned_values = [str(value).strip() for value in raw_language_filter if str(value).strip()]
            settings.language_filter = cleaned_values[0] if cleaned_values else None
        else:
            settings.language_filter = str(raw_language_filter).strip() or None
    if "allow_llm_skip" in config:
        settings.allow_llm_skip = bool(config["allow_llm_skip"])
    if "allow_llm_fallback" in config:
        settings.allow_llm_fallback = bool(config["allow_llm_fallback"])
    if "llm_save_response_preview" in config:
        settings.llm_save_response_preview = bool(config["llm_save_response_preview"])
    if "allow_balanced_subset" in config:
        settings.allow_balanced_subset = bool(config["allow_balanced_subset"])
    if "llm_provider" in config:
        raw_provider = str(config["llm_provider"]).strip().lower() or "openai"
        if raw_provider not in {"openai", "openrouter", "gigachat", "mock"}:
            raise ValueError("llm_provider must be one of: openai, openrouter, gigachat, mock.")
        settings.llm_provider = raw_provider
    if "llm_mode" in config:
        raw_mode = str(config["llm_mode"]).strip().lower() or "real"
        if raw_mode not in {"real", "mock"}:
            raise ValueError("llm_mode must be either 'real' or 'mock'.")
        settings.llm_mode = raw_mode
    if "llm_response_language" in config:
        settings.llm_response_language = normalize_llm_response_language(config["llm_response_language"])
    if "llm_model" in config:
        settings.llm_model = str(config["llm_model"]).strip() or None
    if "gigachat_model" in config:
        settings.gigachat_model = str(config["gigachat_model"]).strip() or settings.gigachat_model
    if "gigachat_auth_key" in config:
        settings.gigachat_auth_key = str(config["gigachat_auth_key"]).strip() or None
    if "gigachat_scope" in config:
        settings.gigachat_scope = str(config["gigachat_scope"]).strip() or settings.gigachat_scope
    if "gigachat_verify_ssl" in config:
        settings.gigachat_verify_ssl = bool(config["gigachat_verify_ssl"])
    if "gigachat_ca_bundle" in config:
        raw_bundle = str(config["gigachat_ca_bundle"]).strip()
        settings.gigachat_ca_bundle = raw_bundle or None
    if "gigachat_oauth_url" in config:
        settings.gigachat_oauth_url = str(config["gigachat_oauth_url"]).strip() or settings.gigachat_oauth_url
    if "gigachat_api_base_url" in config:
        settings.gigachat_api_base_url = str(config["gigachat_api_base_url"]).strip() or settings.gigachat_api_base_url
    if "llm_select_meaningful_users" in config:
        settings.llm_select_meaningful_users = bool(config["llm_select_meaningful_users"])
    if "use_pilot_splits" in config:
        settings.use_pilot_splits = bool(config["use_pilot_splits"])
    if "force_holdout_into_candidate_pool" in config:
        settings.force_holdout_into_candidate_pool = bool(config["force_holdout_into_candidate_pool"])

    int_field_map = {
        "min_review_chars": "min_review_chars",
        "max_review_chars": "max_review_chars",
        "min_reviews_per_game": "min_reviews_per_game",
        "min_user_reviews": "min_user_reviews",
        "min_user_positive_reviews": "min_user_positive_reviews",
        "user_holdout_count": "user_holdout_count",
        "user_candidate_pool_size": "user_candidate_pool_size",
        "max_users_for_experiment": "max_users_for_experiment",
        "max_llm_users": "max_llm_users",
        "top_k": "top_k",
        "baseline_candidate_count": "candidate_pool_size",
        "max_llm_scenarios": "max_llm_scenarios",
        "max_llm_candidates": "max_llm_candidates",
        "random_seed": "random_seed",
    }
    for config_key, settings_field in int_field_map.items():
        if config_key in config and config[config_key] not in {None, ""}:
            setattr(settings, settings_field, int(config[config_key]))


def build_experiment_runtime_config(
    settings: Settings,
    config: dict[str, object],
    config_path: Path | None,
    allow_tiny_override: bool,
    allow_synthetic_override: bool,
    run_llm_override: bool,
) -> dict[str, object]:
    """Combine defaults, config values, and simple CLI overrides."""

    experiment_name = str(
        config.get("experiment_name", "steam_reviews_experiment")
    ).strip() or "steam_reviews_experiment"
    experiment_mode = str(config.get("experiment_mode", "auto")).strip().lower() or "auto"
    output_root = resolve_local_path(
        settings.project_root,
        str(config.get("output_root", "experiments")),
    )
    run_llm = bool(config.get("run_llm", False))
    if run_llm_override:
        run_llm = True

    allow_tiny_dataset = bool(config.get("allow_tiny_dataset", False))
    if allow_tiny_override:
        allow_tiny_dataset = True

    allow_synthetic_scenarios = bool(config.get("allow_synthetic_scenarios", False))
    if allow_synthetic_override:
        allow_synthetic_scenarios = True

    allow_llm_skip = bool(config.get("allow_llm_skip", True))
    allow_llm_fallback = bool(config.get("allow_llm_fallback", getattr(settings, "allow_llm_fallback", True)))
    llm_save_response_preview = bool(
        config.get("llm_save_response_preview", getattr(settings, "llm_save_response_preview", True))
    )
    allow_balanced_subset = bool(config.get("allow_balanced_subset", True))
    llm_mode = str(config.get("llm_mode", getattr(settings, "llm_mode", "real"))).strip().lower() or "real"
    if llm_mode not in {"real", "mock"}:
        raise ValueError("llm_mode must be either 'real' or 'mock'.")
    llm_provider = str(config.get("llm_provider", getattr(settings, "llm_provider", "openai"))).strip().lower() or "openai"
    if llm_provider not in {"openai", "openrouter", "gigachat", "mock"}:
        raise ValueError("llm_provider must be one of: openai, openrouter, gigachat, mock.")
    llm_response_language = normalize_llm_response_language(
        config.get("llm_response_language", getattr(settings, "llm_response_language", "ru"))
    )
    return {
        "experiment_name": experiment_name,
        "experiment_mode": experiment_mode,
        "output_root": output_root,
        "run_llm": run_llm,
        "llm_mode": llm_mode,
        "llm_provider": llm_provider,
        "llm_response_language": llm_response_language,
        "llm_select_meaningful_users": bool(config.get("llm_select_meaningful_users", settings.llm_select_meaningful_users)),
        "gigachat_ca_bundle": str(getattr(settings, "gigachat_ca_bundle", "") or ""),
        "max_llm_users": int(config.get("max_llm_users", settings.max_llm_users)),
        "max_llm_candidates": int(config.get("max_llm_candidates", settings.max_llm_candidates)),
        "max_pilot_splits": int(config.get("max_pilot_splits", settings.max_pilot_splits)),
        "allow_tiny_dataset": allow_tiny_dataset,
        "allow_synthetic_scenarios": allow_synthetic_scenarios,
        "allow_balanced_subset": allow_balanced_subset,
        "allow_llm_skip": allow_llm_skip,
        "allow_llm_fallback": allow_llm_fallback,
        "llm_save_response_preview": llm_save_response_preview,
        "use_pilot_splits": bool(config.get("use_pilot_splits", settings.use_pilot_splits)),
        "force_holdout_into_candidate_pool": bool(
            config.get("force_holdout_into_candidate_pool", settings.force_holdout_into_candidate_pool)
        ),
        "notes": str(config.get("notes", "")).strip(),
        "config_path": str(config_path) if config_path else "",
        "dataset_path": str(settings.reviews_csv_path),
        "scenarios_file": str(settings.scenarios_file) if settings.scenarios_file else "",
        "active_processed_reviews": str(settings.active_processed_reviews_path),
        "pilot_splits_path": str(getattr(settings, "pilot_splits_path", "")),
    }


def run_controlled_experiment(
    settings: Settings,
    config_path: Path | None = None,
    allow_tiny_override: bool = False,
    allow_synthetic_override: bool = False,
    run_llm_override: bool = False,
) -> dict[str, object]:
    """Run the thesis experiment with safety gates and versioned outputs."""

    logger = get_logger()
    config_payload, resolved_config_path = load_experiment_config(
        settings.project_root,
        config_path=config_path,
    )
    apply_experiment_settings_overrides(settings, config_payload)
    runtime_config = build_experiment_runtime_config(
        settings=settings,
        config=config_payload,
        config_path=resolved_config_path,
        allow_tiny_override=allow_tiny_override,
        allow_synthetic_override=allow_synthetic_override,
        run_llm_override=run_llm_override,
    )

    environment_report = run_environment_check(
        settings.project_root,
        dataset_path_override=settings.reviews_csv_path,
    )
    if environment_report["summary"]["errors"] > 0:
        manifest = build_experiment_manifest(
            settings=settings,
            runtime_config=runtime_config,
            preflight_report={},
            status="failed",
            llm_requested=bool(runtime_config["run_llm"]),
            llm_ran=False,
            warnings=["Environment check failed."],
        )
        write_json_report(settings.results_dir / "experiment_manifest.json", manifest)
        message = "Environment check reported critical errors. Fix them before running the experiment."
        logger.warning("%s", message)
        print(message)
        return manifest

    active_dataset_mode = get_active_dataset_mode(settings)
    llm_requested = bool(runtime_config["run_llm"])
    if active_dataset_mode == "balanced_subset" and not bool(runtime_config["allow_balanced_subset"]):
        message = (
            "Balanced subset experiments are disabled by config. Set allow_balanced_subset=true "
            "or point ACTIVE_PROCESSED_REVIEWS at a full processed dataset."
        )
        logger.warning("%s", message)
        print(message)
        manifest = build_experiment_manifest(
            settings=settings,
            runtime_config=runtime_config,
            preflight_report={},
            status="failed",
            llm_requested=llm_requested,
            llm_ran=False,
            warnings=[message],
        )
        write_json_report(settings.results_dir / "experiment_manifest.json", manifest)
        return manifest

    def finalize_experiment(
        *,
        preflight_report: dict[str, object],
        warnings: list[str],
        llm_ran: bool,
        mock_llm_ran: bool = False,
        llm_mode: str = "real",
    ) -> dict[str, object]:
        manifest_status = "completed_with_warnings" if warnings else "completed"
        manifest = build_experiment_manifest(
            settings=settings,
            runtime_config=runtime_config,
            preflight_report=preflight_report,
            status=manifest_status,
            llm_requested=llm_requested,
            llm_ran=llm_ran,
            mock_llm_ran=mock_llm_ran,
            llm_mode=llm_mode,
            warnings=warnings,
        )
        manifest_path = settings.results_dir / "experiment_manifest.json"
        write_json_report(manifest_path, manifest)

        experiment_dir = create_experiment_output_folder(
            output_root=Path(runtime_config["output_root"]),
            experiment_name=str(runtime_config["experiment_name"]),
        )
        config_used_payload = dict(config_payload)
        config_used_payload.update(
            {
                "resolved_dataset_path": str(settings.reviews_csv_path),
                "resolved_active_processed_reviews": str(settings.active_processed_reviews_path),
                "resolved_scenarios_file": str(settings.scenarios_file) if settings.scenarios_file else "",
                "run_llm_effective": llm_requested,
                "llm_mode_effective": str(runtime_config.get("llm_mode", "real")),
                "allow_tiny_dataset_effective": bool(runtime_config["allow_tiny_dataset"]),
                "allow_synthetic_scenarios_effective": bool(runtime_config["allow_synthetic_scenarios"]),
                "allow_balanced_subset_effective": bool(runtime_config["allow_balanced_subset"]),
                "allow_llm_skip_effective": bool(runtime_config["allow_llm_skip"]),
            }
        )
        write_json_report(experiment_dir / "config_used.json", config_used_payload)
        copy_experiment_artifacts(settings, manifest_path, experiment_dir, manifest)
        (experiment_dir / "experiment_manifest.json").write_text(
            manifest_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (experiment_dir / "README_EXPERIMENT.md").write_text(
            build_experiment_readme(
                settings=settings,
                runtime_config=runtime_config,
                manifest=manifest,
                experiment_dir=experiment_dir,
            ),
            encoding="utf-8",
        )
        logger.info("Saved experiment outputs to %s", experiment_dir)
        return manifest

    user_based_workflow = active_dataset_mode == "balanced_subset" or str(runtime_config["experiment_mode"]) == "user_based"

    if user_based_workflow:
        cleaned_reviews_df = load_clean_reviews_for_user_mode(settings)
        if cleaned_reviews_df is None:
            message = (
                "User-based balanced-subset workflow is unavailable because the active processed dataset "
                "could not be loaded."
            )
            logger.warning("%s", message)
            print(message)
            manifest = build_experiment_manifest(
                settings=settings,
                runtime_config=runtime_config,
                preflight_report={},
                status="failed",
                llm_requested=llm_requested,
                llm_ran=False,
                warnings=[message],
            )
            write_json_report(settings.results_dir / "experiment_manifest.json", manifest)
            return manifest

        game_cards = build_game_cards(cleaned_reviews_df, settings)
        build_user_profiles(settings)
        if bool(runtime_config.get("use_pilot_splits", False)):
            build_user_splits_pilot(settings)
        else:
            build_user_evaluation_splits(settings)
        run_user_baseline(settings)
        evaluate_user_baseline(settings)
        preflight_report = build_preflight_report(settings)

        user_llm_llm_requested = bool(runtime_config["run_llm"])
        user_llm_mode = str(runtime_config.get("llm_mode", "real")).strip().lower() or "real"
        user_llm_llm_ran = False
        user_llm_mock_ran = False
        user_llm_summary: dict[str, object] = {}
        if user_llm_llm_requested:
            llm_credentials_configured = provider_credentials_configured(settings)
            if user_llm_mode != "mock" and not llm_credentials_configured and not bool(runtime_config["allow_llm_skip"]):
                message = (
                    f"LLM reranking was requested but credentials for provider '{get_effective_llm_provider(settings)}' are missing. "
                    "Set allow_llm_skip=true to continue baseline-only, or configure credentials."
                )
                logger.warning("%s", message)
                print(message)
                manifest = build_experiment_manifest(
                    settings=settings,
                    runtime_config=runtime_config,
                    preflight_report=preflight_report,
                    status="failed",
                    llm_requested=True,
                    llm_ran=False,
                    warnings=[message],
                )
                write_json_report(settings.results_dir / "experiment_manifest.json", manifest)
                return manifest

            run_user_llm_pilot(settings)
            user_llm_summary = load_json_if_exists(settings.user_llm_reranking_summary_path)
            user_llm_mock_ran = bool(user_llm_summary.get("mock_llm_ran", False))
            user_llm_llm_ran = bool(user_llm_summary.get("llm_ran", False))
            llm_output_rows = read_jsonl(settings.user_llm_results_path) if settings.user_llm_results_path.exists() else []
            has_non_skipped_llm_rows = any(
                str(row.get("status", "")) != "skipped_no_credentials"
                for row in llm_output_rows
            )
            if user_llm_mode == "mock" or has_non_skipped_llm_rows:
                evaluate_user_llm(settings)
                if user_llm_mock_ran:
                    save_user_llm_mock_validation_report(
                        settings,
                        user_llm_summary or load_json_if_exists(settings.user_llm_reranking_summary_path),
                        load_json_if_exists(settings.user_llm_validation_summary_path),
                        read_jsonl(settings.user_llm_results_path),
                    )

        run_data_diagnostics(settings)
        export_available_games(settings)
        generate_thesis_tables(settings)
        build_experiment_readiness_report(settings)
        experiment_preflight_report = dict(preflight_report)
        experiment_preflight_report["scenario_mode"] = "not_used"
        if user_llm_llm_requested:
            experiment_preflight_report["llm_requested"] = True
            experiment_preflight_report["llm_ran"] = user_llm_llm_ran
            experiment_preflight_report["llm_mode"] = user_llm_mode
            experiment_preflight_report["mock_llm_ran"] = user_llm_mock_ran

        warnings = list(preflight_report.get("warnings", []))
        if llm_requested and user_llm_mode != "mock" and not user_llm_llm_ran:
            warnings.append(
                "LLM reranking was requested, but no ranked user-LLM outputs were produced. The pilot remained baseline-only."
            )
        if user_llm_mock_ran:
            warnings.append(
                "LLM reranking ran in mock validation mode. Mock metrics are not scientific results."
            )
        if not user_llm_llm_requested:
            write_jsonl(settings.user_llm_results_path, [])

        warnings = deduplicate_preserve_order(warnings)
        return finalize_experiment(
            preflight_report=experiment_preflight_report,
            warnings=warnings,
            llm_ran=user_llm_llm_ran,
            mock_llm_ran=user_llm_mock_ran,
            llm_mode=user_llm_mode,
        )

    reviews_df = load_reviews_csv(settings)
    cleaned_reviews_df = preprocess_reviews(reviews_df, settings)
    game_cards = build_game_cards(cleaned_reviews_df, settings)
    scenarios = build_scenarios(game_cards, settings)
    validate_scenarios_from_artifacts(settings)
    build_experiment_readiness_report(settings)
    preflight_report = build_preflight_report(settings)

    blocking_messages: list[str] = []
    if preflight_report["blocking_issues"]:
        blocking_messages.append(
            "Preflight reported blocking issues: " + ", ".join(preflight_report["blocking_issues"])
        )
    if not runtime_config["allow_tiny_dataset"] and preflight_report["is_tiny_dataset"]:
        blocking_messages.append(
            "Dataset is too small for a thesis-scale experiment. Use --step smoke_test for technical validation or set allow_tiny_dataset=true in config for debugging."
        )
    if not runtime_config["allow_synthetic_scenarios"] and preflight_report["uses_synthetic_scenarios"]:
        blocking_messages.append(
            "Synthetic scenarios are not allowed for final experiment. Provide manually reviewed scenarios via SCENARIOS_FILE or experiment config."
        )
    if (
        runtime_config["run_llm"]
        and not preflight_report["llm_credentials_configured"]
        and not runtime_config["allow_llm_skip"]
    ):
        blocking_messages.append(
            f"LLM reranking was requested but credentials for provider '{get_effective_llm_provider(settings)}' are missing."
        )

    if blocking_messages:
        for message in blocking_messages:
            logger.warning("%s", message)
            print(message)
        manifest = build_experiment_manifest(
            settings=settings,
            runtime_config=runtime_config,
            preflight_report=preflight_report,
            status="failed",
            llm_requested=bool(runtime_config["run_llm"]),
            llm_ran=False,
            warnings=blocking_messages,
        )
        write_json_report(settings.results_dir / "experiment_manifest.json", manifest)
        return manifest

    baseline_results = run_baseline(scenarios, game_cards, settings)
    llm_results = []
    llm_warnings: list[str] = []
    llm_requested = bool(runtime_config["run_llm"])
    llm_ran = False

    if llm_requested:
        llm_results = run_llm_reranker(scenarios, baseline_results, game_cards, settings)
        llm_ran = any(record.rank is not None and record.game_id for record in llm_results)
        if not llm_ran and not preflight_report["llm_credentials_configured"]:
            llm_warnings.append(
                "LLM reranking was requested but credentials were missing, so the run continued baseline-only."
            )
    else:
        write_jsonl(settings.llm_results_path, [])

    evaluate_recommendations(
        scenarios=scenarios,
        recommendation_sets=[baseline_results, llm_results],
        settings=settings,
        reviews_clean_count=len(cleaned_reviews_df),
        game_cards=game_cards,
    )
    run_analysis_suite(settings)

    warnings = [
        warning
        for warning in preflight_report.get("warnings", [])
        if warning != "llm_credentials_missing" or llm_requested
    ] + llm_warnings
    manifest = finalize_experiment(
        preflight_report=preflight_report,
        warnings=warnings,
        llm_ran=llm_ran,
    )
    return manifest


def resolve_local_path(project_root: Path, raw_path: str) -> Path:
    """Resolve a potentially relative path against the project root."""

    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def create_experiment_output_folder(output_root: Path, experiment_name: str) -> Path:
    """Create a timestamped experiment output folder."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", experiment_name).strip("_") or "experiment"
    experiment_dir = output_root / f"{timestamp}_{slug}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def copy_experiment_artifacts(
    settings: Settings,
    manifest_path: Path,
    experiment_dir: Path,
    manifest: dict[str, object],
) -> None:
    """Copy selected artifacts into the versioned experiment folder."""

    balanced_subset_experiment = settings.active_processed_reviews_path == settings.reviews_clean_balanced_subset_path
    include_user_llm_artifacts = bool(manifest.get("llm_requested", False))
    scenario_only_targets = {
        "scenario_validation_report.csv",
        "metrics_summary.csv",
        "per_scenario_results.csv",
        "rank_comparison.csv",
        "case_studies.json",
        "llm_explanation_checks.csv",
        "experiment_summary.md",
        "thesis_scenario_table.md",
    }
    copy_specs = [
        (settings.environment_check_json_path, "environment_check.json"),
        (settings.preflight_report_json_path, "preflight_report.json"),
        (settings.preprocessing_summary_path, "preprocessing_summary.json"),
        (settings.preprocessing_summary_markdown_path, "preprocessing_summary.md"),
        (settings.preprocessing_subset_summary_path, "preprocessing_subset_summary.json"),
        (settings.preprocessing_subset_summary_markdown_path, "preprocessing_subset_summary.md"),
        (
            settings.preprocessing_balanced_subset_summary_path,
            "preprocessing_balanced_subset_summary.json",
        ),
        (
            settings.preprocessing_balanced_subset_summary_markdown_path,
            "preprocessing_balanced_subset_summary.md",
        ),
        (settings.preprocessing_progress_json_path, "preprocessing_progress.json"),
        (settings.preprocessing_progress_markdown_path, "preprocessing_progress.md"),
        (settings.raw_processed_comparison_json_path, "raw_processed_comparison.json"),
        (settings.raw_processed_comparison_markdown_path, "raw_processed_comparison.md"),
        (settings.raw_processed_subset_comparison_json_path, "raw_processed_subset_comparison.json"),
        (settings.raw_processed_subset_comparison_markdown_path, "raw_processed_subset_comparison.md"),
        (
            settings.raw_processed_balanced_subset_comparison_json_path,
            "raw_processed_balanced_subset_comparison.json",
        ),
        (
            settings.raw_processed_balanced_subset_comparison_markdown_path,
            "raw_processed_balanced_subset_comparison.md",
        ),
        (settings.raw_game_sampling_stats_csv_path, "raw_game_sampling_stats.csv"),
        (settings.raw_game_sampling_stats_json_path, "raw_game_sampling_stats.json"),
        (settings.game_card_summary_path, "game_card_summary.json"),
        (settings.user_profiles_path, "user_profiles.jsonl"),
        (settings.user_profile_summary_path, "user_profile_summary.json"),
        (settings.user_splits_path, "user_evaluation_splits.jsonl"),
        (settings.pilot_splits_path, "user_evaluation_splits_pilot.jsonl"),
        (settings.user_split_summary_path, "user_evaluation_split_summary.json"),
        (settings.user_split_diagnostics_json_path, "user_split_diagnostics.json"),
        (settings.user_split_diagnostics_markdown_path, "user_split_diagnostics.md"),
        (
            settings.user_evaluation_split_pilot_summary_path,
            "user_evaluation_split_pilot_summary.json",
        ),
        (
            settings.user_evaluation_split_pilot_summary_markdown_path,
            "user_evaluation_split_pilot_summary.md",
        ),
        (settings.user_baseline_results_path, "user_baseline_recommendations.jsonl"),
        (settings.user_per_profile_results_path, "user_per_profile_results.csv"),
        (settings.user_metrics_summary_path, "user_metrics_summary.csv"),
        (settings.scenario_validation_report_path, "scenario_validation_report.csv"),
        (settings.metrics_summary_path, "metrics_summary.csv"),
        (settings.per_scenario_results_path, "per_scenario_results.csv"),
        (settings.rank_comparison_csv_path, "rank_comparison.csv"),
        (settings.case_studies_json_path, "case_studies.json"),
        (settings.llm_explanation_checks_csv_path, "llm_explanation_checks.csv"),
        (settings.experiment_summary_path, "experiment_summary.md"),
        (settings.thesis_metrics_table_path, "thesis_metrics_table.md"),
        (settings.reports_dir / "user_thesis_metrics_table.md", "user_thesis_metrics_table.md"),
        (settings.thesis_dataset_table_path, "thesis_dataset_table.md"),
        (
            settings.reports_dir / "thesis_balanced_subset_dataset_table.md",
            "thesis_balanced_subset_dataset_table.md",
        ),
        (settings.reports_dir / "balanced_subset_methodology_note.md", "balanced_subset_methodology_note.md"),
        (settings.thesis_scenario_table_path, "thesis_scenario_table.md"),
        (manifest_path, "experiment_manifest.json"),
    ]
    user_llm_specs = [
        (settings.user_llm_prompt_preview_markdown_path, "user_llm_prompt_preview.md"),
        (settings.user_llm_prompt_preview_json_path, "user_llm_prompt_preview.json"),
        (settings.user_llm_prompt_preview_tiny_markdown_path, "user_llm_prompt_preview_tiny.md"),
        (settings.user_llm_prompt_preview_tiny_json_path, "user_llm_prompt_preview_tiny.json"),
        (settings.user_llm_results_path, "user_llm_recommendations.jsonl"),
        (settings.user_llm_reranking_summary_path, "user_llm_reranking_summary.json"),
        (settings.user_llm_validation_summary_path, "user_llm_validation_summary.json"),
        (settings.user_llm_mock_validation_summary_path, "user_llm_mock_validation_summary.json"),
        (settings.user_llm_per_profile_results_path, "user_llm_per_profile_results.csv"),
        (settings.user_llm_metrics_summary_path, "user_llm_metrics_summary.csv"),
        (settings.user_rank_comparison_path, "user_rank_comparison.csv"),
        (settings.user_rank_comparison_markdown_path, "user_rank_comparison.md"),
        (settings.user_llm_explanation_checks_csv_path, "user_llm_explanation_checks.csv"),
        (settings.user_llm_explanation_checks_markdown_path, "user_llm_explanation_checks.md"),
        (settings.user_llm_explanation_examples_markdown_path, "user_llm_explanation_examples.md"),
        (settings.llm_pilot_candidate_user_report_csv_path, "llm_pilot_candidate_user_report.csv"),
        (settings.llm_pilot_candidate_user_report_markdown_path, "llm_pilot_candidate_user_report.md"),
        (settings.user_llm_metrics_table_path, "user_llm_metrics_table.md"),
        (settings.user_llm_pilot_summary_path, "user_llm_pilot_summary.md"),
        (settings.user_llm_metrics_summary_all_pilot_path, "user_llm_metrics_summary_all_pilot.csv"),
        (settings.user_llm_schema_error_report_json_path, "user_llm_schema_error_report.json"),
        (settings.user_llm_schema_error_report_markdown_path, "user_llm_schema_error_report.md"),
        (settings.user_llm_mock_validation_report_path, "user_llm_mock_validation_report.md"),
        (settings.user_llm_failure_report_json_path, "user_llm_failure_report.json"),
        (settings.user_llm_failure_report_markdown_path, "user_llm_failure_report.md"),
        (settings.llm_check_json_path, "llm_check.json"),
        (settings.llm_check_markdown_path, "llm_check.md"),
        (settings.llm_pilot_readiness_json_path, "llm_pilot_readiness.json"),
        (settings.llm_pilot_readiness_markdown_path, "llm_pilot_readiness.md"),
    ]
    if include_user_llm_artifacts:
        copy_specs.extend(user_llm_specs)
    for source_path, target_name in copy_specs:
        if balanced_subset_experiment and target_name in scenario_only_targets:
            continue
        if source_path.exists():
            shutil.copy2(source_path, experiment_dir / target_name)


def build_experiment_manifest(
    settings: Settings,
    runtime_config: dict[str, object],
    preflight_report: dict[str, object],
    status: str,
    llm_requested: bool,
    llm_ran: bool,
    warnings: list[str],
    mock_llm_ran: bool = False,
    llm_mode: str = "real",
) -> dict[str, object]:
    """Build the machine-readable manifest for a controlled experiment run."""

    preprocessing_summary = load_json_if_exists(get_active_preprocessing_summary_path(settings))
    game_card_summary = load_json_if_exists(settings.game_card_summary_path)
    readiness_report = load_json_if_exists(settings.experiment_readiness_path)
    user_llm_summary = load_json_if_exists(settings.user_llm_reranking_summary_path)
    evaluated_scenarios = 0
    if status != "failed" and settings.metrics_summary_path.exists():
        metrics_df = pd.read_csv(settings.metrics_summary_path)
        if not metrics_df.empty and "evaluated_scenarios" in metrics_df:
            evaluated_scenarios = int(metrics_df["evaluated_scenarios"].max())

    active_dataset_mode = str(
        preflight_report.get("active_dataset_mode", get_active_dataset_mode(settings))
    )
    llm_mode = str(runtime_config.get("llm_mode", user_llm_summary.get("llm_mode", "real"))).strip().lower() or "real"
    llm_provider = str(runtime_config.get("llm_provider", getattr(settings, "llm_provider", "openai"))).strip().lower() or "openai"
    llm_response_language = str(runtime_config.get("llm_response_language", getattr(settings, "llm_response_language", "ru"))).strip().lower() or "ru"
    full_dataset_preprocessing_completed = cleaned_reviews_artifact_is_valid(settings.reviews_clean_path)
    balanced_subset_used = active_dataset_mode == "balanced_subset" or (
        settings.active_processed_reviews_path == settings.reviews_clean_balanced_subset_path
    )
    is_user_based_experiment = str(runtime_config.get("experiment_mode", "auto")).strip().lower() == "user_based"
    if is_user_based_experiment:
        evaluated_scenarios = 0
    metrics_summary_path = settings.metrics_summary_path
    evaluated_users = 0
    evaluated_llm_users = 0
    all_pilot_baseline_evaluated_users = 0
    mock_llm_ran = False
    real_api_calls = 0
    token_requests_attempted = 0
    completion_requests_attempted = 0
    provider_preflight_ok = True
    provider_preflight_status = ""
    not_for_scientific_metrics = False
    if settings.user_metrics_summary_path.exists() and (
        is_user_based_experiment or active_dataset_mode in {"balanced_subset", "full"}
    ):
        try:
            user_metrics_df = pd.read_csv(settings.user_metrics_summary_path)
            if not user_metrics_df.empty and "evaluated_profiles" in user_metrics_df:
                all_pilot_baseline_evaluated_users = int(user_metrics_df["evaluated_profiles"].max())
                evaluated_users = all_pilot_baseline_evaluated_users
            metrics_summary_path = settings.user_metrics_summary_path
        except Exception:
            evaluated_users = 0
    if settings.user_llm_metrics_summary_path.exists() and bool(runtime_config.get("run_llm", False)):
        metrics_summary_path = settings.user_llm_metrics_summary_path
        try:
            llm_metrics_df = pd.read_csv(settings.user_llm_metrics_summary_path)
            if not llm_metrics_df.empty and "evaluated_profiles" in llm_metrics_df:
                evaluated_llm_users = int(llm_metrics_df["evaluated_profiles"].max())
                evaluated_users = int(
                    load_json_if_exists(settings.user_llm_reranking_summary_path).get(
                        "baseline_evaluated_users",
                        evaluated_users,
                    )
                ) or evaluated_users
        except Exception:
            evaluated_llm_users = 0
    if evaluated_llm_users == 0 and user_llm_summary and bool(runtime_config.get("run_llm", False)):
        evaluated_llm_users = int(user_llm_summary.get("evaluated_llm_users", 0))
    if is_user_based_experiment and user_llm_summary:
        evaluated_users = int(user_llm_summary.get("baseline_evaluated_users", evaluated_users))
    mock_llm_ran = bool(user_llm_summary.get("mock_llm_ran", False))
    real_api_calls = int(user_llm_summary.get("real_api_calls_total", user_llm_summary.get("real_api_calls", 0)))
    token_requests_attempted = int(user_llm_summary.get("token_requests_attempted", 0))
    completion_requests_attempted = int(user_llm_summary.get("completion_requests_attempted", 0))
    provider_preflight_ok = bool(user_llm_summary.get("provider_preflight_ok", True))
    provider_preflight_status = str(user_llm_summary.get("provider_preflight_status", ""))
    not_for_scientific_metrics = bool(user_llm_summary.get("not_for_scientific_metrics", False))
    selected_llm_users = int(user_llm_summary.get("selected_llm_users", user_llm_summary.get("selected_users", 0)))
    failed_llm_users = int(user_llm_summary.get("failed_llm_users", user_llm_summary.get("failed_users", 0)))
    selected_meaningful_users = int(user_llm_summary.get("selected_meaningful_users", 0))
    holdout_in_candidate_pool_users = int(user_llm_summary.get("holdout_in_candidate_pool_users", 0))
    holdout_in_baseline_top_k_users = int(user_llm_summary.get("holdout_in_baseline_top_k_users", 0))
    holdout_in_llm_top_k_users = int(user_llm_summary.get("holdout_in_llm_top_k_users", 0))
    scenario_mode_value = map_manifest_scenario_mode(str(preflight_report.get("scenario_mode", readiness_report.get("scenario_mode", "unknown"))))
    if is_user_based_experiment:
        scenario_mode_value = "not_used"
        warnings = [warning for warning in warnings if warning != "synthetic_scenarios_only"]

    return {
        "experiment_name": str(runtime_config["experiment_name"]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "project_version_note": "local research prototype",
        "experiment_mode": str(runtime_config.get("experiment_mode", "auto")),
        "use_pilot_splits": bool(runtime_config.get("use_pilot_splits", False)),
        "dataset_path": str(settings.reviews_csv_path),
        "active_processed_reviews": str(settings.active_processed_reviews_path),
        "active_dataset_mode": active_dataset_mode,
        "active_user_splits_path": str(get_active_user_splits_path(settings)),
        "scenarios_file": str(settings.scenarios_file) if settings.scenarios_file else "",
        "scenario_mode": scenario_mode_value,
        "llm_requested": llm_requested,
        "llm_ran": llm_ran,
        "llm_mode": llm_mode,
        "llm_provider": llm_provider,
        "llm_response_language": llm_response_language,
        "mock_llm_ran": mock_llm_ran,
        "real_api_calls": real_api_calls,
        "real_api_calls_total": real_api_calls,
        "token_requests_attempted": token_requests_attempted,
        "completion_requests_attempted": completion_requests_attempted,
        "provider_preflight_ok": provider_preflight_ok,
        "provider_preflight_status": provider_preflight_status,
        "not_for_scientific_metrics": not_for_scientific_metrics,
        "preflight_status": str(preflight_report.get("status", "not_ready")),
        "full_dataset_preprocessing_completed": full_dataset_preprocessing_completed,
        "balanced_subset_used": balanced_subset_used,
        "baseline_evaluated_users": int(evaluated_users),
        "baseline_all_pilot_evaluated_users": int(all_pilot_baseline_evaluated_users),
        "evaluated_users": evaluated_users,
        "evaluated_llm_users": evaluated_llm_users,
        "selected_llm_users": selected_llm_users,
        "selected_meaningful_users": selected_meaningful_users,
        "failed_llm_users": failed_llm_users,
        "holdout_in_candidate_pool_users": holdout_in_candidate_pool_users,
        "holdout_in_baseline_top_k_users": holdout_in_baseline_top_k_users,
        "holdout_in_llm_top_k_users": holdout_in_llm_top_k_users,
        "cleaned_review_count": int(
            preprocessing_summary.get(
                "final_row_count",
                preprocessing_summary.get("processed_rows_written", preprocessing_summary.get("processed_row_count", 0)),
            )
        ),
        "game_card_count": int(game_card_summary.get("generated_game_card_count", 0)),
        "scenario_count": 0 if is_user_based_experiment else int(readiness_report.get("scenario_count", 0)),
        "evaluated_scenarios": 0 if is_user_based_experiment else evaluated_scenarios,
        "metrics_summary_path": str(metrics_summary_path),
        "warnings": warnings,
    }


def map_manifest_scenario_mode(raw_mode: str) -> str:
    """Map internal scenario mode labels to a compact manifest value."""

    if raw_mode == "manual_or_predefined":
        return "manual"
    if raw_mode == "synthetic_demo_only":
        return "synthetic_demo"
    if raw_mode == "mixed":
        return "mixed"
    return "unknown"


def build_experiment_readme(
    settings: Settings,
    runtime_config: dict[str, object],
    manifest: dict[str, object],
    experiment_dir: Path,
) -> str:
    """Create a brief experiment README for the copied output folder."""

    metrics_lines = ["- Metrics summary unavailable."]
    metrics_source_path = settings.metrics_summary_path
    if (
        str(manifest.get("experiment_mode", "")).strip().lower() == "user_based"
        and bool(runtime_config.get("run_llm", False))
        and settings.user_llm_metrics_summary_path.exists()
    ):
        metrics_source_path = settings.user_llm_metrics_summary_path
    elif (
        str(manifest.get("experiment_mode", "")).strip().lower() == "user_based"
        or str(manifest.get("active_dataset_mode", "")).strip() == "balanced_subset"
    ) and settings.user_metrics_summary_path.exists():
        metrics_source_path = settings.user_metrics_summary_path

    if metrics_source_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_source_path)
        except pd.errors.EmptyDataError:
            metrics_df = pd.DataFrame()
        if not metrics_df.empty:
            if "evaluated_profiles" in metrics_df.columns:
                metrics_lines = [
                    f"- {row['method']}: HitRate@5={format_metric(row['mean_hit_rate_at_5'])}, "
                    f"HitRate@10={format_metric(row['mean_hit_rate_at_10'])}, "
                    f"MRR={format_metric(row['mean_mrr'])}, "
                    f"NDCG@10={format_metric(row['mean_ndcg_at_10'])}, "
                    f"Evaluated users={int(row['evaluated_profiles'])}"
                    for _, row in metrics_df.iterrows()
                ]
            else:
                metrics_lines = [
                    f"- {row['method']}: HitRate@5={format_metric(row['mean_hit_rate_at_5'])}, "
                    f"HitRate@10={format_metric(row['mean_hit_rate_at_10'])}, "
                    f"MRR={format_metric(row['mean_mrr'])}, "
                    f"NDCG@10={format_metric(row['mean_ndcg_at_10'])}"
                    for _, row in metrics_df.iterrows()
                ]

    warnings = manifest.get("warnings", [])
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    return "\n".join(
        [
            "# Experiment README",
            "",
            f"- Experiment name: {manifest['experiment_name']}",
            f"- Run timestamp: {manifest['timestamp']}",
            f"- Experiment folder: `{experiment_dir}`",
            f"- Experiment mode: {manifest.get('experiment_mode', 'auto')}",
            f"- Dataset path: `{manifest['dataset_path']}`",
            f"- Active processed reviews: `{manifest.get('active_processed_reviews', '')}`",
            f"- Active dataset mode: `{manifest.get('active_dataset_mode', 'unknown')}`",
            f"- Active user splits path: `{manifest.get('active_user_splits_path', '')}`",
            f"- Use pilot splits: {manifest.get('use_pilot_splits', False)}",
            f"- Scenario file path: `{manifest['scenarios_file']}`",
            f"- LLM enabled: {runtime_config['run_llm']}",
            f"- LLM actually ran: {manifest['llm_ran']}",
            f"- LLM provider: {manifest.get('llm_provider', 'openai')}",
            f"- LLM mode: {manifest.get('llm_mode', 'real')}",
            f"- LLM response language: {manifest.get('llm_response_language', 'ru')}",
            f"- Mock LLM ran: {manifest.get('mock_llm_ran', False)}",
            f"- Real API calls: {manifest.get('real_api_calls', 0)}",
            f"- Not for scientific metrics: {manifest.get('not_for_scientific_metrics', False)}",
            f"- Evaluated LLM users: {manifest.get('evaluated_llm_users', 0)}",
            f"- Full dataset preprocessing completed: {manifest.get('full_dataset_preprocessing_completed', False)}",
            f"- Balanced subset used: {manifest.get('balanced_subset_used', False)}",
            f"- Cleaned reviews: {manifest['cleaned_review_count']}",
            f"- Game cards: {manifest['game_card_count']}",
            f"- Scenarios: {manifest['scenario_count']}",
            "",
            "## Main Metrics",
            *metrics_lines,
            "",
            "## Warnings",
            *warning_lines,
            "",
            "## Notes",
            "- Mock LLM results are validation-only and must not be interpreted as scientific performance."
            if bool(manifest.get("mock_llm_ran", False))
            else "- none",
            "",
        ]
    ) + "\n"


def _relative_project_path(project_root: Path, path: Path) -> str:
    """Return a project-relative string when possible."""

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _safe_int(value: object, default: int = 0) -> int:
    """Best-effort integer conversion."""

    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(float(value))
    except Exception:
        return default


def _metrics_frame_to_dict(metrics_df: pd.DataFrame, method_name: str) -> dict[str, object]:
    """Extract the first metrics row for a given method name."""

    if metrics_df.empty or "method" not in metrics_df.columns:
        return {}
    try:
        match = metrics_df[metrics_df["method"].astype(str) == method_name]
        if match.empty:
            return {}
        return match.iloc[0].to_dict()
    except Exception:
        return {}


def _load_experiment_candidate(settings: Settings, experiment_dir: Path) -> dict[str, object]:
    """Load an archived experiment folder and score whether it is a valid final result."""

    manifest_path = experiment_dir / "experiment_manifest.json"
    reranking_summary_path = experiment_dir / "user_llm_reranking_summary.json"
    metrics_summary_path = experiment_dir / "user_llm_metrics_summary.csv"
    rank_comparison_path = experiment_dir / "user_rank_comparison.csv"
    explanation_examples_path = experiment_dir / "user_llm_explanation_examples.md"
    required_paths = {
        "experiment_manifest": manifest_path,
        "user_llm_reranking_summary": reranking_summary_path,
        "user_llm_metrics_summary": metrics_summary_path,
        "user_rank_comparison": rank_comparison_path,
    }
    missing_files = [name for name, path in required_paths.items() if not path.exists()]

    manifest = load_json_if_exists(manifest_path)
    reranking_summary = load_json_if_exists(reranking_summary_path)
    try:
        metrics_df = pd.read_csv(metrics_summary_path) if metrics_summary_path.exists() else pd.DataFrame()
    except Exception:
        metrics_df = pd.DataFrame()
    try:
        rank_df = pd.read_csv(rank_comparison_path) if rank_comparison_path.exists() else pd.DataFrame()
    except Exception:
        rank_df = pd.DataFrame()

    baseline_row: dict[str, object] = {}
    llm_row: dict[str, object] = {}
    if not metrics_df.empty and "method" in metrics_df.columns:
        baseline_match = metrics_df[metrics_df["method"].astype(str) == "user_tfidf_baseline"]
        llm_match = metrics_df[metrics_df["method"].astype(str) == "user_llm_reranker"]
        if not baseline_match.empty:
            baseline_row = baseline_match.iloc[0].to_dict()
        if not llm_match.empty:
            llm_row = llm_match.iloc[0].to_dict()

    llm_records_path = experiment_dir / "user_llm_recommendations.jsonl"
    total_records = len(read_jsonl(llm_records_path)) if llm_records_path.exists() else 0
    valid_llm_records = _safe_int(reranking_summary.get("llm_valid_records", 0))
    fallback_records = _safe_int(reranking_summary.get("fallback_record_count", 0))
    candidate_pool_ok = bool(reranking_summary.get("all_game_ids_inside_candidate_pool", False))
    invalid_game_ids = _safe_int(reranking_summary.get("invalid_game_id_count", 0))
    llm_evaluated_users = _safe_int(reranking_summary.get("llm_evaluated_users", 0))
    provider_preflight_ok = bool(reranking_summary.get("provider_preflight_ok", False))
    llm_ran = bool(reranking_summary.get("llm_ran", False))
    llm_provider = str(reranking_summary.get("llm_provider", manifest.get("llm_provider", "")) or "").strip().lower()
    llm_mode = str(reranking_summary.get("llm_mode", manifest.get("llm_mode", "")) or "").strip().lower()
    completion_requests = _safe_int(reranking_summary.get("completion_requests_attempted", 0))
    token_requests = _safe_int(reranking_summary.get("token_requests_attempted", 0))
    real_api_calls = _safe_int(reranking_summary.get("real_api_calls", reranking_summary.get("real_api_calls_total", 0)))
    metrics_llm_profiles = _safe_int(llm_row.get("evaluated_profiles", 0))
    metrics_baseline_profiles = _safe_int(baseline_row.get("evaluated_profiles", 0))
    fallback_only = total_records > 0 and fallback_records >= total_records and valid_llm_records == 0

    reasons: list[str] = []
    if missing_files:
        reasons.append(f"missing_files:{','.join(missing_files)}")
    if str(manifest.get("experiment_mode", "")).strip().lower() != "user_based":
        reasons.append("not_user_based")
    if llm_provider != "gigachat":
        reasons.append(f"provider:{llm_provider or 'unknown'}")
    if llm_mode != "real":
        reasons.append(f"llm_mode:{llm_mode or 'unknown'}")
    if not llm_ran:
        reasons.append("llm_ran_false")
    if completion_requests <= 0:
        reasons.append("no_completion_requests")
    if real_api_calls <= 0:
        reasons.append("no_real_api_calls")
    if not provider_preflight_ok:
        reasons.append(f"provider_preflight:{reranking_summary.get('provider_preflight_status', 'unknown')}")
    if invalid_game_ids > 0:
        reasons.append("invalid_game_ids")
    if not candidate_pool_ok:
        reasons.append("candidate_pool_violation")
    if llm_evaluated_users <= 0:
        reasons.append("no_llm_evaluated_users")
    if metrics_llm_profiles <= 0:
        reasons.append("no_llm_metrics_row")
    if metrics_baseline_profiles <= 0:
        reasons.append("no_baseline_metrics_row")
    if fallback_only:
        reasons.append("fallback_only_output")
    if not rank_df.empty and "holdout_in_candidate_pool" in rank_df.columns:
        if not rank_df["holdout_in_candidate_pool"].fillna(False).astype(bool).any():
            reasons.append("no_holdout_in_candidate_pool_hits")
    if not rank_df.empty and "holdout_in_llm_top_k" in rank_df.columns:
        if not rank_df["holdout_in_llm_top_k"].fillna(False).astype(bool).any():
            reasons.append("no_llm_topk_hits")

    timestamp = str(manifest.get("timestamp", "") or "")
    try:
        timestamp_dt = datetime.fromisoformat(timestamp) if timestamp else datetime.fromtimestamp(experiment_dir.stat().st_mtime)
    except Exception:
        timestamp_dt = datetime.fromtimestamp(experiment_dir.stat().st_mtime)

    return {
        "experiment_dir": experiment_dir,
        "experiment_dir_relative": _relative_project_path(settings.project_root, experiment_dir),
        "experiment_name": str(manifest.get("experiment_name", experiment_dir.name)),
        "timestamp": timestamp_dt,
        "manifest_timestamp": timestamp,
        "active_processed_reviews": str(manifest.get("active_processed_reviews", "")),
        "llm_response_language": str(manifest.get("llm_response_language", "")),
        "llm_provider": llm_provider,
        "llm_mode": llm_mode,
        "llm_ran": llm_ran,
        "provider_preflight_ok": provider_preflight_ok,
        "completion_requests_attempted": completion_requests,
        "token_requests_attempted": token_requests,
        "real_api_calls": real_api_calls,
        "invalid_game_id_count": invalid_game_ids,
        "all_game_ids_inside_candidate_pool": candidate_pool_ok,
        "llm_evaluated_users": llm_evaluated_users,
        "llm_valid_records": valid_llm_records,
        "fallback_record_count": fallback_records,
        "total_records": total_records,
        "metrics_llm_profiles": metrics_llm_profiles,
        "metrics_baseline_profiles": metrics_baseline_profiles,
        "has_explanation_examples": explanation_examples_path.exists(),
        "missing_files": missing_files,
        "reasons": reasons,
        "valid": len(reasons) == 0,
    }


def select_final_experiment(settings: Settings) -> dict[str, object]:
    """Select the best archived experiment run for thesis export."""

    experiments_dir = settings.project_root / "experiments"
    selection_path = settings.results_dir / "final_experiment_selection.json"
    report_path = settings.reports_dir / "final_experiment_selection.md"
    candidate_dirs = (
        sorted(
            [path for path in experiments_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if experiments_dir.exists()
        else []
    )
    candidates = [_load_experiment_candidate(settings, experiment_dir) for experiment_dir in candidate_dirs]
    expected_provider = str(getattr(settings, "llm_provider", "gigachat")).strip().lower() or "gigachat"
    expected_mode = str(getattr(settings, "llm_mode", "real")).strip().lower() or "real"
    expected_active_processed_reviews = str(settings.active_processed_reviews_path)
    expected_response_language = str(getattr(settings, "llm_response_language", "ru")).strip().lower() or "ru"
    filtered_candidates = [
        candidate
        for candidate in candidates
        if candidate["llm_provider"] == expected_provider and candidate["llm_mode"] == expected_mode
        and str(candidate.get("active_processed_reviews", "")) == expected_active_processed_reviews
        and str(candidate.get("llm_response_language", "")).strip().lower() == expected_response_language
    ]
    valid_candidates = [candidate for candidate in filtered_candidates if candidate["valid"]]
    strict_selected = max(
        valid_candidates,
        key=lambda candidate: (
            int(candidate.get("selected_llm_users", candidate.get("llm_evaluated_users", 0)) or 0),
            int(candidate.get("llm_evaluated_users", 0) or 0),
            int(candidate.get("completion_requests_attempted", 0) or 0),
            candidate["timestamp"],
        ),
        default=None,
    )
    inferred_candidates = [
        candidate
        for candidate in filtered_candidates
        if not candidate["valid"]
        and candidate["llm_ran"]
        and candidate["provider_preflight_ok"]
        and candidate["completion_requests_attempted"] > 0
        and candidate["real_api_calls"] > 0
        and candidate["invalid_game_id_count"] == 0
        and candidate["llm_evaluated_users"] > 0
        and candidate["metrics_llm_profiles"] > 0
        and candidate["metrics_baseline_profiles"] > 0
        and candidate["fallback_record_count"] < candidate["total_records"]
    ]
    selected = strict_selected or max(
        inferred_candidates,
        key=lambda candidate: (
        candidate["llm_evaluated_users"],
        candidate["llm_valid_records"],
        candidate["metrics_llm_profiles"],
        candidate["completion_requests_attempted"],
        candidate["real_api_calls"],
        candidate["timestamp"],
    ),
        default=None,
    )

    payload = {
        "status": "selected" if strict_selected else ("selected_with_warnings" if selected else "not_found"),
        "selected_experiment_dir": _relative_project_path(settings.project_root, selected["experiment_dir"]) if selected else "",
        "selected_experiment_name": selected["experiment_name"] if selected else "",
        "selected_timestamp": selected["manifest_timestamp"] if selected else "",
        "selection_reason": (
            "Newest valid archived pilot matching the current provider and model."
            if strict_selected
            else "Newest archived pilot with useful LLM outputs, selected with warnings because archived candidate-pool diagnostics were inconsistent."
            if selected
            else "No archived experiment matched the selection criteria."
        ),
        "selection_criteria": {
            "experiment_mode": "user_based",
            "llm_provider": expected_provider,
            "llm_mode": expected_mode,
            "active_processed_reviews": expected_active_processed_reviews,
            "llm_response_language": expected_response_language,
            "llm_ran": True,
            "completion_requests_attempted_min": 1,
            "invalid_game_id_count_max": 0,
            "all_game_ids_inside_candidate_pool": True,
            "llm_metrics_row_required": True,
            "fallback_only_output": False,
        },
        "candidate_count": len(filtered_candidates),
        "valid_candidate_count": len(valid_candidates),
        "inferred_candidate_count": len(inferred_candidates),
        "selected_candidate_index": (
            next(
                (
                    index
                    for index, candidate in enumerate(filtered_candidates)
                    if selected and candidate["experiment_dir_relative"] == selected["experiment_dir_relative"]
                ),
                -1,
            )
            if selected
            else -1
        ),
        "selected_candidate": (
            {
                key: selected[key]
                for key in [
                    "experiment_dir_relative",
                    "experiment_name",
                    "manifest_timestamp",
                    "llm_provider",
                    "llm_mode",
                    "llm_ran",
                    "provider_preflight_ok",
                    "completion_requests_attempted",
                    "token_requests_attempted",
                    "real_api_calls",
                    "invalid_game_id_count",
                    "all_game_ids_inside_candidate_pool",
                    "llm_evaluated_users",
                    "llm_valid_records",
                    "fallback_record_count",
                    "total_records",
                    "metrics_llm_profiles",
                    "metrics_baseline_profiles",
                    "has_explanation_examples",
                ]
            }
            if selected
            else {}
        ),
        "candidates": [
            {
                "experiment_dir": candidate["experiment_dir_relative"],
                "timestamp": candidate["manifest_timestamp"],
                "llm_provider": candidate["llm_provider"],
                "llm_mode": candidate["llm_mode"],
                "llm_ran": candidate["llm_ran"],
                "provider_preflight_ok": candidate["provider_preflight_ok"],
                "completion_requests_attempted": candidate["completion_requests_attempted"],
                "token_requests_attempted": candidate["token_requests_attempted"],
                "real_api_calls": candidate["real_api_calls"],
                "invalid_game_id_count": candidate["invalid_game_id_count"],
                "all_game_ids_inside_candidate_pool": candidate["all_game_ids_inside_candidate_pool"],
                "llm_evaluated_users": candidate["llm_evaluated_users"],
                "llm_valid_records": candidate["llm_valid_records"],
                "fallback_record_count": candidate["fallback_record_count"],
                "total_records": candidate["total_records"],
                "metrics_llm_profiles": candidate["metrics_llm_profiles"],
                "metrics_baseline_profiles": candidate["metrics_baseline_profiles"],
                "has_explanation_examples": candidate["has_explanation_examples"],
                "valid": candidate["valid"],
                "reasons": candidate["reasons"],
            }
            for candidate in filtered_candidates
        ],
        "warnings": [
            "No valid archived experiment matched the selection criteria."
            if not selected
            else "Selected a controlled pilot run from the experiments archive.",
        ],
    }
    write_json_report(selection_path, payload)

    lines = [
        "# Final Experiment Selection",
        "",
        f"- Status: {payload['status']}",
        f"- Selected experiment: `{payload['selected_experiment_dir'] or 'none'}`",
        f"- Selected experiment name: `{payload['selected_experiment_name'] or 'none'}`",
        f"- Selected timestamp: `{payload['selected_timestamp'] or 'none'}`",
        f"- Selection reason: {payload['selection_reason']}",
        f"- Candidate count: {payload['candidate_count']}",
        f"- Valid candidate count: {payload['valid_candidate_count']}",
        "",
        "## Selection Criteria",
        dataframe_to_markdown(pd.DataFrame([payload["selection_criteria"]])),
        "",
        "## Candidate Summary",
    ]
    if filtered_candidates:
        candidate_frame = pd.DataFrame(filtered_candidates)
        candidate_frame = candidate_frame[
            [
                "experiment_dir_relative",
                "manifest_timestamp",
                "valid",
                "llm_provider",
                "llm_mode",
                "llm_ran",
                "provider_preflight_ok",
                "completion_requests_attempted",
                "token_requests_attempted",
                "real_api_calls",
                "invalid_game_id_count",
                "all_game_ids_inside_candidate_pool",
                "llm_evaluated_users",
                "llm_valid_records",
                "fallback_record_count",
                "total_records",
                "metrics_llm_profiles",
                "metrics_baseline_profiles",
                "has_explanation_examples",
                "reasons",
            ]
        ]
        lines.append(dataframe_to_markdown(candidate_frame.head(20)))
    else:
        lines.append("_No matching candidates found._")
    if selected:
        lines.extend(
            [
                "",
                "## Selected Candidate Details",
                dataframe_to_markdown(pd.DataFrame([payload["selected_candidate"]])),
            ]
        )
    lines.extend(
        [
            "",
            "## Warnings",
            *[f"- {warning}" for warning in payload["warnings"]],
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    get_logger().info(
        "Saved final experiment selection to %s and %s",
        selection_path,
        report_path,
    )
    return payload


def _load_final_experiment_selection(settings: Settings) -> dict[str, object]:
    """Load the selected archived experiment metadata."""

    selection_path = settings.results_dir / "final_experiment_selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(
            f"Final experiment selection not found at '{selection_path}'. Run `./.venv/bin/python main.py --step select_final_experiment` first."
        )
    return load_json_if_exists(selection_path)


def export_thesis_results(settings: Settings) -> dict[str, object]:
    """Copy stable thesis artifacts from the selected archived experiment."""

    selection = _load_final_experiment_selection(settings)
    selected_dir_value = str(selection.get("selected_experiment_dir", "") or "").strip()
    if not selected_dir_value or selection.get("status") not in {"selected", "selected_with_warnings"}:
        raise RuntimeError(
            "No final experiment has been selected yet. Run `./.venv/bin/python main.py --step select_final_experiment` first."
        )

    selected_dir = Path(selected_dir_value)
    if not selected_dir.is_absolute():
        selected_dir = settings.project_root / selected_dir
    if not selected_dir.exists():
        raise FileNotFoundError(f"Selected experiment directory '{selected_dir}' does not exist.")

    final_dir = settings.reports_dir / "final_thesis_artifacts"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    artifact_names = [
        "experiment_manifest.json",
        "user_llm_reranking_summary.json",
        "user_llm_metrics_summary.csv",
        "user_rank_comparison.csv",
        "user_llm_explanation_examples.md",
        "user_llm_pilot_summary.md",
        "thesis_balanced_subset_dataset_table.md",
        "user_thesis_metrics_table.md",
        "balanced_subset_methodology_note.md",
    ]
    copied_files: list[str] = []
    missing_files: list[str] = []
    for artifact_name in artifact_names:
        source_path = selected_dir / artifact_name
        if source_path.exists():
            shutil.copy2(source_path, final_dir / artifact_name)
            copied_files.append(artifact_name)
        else:
            missing_files.append(artifact_name)

    manifest = load_json_if_exists(selected_dir / "experiment_manifest.json")
    user_summary = load_json_if_exists(selected_dir / "user_profile_summary.json")
    split_summary = load_json_if_exists(selected_dir / "user_evaluation_split_pilot_summary.json")
    llm_summary = load_json_if_exists(selected_dir / "user_llm_reranking_summary.json")
    metrics_df = pd.read_csv(selected_dir / "user_llm_metrics_summary.csv") if (selected_dir / "user_llm_metrics_summary.csv").exists() else pd.DataFrame()

    baseline_metrics = _metrics_frame_to_dict(metrics_df, "user_tfidf_baseline")
    llm_metrics = _metrics_frame_to_dict(metrics_df, "user_llm_reranker")
    main_metrics_lines = [
        f"- Baseline HitRate@5: {format_metric(baseline_metrics.get('mean_hit_rate_at_5', 0.0))}",
        f"- Baseline HitRate@10: {format_metric(baseline_metrics.get('mean_hit_rate_at_10', 0.0))}",
        f"- Baseline MRR: {format_metric(baseline_metrics.get('mean_mrr', 0.0))}",
        f"- Baseline NDCG@10: {format_metric(baseline_metrics.get('mean_ndcg_at_10', 0.0))}",
        f"- LLM HitRate@5: {format_metric(llm_metrics.get('mean_hit_rate_at_5', 0.0))}",
        f"- LLM HitRate@10: {format_metric(llm_metrics.get('mean_hit_rate_at_10', 0.0))}",
        f"- LLM MRR: {format_metric(llm_metrics.get('mean_mrr', 0.0))}",
        f"- LLM NDCG@10: {format_metric(llm_metrics.get('mean_ndcg_at_10', 0.0))}",
    ]
    summary_lines = [
        "# Final Thesis Experiment Summary",
        "",
        f"- Dataset mode: {manifest.get('active_dataset_mode', 'unknown')}",
        f"- Processed reviews: {manifest.get('cleaned_review_count', 0)}",
        f"- Games: {manifest.get('game_card_count', 0)}",
        f"- Unique users: {user_summary.get('profile_count', 0)}",
        f"- Eligible users: {user_summary.get('eligible_user_count', 0)}",
        f"- Pilot split count: {split_summary.get('split_count', 0)}",
        f"- LLM provider: {manifest.get('llm_provider', 'unknown')}",
        f"- Response language: {manifest.get('llm_response_language', 'unknown')}",
        f"- Selected users: {manifest.get('selected_users', 0)}",
        f"- Token requests: {manifest.get('token_requests_attempted', 0)}",
        f"- Completion requests: {manifest.get('completion_requests_attempted', 0)}",
        f"- Fallback count: {llm_summary.get('fallback_record_count', 0)}",
        f"- Provider preflight OK: {llm_summary.get('provider_preflight_ok', False)}",
        f"- LLM ran: {manifest.get('llm_ran', False)}",
        f"- Real API calls: {manifest.get('real_api_calls_total', manifest.get('real_api_calls', 0))}",
        "",
        "## Metrics",
        *main_metrics_lines,
        "",
        "## Limitations",
        "This is a controlled pilot, not a full-scale statistically significant experiment.",
        "The archived experiment was selected from the experiments folder, not from mutable data/results outputs.",
        "Fallback-only or fallback-heavy runs are not exported as the final thesis experiment.",
        "",
        "## Exported Artifacts",
        *[f"- `{artifact_name}`" for artifact_name in copied_files],
        "",
    ]
    if missing_files:
        summary_lines.extend(
            [
                "## Missing Artifacts",
                *[f"- `{artifact_name}`" for artifact_name in missing_files],
                "",
            ]
        )
    summary_path = final_dir / "final_experiment_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    get_logger().info(
        "Exported final thesis artifacts to %s from %s",
        final_dir,
        selected_dir,
    )
    return {
        "selected_experiment_dir": selected_dir_value,
        "export_dir": _relative_project_path(settings.project_root, final_dir),
        "copied_files": copied_files,
        "missing_files": missing_files,
        "summary_path": _relative_project_path(settings.project_root, summary_path),
    }


def load_analysis_artifacts(settings: Settings) -> dict[str, object]:
    """Load the recommendation and evaluation artifacts needed for report analysis."""

    scenario_records = load_required_jsonl(
        settings.scenarios_output_path,
        "Run `python main.py --step build_scenarios` or `python main.py` first.",
    )
    game_card_records = load_required_jsonl(
        settings.game_cards_path,
        "Run `python main.py --step build_cards` or `python main.py` first.",
    )
    baseline_rows = load_required_jsonl(
        settings.baseline_results_path,
        "Run `python main.py --step baseline` or `python main.py` first.",
    )
    llm_rows = load_optional_records(settings.llm_results_path)
    per_scenario_df = load_required_csv(
        settings.per_scenario_results_path,
        "Run `python main.py --step evaluate` or `python main.py` first.",
    )

    game_cards = {record["game_id"]: record for record in game_card_records}
    scenarios = {record["scenario_id"]: record for record in scenario_records}
    baseline_by_scenario = group_rows_by_scenario(baseline_rows)
    llm_by_scenario = group_rows_by_scenario(llm_rows)
    llm_has_ranked_rows = any(
        row.get("rank") is not None and row.get("game_id")
        for row in llm_rows
    )
    return {
        "settings": settings,
        "scenario_records": scenario_records,
        "scenarios": scenarios,
        "game_cards": game_cards,
        "baseline_rows": baseline_rows,
        "llm_rows": llm_rows,
        "baseline_by_scenario": baseline_by_scenario,
        "llm_by_scenario": llm_by_scenario,
        "per_scenario_df": per_scenario_df,
        "llm_has_ranked_rows": llm_has_ranked_rows,
        "scenario_mode": determine_scenario_mode(scenario_records),
    }


def load_required_jsonl(path: Path, help_text: str) -> list[dict[str, object]]:
    """Load a required JSONL artifact or raise a clear error."""

    if not path.exists():
        raise FileNotFoundError(f"Required artifact '{path}' not found. {help_text}")
    return read_jsonl(path)


def load_required_csv(path: Path, help_text: str) -> pd.DataFrame:
    """Load a required CSV artifact or raise a clear error."""

    if not path.exists():
        raise FileNotFoundError(f"Required artifact '{path}' not found. {help_text}")
    return pd.read_csv(path)


def group_rows_by_scenario(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Group recommendation rows by scenario id."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id", ""))
        grouped.setdefault(scenario_id, []).append(row)
    return grouped


def build_rank_comparison_frame(artifacts: dict[str, object]) -> pd.DataFrame:
    """Build one row per scenario comparing baseline and LLM target ranks."""

    rows: list[dict[str, object]] = []
    for scenario in artifacts["scenario_records"]:
        scenario_id = str(scenario["scenario_id"])
        ground_truth_ids = [str(game_id) for game_id in scenario.get("ground_truth_game_ids", [])]
        baseline_rank = best_ground_truth_rank(
            rows=artifacts["baseline_by_scenario"].get(scenario_id, []),
            ground_truth_ids=ground_truth_ids,
        )
        llm_rank = best_ground_truth_rank(
            rows=artifacts["llm_by_scenario"].get(scenario_id, []),
            ground_truth_ids=ground_truth_ids,
        )
        llm_status = extract_scenario_status(artifacts["llm_by_scenario"].get(scenario_id, []))

        row = {
            "scenario_id": scenario_id,
            "scenario_type": scenario.get("scenario_type", ""),
            "baseline_best_ground_truth_rank": format_rank_value(baseline_rank),
            "llm_best_ground_truth_rank": format_rank_value(llm_rank, unavailable=not artifacts["llm_has_ranked_rows"]),
            "rank_delta": compute_rank_delta(baseline_rank, llm_rank),
            "baseline_hit_at_5": int(is_hit_at_k(baseline_rank, 5)),
            "llm_hit_at_5": format_hit_value(llm_rank, 5, unavailable=not artifacts["llm_has_ranked_rows"]),
            "baseline_hit_at_10": int(is_hit_at_k(baseline_rank, 10)),
            "llm_hit_at_10": format_hit_value(llm_rank, 10, unavailable=not artifacts["llm_has_ranked_rows"]),
            "llm_status": llm_status,
            "interpretation": build_rank_interpretation(
                baseline_rank=baseline_rank,
                llm_rank=llm_rank,
                llm_status=llm_status,
                llm_available=artifacts["llm_has_ranked_rows"],
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_rank_comparison_outputs(
    settings: Settings,
    rank_frame: pd.DataFrame,
    artifacts: dict[str, object],
) -> None:
    """Save rank-comparison CSV and markdown outputs."""

    rank_frame.to_csv(settings.rank_comparison_csv_path, index=False)
    lines = [
        "# Rank Comparison",
        "",
        "Positive `rank_delta` means the LLM moved a ground-truth game closer to the top of the ranking.",
        "Negative `rank_delta` means the LLM worsened the best ground-truth rank.",
        "",
    ]
    if not artifacts["llm_has_ranked_rows"]:
        lines.append("LLM reranking was unavailable or skipped, so LLM rank columns are marked as unavailable.")
        lines.append("")
    lines.append(dataframe_to_markdown(rank_frame))
    settings.rank_comparison_markdown_path.write_text("\n".join(lines), encoding="utf-8")


def build_llm_explanation_checks_frame(
    artifacts: dict[str, object],
) -> tuple[pd.DataFrame, str]:
    """Build heuristic explanation-check rows for LLM recommendations."""

    rows: list[dict[str, object]] = []
    if not artifacts["llm_has_ranked_rows"]:
        frame = pd.DataFrame(
            [
                {
                    "scenario_id": "",
                    "game_id": "",
                    "game_title": "",
                    "rank": "",
                    "has_explanation": False,
                    "explanation_length": 0,
                    "mentions_preference_keyword": False,
                    "mentions_game_keyword": False,
                    "possible_hallucination_flag": "",
                    "status": "unavailable_skipped_no_credentials",
                }
            ]
        )
        return frame, "LLM explanations are unavailable because reranking was skipped."

    for scenario in artifacts["scenario_records"]:
        scenario_id = str(scenario["scenario_id"])
        llm_rows = sorted_ranked_rows(artifacts["llm_by_scenario"].get(scenario_id, []))
        scenario_tokens = extract_content_tokens(str(scenario.get("preference_text", "")))
        for row in llm_rows:
            game_id = str(row.get("game_id", ""))
            game_card = artifacts["game_cards"].get(game_id, {})
            explanation = str(row.get("notes", "") or "").strip()
            explanation_tokens = extract_content_tokens(explanation)
            game_tokens = set()
            for keyword in game_card.get("positive_keywords", []):
                game_tokens.update(extract_content_tokens(str(keyword)))
            for keyword in game_card.get("negative_keywords", []):
                game_tokens.update(extract_content_tokens(str(keyword)))
            game_tokens.update(extract_content_tokens(str(game_card.get("game_card_text", ""))))

            stray_tokens = sorted(
                token
                for token in explanation_tokens
                if token not in scenario_tokens and token not in game_tokens and token not in GENERIC_EXPLANATION_TOKENS
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "game_id": game_id,
                    "game_title": row.get("game_title", ""),
                    "rank": row.get("rank", ""),
                    "has_explanation": bool(explanation),
                    "explanation_length": len(explanation.split()),
                    "mentions_preference_keyword": bool(explanation_tokens & scenario_tokens),
                    "mentions_game_keyword": bool(explanation_tokens & game_tokens),
                    "possible_hallucination_flag": len(stray_tokens) >= 3,
                    "status": row.get("status", ""),
                }
            )

    return pd.DataFrame(rows), (
        "These explanation checks are heuristic and do not provide a definitive factuality evaluation."
    )


def save_llm_explanation_outputs(
    settings: Settings,
    frame: pd.DataFrame,
    note: str,
) -> None:
    """Save LLM explanation-check CSV and markdown outputs."""

    frame.to_csv(settings.llm_explanation_checks_csv_path, index=False)
    lines = [
        "# LLM Explanation Checks",
        "",
        note,
        "",
        dataframe_to_markdown(frame),
    ]
    settings.llm_explanation_checks_markdown_path.write_text("\n".join(lines), encoding="utf-8")


def select_case_study_records(
    artifacts: dict[str, object],
    rank_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Select representative case studies across baseline and LLM categories."""

    category_rules = {
        "baseline_success": lambda row: row["baseline_hit_at_5"] == 1,
        "baseline_failure": lambda row: row["baseline_best_ground_truth_rank"] == "not_found",
    }
    if artifacts["llm_has_ranked_rows"]:
        category_rules.update(
            {
                "llm_success": lambda row: row["llm_hit_at_5"] == 1,
                "llm_failure": lambda row: row["llm_best_ground_truth_rank"] == "not_found",
                "llm_improved": is_llm_improved_row,
                "llm_worsened": is_llm_worsened_row,
            }
        )

    selected: list[dict[str, object]] = []
    for category, predicate in category_rules.items():
        category_rows = [
            row for row in rank_frame.to_dict(orient="records") if predicate(row)
        ][:3]
        for row in category_rows:
            selected.append(
                build_case_study_record(
                    category=category,
                    scenario=artifacts["scenarios"][row["scenario_id"]],
                    artifacts=artifacts,
                )
            )
    return selected


def build_case_study_record(
    category: str,
    scenario: dict[str, object],
    artifacts: dict[str, object],
) -> dict[str, object]:
    """Build one case-study record with compact interpretation."""

    scenario_id = str(scenario["scenario_id"])
    ground_truth_games = [
        build_game_reference(game_id, artifacts["game_cards"])
        for game_id in scenario.get("ground_truth_game_ids", [])
    ]
    baseline_rows = sorted_ranked_rows(artifacts["baseline_by_scenario"].get(scenario_id, []))
    llm_rows = sorted_ranked_rows(artifacts["llm_by_scenario"].get(scenario_id, []))
    baseline_rank = best_ground_truth_rank(baseline_rows, scenario.get("ground_truth_game_ids", []))
    llm_rank = best_ground_truth_rank(llm_rows, scenario.get("ground_truth_game_ids", []))

    return {
        "scenario_id": scenario_id,
        "scenario_type": scenario.get("scenario_type", ""),
        "category": category,
        "preference_text": scenario.get("preference_text", ""),
        "ground_truth_games": ground_truth_games,
        "baseline_top_recommendations": build_baseline_case_recommendations(
            baseline_rows,
            scenario.get("ground_truth_game_ids", []),
        ),
        "llm_top_recommendations": build_llm_case_recommendations(
            llm_rows,
            scenario.get("ground_truth_game_ids", []),
        ),
        "baseline_target_rank": baseline_rank if baseline_rank is not None else 0,
        "llm_target_rank": llm_rank if llm_rank is not None else 0,
        "short_interpretation": build_case_interpretation(
            category=category,
            baseline_rank=baseline_rank,
            llm_rank=llm_rank,
            llm_status=extract_scenario_status(artifacts["llm_by_scenario"].get(scenario_id, [])),
        ),
    }


def build_case_studies_markdown(payload: dict[str, object]) -> str:
    """Render case studies as a markdown report."""

    lines = ["# Case Studies", ""]
    note = str(payload.get("note", "")).strip()
    if note:
        lines.extend([note, ""])

    case_studies = payload.get("case_studies", [])
    if not case_studies:
        lines.append("_No case studies available._")
        return "\n".join(lines)

    for case in case_studies:
        lines.extend(
            [
                f"## {case['scenario_id']} - {case['category']}",
                "",
                f"- Scenario type: {case['scenario_type']}",
                f"- Preference text: {case['preference_text']}",
                f"- Ground truth: {format_ground_truth_games(case['ground_truth_games'])}",
                f"- Interpretation: {case['short_interpretation']}",
                "",
                "Baseline top recommendations:",
                dataframe_to_markdown(pd.DataFrame(case["baseline_top_recommendations"])),
                "",
            ]
        )
        if case["llm_top_recommendations"]:
            lines.extend(
                [
                    "LLM top recommendations:",
                    dataframe_to_markdown(pd.DataFrame(case["llm_top_recommendations"])),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def build_recommendation_example_section(
    scenario: dict[str, object],
    artifacts: dict[str, object],
) -> list[str]:
    """Render one recommendation example section."""

    scenario_id = str(scenario["scenario_id"])
    baseline_rows = sorted_ranked_rows(artifacts["baseline_by_scenario"].get(scenario_id, []))[:5]
    llm_rows = sorted_ranked_rows(artifacts["llm_by_scenario"].get(scenario_id, []))[:5]
    ground_truth_games = [
        build_game_reference(game_id, artifacts["game_cards"])
        for game_id in scenario.get("ground_truth_game_ids", [])
    ]
    interpretation = build_case_interpretation(
        category="recommendation_example",
        baseline_rank=best_ground_truth_rank(baseline_rows, scenario.get("ground_truth_game_ids", [])),
        llm_rank=best_ground_truth_rank(llm_rows, scenario.get("ground_truth_game_ids", [])),
        llm_status=extract_scenario_status(artifacts["llm_by_scenario"].get(scenario_id, [])),
    )

    lines = [
        f"## {scenario_id}",
        "",
        f"- Scenario type: {scenario.get('scenario_type', '')}",
        f"- Preference text: {scenario.get('preference_text', '')}",
        f"- Ground truth games: {format_ground_truth_games(ground_truth_games)}",
        f"- Interpretation: {interpretation}",
        "",
        "Baseline top-5 recommendations:",
        dataframe_to_markdown(pd.DataFrame(build_baseline_case_recommendations(baseline_rows, scenario.get("ground_truth_game_ids", [])))),
        "",
    ]
    if artifacts["llm_has_ranked_rows"] and llm_rows:
        lines.extend(
            [
                "LLM top-5 recommendations:",
                dataframe_to_markdown(pd.DataFrame(build_llm_case_recommendations(llm_rows, scenario.get("ground_truth_game_ids", [])))),
                "",
            ]
        )
    else:
        lines.append("LLM top-5 recommendations: unavailable.")
        lines.append("")
    return lines


def build_thesis_metrics_table(settings: Settings) -> str:
    """Build a markdown table for method-level metrics."""

    lines = ["# Thesis Metrics Table", ""]
    active_mode = get_active_dataset_mode(settings)
    if active_mode == "balanced_subset" and settings.user_metrics_summary_path.exists():
        metrics_df = pd.read_csv(settings.user_metrics_summary_path)
        if metrics_df.empty:
            lines.append("_No metrics available._")
            return "\n".join(lines)

        lines.append("LLM reranking was not included in this baseline run.")
        lines.append("")
        table_df = pd.DataFrame(
            {
                "Method": metrics_df["method"],
                "HitRate@5": metrics_df["mean_hit_rate_at_5"].map(format_metric),
                "HitRate@10": metrics_df["mean_hit_rate_at_10"].map(format_metric),
                "MRR": metrics_df["mean_mrr"].map(format_metric),
                "NDCG@10": metrics_df["mean_ndcg_at_10"].map(format_metric),
                "Evaluated users": metrics_df["evaluated_profiles"],
            }
        )
        lines.append(dataframe_to_markdown(table_df))
        return "\n".join(lines)

    if settings.metrics_summary_path.exists():
        metrics_df = pd.read_csv(settings.metrics_summary_path)
        if metrics_df.empty:
            lines.append("_No metrics available._")
            return "\n".join(lines)

        llm_present = "llm" in set(metrics_df["method"].astype(str))
        if not llm_present:
            lines.append("LLM reranking was not included in this baseline run.")
            lines.append("")

        table_df = pd.DataFrame(
            {
                "Method": metrics_df["method"],
                "HitRate@5": metrics_df["mean_hit_rate_at_5"].map(format_metric),
                "HitRate@10": metrics_df["mean_hit_rate_at_10"].map(format_metric),
                "MRR": metrics_df["mean_mrr"].map(format_metric),
                "NDCG@10": metrics_df["mean_ndcg_at_10"].map(format_metric),
                "Evaluated scenarios": metrics_df["evaluated_scenarios"],
                "Skipped scenarios": metrics_df["skipped_scenarios"],
            }
        )
        lines.append(dataframe_to_markdown(table_df))
        return "\n".join(lines)

    if settings.user_metrics_summary_path.exists():
        lines.append("LLM reranking was not included in this baseline run.")
        lines.append("")
        return "\n".join(lines + ["_See `user_thesis_metrics_table.md` for the user-based baseline metrics._"])

    lines.append("_No metrics available._")
    return "\n".join(lines)


def build_user_thesis_metrics_table(settings: Settings) -> str:
    """Build a markdown table for the user-based balanced-subset baseline."""

    lines = ["# User Thesis Metrics Table", ""]
    if not settings.user_metrics_summary_path.exists():
        lines.append("_No user-based metrics available._")
        return "\n".join(lines)

    metrics_df = pd.read_csv(settings.user_metrics_summary_path)
    if metrics_df.empty:
        lines.append("_No user-based metrics available._")
        return "\n".join(lines)

    lines.append("LLM reranking was not included in this baseline run.")
    lines.append("")
    table_df = pd.DataFrame(
        {
            "Method": metrics_df["method"],
            "HitRate@5": metrics_df["mean_hit_rate_at_5"].map(format_metric),
            "HitRate@10": metrics_df["mean_hit_rate_at_10"].map(format_metric),
            "MRR": metrics_df["mean_mrr"].map(format_metric),
            "NDCG@10": metrics_df["mean_ndcg_at_10"].map(format_metric),
            "Evaluated users": metrics_df["evaluated_profiles"],
        }
    )
    lines.append(dataframe_to_markdown(table_df))
    return "\n".join(lines)


def build_thesis_dataset_table(settings: Settings) -> str:
    """Build a markdown table summarizing the dataset and preprocessing outputs."""

    preprocessing = load_json_if_exists(get_active_preprocessing_summary_path(settings))
    game_card_summary = load_json_if_exists(settings.game_card_summary_path)
    diagnostics = load_json_if_exists(settings.data_diagnostics_json_path)
    languages = diagnostics.get("language_distribution", {})
    language_text = ", ".join(f"{lang}: {count}" for lang, count in languages.items()) if languages else "unknown"

    table_df = pd.DataFrame(
        [
            {
                "Raw reviews": preprocessing.get("raw_row_count", ""),
                "Cleaned reviews": diagnostics.get("total_cleaned_reviews", preprocessing.get("final_row_count", "")),
                "Unique games": diagnostics.get("unique_games", preprocessing.get("unique_games", "")),
                "Generated game cards": game_card_summary.get("generated_game_card_count", ""),
                "Positive reviews": preprocessing.get("positive_reviews", ""),
                "Negative reviews": preprocessing.get("negative_reviews", ""),
                "Positive review ratio": format_metric(diagnostics.get("positive_review_ratio", "")),
                "Languages": language_text,
            }
        ]
    )
    return "# Thesis Dataset Table\n\n" + dataframe_to_markdown(table_df)


def build_thesis_balanced_subset_dataset_table(settings: Settings) -> str:
    """Build a markdown table summarizing the balanced-subset workflow."""

    preprocessing = load_json_if_exists(settings.preprocessing_balanced_subset_summary_path)
    profile_summary = load_json_if_exists(settings.user_profile_summary_path)
    split_summary = load_json_if_exists(settings.user_split_summary_path)
    metrics_summary = load_csv_summary_if_exists(settings.user_metrics_summary_path)

    if not preprocessing:
        return "# Balanced Subset Dataset Table\n\n_No balanced subset summary available._"

    table_df = pd.DataFrame(
        [
            {
                "Raw rows scanned": preprocessing.get("raw_rows_scanned_pass_1", preprocessing.get("raw_row_count", "")),
                "Processed reviews in subset": preprocessing.get("processed_rows_written", preprocessing.get("final_row_count", "")),
                "Selected games": preprocessing.get("selected_games", 0),
                "Unique users": preprocessing.get("unique_users", 0),
                "Eligible users": preprocessing.get("eligible_users_estimate", 0),
                "User profiles": profile_summary.get("profile_count", 0),
                "Evaluation splits": split_summary.get("split_count", 0),
                "Candidate pool size": settings.user_candidate_pool_size,
                "Baseline evaluated users": metrics_summary.get("evaluated_profiles", 0),
            }
        ]
    )
    return "# Thesis Balanced Subset Dataset Table\n\n" + dataframe_to_markdown(table_df)


def build_balanced_subset_methodology_note(settings: Settings) -> str:
    """Explain the balanced-subset methodology in a thesis-friendly note."""

    preprocessing = load_json_if_exists(settings.preprocessing_balanced_subset_summary_path)
    if not preprocessing:
        return "# Balanced Subset Methodology Note\n\n_No balanced subset summary available._"

    paragraphs = [
        (
            "The full Steam Reviews export is too large for a single-memory preprocessing pass in a lightweight bachelor-thesis prototype, so the project uses a chunked pipeline and a balanced subset for the controlled user-based baseline experiment."
        ),
        (
            "A simple first-N subset was not sufficient because the raw CSV is ordered by app_id, which concentrates early rows on only a few games and produces an overly narrow recommendation benchmark."
        ),
        (
            "The balanced subset was therefore constructed in two passes over the full raw dataset: the first pass collected per-game review statistics, and the second pass selected the most review-rich games that satisfied the minimum review thresholds."
        ),
        (
            f"The resulting subset contains {preprocessing.get('selected_games', 0)} games and {preprocessing.get('processed_rows_written', 0)} processed reviews, while preserving user_id so that user-based offline evaluation remains possible."
        ),
        (
            "The selected subset is appropriate for a controlled thesis workflow, but the resulting metrics must be reported as balanced-subset results rather than full-dataset results."
        ),
        (
            "The same chunked preprocessing pipeline can later be run on the full 21.7 million-row dataset if the thesis work is extended beyond the balanced subset."
        ),
    ]
    return "# Balanced Subset Methodology Note\n\n" + "\n\n".join(paragraphs) + "\n"


def build_thesis_scenario_table(settings: Settings) -> str:
    """Build a markdown table summarizing scenario coverage and validation status."""

    scenario_records = load_required_jsonl(
        settings.scenarios_output_path,
        "Run `python main.py --step build_scenarios` or `python main.py` first.",
    )
    if not scenario_records:
        return "# Thesis Scenario Table\n\n_No scenarios available._"

    validation_df = load_required_csv(
        settings.scenario_validation_report_path,
        "Run `python main.py --step validate_scenarios` first.",
    )
    scenario_df = pd.DataFrame(
        scenario_records,
        columns=[
            "scenario_id",
            "scenario_type",
            "preference_text",
            "seed_game_ids",
            "excluded_game_ids",
            "ground_truth_game_ids",
            "candidate_game_ids",
            "notes",
        ],
    )
    merged_df = scenario_df.merge(validation_df, on="scenario_id", how="left")

    rows: list[dict[str, object]] = []
    for scenario_type, group in merged_df.groupby("scenario_type", sort=False):
        rows.append(
            {
                "Scenario type": scenario_type,
                "Number of scenarios": int(len(group)),
                "Valid scenarios": int((group["status"] == "ok").sum()),
                "Warning scenarios": int((group["status"] == "warning").sum()),
                "Invalid scenarios": int((group["status"] == "invalid").sum()),
                "Average candidate count": round(float(group["candidate_count"].fillna(0).mean()), 2),
                "Average ground truth count": round(float(group["ground_truth_count"].fillna(0).mean()), 2),
            }
        )
    return "# Thesis Scenario Table\n\n" + dataframe_to_markdown(pd.DataFrame(rows))


def build_preflight_next_steps(
    status: str,
    dataset_too_small: bool,
    scenario_mode: str,
    llm_credentials_configured: bool,
    scenario_validation_exists: bool,
    invalid_scenarios: int,
    preprocessing_completed: bool,
    subset_cleaned_reviews_exist: bool,
    full_cleaned_reviews_exist: bool,
    active_dataset_mode: str,
    balanced_subset_recommended: bool = False,
) -> list[str]:
    """Generate concise preflight recommendations."""

    steps: list[str] = []
    if dataset_too_small:
        steps.append("Replace the bundled sample dataset with the real Steam Reviews CSV.")
    if scenario_mode != "manual_or_predefined":
        steps.append("Create and validate manual scenarios before the final thesis experiment.")
    if not scenario_validation_exists:
        steps.append("Run `./.venv/bin/python main.py --step validate_scenarios` to create a validation report.")
    if invalid_scenarios > 0:
        steps.append("Fix invalid scenarios before running the main experiment.")
    if not preprocessing_completed and active_dataset_mode == "full":
        steps.append("Run `./.venv/bin/python main.py --step preprocess` to complete full-dataset preprocessing.")
    if subset_cleaned_reviews_exist and not full_cleaned_reviews_exist:
        steps.append(
            "You may run experiments on the subset by setting ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_subset.csv, but final thesis conclusions should mention this limitation."
        )
    if balanced_subset_recommended:
        steps.append(
            "You may also use ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv for development or a limited thesis experiment, but the thesis must clearly describe it as a subset-based setup."
        )
        steps.append(
            "Run `./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_baseline.json` for the controlled balanced-subset baseline workflow."
        )
    if status == "not_ready":
        steps.append("Run `./.venv/bin/python main.py --step all` after fixing the missing artifacts.")
    if not llm_credentials_configured:
        steps.append("Configure the provider credentials and model if LLM reranking is needed.")
    return deduplicate_preserve_order(steps)


def build_preflight_commands(status: str, scenario_validation_exists: bool) -> list[str]:
    """Suggest the most useful next commands from the current state."""

    commands = [
        "./.venv/bin/python main.py --step check_env",
        "./.venv/bin/python main.py --step preprocess_status",
        "./.venv/bin/python main.py --step all",
    ]
    if not scenario_validation_exists:
        commands.append("./.venv/bin/python main.py --step validate_scenarios")
    else:
        commands.extend(
            [
                "./.venv/bin/python main.py --step validate_scenarios",
                "./.venv/bin/python main.py --step readiness",
            ]
        )
    if status in {"ready_for_baseline_experiment", "ready_for_llm_experiment"}:
        commands.append("./.venv/bin/python main.py --step analysis")
    return deduplicate_preserve_order(commands)


def build_preflight_markdown(report: dict[str, object]) -> str:
    """Render the preflight report as thesis-friendly markdown."""

    lines = [
        "# Preflight Report",
        "",
        "## Status",
        f"- Status: {report['status']}",
        f"- Dataset path: `{report['configured_dataset_path']}`",
        f"- LLM credentials configured: {report['llm_credentials_configured']}",
        f"- Scenario mode: {report['scenario_mode']}",
        f"- Dataset too small: {report['dataset_too_small']}",
        f"- Active processed reviews: `{report.get('active_processed_reviews', '')}`",
        f"- Active dataset mode: {report.get('active_dataset_mode', 'unknown')}",
        f"- Current recommended workflow: {report.get('current_recommended_workflow', 'not_ready')}",
        f"- Has user id: {report['has_user_id']}",
        f"- User-based mode available: {report['user_based_mode_available']}",
        f"- Unique users: {report['unique_users']}",
        f"- Eligible users: {report['eligible_user_count']}",
        f"- Full user-based experiment ready: {report.get('full_user_based_ready', False)}",
        f"- Balanced-subset user-based experiment ready: {report.get('balanced_subset_user_based_ready', False)}",
        f"- Looks like sample or incomplete data: {report.get('looks_like_sample_or_incomplete', False)}",
        f"- Processed artifacts match current dataset: {report.get('artifacts_match_current_dataset', True)}",
        f"- Full cleaned reviews exist: {report.get('full_cleaned_reviews_exist', False)}",
        f"- Subset cleaned reviews exist: {report.get('subset_cleaned_reviews_exist', False)}",
        f"- Balanced subset exists: {report.get('balanced_subset_exists', False)}",
        f"- Balanced subset cleaned reviews: {report.get('balanced_subset_cleaned_reviews', 0)}",
        f"- Balanced subset unique games: {report.get('balanced_subset_unique_games', 0)}",
        f"- Balanced subset unique users: {report.get('balanced_subset_unique_users', 0)}",
        f"- Balanced subset recommended for experiment: {report.get('balanced_subset_recommended_for_experiment', False)}",
        f"- Preprocessing progress status: {report.get('preprocessing_progress_status', 'missing')}",
        f"- Preprocessing progress mode: {report.get('preprocessing_progress_mode', 'unknown')}",
        f"- Full preprocessing completed: {report.get('full_preprocessing_completed', False)}",
        "",
        "## Warnings",
        *(
            [f"- {SAMPLE_WARNING_MESSAGE}"]
            if report.get("looks_like_sample_or_incomplete", False)
            else ["- none"]
        ),
        *(
            [
                "- The balanced subset looks suitable for development or a limited thesis experiment, but it must be documented as a subset-based experiment."
            ]
            if report.get("balanced_subset_recommended_for_experiment", False)
            else []
        ),
        *(
            ["- Balanced subset workflow is ready and can be used as a controlled subset-based experiment."]
            if report.get("balanced_subset_ready", False)
            and report.get("active_dataset_mode") == "balanced_subset"
            else []
        ),
        "",
        "## What Is Ready",
    ]
    ready_items = report.get("what_is_ready", [])
    if ready_items:
        lines.extend(f"- {item}" for item in ready_items)
    else:
        lines.append("- Nothing important is ready yet.")

    lines.extend(["", "## What Is Missing"])
    missing_items = report.get("what_is_missing", [])
    if missing_items:
        lines.extend(f"- {item}" for item in missing_items)
    else:
        lines.append("- No critical gaps were detected.")

    lines.extend(["", "## Recommended Next Steps"])
    next_steps = report.get("recommended_next_steps", [])
    if next_steps:
        lines.extend(f"- {item}" for item in next_steps)
    else:
        lines.append("- No additional preflight actions were suggested.")

    lines.extend(["", "## Commands To Run Next"])
    commands = report.get("commands_to_run_next", [])
    if commands:
        lines.extend(f"- `{command}`" for command in commands)
    else:
        lines.append("- No commands suggested.")

    return "\n".join(lines) + "\n"


def build_case_study_note(
    artifacts: dict[str, object],
    llm_available: bool,
    case_count: int,
) -> str:
    """Build a short note shown at the top of the case-study report."""

    notes: list[str] = []
    if artifacts["scenario_mode"] == "synthetic_demo_only":
        notes.append(
            "These case studies are generated from synthetic demo scenarios and are intended for technical validation only."
        )
    if not llm_available:
        notes.append("LLM case categories are limited because reranking was skipped or unavailable.")
    if case_count < 6:
        notes.append("Only a small number of scenarios were available, so some case-study categories may be missing.")
    return " ".join(notes)


def build_analysis_warning_lines(artifacts: dict[str, object]) -> list[str]:
    """Build prominent warning lines for analysis reports."""

    lines: list[str] = []
    if artifacts["scenario_mode"] == "synthetic_demo_only":
        lines.append(
            "Warning: these examples use synthetic demo scenarios and are intended for technical validation only."
        )
    if len(artifacts["scenario_records"]) < 10:
        lines.append(
            "Warning: the bundled sample dataset is tiny, so these examples are not suitable for scientific conclusions."
        )
    return [f"- {line}" for line in lines]


def build_game_reference(game_id: str, game_cards: dict[str, dict[str, object]]) -> dict[str, str]:
    """Convert a game id to a compact reference object."""

    card = game_cards.get(str(game_id), {})
    return {
        "game_id": str(game_id),
        "game_title": str(card.get("game_title", "unknown")),
    }


def build_baseline_case_recommendations(
    rows: list[dict[str, object]],
    ground_truth_ids: list[str],
) -> list[dict[str, object]]:
    """Build baseline recommendation rows for reports."""

    truth_set = {str(game_id) for game_id in ground_truth_ids}
    return [
        {
            "rank": row.get("rank", ""),
            "game_id": row.get("game_id", ""),
            "game_title": row.get("game_title", ""),
            "score": row.get("score", ""),
            "is_ground_truth": str(row.get("game_id", "")) in truth_set,
        }
        for row in rows[:5]
    ]


def build_llm_case_recommendations(
    rows: list[dict[str, object]],
    ground_truth_ids: list[str],
) -> list[dict[str, object]]:
    """Build LLM recommendation rows for reports."""

    truth_set = {str(game_id) for game_id in ground_truth_ids}
    return [
        {
            "rank": row.get("rank", ""),
            "game_id": row.get("game_id", ""),
            "game_title": row.get("game_title", ""),
            "relevance_score": row.get("score", ""),
            "explanation": row.get("notes", ""),
            "is_ground_truth": str(row.get("game_id", "")) in truth_set,
        }
        for row in rows[:5]
    ]


def sorted_ranked_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Sort recommendation rows that contain actual ranked items."""

    ranked_rows = [row for row in rows if row.get("rank") is not None and row.get("game_id")]
    return sorted(ranked_rows, key=lambda row: int(row.get("rank", 0)))


def best_ground_truth_rank(rows: list[dict[str, object]], ground_truth_ids: list[str]) -> int | None:
    """Return the best observed rank of any ground-truth item."""

    truth_set = {str(game_id) for game_id in ground_truth_ids}
    ranks = [
        int(row["rank"])
        for row in rows
        if row.get("rank") is not None and str(row.get("game_id", "")) in truth_set
    ]
    return min(ranks) if ranks else None


def extract_scenario_status(rows: list[dict[str, object]]) -> str:
    """Extract the status string for one scenario's recommendation rows."""

    if not rows:
        return "unavailable"
    return str(rows[0].get("status", "unavailable"))


def format_rank_value(rank: int | None, unavailable: bool = False) -> object:
    """Format a rank value for CSV export."""

    if unavailable:
        return "unavailable"
    if rank is None:
        return "not_found"
    return rank


def format_hit_value(rank: int | None, k: int, unavailable: bool = False) -> object:
    """Format hit@k for CSV export."""

    if unavailable:
        return "unavailable"
    return int(is_hit_at_k(rank, k))


def compute_rank_delta(baseline_rank: int | None, llm_rank: int | None) -> object:
    """Compute baseline_rank - llm_rank when both are available."""

    if baseline_rank is None or llm_rank is None:
        return ""
    return baseline_rank - llm_rank


def is_hit_at_k(rank: int | None, k: int) -> bool:
    """Return whether a rank falls within the top-k window."""

    return rank is not None and rank <= k


def build_rank_interpretation(
    baseline_rank: int | None,
    llm_rank: int | None,
    llm_status: str,
    llm_available: bool,
) -> str:
    """Generate a one-line interpretation of rank comparison results."""

    if not llm_available:
        return "LLM reranking was unavailable for this run."
    if llm_status.startswith("skipped"):
        return f"LLM reranking status: {llm_status}."
    if baseline_rank is None and llm_rank is None:
        return "Neither method retrieved a ground-truth game in the saved ranking."
    if baseline_rank is None and llm_rank is not None:
        return "LLM retrieved a ground-truth game where the baseline did not."
    if baseline_rank is not None and llm_rank is None:
        return "Baseline retrieved a ground-truth game, but the LLM-ranked list did not."
    if llm_rank < baseline_rank:
        return "LLM improved the best ground-truth rank."
    if llm_rank > baseline_rank:
        return "LLM worsened the best ground-truth rank."
    return "LLM preserved the same best ground-truth rank as the baseline."


def is_llm_improved_row(row: dict[str, object]) -> bool:
    """Return True when LLM rank improved relative to the baseline."""

    baseline_rank = parse_rank_placeholder(row.get("baseline_best_ground_truth_rank"))
    llm_rank = parse_rank_placeholder(row.get("llm_best_ground_truth_rank"))
    if baseline_rank is None and llm_rank is not None:
        return True
    return baseline_rank is not None and llm_rank is not None and llm_rank < baseline_rank


def is_llm_worsened_row(row: dict[str, object]) -> bool:
    """Return True when LLM rank worsened relative to the baseline."""

    baseline_rank = parse_rank_placeholder(row.get("baseline_best_ground_truth_rank"))
    llm_rank = parse_rank_placeholder(row.get("llm_best_ground_truth_rank"))
    if baseline_rank is not None and llm_rank is None:
        return True
    return baseline_rank is not None and llm_rank is not None and llm_rank > baseline_rank


def parse_rank_placeholder(value: object) -> int | None:
    """Convert a stored rank placeholder back into an integer rank or None."""

    if value in {"", None, "not_found", "unavailable"}:
        return None
    return int(value)


def build_case_interpretation(
    category: str,
    baseline_rank: int | None,
    llm_rank: int | None,
    llm_status: str,
) -> str:
    """Generate a short human-readable case interpretation."""

    if category == "baseline_success":
        return f"Baseline retrieved a ground-truth game at rank {baseline_rank}."
    if category == "baseline_failure":
        return "Baseline did not retrieve any ground-truth game in the saved ranking."
    if category == "llm_success":
        return f"LLM reranking retrieved a ground-truth game at rank {llm_rank}."
    if category == "llm_failure":
        return f"LLM reranking status was {llm_status}, and no ground-truth game appeared in the saved ranking."
    if category == "llm_improved":
        return build_rank_interpretation(baseline_rank, llm_rank, llm_status, llm_available=True)
    if category == "llm_worsened":
        return build_rank_interpretation(baseline_rank, llm_rank, llm_status, llm_available=True)
    return build_rank_interpretation(baseline_rank, llm_rank, llm_status, llm_available=True)


def format_ground_truth_games(games: list[dict[str, str]]) -> str:
    """Format ground-truth game references for markdown."""

    if not games:
        return "none"
    return ", ".join(f"{game['game_title']} ({game['game_id']})" for game in games)


def extract_content_tokens(text: str) -> set[str]:
    """Extract simple content tokens for heuristic overlap checks."""

    return {
        token
        for token in re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", text.lower())
        if token not in GENERIC_EXPLANATION_TOKENS
    }


def format_metric(value: object) -> object:
    """Format numeric metrics to three decimals when possible."""

    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return value


def load_json_if_exists(path: Path) -> dict[str, object]:
    """Load a JSON file if present, otherwise return an empty dictionary."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_summary_if_exists(path: Path) -> dict[str, object]:
    """Load the first row of a CSV summary file if present."""

    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


GENERIC_EXPLANATION_TOKENS = {
    "about",
    "because",
    "based",
    "candidate",
    "could",
    "explanation",
    "features",
    "game",
    "games",
    "good",
    "match",
    "player",
    "preference",
    "preferences",
    "recommendation",
    "recommended",
    "relevant",
    "scenario",
    "strong",
    "suggests",
    "this",
    "those",
    "with",
}
