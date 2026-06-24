
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import signal
import time

import pandas as pd

from src.config import Settings
from src.data_loader import (
    build_column_resolution,
    normalize_identifier_series,
    normalize_raw_column_name,
    select_reviews_csv_path,
    validate_dataset_schema,
)
from src.utils import get_logger


LANGUAGE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "ru": "russian",
    "rus": "russian",
    "russian": "russian",
}
BOOL_TRUE_VALUES = {"true", "1", "yes", "y", "recommended", "positive"}
BOOL_FALSE_VALUES = {"false", "0", "no", "n", "not recommended", "negative"}
NORMALIZED_COLUMN_ORDER = [
    "game_id",
    "game_title",
    "review_id",
    "language",
    "review_text",
    "review_clean",
    "timestamp_created",
    "timestamp_updated",
    "recommended",
    "votes_helpful",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "user_id",
    "author_num_games_owned",
    "author_num_reviews",
    "playtime_forever",
    "playtime_last_two_weeks",
    "playtime_at_review",
    "author_last_played",
]
NUMERIC_COLUMNS = [
    "votes_helpful",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "author_num_games_owned",
    "author_num_reviews",
    "playtime_forever",
    "playtime_last_two_weeks",
    "playtime_at_review",
]
INTEGER_NUMERIC_COLUMNS = {
    "votes_helpful",
    "votes_funny",
    "comment_count",
    "author_num_games_owned",
    "author_num_reviews",
}
BOOL_COLUMNS = [
    "recommended",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]
TIMESTAMP_COLUMNS = [
    "timestamp_created",
    "timestamp_updated",
    "author_last_played",
]
SOURCE_TO_NORMALIZED_COLUMNS = {
    "app_id": "game_id",
    "app_name": "game_title",
    "review_id": "review_id",
    "language": "language",
    "review": "review_text",
    "timestamp_created": "timestamp_created",
    "timestamp_updated": "timestamp_updated",
    "recommended": "recommended",
    "votes_helpful": "votes_helpful",
    "votes_funny": "votes_funny",
    "weighted_vote_score": "weighted_vote_score",
    "comment_count": "comment_count",
    "steam_purchase": "steam_purchase",
    "received_for_free": "received_for_free",
    "written_during_early_access": "written_during_early_access",
    "author.steamid": "user_id",
    "author.num_games_owned": "author_num_games_owned",
    "author.num_reviews": "author_num_reviews",
    "author.playtime_forever": "playtime_forever",
    "author.playtime_last_two_weeks": "playtime_last_two_weeks",
    "author.playtime_at_review": "playtime_at_review",
    "author.last_played": "author_last_played",
}
STALE_ARTIFACT_PATHS = [
    "game_cards_path",
    "user_profiles_path",
    "user_splits_path",
    "game_card_summary_path",
    "user_profile_summary_path",
    "user_split_summary_path",
    "user_baseline_results_path",
    "user_per_profile_results_path",
    "user_metrics_summary_path",
]
DEBUG_MAX_ROWS_DEFAULT = 500_000
PREPROCESS_SUMMARY_NOTE = (
    "This is a research subset, not the full dataset."
)
BALANCED_SUBSET_NOTE = (
    "This balanced subset is a research subset, not the full dataset."
)


class PreprocessingInterrupted(RuntimeError):
    pass


def preprocess_reviews(reviews_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:

    validate_preprocessing_settings(settings)
    normalized_df = normalize_chunk_dataframe(reviews_df)
    stats = init_preprocessing_stats(
        settings=settings,
        dataset_path=settings.reviews_csv_path,
        max_rows_for_debug=None,
        mode="full",
        final_output_path=settings.reviews_clean_path,
        temp_output_path=settings.reviews_clean_path.with_suffix(".csv.tmp"),
    )
    cleaned_df, chunk_stats = clean_normalized_chunk(
        normalized_df,
        settings=settings,
        language_filter=normalize_language_filter(settings.language_filter),
        seen_review_ids=set(),
        tracked_review_ids=set(),
        track_ids=False,
    )
    update_stats_from_chunk(stats, normalized_df, cleaned_df, chunk_stats)
    output_path = settings.reviews_clean_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    finalize_stats(stats)
    write_preprocessing_summary(
        summary_path=settings.preprocessing_summary_path,
        markdown_path=settings.preprocessing_summary_markdown_path,
        summary=build_preprocessing_summary(stats),
    )
    write_raw_processed_comparison(
        json_path=settings.raw_processed_comparison_json_path,
        markdown_path=settings.raw_processed_comparison_markdown_path,
        summary=build_preprocessing_summary(stats),
    )
    return cleaned_df


def run_chunked_preprocessing(
    settings: Settings,
    *,
    mode: str = "full",
) -> dict[str, object]:

    logger = get_logger()
    validate_preprocessing_settings(settings)
    dataset_path = select_reviews_csv_path(settings)
    schema_report = validate_dataset_schema(settings, dataset_path)
    if schema_report["status"] == "error":
        raise ValueError(
            "Reviews CSV is missing required columns: "
            + ", ".join(schema_report["missing_expected_columns"])
        )

    run_paths = resolve_preprocessing_run_paths(settings, mode)
    max_rows = resolve_max_rows_for_mode(settings, mode)
    if settings.processed_write_mode != "overwrite":
        raise ValueError("PROCESSED_WRITE_MODE currently supports only 'overwrite'.")

    prepare_preprocessing_outputs(settings, run_paths, mode=mode)
    language_filter = normalize_language_filter(settings.language_filter)
    stats = init_preprocessing_stats(
        settings=settings,
        dataset_path=dataset_path,
        max_rows_for_debug=max_rows,
        mode=mode,
        final_output_path=run_paths["output_path"],
        temp_output_path=run_paths["temp_output_path"],
    )
    stats["column_name_warnings"] = list(schema_report.get("column_name_warnings", []))
    stats["raw_author_steamid_present"] = bool(schema_report.get("user_id_column_detected", False))
    stats["matched_user_id_source_column"] = str(
        schema_report.get("matched_user_id_source_column", "")
    )
    write_progress_report(settings, stats, status="running")

    seen_review_ids: set[int | str] = set()
    tracked_review_ids: set[int | str] = set()
    unique_game_ids: set[int | str] = set()
    unique_user_ids: set[int | str] = set()
    raw_unique_user_ids: set[int | str] = set()
    language_counter: Counter[str] = Counter()
    first_chunk = True
    raw_rows_processed = 0
    chunk_index = 0

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _interrupt_handler(signum, frame):  # noqa: ARG001
        raise PreprocessingInterrupted(f"Received signal {signum}.")

    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    try:
        read_csv_kwargs = {
            "chunksize": settings.preprocess_chunksize,
            "low_memory": False,
        }
        for raw_chunk in pd.read_csv(dataset_path, **read_csv_kwargs):
            if max_rows is not None and raw_rows_processed >= max_rows:
                break

            if max_rows is not None:
                remaining = max_rows - raw_rows_processed
                if remaining <= 0:
                    break
                if len(raw_chunk) > remaining:
                    raw_chunk = raw_chunk.iloc[:remaining].copy()

            chunk_index += 1
            raw_rows_processed += len(raw_chunk)
            normalized_chunk = normalize_chunk_dataframe(raw_chunk)
            cleaned_chunk, chunk_stats = clean_normalized_chunk(
                normalized_chunk,
                settings=settings,
                language_filter=language_filter,
                seen_review_ids=seen_review_ids,
                tracked_review_ids=tracked_review_ids,
                track_ids=True,
            )
            update_stats_from_chunk(stats, normalized_chunk, cleaned_chunk, chunk_stats)
            update_exact_trackers(
                normalized_chunk=normalized_chunk,
                cleaned_chunk=cleaned_chunk,
                raw_unique_user_ids=raw_unique_user_ids,
                unique_game_ids=unique_game_ids,
                unique_user_ids=unique_user_ids,
                language_counter=language_counter,
            )

            cleaned_chunk.to_csv(
                run_paths["temp_output_path"],
                mode="a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            stats["chunks_processed"] = chunk_index
            stats["raw_row_count"] = raw_rows_processed
            stats["unique_reviews"] = len(tracked_review_ids)
            stats["unique_games"] = len(unique_game_ids)
            stats["unique_users"] = len(unique_user_ids)
            stats["raw_unique_author_steamid_count"] = len(raw_unique_user_ids)
            stats["language_distribution_top"] = dict(language_counter.most_common(10))
            write_progress_report(settings, stats, status="running")
            logger.info(
                "Processed chunk %s: raw=%s, kept=%s",
                chunk_index,
                len(raw_chunk),
                len(cleaned_chunk),
            )

        if first_chunk:
            pd.DataFrame(columns=NORMALIZED_COLUMN_ORDER).to_csv(
                run_paths["temp_output_path"],
                index=False,
            )

        os.replace(run_paths["temp_output_path"], run_paths["output_path"])
        stats["raw_row_count"] = raw_rows_processed
        stats["unique_reviews"] = len(tracked_review_ids)
        stats["unique_games"] = len(unique_game_ids)
        stats["unique_users"] = len(unique_user_ids)
        stats["raw_unique_author_steamid_count"] = len(raw_unique_user_ids)
        stats["language_distribution_top"] = dict(language_counter.most_common(10))
        finalize_stats(stats)
        summary = build_preprocessing_summary(stats)
        write_preprocessing_outputs(run_paths, summary, mode=mode)
        write_progress_report(settings, stats, status="completed")
        log_preprocessing_summary(logger, run_paths["summary_path"], summary, mode=mode)
        return summary
    except PreprocessingInterrupted as exc:
        finalize_stats(stats)
        write_progress_report(settings, stats, status="interrupted")
        logger.warning("%s", exc)
        logger.warning(
            "Resume from partial temp file is not implemented. Restart preprocessing to produce a consistent output file."
        )
        raise
    except Exception:
        finalize_stats(stats)
        write_progress_report(settings, stats, status="failed")
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def run_balanced_subset_preprocessing(settings: Settings) -> dict[str, object]:

    logger = get_logger()
    validate_preprocessing_settings(settings)
    dataset_path = select_reviews_csv_path(settings)
    schema_report = validate_dataset_schema(settings, dataset_path)
    if schema_report["status"] == "error":
        raise ValueError(
            "Reviews CSV is missing required columns: "
            + ", ".join(schema_report["missing_expected_columns"])
        )

    run_paths = resolve_preprocessing_run_paths(settings, "balanced_subset")
    prepare_preprocessing_outputs(settings, run_paths, mode="balanced_subset")
    minimal_columns = [
        "app_id",
        "app_name",
        "review",
        "recommended",
        "author.steamid",
    ]
    user_id_raw_column = str(schema_report.get("matched_user_id_source_column", ""))
    game_stats, raw_rows_scanned_pass_1 = collect_raw_game_sampling_stats(
        dataset_path=dataset_path,
        schema_report=schema_report,
        chunksize=settings.balanced_subset_chunksize,
        minimal_columns=minimal_columns,
    )
    selected_games = select_balanced_subset_games(game_stats, settings)
    selected_game_id_set = {str(record["game_id"]) for record in selected_games}
    save_raw_game_sampling_stats(
        settings,
        [
            {
                **record,
                "selected_for_balanced_subset": str(record["game_id"]) in selected_game_id_set,
            }
            for record in game_stats
        ],
    )
    if not selected_games:
        summary = build_balanced_subset_summary(
            settings=settings,
            dataset_path=dataset_path,
            selected_games=[],
            raw_rows_scanned_pass_1=raw_rows_scanned_pass_1,
            raw_rows_scanned_pass_2=0,
            stats={},
            user_id_mapping_status="source_column_missing" if not user_id_raw_column else "source_column_empty",
            eligible_users_estimate=0,
            duration_seconds=0.0,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        write_preprocessing_outputs(run_paths, summary, mode="balanced_subset")
        write_progress_payload(
            settings=settings,
            payload=build_progress_payload_from_summary(summary, status="completed"),
        )
        return summary

    selected_game_ids = {str(record["game_id"]) for record in selected_games}
    per_game_limit = max(
        1,
        settings.balanced_subset_max_processed_rows // max(1, len(selected_game_ids)),
    )
    language_filter = normalize_language_filter(settings.language_filter)
    stats = init_preprocessing_stats(
        settings=settings,
        dataset_path=dataset_path,
        max_rows_for_debug=settings.balanced_subset_max_processed_rows,
        mode="balanced_subset",
        final_output_path=run_paths["output_path"],
        temp_output_path=run_paths["temp_output_path"],
    )
    stats["column_name_warnings"] = list(schema_report.get("column_name_warnings", []))
    stats["raw_author_steamid_present"] = bool(schema_report.get("user_id_column_detected", False))
    stats["matched_user_id_source_column"] = user_id_raw_column
    stats["selected_games"] = len(selected_game_ids)
    stats["target_games"] = settings.balanced_subset_target_games
    started_at = stats["started_at"]
    write_progress_report(settings, stats, status="running")

    seen_review_ids: set[int | str] = set()
    tracked_review_ids: set[int | str] = set()
    unique_game_ids: set[int | str] = set()
    unique_user_ids: set[int | str] = set()
    raw_unique_user_ids: set[int | str] = set()
    language_counter: Counter[str] = Counter()
    user_review_counts: Counter[int | str] = Counter()
    user_positive_counts: Counter[int | str] = Counter()
    per_game_written_counts: Counter[str] = Counter()
    per_game_user_sets = {game_id: set() for game_id in selected_game_ids}
    selected_game_title_map = {
        str(record["game_id"]): str(record["game_title"])
        for record in selected_games
    }
    raw_rows_scanned_pass_2 = 0
    first_chunk = True
    chunk_index = 0

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _interrupt_handler(signum, frame):  # noqa: ARG001
        raise PreprocessingInterrupted(f"Received signal {signum}.")

    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    try:
        for raw_chunk in pd.read_csv(
            dataset_path,
            chunksize=settings.balanced_subset_chunksize,
            low_memory=False,
        ):
            if int(stats["processed_row_count"]) >= settings.balanced_subset_max_processed_rows:
                break

            chunk_index += 1
            raw_rows_scanned_pass_2 += len(raw_chunk)
            normalized_chunk = normalize_chunk_dataframe(raw_chunk)
            normalized_chunk["game_id"] = normalize_identifier_series(normalized_chunk["game_id"])
            selected_mask = normalized_chunk["game_id"].isin(selected_game_ids)
            if not selected_mask.any():
                stats["chunks_processed"] = raw_rows_scanned_pass_2 // settings.balanced_subset_chunksize
                stats["raw_row_count"] = raw_rows_scanned_pass_2
                write_progress_report(settings, stats, status="running")
                continue

            filtered_chunk = normalized_chunk.loc[selected_mask].copy()
            filtered_chunk = enforce_per_game_row_cap(
                filtered_chunk,
                per_game_written_counts=per_game_written_counts,
                per_game_limit=per_game_limit,
                remaining_total=settings.balanced_subset_max_processed_rows - int(stats["processed_row_count"]),
            )
            if filtered_chunk.empty:
                stats["chunks_processed"] = chunk_index
                stats["raw_row_count"] = raw_rows_scanned_pass_2
                write_progress_report(settings, stats, status="running")
                continue

            cleaned_chunk, chunk_stats = clean_normalized_chunk(
                filtered_chunk,
                settings=settings,
                language_filter=language_filter,
                seen_review_ids=seen_review_ids,
                tracked_review_ids=tracked_review_ids,
                track_ids=True,
            )
            cleaned_chunk = enforce_per_game_row_cap_after_cleaning(
                cleaned_chunk,
                per_game_written_counts=per_game_written_counts,
                per_game_limit=per_game_limit,
                remaining_total=settings.balanced_subset_max_processed_rows - int(stats["processed_row_count"]),
            )
            if cleaned_chunk.empty:
                continue

            update_stats_from_chunk(stats, filtered_chunk, cleaned_chunk, chunk_stats)
            update_exact_trackers(
                normalized_chunk=filtered_chunk,
                cleaned_chunk=cleaned_chunk,
                raw_unique_user_ids=raw_unique_user_ids,
                unique_game_ids=unique_game_ids,
                unique_user_ids=unique_user_ids,
                language_counter=language_counter,
            )
            update_balanced_subset_user_stats(
                cleaned_chunk=cleaned_chunk,
                user_review_counts=user_review_counts,
                user_positive_counts=user_positive_counts,
                per_game_written_counts=per_game_written_counts,
                per_game_user_sets=per_game_user_sets,
            )

            cleaned_chunk.to_csv(
                run_paths["temp_output_path"],
                mode="a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            stats["chunks_processed"] = chunk_index
            stats["raw_row_count"] = raw_rows_scanned_pass_2
            stats["unique_reviews"] = len(tracked_review_ids)
            stats["unique_games"] = len(unique_game_ids)
            stats["unique_users"] = len(unique_user_ids)
            stats["raw_unique_author_steamid_count"] = len(raw_unique_user_ids)
            stats["language_distribution_top"] = dict(language_counter.most_common(10))
            write_progress_report(settings, stats, status="running")
            logger.info(
                "Processed chunk %s: raw=%s, kept=%s",
                chunk_index,
                len(raw_chunk),
                len(cleaned_chunk),
            )
            if int(stats["processed_row_count"]) >= settings.balanced_subset_max_processed_rows:
                break

        if first_chunk:
            pd.DataFrame(columns=NORMALIZED_COLUMN_ORDER).to_csv(
                run_paths["temp_output_path"],
                index=False,
            )

        os.replace(run_paths["temp_output_path"], run_paths["output_path"])
        stats["raw_row_count"] = raw_rows_scanned_pass_2
        stats["unique_reviews"] = len(tracked_review_ids)
        stats["unique_games"] = len(unique_game_ids)
        stats["unique_users"] = len(unique_user_ids)
        stats["raw_unique_author_steamid_count"] = len(raw_unique_user_ids)
        stats["language_distribution_top"] = dict(language_counter.most_common(10))
        finalize_stats(stats)

        eligible_users_estimate = estimate_eligible_users(
            user_review_counts=user_review_counts,
            user_positive_counts=user_positive_counts,
            settings=settings,
        )
        summary = build_balanced_subset_summary(
            settings=settings,
            dataset_path=dataset_path,
            selected_games=selected_games,
            raw_rows_scanned_pass_1=raw_rows_scanned_pass_1,
            raw_rows_scanned_pass_2=raw_rows_scanned_pass_2,
            stats=stats,
            user_id_mapping_status=str(stats["user_id_mapping_status"]),
            eligible_users_estimate=eligible_users_estimate,
            duration_seconds=float(stats["duration_seconds"]),
            started_at=started_at,
            finished_at=stats["finished_at"],
        )
        write_preprocessing_outputs(run_paths, summary, mode="balanced_subset")
        write_progress_payload(
            settings=settings,
            payload=build_progress_payload_from_summary(summary, status="completed"),
        )
        log_preprocessing_summary(
            logger,
            run_paths["summary_path"],
            summary,
            mode="balanced_subset",
        )
        return summary
    except PreprocessingInterrupted as exc:
        finalize_stats(stats)
        write_progress_payload(
            settings=settings,
            payload=build_progress_payload_from_stats(settings, stats, status="interrupted"),
        )
        logger.warning("%s", exc)
        logger.warning(
            "Resume from partial temp file is not implemented. Restart preprocessing to produce a consistent output file."
        )
        raise
    except Exception:
        finalize_stats(stats)
        write_progress_payload(
            settings=settings,
            payload=build_progress_payload_from_stats(settings, stats, status="failed"),
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def resolve_preprocessing_run_paths(settings: Settings, mode: str) -> dict[str, Path]:

    if mode == "debug":
        output_path = settings.reviews_clean_debug_path
        summary_path = settings.preprocessing_debug_summary_path
        summary_markdown_path = settings.preprocessing_debug_summary_markdown_path
        comparison_json_path = settings.raw_processed_subset_comparison_json_path.with_name(
            "raw_processed_debug_comparison.json"
        )
        comparison_markdown_path = settings.raw_processed_subset_comparison_markdown_path.with_name(
            "raw_processed_debug_comparison.md"
        )
    elif mode == "subset":
        output_path = settings.reviews_clean_subset_path
        summary_path = settings.preprocessing_subset_summary_path
        summary_markdown_path = settings.preprocessing_subset_summary_markdown_path
        comparison_json_path = settings.raw_processed_subset_comparison_json_path
        comparison_markdown_path = settings.raw_processed_subset_comparison_markdown_path
    elif mode == "balanced_subset":
        output_path = settings.reviews_clean_balanced_subset_path
        summary_path = settings.preprocessing_balanced_subset_summary_path
        summary_markdown_path = settings.preprocessing_balanced_subset_summary_markdown_path
        comparison_json_path = settings.raw_processed_balanced_subset_comparison_json_path
        comparison_markdown_path = settings.raw_processed_balanced_subset_comparison_markdown_path
    else:
        output_path = settings.reviews_clean_path
        summary_path = settings.preprocessing_summary_path
        summary_markdown_path = settings.preprocessing_summary_markdown_path
        comparison_json_path = settings.raw_processed_comparison_json_path
        comparison_markdown_path = settings.raw_processed_comparison_markdown_path

    return {
        "output_path": output_path,
        "temp_output_path": output_path.with_suffix(output_path.suffix + ".tmp"),
        "summary_path": summary_path,
        "summary_markdown_path": summary_markdown_path,
        "comparison_json_path": comparison_json_path,
        "comparison_markdown_path": comparison_markdown_path,
    }


def resolve_max_rows_for_mode(settings: Settings, mode: str) -> int | None:

    if mode == "debug":
        return (
            settings.max_rows_for_debug
            if settings.max_rows_for_debug is not None
            else DEBUG_MAX_ROWS_DEFAULT
        )
    if mode == "subset":
        return settings.subset_max_raw_rows
    if mode == "balanced_subset":
        return settings.balanced_subset_max_processed_rows
    return None


def prepare_preprocessing_outputs(
    settings: Settings,
    run_paths: dict[str, Path],
    *,
    mode: str,
) -> None:

    logger = get_logger()
    run_paths["output_path"].parent.mkdir(parents=True, exist_ok=True)
    run_paths["summary_path"].parent.mkdir(parents=True, exist_ok=True)
    if run_paths["temp_output_path"].exists():
        run_paths["temp_output_path"].unlink()
        logger.warning("Removed previous incomplete preprocessing temp file.")
        logger.warning(
            "Resume from partial temp file is not implemented. Restart preprocessing to produce a consistent output file."
        )
    for path in [
        run_paths["output_path"],
        run_paths["summary_path"],
        run_paths["summary_markdown_path"],
        run_paths["comparison_json_path"],
        run_paths["comparison_markdown_path"],
    ]:
        if path.exists():
            path.unlink()
    for path in [
        settings.preprocessing_progress_json_path,
        settings.preprocessing_progress_markdown_path,
    ]:
        if path.exists():
            path.unlink()
    if mode == "full":
        remove_stale_downstream_artifacts(settings)


def remove_stale_downstream_artifacts(settings: Settings) -> None:

    removed_any = False
    for attr_name in STALE_ARTIFACT_PATHS:
        path = getattr(settings, attr_name)
        if path.exists():
            path.unlink()
            removed_any = True
    message = "Removed stale processed/result artifacts dependent on reviews_clean.csv"
    get_logger().info(message)
    if not removed_any:
        get_logger().info("No dependent stale artifacts were present.")


def collect_raw_game_sampling_stats(
    *,
    dataset_path: Path,
    schema_report: dict[str, object],
    chunksize: int,
    minimal_columns: list[str],
) -> tuple[list[dict[str, object]], int]:

    canonical_to_raw = {
        normalize_raw_column_name(raw_name): raw_name
        for raw_name in schema_report.get("raw_columns", [])
    }
    usecols = [
        canonical_to_raw[column]
        for column in minimal_columns
        if column in canonical_to_raw
    ]
    game_stats: dict[str, dict[str, object]] = {}
    raw_rows_scanned = 0

    for raw_chunk in pd.read_csv(dataset_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        raw_rows_scanned += len(raw_chunk)
        normalized = normalize_chunk_dataframe(raw_chunk)
        normalized["game_id"] = normalize_identifier_series(normalized["game_id"])
        normalized["user_id"] = normalize_identifier_series(normalized["user_id"])
        normalized["review_text"] = normalized["review_text"].fillna("").astype(str)
        normalized["review_clean"] = normalized["review_text"].map(clean_review_text)
        normalized["recommended"] = normalized["recommended"].map(normalize_boolean_value)

        grouped = normalized.groupby(["game_id", "game_title"], sort=False)
        for (game_id, game_title), group in grouped:
            if not game_id:
                continue
            stats = game_stats.setdefault(
                str(game_id),
                {
                    "game_id": str(game_id),
                    "game_title": str(game_title),
                    "review_count": 0,
                    "positive_review_count": 0,
                    "negative_review_count": 0,
                    "non_empty_review_count": 0,
                    "reviews_with_user_id": 0,
                },
            )
            stats["review_count"] += int(len(group))
            positive_count = int(group["recommended"].fillna(False).sum())
            stats["positive_review_count"] += positive_count
            stats["negative_review_count"] += int(len(group) - positive_count)
            stats["non_empty_review_count"] += int(group["review_clean"].str.len().gt(0).sum())
            stats["reviews_with_user_id"] += int(group["user_id"].str.len().gt(0).sum())

    return list(game_stats.values()), raw_rows_scanned


def select_balanced_subset_games(
    game_stats: list[dict[str, object]],
    settings: Settings,
) -> list[dict[str, object]]:

    eligible = [
        record
        for record in game_stats
        if int(record["review_count"]) >= settings.balanced_subset_min_reviews_per_game
        and int(record["positive_review_count"]) >= settings.balanced_subset_min_positive_reviews_per_game
    ]
    eligible.sort(
        key=lambda record: (
            int(record["review_count"]),
            int(record["positive_review_count"]),
            str(record["game_title"]),
        ),
        reverse=True,
    )
    return eligible[: settings.balanced_subset_target_games]


def save_raw_game_sampling_stats(settings: Settings, records: list[dict[str, object]]) -> None:

    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "game_id",
                "game_title",
                "review_count",
                "positive_review_count",
                "negative_review_count",
                "non_empty_review_count",
                "reviews_with_user_id",
                "selected_for_balanced_subset",
            ]
        )
    frame.to_csv(settings.raw_game_sampling_stats_csv_path, index=False)
    settings.raw_game_sampling_stats_json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def enforce_per_game_row_cap(
    chunk: pd.DataFrame,
    *,
    per_game_written_counts: Counter[str],
    per_game_limit: int,
    remaining_total: int,
) -> pd.DataFrame:

    if chunk.empty or remaining_total <= 0:
        return chunk.iloc[0:0].copy()

    keep_indices: list[int] = []
    local_counts: Counter[str] = Counter()
    kept_total = 0
    for index, game_id in zip(chunk.index.tolist(), chunk["game_id"].tolist(), strict=False):
        game_id_str = str(game_id)
        if per_game_written_counts[game_id_str] + local_counts[game_id_str] >= per_game_limit:
            continue
        keep_indices.append(index)
        local_counts[game_id_str] += 1
        kept_total += 1
        if kept_total >= remaining_total:
            break
    return chunk.loc[keep_indices].copy()


def enforce_per_game_row_cap_after_cleaning(
    chunk: pd.DataFrame,
    *,
    per_game_written_counts: Counter[str],
    per_game_limit: int,
    remaining_total: int,
) -> pd.DataFrame:

    if chunk.empty or remaining_total <= 0:
        return chunk.iloc[0:0].copy()

    keep_positions: list[int] = []
    local_counts: Counter[str] = Counter()
    kept_total = 0
    for position, game_id in enumerate(chunk["game_id"].tolist()):
        game_id_str = str(game_id)
        if per_game_written_counts[game_id_str] + local_counts[game_id_str] >= per_game_limit:
            continue
        keep_positions.append(position)
        local_counts[game_id_str] += 1
        kept_total += 1
        if kept_total >= remaining_total:
            break
    return chunk.iloc[keep_positions].copy()


def update_balanced_subset_user_stats(
    *,
    cleaned_chunk: pd.DataFrame,
    user_review_counts: Counter[int | str],
    user_positive_counts: Counter[int | str],
    per_game_written_counts: Counter[str],
    per_game_user_sets: dict[str, set[int | str]],
) -> None:

    for row in cleaned_chunk.itertuples(index=False):
        game_id = str(row.game_id)
        user_id = str(row.user_id)
        per_game_written_counts[game_id] += 1
        if user_id:
            user_key = identifier_tracking_key(user_id)
            user_review_counts[user_key] += 1
            if bool(row.recommended):
                user_positive_counts[user_key] += 1
            if game_id in per_game_user_sets:
                per_game_user_sets[game_id].add(user_key)


def estimate_eligible_users(
    *,
    user_review_counts: Counter[int | str],
    user_positive_counts: Counter[int | str],
    settings: Settings,
) -> int:

    eligible = 0
    for user_id, review_count in user_review_counts.items():
        if (
            int(review_count) >= settings.min_user_reviews
            and int(user_positive_counts.get(user_id, 0)) >= settings.min_user_positive_reviews
        ):
            eligible += 1
    return eligible


def build_balanced_subset_summary(
    *,
    settings: Settings,
    dataset_path: Path,
    selected_games: list[dict[str, object]],
    raw_rows_scanned_pass_1: int,
    raw_rows_scanned_pass_2: int,
    stats: dict[str, object],
    user_id_mapping_status: str,
    eligible_users_estimate: int,
    duration_seconds: float,
    started_at: str,
    finished_at: str,
) -> dict[str, object]:

    return {
        "mode": "balanced_subset",
        "preprocess_mode": "chunked",
        "dataset_path": str(dataset_path),
        "raw_dataset_path": str(dataset_path),
        "target_games": int(settings.balanced_subset_target_games),
        "selected_games": int(len(selected_games)),
        "chunksize": int(settings.balanced_subset_chunksize),
        "max_rows_for_debug": int(settings.balanced_subset_max_processed_rows),
        "chunks_processed": int(stats.get("chunks_processed", 0)),
        "raw_rows_scanned_pass_1": int(raw_rows_scanned_pass_1),
        "raw_rows_scanned_pass_2": int(raw_rows_scanned_pass_2),
        "raw_row_count": int(raw_rows_scanned_pass_2),
        "processed_row_count": int(stats.get("processed_row_count", 0)),
        "processed_rows_written": int(stats.get("processed_row_count", 0)),
        "removed_empty_reviews": int(stats.get("removed_empty_reviews", 0)),
        "removed_short_reviews": int(stats.get("removed_short_reviews", 0)),
        "removed_duplicates": int(stats.get("removed_duplicates", 0)),
        "unique_games": int(stats.get("unique_games", 0)),
        "unique_users": int(stats.get("unique_users", 0)),
        "positive_reviews": int(stats.get("positive_reviews", 0)),
        "negative_reviews": int(stats.get("negative_reviews", 0)),
        "reviews_with_user_id": int(stats.get("reviews_with_user_id", 0)),
        "reviews_without_user_id": int(stats.get("reviews_without_user_id", 0)),
        "raw_author_steamid_present": bool(stats.get("raw_author_steamid_present", False)),
        "raw_author_steamid_non_empty_count": int(stats.get("raw_author_steamid_non_empty_count", 0)),
        "raw_unique_author_steamid_count": int(stats.get("raw_unique_author_steamid_count", 0)),
        "processed_user_id_non_empty_count": int(stats.get("processed_user_id_non_empty_count", 0)),
        "processed_user_id_empty_count": int(stats.get("processed_user_id_empty_count", 0)),
        "eligible_users_estimate": int(eligible_users_estimate),
        "min_reviews_per_game": int(settings.balanced_subset_min_reviews_per_game),
        "min_positive_reviews_per_game": int(settings.balanced_subset_min_positive_reviews_per_game),
        "max_processed_rows": int(settings.balanced_subset_max_processed_rows),
        "user_id_mapping_status": user_id_mapping_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": float(duration_seconds),
        "output_temp_path": stats.get("output_temp_path", ""),
        "final_output_path": stats.get("final_output_path", ""),
        "unique_reviews": int(stats.get("unique_reviews", 0)),
        "language_distribution_top": stats.get("language_distribution_top", {}),
        "column_name_warnings": list(stats.get("column_name_warnings", [])),
        "notes": BALANCED_SUBSET_NOTE,
        "selected_game_ids": [str(record["game_id"]) for record in selected_games],
    }


def build_progress_payload_from_stats(
    settings: Settings,
    stats: dict[str, object],
    *,
    status: str,
) -> dict[str, object]:

    duration_seconds_so_far = (
        round(time.monotonic() - float(stats["_started_monotonic"]), 2)
        if "_started_monotonic" in stats
        else float(stats.get("duration_seconds", 0.0))
    )
    return {
        "dataset_path": stats["dataset_path"],
        "mode": stats["mode"],
        "status": status,
        "chunksize": int(stats["chunksize"]),
        "chunks_processed": int(stats["chunks_processed"]),
        "raw_rows_processed": int(stats["raw_row_count"]),
        "processed_rows_written": int(stats["processed_row_count"]),
        "removed_empty_reviews": int(stats["removed_empty_reviews"]),
        "removed_short_reviews": int(stats["removed_short_reviews"]),
        "removed_duplicates": int(stats["removed_duplicates"]),
        "reviews_with_user_id": int(stats["reviews_with_user_id"]),
        "unique_games_count": int(stats["unique_games"]),
        "unique_users_count": int(stats["unique_users"]),
        "started_at": stats["started_at"],
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds_so_far": duration_seconds_so_far,
        "output_temp_path": stats["output_temp_path"],
        "final_output_path": stats["final_output_path"],
    }


def build_progress_payload_from_summary(
    summary: dict[str, object],
    *,
    status: str,
) -> dict[str, object]:

    return {
        "dataset_path": summary.get("dataset_path", summary.get("raw_dataset_path", "")),
        "mode": summary.get("mode", "unknown"),
        "status": status,
        "chunksize": int(summary.get("chunksize", 0)),
        "chunks_processed": int(summary.get("chunks_processed", 0)),
        "raw_rows_processed": int(summary.get("raw_row_count", summary.get("raw_rows_scanned_pass_2", 0))),
        "processed_rows_written": int(summary.get("processed_row_count", summary.get("processed_rows_written", 0))),
        "removed_empty_reviews": int(summary.get("removed_empty_reviews", 0)),
        "removed_short_reviews": int(summary.get("removed_short_reviews", 0)),
        "removed_duplicates": int(summary.get("removed_duplicates", 0)),
        "reviews_with_user_id": int(summary.get("reviews_with_user_id", 0)),
        "unique_games_count": int(summary.get("unique_games", 0)),
        "unique_users_count": int(summary.get("unique_users", 0)),
        "started_at": summary.get("started_at", ""),
        "last_updated_at": summary.get("finished_at", ""),
        "duration_seconds_so_far": float(summary.get("duration_seconds", 0.0)),
        "output_temp_path": summary.get("output_temp_path", ""),
        "final_output_path": summary.get("final_output_path", ""),
    }


def normalize_chunk_dataframe(raw_chunk: pd.DataFrame) -> pd.DataFrame:

    normalized: dict[str, object] = {}
    for source_name, target_name in SOURCE_TO_NORMALIZED_COLUMNS.items():
        normalized[target_name] = extract_source_series(raw_chunk, source_name)
    frame = pd.DataFrame(normalized, index=raw_chunk.index)
    for column in NORMALIZED_COLUMN_ORDER:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, NORMALIZED_COLUMN_ORDER].copy()


def extract_source_series(raw_chunk: pd.DataFrame, canonical_name: str) -> pd.Series:

    matched_columns = [
        raw_column
        for raw_column in raw_chunk.columns
        if normalize_raw_column_name(raw_column) == canonical_name
    ]
    if not matched_columns:
        return pd.Series([pd.NA] * len(raw_chunk), index=raw_chunk.index)
    return raw_chunk[matched_columns[0]].copy()


def clean_normalized_chunk(
    normalized_chunk: pd.DataFrame,
    *,
    settings: Settings,
    language_filter: str | None,
    seen_review_ids: set[int | str],
    tracked_review_ids: set[int | str],
    track_ids: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:

    chunk = normalized_chunk.copy()
    chunk["game_id"] = normalize_identifier_series(chunk["game_id"])
    chunk["review_id"] = normalize_identifier_series(chunk["review_id"])
    chunk["user_id"] = normalize_identifier_series(chunk["user_id"])
    chunk["game_title"] = chunk["game_title"].fillna("").astype(str).str.strip()
    chunk["review_text"] = chunk["review_text"].fillna("").astype(str)
    chunk["review_clean"] = chunk["review_text"].map(clean_review_text)
    chunk["language"] = chunk["language"].fillna("").astype(str).map(normalize_language_value)

    chunk_stats = {
        "removed_empty_reviews": 0,
        "removed_short_reviews": 0,
        "removed_duplicates": 0,
        "removed_invalid_recommended": 0,
    }

    before_empty_filter = len(chunk)
    chunk = chunk[chunk["review_clean"].str.len() > 0].copy()
    chunk_stats["removed_empty_reviews"] = int(before_empty_filter - len(chunk))

    for column in BOOL_COLUMNS:
        chunk[column] = chunk[column].map(normalize_boolean_value)
    for column in NUMERIC_COLUMNS:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    before_invalid_recommended = len(chunk)
    chunk = chunk.dropna(subset=["recommended"]).copy()
    chunk_stats["removed_invalid_recommended"] = int(
        before_invalid_recommended - len(chunk)
    )
    chunk["recommended"] = chunk["recommended"].astype(bool)

    if language_filter:
        chunk = chunk[chunk["language"] == language_filter].copy()

    chunk["review_clean"] = chunk["review_clean"].map(
        lambda text: truncate_review_text(text, settings.max_review_chars)
    )
    before_short_filter = len(chunk)
    chunk = chunk[chunk["review_clean"].str.len() >= settings.min_review_chars].copy()
    chunk_stats["removed_short_reviews"] = int(before_short_filter - len(chunk))

    chunk = filter_duplicate_review_ids(
        chunk,
        seen_review_ids=seen_review_ids,
        tracked_review_ids=tracked_review_ids,
        track_ids=track_ids,
        chunk_stats=chunk_stats,
    )

    chunk["votes_helpful"] = chunk["votes_helpful"].fillna(0).clip(lower=0).astype(int)
    chunk["votes_funny"] = chunk["votes_funny"].fillna(0).clip(lower=0).astype(int)
    for column in NUMERIC_COLUMNS:
        if column not in {"votes_helpful", "votes_funny"}:
            if column in INTEGER_NUMERIC_COLUMNS:
                chunk[column] = chunk[column].fillna(0).astype(int)
            else:
                chunk[column] = chunk[column].astype(float)

    for column in TIMESTAMP_COLUMNS:
        chunk[column] = normalize_timestamp_series(chunk[column])

    chunk["review_text"] = chunk["review_text"].astype(str)
    chunk["review_clean"] = chunk["review_clean"].astype(str)
    chunk["user_id"] = normalize_identifier_series(chunk["user_id"])
    chunk = chunk.loc[:, NORMALIZED_COLUMN_ORDER].reset_index(drop=True)
    return chunk, chunk_stats


def filter_duplicate_review_ids(
    chunk: pd.DataFrame,
    *,
    seen_review_ids: set[int | str],
    tracked_review_ids: set[int | str],
    track_ids: bool,
    chunk_stats: dict[str, int],
) -> pd.DataFrame:

    if chunk.empty:
        return chunk

    keep_mask: list[bool] = []
    for review_id in chunk["review_id"].tolist():
        if not review_id:
            keep_mask.append(True)
            continue
        key = identifier_tracking_key(review_id)
        if key in seen_review_ids:
            keep_mask.append(False)
            continue
        seen_review_ids.add(key)
        if track_ids:
            tracked_review_ids.add(key)
        keep_mask.append(True)

    filtered = chunk.loc[keep_mask].copy()
    chunk_stats["removed_duplicates"] = int(len(chunk) - len(filtered))
    return filtered


def update_exact_trackers(
    *,
    normalized_chunk: pd.DataFrame,
    cleaned_chunk: pd.DataFrame,
    raw_unique_user_ids: set[int | str],
    unique_game_ids: set[int | str],
    unique_user_ids: set[int | str],
    language_counter: Counter[str],
) -> None:

    for user_id in normalize_identifier_series(normalized_chunk["user_id"]).tolist():
        if user_id:
            raw_unique_user_ids.add(identifier_tracking_key(user_id))
    for game_id in cleaned_chunk["game_id"].tolist():
        if game_id:
            unique_game_ids.add(identifier_tracking_key(game_id))
    for user_id in cleaned_chunk["user_id"].tolist():
        if user_id:
            unique_user_ids.add(identifier_tracking_key(user_id))
    language_counter.update(
        language
        for language in cleaned_chunk["language"].tolist()
        if language
    )


def init_preprocessing_stats(
    *,
    settings: Settings,
    dataset_path: Path,
    max_rows_for_debug: int | None,
    mode: str,
    final_output_path: Path,
    temp_output_path: Path,
) -> dict[str, object]:

    started_at = datetime.now(timezone.utc)
    return {
        "dataset_path": str(dataset_path),
        "preprocess_mode": "chunked",
        "mode": mode,
        "chunksize": settings.preprocess_chunksize,
        "max_rows_for_debug": max_rows_for_debug,
        "chunks_processed": 0,
        "raw_row_count": 0,
        "processed_row_count": 0,
        "removed_empty_reviews": 0,
        "removed_short_reviews": 0,
        "removed_duplicates": 0,
        "removed_invalid_recommended": 0,
        "unique_reviews": 0,
        "unique_games": 0,
        "unique_users": 0,
        "positive_reviews": 0,
        "negative_reviews": 0,
        "reviews_with_user_id": 0,
        "reviews_without_user_id": 0,
        "reviews_with_playtime_forever": 0,
        "reviews_with_playtime_at_review": 0,
        "reviews_with_weighted_vote_score": 0,
        "language_distribution_top": {},
        "raw_author_steamid_present": False,
        "raw_author_steamid_non_empty_count": 0,
        "raw_unique_author_steamid_count": 0,
        "processed_user_id_non_empty_count": 0,
        "processed_user_id_empty_count": 0,
        "user_id_mapping_status": "source_column_missing",
        "started_at": started_at.isoformat(),
        "finished_at": "",
        "duration_seconds": 0,
        "output_temp_path": str(temp_output_path.relative_to(settings.project_root)),
        "final_output_path": str(final_output_path.relative_to(settings.project_root)),
        "min_review_chars": int(settings.min_review_chars),
        "max_review_chars": int(settings.max_review_chars),
        "column_name_warnings": [],
        "_started_monotonic": time.monotonic(),
    }


def update_stats_from_chunk(
    stats: dict[str, object],
    normalized_chunk: pd.DataFrame,
    cleaned_chunk: pd.DataFrame,
    chunk_stats: dict[str, int],
) -> None:

    stats["raw_author_steamid_non_empty_count"] += int(
        normalize_identifier_series(normalized_chunk["user_id"]).str.len().gt(0).sum()
    )
    stats["processed_row_count"] += int(len(cleaned_chunk))
    stats["removed_empty_reviews"] += int(chunk_stats["removed_empty_reviews"])
    stats["removed_short_reviews"] += int(chunk_stats["removed_short_reviews"])
    stats["removed_duplicates"] += int(chunk_stats["removed_duplicates"])
    stats["removed_invalid_recommended"] += int(chunk_stats["removed_invalid_recommended"])
    stats["positive_reviews"] += int(cleaned_chunk["recommended"].sum())
    stats["negative_reviews"] += int((~cleaned_chunk["recommended"]).sum())
    reviews_with_user_id = int(cleaned_chunk["user_id"].str.len().gt(0).sum())
    stats["reviews_with_user_id"] += reviews_with_user_id
    stats["reviews_without_user_id"] += int(len(cleaned_chunk) - reviews_with_user_id)
    stats["reviews_with_playtime_forever"] += int(cleaned_chunk["playtime_forever"].fillna(0).gt(0).sum())
    stats["reviews_with_playtime_at_review"] += int(cleaned_chunk["playtime_at_review"].fillna(0).gt(0).sum())
    stats["reviews_with_weighted_vote_score"] += int(cleaned_chunk["weighted_vote_score"].notna().sum())


def finalize_stats(stats: dict[str, object]) -> None:

    stats["processed_user_id_non_empty_count"] = int(stats["reviews_with_user_id"])
    stats["processed_user_id_empty_count"] = int(stats["reviews_without_user_id"])
    stats["user_id_mapping_status"] = determine_user_id_mapping_status(
        raw_author_steamid_present=bool(stats["raw_author_steamid_present"]),
        raw_author_steamid_non_empty_count=int(stats["raw_author_steamid_non_empty_count"]),
        processed_user_id_non_empty_count=int(stats["processed_user_id_non_empty_count"]),
    )
    finished_at = datetime.now(timezone.utc)
    stats["finished_at"] = finished_at.isoformat()
    started_monotonic = stats.pop("_started_monotonic", None)
    if started_monotonic is None:
        stats["duration_seconds"] = round(float(stats.get("duration_seconds", 0.0)), 2)
        return
    stats["duration_seconds"] = round(
        time.monotonic() - float(started_monotonic),
        2,
    )


def build_preprocessing_summary(stats: dict[str, object]) -> dict[str, object]:

    return {
        "dataset_path": stats["dataset_path"],
        "preprocess_mode": stats["preprocess_mode"],
        "mode": stats["mode"],
        "chunksize": int(stats["chunksize"]),
        "max_rows_for_debug": stats["max_rows_for_debug"],
        "chunks_processed": int(stats["chunks_processed"]),
        "raw_row_count": int(stats["raw_row_count"]),
        "processed_row_count": int(stats["processed_row_count"]),
        "final_row_count": int(stats["processed_row_count"]),
        "removed_empty_reviews": int(stats["removed_empty_reviews"]),
        "removed_short_reviews": int(stats["removed_short_reviews"]),
        "removed_duplicates": int(stats["removed_duplicates"]),
        "removed_invalid_recommended": int(stats["removed_invalid_recommended"]),
        "unique_reviews": int(stats["unique_reviews"]),
        "unique_games": int(stats["unique_games"]),
        "unique_users": int(stats["unique_users"]),
        "positive_reviews": int(stats["positive_reviews"]),
        "negative_reviews": int(stats["negative_reviews"]),
        "reviews_with_user_id": int(stats["reviews_with_user_id"]),
        "reviews_without_user_id": int(stats["reviews_without_user_id"]),
        "reviews_with_playtime_forever": int(stats["reviews_with_playtime_forever"]),
        "reviews_with_playtime_at_review": int(stats["reviews_with_playtime_at_review"]),
        "reviews_with_weighted_vote_score": int(stats["reviews_with_weighted_vote_score"]),
        "language_distribution_top": stats["language_distribution_top"],
        "raw_author_steamid_present": bool(stats["raw_author_steamid_present"]),
        "raw_author_steamid_non_empty_count": int(stats["raw_author_steamid_non_empty_count"]),
        "raw_unique_author_steamid_count": int(stats["raw_unique_author_steamid_count"]),
        "processed_user_id_non_empty_count": int(stats["processed_user_id_non_empty_count"]),
        "processed_user_id_empty_count": int(stats["processed_user_id_empty_count"]),
        "user_id_mapping_status": str(stats["user_id_mapping_status"]),
        "started_at": stats["started_at"],
        "finished_at": stats["finished_at"],
        "duration_seconds": float(stats["duration_seconds"]),
        "output_temp_path": stats["output_temp_path"],
        "final_output_path": stats["final_output_path"],
        "min_review_chars": int(stats["min_review_chars"]),
        "max_review_chars": int(stats["max_review_chars"]),
        "column_name_warnings": list(stats["column_name_warnings"]),
        "matched_user_id_source_column": stats.get("matched_user_id_source_column", ""),
        "notes": (
            PREPROCESS_SUMMARY_NOTE
            if stats["mode"] == "subset"
            else BALANCED_SUBSET_NOTE
            if stats["mode"] == "balanced_subset"
            else ""
        ),
    }


def write_preprocessing_outputs(
    run_paths: dict[str, Path],
    summary: dict[str, object],
    *,
    mode: str,
) -> None:

    write_preprocessing_summary(
        summary_path=run_paths["summary_path"],
        markdown_path=run_paths["summary_markdown_path"],
        summary=summary,
    )
    if mode != "debug":
        write_raw_processed_comparison(
            json_path=run_paths["comparison_json_path"],
            markdown_path=run_paths["comparison_markdown_path"],
            summary=summary,
        )


def write_preprocessing_summary(
    *,
    summary_path: Path,
    markdown_path: Path,
    summary: dict[str, object],
) -> None:

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    markdown_path.write_text(
        build_preprocessing_summary_markdown(summary),
        encoding="utf-8",
    )


def write_progress_report(
    settings: Settings,
    stats: dict[str, object],
    *,
    status: str,
) -> None:

    now = datetime.now(timezone.utc).isoformat()
    if "_started_monotonic" in stats:
        duration_seconds_so_far = round(
            time.monotonic() - float(stats["_started_monotonic"]),
            2,
        )
    else:
        duration_seconds_so_far = float(stats.get("duration_seconds", 0.0))
    payload = {
        "dataset_path": stats["dataset_path"],
        "mode": stats["mode"],
        "status": status,
        "chunksize": int(stats["chunksize"]),
        "chunks_processed": int(stats["chunks_processed"]),
        "raw_rows_processed": int(stats["raw_row_count"]),
        "processed_rows_written": int(stats["processed_row_count"]),
        "removed_empty_reviews": int(stats["removed_empty_reviews"]),
        "removed_short_reviews": int(stats["removed_short_reviews"]),
        "removed_duplicates": int(stats["removed_duplicates"]),
        "reviews_with_user_id": int(stats["reviews_with_user_id"]),
        "unique_games_count": int(stats["unique_games"]),
        "unique_users_count": int(stats["unique_users"]),
        "started_at": stats["started_at"],
        "last_updated_at": now,
        "duration_seconds_so_far": duration_seconds_so_far,
        "output_temp_path": stats["output_temp_path"],
        "final_output_path": stats["final_output_path"],
    }
    settings.preprocessing_progress_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.preprocessing_progress_markdown_path.write_text(
        build_preprocessing_progress_markdown(payload),
        encoding="utf-8",
    )


def write_progress_payload(
    *,
    settings: Settings,
    payload: dict[str, object],
) -> None:

    settings.preprocessing_progress_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.preprocessing_progress_markdown_path.write_text(
        build_preprocessing_progress_markdown(payload),
        encoding="utf-8",
    )


def build_preprocessing_progress_markdown(progress: dict[str, object]) -> str:

    return "\n".join(
        [
            "# Preprocessing Progress",
            "",
            f"- Dataset path: `{progress['dataset_path']}`",
            f"- Mode: {progress['mode']}",
            f"- Status: {progress['status']}",
            f"- Chunk size: {progress['chunksize']}",
            f"- Chunks processed: {progress['chunks_processed']}",
            f"- Raw rows processed: {progress['raw_rows_processed']}",
            f"- Processed rows written: {progress['processed_rows_written']}",
            f"- Removed empty reviews: {progress['removed_empty_reviews']}",
            f"- Removed short reviews: {progress['removed_short_reviews']}",
            f"- Removed duplicates: {progress['removed_duplicates']}",
            f"- Reviews with user_id: {progress['reviews_with_user_id']}",
            f"- Unique games so far: {progress['unique_games_count']}",
            f"- Unique users so far: {progress['unique_users_count']}",
            f"- Started at: {progress['started_at']}",
            f"- Last updated at: {progress['last_updated_at']}",
            f"- Duration so far (s): {progress['duration_seconds_so_far']}",
            f"- Output temp path: `{progress['output_temp_path']}`",
            f"- Final output path: `{progress['final_output_path']}`",
            "",
        ]
    ) + "\n"


def write_raw_processed_comparison(
    *,
    json_path: Path,
    markdown_path: Path,
    summary: dict[str, object],
) -> None:

    comparison = {
        "raw_row_count": int(summary["raw_row_count"]),
        "processed_row_count": int(summary["processed_row_count"]),
        "raw_author_steamid_non_empty_count": int(summary["raw_author_steamid_non_empty_count"]),
        "processed_user_id_non_empty_count": int(summary["processed_user_id_non_empty_count"]),
        "raw_unique_author_steamid_count": int(summary["raw_unique_author_steamid_count"]),
        "processed_unique_user_id_count": int(summary["unique_users"]),
        "user_ids_preserved": bool(
            int(summary["raw_author_steamid_non_empty_count"]) == 0
            or int(summary["processed_user_id_non_empty_count"]) > 0
        ),
        "warning": (
            "Processed user_id values were lost during preprocessing."
            if summary["user_id_mapping_status"] == "mapping_failed"
            else ""
        ),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
    markdown_path.write_text(
        build_raw_processed_comparison_markdown(comparison),
        encoding="utf-8",
    )


def build_preprocessing_summary_markdown(summary: dict[str, object]) -> str:

    language_lines = [
        f"- {language}: {count}"
        for language, count in summary["language_distribution_top"].items()
    ] or ["- none"]
    warning_lines = [
        f"- {warning}" for warning in summary.get("column_name_warnings", [])
    ] or ["- none"]
    return "\n".join(
        [
            "# Preprocessing Summary",
            "",
            f"- Dataset path: `{summary['dataset_path']}`",
            f"- Preprocess mode: {summary['preprocess_mode']}",
            f"- Mode: {summary['mode']}",
            f"- Chunk size: {summary['chunksize']}",
            f"- Max rows for debug: {summary['max_rows_for_debug']}",
            f"- Chunks processed: {summary['chunks_processed']}",
            f"- Raw rows processed: {summary['raw_row_count']}",
            f"- Processed rows kept: {summary['processed_row_count']}",
            f"- Removed empty reviews: {summary['removed_empty_reviews']}",
            f"- Removed short reviews: {summary['removed_short_reviews']}",
            f"- Removed duplicates: {summary['removed_duplicates']}",
            f"- Unique reviews: {summary['unique_reviews']}",
            f"- Unique games: {summary['unique_games']}",
            f"- Unique users: {summary['unique_users']}",
            f"- Reviews with user id: {summary['reviews_with_user_id']}",
            f"- Reviews without user id: {summary['reviews_without_user_id']}",
            f"- Positive reviews: {summary['positive_reviews']}",
            f"- Negative reviews: {summary['negative_reviews']}",
            f"- Raw `author.steamid` present: {summary['raw_author_steamid_present']}",
            f"- Raw non-empty `author.steamid` count: {summary['raw_author_steamid_non_empty_count']}",
            f"- Processed non-empty `user_id` count: {summary['processed_user_id_non_empty_count']}",
            f"- `user_id` mapping status: {summary['user_id_mapping_status']}",
            f"- Started at: {summary['started_at']}",
            f"- Finished at: {summary['finished_at']}",
            f"- Duration (s): {summary['duration_seconds']}",
            f"- Output temp path: `{summary['output_temp_path']}`",
            f"- Final output path: `{summary['final_output_path']}`",
            "",
            "## Column Name Diagnostics",
            *warning_lines,
            "",
            "## Top Languages",
            *language_lines,
            "",
            "## Notes",
            f"- {summary['notes']}" if summary.get("notes") else "- none",
            "",
        ]
    ) + "\n"


def build_raw_processed_comparison_markdown(report: dict[str, object]) -> str:

    warning_lines = [f"- {report['warning']}"] if report["warning"] else ["- none"]
    return "\n".join(
        [
            "# Raw to Processed Comparison",
            "",
            f"- Raw row count: {report['raw_row_count']}",
            f"- Processed row count: {report['processed_row_count']}",
            f"- Raw non-empty `author.steamid` count: {report['raw_author_steamid_non_empty_count']}",
            f"- Processed non-empty `user_id` count: {report['processed_user_id_non_empty_count']}",
            f"- Raw unique `author.steamid` count: {report['raw_unique_author_steamid_count']}",
            f"- Processed unique `user_id` count: {report['processed_unique_user_id_count']}",
            f"- User IDs preserved: {report['user_ids_preserved']}",
            "",
            "## Warnings",
            *warning_lines,
            "",
        ]
    ) + "\n"


def clean_review_text(text: str) -> str:

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_review_text(text: str, max_chars: int) -> str:

    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def normalize_boolean_value(value: object) -> bool | None:

    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in BOOL_TRUE_VALUES:
        return True
    if normalized in BOOL_FALSE_VALUES:
        return False
    return None


def normalize_language_filter(value: str | None) -> str | None:

    if value is None:
        return None
    normalized = LANGUAGE_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise ValueError(
            "LANGUAGE_FILTER must be one of: english, russian, en, eng, ru, rus."
        )
    return normalized


def normalize_language_value(value: str) -> str:

    normalized = value.strip().lower()
    return LANGUAGE_ALIASES.get(normalized, normalized)


def normalize_timestamp_series(values: pd.Series) -> pd.Series:

    numeric = pd.to_numeric(values, errors="coerce")
    parsed_numeric = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    fallback_source = values.where(numeric.isna())
    parsed_fallback = pd.to_datetime(fallback_source, errors="coerce", utc=True)
    combined = parsed_numeric.fillna(parsed_fallback)
    combined = combined.dt.tz_localize(None)
    formatted = combined.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return formatted.fillna("")


def determine_user_id_mapping_status(
    *,
    raw_author_steamid_present: bool,
    raw_author_steamid_non_empty_count: int,
    processed_user_id_non_empty_count: int,
) -> str:

    if not raw_author_steamid_present:
        return "source_column_missing"
    if raw_author_steamid_non_empty_count == 0:
        return "source_column_empty"
    if processed_user_id_non_empty_count == 0:
        return "mapping_failed"
    return "ok"


def identifier_tracking_key(value: str) -> int | str:

    if value.isdigit():
        return int(value)
    return value


def validate_preprocessing_settings(settings: Settings) -> None:

    if settings.min_review_chars < 0:
        raise ValueError("MIN_REVIEW_CHARS must be non-negative.")
    if settings.max_review_chars <= 0:
        raise ValueError("MAX_REVIEW_CHARS must be positive.")
    if settings.min_review_chars > settings.max_review_chars:
        raise ValueError("MIN_REVIEW_CHARS cannot be greater than MAX_REVIEW_CHARS.")
    if settings.preprocess_chunksize <= 0:
        raise ValueError("PREPROCESS_CHUNKSIZE must be positive.")


def log_preprocessing_summary(
    logger,
    summary_path: Path,
    summary: dict[str, object],
    *,
    mode: str,
) -> None:

    label = (
        "Preprocessing debug summary"
        if mode == "debug"
        else "Preprocessing subset summary"
        if mode == "subset"
        else "Preprocessing summary"
    )
    logger.info("%s:", label)
    logger.info("  Raw row count: %s", summary["raw_row_count"])
    logger.info("  Processed row count: %s", summary["processed_row_count"])
    logger.info("  Unique users: %s", summary["unique_users"])
    logger.info("  Reviews with user id: %s", summary["reviews_with_user_id"])
    logger.info("  User id mapping status: %s", summary["user_id_mapping_status"])
    logger.info("  Duration (s): %s", summary["duration_seconds"])
    logger.info("Saved preprocessing summary to %s", summary_path)


def print_preprocessing_status(settings: Settings) -> bool:

    if not settings.preprocessing_progress_json_path.exists():
        print("No preprocessing progress file found.")
        return False

    progress = json.loads(
        settings.preprocessing_progress_json_path.read_text(encoding="utf-8")
    )
    print(f"Preprocessing status: {progress['status']}")
    print(f"Chunks processed: {progress['chunks_processed']}")
    print(f"Raw rows processed: {progress['raw_rows_processed']}")
    print(f"Processed rows written: {progress['processed_rows_written']}")
    print(f"Reviews with user_id: {progress['reviews_with_user_id']}")
    print(f"Elapsed time: {progress['duration_seconds_so_far']} seconds")
    print(f"Output temp path: {progress['output_temp_path']}")
    print(f"Final output path: {progress['final_output_path']}")
    return True
