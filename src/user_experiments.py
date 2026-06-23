"""Lightweight user-based recommendation helpers for the full Steam Reviews dataset."""

from __future__ import annotations

from collections import Counter
import math
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings
from src.game_card_builder import GameCard
from src.utils import get_logger, read_jsonl, write_jsonl


def load_clean_reviews_for_user_mode(settings: Settings) -> pd.DataFrame | None:
    """Load cleaned reviews and return None when user mode is unavailable."""

    logger = get_logger()
    active_path = settings.active_processed_reviews_path
    if not active_path.exists():
        logger.warning(
            "Cleaned reviews not found at %s. Run `./.venv/bin/python main.py --step preprocess` first.",
            active_path,
        )
        return None

    reviews_df = pd.read_csv(active_path)
    raw_inspection = load_json_if_exists(settings.raw_dataset_inspection_json_path)
    preprocessing_summary_path = settings.preprocessing_summary_path
    if active_path == settings.reviews_clean_subset_path:
        preprocessing_summary_path = settings.preprocessing_subset_summary_path
    elif active_path == settings.reviews_clean_debug_path:
        preprocessing_summary_path = settings.preprocessing_debug_summary_path
    preprocessing_summary = load_json_if_exists(preprocessing_summary_path)
    if (
        active_path == settings.reviews_clean_path
        and raw_inspection
        and preprocessing_summary
    ):
        if int(raw_inspection.get("raw_row_count", 0)) != int(
            preprocessing_summary.get("raw_row_count", 0)
        ):
            logger.warning(
                "User-based mode is unavailable because cleaned reviews are out of sync with the current raw dataset. Re-run preprocessing first."
            )
            return None
    if "user_id" not in reviews_df.columns:
        logger.warning("User-based mode is unavailable because user_id is not present in cleaned reviews.")
        return None

    non_empty_user_ids = reviews_df["user_id"].fillna("").astype(str).str.len() > 0
    if not non_empty_user_ids.any():
        logger.warning("User-based mode is unavailable because no user_id values are present.")
        return None
    return reviews_df


def get_active_user_splits_path(settings: Settings) -> Path:
    """Return the split file that should be used for the current workflow."""

    if bool(getattr(settings, "use_pilot_splits", False)):
        return getattr(settings, "pilot_splits_path", settings.user_splits_path)
    return settings.user_splits_path


def get_active_user_split_mode(settings: Settings) -> str:
    """Return the active split mode label for the current workflow."""

    return "pilot" if bool(getattr(settings, "use_pilot_splits", False)) else "main"


def load_game_cards_by_id(settings: Settings) -> dict[str, GameCard]:
    """Load game cards into an id lookup when available."""

    if not settings.game_cards_path.exists():
        return {}
    return {card.game_id: card for card in (GameCard(**record) for record in read_jsonl(settings.game_cards_path))}


def build_candidate_game_ids_for_user(
    *,
    user_reviews: pd.DataFrame,
    holdout_game_ids: list[str],
    game_cards_by_id: dict[str, GameCard],
    settings: Settings,
    vectorizer: TfidfVectorizer | None = None,
) -> list[str]:
    """Build a candidate pool that always contains the holdout games."""

    all_game_ids = list(game_cards_by_id.keys()) or user_reviews["game_id"].astype(str).drop_duplicates().tolist()
    reviewed_game_ids = user_reviews["game_id"].astype(str).drop_duplicates().tolist()
    blocked_ids = set(reviewed_game_ids)
    holdout_set = {str(game_id) for game_id in holdout_game_ids if str(game_id)}
    blocked_ids -= holdout_set

    candidate_ids: list[str] = []
    for game_id in holdout_game_ids:
        game_id = str(game_id)
        if game_id and game_id in all_game_ids and game_id not in candidate_ids:
            candidate_ids.append(game_id)

    remaining_ids = [
        game_id
        for game_id in all_game_ids
        if game_id not in blocked_ids and game_id not in candidate_ids
    ]

    similar_ids = score_candidate_games_from_train(
        train_game_ids=[
            str(game_id)
            for game_id in user_reviews.loc[user_reviews["recommended"], "game_id"].astype(str).drop_duplicates().tolist()
            if str(game_id) not in holdout_set
        ],
        candidate_game_ids=remaining_ids,
        game_cards_by_id=game_cards_by_id,
        vectorizer=vectorizer,
    )
    similar_ids = [game_id for game_id, _ in similar_ids if game_id not in candidate_ids]

    for game_id in similar_ids:
        if len(candidate_ids) >= settings.user_candidate_pool_size:
            break
        candidate_ids.append(game_id)

    if len(candidate_ids) < settings.user_candidate_pool_size:
        filler_ids = [game_id for game_id in remaining_ids if game_id not in candidate_ids]
        rng = random.Random(f"{settings.random_seed}:{','.join(holdout_game_ids)}")
        rng.shuffle(filler_ids)
        for game_id in filler_ids:
            if len(candidate_ids) >= settings.user_candidate_pool_size:
                break
            candidate_ids.append(game_id)

    return candidate_ids[: settings.user_candidate_pool_size]


def score_candidate_games_from_train(
    *,
    train_game_ids: list[str],
    candidate_game_ids: list[str],
    game_cards_by_id: dict[str, GameCard],
    vectorizer: TfidfVectorizer | None = None,
) -> list[tuple[str, float]]:
    """Score candidate games against the user's positive training games."""

    candidate_ids = [game_id for game_id in candidate_game_ids if game_id in game_cards_by_id]
    if not candidate_ids:
        return []
    train_ids = [game_id for game_id in train_game_ids if game_id in game_cards_by_id]
    if not train_ids:
        return [(game_id, 0.0) for game_id in candidate_ids]

    cards = [game_cards_by_id[game_id] for game_id in candidate_ids]
    if vectorizer is None:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
        vectorizer.fit([card.game_card_text for card in game_cards_by_id.values()])
    matrix = vectorizer.transform([card.game_card_text for card in cards])
    query_text = "\n\n".join(game_cards_by_id[game_id].game_card_text for game_id in train_ids)
    scores = cosine_similarity(vectorizer.transform([query_text]), matrix).flatten()
    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [(game_id, float(score)) for game_id, score in ranked]


def build_user_split_record(
    *,
    user_id: str,
    masked_user_id: str,
    user_reviews: pd.DataFrame,
    game_cards_by_id: dict[str, GameCard],
    settings: Settings,
    vectorizer: TfidfVectorizer | None = None,
) -> dict[str, object] | None:
    """Build one split record with an explicit candidate pool."""

    ordered_reviews = user_reviews.sort_values(
        by=["timestamp_created", "review_id"],
        ascending=[True, True],
    ).copy()
    unique_game_history = ordered_reviews.drop_duplicates(subset=["game_id"], keep="last")["game_id"].astype(str).tolist()
    if len(unique_game_history) <= settings.user_holdout_count:
        return None

    holdout_game_ids = unique_game_history[-settings.user_holdout_count :]
    train_game_ids = unique_game_history[:-settings.user_holdout_count]
    if len(train_game_ids) < max(settings.min_user_reviews - settings.user_holdout_count, 1):
        return None

    positive_history_ids = (
        ordered_reviews.loc[ordered_reviews["recommended"], "game_id"].astype(str).drop_duplicates().tolist()
    )
    if len(positive_history_ids) < settings.min_user_positive_reviews:
        return None

    candidate_game_ids = build_candidate_game_ids_for_user(
        user_reviews=ordered_reviews,
        holdout_game_ids=holdout_game_ids,
        game_cards_by_id=game_cards_by_id,
        settings=settings,
        vectorizer=vectorizer,
    )

    ranked_candidates = score_candidate_games_from_train(
        train_game_ids=train_game_ids,
        candidate_game_ids=candidate_game_ids,
        game_cards_by_id=game_cards_by_id,
        vectorizer=vectorizer,
    )
    baseline_rank = next(
        (index for index, (game_id, _) in enumerate(ranked_candidates, start=1) if game_id in set(holdout_game_ids)),
        None,
    )
    history_game_ids = [
        game_id
        for game_id in ordered_reviews["game_id"].astype(str).drop_duplicates().tolist()
        if game_id not in set(holdout_game_ids)
    ]
    return {
        "user_id": user_id,
        "masked_user_id": masked_user_id,
        "history_game_ids": history_game_ids,
        "train_game_ids": train_game_ids,
        "holdout_game_ids": holdout_game_ids,
        "ground_truth_game_ids": holdout_game_ids,
        "candidate_game_ids": candidate_game_ids,
        "candidate_pool_size": len(candidate_game_ids),
        "baseline_best_holdout_rank": format_rank_value(baseline_rank),
        "baseline_hit_at_10": int(is_hit_at_k(baseline_rank, 10)),
        "holdout_in_candidate_pool": bool(set(holdout_game_ids) & set(candidate_game_ids)),
        "candidate_pool_too_small": bool(len(candidate_game_ids) < settings.user_candidate_pool_size),
        "user_history_too_small": bool(len(unique_game_history) < settings.min_user_reviews),
    }


def build_user_split_diagnostics_frame(
    settings: Settings,
    splits: list[dict[str, object]],
) -> pd.DataFrame:
    """Build diagnostics for user evaluation splits."""

    rows: list[dict[str, object]] = []
    for split in splits:
        holdout_ids = [str(game_id) for game_id in split.get("holdout_game_ids", split.get("ground_truth_game_ids", []))]
        candidate_ids = [str(game_id) for game_id in split.get("candidate_game_ids", [])]
        history_ids = [str(game_id) for game_id in split.get("history_game_ids", split.get("train_game_ids", []))]
        baseline_rank_raw = split.get("baseline_best_holdout_rank", "not_found")
        baseline_rank = None
        if baseline_rank_raw not in {None, "", "not_found"}:
            try:
                baseline_rank = int(baseline_rank_raw)
            except Exception:
                baseline_rank = None
        holdout_in_candidate_pool = bool(set(holdout_ids) & set(candidate_ids))
        reasons: list[str] = []
        if not holdout_ids:
            reasons.append("user_history_too_small")
        if holdout_ids and not holdout_in_candidate_pool:
            reasons.append("holdout_missing_from_candidate_pool")
        if int(split.get("candidate_pool_size", len(candidate_ids))) < settings.user_candidate_pool_size:
            reasons.append("candidate_pool_too_small")
        if baseline_rank is None:
            reasons.append("no_baseline_recommendations")
        elif baseline_rank > settings.top_k:
            reasons.append("holdout_not_in_top_k")
        if int(split.get("user_history_too_small", False)) or len(history_ids) < settings.min_user_reviews:
            reasons.append("user_history_too_small")
        rows.append(
            {
                "masked_user_id": str(split.get("masked_user_id", "")),
                "holdout_count": len(holdout_ids),
                "candidate_pool_size": int(split.get("candidate_pool_size", len(candidate_ids))),
                "holdout_in_candidate_pool": holdout_in_candidate_pool,
                "baseline_best_holdout_rank": baseline_rank_raw,
                "baseline_hit_at_10": int(split.get("baseline_hit_at_10", int(is_hit_at_k(baseline_rank, 10)))),
                "holdout_in_baseline_top_10": bool(baseline_rank is not None and baseline_rank <= 10),
                "holdout_in_baseline_top_50": bool(baseline_rank is not None and baseline_rank <= 50),
                "reason": ", ".join(deduplicate_preserve_order(reasons)) if reasons else "meaningful",
            }
        )
    return pd.DataFrame(rows)


def save_user_split_diagnostics(
    settings: Settings,
    splits: list[dict[str, object]],
    path_json: Path,
    path_markdown: Path,
    title: str,
) -> dict[str, object]:
    """Save split diagnostics as JSON and markdown."""

    diagnostics_df = build_user_split_diagnostics_frame(settings, splits)
    reason_counts = Counter()
    for reason_text in diagnostics_df.get("reason", pd.Series(dtype=str)).astype(str).tolist():
        for reason in [value.strip() for value in reason_text.split(",") if value.strip() and value.strip() != "meaningful"]:
            reason_counts[reason] += 1

    summary = {
        "title": title,
        "split_count": int(len(splits)),
        "number_of_splits": int(len(splits)),
        "average_candidate_pool_size": round(
            float(diagnostics_df["candidate_pool_size"].mean()), 6
        ) if not diagnostics_df.empty and "candidate_pool_size" in diagnostics_df.columns else 0.0,
        "splits_with_holdout_game_ids": int(
            diagnostics_df["holdout_count"].fillna(0).astype(int).gt(0).sum()
        ) if not diagnostics_df.empty and "holdout_count" in diagnostics_df.columns else 0,
        "splits_with_holdout_in_candidate_pool": int(
            diagnostics_df["holdout_in_candidate_pool"].fillna(False).astype(bool).sum()
        ) if not diagnostics_df.empty and "holdout_in_candidate_pool" in diagnostics_df.columns else 0,
        "splits_without_holdout_in_candidate_pool": int(
            len(diagnostics_df)
            - (diagnostics_df["holdout_in_candidate_pool"].fillna(False).astype(bool).sum())
        ) if not diagnostics_df.empty and "holdout_in_candidate_pool" in diagnostics_df.columns else 0,
        "splits_with_holdout_in_baseline_top_10": int(
            diagnostics_df["holdout_in_baseline_top_10"].fillna(False).astype(bool).sum()
        ) if not diagnostics_df.empty and "holdout_in_baseline_top_10" in diagnostics_df.columns else 0,
        "splits_with_holdout_in_baseline_top_50": int(
            diagnostics_df["holdout_in_baseline_top_50"].fillna(False).astype(bool).sum()
        ) if not diagnostics_df.empty and "holdout_in_baseline_top_50" in diagnostics_df.columns else 0,
        "reason_counts": dict(reason_counts),
    }
    path_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {title}",
        "",
        f"- Splits: {summary['split_count']}",
        f"- Average candidate pool size: {summary['average_candidate_pool_size']}",
        f"- Splits with holdout_game_ids: {summary['splits_with_holdout_game_ids']}",
        f"- Splits with holdout in candidate pool: {summary['splits_with_holdout_in_candidate_pool']}",
        f"- Splits without holdout in candidate pool: {summary['splits_without_holdout_in_candidate_pool']}",
        f"- Splits with holdout in baseline top-10: {summary['splits_with_holdout_in_baseline_top_10']}",
        f"- Splits with holdout in baseline top-50: {summary['splits_with_holdout_in_baseline_top_50']}",
        "",
        "## Meaningfulness reasons",
    ]
    if reason_counts:
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(reason_counts.items()))
    else:
        lines.append("- none")
    if not diagnostics_df.empty:
        lines.extend(["", dataframe_to_markdown(diagnostics_df.head(20))])
    path_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    """Return values without duplicates while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def format_rank_value(rank: int | None) -> int | str:
    """Format a rank for report output."""

    return "not_found" if rank is None else int(rank)


def is_hit_at_k(rank: int | None, k: int) -> bool:
    """Return whether a rank is within the top-k."""

    return bool(rank is not None and rank <= k)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a tiny markdown table."""

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


def summarize_user_eligibility(settings: Settings, reviews_df: pd.DataFrame) -> dict[str, int | bool]:
    """Compute user-based availability statistics from cleaned reviews."""

    if "user_id" not in reviews_df.columns:
        return {
            "has_user_id": False,
            "user_based_mode_available": False,
            "unique_users": 0,
            "eligible_user_count": 0,
            "min_user_reviews": settings.min_user_reviews,
            "min_user_positive_reviews": settings.min_user_positive_reviews,
            "can_run_user_experiment": False,
        }

    user_reviews = reviews_df[reviews_df["user_id"].fillna("").astype(str).str.len() > 0].copy()
    if user_reviews.empty:
        return {
            "has_user_id": False,
            "user_based_mode_available": False,
            "unique_users": 0,
            "eligible_user_count": 0,
            "min_user_reviews": settings.min_user_reviews,
            "min_user_positive_reviews": settings.min_user_positive_reviews,
            "can_run_user_experiment": False,
        }

    grouped = user_reviews.groupby("user_id", sort=False)
    per_user = grouped.agg(
        review_count=("review_id", "count"),
        positive_review_count=("recommended", lambda values: int(values.fillna(False).sum())),
    )
    eligible_user_count = int(
        (
            (per_user["review_count"] >= settings.min_user_reviews)
            & (per_user["positive_review_count"] >= settings.min_user_positive_reviews)
        ).sum()
    )
    return {
        "has_user_id": True,
        "user_based_mode_available": True,
        "unique_users": int(per_user.shape[0]),
        "eligible_user_count": eligible_user_count,
        "min_user_reviews": settings.min_user_reviews,
        "min_user_positive_reviews": settings.min_user_positive_reviews,
        "can_run_user_experiment": eligible_user_count > 0,
    }


def build_user_profiles(settings: Settings) -> list[dict[str, object]]:
    """Build compact user preference profiles from cleaned review histories."""

    logger = get_logger()
    reviews_df = load_clean_reviews_for_user_mode(settings)
    if reviews_df is None:
        return []

    user_reviews = reviews_df[reviews_df["user_id"].fillna("").astype(str).str.len() > 0].copy()
    grouped = user_reviews.groupby("user_id", sort=False)
    profiles: list[dict[str, object]] = []
    masked_lookup = build_masked_user_lookup(grouped.groups.keys())

    for user_id, group in grouped:
        positive_reviews = group[group["recommended"]].copy()
        unique_positive_games = positive_reviews["game_id"].astype(str).drop_duplicates().tolist()
        if len(group) < settings.min_user_reviews:
            continue
        if len(unique_positive_games) < settings.min_user_positive_reviews:
            continue

        profiles.append(
            {
                "user_id": str(user_id),
                "masked_user_id": masked_lookup[str(user_id)],
                "review_count": int(len(group)),
                "positive_review_count": int(len(positive_reviews)),
                "positive_game_ids": unique_positive_games,
                "negative_game_ids": group.loc[~group["recommended"], "game_id"].astype(str).drop_duplicates().tolist(),
                "positive_review_text": " ".join(positive_reviews["review_clean"].astype(str).tolist()),
            }
        )

    write_jsonl(settings.user_profiles_path, profiles)
    summary = {
        "profile_count": len(profiles),
        "eligible_user_count": len(profiles),
        "min_user_reviews": settings.min_user_reviews,
        "min_user_positive_reviews": settings.min_user_positive_reviews,
    }
    settings.user_profile_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Saved %s user profiles to %s",
        len(profiles),
        settings.user_profiles_path,
    )
    return profiles


def build_user_evaluation_splits(settings: Settings) -> list[dict[str, object]]:
    """Create controlled holdout splits with explicit candidate pools."""

    return _build_user_splits(
        settings=settings,
        output_path=settings.user_splits_path,
        summary_path=settings.user_split_summary_path,
        summary_markdown_path=None,
        diagnostics_json_path=settings.user_split_diagnostics_json_path,
        diagnostics_markdown_path=settings.user_split_diagnostics_markdown_path,
        split_count_limit=settings.max_users_for_experiment,
        pilot_mode=False,
        title="User Split Diagnostics",
    )


def build_user_splits_pilot(settings: Settings) -> list[dict[str, object]]:
    """Create pilot-friendly splits with holdouts guaranteed in the candidate pool."""

    return _build_user_splits(
        settings=settings,
        output_path=settings.pilot_splits_path,
        summary_path=settings.user_evaluation_split_pilot_summary_path,
        summary_markdown_path=settings.user_evaluation_split_pilot_summary_markdown_path,
        diagnostics_json_path=settings.user_split_diagnostics_json_path,
        diagnostics_markdown_path=settings.user_split_diagnostics_markdown_path,
        split_count_limit=settings.max_pilot_splits,
        pilot_mode=True,
        title="User Split Diagnostics",
    )


def _build_user_splits(
    *,
    settings: Settings,
    output_path: Path,
    summary_path: Path,
    summary_markdown_path: Path | None,
    diagnostics_json_path: Path,
    diagnostics_markdown_path: Path,
    split_count_limit: int,
    pilot_mode: bool,
    title: str,
) -> list[dict[str, object]]:
    """Shared split builder for the main and pilot workflows."""

    logger = get_logger()
    profiles = read_jsonl(settings.user_profiles_path)
    if not profiles:
        logger.warning("User-based mode is unavailable or no user profiles have been built yet.")
        return []

    reviews_df = load_clean_reviews_for_user_mode(settings)
    if reviews_df is None:
        return []

    game_cards_by_id = load_game_cards_by_id(settings)
    if not game_cards_by_id:
        logger.warning("Game cards are unavailable; split generation will be limited.")
    split_vectorizer: TfidfVectorizer | None = None
    if game_cards_by_id:
        split_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
        split_vectorizer.fit([card.game_card_text for card in game_cards_by_id.values()])
    user_reviews_df = reviews_df[reviews_df["user_id"].fillna("").astype(str).str.len() > 0].copy()
    user_reviews_groups = {
        str(user_id): group.copy()
        for user_id, group in user_reviews_df.groupby("user_id", sort=False)
    }

    eligible_profiles = [profile for profile in profiles if str(profile.get("user_id", ""))]
    candidate_splits: list[dict[str, object]] = []
    meaningful_pilot_splits: list[dict[str, object]] = []

    for profile in eligible_profiles:
        user_id = str(profile["user_id"])
        masked_user_id = str(profile.get("masked_user_id", "")) or mask_user_id(user_id)
        user_reviews = user_reviews_groups.get(user_id)
        if user_reviews is None or user_reviews.empty:
            continue

        split = build_user_split_record(
            user_id=user_id,
            masked_user_id=masked_user_id,
            user_reviews=user_reviews,
            game_cards_by_id=game_cards_by_id,
            settings=settings,
            vectorizer=split_vectorizer,
        )
        if split is None:
            continue
        candidate_splits.append(split)
        baseline_rank = split.get("baseline_best_holdout_rank", "not_found")
        try:
            baseline_rank_value = None if baseline_rank in {None, "", "not_found"} else int(baseline_rank)
        except Exception:
            baseline_rank_value = None
        if (
            split.get("holdout_in_candidate_pool", False)
            and baseline_rank_value is not None
            and baseline_rank_value <= settings.max_llm_candidates
            and baseline_rank_value > 1
        ):
            meaningful_pilot_splits.append(split)
            if pilot_mode and len(meaningful_pilot_splits) >= split_count_limit:
                break

    meaningful_pilot_splits.sort(
        key=lambda split: int(
            split.get("baseline_best_holdout_rank", settings.max_llm_candidates + 1)
            if str(split.get("baseline_best_holdout_rank", "not_found")) not in {"", "not_found"}
            else settings.max_llm_candidates + 1
        )
    )
    candidate_splits.sort(
        key=lambda split: int(
            split.get("baseline_best_holdout_rank", settings.max_llm_candidates + 1)
            if str(split.get("baseline_best_holdout_rank", "not_found")) not in {"", "not_found"}
            else settings.max_llm_candidates + 1
        )
    )

    selected_splits: list[dict[str, object]]
    warnings: list[str] = []
    if pilot_mode:
        selected_splits = meaningful_pilot_splits[:split_count_limit]
        if len(selected_splits) < split_count_limit:
            remaining_splits = [split for split in candidate_splits if split not in selected_splits]
            selected_splits.extend(remaining_splits[: split_count_limit - len(selected_splits)])
            if meaningful_pilot_splits:
                warnings.append(
                    f"Only {len(meaningful_pilot_splits)} meaningful pilot splits were available; the remainder were filled from fallback users."
                )
            else:
                warnings.append(
                    "No clearly meaningful pilot splits were found; fallback users were selected so the pilot can still be inspected."
                )
    else:
        selected_splits = candidate_splits[:split_count_limit]

    write_jsonl(output_path, selected_splits)
    diagnostics_summary = save_user_split_diagnostics(
        settings,
        selected_splits,
        diagnostics_json_path,
        diagnostics_markdown_path,
        title,
    )
    summary = {
        "split_count": len(selected_splits),
        "selected_split_count": len(selected_splits),
        "user_holdout_count": settings.user_holdout_count,
        "max_users_for_experiment": split_count_limit,
        "force_holdout_into_candidate_pool": bool(settings.force_holdout_into_candidate_pool),
        "candidate_pool_size": settings.user_candidate_pool_size,
        "use_pilot_splits": pilot_mode,
        "meaningful_pilot_split_count": len(meaningful_pilot_splits) if pilot_mode else 0,
        "fallback_split_count": max(len(selected_splits) - len(meaningful_pilot_splits), 0) if pilot_mode else 0,
        "splits_with_holdout_in_candidate_pool": int(diagnostics_summary.get("splits_with_holdout_in_candidate_pool", 0)),
        "splits_without_holdout_in_candidate_pool": int(diagnostics_summary.get("splits_without_holdout_in_candidate_pool", 0)),
        "splits_with_holdout_in_baseline_top_10": int(diagnostics_summary.get("splits_with_holdout_in_baseline_top_10", 0)),
        "splits_with_holdout_in_baseline_top_50": int(diagnostics_summary.get("splits_with_holdout_in_baseline_top_50", 0)),
        "warnings": warnings,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary_markdown_path is not None:
        lines = [
            f"# {title}",
            "",
            f"- Split count: {summary['split_count']}",
            f"- Selected split count: {summary['selected_split_count']}",
            f"- Mean candidate pool size: {summary['candidate_pool_size']}",
            f"- User holdout count: {summary['user_holdout_count']}",
            f"- Force holdout into candidate pool: {summary['force_holdout_into_candidate_pool']}",
            f"- Use pilot splits: {summary['use_pilot_splits']}",
            f"- Meaningful pilot splits: {summary['meaningful_pilot_split_count']}",
            f"- Fallback split count: {summary['fallback_split_count']}",
            f"- Splits with holdout in candidate pool: {summary['splits_with_holdout_in_candidate_pool']}",
            f"- Splits without holdout in candidate pool: {summary['splits_without_holdout_in_candidate_pool']}",
            f"- Splits with holdout in baseline top-10: {summary['splits_with_holdout_in_baseline_top_10']}",
            f"- Splits with holdout in baseline top-50: {summary['splits_with_holdout_in_baseline_top_50']}",
            "",
            "## Warnings",
        ]
        if warnings:
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("- none")
        summary_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_label = "pilot user evaluation splits" if pilot_mode else "user evaluation splits"
    logger.info("Saved %s to %s", output_label, output_path)
    return selected_splits


def run_user_baseline(settings: Settings) -> list[dict[str, object]]:
    """Rank game cards for user profiles using TF-IDF over held-in liked games."""

    logger = get_logger()
    split_path = get_active_user_splits_path(settings)
    if not split_path.exists():
        logger.warning("User evaluation splits not found. Run `--step build_user_splits` first.")
        return []
    if not settings.game_cards_path.exists():
        logger.warning("Game cards not found. Run `--step build_cards` first.")
        return []

    splits = read_jsonl(split_path)
    game_cards = [GameCard(**record) for record in read_jsonl(settings.game_cards_path)]
    if not splits or not game_cards:
        logger.warning("User-based baseline is unavailable because splits or game cards are missing.")
        return []

    texts = [card.game_card_text for card in game_cards]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    matrix = vectorizer.fit_transform(texts)
    id_to_card = {card.game_id: card for card in game_cards}
    id_to_index = {card.game_id: index for index, card in enumerate(game_cards)}
    rows: list[dict[str, object]] = []

    for split in splits:
        train_game_ids = [game_id for game_id in split["train_game_ids"] if game_id in id_to_card]
        candidate_game_ids = [
            game_id
            for game_id in split.get("candidate_game_ids", [])
            if game_id in id_to_card and game_id not in train_game_ids
        ]
        if not candidate_game_ids:
            candidate_game_ids = [card.game_id for card in game_cards if card.game_id not in set(train_game_ids)]
        if not train_game_ids:
            rows.append(
                {
                    "user_id": split["user_id"],
                    "masked_user_id": split["masked_user_id"],
                    "method": "user_baseline",
                    "status": "no_train_games",
                    "rank": None,
                    "game_id": None,
                    "game_title": None,
                    "score": None,
                }
            )
            continue

        query_text = "\n\n".join(id_to_card[game_id].game_card_text for game_id in train_game_ids)
        scores = cosine_similarity(vectorizer.transform([query_text]), matrix).flatten()
        blocked_ids = set(train_game_ids)
        ranked_ids = sorted(
            [game_id for game_id in candidate_game_ids if game_id not in blocked_ids],
            key=lambda game_id: float(scores[id_to_index[game_id]]),
            reverse=True,
        )[: settings.top_k]

        for rank, game_id in enumerate(ranked_ids, start=1):
            rows.append(
                {
                    "user_id": split["user_id"],
                    "masked_user_id": split["masked_user_id"],
                    "method": "user_baseline",
                    "status": "ok",
                    "rank": rank,
                    "game_id": game_id,
                    "game_title": id_to_card[game_id].game_title,
                    "score": round(float(scores[id_to_index[game_id]]), 6),
                }
            )

    write_jsonl(settings.user_baseline_results_path, rows)
    logger.info("Saved user baseline recommendations to %s", settings.user_baseline_results_path)
    return rows


def evaluate_user_baseline(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate user-based baseline recommendations against holdout games."""

    logger = get_logger()
    split_path = get_active_user_splits_path(settings)
    if not split_path.exists() or not settings.user_baseline_results_path.exists():
        logger.warning("User-based evaluation is unavailable. Build splits and user baseline first.")
        empty = pd.DataFrame()
        return empty, empty

    splits = read_jsonl(split_path)
    rows = read_jsonl(settings.user_baseline_results_path)
    grouped_rows = group_rows_by_user(rows)

    per_profile_rows: list[dict[str, object]] = []
    for split in splits:
        user_id = str(split["user_id"])
        ground_truth_ids = {str(game_id) for game_id in split["ground_truth_game_ids"]}
        ranked_rows = sorted(
            [
                row
                for row in grouped_rows.get(user_id, [])
                if row.get("rank") is not None and row.get("game_id")
            ],
            key=lambda row: int(row["rank"]),
        )
        best_rank = next(
            (
                int(row["rank"])
                for row in ranked_rows
                if str(row["game_id"]) in ground_truth_ids
            ),
            None,
        )
        hit_at_5 = int(best_rank is not None and best_rank <= 5)
        hit_at_10 = int(best_rank is not None and best_rank <= 10)
        mrr = 0.0 if best_rank is None else 1 / best_rank
        ndcg_at_10 = 0.0 if best_rank is None or best_rank > 10 else 1 / math.log2(best_rank + 1)
        per_profile_rows.append(
            {
                "user_id": user_id,
                "masked_user_id": split["masked_user_id"],
                "hit_rate_at_5": hit_at_5,
                "hit_rate_at_10": hit_at_10,
                "mrr": round(float(mrr), 6),
                "ndcg_at_10": round(float(ndcg_at_10), 6),
                "ground_truth_count": len(ground_truth_ids),
                "status": "ok" if ranked_rows else "missing_results",
            }
        )

    per_profile_df = pd.DataFrame(per_profile_rows)
    per_profile_df.to_csv(settings.user_per_profile_results_path, index=False)
    if per_profile_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = pd.DataFrame(
            [
                {
                    "method": "user_baseline",
                    "evaluated_profiles": int(len(per_profile_df)),
                    "mean_hit_rate_at_5": round(float(per_profile_df["hit_rate_at_5"].mean()), 6),
                    "mean_hit_rate_at_10": round(float(per_profile_df["hit_rate_at_10"].mean()), 6),
                    "mean_mrr": round(float(per_profile_df["mrr"].mean()), 6),
                    "mean_ndcg_at_10": round(float(per_profile_df["ndcg_at_10"].mean()), 6),
                }
            ]
        )
    summary_df.to_csv(settings.user_metrics_summary_path, index=False)
    logger.info("Saved user-based evaluation outputs to %s", settings.user_metrics_summary_path)
    return per_profile_df, summary_df


def run_user_experiment(settings: Settings) -> None:
    """Run the full user-based baseline experiment when user ids are available."""

    if not settings.active_processed_reviews_path.exists():
        logger = get_logger()
        logger.warning(
            "Run preprocessing first or set ACTIVE_PROCESSED_REVIEWS to a cleaned reviews file before user_experiment."
        )
        return
    build_user_profiles(settings)
    if bool(getattr(settings, "use_pilot_splits", False)):
        build_user_splits_pilot(settings)
    else:
        build_user_evaluation_splits(settings)
    run_user_baseline(settings)
    evaluate_user_baseline(settings)


def build_masked_user_lookup(user_ids) -> dict[str, str]:
    """Create stable masked identifiers for human-readable reports."""

    return {
        str(user_id): f"user_{index:03d}"
        for index, user_id in enumerate(sorted(str(value) for value in user_ids), start=1)
    }


def group_rows_by_user(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Group row-wise user recommendation outputs by user id."""

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        user_id = str(row.get("user_id", ""))
        grouped.setdefault(user_id, []).append(row)
    return grouped


def load_json_if_exists(path) -> dict[str, object]:
    """Load a JSON file when it exists."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
