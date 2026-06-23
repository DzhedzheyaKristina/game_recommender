"""Aggregate cleaned reviews into compact game cards."""

from __future__ import annotations

import json

import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.config import Settings
from src.utils import get_logger, model_to_dict, truncate_text, write_jsonl


RUSSIAN_STOPWORDS = {
    "а",
    "без",
    "более",
    "бы",
    "был",
    "была",
    "были",
    "было",
    "быть",
    "в",
    "вам",
    "вас",
    "весь",
    "во",
    "вот",
    "все",
    "всё",
    "вы",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "её",
    "если",
    "есть",
    "еще",
    "ещё",
    "же",
    "за",
    "здесь",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "когда",
    "ли",
    "мне",
    "можно",
    "мы",
    "на",
    "над",
    "не",
    "него",
    "нее",
    "неё",
    "нет",
    "но",
    "ну",
    "о",
    "об",
    "однако",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "очень",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "также",
    "такой",
    "там",
    "те",
    "тем",
    "то",
    "того",
    "тоже",
    "только",
    "том",
    "ты",
    "у",
    "уже",
    "хоть",
    "хорошо",
    "чем",
    "что",
    "чтобы",
    "эта",
    "эти",
    "это",
    "я",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS).union(RUSSIAN_STOPWORDS))


class GameCard(BaseModel):
    """Compact representation of a game built from review text only."""

    game_id: str
    game_title: str
    review_count: int
    positive_review_count: int
    negative_review_count: int
    positive_ratio: float
    positive_keywords: list[str]
    negative_keywords: list[str]
    representative_positive_reviews: list[str]
    representative_negative_reviews: list[str]
    average_playtime_forever: float = 0.0
    median_playtime_forever: float = 0.0
    weighted_vote_score_mean: float = 0.0
    steam_purchase_ratio: float = 0.0
    received_for_free_ratio: float = 0.0
    early_access_review_ratio: float = 0.0
    game_card_text: str


def build_game_cards(reviews_df: pd.DataFrame, settings: Settings) -> list[GameCard]:
    """Create one card per game from cleaned review data."""

    logger = get_logger()
    validate_card_settings(settings)
    game_cards: list[GameCard] = []
    skipped_games = 0
    review_text_column = "review_clean" if "review_clean" in reviews_df.columns else "review_text"
    grouped = reviews_df.groupby(["game_id", "game_title"], sort=False)

    for (game_id, game_title), group in grouped:
        review_count = int(len(group))
        if review_count < settings.min_reviews_per_game:
            skipped_games += 1
            continue

        selected_reviews = select_reviews_for_card(
            group,
            max_reviews=settings.max_reviews_per_game_for_card,
        )
        positive_reviews = selected_reviews[selected_reviews["recommended"]]
        negative_reviews = selected_reviews[~selected_reviews["recommended"]]

        positive_count = int(group["recommended"].sum())
        negative_count = review_count - positive_count
        positive_ratio = positive_count / review_count if review_count else 0.0
        average_playtime_forever = safe_mean(group.get("playtime_forever"))
        median_playtime_forever = safe_median(group.get("playtime_forever"))
        weighted_vote_score_mean = safe_mean(group.get("weighted_vote_score"))
        steam_purchase_ratio = safe_ratio(group.get("steam_purchase"))
        received_for_free_ratio = safe_ratio(group.get("received_for_free"))
        early_access_review_ratio = safe_ratio(group.get("written_during_early_access"))

        positive_keywords = extract_keywords(
            positive_reviews[review_text_column].tolist(),
            top_n=settings.max_keywords,
        )
        negative_keywords = extract_keywords(
            negative_reviews[review_text_column].tolist(),
            top_n=settings.max_keywords,
        )

        representative_positive_reviews = select_representative_reviews(
            positive_reviews,
            count=settings.max_representative_reviews,
            max_chars=settings.max_review_chars,
            review_text_column=review_text_column,
        )
        representative_negative_reviews = select_representative_reviews(
            negative_reviews,
            count=settings.max_representative_reviews,
            max_chars=settings.max_review_chars,
            review_text_column=review_text_column,
        )

        game_card_text = compose_game_card_text(
            game_title=str(game_title),
            positive_ratio=positive_ratio,
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords,
            representative_positive_reviews=representative_positive_reviews,
            representative_negative_reviews=representative_negative_reviews,
            average_playtime_forever=average_playtime_forever,
            weighted_vote_score_mean=weighted_vote_score_mean,
            steam_purchase_ratio=steam_purchase_ratio,
            received_for_free_ratio=received_for_free_ratio,
        )

        game_cards.append(
            GameCard(
                game_id=str(game_id),
                game_title=str(game_title),
                review_count=review_count,
                positive_review_count=positive_count,
                negative_review_count=negative_count,
                positive_ratio=round(positive_ratio, 4),
                positive_keywords=positive_keywords,
                negative_keywords=negative_keywords,
                representative_positive_reviews=representative_positive_reviews,
                representative_negative_reviews=representative_negative_reviews,
                average_playtime_forever=average_playtime_forever,
                median_playtime_forever=median_playtime_forever,
                weighted_vote_score_mean=weighted_vote_score_mean,
                steam_purchase_ratio=steam_purchase_ratio,
                received_for_free_ratio=received_for_free_ratio,
                early_access_review_ratio=early_access_review_ratio,
                game_card_text=game_card_text,
            )
        )

    write_jsonl(settings.game_cards_path, [model_to_dict(card) for card in game_cards])
    write_game_card_summary(
        settings=settings,
        input_review_count=int(len(reviews_df)),
        game_cards=game_cards,
        skipped_games=skipped_games,
    )
    logger.info("Saved game card summary to %s", settings.game_card_summary_path)
    return game_cards


def validate_card_settings(settings: Settings) -> None:
    """Validate configuration values used for game-card generation."""

    if settings.min_reviews_per_game <= 0:
        raise ValueError("MIN_REVIEWS_PER_GAME must be positive.")
    if settings.max_reviews_per_game_for_card <= 0:
        raise ValueError("MAX_REVIEWS_PER_GAME_FOR_CARD must be positive.")
    if settings.max_representative_reviews <= 0:
        raise ValueError("MAX_REPRESENTATIVE_REVIEWS must be positive.")
    if settings.max_keywords <= 0:
        raise ValueError("MAX_KEYWORDS must be positive.")


def select_reviews_for_card(reviews_df: pd.DataFrame, max_reviews: int) -> pd.DataFrame:
    """Cap large review groups while preserving both sentiment subsets."""

    ordered = sort_reviews(reviews_df)
    if len(ordered) <= max_reviews:
        return ordered

    positive_reviews = sort_reviews(ordered[ordered["recommended"]])
    negative_reviews = sort_reviews(ordered[~ordered["recommended"]])
    positive_limit, negative_limit = allocate_review_budget(
        positive_count=len(positive_reviews),
        negative_count=len(negative_reviews),
        max_reviews=max_reviews,
    )
    selected = pd.concat(
        [
            positive_reviews.head(positive_limit),
            negative_reviews.head(negative_limit),
        ],
        axis=0,
    )
    return sort_reviews(selected)


def sort_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Order reviews by helpfulness and stable timestamp tiebreakers."""

    return reviews_df.sort_values(
        by=["votes_helpful", "timestamp_created", "review_id"],
        ascending=[False, True, True],
    )


def allocate_review_budget(
    positive_count: int,
    negative_count: int,
    max_reviews: int,
) -> tuple[int, int]:
    """Allocate a capped review budget across positive and negative subsets."""

    total_count = positive_count + negative_count
    if total_count <= max_reviews:
        return positive_count, negative_count

    allocations = {"positive": 0, "negative": 0}
    counts = {"positive": positive_count, "negative": negative_count}
    non_empty_labels = [label for label, count in counts.items() if count > 0]
    for label in non_empty_labels:
        proportional = round(max_reviews * counts[label] / total_count)
        allocations[label] = min(counts[label], max(1, proportional))

    while sum(allocations.values()) > max_reviews:
        reducible = [
            label for label in non_empty_labels if allocations[label] > 1
        ] or [
            label for label in non_empty_labels if allocations[label] > 0
        ]
        label_to_reduce = max(reducible, key=lambda label: allocations[label])
        allocations[label_to_reduce] -= 1

    while sum(allocations.values()) < max_reviews:
        expandable = [
            label for label in non_empty_labels if allocations[label] < counts[label]
        ]
        if not expandable:
            break
        label_to_expand = max(
            expandable,
            key=lambda label: counts[label] - allocations[label],
        )
        allocations[label_to_expand] += 1

    return allocations["positive"], allocations["negative"]


def extract_keywords(reviews: list[str], top_n: int) -> list[str]:
    """Extract TF-IDF keywords from a subset of reviews."""

    usable_reviews = [review for review in reviews if review.strip()]
    if not usable_reviews:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words=STOPWORDS,
            lowercase=True,
            token_pattern=r"(?u)\b[^\W\d_]{3,}\b",
        )
        matrix = vectorizer.fit_transform(usable_reviews)
    except ValueError:
        return []

    weights = matrix.mean(axis=0).A1
    features = vectorizer.get_feature_names_out()
    ranked = sorted(
        zip(features, weights, strict=False),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    return [token for token, weight in ranked if weight > 0][:top_n]


def select_representative_reviews(
    reviews_df: pd.DataFrame,
    count: int,
    max_chars: int,
    review_text_column: str = "review",
) -> list[str]:
    """Choose compact representative snippets using helpful-vote sorting."""

    if reviews_df.empty:
        return []

    selected = sort_reviews(reviews_df)[review_text_column].head(count).tolist()
    return [truncate_text(review, max_chars) for review in selected]


def compose_game_card_text(
    game_title: str,
    positive_ratio: float,
    positive_keywords: list[str],
    negative_keywords: list[str],
    representative_positive_reviews: list[str],
    representative_negative_reviews: list[str],
    average_playtime_forever: float,
    weighted_vote_score_mean: float,
    steam_purchase_ratio: float,
    received_for_free_ratio: float,
) -> str:
    """Compose a compact game card text for retrieval and reranking."""

    sections = [
        f"Game: {game_title}.",
        f"Positive ratio: {positive_ratio:.2f}.",
        "Players praise: " + ", ".join(positive_keywords or ["none"]) + ".",
        "Players criticize: " + ", ".join(negative_keywords or ["none"]) + ".",
        f"Average playtime forever: {average_playtime_forever:.2f}.",
        f"Weighted vote score mean: {weighted_vote_score_mean:.3f}.",
        f"Steam purchase ratio: {steam_purchase_ratio:.2f}.",
        f"Received for free ratio: {received_for_free_ratio:.2f}.",
        "Positive review examples: "
        + " | ".join(representative_positive_reviews or ["none"]),
        "Negative review examples: "
        + " | ".join(representative_negative_reviews or ["none"]),
    ]
    return "\n".join(sections)


def safe_mean(series: pd.Series | None) -> float:
    """Return a rounded mean when numeric values are available."""

    if series is None:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return round(float(numeric.mean()), 4)


def safe_median(series: pd.Series | None) -> float:
    """Return a rounded median when numeric values are available."""

    if series is None:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return round(float(numeric.median()), 4)


def safe_ratio(series: pd.Series | None) -> float:
    """Return a ratio over boolean-like values."""

    if series is None:
        return 0.0
    boolean_series = series.fillna(False).astype(bool)
    if boolean_series.empty:
        return 0.0
    return round(float(boolean_series.mean()), 4)


def write_game_card_summary(
    settings: Settings,
    input_review_count: int,
    game_cards: list[GameCard],
    skipped_games: int,
) -> None:
    """Write a short summary for thesis reporting."""

    average_reviews_per_card = 0.0
    if game_cards:
        average_reviews_per_card = round(
            sum(card.review_count for card in game_cards) / len(game_cards),
            2,
        )

    summary = {
        "input_review_count": input_review_count,
        "generated_game_card_count": len(game_cards),
        "average_reviews_per_card": average_reviews_per_card,
        "skipped_games": skipped_games,
        "min_review_threshold": settings.min_reviews_per_game,
    }
    with settings.game_card_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
