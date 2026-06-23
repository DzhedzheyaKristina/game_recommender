"""Build or load scenario-based recommendation profiles."""

from __future__ import annotations

import random

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings
from src.data_loader import load_external_scenarios
from src.game_card_builder import GameCard
from src.utils import get_logger, model_to_dict, write_jsonl


class Scenario(BaseModel):
    """Canonical scenario format for evaluation and ranking."""

    scenario_id: str
    scenario_type: str
    preference_text: str
    seed_game_ids: list[str] = Field(default_factory=list)
    excluded_game_ids: list[str] = Field(default_factory=list)
    ground_truth_game_ids: list[str] = Field(default_factory=list)
    candidate_game_ids: list[str] = Field(default_factory=list)
    notes: str = ""


def build_scenarios(
    game_cards: list[GameCard],
    settings: Settings,
) -> list[Scenario]:
    """Load predefined scenarios or derive synthetic demo scenarios."""

    logger = get_logger()
    id_to_card = {card.game_id: card for card in game_cards}

    external_records: list[dict[str, object]] = []
    if settings.scenarios_file is not None:
        try:
            external_records = load_external_scenarios(settings.scenarios_file)
        except FileNotFoundError:
            logger.warning(
                "Scenario file %s was not found. Falling back to synthetic demo scenarios.",
                settings.scenarios_file,
            )

    if external_records:
        logger.info("Using predefined scenarios from %s", settings.scenarios_file)
        scenarios = normalize_predefined_scenarios(
            external_records=external_records,
            id_to_card=id_to_card,
        )
    else:
        logger.warning(
            "Using synthetic demo scenarios only. These scenarios are for local technical "
            "validation and not strong scientific evidence."
        )
        scenarios = generate_synthetic_scenarios(game_cards, settings)

    write_jsonl(
        settings.scenarios_output_path,
        [model_to_dict(scenario) for scenario in scenarios],
    )
    return scenarios


def normalize_predefined_scenarios(
    external_records: list[dict[str, object]],
    id_to_card: dict[str, GameCard],
) -> list[Scenario]:
    """Normalize externally provided scenarios and annotate partial issues."""

    logger = get_logger()
    scenarios: list[Scenario] = []
    for index, record in enumerate(external_records, start=1):
        scenario_id = str(record.get("scenario_id", "")).strip()
        if not scenario_id:
            logger.warning("Skipping predefined scenario row %s because scenario_id is missing.", index)
            continue

        preference_text = str(record.get("preference_text", "")).strip()
        if not preference_text:
            logger.warning(
                "Skipping predefined scenario %s because preference_text is missing.",
                scenario_id,
            )
            continue

        seed_game_ids = normalize_id_list(record.get("seed_game_ids"))
        excluded_game_ids = normalize_id_list(record.get("excluded_game_ids"))
        ground_truth_game_ids = normalize_id_list(record.get("ground_truth_game_ids"))
        candidate_game_ids = normalize_id_list(record.get("candidate_game_ids"))

        notes = normalize_notes(record.get("notes"))
        missing_notes: list[str] = []
        seed_game_ids = keep_existing_ids(seed_game_ids, id_to_card, "seed_game_ids", missing_notes)
        excluded_game_ids = keep_existing_ids(
            excluded_game_ids,
            id_to_card,
            "excluded_game_ids",
            missing_notes,
        )
        ground_truth_game_ids = keep_existing_ids(
            ground_truth_game_ids,
            id_to_card,
            "ground_truth_game_ids",
            missing_notes,
        )
        candidate_game_ids = keep_existing_ids(
            candidate_game_ids,
            id_to_card,
            "candidate_game_ids",
            missing_notes,
        )

        if not ground_truth_game_ids:
            missing_notes.append(
                "Warning: ground_truth_game_ids is empty after validation; this scenario "
                "will be skipped during evaluation."
            )

        if candidate_game_ids and set(candidate_game_ids).issubset(set(seed_game_ids + excluded_game_ids)):
            missing_notes.append(
                "Warning: candidate_game_ids contain no usable games after excluding "
                "seed and excluded ids."
            )

        scenario_type = normalize_predefined_scenario_type(
            raw_value=record.get("scenario_type"),
            has_seed_games=bool(seed_game_ids),
        )
        all_notes = " ".join(part for part in [notes, *missing_notes] if part).strip()

        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                preference_text=preference_text,
                seed_game_ids=seed_game_ids,
                excluded_game_ids=excluded_game_ids,
                ground_truth_game_ids=ground_truth_game_ids,
                candidate_game_ids=candidate_game_ids,
                notes=all_notes,
            )
        )

    return scenarios


def normalize_predefined_scenario_type(raw_value: object, has_seed_games: bool) -> str:
    """Normalize predefined scenarios into the allowed scenario types."""

    normalized = str(raw_value or "").strip().lower()
    if normalized in {"manual", "manual_draft", "seed_games", "synthetic_demo"}:
        return normalized
    return "seed_games" if has_seed_games else "manual"


def normalize_notes(value: object) -> str:
    """Normalize scenario notes into a compact string."""

    if value is None:
        return ""
    return str(value).strip()


def normalize_id_list(value: object) -> list[str]:
    """Normalize a list-like field into a unique list of string game ids."""

    if value is None:
        return []
    if isinstance(value, list):
        return deduplicate_preserve_order(str(item).strip() for item in value if str(item).strip())

    raw_value = str(value).strip()
    if not raw_value or raw_value.lower() == "nan":
        return []
    if raw_value.startswith("[") and raw_value.endswith("]"):
        stripped = raw_value[1:-1].strip()
        if not stripped:
            return []
        parts = [part.strip().strip("'\"") for part in stripped.split(",")]
        return deduplicate_preserve_order(part for part in parts if part)

    separator = "|" if "|" in raw_value else ","
    parts = [part.strip().strip("'\"") for part in raw_value.split(separator)]
    return deduplicate_preserve_order(part for part in parts if part)


def deduplicate_preserve_order(values) -> list[str]:
    """Deduplicate iterables while keeping the first occurrence order."""

    seen: set[str] = set()
    deduplicated: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduplicated.append(value)
    return deduplicated


def keep_existing_ids(
    game_ids: list[str],
    id_to_card: dict[str, GameCard],
    field_name: str,
    notes: list[str],
) -> list[str]:
    """Filter unknown game ids and record a warning instead of crashing."""

    kept_ids: list[str] = []
    missing_ids: list[str] = []
    for game_id in game_ids:
        if game_id in id_to_card:
            kept_ids.append(game_id)
        else:
            missing_ids.append(game_id)

    if missing_ids:
        notes.append(
            f"Warning: {field_name} contains missing game ids that were removed: "
            + ", ".join(missing_ids)
            + ". Scenario is partially invalid."
        )
    return kept_ids


def generate_synthetic_scenarios(
    game_cards: list[GameCard],
    settings: Settings,
) -> list[Scenario]:
    """Generate lightweight synthetic demo scenarios for local validation."""

    if not game_cards:
        return []

    rng = random.Random(settings.random_seed)
    eligible_cards = [
        card
        for card in sorted(
            game_cards,
            key=lambda card: (card.positive_ratio, card.review_count),
            reverse=True,
        )
        if card.review_count >= settings.min_reviews_per_game and card.positive_keywords
    ]
    if not eligible_cards:
        eligible_cards = sorted(game_cards, key=lambda card: card.review_count, reverse=True)

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    matrix = vectorizer.fit_transform([card.game_card_text for card in game_cards])
    id_to_index = {card.game_id: index for index, card in enumerate(game_cards)}

    scenarios: list[Scenario] = []
    used_seed_ids: set[str] = set()
    for seed_card in eligible_cards:
        if len(scenarios) >= settings.synthetic_scenario_count:
            break
        if seed_card.game_id in used_seed_ids:
            continue

        similar_cards = find_similar_cards(
            seed_card=seed_card,
            game_cards=game_cards,
            matrix=matrix,
            id_to_index=id_to_index,
        )
        if not similar_cards:
            continue

        ground_truth_card = similar_cards[0]
        distractor_ids = [
            card.game_id
            for card in game_cards
            if card.game_id not in {seed_card.game_id, ground_truth_card.game_id}
        ]
        rng.shuffle(distractor_ids)
        candidate_game_ids = [ground_truth_card.game_id] + distractor_ids
        candidate_game_ids = deduplicate_preserve_order(candidate_game_ids)
        candidate_game_ids = candidate_game_ids[: settings.candidate_pool_size]

        scenarios.append(
            Scenario(
                scenario_id=f"synthetic_demo_{len(scenarios) + 1}",
                scenario_type="synthetic_demo",
                preference_text=build_synthetic_preference_text(seed_card),
                seed_game_ids=[seed_card.game_id],
                excluded_game_ids=[],
                ground_truth_game_ids=[ground_truth_card.game_id],
                candidate_game_ids=candidate_game_ids,
                notes=(
                    "Synthetic demo scenario for local validation only; these results are "
                    "not strong scientific evidence."
                ),
            )
        )
        used_seed_ids.add(seed_card.game_id)

    return scenarios


def find_similar_cards(
    seed_card: GameCard,
    game_cards: list[GameCard],
    matrix,
    id_to_index: dict[str, int],
) -> list[GameCard]:
    """Find similar game cards with TF-IDF cosine similarity."""

    seed_index = id_to_index[seed_card.game_id]
    similarities = cosine_similarity(matrix[seed_index], matrix).flatten()
    ranked_indices = sorted(
        range(len(game_cards)),
        key=lambda index: similarities[index],
        reverse=True,
    )

    related_cards: list[GameCard] = []
    for index in ranked_indices:
        candidate = game_cards[index]
        if candidate.game_id == seed_card.game_id:
            continue
        if candidate.review_count < 1:
            continue
        related_cards.append(candidate)
    return related_cards


def build_synthetic_preference_text(seed_card: GameCard) -> str:
    """Build synthetic preference text from a high-quality seed game card."""

    positive_keywords = ", ".join(seed_card.positive_keywords[:5] or ["none"])
    negative_keywords = ", ".join(seed_card.negative_keywords[:5] or ["none"])
    return (
        f"I want a game similar to {seed_card.game_title}. "
        f"Preferred qualities: {positive_keywords}. "
        f"Avoided qualities: {negative_keywords}."
    )
