
from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.config import Settings
from src.game_card_builder import GameCard
from src.scenario_builder import Scenario
from src.utils import write_jsonl


@dataclass(slots=True)
class RecommendationRecord:

    scenario_id: str
    method: str
    status: str
    rank: int | None = None
    game_id: str | None = None
    game_title: str | None = None
    score: float | None = None
    candidate_size: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, object]:

        return {
            "scenario_id": self.scenario_id,
            "method": self.method,
            "rank": self.rank,
            "game_id": self.game_id,
            "game_title": self.game_title,
            "score": self.score,
            "status": self.status,
            "candidate_size": self.candidate_size,
            "notes": self.notes,
        }


def run_baseline(
    scenarios: list[Scenario],
    game_cards: list[GameCard],
    settings: Settings,
) -> list[RecommendationRecord]:

    if not game_cards:
        raise ValueError("No game cards are available for the TF-IDF baseline.")

    texts = [card.game_card_text for card in game_cards]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[^\W\d_]{2,}\b")
    matrix = vectorizer.fit_transform(texts)

    id_to_card = {card.game_id: card for card in game_cards}
    id_to_index = {card.game_id: index for index, card in enumerate(game_cards)}

    results: list[RecommendationRecord] = []
    for scenario in tqdm(scenarios, desc="Baseline scenarios", leave=False):
        query_text = build_query_text(scenario, id_to_card, settings)
        scenario_vector = vectorizer.transform([query_text])
        scores = cosine_similarity(scenario_vector, matrix).flatten()
        results.extend(
            rank_candidates(
                scenario=scenario,
                scores=scores,
                game_cards=game_cards,
                id_to_card=id_to_card,
                id_to_index=id_to_index,
                settings=settings,
            )
        )

    write_jsonl(settings.baseline_results_path, [result.to_dict() for result in results])
    return results


def build_query_text(
    scenario: Scenario,
    id_to_card: dict[str, GameCard],
    settings: Settings,
) -> str:

    query_parts = [scenario.preference_text.strip()]
    if settings.use_seed_cards_in_query:
        seed_card_texts = [
            id_to_card[game_id].game_card_text
            for game_id in scenario.seed_game_ids
            if game_id in id_to_card
        ]
        if seed_card_texts:
            query_parts.append("\n".join(seed_card_texts))
    return "\n\n".join(part for part in query_parts if part)


def rank_candidates(
    scenario: Scenario,
    scores,
    game_cards: list[GameCard],
    id_to_card: dict[str, GameCard],
    id_to_index: dict[str, int],
    settings: Settings,
) -> list[RecommendationRecord]:

    excluded_ids = set(scenario.seed_game_ids) | set(scenario.excluded_game_ids)
    if scenario.candidate_game_ids:
        candidate_ids = [
            game_id
            for game_id in scenario.candidate_game_ids
            if game_id in id_to_card and game_id not in excluded_ids
        ]
    else:
        candidate_ids = [
            card.game_id
            for card in game_cards
            if card.game_id not in excluded_ids
        ]

    candidate_size = len(candidate_ids)
    if candidate_size == 0:
        return [
            RecommendationRecord(
                scenario_id=scenario.scenario_id,
                method="baseline",
                status="no_valid_candidates",
                candidate_size=0,
                notes="No valid candidates remained after scenario filtering.",
            )
        ]

    ranked_candidates = sorted(
        (
            (
                game_id,
                float(scores[id_to_index[game_id]]),
            )
            for game_id in candidate_ids
        ),
        key=lambda item: item[1],
        reverse=True,
    )[: settings.top_k]

    return [
        RecommendationRecord(
            scenario_id=scenario.scenario_id,
            method="baseline",
            rank=rank,
            game_id=game_id,
            game_title=id_to_card[game_id].game_title,
            score=round(score, 6),
            status="ok",
            candidate_size=candidate_size,
            notes="TF-IDF ranking over preference text and compact game cards.",
        )
        for rank, (game_id, score) in enumerate(ranked_candidates, start=1)
    ]
