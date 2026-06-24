
from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.baseline_tfidf import RecommendationRecord
from src.config import Settings
from src.game_card_builder import GameCard
from src.llm_provider import (
    generate_llm_json_response,
    get_effective_llm_model,
    get_effective_llm_provider,
    provider_credentials_configured,
)
from src.scenario_builder import Scenario
from src.utils import load_text, write_jsonl


class LLMRankingResponse(BaseModel):

    ranked_game_ids: list[str]
    rationale: str = ""


def run_llm_reranker(
    scenarios: list[Scenario],
    baseline_results: list[RecommendationRecord],
    game_cards: list[GameCard],
    settings: Settings,
) -> list[RecommendationRecord]:

    baseline_by_scenario = group_records_by_scenario(baseline_results, method="baseline")
    id_to_card = {card.game_id: card for card in game_cards}
    provider = get_effective_llm_provider(settings)
    model = get_effective_llm_model(settings)

    if not provider_credentials_configured(settings):
        results = [
            RecommendationRecord(
                scenario_id=scenario.scenario_id,
                method="llm",
                status="skipped_no_credentials",
                notes=f"Credentials for provider '{provider}' are missing.",
            )
            for scenario in scenarios
        ]
        write_jsonl(settings.llm_results_path, [record.to_dict() for record in results])
        return results

    system_prompt = load_text(settings.system_prompt_path)
    user_template = load_text(settings.reranking_prompt_template_path)

    results: list[RecommendationRecord] = []
    for index, scenario in enumerate(
        tqdm(scenarios, desc="LLM scenarios", leave=False),
        start=1,
    ):
        baseline_rows = baseline_by_scenario.get(scenario.scenario_id, [])
        if index > settings.max_llm_scenarios:
            results.append(
                RecommendationRecord(
                    scenario_id=scenario.scenario_id,
                    method="llm",
                    status="skipped_scenario_limit",
                    notes="Skipped because MAX_LLM_SCENARIOS was reached.",
                )
            )
            continue

        if not has_ranked_rows(baseline_rows):
            results.append(
                RecommendationRecord(
                    scenario_id=scenario.scenario_id,
                    method="llm",
                    status="no_valid_candidates",
                    candidate_size=get_candidate_size(baseline_rows),
                    notes="Baseline produced no valid candidate ranking for this scenario.",
                )
            )
            continue

        results.extend(
            rerank_single_scenario(
                scenario=scenario,
                baseline_rows=baseline_rows,
                id_to_card=id_to_card,
                system_prompt=system_prompt,
                user_template=user_template,
                provider=provider,
                model=model,
                settings=settings,
            )
        )

    write_jsonl(settings.llm_results_path, [record.to_dict() for record in results])
    return results


def rerank_single_scenario(
    scenario: Scenario,
    baseline_rows: list[RecommendationRecord],
    id_to_card: dict[str, GameCard],
    system_prompt: str,
    user_template: str,
    provider: str,
    model: str,
    settings: Settings,
) -> list[RecommendationRecord]:

    baseline_ranked_rows = sorted(
        [row for row in baseline_rows if row.rank is not None and row.game_id],
        key=lambda row: row.rank or 0,
    )
    rerank_pool = baseline_ranked_rows[: settings.max_llm_candidates]
    baseline_ids = [row.game_id for row in rerank_pool if row.game_id is not None]
    baseline_score_map = {
        row.game_id: row.score for row in baseline_ranked_rows if row.game_id is not None
    }
    candidate_size = get_candidate_size(baseline_rows)
    target_count = min(settings.top_k, len(baseline_ranked_rows))

    candidate_lines = []
    for row in rerank_pool:
        if row.game_id is None:
            continue
        card = id_to_card[row.game_id]
        candidate_lines.append(f"{card.game_id} | {card.game_title}\n{card.game_card_text}")

    user_prompt = user_template.format(
        preference_text=scenario.preference_text,
        candidate_ids=", ".join(baseline_ids),
        candidate_games="\n\n".join(candidate_lines),
    )

    try:
        content = generate_llm_json_response(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            settings=settings,
        )
        parsed = parse_llm_response(content)
        valid_ids = keep_valid_reranked_ids(parsed.ranked_game_ids, baseline_ids)

        fallback_ids = [
            row.game_id
            for row in baseline_ranked_rows
            if row.game_id is not None and row.game_id not in valid_ids
        ]
        final_ids = (valid_ids + fallback_ids)[:target_count]
        final_status = "ok"
        if len(valid_ids) < target_count:
            final_status = "partial_fallback_to_baseline"

        return build_ranked_records(
            scenario_id=scenario.scenario_id,
            method="llm",
            status=final_status,
            ranked_game_ids=final_ids,
            id_to_card=id_to_card,
            score_map=baseline_score_map,
            candidate_size=candidate_size,
            notes=parsed.rationale or "LLM reranking completed.",
        )
    except Exception as exc:  # noqa: BLE001
        fallback_ids = [
            row.game_id
            for row in baseline_ranked_rows[:target_count]
            if row.game_id is not None
        ]
        return build_ranked_records(
            scenario_id=scenario.scenario_id,
            method="llm",
            status="fallback_to_baseline",
            ranked_game_ids=fallback_ids,
            id_to_card=id_to_card,
            score_map=baseline_score_map,
            candidate_size=candidate_size,
            notes=f"LLM reranking failed. Baseline order used instead. Error: {exc}",
        )


def build_ranked_records(
    scenario_id: str,
    method: str,
    status: str,
    ranked_game_ids: list[str],
    id_to_card: dict[str, GameCard],
    score_map: dict[str, float | None],
    candidate_size: int,
    notes: str,
) -> list[RecommendationRecord]:

    return [
        RecommendationRecord(
            scenario_id=scenario_id,
            method=method,
            rank=rank,
            game_id=game_id,
            game_title=id_to_card[game_id].game_title,
            score=score_map.get(game_id),
            status=status,
            candidate_size=candidate_size,
            notes=notes,
        )
        for rank, game_id in enumerate(ranked_game_ids, start=1)
    ]


def group_records_by_scenario(
    records: list[RecommendationRecord],
    method: str,
) -> dict[str, list[RecommendationRecord]]:

    grouped: dict[str, list[RecommendationRecord]] = {}
    for record in records:
        if record.method != method:
            continue
        grouped.setdefault(record.scenario_id, []).append(record)
    return grouped


def has_ranked_rows(records: list[RecommendationRecord]) -> bool:

    return any(record.rank is not None and record.game_id for record in records)


def get_candidate_size(records: list[RecommendationRecord]) -> int:

    if not records:
        return 0
    return max(record.candidate_size for record in records)


def parse_llm_response(content: str) -> LLMRankingResponse:

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("LLM response did not contain JSON.") from None
        payload = json.loads(content[start : end + 1])

    try:
        return LLMRankingResponse.model_validate(payload)
    except AttributeError:
        return LLMRankingResponse.parse_obj(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid LLM response schema: {exc}") from exc


def keep_valid_reranked_ids(ranked_ids: list[str], baseline_ids: list[str]) -> list[str]:

    allowed_ids = set(baseline_ids)
    filtered: list[str] = []
    for game_id in ranked_ids:
        normalized = str(game_id)
        if normalized in allowed_ids and normalized not in filtered:
            filtered.append(normalized)
    return filtered
