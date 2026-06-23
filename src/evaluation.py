"""Evaluate recommendation outputs and write simple report artifacts."""

from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path
import platform

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import pandas as pd

from src.baseline_tfidf import RecommendationRecord
from src.config import Settings
from src.game_card_builder import GameCard
from src.scenario_builder import Scenario


EVALUATED_STATUSES = {"ok", "partial_fallback_to_baseline", "fallback_to_baseline"}


def evaluate_recommendations(
    scenarios: list[Scenario],
    recommendation_sets: list[list[RecommendationRecord]],
    settings: Settings,
    reviews_clean_count: int,
    game_cards: list[GameCard],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate baseline and optional LLM outputs against scenario ground truth."""

    flattened_records = [
        record
        for recommendation_set in recommendation_sets
        for record in recommendation_set
    ]
    grouped_records = group_records_by_method_and_scenario(flattened_records)

    llm_rows = [record for record in flattened_records if record.method == "llm"]
    llm_has_ranked_rows = any(
        record.rank is not None and record.game_id is not None
        for record in llm_rows
    )
    methods_to_evaluate = ["baseline"]
    if llm_has_ranked_rows:
        methods_to_evaluate.append("llm")

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for method in methods_to_evaluate:
            rows.append(
                build_per_scenario_result_row(
                    scenario=scenario,
                    method=method,
                    records=grouped_records.get((method, scenario.scenario_id), []),
                )
            )

    per_scenario_df = pd.DataFrame(rows)
    per_scenario_df.to_csv(settings.per_scenario_results_path, index=False)

    metrics_summary_df = summarize_metrics(per_scenario_df)
    metrics_summary_df.to_csv(settings.metrics_summary_path, index=False)

    save_metrics_plot(metrics_summary_df, settings.metrics_plot_path)
    write_experiment_summary(
        settings=settings,
        reviews_clean_count=reviews_clean_count,
        game_cards=game_cards,
        scenarios=scenarios,
        metrics_summary_df=metrics_summary_df,
        llm_rows=llm_rows,
        llm_has_ranked_rows=llm_has_ranked_rows,
    )

    return per_scenario_df, metrics_summary_df


def group_records_by_method_and_scenario(
    records: list[RecommendationRecord],
) -> dict[tuple[str, str], list[RecommendationRecord]]:
    """Group recommendation rows by method and scenario id."""

    grouped: dict[tuple[str, str], list[RecommendationRecord]] = {}
    for record in records:
        key = (record.method, record.scenario_id)
        grouped.setdefault(key, []).append(record)
    return grouped


def build_per_scenario_result_row(
    scenario: Scenario,
    method: str,
    records: list[RecommendationRecord],
) -> dict[str, object]:
    """Build one evaluation row for a method-scenario pair."""

    candidate_size = max((record.candidate_size for record in records), default=0)
    ground_truth_count = len(scenario.ground_truth_game_ids)
    ranked_records = sorted(
        [record for record in records if record.rank is not None and record.game_id],
        key=lambda record: record.rank or 0,
    )
    status = records[0].status if records else "missing_results"

    row = {
        "scenario_id": scenario.scenario_id,
        "scenario_type": scenario.scenario_type,
        "method": method,
        "hit_rate_at_5": None,
        "hit_rate_at_10": None,
        "mrr": None,
        "ndcg_at_10": None,
        "candidate_size": candidate_size,
        "ground_truth_count": ground_truth_count,
        "status": status,
    }

    if ground_truth_count == 0:
        row["status"] = "skipped_empty_ground_truth"
        return row

    if not ranked_records:
        return row

    if status not in EVALUATED_STATUSES:
        return row

    ranked_ids = [record.game_id for record in ranked_records if record.game_id is not None]
    truth_ids = scenario.ground_truth_game_ids
    row["hit_rate_at_5"] = hit_rate_at_k(ranked_ids, truth_ids, 5)
    row["hit_rate_at_10"] = hit_rate_at_k(ranked_ids, truth_ids, 10)
    row["mrr"] = mean_reciprocal_rank(ranked_ids, truth_ids)
    row["ndcg_at_10"] = ndcg_at_k(ranked_ids, truth_ids, 10)
    return row


def hit_rate_at_k(ranked_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    """Return 1.0 if any relevant game appears in the top-k ranking."""

    truth_set = set(ground_truth_ids)
    return float(any(game_id in truth_set for game_id in ranked_ids[:k]))


def mean_reciprocal_rank(ranked_ids: list[str], ground_truth_ids: list[str]) -> float:
    """Compute reciprocal rank for the first relevant hit."""

    truth_set = set(ground_truth_ids)
    for index, game_id in enumerate(ranked_ids, start=1):
        if game_id in truth_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    """Compute a simple binary-relevance NDCG at k."""

    truth_set = set(ground_truth_ids)
    dcg = 0.0
    for index, game_id in enumerate(ranked_ids[:k], start=1):
        if game_id in truth_set:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(ground_truth_ids), k)
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def summarize_metrics(per_scenario_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by method using evaluated scenarios only."""

    if per_scenario_df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "evaluated_scenarios",
                "skipped_scenarios",
                "mean_hit_rate_at_5",
                "mean_hit_rate_at_10",
                "mean_mrr",
                "mean_ndcg_at_10",
            ]
        )

    summary_rows: list[dict[str, object]] = []
    for method, method_df in per_scenario_df.groupby("method", sort=False):
        evaluated_df = method_df[method_df["status"].isin(EVALUATED_STATUSES)]
        if evaluated_df.empty:
            continue
        summary_rows.append(
            {
                "method": method,
                "evaluated_scenarios": int(len(evaluated_df)),
                "skipped_scenarios": int(len(method_df) - len(evaluated_df)),
                "mean_hit_rate_at_5": round(evaluated_df["hit_rate_at_5"].mean(), 4),
                "mean_hit_rate_at_10": round(evaluated_df["hit_rate_at_10"].mean(), 4),
                "mean_mrr": round(evaluated_df["mrr"].mean(), 4),
                "mean_ndcg_at_10": round(evaluated_df["ndcg_at_10"].mean(), 4),
            }
        )

    return pd.DataFrame(summary_rows)


def save_metrics_plot(metrics_summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a small comparison figure for aggregate metrics."""

    metric_columns = [
        "mean_hit_rate_at_5",
        "mean_hit_rate_at_10",
        "mean_mrr",
        "mean_ndcg_at_10",
    ]

    if metrics_summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No evaluated methods", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    plot_df = metrics_summary_df.set_index("method")[metric_columns]
    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Score")
    ax.set_xlabel("Method")
    ax.set_title("Recommendation Metric Comparison")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_experiment_summary(
    settings: Settings,
    reviews_clean_count: int,
    game_cards: list[GameCard],
    scenarios: list[Scenario],
    metrics_summary_df: pd.DataFrame,
    llm_rows: list[RecommendationRecord],
    llm_has_ranked_rows: bool,
) -> None:
    """Write a compact markdown experiment summary."""

    scenario_type_distribution = dict(Counter(scenario.scenario_type for scenario in scenarios))
    scenario_source = determine_scenario_source(scenarios)
    llm_summary_line = build_llm_summary_line(llm_rows, llm_has_ranked_rows)
    unique_game_count = len({card.game_id for card in game_cards})
    manual_scenario_count = sum(
        1 for scenario in scenarios if scenario.scenario_type in {"manual", "seed_games"}
    )
    validation_summary = load_validation_summary(settings.scenario_validation_report_path)
    dataset_too_small = (
        reviews_clean_count < settings.min_reviews_for_real_experiment
        or len(game_cards) < settings.min_game_cards_for_real_experiment
    )
    warnings: list[str] = []
    if dataset_too_small:
        warnings.append(
            "Dataset is below the configured thesis-scale thresholds and is too small for strong conclusions."
        )
    if scenario_source == "synthetic_demo":
        warnings.append(
            "Only synthetic demo scenarios were used. These results are suitable for technical validation only."
        )
    if validation_summary["invalid_count"] > 0:
        warnings.append(
            f"Scenario validation currently reports {validation_summary['invalid_count']} invalid scenarios."
        )
    llm_status_line = (
        f"- {llm_summary_line}"
        if llm_summary_line.startswith("llm_status = ")
        else f"- LLM status: {llm_summary_line}"
    )

    summary_lines = [
        "# Experiment Summary",
        "",
        "## Run Overview",
        f"- Reviews CSV: `{settings.reviews_csv_path}`",
        llm_status_line,
        "",
        "## Dataset and Scenario Summary",
        f"- Dataset size: {reviews_clean_count} cleaned reviews",
        f"- Number of games: {unique_game_count}",
        f"- Number of game cards: {len(game_cards)}",
        f"- Number of scenarios: {len(scenarios)}",
        f"- Scenario type distribution: {format_distribution(scenario_type_distribution)}",
        f"- Scenario mode: {scenario_source}",
        f"- Manual/predefined scenarios: {manual_scenario_count}",
        "",
        "## Warnings",
    ]
    if warnings:
        summary_lines.extend(f"- {warning}" for warning in warnings)
    else:
        summary_lines.append("- No major experiment-readiness warnings were detected in this run.")

    summary_lines.extend(
        [
            "",
            "## Aggregate Metrics",
            dataframe_to_markdown(metrics_summary_df),
            "",
            "## Scientific Validity Notes",
            "- No user identifiers are available in the Steam Reviews dataset.",
            "- Real user histories cannot be reconstructed from this dataset.",
            "- Scenario-based preference profiles are therefore used instead of user-level offline evaluation.",
            "- Synthetic demo scenarios are only for technical validation.",
            "- Manual scenarios are recommended for the final thesis experiment.",
            "- Metrics on the tiny bundled sample data should not be interpreted as scientific results.",
            "- The LLM reranker, when enabled, only reorders baseline candidates.",
            "",
            "## Report-Ready Thesis Artifacts",
            "- `data/results/environment_check.json`: machine-readable environment and dependency check report.",
            "- `data/results/environment_check.md`: markdown summary of local environment readiness.",
            "- `data/results/preflight_report.md`: stricter pre-flight classification before a real thesis experiment.",
            "- `data/results/smoke_test_report.json`: latest baseline-only smoke test result.",
            "- `data/results/data_diagnostics.md`: dataset overview and descriptive statistics.",
            "- `data/results/available_games.csv`: export of valid games for scenario authoring.",
            "- `data/results/available_games.md`: markdown version of the available games list.",
            "- `data/results/scenario_validation_report.csv`: scenario validation outcomes and warnings.",
            "- `data/results/experiment_readiness.json`: readiness summary for thesis-scale evaluation.",
            "- `reports/case_studies.md`: representative success and failure examples.",
            "- `reports/rank_comparison.md`: scenario-level baseline versus LLM rank comparison.",
            "- `reports/recommendation_examples.md`: recommendation examples for the experimental chapter.",
            "- `reports/thesis_metrics_table.md`: markdown-ready metrics table.",
            "- `reports/thesis_dataset_table.md`: markdown-ready dataset summary table.",
            "- `reports/thesis_scenario_table.md`: markdown-ready scenario coverage table.",
            "- `reports/llm_explanation_checks.md`: heuristic LLM explanation checks.",
            "",
            "## Execution and Reproducibility",
            f"- Python version: {platform.python_version()}",
            f"- LLM credentials configured: {bool(settings.openai_api_key and settings.llm_model)}",
            f"- Experiment scale classification: {'sample_or_technical_validation' if dataset_too_small or scenario_source == 'synthetic_demo' else 'real_experiment_candidate'}",
            "- Recommended reproduction command: `./.venv/bin/python main.py`",
            "",
            "## Recommended Next Steps",
            *build_recommended_next_steps(
                dataset_too_small=dataset_too_small,
                scenario_source=scenario_source,
                llm_summary_line=llm_summary_line,
                manual_scenario_count=manual_scenario_count,
                settings=settings,
                validation_summary=validation_summary,
            ),
        ]
    )

    settings.experiment_summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def determine_scenario_source(scenarios: list[Scenario]) -> str:
    """Describe whether predefined or synthetic scenarios were used."""

    if scenarios and all(scenario.scenario_type == "synthetic_demo" for scenario in scenarios):
        return "synthetic_demo"
    if scenarios and all(scenario.scenario_type in {"manual", "seed_games"} for scenario in scenarios):
        return "manual_or_predefined"
    return "mixed"


def build_llm_summary_line(
    llm_rows: list[RecommendationRecord],
    llm_has_ranked_rows: bool,
) -> str:
    """Summarize whether the LLM reranker ran or was skipped."""

    if not llm_rows:
        return "no llm output"

    if not llm_has_ranked_rows:
        unique_statuses = sorted({row.status for row in llm_rows})
        if unique_statuses == ["skipped_no_credentials"]:
            return "llm_status = skipped_no_credentials"
        return "skipped (" + ", ".join(unique_statuses) + ")"

    status_counts = (
        pd.Series([row.status for row in llm_rows])
        .value_counts()
        .to_dict()
    )
    parts = [f"{status}: {count}" for status, count in status_counts.items()]
    return "ran (" + ", ".join(parts) + ")"


def format_distribution(distribution: dict[str, int]) -> str:
    """Format a small scenario-type distribution for markdown output."""

    if not distribution:
        return "none"
    return ", ".join(
        f"{scenario_type}: {count}"
        for scenario_type, count in sorted(distribution.items())
    )


def load_validation_summary(path: Path) -> dict[str, int]:
    """Load a tiny summary of the most recent scenario validation report."""

    if not path.exists():
        return {"ok_count": 0, "warning_count": 0, "invalid_count": 0}

    frame = pd.read_csv(path)
    return {
        "ok_count": int((frame["status"] == "ok").sum()),
        "warning_count": int((frame["status"] == "warning").sum()),
        "invalid_count": int((frame["status"] == "invalid").sum()),
    }


def build_recommended_next_steps(
    dataset_too_small: bool,
    scenario_source: str,
    llm_summary_line: str,
    manual_scenario_count: int,
    settings: Settings,
    validation_summary: dict[str, int],
) -> list[str]:
    """Generate a small set of next-step bullets from the current run state."""

    steps: list[str] = []
    if dataset_too_small:
        steps.append(
            "- Replace the bundled sample CSV with the real Steam Reviews dataset before drawing thesis conclusions."
        )
    if scenario_source == "synthetic_demo":
        steps.append(
            "- Create or review manual scenarios before using the metrics in the thesis."
        )
    if llm_summary_line == "llm_status = skipped_no_credentials":
        steps.append(
            "- Configure `OPENAI_API_KEY` and `OPENAI_MODEL` to run LLM reranking, or report baseline-only results."
        )
    if validation_summary["invalid_count"] > 0:
        steps.append(
            "- Fix invalid scenarios listed in `data/results/scenario_validation_report.csv` before the main experiment."
        )
    if (
        not dataset_too_small
        and scenario_source != "synthetic_demo"
        and validation_summary["invalid_count"] == 0
        and manual_scenario_count >= settings.min_manual_scenarios_for_real_experiment
    ):
        steps.append("- The project appears ready for the main thesis experiment.")
    if not steps:
        steps.append("- Continue reviewing artifacts and scenario quality before writing up results.")
    return steps


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small markdown table without optional dependencies."""

    if frame.empty:
        return "_No metrics available._"

    headers = [str(column) for column in frame.columns]
    separator = ["---"] * len(headers)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in frame.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
