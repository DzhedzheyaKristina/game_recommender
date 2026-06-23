"""Entrypoint for the scenario-based Steam review recommender prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.environment_tools import (
    REQUIRED_IMPORTS,
    find_missing_runtime_dependencies,
    run_environment_check,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Steam review recommender prototype")
    parser.add_argument(
        "--step",
        default="all",
        choices=[
            "check_env",
            "discover_datasets",
            "inspect_raw",
            "schema_check",
            "preflight",
            "run_experiment",
            "preprocess",
            "preprocess_debug",
            "preprocess_subset",
            "preprocess_balanced_subset",
            "preprocess_status",
            "build_cards",
            "build_scenarios",
            "baseline",
            "llm",
            "evaluate",
            "data_diagnostics",
            "case_studies",
            "recommendation_examples",
            "thesis_tables",
            "analysis",
            "select_final_experiment",
            "export_thesis_results",
            "demo_info",
            "validate_scenarios",
            "readiness",
            "list_games",
            "draft_scenarios",
            "normalize_scenarios",
            "build_user_profiles",
            "build_user_splits",
            "build_user_splits_pilot",
            "user_baseline",
            "user_evaluate",
            "user_experiment",
            "user_llm",
            "user_llm_dry_run",
            "llm_check",
            "llm_pilot_readiness",
            "user_llm_evaluate",
            "smoke_test",
            "all",
        ],
        help="Run a single pipeline step or the full pipeline.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to an experiment configuration JSON file.",
    )
    parser.add_argument(
        "--allow-tiny",
        action="store_true",
        help="Allow tiny datasets in run_experiment for debugging.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic scenarios in run_experiment for debugging.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        help="Request LLM reranking in run_experiment.",
    )
    return parser.parse_args()


def ensure_runtime_dependencies(step: str) -> None:
    """Stop early with a readable message when key packages are missing."""

    if step == "check_env":
        return

    missing_modules = find_missing_runtime_dependencies()
    if not missing_modules:
        return

    first_missing = missing_modules[0]
    package_name = REQUIRED_IMPORTS[first_missing]
    print(f"Missing required dependency: {package_name}")
    print("Please activate the project virtual environment and install dependencies:")
    print("python -m venv .venv")
    print("source .venv/bin/activate")
    print("pip install -r requirements.txt")
    print("Or run with:")
    print("./.venv/bin/python main.py")
    sys.exit(1)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the requested part of the pipeline."""

    step = args.step
    project_root = Path(__file__).resolve().parent
    if step == "check_env":
        report = run_environment_check(project_root)
        summary = report["summary"]
        print(
            "Environment check completed: "
            f"{summary['ok']} ok, {summary['warnings']} warnings, {summary['errors']} errors."
        )
        return

    ensure_runtime_dependencies(step)

    import pandas as pd

    from src.baseline_tfidf import RecommendationRecord, run_baseline
    from src.config import Settings, load_settings
    from src.data_loader import discover_datasets, inspect_raw_dataset, load_reviews_csv, run_schema_check
    from src.evaluation import evaluate_recommendations
    from src.experiment_tools import (
        apply_experiment_settings_overrides,
        load_experiment_config,
        build_experiment_readiness_report,
        build_preflight_report,
        build_llm_pilot_readiness_report,
        generate_case_studies,
        generate_draft_scenarios,
        generate_recommendation_examples,
        generate_thesis_tables,
        export_available_games,
        export_thesis_results,
        normalize_scenarios_file,
        select_final_experiment,
        run_controlled_experiment,
        run_analysis_suite,
        run_data_diagnostics,
        run_smoke_test,
        validate_scenarios_from_artifacts,
    )
    from src.game_card_builder import GameCard, build_game_cards
    from src.llm_reranker import run_llm_reranker
    from src.preprocessing import (
        preprocess_reviews,
        print_preprocessing_status,
        run_balanced_subset_preprocessing,
        run_chunked_preprocessing,
    )
    from src.scenario_builder import Scenario, build_scenarios
    from src.user_experiments import (
        build_user_evaluation_splits,
        build_user_splits_pilot,
        build_user_profiles,
        evaluate_user_baseline,
        run_user_baseline,
        run_user_experiment,
    )
    from src.user_llm import evaluate_user_llm, run_llm_check, run_user_llm_dry_run, run_user_llm_pilot
    from src.utils import ensure_directories, get_logger

    def prepare_directories(settings: Settings) -> None:
        """Ensure the required output directories exist."""

        ensure_directories(
            [
                settings.raw_data_dir,
                settings.scenarios_dir,
                settings.processed_data_dir,
                settings.results_dir,
                settings.reports_dir,
                settings.figures_dir,
                settings.prompts_dir,
            ]
        )

    settings = load_settings()
    logger = get_logger()
    prepare_directories(settings)

    config_override_steps = {
        "build_user_profiles",
        "build_user_splits",
        "build_user_splits_pilot",
        "preflight",
        "readiness",
        "user_baseline",
        "user_evaluate",
        "user_experiment",
        "user_llm",
        "user_llm_dry_run",
        "user_llm_evaluate",
        "llm_check",
        "llm_pilot_readiness",
        "select_final_experiment",
        "export_thesis_results",
        "demo_info",
    }
    if args.config and step in config_override_steps:
        config_payload, _ = load_experiment_config(settings.project_root, config_path=Path(args.config))
        apply_experiment_settings_overrides(settings, config_payload)

    reviews_df: pd.DataFrame | None = None
    cleaned_reviews_df: pd.DataFrame | None = None
    game_cards: list[GameCard] = []
    scenarios: list[Scenario] = []
    baseline_results: list[RecommendationRecord] = []

    if step == "list_games":
        export_available_games(settings)
        return

    if step == "discover_datasets":
        reports = discover_datasets(settings)
        print(f"Dataset discovery completed: {len(reports)} CSV file(s) inspected.")
        return

    if step == "inspect_raw":
        report = inspect_raw_dataset(settings)
        print(
            "Raw inspection completed: "
            f"rows={report['raw_row_count']}, "
            f"author.steamid_non_empty={report['author_steamid_non_empty_count']}"
        )
        return

    if step == "schema_check":
        report = run_schema_check(settings)
        print(
            "Schema check completed: "
            f"status={report['status']}, "
            f"user_mode={report['required_for_user_mode_present']}"
        )
        return

    if step == "data_diagnostics":
        run_data_diagnostics(settings)
        return

    if step == "preflight":
        build_preflight_report(settings)
        return

    if step == "preprocess":
        run_chunked_preprocessing(settings, mode="full")
        logger.info("Saved cleaned reviews to %s", settings.reviews_clean_path)
        return

    if step == "preprocess_debug":
        run_chunked_preprocessing(settings, mode="debug")
        logger.info("Saved debug cleaned reviews to %s", settings.reviews_clean_debug_path)
        return

    if step == "preprocess_subset":
        run_chunked_preprocessing(settings, mode="subset")
        logger.info("Saved subset cleaned reviews to %s", settings.reviews_clean_subset_path)
        return

    if step == "preprocess_balanced_subset":
        run_balanced_subset_preprocessing(settings)
        logger.info(
            "Saved balanced subset cleaned reviews to %s",
            settings.reviews_clean_balanced_subset_path,
        )
        return

    if step == "preprocess_status":
        print_preprocessing_status(settings)
        return

    if step == "run_experiment":
        run_controlled_experiment(
            settings=settings,
            config_path=Path(args.config) if args.config else None,
            allow_tiny_override=args.allow_tiny,
            allow_synthetic_override=args.allow_synthetic,
            run_llm_override=args.run_llm,
        )
        return

    if step == "draft_scenarios":
        generate_draft_scenarios(settings)
        return

    if step == "build_user_profiles":
        build_user_profiles(settings)
        return

    if step == "build_user_splits":
        build_user_evaluation_splits(settings)
        return

    if step == "build_user_splits_pilot":
        build_user_splits_pilot(settings)
        return

    if step == "user_baseline":
        run_user_baseline(settings)
        return

    if step == "user_evaluate":
        evaluate_user_baseline(settings)
        return

    if step == "user_experiment":
        run_user_experiment(settings)
        return

    if step == "user_llm_dry_run":
        if args.config:
            config_payload, _ = load_experiment_config(settings.project_root, config_path=Path(args.config))
            apply_experiment_settings_overrides(settings, config_payload)
        run_user_llm_dry_run(settings)
        return

    if step == "llm_check":
        if args.config:
            config_payload, _ = load_experiment_config(settings.project_root, config_path=Path(args.config))
            apply_experiment_settings_overrides(settings, config_payload)
        run_llm_check(settings)
        return

    if step == "llm_pilot_readiness":
        build_llm_pilot_readiness_report(settings)
        return

    if step == "user_llm":
        run_user_llm_pilot(settings)
        return

    if step == "user_llm_evaluate":
        evaluate_user_llm(settings)
        return

    if step == "case_studies":
        generate_case_studies(settings)
        return

    if step == "recommendation_examples":
        generate_recommendation_examples(settings)
        return

    if step == "thesis_tables":
        generate_thesis_tables(settings)
        return

    if step == "analysis":
        run_analysis_suite(settings)
        return

    if step == "select_final_experiment":
        select_final_experiment(settings)
        return

    if step == "export_thesis_results":
        export_thesis_results(settings)
        return

    if step == "demo_info":
        final_dir = settings.reports_dir / "final_thesis_artifacts"
        required_files = [
            "final_experiment_summary.md",
            "experiment_manifest.json",
            "user_llm_reranking_summary.json",
            "user_llm_metrics_summary.csv",
            "user_rank_comparison.csv",
            "user_llm_explanation_examples.md",
            "balanced_subset_methodology_note.md",
        ]
        print("To launch the local demo interface:")
        print("streamlit run app.py")
        print()
        print("Artifact check:")
        for filename in required_files:
            exists = (final_dir / filename).exists()
            status = "OK" if exists else "MISSING"
            print(f"- {filename}: {status}")
        return

    if step == "normalize_scenarios":
        normalize_scenarios_file(settings)
        return

    if step == "validate_scenarios":
        validate_scenarios_from_artifacts(settings)
        return

    if step == "readiness":
        build_experiment_readiness_report(settings)
        return

    if step == "smoke_test":
        run_smoke_test(settings)
        return

    if step == "all":
        reviews_df = load_reviews_csv(settings)
        cleaned_reviews_df = preprocess_reviews(reviews_df, settings)
        logger.info("Saved cleaned reviews to %s", settings.reviews_clean_path)

    if step in {"build_cards", "build_scenarios", "baseline", "llm", "evaluate"}:
        if not settings.active_processed_reviews_path.exists():
            raise RuntimeError(
                "Cleaned reviews are required before downstream steps. Run `./.venv/bin/python main.py --step preprocess` first."
            )
        cleaned_reviews_df = pd.read_csv(settings.active_processed_reviews_path)

    if step in {"build_cards", "build_scenarios", "baseline", "llm", "evaluate", "all"}:
        if cleaned_reviews_df is None:
            raise RuntimeError("Cleaned reviews are required before building game cards.")
        game_cards = build_game_cards(cleaned_reviews_df, settings)
        logger.info("Saved %s game cards to %s", len(game_cards), settings.game_cards_path)
        if step == "build_cards":
            return

    if step in {"build_scenarios", "baseline", "llm", "evaluate", "all"}:
        scenarios = build_scenarios(game_cards, settings)
        logger.info("Saved %s scenarios to %s", len(scenarios), settings.scenarios_output_path)
        if step == "build_scenarios":
            return

    if step in {"baseline", "llm", "evaluate", "all"}:
        baseline_results = run_baseline(scenarios, game_cards, settings)
        logger.info("Saved baseline recommendations to %s", settings.baseline_results_path)
        if step == "baseline":
            return

    llm_results: list[RecommendationRecord] = []
    if step in {"llm", "evaluate", "all"}:
        llm_results = run_llm_reranker(scenarios, baseline_results, game_cards, settings)
        logger.info("Saved LLM recommendation records to %s", settings.llm_results_path)
        if step == "llm":
            return

    if step in {"evaluate", "all"}:
        if cleaned_reviews_df is None:
            raise RuntimeError("Cleaned reviews are required for evaluation.")
        evaluate_recommendations(
            scenarios=scenarios,
            recommendation_sets=[baseline_results, llm_results],
            settings=settings,
            reviews_clean_count=len(cleaned_reviews_df),
            game_cards=game_cards,
        )
        logger.info("Saved evaluation outputs to %s", settings.results_dir)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
