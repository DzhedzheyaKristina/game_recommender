
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

from src.environment_tools import (
    REQUIRED_IMPORTS,
    find_missing_runtime_dependencies,
    run_environment_check,
)


DEMO_REQUIRED_FILES = [
    "final_experiment_summary.md",
    "experiment_manifest.json",
    "user_llm_reranking_summary.json",
    "user_llm_metrics_summary.csv",
    "user_rank_comparison.csv",
    "user_llm_explanation_examples.md",
    "balanced_subset_methodology_note.md",
]


def parse_args() -> argparse.Namespace:

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


def run_demo_info_step(reports_dir: Path) -> None:

    final_dir = reports_dir / "final_thesis_artifacts"
    print("To launch the local demo interface:")
    print("streamlit run app.py")
    print()
    print("Artifact check:")
    for filename in DEMO_REQUIRED_FILES:
        exists = (final_dir / filename).exists()
        status = "OK" if exists else "MISSING"
        print(f"- {filename}: {status}")


def run_pipeline(args: argparse.Namespace) -> None:

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

    def apply_optional_config_override(settings: Settings) -> None:

        if not args.config or step not in config_override_steps:
            return
        config_payload, _ = load_experiment_config(settings.project_root, config_path=Path(args.config))
        apply_experiment_settings_overrides(settings, config_payload)

    def run_discover_datasets_step() -> None:
        reports = discover_datasets(settings)
        print(f"Dataset discovery completed: {len(reports)} CSV file(s) inspected.")

    def run_inspect_raw_step() -> None:
        report = inspect_raw_dataset(settings)
        print(
            "Raw inspection completed: "
            f"rows={report['raw_row_count']}, "
            f"author.steamid_non_empty={report['author_steamid_non_empty_count']}"
        )

    def run_schema_check_step() -> None:
        report = run_schema_check(settings)
        print(
            "Schema check completed: "
            f"status={report['status']}, "
            f"user_mode={report['required_for_user_mode_present']}"
        )

    def run_chunked_preprocessing_step(mode: str, output_path: Path, label: str) -> None:
        run_chunked_preprocessing(settings, mode=mode)
        logger.info("Saved %s to %s", label, output_path)

    def run_balanced_subset_preprocessing_step() -> None:
        run_balanced_subset_preprocessing(settings)
        logger.info(
            "Saved balanced subset cleaned reviews to %s",
            settings.reviews_clean_balanced_subset_path,
        )

    def run_experiment_step() -> None:
        run_controlled_experiment(
            settings=settings,
            config_path=Path(args.config) if args.config else None,
            allow_tiny_override=args.allow_tiny,
            allow_synthetic_override=args.allow_synthetic,
            run_llm_override=args.run_llm,
        )

    def load_required_cleaned_reviews() -> pd.DataFrame:
        if not settings.active_processed_reviews_path.exists():
            raise RuntimeError(
                "Cleaned reviews are required before downstream steps. Run `./.venv/bin/python main.py --step preprocess` first."
            )
        return pd.read_csv(settings.active_processed_reviews_path)

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
    apply_optional_config_override(settings)

    simple_step_handlers: dict[str, Callable[[], None]] = {
        "list_games": lambda: export_available_games(settings),
        "discover_datasets": run_discover_datasets_step,
        "inspect_raw": run_inspect_raw_step,
        "schema_check": run_schema_check_step,
        "data_diagnostics": lambda: run_data_diagnostics(settings),
        "preflight": lambda: build_preflight_report(settings),
        "preprocess": lambda: run_chunked_preprocessing_step("full", settings.reviews_clean_path, "cleaned reviews"),
        "preprocess_debug": lambda: run_chunked_preprocessing_step("debug", settings.reviews_clean_debug_path, "debug cleaned reviews"),
        "preprocess_subset": lambda: run_chunked_preprocessing_step("subset", settings.reviews_clean_subset_path, "subset cleaned reviews"),
        "preprocess_balanced_subset": run_balanced_subset_preprocessing_step,
        "preprocess_status": lambda: print_preprocessing_status(settings),
        "run_experiment": run_experiment_step,
        "draft_scenarios": lambda: generate_draft_scenarios(settings),
        "build_user_profiles": lambda: build_user_profiles(settings),
        "build_user_splits": lambda: build_user_evaluation_splits(settings),
        "build_user_splits_pilot": lambda: build_user_splits_pilot(settings),
        "user_baseline": lambda: run_user_baseline(settings),
        "user_evaluate": lambda: evaluate_user_baseline(settings),
        "user_experiment": lambda: run_user_experiment(settings),
        "user_llm_dry_run": lambda: run_user_llm_dry_run(settings),
        "llm_check": lambda: run_llm_check(settings),
        "llm_pilot_readiness": lambda: build_llm_pilot_readiness_report(settings),
        "user_llm": lambda: run_user_llm_pilot(settings),
        "user_llm_evaluate": lambda: evaluate_user_llm(settings),
        "case_studies": lambda: generate_case_studies(settings),
        "recommendation_examples": lambda: generate_recommendation_examples(settings),
        "thesis_tables": lambda: generate_thesis_tables(settings),
        "analysis": lambda: run_analysis_suite(settings),
        "select_final_experiment": lambda: select_final_experiment(settings),
        "export_thesis_results": lambda: export_thesis_results(settings),
        "demo_info": lambda: run_demo_info_step(settings.reports_dir),
        "normalize_scenarios": lambda: normalize_scenarios_file(settings),
        "validate_scenarios": lambda: validate_scenarios_from_artifacts(settings),
        "readiness": lambda: build_experiment_readiness_report(settings),
        "smoke_test": lambda: run_smoke_test(settings),
    }

    if step in simple_step_handlers:
        simple_step_handlers[step]()
        return

    reviews_df: pd.DataFrame | None = None
    cleaned_reviews_df: pd.DataFrame | None = None
    game_cards: list[GameCard] = []
    scenarios: list[Scenario] = []
    baseline_results: list[RecommendationRecord] = []

    pipeline_steps = {"build_cards", "build_scenarios", "baseline", "llm", "evaluate", "all"}
    scenario_steps = {"build_scenarios", "baseline", "llm", "evaluate", "all"}
    baseline_steps = {"baseline", "llm", "evaluate", "all"}
    llm_steps = {"llm", "evaluate", "all"}
    evaluation_steps = {"evaluate", "all"}

    if step == "all":
        reviews_df = load_reviews_csv(settings)
        cleaned_reviews_df = preprocess_reviews(reviews_df, settings)
        logger.info("Saved cleaned reviews to %s", settings.reviews_clean_path)

    if step in pipeline_steps - {"all"}:
        cleaned_reviews_df = load_required_cleaned_reviews()

    if step in pipeline_steps:
        if cleaned_reviews_df is None:
            raise RuntimeError("Cleaned reviews are required before building game cards.")
        game_cards = build_game_cards(cleaned_reviews_df, settings)
        logger.info("Saved %s game cards to %s", len(game_cards), settings.game_cards_path)
        if step == "build_cards":
            return

    if step in scenario_steps:
        scenarios = build_scenarios(game_cards, settings)
        logger.info("Saved %s scenarios to %s", len(scenarios), settings.scenarios_output_path)
        if step == "build_scenarios":
            return

    if step in baseline_steps:
        baseline_results = run_baseline(scenarios, game_cards, settings)
        logger.info("Saved baseline recommendations to %s", settings.baseline_results_path)
        if step == "baseline":
            return

    llm_results: list[RecommendationRecord] = []
    if step in llm_steps:
        llm_results = run_llm_reranker(scenarios, baseline_results, game_cards, settings)
        logger.info("Saved LLM recommendation records to %s", settings.llm_results_path)
        if step == "llm":
            return

    if step in evaluation_steps:
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
