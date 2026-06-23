"""Configuration for the research prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    """Application settings loaded from defaults and optional environment variables."""

    project_root: Path
    raw_data_dir: Path
    scenarios_dir: Path
    processed_data_dir: Path
    results_dir: Path
    reports_dir: Path
    figures_dir: Path
    prompts_dir: Path
    reviews_csv_path: Path
    scenarios_file: Path | None
    reviews_clean_path: Path
    active_processed_reviews_path: Path
    reviews_clean_debug_path: Path
    reviews_clean_subset_path: Path
    reviews_clean_balanced_subset_path: Path
    dataset_discovery_report_csv_path: Path
    dataset_discovery_report_json_path: Path
    dataset_discovery_report_markdown_path: Path
    raw_dataset_inspection_json_path: Path
    raw_dataset_inspection_markdown_path: Path
    schema_validation_report_json_path: Path
    schema_validation_report_markdown_path: Path
    preprocessing_summary_path: Path
    preprocessing_summary_markdown_path: Path
    preprocessing_debug_summary_path: Path
    preprocessing_debug_summary_markdown_path: Path
    preprocessing_subset_summary_path: Path
    preprocessing_subset_summary_markdown_path: Path
    preprocessing_balanced_subset_summary_path: Path
    preprocessing_balanced_subset_summary_markdown_path: Path
    preprocessing_progress_json_path: Path
    preprocessing_progress_markdown_path: Path
    raw_processed_comparison_json_path: Path
    raw_processed_comparison_markdown_path: Path
    raw_processed_subset_comparison_json_path: Path
    raw_processed_subset_comparison_markdown_path: Path
    raw_processed_balanced_subset_comparison_json_path: Path
    raw_processed_balanced_subset_comparison_markdown_path: Path
    raw_game_sampling_stats_csv_path: Path
    raw_game_sampling_stats_json_path: Path
    game_card_summary_path: Path
    game_cards_path: Path
    user_profiles_path: Path
    user_profile_summary_path: Path
    user_splits_path: Path
    pilot_splits_path: Path
    user_split_summary_path: Path
    user_split_diagnostics_json_path: Path
    user_split_diagnostics_markdown_path: Path
    user_evaluation_split_pilot_summary_path: Path
    user_evaluation_split_pilot_summary_markdown_path: Path
    user_baseline_results_path: Path
    user_per_profile_results_path: Path
    user_metrics_summary_path: Path
    user_llm_prompt_preview_markdown_path: Path
    user_llm_prompt_preview_json_path: Path
    user_llm_prompt_preview_tiny_markdown_path: Path
    user_llm_prompt_preview_tiny_json_path: Path
    user_llm_prompt_preview_10_gigachat_markdown_path: Path
    user_llm_prompt_preview_10_gigachat_json_path: Path
    user_llm_results_path: Path
    user_llm_reranking_summary_path: Path
    user_llm_validation_summary_path: Path
    user_llm_per_profile_results_path: Path
    user_llm_metrics_summary_path: Path
    user_llm_metrics_summary_all_pilot_path: Path
    user_rank_comparison_path: Path
    user_rank_comparison_markdown_path: Path
    user_llm_explanation_checks_csv_path: Path
    user_llm_explanation_checks_markdown_path: Path
    user_llm_explanation_examples_markdown_path: Path
    user_llm_schema_error_report_json_path: Path
    user_llm_schema_error_report_markdown_path: Path
    llm_pilot_candidate_user_report_csv_path: Path
    llm_pilot_candidate_user_report_markdown_path: Path
    user_llm_metrics_table_path: Path
    user_llm_pilot_summary_path: Path
    user_llm_mock_validation_summary_path: Path
    user_llm_mock_validation_report_path: Path
    user_llm_failure_report_json_path: Path
    user_llm_failure_report_markdown_path: Path
    llm_check_json_path: Path
    llm_check_markdown_path: Path
    llm_pilot_readiness_json_path: Path
    llm_pilot_readiness_markdown_path: Path
    scenarios_output_path: Path
    scenario_validation_report_path: Path
    environment_check_json_path: Path
    environment_check_markdown_path: Path
    preflight_report_json_path: Path
    preflight_report_markdown_path: Path
    smoke_test_report_path: Path
    data_diagnostics_json_path: Path
    data_diagnostics_markdown_path: Path
    available_games_path: Path
    available_games_markdown_path: Path
    experiment_readiness_path: Path
    case_studies_json_path: Path
    rank_comparison_csv_path: Path
    llm_explanation_checks_csv_path: Path
    manual_scenarios_template_jsonl_path: Path
    manual_scenarios_template_csv_path: Path
    draft_scenarios_jsonl_path: Path
    draft_scenarios_csv_path: Path
    baseline_results_path: Path
    llm_results_path: Path
    per_scenario_results_path: Path
    metrics_summary_path: Path
    metrics_plot_path: Path
    review_language_distribution_plot_path: Path
    review_recommendation_distribution_plot_path: Path
    reviews_per_user_distribution_plot_path: Path
    reviews_per_game_distribution_plot_path: Path
    playtime_forever_distribution_plot_path: Path
    top_games_by_review_count_plot_path: Path
    experiment_summary_path: Path
    case_studies_markdown_path: Path
    rank_comparison_markdown_path: Path
    llm_explanation_checks_markdown_path: Path
    recommendation_examples_markdown_path: Path
    thesis_metrics_table_path: Path
    thesis_dataset_table_path: Path
    thesis_scenario_table_path: Path
    system_prompt_path: Path
    reranking_prompt_template_path: Path
    required_columns: tuple[str, ...] = (
        "app_id",
        "app_name",
        "review_id",
        "language",
        "review",
        "timestamp_created",
        "timestamp_updated",
        "recommended",
        "votes_helpful",
    )
    expected_full_schema_columns: tuple[str, ...] = (
        "Index",
        "app_id",
        "app_name",
        "review_id",
        "language",
        "review",
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
        "author.steamid",
        "author.num_games_owned",
        "author.num_reviews",
        "author.playtime_forever",
        "author.playtime_last_two_weeks",
        "author.playtime_at_review",
        "author.last_played",
    )
    language_filter: str | None = field(default=None)
    min_review_chars: int = field(default=20)
    max_review_chars: int = field(default=3000)
    min_reviews_per_game: int = field(default=3)
    max_reviews_per_game_for_card: int = field(default=50)
    max_representative_reviews: int = field(default=3)
    max_keywords: int = field(default=12)
    top_k: int = field(default=10)
    candidate_pool_size: int = field(default=50)
    synthetic_scenario_count: int = field(default=5)
    use_seed_cards_in_query: bool = field(default=True)
    max_llm_scenarios: int = field(default=20)
    max_llm_candidates: int = field(default=20)
    available_games_markdown_limit: int = field(default=100)
    draft_scenario_count: int = field(default=12)
    draft_scenario_candidate_count: int = field(default=20)
    draft_scenario_ground_truth_count: int = field(default=3)
    min_reviews_for_draft_scenario_seed: int = field(default=5)
    min_reviews_for_real_experiment: int = field(default=1000)
    min_game_cards_for_real_experiment: int = field(default=50)
    min_manual_scenarios_for_real_experiment: int = field(default=10)
    min_user_reviews: int = field(default=5)
    min_user_positive_reviews: int = field(default=3)
    user_holdout_count: int = field(default=1)
    user_candidate_pool_size: int = field(default=50)
    max_users_for_experiment: int = field(default=100)
    max_llm_users: int = field(default=10)
    llm_select_meaningful_users: bool = field(default=True)
    use_pilot_splits: bool = field(default=False)
    max_pilot_splits: int = field(default=10)
    force_holdout_into_candidate_pool: bool = field(default=True)
    preprocess_chunksize: int = field(default=100_000)
    max_rows_for_debug: int | None = field(default=None)
    subset_max_raw_rows: int = field(default=2_000_000)
    balanced_subset_target_games: int = field(default=200)
    balanced_subset_min_reviews_per_game: int = field(default=100)
    balanced_subset_min_positive_reviews_per_game: int = field(default=30)
    balanced_subset_max_processed_rows: int = field(default=1_000_000)
    balanced_subset_chunksize: int = field(default=100_000)
    processed_write_mode: str = field(default="overwrite")
    llm_provider: str = field(default="openai")
    llm_mode: str = field(default="real")
    llm_response_language: str = field(default="ru")
    llm_model: str | None = field(default=None)
    openai_api_key: str | None = field(default=None)
    gigachat_auth_key: str | None = field(default=None)
    gigachat_scope: str = field(default="GIGACHAT_API_PERS")
    gigachat_model: str = field(default="GigaChat")
    gigachat_verify_ssl: bool = field(default=True)
    gigachat_ca_bundle: str | None = field(default=None)
    gigachat_oauth_url: str = field(default="https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
    gigachat_api_base_url: str = field(default="https://gigachat.devices.sberbank.ru/api/v1")
    allow_llm_skip: bool = field(default=True)
    allow_llm_fallback: bool = field(default=True)
    llm_save_response_preview: bool = field(default=True)
    allow_balanced_subset: bool = field(default=True)
    random_seed: int = field(default=42)


def load_settings() -> Settings:
    """Load settings from defaults and optional `.env` variables."""

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    raw_data_dir = project_root / "data" / "raw"
    scenarios_dir = project_root / "data" / "scenarios"
    processed_data_dir = project_root / "data" / "processed"
    results_dir = project_root / "data" / "results"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    prompts_dir = project_root / "prompts"

    reviews_csv_path = project_root / os.getenv(
        "STEAM_REVIEWS_CSV",
        "data/raw/steam_reviews.csv",
    )
    active_processed_reviews_value = os.getenv(
        "ACTIVE_PROCESSED_REVIEWS",
        "data/processed/reviews_clean.csv",
    ).strip()
    if not active_processed_reviews_value:
        active_processed_reviews_value = "data/processed/reviews_clean.csv"
    active_processed_reviews_path = (
        Path(active_processed_reviews_value)
        if Path(active_processed_reviews_value).is_absolute()
        else project_root / active_processed_reviews_value
    )

    scenarios_file_value = os.getenv("SCENARIOS_FILE", "").strip()
    scenarios_file = project_root / scenarios_file_value if scenarios_file_value else None

    return Settings(
        project_root=project_root,
        raw_data_dir=raw_data_dir,
        scenarios_dir=scenarios_dir,
        processed_data_dir=processed_data_dir,
        results_dir=results_dir,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        prompts_dir=prompts_dir,
        reviews_csv_path=reviews_csv_path,
        scenarios_file=scenarios_file,
        reviews_clean_path=processed_data_dir / "reviews_clean.csv",
        active_processed_reviews_path=active_processed_reviews_path,
        reviews_clean_debug_path=processed_data_dir / "reviews_clean_debug.csv",
        reviews_clean_subset_path=processed_data_dir / "reviews_clean_subset.csv",
        reviews_clean_balanced_subset_path=processed_data_dir / "reviews_clean_balanced_subset.csv",
        dataset_discovery_report_csv_path=results_dir / "dataset_discovery_report.csv",
        dataset_discovery_report_json_path=results_dir / "dataset_discovery_report.json",
        dataset_discovery_report_markdown_path=results_dir / "dataset_discovery_report.md",
        raw_dataset_inspection_json_path=results_dir / "raw_dataset_inspection.json",
        raw_dataset_inspection_markdown_path=results_dir / "raw_dataset_inspection.md",
        schema_validation_report_json_path=results_dir / "schema_validation_report.json",
        schema_validation_report_markdown_path=results_dir / "schema_validation_report.md",
        preprocessing_summary_path=results_dir / "preprocessing_summary.json",
        preprocessing_summary_markdown_path=results_dir / "preprocessing_summary.md",
        preprocessing_debug_summary_path=results_dir / "preprocessing_debug_summary.json",
        preprocessing_debug_summary_markdown_path=results_dir / "preprocessing_debug_summary.md",
        preprocessing_subset_summary_path=results_dir / "preprocessing_subset_summary.json",
        preprocessing_subset_summary_markdown_path=results_dir / "preprocessing_subset_summary.md",
        preprocessing_balanced_subset_summary_path=results_dir / "preprocessing_balanced_subset_summary.json",
        preprocessing_balanced_subset_summary_markdown_path=results_dir / "preprocessing_balanced_subset_summary.md",
        preprocessing_progress_json_path=results_dir / "preprocessing_progress.json",
        preprocessing_progress_markdown_path=results_dir / "preprocessing_progress.md",
        raw_processed_comparison_json_path=results_dir / "raw_processed_comparison.json",
        raw_processed_comparison_markdown_path=results_dir / "raw_processed_comparison.md",
        raw_processed_subset_comparison_json_path=results_dir / "raw_processed_subset_comparison.json",
        raw_processed_subset_comparison_markdown_path=results_dir / "raw_processed_subset_comparison.md",
        raw_processed_balanced_subset_comparison_json_path=results_dir / "raw_processed_balanced_subset_comparison.json",
        raw_processed_balanced_subset_comparison_markdown_path=results_dir / "raw_processed_balanced_subset_comparison.md",
        raw_game_sampling_stats_csv_path=results_dir / "raw_game_sampling_stats.csv",
        raw_game_sampling_stats_json_path=results_dir / "raw_game_sampling_stats.json",
        game_card_summary_path=results_dir / "game_card_summary.json",
        game_cards_path=processed_data_dir / "game_cards.jsonl",
        user_profiles_path=processed_data_dir / "user_profiles.jsonl",
        user_profile_summary_path=results_dir / "user_profile_summary.json",
        user_splits_path=processed_data_dir / "user_evaluation_splits.jsonl",
        pilot_splits_path=processed_data_dir / "user_evaluation_splits_pilot.jsonl",
        user_split_summary_path=results_dir / "user_evaluation_split_summary.json",
        user_split_diagnostics_json_path=results_dir / "user_split_diagnostics.json",
        user_split_diagnostics_markdown_path=reports_dir / "user_split_diagnostics.md",
        user_evaluation_split_pilot_summary_path=results_dir / "user_evaluation_split_pilot_summary.json",
        user_evaluation_split_pilot_summary_markdown_path=reports_dir / "user_evaluation_split_pilot_summary.md",
        user_baseline_results_path=results_dir / "user_baseline_recommendations.jsonl",
        user_per_profile_results_path=results_dir / "user_per_profile_results.csv",
        user_metrics_summary_path=results_dir / "user_metrics_summary.csv",
        user_llm_prompt_preview_markdown_path=results_dir / "user_llm_prompt_preview.md",
        user_llm_prompt_preview_json_path=results_dir / "user_llm_prompt_preview.json",
        user_llm_prompt_preview_tiny_markdown_path=results_dir / "user_llm_prompt_preview_tiny.md",
        user_llm_prompt_preview_tiny_json_path=results_dir / "user_llm_prompt_preview_tiny.json",
        user_llm_prompt_preview_10_gigachat_markdown_path=results_dir / "user_llm_prompt_preview_10_gigachat.md",
        user_llm_prompt_preview_10_gigachat_json_path=results_dir / "user_llm_prompt_preview_10_gigachat.json",
        user_llm_results_path=results_dir / "user_llm_recommendations.jsonl",
        user_llm_reranking_summary_path=results_dir / "user_llm_reranking_summary.json",
        user_llm_validation_summary_path=results_dir / "user_llm_validation_summary.json",
        user_llm_per_profile_results_path=results_dir / "user_llm_per_profile_results.csv",
        user_llm_metrics_summary_path=results_dir / "user_llm_metrics_summary.csv",
        user_llm_metrics_summary_all_pilot_path=results_dir / "user_llm_metrics_summary_all_pilot.csv",
        user_rank_comparison_path=results_dir / "user_rank_comparison.csv",
        user_rank_comparison_markdown_path=reports_dir / "user_rank_comparison.md",
        user_llm_explanation_checks_csv_path=results_dir / "user_llm_explanation_checks.csv",
        user_llm_explanation_checks_markdown_path=reports_dir / "user_llm_explanation_checks.md",
        user_llm_explanation_examples_markdown_path=reports_dir / "user_llm_explanation_examples.md",
        user_llm_schema_error_report_json_path=results_dir / "user_llm_schema_error_report.json",
        user_llm_schema_error_report_markdown_path=reports_dir / "user_llm_schema_error_report.md",
        llm_pilot_candidate_user_report_csv_path=results_dir / "llm_pilot_candidate_user_report.csv",
        llm_pilot_candidate_user_report_markdown_path=reports_dir / "llm_pilot_candidate_user_report.md",
        user_llm_metrics_table_path=reports_dir / "user_llm_metrics_table.md",
        user_llm_pilot_summary_path=reports_dir / "user_llm_pilot_summary.md",
        user_llm_mock_validation_summary_path=results_dir / "user_llm_mock_validation_summary.json",
        user_llm_mock_validation_report_path=reports_dir / "user_llm_mock_validation_report.md",
        user_llm_failure_report_json_path=results_dir / "user_llm_failure_report.json",
        user_llm_failure_report_markdown_path=reports_dir / "user_llm_failure_report.md",
        llm_check_json_path=results_dir / "llm_check.json",
        llm_check_markdown_path=results_dir / "llm_check.md",
        llm_pilot_readiness_json_path=results_dir / "llm_pilot_readiness.json",
        llm_pilot_readiness_markdown_path=results_dir / "llm_pilot_readiness.md",
        scenarios_output_path=processed_data_dir / "recommendation_scenarios.jsonl",
        scenario_validation_report_path=results_dir / "scenario_validation_report.csv",
        environment_check_json_path=results_dir / "environment_check.json",
        environment_check_markdown_path=results_dir / "environment_check.md",
        preflight_report_json_path=results_dir / "preflight_report.json",
        preflight_report_markdown_path=results_dir / "preflight_report.md",
        smoke_test_report_path=results_dir / "smoke_test_report.json",
        data_diagnostics_json_path=results_dir / "data_diagnostics.json",
        data_diagnostics_markdown_path=results_dir / "data_diagnostics.md",
        available_games_path=results_dir / "available_games.csv",
        available_games_markdown_path=results_dir / "available_games.md",
        experiment_readiness_path=results_dir / "experiment_readiness.json",
        case_studies_json_path=results_dir / "case_studies.json",
        rank_comparison_csv_path=results_dir / "rank_comparison.csv",
        llm_explanation_checks_csv_path=results_dir / "llm_explanation_checks.csv",
        manual_scenarios_template_jsonl_path=scenarios_dir / "manual_scenarios_template.jsonl",
        manual_scenarios_template_csv_path=scenarios_dir / "manual_scenarios_template.csv",
        draft_scenarios_jsonl_path=scenarios_dir / "draft_manual_scenarios.jsonl",
        draft_scenarios_csv_path=scenarios_dir / "draft_manual_scenarios.csv",
        baseline_results_path=results_dir / "baseline_recommendations.jsonl",
        llm_results_path=results_dir / "llm_recommendations.jsonl",
        per_scenario_results_path=results_dir / "per_scenario_results.csv",
        metrics_summary_path=results_dir / "metrics_summary.csv",
        metrics_plot_path=figures_dir / "metrics_comparison.png",
        review_language_distribution_plot_path=figures_dir / "language_distribution.png",
        review_recommendation_distribution_plot_path=figures_dir / "recommendation_distribution.png",
        reviews_per_user_distribution_plot_path=figures_dir / "reviews_per_user_distribution.png",
        reviews_per_game_distribution_plot_path=figures_dir / "reviews_per_game_distribution.png",
        playtime_forever_distribution_plot_path=figures_dir / "playtime_forever_distribution.png",
        top_games_by_review_count_plot_path=figures_dir / "top_games_by_review_count.png",
        experiment_summary_path=reports_dir / "experiment_summary.md",
        case_studies_markdown_path=reports_dir / "case_studies.md",
        rank_comparison_markdown_path=reports_dir / "rank_comparison.md",
        llm_explanation_checks_markdown_path=reports_dir / "llm_explanation_checks.md",
        recommendation_examples_markdown_path=reports_dir / "recommendation_examples.md",
        thesis_metrics_table_path=reports_dir / "thesis_metrics_table.md",
        thesis_dataset_table_path=reports_dir / "thesis_dataset_table.md",
        thesis_scenario_table_path=reports_dir / "thesis_scenario_table.md",
        system_prompt_path=prompts_dir / "system_prompt.txt",
        reranking_prompt_template_path=prompts_dir / "reranking_prompt_template.txt",
        language_filter=os.getenv("LANGUAGE_FILTER") or None,
        min_review_chars=int(os.getenv("MIN_REVIEW_CHARS", "20")),
        max_review_chars=int(os.getenv("MAX_REVIEW_CHARS", "3000")),
        min_reviews_per_game=int(os.getenv("MIN_REVIEWS_PER_GAME", "3")),
        max_reviews_per_game_for_card=int(
            os.getenv("MAX_REVIEWS_PER_GAME_FOR_CARD", "50")
        ),
        max_representative_reviews=int(os.getenv("MAX_REPRESENTATIVE_REVIEWS", "3")),
        max_keywords=int(os.getenv("MAX_KEYWORDS", "12")),
        top_k=int(os.getenv("TOP_K", "10")),
        candidate_pool_size=int(os.getenv("CANDIDATE_POOL_SIZE", "50")),
        synthetic_scenario_count=int(os.getenv("SYNTHETIC_SCENARIO_COUNT", "5")),
        use_seed_cards_in_query=_load_bool_env("USE_SEED_CARDS_IN_QUERY", True),
        max_llm_scenarios=int(os.getenv("MAX_LLM_SCENARIOS", "20")),
        max_llm_candidates=int(os.getenv("MAX_LLM_CANDIDATES", "20")),
        available_games_markdown_limit=int(
            os.getenv("AVAILABLE_GAMES_MARKDOWN_LIMIT", "100")
        ),
        draft_scenario_count=int(os.getenv("DRAFT_SCENARIO_COUNT", "12")),
        draft_scenario_candidate_count=int(
            os.getenv("DRAFT_SCENARIO_CANDIDATE_COUNT", "20")
        ),
        draft_scenario_ground_truth_count=int(
            os.getenv("DRAFT_SCENARIO_GROUND_TRUTH_COUNT", "3")
        ),
        min_reviews_for_draft_scenario_seed=int(
            os.getenv("MIN_REVIEWS_FOR_DRAFT_SCENARIO_SEED", "5")
        ),
        min_reviews_for_real_experiment=int(
            os.getenv("MIN_REVIEWS_FOR_REAL_EXPERIMENT", "1000")
        ),
        min_game_cards_for_real_experiment=int(
            os.getenv("MIN_GAME_CARDS_FOR_REAL_EXPERIMENT", "50")
        ),
        min_manual_scenarios_for_real_experiment=int(
            os.getenv("MIN_MANUAL_SCENARIOS_FOR_REAL_EXPERIMENT", "10")
        ),
        min_user_reviews=int(os.getenv("MIN_USER_REVIEWS", "5")),
        min_user_positive_reviews=int(os.getenv("MIN_USER_POSITIVE_REVIEWS", "3")),
        user_holdout_count=int(os.getenv("USER_HOLDOUT_COUNT", "1")),
        user_candidate_pool_size=int(os.getenv("USER_CANDIDATE_POOL_SIZE", "50")),
        max_users_for_experiment=int(os.getenv("MAX_USERS_FOR_EXPERIMENT", "100")),
        max_llm_users=int(os.getenv("MAX_LLM_USERS", "10")),
        llm_select_meaningful_users=_load_bool_env("LLM_SELECT_MEANINGFUL_USERS", True),
        use_pilot_splits=_load_bool_env("USE_PILOT_SPLITS", False),
        max_pilot_splits=int(os.getenv("MAX_PILOT_SPLITS", "10")),
        force_holdout_into_candidate_pool=_load_bool_env("FORCE_HOLDOUT_INTO_CANDIDATE_POOL", True),
        preprocess_chunksize=int(os.getenv("PREPROCESS_CHUNKSIZE", "100000")),
        max_rows_for_debug=_load_optional_int_env("MAX_ROWS_FOR_DEBUG"),
        subset_max_raw_rows=int(os.getenv("SUBSET_MAX_RAW_ROWS", "2000000")),
        balanced_subset_target_games=int(os.getenv("BALANCED_SUBSET_TARGET_GAMES", "200")),
        balanced_subset_min_reviews_per_game=int(os.getenv("BALANCED_SUBSET_MIN_REVIEWS_PER_GAME", "100")),
        balanced_subset_min_positive_reviews_per_game=int(os.getenv("BALANCED_SUBSET_MIN_POSITIVE_REVIEWS_PER_GAME", "30")),
        balanced_subset_max_processed_rows=int(os.getenv("BALANCED_SUBSET_MAX_PROCESSED_ROWS", "1000000")),
        balanced_subset_chunksize=int(os.getenv("BALANCED_SUBSET_CHUNKSIZE", "100000")),
        processed_write_mode=os.getenv("PROCESSED_WRITE_MODE", "overwrite").strip().lower() or "overwrite",
        llm_provider=(os.getenv("LLM_PROVIDER") or "openai").strip().lower() or "openai",
        llm_mode=(os.getenv("LLM_MODE") or "real").strip().lower() or "real",
        llm_response_language=normalize_llm_response_language(
            os.getenv("LLM_RESPONSE_LANGUAGE", "ru")
        ),
        llm_model=os.getenv("OPENAI_MODEL") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        gigachat_auth_key=os.getenv("GIGACHAT_AUTH_KEY") or None,
        gigachat_scope=(os.getenv("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS").strip() or "GIGACHAT_API_PERS",
        gigachat_model=(os.getenv("GIGACHAT_MODEL") or "GigaChat").strip() or "GigaChat",
        gigachat_verify_ssl=_load_bool_env("GIGACHAT_VERIFY_SSL", True),
        gigachat_ca_bundle=(os.getenv("GIGACHAT_CA_BUNDLE") or "").strip() or None,
        gigachat_oauth_url=(os.getenv("GIGACHAT_OAUTH_URL") or "https://ngw.devices.sberbank.ru:9443/api/v2/oauth").strip() or "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        gigachat_api_base_url=(os.getenv("GIGACHAT_API_BASE_URL") or "https://gigachat.devices.sberbank.ru/api/v1").strip() or "https://gigachat.devices.sberbank.ru/api/v1",
        allow_llm_skip=_load_bool_env("ALLOW_LLM_SKIP", True),
        allow_llm_fallback=_load_bool_env("ALLOW_LLM_FALLBACK", True),
        llm_save_response_preview=_load_bool_env("LLM_SAVE_RESPONSE_PREVIEW", True),
        allow_balanced_subset=_load_bool_env("ALLOW_BALANCED_SUBSET", True),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )


def normalize_llm_response_language(value: object) -> str:
    """Normalize the configured LLM response language."""

    normalized = str(value or "ru").strip().lower() or "ru"
    if normalized not in {"ru", "en"}:
        raise ValueError("LLM_RESPONSE_LANGUAGE must be either 'ru' or 'en'.")
    return normalized


def _load_bool_env(name: str, default: bool) -> bool:
    """Load a boolean environment variable with permissive parsing."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value.")


def _load_optional_int_env(name: str) -> int | None:
    """Load an optional integer environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    return int(raw_value)
