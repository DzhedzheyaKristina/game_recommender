# Steam Reviews Game Recommender Prototype

This repository contains a small Python 3.11+ research prototype for a bachelor thesis on game recommendation using Steam review text and optional LLM reranking.

The design is intentionally simple: it builds compact game cards from review text, defines scenario-based gamer preference profiles, retrieves candidate games with TF-IDF, and optionally reranks baseline results with an LLM.

## Real Steam Reviews Dataset

The full Steam Reviews dataset can contain these columns:

- `Index`
- `app_id`
- `app_name`
- `review_id`
- `language`
- `review`
- `timestamp_created`
- `timestamp_updated`
- `recommended`
- `votes_helpful`
- `votes_funny`
- `weighted_vote_score`
- `comment_count`
- `steam_purchase`
- `received_for_free`
- `written_during_early_access`
- `author.steamid`
- `author.num_games_owned`
- `author.num_reviews`
- `author.playtime_forever`
- `author.playtime_last_two_weeks`
- `author.playtime_at_review`
- `author.last_played`

The preprocessing step normalizes them into these internal columns:

- `game_id`
- `game_title`
- `review_id`
- `language`
- `review_text`
- `review_clean`
- `timestamp_created`
- `timestamp_updated`
- `recommended`
- `votes_helpful`
- `votes_funny`
- `weighted_vote_score`
- `comment_count`
- `steam_purchase`
- `received_for_free`
- `written_during_early_access`
- `user_id`
- `author_num_games_owned`
- `author_num_reviews`
- `playtime_forever`
- `playtime_last_two_weeks`
- `playtime_at_review`
- `author_last_played`

## Dataset Schema

The expected Steam Reviews CSV must be placed in `data/raw/` and include these columns:

- `app_id`
- `app_name`
- `review_id`
- `language`
- `review`
- `timestamp_created`
- `timestamp_updated`
- `recommended`
- `votes_helpful`

The dataset does **not** contain `user_id`, `author.steamid`, or real user histories. This project therefore uses **scenario-based recommendation profiles** instead of user-based recommendation.

## Scenario-Based Recommendation

Because there is no user identifier in the dataset, the prototype evaluates recommendations using predefined or synthetic scenarios.

Each scenario contains:

- `scenario_id`
- `scenario_type`
- `preference_text`
- `seed_game_ids`
- `excluded_game_ids`
- `ground_truth_game_ids`
- `candidate_game_ids`
- `notes`

If `SCENARIOS_FILE` is provided, predefined scenarios from JSONL or CSV are used and preferred for thesis evaluation.

If `SCENARIOS_FILE` is not provided, the project generates a few synthetic demo scenarios from the game cards. These synthetic scenarios are only for local technical validation and should not be treated as strong scientific evidence.

## Project Structure

```text
src/
  __init__.py
  config.py
  environment_tools.py
  data_loader.py
  preprocessing.py
  game_card_builder.py
  scenario_builder.py
  baseline_tfidf.py
  llm_reranker.py
  evaluation.py
  utils.py
scripts/
  run_full_pipeline.sh
  run_analysis.sh
  run_preflight.sh
  smoke_test.sh
configs/
  experiment_config.example.json
  experiment_config.debug.json
data/
  raw/
  scenarios/
  processed/
  results/
reports/
  figures/
prompts/
  system_prompt.txt
  reranking_prompt_template.txt
Makefile
.gitignore
main.py
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

4. Place your Steam reviews CSV at `data/raw/steam_reviews.csv`, or update `STEAM_REVIEWS_CSV` in `.env`.

The repository also includes a tiny sample CSV so the pipeline can be validated locally immediately.

## Correct Way to Run the Project

Dependencies must be installed and the virtual environment should be activated before running the project.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Alternative without activating the environment:

```bash
./.venv/bin/python main.py
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

This avoids accidentally using the host interpreter. If you run `python main.py` and a dependency such as `pandas` is missing, the CLI prints a friendly message telling you to activate `.venv` or run `./.venv/bin/python main.py`.

## Running the Pipeline

Run the full pipeline:

```bash
./.venv/bin/python main.py
```

Run a single step:

```bash
./.venv/bin/python main.py --step check_env
./.venv/bin/python main.py --step schema_check
./.venv/bin/python main.py --step preflight
./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.example.json
./.venv/bin/python main.py --step preprocess
./.venv/bin/python main.py --step preprocess_debug
./.venv/bin/python main.py --step preprocess_subset
./.venv/bin/python main.py --step preprocess_status
./.venv/bin/python main.py --step build_cards
./.venv/bin/python main.py --step build_scenarios
./.venv/bin/python main.py --step baseline
./.venv/bin/python main.py --step llm
./.venv/bin/python main.py --step evaluate
./.venv/bin/python main.py --step data_diagnostics
./.venv/bin/python main.py --step list_games
./.venv/bin/python main.py --step draft_scenarios
./.venv/bin/python main.py --step normalize_scenarios
./.venv/bin/python main.py --step build_user_profiles
./.venv/bin/python main.py --step build_user_splits
./.venv/bin/python main.py --step user_baseline
./.venv/bin/python main.py --step user_evaluate
./.venv/bin/python main.py --step user_experiment
./.venv/bin/python main.py --step case_studies
./.venv/bin/python main.py --step recommendation_examples
./.venv/bin/python main.py --step thesis_tables
./.venv/bin/python main.py --step analysis
./.venv/bin/python main.py --step validate_scenarios
./.venv/bin/python main.py --step readiness
./.venv/bin/python main.py --step smoke_test
```

## Optional LLM Reranking

The TF-IDF baseline works without LLM credentials.

LLM reranking is optional:

- if the selected provider credentials are missing, reranking is skipped cleanly
- skipped status records are still written to `data/results/llm_recommendations.jsonl`
- the LLM is only allowed to rerank provided baseline candidates

Baseline ranking does not need LLM credentials.

LLM reranking requires the selected provider credentials and a model.

If `SCENARIOS_FILE` is empty, synthetic demo scenarios are used.

Final thesis experiments should use manually reviewed scenarios.

## Output Files

Processed data:

- `data/processed/reviews_clean.csv`
- `data/processed/game_cards.jsonl`
- `data/processed/recommendation_scenarios.jsonl`

Results:

- `data/results/baseline_recommendations.jsonl`
- `data/results/llm_recommendations.jsonl`
- `data/results/preprocessing_summary.json`
- `data/results/game_card_summary.json`
- `data/results/environment_check.json`
- `data/results/environment_check.md`
- `data/results/schema_validation_report.json`
- `data/results/schema_validation_report.md`
- `data/results/preflight_report.json`
- `data/results/preflight_report.md`
- `data/results/smoke_test_report.json`
- `data/results/experiment_manifest.json`
- `data/results/data_diagnostics.json`
- `data/results/data_diagnostics.md`
- `data/results/available_games.csv`
- `data/results/available_games.md`
- `data/results/case_studies.json`
- `data/results/rank_comparison.csv`
- `data/results/llm_explanation_checks.csv`
- `data/results/scenario_validation_report.csv`
- `data/results/experiment_readiness.json`
- `data/results/per_scenario_results.csv`
- `data/results/metrics_summary.csv`
- `data/results/user_profile_summary.json`
- `data/results/user_evaluation_split_summary.json`
- `data/results/user_per_profile_results.csv`
- `data/results/user_metrics_summary.csv`

Reports:

- `reports/figures/metrics_comparison.png`
- `reports/experiment_summary.md`
- `reports/case_studies.md`
- `reports/rank_comparison.md`
- `reports/recommendation_examples.md`
- `reports/thesis_metrics_table.md`
- `reports/thesis_dataset_table.md`
- `reports/thesis_scenario_table.md`
- `reports/llm_explanation_checks.md`

## Recommended Workflow for the Thesis Experiment

1. Replace the bundled sample CSV with the real Steam Reviews CSV.
2. Check the local environment:

   ```bash
   ./.venv/bin/python main.py --step check_env
   ```

3. Run a strict pre-flight check:

   ```bash
   ./.venv/bin/python main.py --step preflight
   ```

4. Run a small smoke test:

   ```bash
   ./.venv/bin/python main.py --step smoke_test
   ```

5. Run:

   ```bash
   ./.venv/bin/python main.py --step all
   ```

6. Inspect generated game cards and supporting reports:

   ```bash
   ./.venv/bin/python main.py --step data_diagnostics
   ./.venv/bin/python main.py --step list_games
   ./.venv/bin/python main.py --step draft_scenarios
   ```

7. Create manual scenarios in:

   - `data/scenarios/manual_scenarios_template.jsonl`
   - optionally starting from `data/scenarios/draft_manual_scenarios.jsonl`

8. Set in `.env`:

   ```bash
   SCENARIOS_FILE=data/scenarios/manual_scenarios_template.jsonl
   ```

9. Normalize and validate scenarios:

   ```bash
   ./.venv/bin/python main.py --step normalize_scenarios
   ./.venv/bin/python main.py --step validate_scenarios
   ```

10. Run:

   ```bash
   ./.venv/bin/python main.py --step all
   ```

11. Check readiness:

   ```bash
   ./.venv/bin/python main.py --step readiness
   ```

12. Use output metrics and recommendation examples in the thesis.

## Manual Scenario Design Guidelines

- Choose a clear gamer preference profile for each scenario.
- Choose seed games only if they exist in the dataset.
- Exclude seed games from recommendations.
- Choose 1-3 expected relevant games as ground truth.
- Candidate games can be omitted to let the system rank all games.
- Use candidate games when you want a controlled comparison set.
- Avoid scenarios where ground truth games are not present in the dataset.
- Use `./.venv/bin/python main.py --step list_games` to export valid `app_id` values before editing manual scenarios.

## Dataset Diagnostics

Run:

```bash
./.venv/bin/python main.py --step data_diagnostics
```

This step reads `data/processed/reviews_clean.csv` and generates:

- `data/results/data_diagnostics.json`
- `data/results/data_diagnostics.md`
- `reports/figures/review_language_distribution.png`
- `reports/figures/review_recommendation_distribution.png`
- `reports/figures/top_games_by_review_count.png`

Use these outputs to inspect dataset size, language balance, review-length statistics, and whether enough games are available for a meaningful thesis experiment.

## Choosing Games for Manual Scenarios

Run:

```bash
./.venv/bin/python main.py --step list_games
```

This step generates:

- `data/results/available_games.csv`
- `data/results/available_games.md`

These files help you choose valid `app_id` values by showing review counts, sentiment balance, review-derived keywords, representative reviews, and a short `scenario_hint` for manual scenario design.

## Drafting Manual Scenarios

Run:

```bash
./.venv/bin/python main.py --step draft_scenarios
```

This step generates:

- `data/scenarios/draft_manual_scenarios.jsonl`
- `data/scenarios/draft_manual_scenarios.csv`

Draft scenarios are not final evaluation scenarios. They must be reviewed manually. They are generated from review-derived keywords and TF-IDF similarity between game cards, and should be converted into final manual scenarios before thesis evaluation.

## Normalizing Scenario Files

Run:

```bash
./.venv/bin/python main.py --step normalize_scenarios
```

This step reads `SCENARIOS_FILE` from `.env`, loads JSONL or CSV scenarios, normalizes them into canonical JSONL format, saves them to `data/processed/recommendation_scenarios.jsonl`, and runs validation after normalization.

## Environment Check

Run:

```bash
./.venv/bin/python main.py --step check_env
```

`check_env` creates:

- `data/results/environment_check.json`
- `data/results/environment_check.md`

It verifies Python 3.11+, required packages, project directories, `.env` files, the configured dataset path, and whether the selected LLM provider credentials are present.

## Checking the Real Dataset

Run:

```bash
./.venv/bin/python main.py --step schema_check
./.venv/bin/python main.py --step preprocess
./.venv/bin/python main.py --step data_diagnostics
./.venv/bin/python main.py --step preflight
```

`schema_check` locates the dataset, loads only the header, validates the expected Steam Reviews columns, writes schema reports, and prints a concise summary.

The schema validation outputs are:

- `data/results/schema_validation_report.json`
- `data/results/schema_validation_report.md`

## Processing the Full Real Dataset

Run:

```bash
./.venv/bin/python main.py --step preprocess
./.venv/bin/python main.py --step preprocess_status
```

The full real dataset contains about 21.7 million rows and may take a long time to preprocess. The project writes a temporary output file and updates `data/results/preprocessing_progress.json` and `data/results/preprocessing_progress.md` after each chunk.

If preprocessing is interrupted, the temporary file is left in place for inspection, but it is not treated as a valid cleaned dataset. Resume from a partial temp file is not implemented; restart preprocessing to produce a consistent final output.

## Creating a Research Subset

Run:

```bash
./.venv/bin/python main.py --step preprocess_subset
```

This creates a manageable subset for development or a limited thesis experiment:

- `data/processed/reviews_clean_subset.csv`
- `data/results/preprocessing_subset_summary.json`
- `data/results/preprocessing_subset_summary.md`
- `data/results/raw_processed_subset_comparison.json`
- `data/results/raw_processed_subset_comparison.md`

The subset preserves `user_id`, uses the same chunked preprocessing logic, and is useful when the full preprocessing run is too time-consuming for iterative work.

Results based on a subset should not be presented as results on the full dataset.

## Balanced Subset for User-Based Experiments

The first-N subset can be narrow when the raw CSV is ordered by `app_id`, because it may collect many rows for only a few games. For user-based offline evaluation, use the balanced subset mode instead.

Run:

```bash
./.venv/bin/python main.py --step preprocess_balanced_subset
```

Then point the active processed dataset at the balanced subset:

```bash
export ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv
./.venv/bin/python main.py --step data_diagnostics
./.venv/bin/python main.py --step build_cards
./.venv/bin/python main.py --step build_user_profiles
```

On Windows or in `.env`, set:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv
```

If `ACTIVE_PROCESSED_REVIEWS` is set, user-based steps use that file. If it is empty, the default full processed file is used. The final thesis text must state whether the experiment used the full dataset or the balanced subset.

The balanced subset is useful for development or a limited thesis experiment, but it must be described in the thesis as a subset-based setup. Do not present subset-based results as full-dataset results.

## Pre-flight Check Before Real Experiment

Run:

```bash
./.venv/bin/python main.py --step preflight
```

This step creates:

- `data/results/preflight_report.json`
- `data/results/preflight_report.md`

The preflight report classifies the current state as one of:

- `technical_validation_only`
- `ready_for_baseline_experiment`
- `ready_for_llm_experiment`
- `not_ready`

It inspects the environment check, dataset path, preprocessing and game-card summaries, scenario validation, experiment readiness, and LLM credential status.

If only the subset file exists, preflight recommends setting:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_subset.csv
```

This allows subset-based experiments, but the thesis should clearly state that a subset was used.

If the balanced subset exists and looks suitable, preflight also recommends:

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv
```

This is still a subset-based experiment and must be documented that way.

## Smoke Test

Run:

```bash
./.venv/bin/python main.py --step smoke_test
```

This step creates:

- `data/results/smoke_test_report.json`

The smoke test runs a lightweight baseline-only validation sequence on the current dataset: environment check, preprocessing, game-card generation, scenario building, baseline ranking, evaluation, and thesis-table generation. It does not call the LLM.

## User-Based Experiment

User-based mode is available when the dataset contains `author.steamid`, which is normalized to `user_id` during preprocessing.

Run:

```bash
./.venv/bin/python main.py --step build_user_profiles
./.venv/bin/python main.py --step build_user_splits
./.venv/bin/python main.py --step build_user_splits_pilot
./.venv/bin/python main.py --step user_baseline
./.venv/bin/python main.py --step user_evaluate
./.venv/bin/python main.py --step user_experiment
```

If `author.steamid` is missing, these steps do not crash. They print a clear message that user-based mode is unavailable and the scenario-based pipeline remains available as fallback. The same steps also respect `ACTIVE_PROCESSED_REVIEWS`, so you can point them at `data/processed/reviews_clean_balanced_subset.csv` without renaming files.
`build_user_splits_pilot` creates a separate pilot split file with holdout-in-candidate coverage for controlled LLM testing. The main 100-split evaluation file is preserved unless you explicitly overwrite it.

## Balanced Subset Baseline Experiment

This is the recommended controlled workflow while the full 21.7 million-row preprocessing run has not been completed.

Run:

```bash
export ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv
./.venv/bin/python main.py --step preflight
./.venv/bin/python main.py --step build_user_splits_pilot
./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_baseline.json
```

This is a valid thesis experiment only if you describe it honestly as a balanced-subset experiment. It is not a result on the full Steam Reviews dataset. Full-dataset results require completing `./.venv/bin/python main.py --step preprocess`.

## Running Controlled LLM Pilot

Use this workflow when you want to test LLM reranking on a small, controlled subset of the balanced-subset user evaluation splits.

```bash
export ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv
./.venv/bin/python main.py --step build_user_splits_pilot
./.venv/bin/python main.py --step user_llm_dry_run
./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_llm_pilot.json
```

- The dry run does not call the API.
- The pilot uses only a small number of users first.
- LLM metrics are only meaningful for the evaluated pilot users.
- This is the real LLM path. Keep `llm_mode=real` and configure the selected provider credentials and model if you want the API call to run.
- If credentials are missing and `allow_llm_skip=false`, the run stops clearly before any API call.
- Do not present pilot results as a full experiment result. It is a limited LLM pilot on the balanced subset.

## Russian-language LLM explanations

The controlled user-based prompts support Russian-language explanations while keeping JSON keys in English for program compatibility.

- `explanation`, `matched_preferences`, and `possible_risks` are written in Russian when `LLM_RESPONSE_LANGUAGE=ru`.
- Game titles are never translated.
- The prompt still instructs the model to return valid JSON only and to rank only the provided candidate games.

## GigaChat API Provider

The project can use GigaChat for Russian-language recommendation explanations.

- GigaChat uses `GIGACHAT_AUTH_KEY` to request a temporary access token.
- The access token is used for API calls and must be refreshed after expiration.
- The project handles token retrieval automatically.
- Do not commit `GIGACHAT_AUTH_KEY`.
- Explanations can be generated in Russian.
- If SSL certificate issues occur, install the required certificates. Do not disable verification except for local debugging.

Example commands:

```bash
./.venv/bin/python main.py --step llm_check
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_llm_dry_run --config configs/experiment_config.balanced_subset_llm_tiny_gigachat.json
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_llm_tiny_gigachat.json
```

## Tiny LLM Pilot

Use this only to validate the LLM integration path on 3 users.

```bash
./.venv/bin/python main.py --step llm_check
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_llm_dry_run --config configs/experiment_config.balanced_subset_llm_tiny.json
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_llm_tiny.json
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step llm_pilot_readiness
```

- The tiny pilot is for validating LLM integration, not for final thesis metrics.
- It uses only 3 users.
- It checks JSON parsing, candidate-pool safety, and explanation generation.
- If credentials are missing, the run stops before any API call and does not fake results.
The pilot should use the dedicated pilot splits created by `build_user_splits_pilot`, not the main 100-split evaluation file.

## Before running real LLM pilot

Run these checks before attempting a real API-backed pilot:

```bash
./.venv/bin/python main.py --step llm_check
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_llm_dry_run --config configs/experiment_config.balanced_subset_llm_tiny.json
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step llm_pilot_readiness
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step build_user_splits_pilot
```

- If credentials are missing, the real LLM pilot cannot run.
- Mock mode is only for pipeline validation.
- Start with 1 to 3 users and confirm the prompt preview before spending API calls.
- The report `data/results/llm_pilot_candidate_user_report.csv` helps select more informative pilot users without using holdout labels in the prompt.
- `data/results/user_split_diagnostics.json` and `reports/user_split_diagnostics.md` summarize whether holdout games are actually present in the candidate pools.
- The final thesis text must state whether the experiment used the balanced subset or the full dataset.

If you are using GigaChat, set the provider and credential variables in `.env`:

- `LLM_PROVIDER=gigachat`
- `GIGACHAT_AUTH_KEY`
- `GIGACHAT_SCOPE`
- `GIGACHAT_MODEL`
- `GIGACHAT_VERIFY_SSL`
- `GIGACHAT_CA_BUNDLE`
- `GIGACHAT_OAUTH_URL`
- `GIGACHAT_API_BASE_URL`

## GigaChat CA bundle

If `curl` works but Python `requests` still returns `ssl_error`, configure a custom CA bundle and keep SSL verification enabled:

```bash
GIGACHAT_VERIFY_SSL=true
GIGACHAT_CA_BUNDLE=/home/kristina/certs-gigachat/gigachat_ca_bundle.pem
```

Create the bundle by concatenating the trusted root and subordinate certificates, for example:

```bash
cat ~/certs-gigachat/russian_trusted_root_ca_pem.crt ~/certs-gigachat/russian_trusted_sub_ca_pem.crt > ~/certs-gigachat/gigachat_ca_bundle.pem
```

Then run:

```bash
./.venv/bin/python main.py --step llm_check
```

Expected result after successful token retrieval:

- `status = configured`
- `token_status = ok`

Do not disable SSL verification except for local debugging.

## Mock LLM Validation Mode

Use this mode only to validate the downstream pipeline when you do not have provider credentials yet.

```bash
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.balanced_subset_llm_mock.json
```

- Mock mode does not call any provider API.
- Mock mode is only for testing downstream files, evaluation, rank comparison, and report generation.
- Mock metrics must not be used as evidence of LLM effectiveness.
- Real LLM pilot runs still require credentials and `llm_mode=real`.

## Running the Final Thesis Experiment

Run:

```bash
./.venv/bin/python main.py --step run_experiment --config configs/experiment_config.example.json
```

Recommended flow:

- replace the bundled sample CSV with the real Steam Reviews dataset
- create manually reviewed scenarios
- set the dataset and scenario paths in the experiment config
- run `check_env` and `preflight`
- run the final controlled experiment
- inspect `experiments/<timestamp>_<experiment_name>/`

The controlled run creates a versioned experiment folder, copies the key outputs there, and writes `data/results/experiment_manifest.json`.

## Experiment Safety Gates

- Tiny datasets are blocked by default for final experiments.
- Synthetic scenarios are blocked by default for final experiments.
- LLM usage can be optional or required depending on the experiment config.
- `smoke_test` remains available for debugging and technical validation.
- If `run_llm=true` and credentials are missing, the runner only downgrades to baseline-only when `allow_llm_skip=true`.

For a repeatable pre-flight sequence, use:

```bash
./scripts/run_preflight.sh
./scripts/smoke_test.sh
```

## Preserving final thesis experiment

`run_experiment` writes to `data/results/` and can overwrite mutable outputs from the latest run. Stable experiment results are archived under `experiments/`.

Use the archive selector to choose the best successful run:

```bash
./.venv/bin/python main.py --step select_final_experiment --config configs/experiment_config.balanced_subset_llm_tiny_gigachat.json
```

Then export thesis-ready artifacts from the selected archived experiment:

```bash
./.venv/bin/python main.py --step export_thesis_results
```

- `select_final_experiment` inspects archived experiment folders and prefers a real user-based GigaChat pilot with usable LLM outputs.
- `export_thesis_results` copies stable artifacts from the selected experiment folder into `reports/final_thesis_artifacts/`.
- The exported bundle is intended for thesis writing and preserves the controlled pilot results without depending on the current mutable `data/results` state.

## Локальный демонстрационный интерфейс

Для демонстрации и защиты доступен локальный Streamlit-интерфейс. Он читает только финальные артефакты и не запускает preprocessing, baseline или LLM заново.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Интерфейс предназначен для demo-режима и показывает выбранный архивированный эксперимент из `reports/final_thesis_artifacts/`. Он не обращается к GigaChat/OpenAI автоматически.

## Подготовка 10-пользовательского LLM-пилота без API-запуска

Эта процедура готовит конфигурацию и dry-run-превью для 10-пользовательского пилота. Она не отправляет запросы в GigaChat и не запускает реальный `run_experiment`.

```bash
python -m py_compile main.py src/*.py
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step user_llm_dry_run --config configs/experiment_config.balanced_subset_llm_10_gigachat.json
ACTIVE_PROCESSED_REVIEWS=data/processed/reviews_clean_balanced_subset.csv ./.venv/bin/python main.py --step llm_pilot_readiness --config configs/experiment_config.balanced_subset_llm_10_gigachat.json
```

Перед реальным запуском LLM-пилота нужно отдельно проверить сеть, доступ к API и только затем вручную запускать `run_experiment`.

## Reproducible Run Scripts

The `scripts/` directory contains small wrappers that always use the project virtual environment:

- `./scripts/run_full_pipeline.sh`
- `./scripts/run_analysis.sh`
- `./scripts/run_preflight.sh`
- `./scripts/smoke_test.sh`

These are useful when moving from the bundled sample data to the real Steam Reviews dataset and wanting a predictable local execution path.

An optional `Makefile` is also included with these convenience targets:

- `make setup`
- `make run`
- `make smoke`
- `make analysis`
- `make preflight`
- `make clean-results`

## Experiment Analysis

Run:

```bash
./.venv/bin/python main.py --step case_studies
./.venv/bin/python main.py --step recommendation_examples
./.venv/bin/python main.py --step thesis_tables
./.venv/bin/python main.py --step analysis
```

These steps generate report-ready artifacts for the thesis:

- `./.venv/bin/python main.py --step case_studies`
  Creates `data/results/case_studies.json`, `data/results/rank_comparison.csv`, `data/results/llm_explanation_checks.csv`, `reports/case_studies.md`, `reports/rank_comparison.md`, and `reports/llm_explanation_checks.md`.
- `./.venv/bin/python main.py --step recommendation_examples`
  Creates `reports/recommendation_examples.md` with scenario-level baseline and optional LLM recommendation examples.
- `./.venv/bin/python main.py --step thesis_tables`
  Creates `reports/thesis_metrics_table.md`, `reports/thesis_dataset_table.md`, and `reports/thesis_scenario_table.md`.
- `./.venv/bin/python main.py --step analysis`
  Runs the local report helpers together: data diagnostics, available games export, scenario validation, readiness check, case studies, recommendation examples, and thesis tables.

Generated reports from the bundled sample dataset are for validation only. Final thesis conclusions require the real dataset and manually reviewed scenarios.

## Limitations

- The project is a thesis-oriented prototype, not a production recommender.
- Predefined scenarios are preferred for real thesis evaluation.
- Synthetic scenarios are useful for technical validation, not strong scientific evidence.
- Only review text and review-level metadata are used. No genres, tags, or external metadata are invented.
- The evaluation is scenario-based because the dataset has no user identities or session histories.
