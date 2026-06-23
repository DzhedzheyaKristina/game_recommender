# Experiment Summary

## Run Overview
- Reviews CSV: `/home/kristina/projects/diplom/data/raw/steam_reviews.csv`
- LLM status: no llm output

## Dataset and Scenario Summary
- Dataset size: 18 cleaned reviews
- Number of games: 6
- Number of game cards: 6
- Number of scenarios: 5
- Scenario type distribution: synthetic_demo: 5
- Scenario mode: synthetic_demo
- Manual/predefined scenarios: 0

## Warnings
- Dataset is below the configured thesis-scale thresholds and is too small for strong conclusions.
- Only synthetic demo scenarios were used. These results are suitable for technical validation only.

## Aggregate Metrics
| method | evaluated_scenarios | skipped_scenarios | mean_hit_rate_at_5 | mean_hit_rate_at_10 | mean_mrr | mean_ndcg_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 5 | 0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Scientific Validity Notes
- No user identifiers are available in the Steam Reviews dataset.
- Real user histories cannot be reconstructed from this dataset.
- Scenario-based preference profiles are therefore used instead of user-level offline evaluation.
- Synthetic demo scenarios are only for technical validation.
- Manual scenarios are recommended for the final thesis experiment.
- Metrics on the tiny bundled sample data should not be interpreted as scientific results.
- The LLM reranker, when enabled, only reorders baseline candidates.

## Report-Ready Thesis Artifacts
- `data/results/environment_check.json`: machine-readable environment and dependency check report.
- `data/results/environment_check.md`: markdown summary of local environment readiness.
- `data/results/preflight_report.md`: stricter pre-flight classification before a real thesis experiment.
- `data/results/smoke_test_report.json`: latest baseline-only smoke test result.
- `data/results/data_diagnostics.md`: dataset overview and descriptive statistics.
- `data/results/available_games.csv`: export of valid games for scenario authoring.
- `data/results/available_games.md`: markdown version of the available games list.
- `data/results/scenario_validation_report.csv`: scenario validation outcomes and warnings.
- `data/results/experiment_readiness.json`: readiness summary for thesis-scale evaluation.
- `reports/case_studies.md`: representative success and failure examples.
- `reports/rank_comparison.md`: scenario-level baseline versus LLM rank comparison.
- `reports/recommendation_examples.md`: recommendation examples for the experimental chapter.
- `reports/thesis_metrics_table.md`: markdown-ready metrics table.
- `reports/thesis_dataset_table.md`: markdown-ready dataset summary table.
- `reports/thesis_scenario_table.md`: markdown-ready scenario coverage table.
- `reports/llm_explanation_checks.md`: heuristic LLM explanation checks.

## Execution and Reproducibility
- Python version: 3.14.3
- LLM credentials configured: False
- Experiment scale classification: sample_or_technical_validation
- Recommended reproduction command: `./.venv/bin/python main.py`

## Recommended Next Steps
- Replace the bundled sample CSV with the real Steam Reviews dataset before drawing thesis conclusions.
- Create or review manual scenarios before using the metrics in the thesis.