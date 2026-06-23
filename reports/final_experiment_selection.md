# Final Experiment Selection

- Status: selected_with_warnings
- Selected experiment: `experiments/2026-06-06_183742_steam_reviews_balanced_subset_llm_10_gigachat`
- Selected experiment name: `steam_reviews_balanced_subset_llm_10_gigachat`
- Selected timestamp: `2026-06-06T18:37:42`
- Selection reason: Newest archived pilot with useful LLM outputs, selected with warnings because archived candidate-pool diagnostics were inconsistent.
- Candidate count: 13
- Valid candidate count: 0

## Selection Criteria
| experiment_mode | llm_provider | llm_mode | active_processed_reviews | llm_response_language | llm_ran | completion_requests_attempted_min | invalid_game_id_count_max | all_game_ids_inside_candidate_pool | llm_metrics_row_required | fallback_only_output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| user_based | gigachat | real | /home/kristina/projects/diplom/data/processed/reviews_clean_balanced_subset.csv | ru | True | 1 | 0 | True | True | False |

## Candidate Summary
| experiment_dir_relative | manifest_timestamp | valid | llm_provider | llm_mode | llm_ran | provider_preflight_ok | completion_requests_attempted | token_requests_attempted | real_api_calls | invalid_game_id_count | all_game_ids_inside_candidate_pool | llm_evaluated_users | llm_valid_records | fallback_record_count | total_records | metrics_llm_profiles | metrics_baseline_profiles | has_explanation_examples | reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/2026-06-06_183742_steam_reviews_balanced_subset_llm_10_gigachat | 2026-06-06T18:37:42 | False | gigachat | real | True | True | 10 | 1 | 10 | 0 | False | 4 | 40 | 60 | 100 | 4 | 10 | True | ['candidate_pool_violation'] |
| experiments/2026-06-06_025824_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T02:58:24 | False | gigachat | real | True | True | 3 | 1 | 3 | 0 | False | 1 | 10 | 20 | 30 | 1 | 3 | True | ['candidate_pool_violation'] |
| experiments/2026-06-06_024000_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T02:40:00 | False | gigachat | real | False | False | 0 | 3 | 3 | 0 | False | 0 | 0 | 30 | 30 | 0 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:connection_error', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_llm_metrics_row', 'fallback_only_output'] |
| experiments/2026-06-06_023202_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T02:32:02 | False | gigachat | real | False | False | 0 | 3 | 3 | 0 | False | 0 | 0 | 30 | 30 | 0 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:connection_error', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_llm_metrics_row', 'fallback_only_output'] |
| experiments/2026-06-06_015443_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T01:54:43 | False | gigachat | real | True | True | 3 | 1 | 3 | 0 | False | 2 | 0 | 10 | 30 | 2 | 3 | True | ['candidate_pool_violation', 'no_holdout_in_candidate_pool_hits'] |
| experiments/2026-06-06_011953_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T01:19:53 | False | gigachat | real | False | False | 0 | 3 | 3 | 0 | False | 0 | 0 | 30 | 30 | 0 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:connection_error', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_llm_metrics_row', 'fallback_only_output', 'no_holdout_in_candidate_pool_hits'] |
| experiments/2026-06-06_011308_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T01:13:08 | False | gigachat | real | False | False | 0 | 3 | 3 | 0 | False | 0 | 0 | 30 | 30 | 0 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:connection_error', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_llm_metrics_row', 'fallback_only_output', 'no_holdout_in_candidate_pool_hits'] |
| experiments/2026-06-06_004814_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T00:48:14 | False | gigachat | real | False | False | 0 | 0 | 3 | 0 | False | 0 | 0 | 30 | 30 | 0 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:unknown', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_llm_metrics_row', 'fallback_only_output', 'no_holdout_in_candidate_pool_hits'] |
| experiments/2026-06-06_004102_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-06T00:41:02 | False | gigachat | real | False | False | 0 | 0 | 3 | 0 | False | 0 | 0 | 30 | 30 | 0 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:unknown', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_llm_metrics_row', 'fallback_only_output', 'no_holdout_in_candidate_pool_hits'] |
| experiments/2026-06-05_223444_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-05T22:34:44 | False | gigachat | real | False | False | 0 | 0 | 3 | 0 | False | 0 | 0 | 0 | 3 | 2 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:unknown', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_holdout_in_candidate_pool_hits', 'no_llm_topk_hits'] |
| experiments/2026-06-05_222852_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-05T22:28:52 | False | gigachat | real | False | False | 0 | 0 | 3 | 0 | False | 0 | 0 | 0 | 3 | 2 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:unknown', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_holdout_in_candidate_pool_hits', 'no_llm_topk_hits'] |
| experiments/2026-06-05_222702_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-05T22:27:02 | False | gigachat | real | False | False | 0 | 0 | 3 | 0 | False | 0 | 0 | 0 | 3 | 2 | 3 | True | ['llm_ran_false', 'no_completion_requests', 'provider_preflight:unknown', 'candidate_pool_violation', 'no_llm_evaluated_users', 'no_holdout_in_candidate_pool_hits', 'no_llm_topk_hits'] |
| experiments/2026-06-05_215444_steam_reviews_balanced_subset_llm_tiny_gigachat | 2026-06-05T21:54:44 | False | gigachat | real | True | False | 0 | 0 | 3 | 0 | False | 2 | 0 | 0 | 21 | 0 | 0 | True | ['no_completion_requests', 'provider_preflight:unknown', 'candidate_pool_violation', 'no_llm_metrics_row', 'no_baseline_metrics_row'] |

## Selected Candidate Details
| experiment_dir_relative | experiment_name | manifest_timestamp | llm_provider | llm_mode | llm_ran | provider_preflight_ok | completion_requests_attempted | token_requests_attempted | real_api_calls | invalid_game_id_count | all_game_ids_inside_candidate_pool | llm_evaluated_users | llm_valid_records | fallback_record_count | total_records | metrics_llm_profiles | metrics_baseline_profiles | has_explanation_examples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiments/2026-06-06_183742_steam_reviews_balanced_subset_llm_10_gigachat | steam_reviews_balanced_subset_llm_10_gigachat | 2026-06-06T18:37:42 | gigachat | real | True | True | 10 | 1 | 10 | 0 | False | 4 | 40 | 60 | 100 | 4 | 10 | True |

## Warnings
- Selected a controlled pilot run from the experiments archive.

