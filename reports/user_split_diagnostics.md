# User Split Diagnostics

- Splits: 10
- Average candidate pool size: 50.0
- Splits with holdout_game_ids: 10
- Splits with holdout in candidate pool: 10
- Splits without holdout in candidate pool: 0
- Splits with holdout in baseline top-10: 10
- Splits with holdout in baseline top-50: 10

## Meaningfulness reasons
- user_history_too_small: 6

| masked_user_id | holdout_count | candidate_pool_size | holdout_in_candidate_pool | baseline_best_holdout_rank | baseline_hit_at_10 | holdout_in_baseline_top_10 | holdout_in_baseline_top_50 | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| user_5811 | 1 | 50 | True | 2 | 1 | True | True | meaningful |
| user_119717 | 1 | 50 | True | 3 | 1 | True | True | user_history_too_small |
| user_612947 | 1 | 50 | True | 4 | 1 | True | True | user_history_too_small |
| user_64018 | 1 | 50 | True | 4 | 1 | True | True | user_history_too_small |
| user_551 | 1 | 50 | True | 5 | 1 | True | True | meaningful |
| user_432875 | 1 | 50 | True | 6 | 1 | True | True | user_history_too_small |
| user_244467 | 1 | 50 | True | 7 | 1 | True | True | meaningful |
| user_544810 | 1 | 50 | True | 7 | 1 | True | True | user_history_too_small |
| user_238552 | 1 | 50 | True | 8 | 1 | True | True | meaningful |
| user_294952 | 1 | 50 | True | 10 | 1 | True | True | user_history_too_small |
