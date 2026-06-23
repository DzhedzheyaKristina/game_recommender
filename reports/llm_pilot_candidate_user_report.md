# LLM Pilot Candidate User Report

This report is for pilot-case selection only and does not use holdout labels as model input.

| masked_user_id | holdout_count | candidate_pool_size | holdout_in_candidate_pool | baseline_best_holdout_rank | baseline_hit_at_10 | eligible_for_meaningful_llm_pilot |
| --- | --- | --- | --- | --- | --- | --- |
| user_5811 | 1 | 50 | True | 2 | 1 | True |
| user_119717 | 1 | 50 | True | 3 | 1 | True |
| user_612947 | 1 | 50 | True | 4 | 1 | True |
| user_64018 | 1 | 50 | True | 4 | 1 | True |
| user_551 | 1 | 50 | True | 5 | 1 | True |
| user_432875 | 1 | 50 | True | 6 | 1 | True |
| user_244467 | 1 | 50 | True | 7 | 1 | True |
| user_544810 | 1 | 50 | True | 7 | 1 | True |
| user_238552 | 1 | 50 | True | 8 | 1 | True |
| user_294952 | 1 | 50 | True | 10 | 1 | True |

- Total users inspected: 10
- Eligible for meaningful pilot: 10
- A user is eligible when the holdout is inside the candidate pool and the baseline rank is not already 1.
