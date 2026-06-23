# Final Thesis Experiment Summary

- Dataset mode: balanced_subset
- Processed reviews: 791254
- Games: 200
- Unique users: 1813
- Eligible users: 1813
- Pilot split count: 10
- LLM provider: gigachat
- Response language: ru
- Selected users: 0
- Token requests: 1
- Completion requests: 10
- Fallback count: 60
- Provider preflight OK: True
- LLM ran: True
- Real API calls: 10

## Metrics
- Baseline HitRate@5: 0.500
- Baseline HitRate@10: 1.000
- Baseline MRR: 0.221
- Baseline NDCG@10: 0.401
- LLM HitRate@5: 0.750
- LLM HitRate@10: 1.000
- LLM MRR: 0.375
- LLM NDCG@10: 0.516

## Limitations
This is a controlled pilot, not a full-scale statistically significant experiment.
The archived experiment was selected from the experiments folder, not from mutable data/results outputs.
Fallback-only or fallback-heavy runs are not exported as the final thesis experiment.

## Exported Artifacts
- `experiment_manifest.json`
- `user_llm_reranking_summary.json`
- `user_llm_metrics_summary.csv`
- `user_rank_comparison.csv`
- `user_llm_explanation_examples.md`
- `user_llm_pilot_summary.md`
- `thesis_balanced_subset_dataset_table.md`
- `user_thesis_metrics_table.md`
- `balanced_subset_methodology_note.md`

