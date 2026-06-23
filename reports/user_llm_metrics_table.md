# User LLM Metrics Table

LLM reranking was only evaluated on a limited pilot subset.

| Method | LLM mode | HitRate@5 | HitRate@10 | MRR | NDCG@10 | Evaluated users |
| --- | --- | --- | --- | --- | --- | --- |
| user_tfidf_baseline | baseline | 0.500 | 1.000 | 0.221 | 0.401 | 10 |
| user_llm_reranker | real | 0.750 | 1.000 | 0.375 | 0.516 | 4 |
