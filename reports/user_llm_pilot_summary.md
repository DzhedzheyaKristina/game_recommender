# User LLM Pilot Summary

This is a controlled pilot on a small subset of user-based evaluation splits.

- Status: completed_with_warnings
- Selected users: 10
- Candidates per user: 10
- Baseline evaluated users: 10
- LLM evaluated users: 4
- LLM improved users: 1
- LLM worsened users: 2
- LLM unchanged users: 7
- LLM mode: real
- LLM provider: gigachat
- Response language: ru
- Mock mode: False
- Real API calls: 10
- Real API calls total: 10
- Not for scientific metrics: False
- Holdout in candidate pool users: 10
- Holdout in baseline top-k users: 10
- Holdout in LLM top-k users: 10
- LLM valid records: 40
- Users with any valid LLM records: 4
- Users with only fallback: 6
- Users with partial fallback: 1
- Users with schema errors: 5
- Provider preflight OK: True
- Provider preflight status: `ok`
- Token requests attempted: 1
- Completion requests attempted: 10
- Provider failed users: 0

## Failure diagnostics
- Failed users: 0
- Failed records: 0
- Fallback users: 6
- Fallback records: 60
- Valid LLM records: 40
- Provider failed users: 0
- Invalid JSON count: 1
- Empty response count: 0
- Schema validation error count: 5
- Users with schema errors: 5
- Candidate pool error count: 1
- Response preview saved count: 10
- LLM responses received: 10
- LLM extraction failures: 7

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
This is a controlled pilot on a small subset of user-based evaluation splits. The results should not be presented as a full-experiment conclusion.

## Candidate Selection
- Selected meaningful pilot users: 10
- Failed LLM users: 0
- Partial fallback users: 1

