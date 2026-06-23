# Balanced Subset Methodology Note

The full Steam Reviews export is too large for a single-memory preprocessing pass in a lightweight bachelor-thesis prototype, so the project uses a chunked pipeline and a balanced subset for the controlled user-based baseline experiment.

A simple first-N subset was not sufficient because the raw CSV is ordered by app_id, which concentrates early rows on only a few games and produces an overly narrow recommendation benchmark.

The balanced subset was therefore constructed in two passes over the full raw dataset: the first pass collected per-game review statistics, and the second pass selected the most review-rich games that satisfied the minimum review thresholds.

The resulting subset contains 200 games and 791254 processed reviews, while preserving user_id so that user-based offline evaluation remains possible.

The selected subset is appropriate for a controlled thesis workflow, but the resulting metrics must be reported as balanced-subset results rather than full-dataset results.

The same chunked preprocessing pipeline can later be run on the full 21.7 million-row dataset if the thesis work is extended beyond the balanced subset.
