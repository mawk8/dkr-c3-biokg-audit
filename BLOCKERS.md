# Outstanding work (ordered by impact on the main result)

1. **Learned models under matched negatives.** Weights, per-query CSVs and loss/time/VRAM logs from the Kaggle pilots were not exported, so HeadBlind/TransE/ComplEx cannot yet be re-scored on the matched candidates. Needed: re-run with checkpoint export, then call `evaluate_validation(..., candidate_sets=...)`.
2. **ComplEx is at chance level** (MRR 0.015 after 500 updates). Inspect loss curves, add regularisation and tune the learning rate, and rule out an implementation bug before any seed sweep.
3. **Under-trained learned pilots.** HeadBlind and TransE were still improving at update 500, and only seed 101 exists. Choose a common horizon, then run seeds 101/202/303.
4. **Full-validation run.** All matched/CI results are on the 20-triples-per-relation subset (zero stratum n=25). Run `python scripts/run_audit.py --preset full`.
5. **Popularity matchability.** 72% of subset queries cannot be balanced by popularity. Report matched results split by `popularity_matchable`, and decide whether to also give a matchable-only headline.
6. **Reverse stratum empty.** No validation query has its reverse triple in train, so the available-vs-absent effect is not estimable (N/A, not zero).
7. **Final test evaluation** is not run (test positives are used only as the negative filter mask).
8. **Strict resume** across sessions is implemented but unverified. The original Kaggle FileLink pointed at an internal path, not a downloadable URL.

Resolved since the first checkpoint draft: matched candidate generation and balance diagnostics; cluster-bootstrap CIs and paired differences; test-positive filtering of matched negatives; global-degree baseline; one-command pipeline; pinned CPU environment; pytest suite.
