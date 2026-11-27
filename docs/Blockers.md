# Outstanding work (ordered by impact on the main result)

1. **Learned models under matched negatives.** Weights, per-query CSVs and loss/time/VRAM logs from the Kaggle pilots were not exported, so HeadBlind/TransE/ComplEx cannot yet be re-scored on the matched candidates. The updated Kaggle notebook exports scoring weights. Recover an original snapshot with its manifest or run one export-enabled pilot, then check the single-seed comparison in the README. H3 requires both models at the same completed last step. Historical best-only weights can still be analysed using the baseline audit command. The synthetic CLI test covers this path.
2. **ComplEx has a pilot MRR close to the random baseline** (MRR 0.015 after 500 updates). Inspect loss curves, add regularisation and tune the learning rate, and rule out an implementation bug before any seed sweep.
3. **Under-trained learned pilots.** HeadBlind and TransE were still improving at update 500, and only seed 101 exists. Choose a common horizon, then run seeds 101/202/303.
4. **Full-validation run.** All matched/CI results are on the 20-triples-per-relation subset (zero stratum n=25). Run `python scripts/run_audit.py --preset full`.
5. **Popularity matchability.** 72% of subset queries fall below the operational threshold for the historical nearest-frequency sampler. The README reports both subsets. The threshold itself bounds the count-prior rank, so use the whole paired sample for the headline.
6. **Reverse stratum empty.** No validation query has its reverse triple in train, so the available-vs-absent effect is not estimable (N/A, not zero).
7. **Final test evaluation** is not run (test positives are used only as the negative filter mask).
8. **Strict resume** across sessions is implemented but unverified. The original Kaggle FileLink pointed at an internal path, not a downloadable URL.

Resolved since the first checkpoint draft: matched candidate generation and balance diagnostics; cluster-bootstrap CIs and paired differences; test-positive filtering of matched negatives; global-degree baseline; one-command pipeline; pinned CPU environment; pytest suite.
