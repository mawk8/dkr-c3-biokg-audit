# Outstanding work

- Full learned-run exports are missing: weights, per-query CSV/hashes and loss/time/VRAM JSON have not been inspected locally. Original executed notebooks are available.
- The two-KGE-model by three-seed proposal matrix is incomplete. Only seed 101 and 500 updates per learned scorer are observed.
- ComplEx after 500 updates is near random on the selected validation subset. Inspect learning/loss diagnostics and choose the training horizon before multiplying seeds.
- Reverse-available validation stratum is empty; its interaction is N/A, not an estimated zero effect. Preserve this scientific limitation.
- Validation negatives were audited against train+valid only. Test/all-known audit and final protocol lock remain pending.
- Matched candidate generation/balance, bootstrap/CI, final-test evaluation and fresh-cloud reproduction are incomplete.
- Strict resume is implemented but cross-session equivalence has not been verified.
- The original displayed FileLink pointed at an internal Linux path, not a valid Kaggle download route. Preserve existing artifacts; do not rerun training for a download link.
