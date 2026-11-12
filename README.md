# C3: Popularity, Leakage, or Structure?

Biomedical link prediction audit on ogbl-biokg. Midway snapshot, 12 November 2026.
This is a preliminary checkpoint, not the completed final experiment.

## Quickstart (main result in one command)

```bash
git clone <this repo> && cd dkr-c3-biokg-audit
python -m venv .venv && . .venv/bin/activate      # Python 3.14 verified
pip install -r requirements.txt
pytest -q tests                                   # 10 tests, ~10 s, synthetic graph
python scripts/run_audit.py --download            # ~1 GB one-time download, then ~1 min
```

This writes `results/quick/`:

| File | Contents |
|---|---|
| `main_table.csv` | MRR and Hits@1/3/10 with 95% CIs, per protocol (official / matched_nearest / matched_bin) and scorer |
| `ci_by_stratum.csv` | The same metrics split by frequency bin, popularity matchability and direction |
| `paired_differences.csv` | Paired MRR deltas (e.g. count prior matched vs official) with CIs |
| `mrr_by_frequency.png` | The main figure |
| `reverse_audit_train.csv` | Reverse-edge closure per relation in train |
| `run_info.json` | Seeds, hashes, negative-balance diagnostics, environment |

Expected headline: count prior MRR 0.478 (official) falls to 0.119 under
matched negatives. `--preset full` uses all 162,886 validation triples.
Test triples are never scored. Their head/relation/tail are used only to
filter matched negatives.

## Read first

- [Checkpoint report](CHECKPOINT_REPORT.md): task, protocol, results and limitations.
- [Results CSV](results/main_results.csv): all six scorers, with evidence labels.
- [Outstanding work](BLOCKERS.md).
- [Run instructions](RUN_GUIDE.md).
- [Provenance](provenance.json).

## Repository contents

`src/biokg_audit/` contains the current source (`matched.py`: frequency-matched negatives; `stats.py`: cluster-bootstrap CIs). `scripts/run_audit.py` is the one-command pipeline; `tests/` holds the pytest suite. `notebooks/` contains the standalone
English runnable notebook. It is an unexecuted translation; numerical settings
are unchanged. `configs/` contains pilot defaults. `results/` contains small tables.
`evidence/baseline_run/` contains the received original baseline CSV/JSON evidence.
`evidence/original_notebooks/` preserves the four original executed notebooks,
including their Russian console text. Their output is not translated or re-run,
so the historical evidence remains intact. All submission-facing explanation
is provided in English in the report and runnable notebook.

Raw dataset files, model weights and export ZIPs are excluded. `requirements.txt`
is a pinned, fresh-clone-verified CPU environment for the audit pipeline. The
earlier Kaggle GPU pilots used the versions in environment_observed.json; on
Kaggle keep the preinstalled CUDA Torch.

Metric validation uses the official OGB 1.3.6 CPU Torch branch; NumPy-branch
compatibility in the user's installed build failed and was not silently skipped.
See report for the reverse-stratum correction and remaining commitments.
