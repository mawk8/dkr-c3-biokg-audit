# C3: Popularity, Leakage, or Structure?

Biomedical link prediction audit on ogbl-biokg. Midway snapshot, 12 November 2026.
This is a preliminary checkpoint, not the completed final experiment.

## Read first

- [Checkpoint report](CHECKPOINT_REPORT.md): task, protocol, results and limitations.
- [Results CSV](results/main_results.csv): all six scorers, with evidence labels.
- [Outstanding work](BLOCKERS.md).
- [Run instructions](RUN_GUIDE.md).
- [Provenance](provenance.json).

## Repository contents

`src/biokg_audit/` contains the current source. `notebooks/` contains the standalone
English runnable notebook. It is an unexecuted translation; numerical settings
are unchanged. `configs/` contains pilot defaults. `results/` contains small tables.
`evidence/baseline_run/` contains the received original baseline CSV/JSON evidence.
`evidence/original_notebooks/` preserves the four original executed notebooks,
including their Russian console text. Their output is not translated or re-run,
so the historical evidence remains intact. All submission-facing explanation
is provided in English in the report and runnable notebook.

Raw dataset files, model weights and export ZIPs are excluded. Observed package
versions are recorded in environment_observed.json, not offered as a blindly
installable environment lock. Fresh-cloud reproduction remains unverified.

Metric validation uses the official OGB 1.3.6 CPU Torch branch; NumPy-branch
compatibility in the user's installed build failed and was not silently skipped.
See report for the reverse-stratum correction and remaining commitments.
