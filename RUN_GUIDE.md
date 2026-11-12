# Running the English notebook

1. Import notebooks/C3_Kaggle_Baselines_and_Pilot_EN.ipynb into Kaggle. The notebook includes the source package; no separate GitHub download is required.
2. The default mode is prepare_only. It does not download data or run experiments.
3. For a new baseline run set MODE="baselines", ALLOW_DOWNLOAD=True and ALLOW_OGB_INSTALL=True; enable Internet and run top to bottom. Baselines can use CPU. Do not do this merely to translate results that already exist.
4. For a deliberate new pilot enable a GPU, set MODE="pilot" and choose one of head_blind, complex, transe. Defaults are 128 real scalars, 500 updates, seed 101, batch 256, and 64 negatives. There is no automatic seed sweep.
5. Metric/data gates must pass. A stopped run is not automatically a completed pilot. Keep the printed completed_steps and stop_reason.
6. Results are written under /kaggle/working/dkr_c3_en. The new workspace avoids overwriting or colliding with original Russian-notebook source files. Existing Kaggle dkr_c3 artifacts must not be deleted. A new workspace may need new preparation; translation alone does not require another experiment.
7. Download existing outputs with Kaggle's native Output file controls. Do not treat /kaggle/working/... as a website URL. Save notebook outputs as evidence as well.

The notebook is not a one-command final scientific pipeline. Test, matched candidates and confidence intervals are outside its current implemented scope. Keep Kaggle's supplied Torch; only OGB 1.3.6 may be installed without dependency upgrades after explicit opt-in.
