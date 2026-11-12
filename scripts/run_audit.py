"""One command to the main C3 result: baseline ladder x {official, matched}.

    python scripts/run_audit.py --download                 # quick: 20 valid triples/relation
    python scripts/run_audit.py --preset full              # all 162,886 valid triples

Steps: load/cached ogbl-biokg (train+valid; test positives only as a negative
filter mask) -> train-only statistics -> reverse-edge audit -> evaluate the
non-learned ladder (random, count prior, global degree, reverse lookup) under
official, matched_nearest and matched_bin negatives -> cluster-bootstrap CIs and
paired differences -> results/<preset>/ (small CSV/JSON/PNG, committed).
Per-query CSVs stay in <workspace>/runs/<id>/ (large, git-ignored).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from biokg_audit.baselines import (CountPrior, GlobalDegreePrior, RandomScorer,  # noqa: E402
                                   ReverseLookup, prepare_statistics)
from biokg_audit.data import load_data, load_test_positive_keys  # noqa: E402
from biokg_audit.evaluate import (MATCHABLE_THRESHOLD, evaluate_validation,  # noqa: E402
                                  metric_gates, select_validation_rows)
from biokg_audit.matched import build_matched_negatives  # noqa: E402
from biokg_audit.stats import paired_difference, summarize  # noqa: E402
from biokg_audit.workflow import environment_report, new_run, save_json  # noqa: E402

PRESETS = {"quick": 20, "full": None}
BINS = ["zero", "low", "mid", "high"]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="quick")
    p.add_argument("--workspace", type=Path, default=ROOT / "work")
    p.add_argument("--input-dir", type=Path, help="trusted local OGB biokg folder instead of download")
    p.add_argument("--download", action="store_true", help="allow the one-time official download (~1 GB)")
    p.add_argument("--row-seed", type=int, default=71)
    p.add_argument("--neg-seed", type=int, default=2024)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-test-filter", action="store_true", help="ablation: do not mask test positives")
    p.add_argument("--out", type=Path, help="default: results/<preset>")
    return p.parse_args(argv)


def figure(ci: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    part = ci[(ci.metric == "rr") & ci.group.isin(BINS)]
    protocols = [p for p in ("official", "matched_nearest") if p in set(part.protocol)]
    models = list(dict.fromkeys(part.model))
    fig, axes = plt.subplots(1, len(protocols), figsize=(5 * len(protocols), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(BINS))
    width = 0.8 / len(models)
    for ax, protocol in zip(axes, protocols):
        for i, model in enumerate(models):
            rows = part[(part.protocol == protocol) & (part.model == model)].set_index("group").reindex(BINS)
            err = np.vstack([rows["mean"] - rows.ci_low, rows.ci_high - rows["mean"]])
            ax.bar(x + (i - (len(models) - 1) / 2) * width, rows["mean"], width, yerr=err,
                   capsize=2, label=model)
        ax.set_xticks(x, [f"{b}\n(n={int(n) if n == n else 0})" for b, n in
                          zip(BINS, part[(part.protocol == protocol) & (part.model == models[0])]
                              .set_index("group").reindex(BINS).n_queries)])
        ax.set_title(protocol)
        ax.set_xlabel("train frequency of the true answer")
    axes[0].set_ylabel("MRR (95% cluster-bootstrap CI)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    a = parse(argv)
    out = a.out or ROOT / "results" / a.preset
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    data = load_data(a.workspace, a.input_dir, a.download)
    gates = metric_gates()
    stats = prepare_statistics(data)
    test_keys = None if a.no_test_filter else load_test_positive_keys(data)
    rows = select_validation_rows(data, PRESETS[a.preset], a.row_seed)
    config = {k: str(v) for k, v in vars(a).items()} | {"n_rows": int(len(rows))}
    run = new_run(a.workspace, config, f"audit_{a.preset}")
    print(f"[{a.preset}] {len(rows)} valid triples, {2 * len(rows)} queries -> {run}", flush=True)

    reverse = pd.DataFrame(stats["reverse_diagnostics"])
    reverse.to_csv(out / "reverse_audit_train.csv", index=False)

    protocols = {"official": None}
    diagnostics = {}
    for mode in tqdm(("nearest", "bin"), desc="matched protocols", unit="protocol"):
        t = time.perf_counter()
        sets = build_matched_negatives(data, stats, rows, mode=mode, seed=a.neg_seed, test_keys=test_keys,
                                       progress=True)
        name = sets["metadata"]["protocol"]
        protocols[name] = sets
        diagnostics[name] = sets["metadata"] | {
            k: sets["diagnostics"][k] for k in ("same_bin_fraction", "mean_abs_log_count_gap",
                                                "frac_neg_count_ge_target")
        } | {"seconds": time.perf_counter() - t}
        tqdm.write(f"built {name} in {diagnostics[name]['seconds']:.0f}s")

    scorers = {"random": RandomScorer(101), "count_prior": CountPrior(stats),
               "global_degree": GlobalDegreePrior(stats), "reverse_lookup": ReverseLookup(stats)}
    frames, summaries = {}, []
    jobs = [(p, s, n, sc) for p, s in protocols.items() for n, sc in scorers.items()]
    for protocol, sets, name, scorer in tqdm(jobs, desc="evaluating scorers", unit="run"):
        res = evaluate_validation(scorer, data, stats, rows, run / protocol / name, model_name=name,
                                  candidate_sets=sets, verbose=True)
        frames[protocol, name] = pd.read_csv(run / protocol / name / "per_query.csv")
        summaries.append(res)
        tqdm.write(f"{protocol:16s} {name:15s} MRR={res['mrr']:.4f} H@10={res['hits10']:.4f}")

    kw = {"n_boot": a.n_boot, "seed": 0}
    ci = []
    for (protocol, name), frame in tqdm(frames.items(), desc="bootstrap CIs", unit="run"):
        for by in (None, "frequency_bin", "popularity_matchable", "direction"):
            part = summarize(frame, by=by, **kw)
            part.insert(0, "breakdown", by or "all")
            part.insert(0, "model", name)
            part.insert(0, "protocol", protocol)
            ci.append(part)
    ci = pd.concat(ci, ignore_index=True)
    ci["group"] = ci["group"].astype(str)
    ci.to_csv(out / "ci_by_stratum.csv", index=False)

    main_table = ci[(ci.breakdown == "all")].pivot_table(
        index=["protocol", "model"], columns="metric", values=["mean", "ci_low", "ci_high"])
    main_table.columns = [f"{m}_{s}" for s, m in main_table.columns]
    main_table = main_table[[f"{m}_{s}" for m in ("rr", "hits1", "hits3", "hits10")
                             for s in ("mean", "ci_low", "ci_high")]].reset_index()
    main_table.insert(2, "n_queries", 2 * len(rows))
    main_table.to_csv(out / "main_table.csv", index=False)

    comparisons = [
        ("count_prior: matched_nearest - official", frames["matched_nearest", "count_prior"], frames["official", "count_prior"]),
        ("global_degree - count_prior (official)", frames["official", "global_degree"], frames["official", "count_prior"]),
        ("global_degree - count_prior (matched_nearest)", frames["matched_nearest", "global_degree"], frames["matched_nearest", "count_prior"]),
        ("reverse_lookup - count_prior (official)", frames["official", "reverse_lookup"], frames["official", "count_prior"]),
        ("count_prior - random (matched_nearest)", frames["matched_nearest", "count_prior"], frames["matched_nearest", "random"]),
    ]
    paired = []
    for label, left, right in tqdm(comparisons, desc="paired differences", unit="cmp"):
        for by in (None, "frequency_bin"):
            part = paired_difference(left, right, "rr", by=by, **kw)
            part.insert(0, "breakdown", by or "all")
            part.insert(0, "comparison", label)
            paired.append(part)
    pd.concat(paired, ignore_index=True).to_csv(out / "paired_differences.csv", index=False)

    figure(ci, out / "mrr_by_frequency.png")
    save_json_over(out / "run_info.json", {
        "preset": a.preset, "n_valid_triples": int(len(rows)), "n_queries": int(2 * len(rows)),
        "row_seed": a.row_seed, "neg_seed": a.neg_seed, "n_boot": a.n_boot,
        "matchable_threshold": MATCHABLE_THRESHOLD, "test_positive_mask": test_keys is not None,
        "negative_diagnostics": diagnostics, "metric_gates": gates,
        "candidate_sha256": {f"{s['protocol']}/{s['model']}": s["candidate_sha256"] for s in summaries},
        "validation_rows_sha256": summaries[0]["validation_rows_sha256"],
        "run_dir": str(run.relative_to(a.workspace.resolve().parent)) if run.is_relative_to(a.workspace.resolve().parent) else run.name,
        "environment": environment_report(), "seconds": time.perf_counter() - t0,
        "stage": "validation only; test triples are never scored",
    })
    print(main_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"done in {time.perf_counter() - t0:.0f}s -> {out}")


def save_json_over(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
