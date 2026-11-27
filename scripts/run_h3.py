"""Validation-only ComplEx versus HeadBlind at fixed completed horizons."""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from biokg_audit.baselines import classify_frequency, prepare_statistics, reverse_available
from biokg_audit.data import _no_links, _sha256, load_data, load_test_positive_keys
from biokg_audit.evaluate import evaluate_validation, metric_gates, select_validation_rows
from biokg_audit.matched import (
    CANDIDATE_SEEDS, POOL_SEED, POOL_SIZE, RANK_POOL_PROTOCOL, SAMPLE_SIZE, RankPoolSampler,
)
from biokg_audit.stats import KEY, METRICS, aligned_queries, bootstrap_mean
from biokg_audit.training import restore_scoring_checkpoint, selected_validation_record
from biokg_audit.workflow import new_run, save_json

MODELS = ("complex", "head_blind")
TRAINING_SEEDS = (101, 202, 303)
BOOTSTRAP_SEED = 20261110
BOOTSTRAP_REPLICATES = 1000
BALANCE_FIELDS = (
    "mean_abs_rank_distance", "mean_signed_log_count_difference", "mean_abs_log_count_difference",
    "fraction_equal_count", "fraction_lower_count", "fraction_higher_count",
)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--full-validation", action="store_true", help="all validation rows and 6 exports")
    scope.add_argument("--pilot", action="store_true", help="bounded validation sample and 2 exports")
    parser.add_argument("--max-per-relation", type=int, help="pilot row cap per relation (default: 20)")
    parser.add_argument("--row-seed", type=int, default=71, help="pilot validation sampling seed")
    parser.add_argument("--workspace", type=Path, default=ROOT / "work")
    parser.add_argument("--input-dir", type=Path, help="trusted local OGB data root")
    parser.add_argument("--download", action="store_true", help="allow official dataset download")
    parser.add_argument("--checkpoint", required=True, action="append", type=Path,
                        help="repeat for each scoring export (last for pilot, selected best for full validation)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.pilot and args.max_per_relation is not None:
        parser.error("--max-per-relation requires --pilot")
    if args.pilot and args.max_per_relation is None:
        args.max_per_relation = 20
    if args.pilot and args.max_per_relation < 1:
        parser.error("--max-per-relation must be positive")
    if len(args.checkpoint) != (2 if args.pilot else 6):
        parser.error("Supply both models for each training seed (2 pilot or 6 full validation exports)")
    return args


def selection_provenance(data):
    rows = select_validation_rows(data, max_per_relation=200, seed=71).astype("<i8")
    split = data["valid"]
    fingerprint = hashlib.sha256()
    # Match the evaluator's row identity and negative serialization.
    for direction, target, fixed in ((0, "head", "tail"), (1, "tail", "head")):
        records = np.empty((len(rows), 505), dtype="<i8")
        records[:, 0] = rows
        records[:, 1] = direction
        records[:, 2] = split["relation"][rows]
        records[:, 3] = split[fixed][rows]
        records[:, 4] = split[target][rows]
        records[:, 5:] = split[f"{target}_neg"][rows]
        fingerprint.update(records)
    return {
        "split": "valid", "protocol": "official", "n_triples": len(rows),
        "n_queries": 2 * len(rows), "negatives_per_direction": 500,
        "validation_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
        "candidate_sha256": fingerprint.hexdigest(),
    }


def check_selected_export(info, provenance):
    config = info["config"]
    steps = config["steps"]
    message = "Full validation requires "
    completion = info.get("completed_run")
    if (info["kind"] != "best" or type(steps) is not int or steps <= 0
            or not isinstance(completion, dict)
            or type(completion.get("completed_steps")) is not int
            or completion["completed_steps"] != steps
            or type(completion.get("selected_step")) is not int
            or completion["selected_step"] != info["step"]
            or not isinstance(completion.get("last_checkpoint_sha256"), str)
            or not re.fullmatch("[0-9a-f]{64}", completion["last_checkpoint_sha256"])):
        raise ValueError(message + "a selected best checkpoint with completed run metadata")
    interval = max(1000, steps // 5)
    if (config["selection_max_per_relation"] != 200 or config["selection_seed"] != 71
            or config["validation_every"] != interval):
        raise ValueError(message + "the pre-specified validation selection recipe")
    history = completion.get("validation_history")
    try:
        winner = selected_validation_record(history, steps, interval)
    except ValueError as error:
        raise ValueError(message + str(error)) from error
    expected = provenance | {"model": info["model_config"]["name"], "training_seed": config["seed"]}
    if any(row.get(key) != value for row in history for key, value in expected.items()):
        raise ValueError(message + "official selection-set provenance for every checkpoint")
    prefix = [row for row in history if row["step"] <= info["step"]]
    if winner["step"] != info["step"] or info.get("validation_history") != prefix:
        raise ValueError(message + "the earliest maximum-MRR checkpoint and its original history")


def check_exports(paths, data, pilot):
    exports, metadata, lrs = {}, {}, {}
    recipe = None
    provenance = None if pilot else selection_provenance(data)
    for path in paths:
        path = _no_links(path)
        _, info = restore_scoring_checkpoint(path, data)
        model, config = info["model_config"], info["config"]
        name, seed = model["name"], model["seed"]
        key = (name, seed)
        if name not in MODELS or key in exports or config["seed"] != seed:
            raise ValueError("Expected one ComplEx and one HeadBlind export for every training seed")
        if pilot:
            if info["kind"] != "last" or info["step"] <= 0 or info["step"] != config["steps"]:
                raise ValueError("H3 pilot requires the last checkpoint at a completed configured horizon")
        else:
            check_selected_export(info, provenance)
        budget = {k: config[k] for k in ("steps", "batch_size", "negatives",
                                         "selection_max_per_relation", "selection_seed", "validation_every")}
        budget.update(dim=model["dim"], precision=model["precision"], source_sha256=info["source_sha256"])
        if recipe is None:
            recipe = budget
        if budget != recipe:
            raise ValueError("H3 checkpoints must use a common trainer, budget, dimension and validation recipe")
        if config["learning_rate"] != lrs.setdefault(name, config["learning_rate"]):
            raise ValueError("H3 requires a fixed learning rate across seeds for each model")
        exports[key] = path
        metadata[f"{name}/{seed}"] = info | {"file": path.name, "file_sha256": _sha256(path)}
    seeds = sorted({seed for _, seed in exports})
    expected = seeds if pilot else TRAINING_SEEDS
    if (pilot and len(seeds) != 1) or set(exports) != {(name, seed) for name in MODELS for seed in expected}:
        raise ValueError("Mismatched training seed labels; full validation requires 101, 202 and 303")
    return exports, metadata, seeds


def prepare_cohort(run, data, stats, rows, test_keys):
    sampler = RankPoolSampler(data, stats, test_keys)
    arrays = {
        seed: np.lib.format.open_memmap(run / f"candidates_{seed}.npy", mode="w+", dtype="<u4",
                                       shape=(2, len(rows), SAMPLE_SIZE))
        for seed in CANDIDATE_SEEDS
    }
    mask = np.zeros((2, len(rows)), dtype=bool)
    split = data["valid"]
    head, rel, tail = (np.asarray(split[key][rows]) for key in ("head", "relation", "tail"))
    applicable, inverse = reverse_available(stats, rel, head, tail)
    reverse = np.where(~applicable, "not_applicable",
                       np.where(head == tail, "self_loop", np.where(inverse, "available", "absent")))
    cohort_parts = []
    with (run / "balance.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = ("row_id", "direction", "candidate_seed", "eligible_count", "pool_count", "status",
                  *BALANCE_FIELDS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for direction, (fixed, target, role) in enumerate(((tail, head, "head"), (head, tail, "tail"))):
            count_info = []
            for pos, row in enumerate(rows):
                info, samples = sampler.sample_one(int(row), direction, int(rel[pos]),
                                                   int(fixed[pos]), int(target[pos]))
                count_info.append(info)
                mask[direction, pos] = bool(samples)
                for seed in CANDIDATE_SEEDS:
                    if seed in samples:
                        candidates, balance = samples[seed]
                        arrays[seed][direction, pos] = candidates
                    else:
                        balance = {}
                    writer.writerow(dict(row_id=int(row), direction=role, candidate_seed=seed, **info, **balance))
            cohort_parts.append(pd.DataFrame(count_info).assign(
                row_id=rows, direction=role, relation=rel,
                frequency_bin=classify_frequency(stats, rel, direction, target), reverse_stratum=reverse))
    for array in arrays.values():
        array.flush()
    cohort = pd.concat(cohort_parts, ignore_index=True)
    cohort.to_csv(run / "query_cohort.csv", index=False)
    coverage = []
    for column in ("relation", "direction", "frequency_bin", "reverse_stratum"):
        table = (cohort.groupby([column, "status"]).size().unstack(fill_value=0)
                 .reindex(columns=["full_pool", "reduced_pool", "insufficient_eligible"], fill_value=0))
        table["total"] = table.sum(axis=1)
        table["excluded"] = table["total"] - table["full_pool"]
        table["excluded_percent"] = 100 * table["excluded"] / table["total"]
        coverage.append(table.reset_index(names="value").assign(stratum=column))
    pd.concat(coverage, ignore_index=True).to_csv(run / "coverage.csv", index=False)
    np.save(run / "validation_rows.npy", rows, allow_pickle=False)
    np.save(run / "matched_query_mask.npy", mask, allow_pickle=False)
    return mask, cohort


def pair_files(run, suffix, expected_index):
    columns = KEY + ["target_global", "cluster_id", *METRICS]
    a, b = [pd.read_csv(run / name / suffix / "per_query.csv", usecols=columns) for name in MODELS]
    a, b = aligned_queries(a, b)
    if not a.index.equals(expected_index):
        raise ValueError("Scoring did not preserve the expected validation query cohort")
    return a, b


def metric_row(a, b, label, train_seed, candidate_seed=None):
    values = {f"{metric}_{name}": float(scores[metric].mean())
              for name, scores in zip(MODELS, (a, b)) for metric in METRICS}
    return dict(protocol=label, training_seed=train_seed, candidate_seed=candidate_seed,
                n_queries=len(a), delta_rr=values["rr_complex"] - values["rr_head_blind"], **values)


def main(argv=None):
    args = parse(argv)
    start = time.perf_counter()
    data = load_data(args.workspace, args.input_dir, args.download)
    exports, metadata, train_seeds = check_exports(args.checkpoint, data, args.pilot)
    gates = metric_gates()
    stats = prepare_statistics(data)
    rows = select_validation_rows(data, args.max_per_relation, args.row_seed)
    stage = "preliminary_validation_pilot" if args.pilot else "preliminary_validation_h3_grid"
    run = new_run(_no_links(args.workspace), {
        "stage": stage, "n_validation_triples_selected": int(len(rows)),
        "n_validation_triples_total": int(len(data["valid"]["head"])),
        "max_per_relation": args.max_per_relation, "row_seed": args.row_seed, "device": args.device,
        "protocol": RANK_POOL_PROTOCOL, "pool_seed": POOL_SEED, "pool_size": POOL_SIZE,
        "negatives_per_query": SAMPLE_SIZE, "candidate_seeds": list(CANDIDATE_SEEDS),
        "training_seeds": train_seeds,
    }, "h3_validation")
    print(f"{stage} -> {run}", flush=True)
    mask, cohort = prepare_cohort(run, data, stats, rows, load_test_positive_keys(data))
    arrays = {seed: np.load(run / f"candidates_{seed}.npy", mmap_mode="r", allow_pickle=False)
              for seed in CANDIDATE_SEEDS}
    fingerprints = {}
    save_json(run / "provenance.json", {
        "metric_gates": gates, "checkpoints": metadata,
        "test_positive_mask": data.get("test_positive_mask_provenance"),
        "candidate_files": {str(seed): _sha256(run / f"candidates_{seed}.npy") for seed in CANDIDATE_SEEDS},
        "validation_rows_sha256": _sha256(run / "validation_rows.npy"),
        "matched_query_mask_sha256": _sha256(run / "matched_query_mask.npy"),
    })
    full_index = pd.MultiIndex.from_frame(cohort[KEY])
    matched_index = full_index[mask.reshape(-1)]
    matched_seeds = CANDIDATE_SEEDS if len(matched_index) else ()
    result_rows = []
    matched_deltas = np.zeros(len(matched_index))
    official_deltas = np.zeros(len(matched_index))
    clusters = None
    paired_path = run / "paired_per_query.csv"
    for train_seed in train_seeds:
        score_dir = run / f"seed_{train_seed}"
        for name in MODELS:
            path = exports[name, train_seed]
            if _sha256(path) != metadata[f"{name}/{train_seed}"]["file_sha256"]:
                raise ValueError("Scoring checkpoint changed during the run")
            model, _ = restore_scoring_checkpoint(path, data, device=args.device)
            for candidate_seed in (None, *matched_seeds):
                candidates = None if candidate_seed is None else {
                    "head_neg": arrays[candidate_seed][0], "tail_neg": arrays[candidate_seed][1],
                    "metadata": {"protocol": RANK_POOL_PROTOCOL},
                }
                suffix = "official" if candidate_seed is None else f"matched_{candidate_seed}"
                result = evaluate_validation(
                    model, data, stats, rows, score_dir / name / suffix, model_name=name,
                    training_seed=train_seed, candidate_sets=candidates,
                    query_mask=None if candidates is None else mask,
                )
                fingerprint = (result["validation_rows_sha256"], result["candidate_sha256"])
                if fingerprint != fingerprints.setdefault(suffix, fingerprint):
                    raise ValueError("Scorers did not see identical validation rows and candidate arrays")
            del model

        a, b = pair_files(score_dir, "official", full_index)
        result_rows.append(metric_row(a, b, "official_full_selected", train_seed))
        if not matched_seeds:
            continue
        a, b = a.loc[matched_index], b.loc[matched_index]
        result_rows.append(metric_row(a, b, "official_common_cohort", train_seed))
        official_deltas += (a["rr"] - b["rr"]).to_numpy()
        if clusters is None:
            clusters = a["cluster_id"]
        if not a["cluster_id"].equals(clusters):
            raise ValueError("Official clusters disagree across training seeds")
        for seed in matched_seeds:
            a, b = pair_files(score_dir, f"matched_{seed}", matched_index)
            if not a["cluster_id"].equals(clusters):
                raise ValueError("Matched and official clusters disagree")
            pair = pd.DataFrame({
                "cluster_id": clusters, "training_seed": train_seed, "candidate_seed": seed,
                "rr_complex": a["rr"], "rr_head_blind": b["rr"], "delta_rr": a["rr"] - b["rr"],
            })
            first = not paired_path.exists()
            pair.to_csv(paired_path, mode="x" if first else "a", header=first)
            matched_deltas += pair["delta_rr"].to_numpy()
            result_rows.append(metric_row(a, b, RANK_POOL_PROTOCOL, train_seed, seed))
    per_seed = pd.DataFrame(result_rows)
    per_seed.to_csv(run / "per_seed.csv", index=False)
    matched = per_seed[per_seed["protocol"] == RANK_POOL_PROTOCOL]
    train_means = matched.groupby("training_seed")["delta_rr"].mean()
    candidate_means = matched.groupby("candidate_seed")["delta_rr"].mean()
    matched_deltas /= len(train_seeds) * len(CANDIDATE_SEEDS)
    official_deltas /= len(train_seeds)
    if matched_seeds:
        pd.DataFrame({
            "cluster_id": clusters, "matched_delta_rr": matched_deltas, "official_delta_rr": official_deltas,
        }).to_csv(run / "averaged_per_query.csv")
    boot = dict(n_boot=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    save_json(run / "summary.json", {
        "stage": stage, "split": "valid", "protocol": RANK_POOL_PROTOCOL,
        "training_seeds": train_seeds, "candidate_seeds": list(CANDIDATE_SEEDS),
        "matched": bootstrap_mean(matched_deltas, clusters, **boot) if matched_seeds else None,
        "official_common_cohort": bootstrap_mean(official_deltas, clusters, **boot) if matched_seeds else None,
        "matched_status": "scored" if matched_seeds else "no_full_pool_queries",
        "official_full_selected_n_queries": len(full_index),
        "training_seed_delta_rr": {str(k): v for k, v in train_means.items()},
        "training_seed_sample_sd": float(train_means.std()) if len(train_means) > 1 else None,
        "candidate_seed_delta_rr": {str(int(k)): v for k, v in candidate_means.items()},
        "fingerprints": fingerprints, "bootstrap_seed": BOOTSTRAP_SEED,
        "wall_seconds": time.perf_counter() - start,
        "candidate_files_bytes": sum((run / f"candidates_{seed}.npy").stat().st_size
                                     for seed in CANDIDATE_SEEDS),
        "scope": "Validation only. Test positives are used solely for filtering matched negatives.",
        "uncertainty": "Conditional on the supplied exports and negative samples. No confirmatory p-value.",
    })


if __name__ == "__main__":
    main()
