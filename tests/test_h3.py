import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from biokg_audit import training
from biokg_audit.baselines import prepare_statistics
from biokg_audit.data import triple_keys
from biokg_audit.evaluate import evaluate_validation
from biokg_audit.matched import RankPoolSampler
from biokg_audit.stats import bootstrap_mean


@pytest.mark.parametrize("mode,seeds", [("pilot", (101,)), ("full-validation", (101, 202, 303))])
def test_h3_cli(tmp_path, monkeypatch, mode, seeds):
    def split(h, t):
        return {k: np.asarray(v) for k, v in {
            "head": h, "tail": t, "relation": [0] * len(h),
            "head_type": [0] * len(h), "tail_type": [1] * len(h),
        }.items()}

    valid = split([0, 1], [1002, 1003])
    valid["head_neg"] = np.tile(np.arange(2, 502), (2, 1))
    valid["tail_neg"] = np.tile(np.arange(1004, 1504), (2, 1))
    data = {
        "train": split([2, 3], [1004, 1005]), "valid": valid,
        "n_entities": 2004, "n_relations": 1, "type_offsets": np.array([0, 1002]),
        "type_sizes": np.array([1002, 1002]), "schema_head": np.array([0]),
        "schema_tail": np.array([1]), "mapping": {"types": ["a", "b"]},
        "manifest": {"dataset": "synthetic-h3"},
    }
    # Two held-out edges leave row 0's head at 999 eligible; row 1's
    # tail lands exactly at 1000 and its held-out edge stays excluded.
    test_keys = np.sort(triple_keys([4, 5, 1], 0, [1002, 1002, 1006], 2004, 1))
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_h3.py"
    spec = importlib.util.spec_from_file_location("run_h3", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "load_data", lambda *args: data)
    monkeypatch.setattr(cli, "load_test_positive_keys", lambda data: test_keys)
    monkeypatch.setattr(cli, "metric_gates", lambda: {"status": "synthetic_fixture"})
    argv = [f"--{mode}", "--workspace", str(tmp_path / "work")]
    if mode == "pilot":
        argv.extend(["--max-per-relation", "2"])
    exports = []
    last_exports = []
    simulated_exports = []
    stats = prepare_statistics(data)
    for seed in seeds:
        for name in ("complex", "head_blind"):
            selection = {"selection_max_per_relation": 200, "validation_every": 1000} if mode != "pilot" else {}
            stem = f"{name}_{seed}"
            model = training.make_model(name, data, dim=8, seed=seed, device="cpu")
            result = training.train_model(
                model, data, training.TrainConfig(seed=seed, steps=1, batch_size=2, negatives=2, **selection),
                tmp_path / stem, evaluate_fn=lambda model, step: evaluate_validation(
                    model, data, stats, np.arange(2), model_name=name, training_seed=seed,
                    save_queries=False, verbose=False), progress=False)
            export = tmp_path / f"{stem}.pt"
            training.export_scoring_checkpoint(result["last_checkpoint"], export, data=data)
            last_exports.append(export)
            if mode == "full-validation":
                export = tmp_path / f"{stem}_best.pt"
                training.export_scoring_checkpoint(
                    result["best_checkpoint"], export, data=data,
                    completed_checkpoint_path=result["last_checkpoint"])
                # Simulate longer history for validating scheduled records and tied winners.
                snapshot = torch.load(export, weights_only=True, map_location="cpu")
                info = snapshot["metadata"]
                info["kind"] = "best"
                info["step"] = 1000
                info["config"].update(steps=1001, selection_max_per_relation=200, validation_every=1000)
                info["validation_history"][0]["step"] = 1000
                first = info["validation_history"][0]
                info["completed_run"] = {
                    "completed_steps": 1001, "selected_step": 1000, "last_checkpoint_sha256": "a" * 64,
                    "validation_history": [first, first | {"step": 1001}],
                }
                selected = tmp_path / f"{stem}_selected.pt"
                torch.save(snapshot, selected)
                simulated_exports.append(selected)
            exports.append(export)
            argv.extend(["--checkpoint", str(export)])

    with pytest.raises(SystemExit) as help_exit:
        cli.main(["--help"])
    assert help_exit.value.code == 0
    with pytest.raises(SystemExit) as args_exit:
        cli.main(argv[:-2])
    assert args_exit.value.code == 2
    if mode == "pilot":
        snapshot = torch.load(exports[1], weights_only=True, map_location="cpu")
        snapshot["metadata"]["config"]["steps"] += 1
        incomplete = tmp_path / "incomplete.pt"
        torch.save(snapshot, incomplete)
        with pytest.raises(ValueError, match="completed configured horizon"):
            cli.check_exports([exports[0], incomplete], data, pilot=True)
    else:
        with pytest.raises(ValueError, match="Full validation requires"):
            cli.check_exports(last_exports, data, pilot=False)
        with pytest.raises(ValueError, match="Mismatched training seed labels"):
            cli.check_exports([last_exports[0], last_exports[3]], data, pilot=True)
        with pytest.raises(ValueError, match="completed configured horizon"):
            cli.check_exports([exports[0], exports[1]], data, pilot=True)
        cli.check_exports(simulated_exports, data, pilot=False)
        for label, corrupt in (
            ("missing", lambda info: info.pop("completed_run")),
            ("wrong_horizon", lambda info: info["completed_run"].update(completed_steps=1000)),
            ("bad_last_digest", lambda info: info["completed_run"].update(last_checkpoint_sha256="bad")),
            ("missing_eval", lambda info: info["completed_run"]["validation_history"].pop()),
            ("wrong_winner", lambda info: info["completed_run"]["validation_history"][1].update(
                mrr=(info["completed_run"]["validation_history"][0]["mrr"] + 1) / 2)),
            ("wrong_rows", lambda info: info["completed_run"]["validation_history"][1].update(
                validation_rows_sha256="0" * 64)),
            ("wrong_candidates", lambda info: info["completed_run"]["validation_history"][1].update(
                candidate_sha256="0" * 64)),
            ("wrong_seed", lambda info: info["completed_run"]["validation_history"][1].update(
                training_seed=404)),
        ):
            snapshot = torch.load(simulated_exports[-1], weights_only=True, map_location="cpu")
            corrupt(snapshot["metadata"])
            invalid = tmp_path / f"{label}.pt"
            torch.save(snapshot, invalid)
            with pytest.raises(ValueError, match="Full validation requires"):
                cli.check_exports([*simulated_exports[:-1], invalid], data, pilot=False)

    cli.main(argv)
    run, = (tmp_path / "work" / "runs").iterdir()
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    stage = "preliminary_validation_pilot" if mode == "pilot" else "preliminary_validation_h3_grid"
    assert summary["stage"] == stage
    assert summary["wall_seconds"] > 0
    assert summary["matched"]["n_queries"] == 3
    assert summary["matched"]["n_clusters"] == 2
    assert summary["matched"]["n_boot"] == 1000
    assert summary["candidate_seeds"] == [311, 312, 313]
    assert summary["training_seeds"] == list(seeds)
    coverage = pd.read_csv(run / "query_cohort.csv").set_index(["row_id", "direction"])
    assert coverage.loc[(0, "head"), "status"] == "reduced_pool"
    assert coverage.loc[(0, "tail"), "status"] == "full_pool"
    assert coverage.loc[(1, "tail"), "eligible_count"] == 1000
    candidates = [np.load(run / f"candidates_{seed}.npy", allow_pickle=False)
                  for seed in summary["candidate_seeds"]]
    assert all(len(set(candidate[1, 0])) == 500 for candidate in candidates)
    assert all(1003 not in candidate[1, 1] and 1006 not in candidate[1, 1] for candidate in candidates)
    assert not np.array_equal(candidates[0][1, 0], candidates[1][1, 0])
    per_query = pd.read_csv(run / "paired_per_query.csv")
    assert len(per_query) == 3 * 3 * len(seeds)
    np.testing.assert_allclose(per_query["delta_rr"],
                               per_query["rr_complex"] - per_query["rr_head_blind"])
    averaged = per_query.groupby(["row_id", "direction", "cluster_id"])["delta_rr"].mean()
    expected = bootstrap_mean(averaged.to_numpy(), averaged.index.get_level_values("cluster_id"),
                              n_boot=1000, seed=20261110)
    assert summary["matched"] == pytest.approx(expected)
    official_pairs = []
    for seed in seeds:
        score_dir = run / f"seed_{seed}"
        a, b = [pd.read_csv(score_dir / name / "official" / "per_query.csv")
                .set_index(["row_id", "direction", "cluster_id"])["rr"]
                for name in ("complex", "head_blind")]
        common = (a.index.get_level_values("row_id") != 0) | (a.index.get_level_values("direction") != "head")
        official_pairs.append((a - b)[common])
    official_averaged = pd.concat(official_pairs, axis=1).mean(axis=1)
    official_expected = bootstrap_mean(
        official_averaged.to_numpy(), official_averaged.index.get_level_values("cluster_id"),
        n_boot=1000, seed=20261110)
    assert summary["official_common_cohort"] == pytest.approx(official_expected)
    assert summary["official_full_selected_n_queries"] == 4

    sampler = RankPoolSampler(data, prepare_statistics(data), test_keys)
    _, independently_resampled = sampler.sample_one(1, 1, 0, 1, 1003)
    for arr, seed in zip(candidates, summary["candidate_seeds"]):
        np.testing.assert_array_equal(arr[1, 1], independently_resampled[seed][0])
