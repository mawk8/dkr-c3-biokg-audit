"""CPU integration checks using synthetic typed triples only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from biokg_audit import training
from biokg_audit.matched import build_matched_negatives


@pytest.fixture
def typed_data(data, stats):
    return {
        **{key: value for key, value in data.items() if key != "test"},
        "schema_head": stats["schema_head"],
        "schema_tail": stats["schema_tail"],
        "mapping": {"types": ["a", "b"]},
        "manifest": {"dataset": "synthetic-checkpoint-test"},
    }


@pytest.fixture
def trained_checkpoint(tmp_path, typed_data):
    config = training.TrainConfig(seed=47, steps=1, batch_size=8, negatives=4)
    model = training.make_model("complex", typed_data, dim=8, seed=config.seed, device="cpu")
    run = training.train_model(model, typed_data, config, tmp_path / "training", progress=False)
    return model, Path(run["last_checkpoint"])


def test_scoring_export_restores_scores_and_checks_mapping(tmp_path, typed_data, trained_checkpoint):
    model, checkpoint = trained_checkpoint
    export = tmp_path / "scoring.pt"
    training.export_scoring_checkpoint(checkpoint, export, data=typed_data)
    snapshot = torch.load(export, map_location="cpu", weights_only=True)
    assert not {"optimizer_state", "positive_rng", "negative_rng", "run_dir"}.intersection(snapshot)

    restored, metadata = training.restore_scoring_checkpoint(export, typed_data)
    assert (metadata["step"], metadata["kind"], metadata["model_config"]["seed"]) == (1, "last", 47)
    for direction, fixed, candidates in ((0, [300, 302], [[0, 2], [301, 303]]),
                                         (1, [0, 305], [[300, 302], [301, 303]])):
        query = {"direction": direction, "relation": np.array([0, 1]), "fixed": np.array(fixed)}
        ids = np.asarray(candidates)
        np.testing.assert_array_equal(restored.score_candidates(query, ids), model.score_candidates(query, ids))

    wrong_mapping = {**typed_data, "mapping": {"types": ["b", "a"]}}
    with pytest.raises(ValueError, match="mapping|identity"):
        training.restore_scoring_checkpoint(export, wrong_mapping)


@pytest.fixture
def selected_run(tmp_path, typed_data):
    config = training.TrainConfig(seed=47, steps=4, batch_size=8, negatives=4, validation_every=1)
    model = training.make_model("complex", typed_data, dim=8, seed=config.seed, device="cpu")

    def evaluate(_, step):
        return {"mrr": (0.2, 0.5, 0.5, 0.1)[step - 1], "split": "valid", "protocol": "official",
                "validation_rows_sha256": "a" * 64, "candidate_sha256": "b" * 64}

    return training.train_model(model, typed_data, config, tmp_path / "selection",
                                evaluate_fn=evaluate, progress=False)


def test_export_attests_completed_run_and_earliest_best(tmp_path, typed_data, selected_run):
    best, last = (Path(selected_run[f"{kind}_checkpoint"]) for kind in ("best", "last"))
    export = tmp_path / "scoring.pt"
    training.export_scoring_checkpoint(best, export, data=typed_data, completed_checkpoint_path=last)
    _, info = training.restore_scoring_checkpoint(export, typed_data)
    assert (info["kind"], info["step"]) == ("best", 2)
    assert info["completed_run"] == {
        "completed_steps": 4, "selected_step": 2,
        "last_checkpoint_sha256": hashlib.sha256(last.read_bytes()).hexdigest(),
        "validation_history": selected_run["validation_history"],
    }


def test_migration_cli_accepts_relocated_checkpoints_and_original_metadata(tmp_path, typed_data, selected_run):
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_scoring_checkpoint.py"
    spec = importlib.util.spec_from_file_location("export_checkpoint", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    best, last = (tmp_path / f"{kind}.pt" for kind in ("best", "last"))
    for kind, target in (("best", best), ("last", last)):
        target.write_bytes(Path(selected_run[f"{kind}_checkpoint"]).read_bytes())
    (tmp_path / "entity_mapping.json").write_text(json.dumps(typed_data["mapping"]), encoding="utf-8")
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(typed_data["manifest"]), encoding="utf-8")

    export = tmp_path / "scoring.pt"
    cli.main([str(best), str(export), "--original-run", str(tmp_path), "--completed-checkpoint", str(last)])
    _, info = training.restore_scoring_checkpoint(export, typed_data)
    assert info["completed_run"]["selected_step"] == 2


@pytest.mark.parametrize("invalid,reason", [
    ("unfinished", "horizon"), ("different_run", "run_dir"), ("different_config", "config"),
    ("other_candidates", "provenance"), ("missing_validation", "schedule"),
    ("later_better", "maximum"), ("later_tie", "maximum"),
])
def test_export_rejects_invalid_completion_evidence(tmp_path, typed_data, selected_run, invalid, reason):
    best, last = (Path(selected_run[f"{kind}_checkpoint"]) for kind in ("best", "last"))
    payload = torch.load(last, map_location="cpu", weights_only=False)
    if invalid == "unfinished":
        payload["step"] = 3
    elif invalid == "different_run":
        payload["run_dir"] += "-other"
    elif invalid == "different_config":
        payload["config"]["negatives"] += 1
    elif invalid == "other_candidates":
        payload["validation_history"][-1]["candidate_sha256"] = "c" * 64
    elif invalid == "missing_validation":
        payload["validation_history"].pop(2)
    elif invalid == "later_better":
        payload["validation_history"][-1]["mrr"] = 0.8
    else:
        early_best = torch.load(best, map_location="cpu", weights_only=False)
        early_best["step"] = 3
        early_best["validation_history"] = payload["validation_history"][:3]
        best = tmp_path / "late_best.pt"
        early_best["best_checkpoint"] = payload["best_checkpoint"] = str(best)
        torch.save(early_best, best)
    altered_last = tmp_path / "altered_last.pt"
    torch.save(payload, altered_last)
    export = tmp_path / "invalid_export.pt"
    with pytest.raises(ValueError, match=reason):
        training.export_scoring_checkpoint(best, export, data=typed_data, completed_checkpoint_path=altered_last)
    assert not export.exists()


def test_learned_cli_uses_validation_and_baseline_candidates(tmp_path, monkeypatch, typed_data,
                                                               trained_checkpoint, test_keys):
    import pandas as pd

    export = tmp_path / "scoring.pt"
    _, checkpoint = trained_checkpoint
    training.export_scoring_checkpoint(checkpoint, export, data=typed_data)
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_audit.py"
    spec = importlib.util.spec_from_file_location("run_audit", path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    # Substitute data and reporting boundaries. Real matched sampling runs with a
    # smaller K because this fixture has fewer than 500 entities of each type.
    monkeypatch.setattr(audit, "load_data", lambda *args: typed_data)
    monkeypatch.setattr(audit, "load_test_positive_keys", lambda data: test_keys)
    monkeypatch.setattr(audit, "metric_gates", lambda: {"status": "synthetic_fixture"})
    monkeypatch.setattr(audit, "figure", lambda *args: None)
    monkeypatch.setattr(audit, "build_matched_negatives",
                        lambda *args, **kwargs: build_matched_negatives(*args, **kwargs, k=8))

    workspace = tmp_path / "work"
    out = tmp_path / "tables"
    audit.main(["--workspace", str(workspace), "--out", str(out), "--n-boot", "25",
                "--checkpoint", str(export)])
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    paired = pd.read_csv(out / "paired_differences.csv")
    run, = (workspace / "runs").iterdir()
    learned = "complex_seed47_step1_last"
    assert info["checkpoints"][learned]["step"] == 1
    for protocol in ("official", "matched_nearest", "matched_bin"):
        hashes = info["candidate_sha256"]
        assert hashes[f"{protocol}/{learned}"] == hashes[f"{protocol}/count_prior"]
        scores = pd.read_csv(run / protocol / learned / "per_query.csv")
        baseline = pd.read_csv(run / protocol / "count_prior" / "per_query.csv")
        assert set(scores["split"]) == {"valid"}
        assert set(scores["training_seed"]) == {47}
        keys = ["row_id", "direction", "target_global", "n_negative"]
        pd.testing.assert_frame_equal(scores[keys], baseline[keys])
        comparison = f"{learned} - count_prior ({protocol})"
        pooled = paired.loc[(paired["comparison"] == comparison) & (paired["breakdown"] == "all")]
        assert pooled["mean"].item() == pytest.approx((scores["rr"] - baseline["rr"]).mean())
