import numpy as np
import pandas as pd
import pytest

from biokg_audit.baselines import CountPrior, GlobalDegreePrior, RandomScorer, ReverseLookup
from biokg_audit.data import contains_keys, triple_keys
from biokg_audit.evaluate import baseline_gates, evaluate_validation, metric_gates, midrank_metrics
from biokg_audit.matched import build_matched_negatives, known_positive_keys, official_count_gap
from biokg_audit.stats import bootstrap_mean, paired_difference


def rows(data):
    return np.arange(len(data["valid"]["head"]))


def test_metric_matches_official_ogb_evaluator():
    assert metric_gates()["status"] == "passed"


def test_midrank_ties():
    m = midrank_metrics(np.array([1.0, 1.0]), np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert m["rank"].tolist() == [1.0, 2.0]


def test_baseline_invariants(data, stats):
    assert baseline_gates(data, stats)["status"] == "passed"


@pytest.mark.parametrize("mode", ["nearest", "bin"])
def test_matched_negatives_are_valid(data, stats, test_keys, mode):
    r = rows(data)
    sets = build_matched_negatives(data, stats, r, mode=mode, k=50, seed=3, test_keys=test_keys)
    known = known_positive_keys(data, stats, test_keys)
    v = data["valid"]
    for key, direction in (("head_neg", 0), ("tail_neg", 1)):
        neg = sets[key]
        assert neg.shape == (len(r), 50)
        assert all(len(np.unique(x)) == 50 for x in neg)
        types = v["head_type" if direction == 0 else "tail_type"][r]
        low = data["type_offsets"][types][:, None]
        assert ((neg >= low) & (neg < low + data["type_sizes"][types][:, None])).all()
        fixed = (v["tail"] if direction == 0 else v["head"])[r][:, None]
        rel = v["relation"][r][:, None]
        keys = triple_keys(neg, rel, fixed, stats["n_entities"], stats["n_relations"]) if direction == 0 \
            else triple_keys(fixed, rel, neg, stats["n_entities"], stats["n_relations"])
        assert not contains_keys(known, keys).any(), "a known positive (incl. test) leaked into negatives"
    assert "test" in sets["metadata"]["filter"]


def test_matched_is_deterministic(data, stats):
    a = build_matched_negatives(data, stats, rows(data), k=30, seed=9)
    b = build_matched_negatives(data, stats, rows(data), k=30, seed=9)
    c = build_matched_negatives(data, stats, rows(data), k=30, seed=10)
    assert np.array_equal(a["tail_neg"], b["tail_neg"])
    assert not np.array_equal(a["tail_neg"], c["tail_neg"])


def test_nearest_is_more_balanced_than_official(data, stats):
    # Share of negatives at least as popular as the positive: 0.5 = no popularity signal.
    sets = build_matched_negatives(data, stats, rows(data), k=50)
    official = official_count_gap(data, stats, rows(data))
    assert sets["diagnostics"]["frac_neg_count_ge_target"] > official["frac_neg_count_ge_target"]


def test_candidate_sets_reproduce_official(data, stats):
    r = rows(data)
    sets = {"head_neg": np.asarray(data["valid"]["head_neg"]), "tail_neg": np.asarray(data["valid"]["tail_neg"]),
            "metadata": {"protocol": "official_copy"}}
    for scorer in (CountPrior(stats), RandomScorer(1), ReverseLookup(stats), GlobalDegreePrior(stats)):
        a = evaluate_validation(scorer, data, stats, r, verbose=False)
        b = evaluate_validation(scorer, data, stats, r, verbose=False, candidate_sets=sets)
        assert a["mrr"] == b["mrr"] and a["candidate_sha256"] == b["candidate_sha256"]


def test_bootstrap_ci_covers_mean_and_clusters():
    rng = np.random.default_rng(0)
    v = rng.random(400)
    res = bootstrap_mean(v, np.repeat(np.arange(200), 2), n_boot=300)
    assert res["ci_low"] <= res["mean"] <= res["ci_high"] and res["n_clusters"] == 200


def test_paired_difference(tmp_path, data, stats):
    r = rows(data)
    out = {}
    for name, scorer in (("a", CountPrior(stats)), ("b", RandomScorer(1))):
        evaluate_validation(scorer, data, stats, r, tmp_path / name, verbose=False)
        out[name] = pd.read_csv(tmp_path / name / "per_query.csv")
    same = paired_difference(out["a"], out["a"], n_boot=100)
    assert same["mean"].iloc[0] == 0 and not same["excludes_zero"].iloc[0]
    diff = paired_difference(out["a"], out["b"], n_boot=200)
    assert diff["excludes_zero"].iloc[0]
    with pytest.raises(ValueError):
        paired_difference(out["a"], out["b"].iloc[:-1])
