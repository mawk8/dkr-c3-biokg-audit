"""Cluster-bootstrap confidence intervals over per-query results.

Queries are not independent: the head and tail query of one triple share an
entity pair, and for same-type relations (h,r,t)/(t,r,h) share the unordered
pair. The evaluator writes this as ``cluster_id``; we resample clusters, not
queries, and use the ratio estimator sum(w * cluster_sum) / sum(w * cluster_n).

Paired differences (model A - model B, or protocol A - protocol B) are
computed per query on identical (row_id, direction) keys before resampling,
so the shared query difficulty cancels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("rr", "hits1", "hits3", "hits10")
KEY = ["row_id", "direction"]


def _cluster_arrays(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(codes))
    counts = np.bincount(inverse, minlength=len(codes)).astype(np.float64)
    return sums, counts


def bootstrap_mean(values, clusters, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05, chunk: int = 100) -> dict:
    """Mean with a percentile cluster-bootstrap CI."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.shape != clusters.shape or values.ndim != 1 or not len(values):
        raise ValueError("values and clusters must be matching non-empty [N] arrays")
    sums, counts = _cluster_arrays(values, clusters)
    n = len(sums)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for begin in range(0, n_boot, chunk):
        size = min(chunk, n_boot - begin)
        # Multinomial cluster weights == resampling n clusters with replacement.
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=size).astype(np.float64)
        estimates[begin:begin + size] = (weights @ sums) / (weights @ counts)
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_queries": int(len(values)), "n_clusters": int(n), "n_boot": int(n_boot)}


def summarize(frame: pd.DataFrame, by: str | None = None, metrics=METRICS, **kwargs) -> pd.DataFrame:
    """Per-group CIs for one model/protocol per-query frame."""
    groups = [("all", frame)] if by is None else list(frame.groupby(by, dropna=False))
    rows = []
    for key, part in groups:
        for metric in metrics:
            rows.append({"group": key, "metric": metric,
                         **bootstrap_mean(part[metric].to_numpy(float), part["cluster_id"].to_numpy(), **kwargs)})
    return pd.DataFrame(rows)


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, metric: str = "rr",
                      by: str | None = None, **kwargs) -> pd.DataFrame:
    """CI of mean(a - b) on identical queries; raises if the query sets differ."""
    left = a.set_index(KEY)
    right = b.set_index(KEY)
    if len(left) != len(right) or not left.index.sort_values().equals(right.index.sort_values()):
        raise ValueError("Paired comparison needs identical (row_id, direction) query sets")
    right = right.loc[left.index]
    if not (left["target_global"].to_numpy() == right["target_global"].to_numpy()).all():
        raise ValueError("Paired queries disagree on the positive target")
    diff = pd.DataFrame({"diff": left[metric].to_numpy(float) - right[metric].to_numpy(float),
                         "cluster_id": left["cluster_id"].to_numpy()})
    if by is not None:
        diff[by] = left[by].to_numpy()
    groups = [("all", diff)] if by is None else list(diff.groupby(by, dropna=False))
    rows = []
    for key, part in groups:
        result = bootstrap_mean(part["diff"].to_numpy(), part["cluster_id"].to_numpy(), **kwargs)
        # Tolerance: identical scorers give +/-1e-17 float noise, not an effect.
        result["excludes_zero"] = bool(result["ci_low"] > 1e-12 or result["ci_high"] < -1e-12)
        rows.append({"group": key, "metric": f"delta_{metric}", **result})
    return pd.DataFrame(rows)
