"""Frequency-matched negative candidates (the required C3 control).

Official ogbl-biokg negatives are uniform over the target type. Because the
true answer is usually a popular entity, a pure popularity scorer ranks it
high against them. Here each query gets K negatives whose TRAIN frequency in
the same (relation, role) resembles the positive's, so popularity alone
carries no signal.

Modes
-----
``bin``      negatives drawn uniformly from the positive's zero/low/mid/high
             train-frequency bin (same frozen thresholds as the strata).
``nearest``  a window of K type-valid entities centred on the positive's
             train-count rank (K/2 less popular, K/2 more popular, borrowing
             at the extremes; random order among equal counts). Stricter:
             the count prior is near-chance except for the most popular
             positives, which cannot be matched from above.

Diagnostic ``frac_neg_count_ge_target``: 0.5 means popularity is
uninformative; near 0 means the positive out-ranks its negatives by count.

Negatives are filtered against all known positives: TRAIN + full VALID +
TEST positives (test head/relation/tail only, used solely as a mask; test
negatives are never read and test triples are never scored).
Sampling is deterministic given (seed, row_id, direction).
"""
from __future__ import annotations

import hashlib

import numpy as np

from .baselines import classify_frequency
from .data import contains_keys, triple_keys

MODES = ("bin", "nearest")


def known_positive_keys(data: dict, stats: dict, extra_keys: np.ndarray | None = None) -> np.ndarray:
    """Sorted unique keys of TRAIN + all VALID (+ optional TEST) positives."""
    valid = data["valid"]
    valid_keys = triple_keys(np.asarray(valid["head"], dtype=np.int64),
                             np.asarray(valid["relation"], dtype=np.int64),
                             np.asarray(valid["tail"], dtype=np.int64),
                             stats["n_entities"], stats["n_relations"])
    parts = [stats["train_keys"], valid_keys]
    if extra_keys is not None:
        parts.append(np.asarray(extra_keys, dtype=np.uint64))
    return np.unique(np.concatenate(parts))


def _rng(seed: int, row_id: int, direction: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{row_id}:{direction}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _bin_codes(stats: dict, relation: int, direction: int, counts: np.ndarray) -> np.ndarray:
    q1, q2 = stats["thresholds"][relation, direction]
    codes = np.full(counts.shape, 3, dtype=np.int8)
    codes[counts <= q2] = 2
    codes[counts <= q1] = 1
    codes[counts == 0] = 0
    return codes


class MatchedSampler:
    def __init__(self, data: dict, stats: dict, mode: str = "nearest", k: int = 500,
                 seed: int = 2024, test_keys: np.ndarray | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if k < 1:
            raise ValueError("k must be positive")
        self.stats, self.mode, self.k, self.seed = stats, mode, int(k), int(seed)
        self.known = known_positive_keys(data, stats, test_keys)
        self.filter_label = ("train+valid+test positives removed" if test_keys is not None
                             else "train+valid positives removed; TEST NOT FILTERED")
        self._cells = {}

    def _cell(self, relation: int, direction: int):
        key = (relation, direction)
        if key not in self._cells:
            stats = self.stats
            type_id = int(stats["schema_head" if direction == 0 else "schema_tail"][relation])
            low = int(stats["type_offsets"][type_id])
            ids = np.arange(low, low + int(stats["type_sizes"][type_id]), dtype=np.int64)
            counts = np.asarray(stats["counts"][relation, direction, ids])
            cell = {"ids": ids, "counts": counts,
                    "bins": _bin_codes(stats, relation, direction, counts)}
            if self.mode == "nearest":
                # Fixed random tie order inside equal counts, then sort by count.
                tie = np.random.default_rng([self.seed, relation, direction]).permutation(len(ids))
                order = np.lexsort((tie, counts))
                cell["order"] = order
                cell["sorted_counts"] = counts[order]
            self._cells[key] = cell
        return self._cells[key]

    def _is_known(self, relation: int, direction: int, fixed: int, cand: np.ndarray) -> np.ndarray:
        n, r = self.stats["n_entities"], self.stats["n_relations"]
        keys = (triple_keys(cand, relation, fixed, n, r) if direction == 0
                else triple_keys(fixed, relation, cand, n, r))
        return contains_keys(self.known, keys)

    def sample_one(self, row_id: int, direction: int, relation: int, fixed: int,
                   target: int) -> tuple[np.ndarray, dict]:
        cell = self._cell(relation, direction)
        offset = int(cell["ids"][0])
        target_count = int(self.stats["counts"][relation, direction, target])
        rng = _rng(self.seed, row_id, direction)
        if self.mode == "bin":
            target_bin = int(_bin_codes(self.stats, relation, direction,
                                        np.array([target_count]))[0])
            # Fill from the positive's bin first, then nearest bins outward.
            order = sorted(range(4), key=lambda b: (abs(b - target_bin), b))
            picked, matched = [], 0
            for b in order:
                pool = cell["ids"][cell["bins"] == b]
                pool = pool[(pool != target) & ~self._is_known(relation, direction, fixed, pool)]
                need = self.k - sum(len(x) for x in picked)
                take = pool if len(pool) <= need else rng.choice(pool, size=need, replace=False)
                picked.append(take)
                if b == target_bin:
                    matched = len(take)
                if sum(len(x) for x in picked) >= self.k:
                    break
            cand = np.concatenate(picked)
        else:
            # Rank-centred window: k/2 entities just below the positive's
            # count rank and k/2 just above (borrowing from the other side at
            # the extremes). Distance-nearest would favour the dense low tail.
            order = cell["order"]
            ranked = cell["ids"][order]
            valid = (ranked != target) & ~self._is_known(relation, direction, fixed, ranked)
            # Place the positive in the middle of its tie group (random rank among equals).
            sc = cell["sorted_counts"]
            lo_t, hi_t = np.searchsorted(sc, target_count, "left"), np.searchsorted(sc, target_count, "right")
            center = int(rng.integers(lo_t, hi_t + 1))
            below = ranked[:center][valid[:center]][::-1]
            above = ranked[center:][valid[center:]]
            n_below = min(len(below), max(self.k // 2, self.k - len(above)))
            n_above = min(len(above), self.k - n_below)
            cand = np.concatenate([below[:n_below], above[:n_above]])
            matched = int(np.count_nonzero(
                _bin_codes(self.stats, relation, direction, cell["counts"][cand - offset])
                == _bin_codes(self.stats, relation, direction, np.array([target_count]))[0]))
        if len(cand) < self.k:
            raise ValueError(f"Relation {relation}/{direction}: only {len(cand)} valid negatives")
        cand = cand[rng.permutation(len(cand))]  # no positional signal
        neg_counts = cell["counts"][cand - offset]
        # 0.5 = popularity carries no signal; ~0 = positive is the most popular.
        info = {"n_same_bin": matched,
                "frac_neg_count_ge": float(np.mean(neg_counts >= target_count)),
                "mean_abs_log_count_gap": float(np.mean(np.abs(
                    np.log1p(cell["counts"][cand - offset]) - np.log1p(target_count))))}
        return cand.astype(np.int64), info


def build_matched_negatives(data: dict, stats: dict, rows: np.ndarray, mode: str = "nearest",
                            k: int = 500, seed: int = 2024,
                            test_keys: np.ndarray | None = None) -> dict:
    """Return {'head_neg','tail_neg': int64[len(rows),k], diagnostics, metadata}.

    Arrays are aligned with ``rows`` (row position, not row ID).
    """
    sampler = MatchedSampler(data, stats, mode=mode, k=k, seed=seed, test_keys=test_keys)
    split = data["valid"]
    rows = np.asarray(rows, dtype=np.int64)
    out = {"head_neg": np.empty((len(rows), k), dtype=np.int64),
           "tail_neg": np.empty((len(rows), k), dtype=np.int64)}
    same_bin = np.empty((2, len(rows)), dtype=np.int64)
    gap = np.empty((2, len(rows)), dtype=np.float64)
    balance = np.empty((2, len(rows)), dtype=np.float64)
    for i, row in enumerate(rows):
        h, r, t = int(split["head"][row]), int(split["relation"][row]), int(split["tail"][row])
        for direction, fixed, target, key in ((0, t, h, "head_neg"), (1, h, t, "tail_neg")):
            cand, info = sampler.sample_one(int(row), direction, r, fixed, target)
            out[key][i] = cand
            same_bin[direction, i] = info["n_same_bin"]
            gap[direction, i] = info["mean_abs_log_count_gap"]
            balance[direction, i] = info["frac_neg_count_ge"]
    out["diagnostics"] = {"same_bin_fraction": float(same_bin.sum() / (2 * len(rows) * k)),
                          "mean_abs_log_count_gap": float(gap.mean()),
                          "frac_neg_count_ge_target": float(balance.mean()),
                          "same_bin": same_bin, "log_gap": gap, "balance": balance}
    out["metadata"] = {"protocol": f"matched_{mode}", "k": k, "seed": seed,
                       "filter": sampler.filter_label,
                       "n_rows": len(rows)}
    return out


def official_count_gap(data: dict, stats: dict, rows: np.ndarray) -> dict:
    """Same diagnostics for the official negatives, for the report table."""
    split = data["valid"]
    rows = np.asarray(rows, dtype=np.int64)
    rel = np.asarray(split["relation"][rows], dtype=np.int64)
    result = {}
    for direction, target_key, neg_key in ((0, "head", "head_neg"), (1, "tail", "tail_neg")):
        targets = np.asarray(split[target_key][rows], dtype=np.int64)
        neg = np.asarray(split[neg_key][rows], dtype=np.int64)
        tc = stats["counts"][rel, direction, targets]
        nc = stats["counts"][rel[:, None], direction, neg]
        tb = classify_frequency(stats, rel, direction, targets)
        nb = np.stack([classify_frequency(stats, rel, direction, neg[:, j])
                       for j in range(neg.shape[1])], axis=1)
        result[direction] = (np.mean(nb == tb[:, None]),
                             float(np.mean(np.abs(np.log1p(nc) - np.log1p(tc)[:, None]))),
                             float(np.mean(nc >= tc[:, None])))
    return {"same_bin_fraction": float(np.mean([v[0] for v in result.values()])),
            "mean_abs_log_count_gap": float(np.mean([v[1] for v in result.values()])),
            "frac_neg_count_ge_target": float(np.mean([v[2] for v in result.values()]))}
