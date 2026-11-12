"""Train-only statistics and cheap candidate scorers.

The scorer input has row_id[B], scalar direction (head=0, tail=1), relation[B],
fixed[B], target_type[B]. It contains no target or positive flag. Candidate
matrices are GLOBAL IDs [B,K]; larger scores mean a better candidate.
"""

from __future__ import annotations

import numpy as np

from .data import contains_keys, triple_keys


def _train_arrays(data: dict) -> tuple:
    train = data["train"]
    return tuple(np.asarray(train[k], dtype=np.int64) for k in ("head", "relation", "tail"))


def prepare_statistics(data: dict) -> dict:
    """Return count tables, compact TRAIN index, and train-universe thresholds.

    Validation/test labels are never read. Schema is inferred independently
    from train type IDs, rather than using potentially held-out schema labels.
    Count arrays include zeros for every entity and retain duplicate train rows.
    """
    n, r = int(data["n_entities"]), int(data["n_relations"])
    train = data["train"]
    heads, relations, tails = _train_arrays(data)
    sizes = np.asarray(data["type_sizes"], dtype=np.int64)
    offsets = np.asarray(data["type_offsets"], dtype=np.int64)
    counts = np.zeros((r, 2, n), dtype=np.int64)
    np.add.at(counts[:, 0, :], (relations, heads), 1)
    np.add.at(counts[:, 1, :], (relations, tails), 1)
    schema = np.full((r, 2), -1, dtype=np.int64)
    thresholds = np.zeros((r, 2, 2), dtype=np.int64)
    cells = []
    for relation in range(r):
        selected = relations == relation
        if not np.any(selected):
            raise ValueError(f"Relation {relation}: no train schema; held-out schema cannot fill it")
        for direction, role in ((0, "head"), (1, "tail")):
            types = np.unique(train[f"{role}_type"][selected])
            if len(types) != 1:
                raise ValueError(f"Relation {relation}, {role}: multiple train schemas")
            type_id = int(types[0])
            schema[relation, direction] = type_id
            start, stop = int(offsets[type_id]), int(offsets[type_id] + sizes[type_id])
            universe = counts[relation, direction, start:stop]
            positive = np.sort(universe[universe > 0])
            q1 = int(positive[(len(positive) + 2) // 3 - 1]) if len(positive) else 0
            q2 = int(positive[(2 * len(positive) + 2) // 3 - 1]) if len(positive) else 0
            thresholds[relation, direction] = (q1, q2)
            total = int(universe.sum(dtype=np.int64))
            ordered = np.sort(universe)
            ranks = np.arange(1, len(ordered) + 1, dtype=np.float64)
            gini = float(2 * np.dot(ranks, ordered.astype(np.float64)) / (len(ordered) * total)
                         - (len(ordered) + 1) / len(ordered)) if total else None
            cells.append({"relation": relation, "direction": direction, "target_type": type_id,
                          "universe_size": len(universe), "nonzero_entities": int(len(positive)),
                          "zero_fraction": float(np.mean(universe == 0)), "gini": gini,
                          "q1": q1, "q2": q2, "train_role_rows": total})
    train_keys = np.unique(triple_keys(heads, relations, tails, n, r))
    same_type = schema[:, 0] == schema[:, 1]
    reverse = contains_keys(train_keys, triple_keys(tails, relations, heads, n, r)) & same_type[relations]
    reverse_rows = []
    for relation in range(r):
        selected = relations == relation
        self_loop = selected & (heads == tails)
        nonself = selected & (heads != tails)
        applicable = bool(same_type[relation])
        denominator = int(np.count_nonzero(nonself))
        reverse_rows.append({"relation": relation, "applicable": applicable,
                             "n_train_rows": int(np.count_nonzero(selected)),
                             "self_loop_rows": int(np.count_nonzero(self_loop)),
                             "nonself_train_rows": denominator,
                             "reverse_rows": int(np.count_nonzero(reverse & selected)) if applicable else None,
                             "reverse_nonself_fraction": float(np.count_nonzero(reverse & nonself) / denominator) if applicable and denominator else None})
    return {"counts": counts, "train_keys": train_keys, "same_type": same_type,
            "schema_head": schema[:, 0], "schema_tail": schema[:, 1],
            "thresholds": thresholds, "bins": ("zero", "low", "mid", "high"),
            "frequency_cells": cells, "reverse_diagnostics": reverse_rows,
            "type_offsets": offsets.copy(), "type_sizes": sizes.copy(),
            "n_entities": n, "n_relations": r,
            "statistics_source": "train_only; duplicate original rows retained for counts"}


def _query(query: dict, candidates: np.ndarray) -> tuple:
    forbidden = {"target", "target_global", "positive", "positive_flag", "frequency_bin", "inverse_available_positive"}
    if forbidden.intersection(query):
        raise ValueError("Scorers must not receive labels or diagnostic features")
    candidates = np.asarray(candidates)
    if candidates.ndim != 2 or candidates.dtype.kind not in "iu":
        raise ValueError("Candidates must be integer GLOBAL IDs [B,K]")
    direction = query["direction"]
    if np.ndim(direction) != 0 or int(direction) not in (0, 1):
        raise ValueError("Each scorer batch must have one scalar direction (0=head, 1=tail)")
    size = candidates.shape[0]
    values = []
    for name in ("row_id", "relation", "fixed", "target_type"):
        array = np.asarray(query[name])
        if array.shape != (size,) or array.dtype.kind not in "iu":
            raise ValueError(f"query.{name} must be integer [B]")
        values.append(array.astype(np.int64, copy=False))
    return (int(direction), *values, candidates.astype(np.int64, copy=False))


def _typed_candidates(stats: dict, direction, relations, fixed, types, candidates) -> None:
    if np.any(relations < 0) or np.any(relations >= stats["n_relations"]):
        raise ValueError("Query relation outside train schema")
    expected_type = stats["schema_head" if direction == 0 else "schema_tail"][relations]
    if not np.array_equal(types, expected_type):
        raise ValueError("Query target type disagrees with train-only schema")
    low = stats["type_offsets"][types, None]
    high = low + stats["type_sizes"][types, None]
    if np.any(candidates < low) or np.any(candidates >= high):
        raise ValueError("Candidate GLOBAL ID has the wrong endpoint type")
    fixed_types = stats["schema_tail" if direction == 0 else "schema_head"][relations]
    fixed_low = stats["type_offsets"][fixed_types]
    if np.any(fixed < fixed_low) or np.any(fixed >= fixed_low + stats["type_sizes"][fixed_types]):
        raise ValueError("Fixed GLOBAL ID has the wrong endpoint type")


class CountPrior:
    """Count of candidate in the requested relation/role in TRAIN, int64."""

    def __init__(self, stats: dict):
        self.stats = stats

    def score_candidates(self, query: dict, candidates: np.ndarray) -> np.ndarray:
        direction, _, relations, fixed, types, candidates = _query(query, candidates)
        _typed_candidates(self.stats, direction, relations, fixed, types, candidates)
        return self.stats["counts"][relations[:, None], direction, candidates]


class GlobalDegreePrior:
    """Total TRAIN degree of the candidate over ALL relations and roles.

    Unlike CountPrior it ignores the queried relation, so frequency-matched
    negatives (matched on the relation/role count) do not neutralise it by
    construction. It tests whether generic hub-ness survives matching.
    """

    def __init__(self, stats: dict):
        self.stats = stats
        self.degree = stats["counts"].sum(axis=(0, 1), dtype=np.int64)

    def score_candidates(self, query: dict, candidates: np.ndarray) -> np.ndarray:
        direction, _, relations, fixed, types, candidates = _query(query, candidates)
        _typed_candidates(self.stats, direction, relations, fixed, types, candidates)
        return self.degree[candidates]


class ReverseLookup:
    """TRAIN reverse indicator + bounded count score; float64, no labels."""

    def __init__(self, stats: dict, reverse_only: bool = False):
        self.stats = stats
        self.reverse_only = bool(reverse_only)

    def score_candidates(self, query: dict, candidates: np.ndarray) -> np.ndarray:
        direction, _, relations, fixed, types, candidates = _query(query, candidates)
        _typed_candidates(self.stats, direction, relations, fixed, types, candidates)
        # For (?,r,t), candidate reverse is (t,r,c); for (h,r,?),
        # candidate reverse is (c,r,h). Cross-type swaps are inapplicable.
        keys = (triple_keys(fixed[:, None], relations[:, None], candidates,
                            self.stats["n_entities"], self.stats["n_relations"]) if direction == 0
                else triple_keys(candidates, relations[:, None], fixed[:, None],
                                 self.stats["n_entities"], self.stats["n_relations"]))
        indicator = contains_keys(self.stats["train_keys"], keys)
        indicator &= self.stats["same_type"][relations, None]
        score = indicator.astype(np.float64)
        if not self.reverse_only:
            counts = self.stats["counts"][relations[:, None], direction, candidates].astype(np.float64)
            score += 0.25 * counts / (counts + 1.0)
        return score


def _mix64(value: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer; arithmetic modulo 2**64 is intentional.

    Algorithm constants: Steele, Lea, Flood, SPLIT/OOPSLA 2014;
    https://prng.di.unimi.it/splitmix64.c (public-domain reference).
    """
    with np.errstate(over="ignore"):
        value = np.asarray(value, dtype=np.uint64).copy()
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


class RandomScorer:
    """Stable candidate-specific float64 [0,1), independent of column order.

    The optional split defaults to valid for this validation-only backend.
    There is no Python hash, RNG state, target argument, or special column 0.
    Identical candidates in one query receive identical scores.
    """

    def __init__(self, seed: int = 101):
        if not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**64:
            raise ValueError("Random seed must fit uint64")
        self.seed = int(seed)

    def score_candidates(self, query: dict, candidates: np.ndarray) -> np.ndarray:
        direction, rows, _, _, _, candidates = _query(query, candidates)
        split_codes = {"train": 1, "valid": 2, "test": 3}
        split = query.get("split", "valid")
        if split not in split_codes:
            raise ValueError("Random scorer needs a stable known split name")
        if np.any(rows < 0) or np.any(candidates < 0):
            raise ValueError("Rows and GLOBAL candidate IDs must be nonnegative")
        # Stable serialization is a sequence of uint64 fields mixed with
        # separate domain tags, not ambiguous decimal/string concatenation.
        seed_key = _mix64(np.array([self.seed], dtype=np.uint64))[0]
        row_key = _mix64(rows.astype(np.uint64) ^ np.uint64(0x243F6A8885A308D3))
        domain = np.uint64(split_codes[split] * 2 + direction)
        query_key = _mix64(row_key ^ seed_key ^ _mix64(np.array([domain]))[0])
        mixed = _mix64(query_key[:, None] ^ _mix64(candidates.astype(np.uint64) ^ np.uint64(0x13198A2E03707344)))
        return (mixed >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def classify_frequency(stats: dict, relations, direction: int, targets) -> np.ndarray:
    """Analysis-only zero/low/mid/high labels using frozen TRAIN thresholds."""
    if direction not in (0, 1):
        raise ValueError("direction must be head=0 or tail=1")
    relations = np.asarray(relations, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if relations.shape != targets.shape or relations.ndim != 1:
        raise ValueError("Relations and target IDs must be matching [B] arrays")
    counts = stats["counts"][relations, direction, targets]
    q1 = stats["thresholds"][relations, direction, 0]
    q2 = stats["thresholds"][relations, direction, 1]
    labels = np.full(counts.shape, "high", dtype="<U4")
    labels[counts <= q2] = "mid"
    labels[counts <= q1] = "low"
    labels[counts == 0] = "zero"
    return labels


def reverse_available(stats: dict, relations, heads, tails) -> tuple[np.ndarray, np.ndarray]:
    """Analysis-only (applicable, available) booleans for positive triples.

    Available=False for cross-type relations is masked by applicable=False;
    exported tables must represent that diagnostic stratum as N/A.
    """
    relations = np.asarray(relations, dtype=np.int64)
    heads, tails = np.asarray(heads, dtype=np.int64), np.asarray(tails, dtype=np.int64)
    if relations.ndim != 1 or heads.shape != relations.shape or tails.shape != relations.shape:
        raise ValueError("Relations, heads, tails must be matching [B] arrays")
    applicable = stats["same_type"][relations]
    available = contains_keys(stats["train_keys"], triple_keys(tails, relations, heads,
                              stats["n_entities"], stats["n_relations"])) & applicable
    return applicable, available
