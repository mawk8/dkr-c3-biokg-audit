"""Small synthetic typed KG with the same dict layout as data.load_data()."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SIZES = np.array([300, 250], dtype=np.int64)
OFFSETS = np.array([0, 300], dtype=np.int64)
SCHEMA = [(0, 1), (1, 1)]  # relation 0 cross-type, relation 1 same-type


def _split(rng, n, n_neg=None):
    r = rng.integers(0, len(SCHEMA), n)
    h, t = np.empty(n, np.int64), np.empty(n, np.int64)
    for i, (a, b) in enumerate(SCHEMA):
        m = r == i
        h[m] = OFFSETS[a] + np.minimum(rng.zipf(1.6, m.sum()), SIZES[a]) - 1
        t[m] = OFFSETS[b] + np.minimum(rng.zipf(1.6, m.sum()), SIZES[b]) - 1
    out = {"head": h, "relation": r, "tail": t,
           "head_type": np.array([SCHEMA[x][0] for x in r]),
           "tail_type": np.array([SCHEMA[x][1] for x in r])}
    if n_neg:
        out["head_neg"] = np.stack([OFFSETS[SCHEMA[x][0]] + rng.integers(0, SIZES[SCHEMA[x][0]], n_neg) for x in r])
        out["tail_neg"] = np.stack([OFFSETS[SCHEMA[x][1]] + rng.integers(0, SIZES[SCHEMA[x][1]], n_neg) for x in r])
    return out


@pytest.fixture(scope="session")
def data():
    rng = np.random.default_rng(0)
    return {"train": _split(rng, 6000), "valid": _split(rng, 120, 500), "test": _split(rng, 120),
            "n_entities": int(SIZES.sum()), "n_relations": len(SCHEMA),
            "type_offsets": OFFSETS, "type_sizes": SIZES}


@pytest.fixture(scope="session")
def stats(data):
    from biokg_audit.baselines import prepare_statistics
    return prepare_statistics(data)


@pytest.fixture(scope="session")
def test_keys(data, stats):
    from biokg_audit.data import triple_keys
    t = data["test"]
    return np.unique(triple_keys(t["head"], t["relation"], t["tail"], stats["n_entities"], stats["n_relations"]))
