import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from biokg_audit import data as loader
from biokg_audit.data import load_test_positive_keys, triple_keys


@pytest.fixture
def source(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    root = tmp_path / "local" / "biokg"
    test = root / "split" / "random" / "test.pt"
    test.parent.mkdir(parents=True)
    test.write_bytes(b"test source")
    raw = {"head": np.array([2]), "tail": np.array([3]), "relation": np.array([0]),
           "head_type": ["drug"], "tail_type": ["disease"]}
    calls = []

    def load(stream, **kwargs):
        calls.append(stream.read())
        return raw

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    data = {"cache_dir": cache, "public_source": {"root": root, "archive": None},
            "manifest": {"source_files": {"train": {"sha256": "train"}, "valid": {"sha256": "valid"}}}}
    return data, test, raw, calls


def test_local_test_source_ignores_stale_pointer_and_unverified_cache(source):
    data, test, _, calls = source
    cache = data["cache_dir"]
    (cache.parent / "public_source.json").write_text('{"relative_root": "stale/biokg"}')
    np.save(cache / "test_positive_keys.npy", np.array([0], dtype=np.uint64))
    expected = triple_keys([10689], [0], [3])
    np.testing.assert_array_equal(load_test_positive_keys(data), expected)
    np.testing.assert_array_equal(load_test_positive_keys(data), expected)
    assert calls == [test.read_bytes()]


def test_load_data_uses_explicit_root_for_default_mask(source, tmp_path, monkeypatch):
    _, test, _, calls = source
    root = test.parents[2]
    for split in ("train", "valid"):
        test.with_name(f"{split}.pt").write_bytes(split.encode())

    def prepare_split(root, cache, split, source_hash):
        folder = cache / split
        folder.mkdir()
        columns = loader._SPLIT_COLUMNS + (("head_neg", "tail_neg") if split == "valid" else ())
        entries = {}
        for name in columns:
            path = folder / f"{name}.npy"
            np.save(path, np.zeros(1, dtype=np.uint8))
            entries[name] = {"dtype": "uint8", "shape": [1], "sha256": loader._sha256(path)}
        return {"arrays": entries}

    monkeypatch.setattr(loader, "_prepare_split", prepare_split)
    monkeypatch.setattr(loader, "_schema", lambda *_: (np.zeros(1), np.zeros(1)))
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "public_source.json").write_text('{"relative_root": "stale/biokg"}')
    data = loader.load_data(workspace, input_dir=root.parent)
    assert calls == []
    np.testing.assert_array_equal(load_test_positive_keys(data), triple_keys([10689], [0], [3]))
    assert calls == [test.read_bytes()]


@pytest.mark.parametrize(("field", "bad"), [("head", [-1]), ("tail", [10687]),
                                           ("relation", [-1]), ("relation", [51]),
                                           ("relation", [0.5]), ("head", [[2]])])
def test_rejects_invalid_typed_rows(source, field, bad):
    data, _, raw, _ = source
    raw[field] = np.array(bad)
    with pytest.raises(ValueError, match=f"test.{field}"):
        load_test_positive_keys(data)


def test_cache_changes_with_test_bytes_and_rejects_corruption(source):
    data, test, _, calls = source
    load_test_positive_keys(data)
    test.write_bytes(b"replacement")
    load_test_positive_keys(data)
    assert calls == [b"test source", b"replacement"]
    relative = Path(data["test_positive_mask_provenance"]["cache_path"])
    assert not relative.is_absolute()
    keys_path = data["cache_dir"] / relative
    np.save(keys_path, np.array([0], dtype=np.uint64))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_test_positive_keys(data)


def test_reads_only_selected_download_release(source):
    data, path, _, calls = source
    archive = path.parents[3] / "biokg.zip"
    with ZipFile(archive, "x") as bundle:
        bundle.writestr("biokg/split/random/test.pt", b"downloaded test source")
        for split in ("train", "valid"):
            bundle.writestr(f"biokg/split/random/{split}.pt", split.encode())
            data["manifest"]["source_files"][split]["sha256"] = hashlib.sha256(split.encode()).hexdigest()
    (archive.parent / "download_provenance.json").write_text(json.dumps({
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))
    data["public_source"]["archive"] = archive
    load_test_positive_keys(data)
    assert calls == [b"downloaded test source"]
    data["manifest"]["source_files"]["train"]["sha256"] = "other data"
    with pytest.raises(ValueError, match="archive train hash mismatch"):
        load_test_positive_keys(data)


def test_interrupted_cache_is_preserved(source, monkeypatch):
    data, _, _, calls = source
    with monkeypatch.context() as patch:
        def interrupted(*args):
            raise OSError("interrupted before cache publish")

        patch.setattr(loader.os, "rename", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            load_test_positive_keys(data)
    partial = list(data["cache_dir"].rglob("manifest.json.partial"))
    load_test_positive_keys(data)
    assert len(partial) == 1 and partial[0].is_file()
    assert len(calls) == 2
