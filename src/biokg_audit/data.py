"""Safe, validation-only BioKG preparation for a Kaggle/Colab workspace.

Inputs are the public OGB train.pt/valid.pt files. Outputs use typed GLOBAL
entity IDs and read-only .npy memory maps. No test file is opened, extracted,
or passed to a trainer. Calling load_data is an action; importing is inert.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from datetime import datetime, timezone
import urllib.request
import uuid
import zipfile

import numpy as np
from tqdm.auto import tqdm


TYPE_NAMES = ("disease", "drug", "function", "protein", "sideeffect")
TYPE_SIZES = np.array([10687, 10533, 45085, 17499, 9969], dtype=np.int64)
TYPE_OFFSETS = np.array([0, 10687, 21220, 66305, 83804], dtype=np.int64)
N_ENTITIES = 93773
N_RELATIONS = 51
EXPECTED_COUNTS = {"train": 4762678, "valid": 162886}
OFFICIAL_URL = "https://snap.stanford.edu/ogb/data/linkproppred/biokg.zip"
CACHE_VERSION = 1
CHUNK_ROWS = 1024
_SPLIT_COLUMNS = ("row_id", "head", "relation", "tail", "head_type", "tail_type")
_SOURCE_REFS = {
    "loader": "https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/dataset.py",
    "typed_ids": "https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/biokg/dataloader.py",
    "node_metadata": "https://github.com/snap-stanford/ogb/blob/master/ogb/io/read_graph_raw.py",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _no_links(path: Path) -> Path:
    """Reject existing symlink/junction components, including input parents."""
    path = path.expanduser().absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError(f"Symlinks/junctions are not allowed: {item}")
    return path


def _write_json_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _mapping() -> dict:
    return {
        "version": "biokg_global_lexicographic_v1",
        "type_order": list(TYPE_NAMES),
        "type_sizes": TYPE_SIZES.tolist(),
        "type_offsets": TYPE_OFFSETS.tolist(),
        "global_id_formula": "type_offsets[type_id] + local_id",
        "n_entities": N_ENTITIES,
        "n_relations": N_RELATIONS,
        "source": "docs/SPEC.md section 4; checked against raw num-node-dict when present",
    }


def _find_public_root(input_dir: Path) -> Path:
    """Only exact supported locations; no recursive walks or test reads."""
    root = _no_links(input_dir)
    candidates = (root, root / "biokg", root / "ogbl_biokg")
    matches = []
    for candidate in candidates:
        _no_links(candidate)
        if all(_no_links(candidate / "split" / "random" / f"{s}.pt").is_file()
               for s in ("train", "valid")):
            matches.append(candidate)
    if len(matches) != 1:
        raise FileNotFoundError(
            "INPUT_DIR must uniquely contain split/random/train.pt and valid.pt "
            "directly, under biokg/, or under ogbl_biokg/. No recursive search."
        )
    return matches[0]


def _download_public(data_dir: Path) -> Path:
    """Explicit opt-in, append-only download and selective extraction.

    A unique exclusive directory contains both names in the partial->complete
    rename. Interrupted partial files and the complete ZIP are retained.
    """
    target_dir = _no_links(data_dir / f"public_ogb_{uuid.uuid4().hex}")
    target_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    partial = target_dir / "biokg.zip.partial"
    archive = target_dir / "biokg.zip"
    print("Downloading public OGB archive; it will be retained:", OFFICIAL_URL, flush=True)
    with urllib.request.urlopen(OFFICIAL_URL, timeout=120) as response:
        # Reject downgrade or a redirect to an unrelated distribution host.
        final_url = response.geturl()
        if not final_url.startswith("https://snap.stanford.edu/ogb/"):
            raise ValueError(f"Unexpected download redirect: {final_url}")
        expected_bytes = int(response.headers.get("Content-Length", 0))
        if expected_bytes:
            print(f"Archive size: {expected_bytes / 2**30:.2f} GiB", flush=True)
            if shutil.disk_usage(target_dir).free < expected_bytes + 4 * 2**30:
                raise OSError("Not enough free disk for archive, train/valid files and cache; need archive size plus 4 GiB reserve")
        with partial.open("xb") as stream, tqdm(
                total=expected_bytes or None, unit="B", unit_scale=True, unit_divisor=1024,
                desc="Downloading biokg.zip") as bar:
            downloaded = 0
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                downloaded += len(block)
                bar.update(len(block))
            if expected_bytes and downloaded != expected_bytes:
                raise OSError("Download length mismatch; partial file retained, no extraction performed")
    if archive.exists():
        raise FileExistsError(archive)
    # The target is a new private UUID directory, not a shared existing path.
    os.rename(partial, archive)
    print("Extracting only train, valid and small metadata (not test).", flush=True)
    wanted = {
        "biokg/split/random/train.pt", "biokg/split/random/valid.pt",
        "biokg/raw/num-node-dict.csv.gz", "biokg/RELEASE_v1.txt",
    }
    required = {"biokg/split/random/train.pt", "biokg/split/random/valid.pt"}
    seen = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in tqdm(bundle.infolist(), desc="Validating ZIP members", unit="file"):
            # Validate every archive path even though only allowlisted files
            # are extracted. No ZIP-controlled path is passed to extractall.
            pure = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (pure.is_absolute() or ".." in pure.parts or "\\" in member.filename
                    or ":" in member.filename or stat.S_ISLNK(mode)):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.filename not in wanted:
                continue
            if member.filename in seen or member.is_dir():
                raise ValueError(f"Duplicate/non-file selected ZIP member: {member.filename}")
            seen.add(member.filename)
            output = _no_links(target_dir.joinpath(*pure.parts))
            output.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, output.open("xb") as stream:
                shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
    if not required.issubset(seen):
        raise ValueError(f"Official ZIP missing required members: {sorted(required - seen)}")
    _write_json_new(target_dir / "download_provenance.json", {
        "url": OFFICIAL_URL, "final_url": final_url, "retrieved_at_utc": _utc(),
        "archive_sha256": _sha256(archive), "archive_bytes": archive.stat().st_size,
        "extracted_members": sorted(seen), "test_extracted": False,
    })
    return target_dir / "biokg"


def _node_metadata(root: Path) -> dict:
    path = _no_links(root / "raw" / "num-node-dict.csv.gz")
    if not path.is_file():
        return {"status": "not_available", "sizes_source": "fixed SPEC mapping"}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != 1 or set(rows[0]) != set(TYPE_NAMES):
        raise ValueError("num-node-dict must contain one graph and exactly the five expected types")
    actual = {name: int(rows[0][name]) for name in TYPE_NAMES}
    expected = dict(zip(TYPE_NAMES, TYPE_SIZES.tolist()))
    if actual != expected:
        raise ValueError(f"Node-size metadata mismatch: {actual}; expected {expected}")
    return {"status": "verified", "num_nodes_dict": actual, "sha256": _sha256(path)}


def _as_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _integer(name: str, values: np.ndarray, shape: tuple, low, high) -> None:
    if values.shape != shape or values.dtype.kind not in "iu":
        raise ValueError(f"{name}: expected integer shape {shape}, got {values.shape}/{values.dtype}")
    if values.size and (np.any(values < low) or np.any(values >= high)):
        raise ValueError(f"{name}: values outside the allowed typed range")


def _type_ids(name: str, value, size: int) -> np.ndarray:
    array = _as_array(value)
    if array.shape != (size,):
        raise ValueError(f"{name}: incorrect type-label shape {array.shape}")
    if array.dtype.kind in "iu":
        _integer(name, array, (size,), 0, 5)
        return array.astype(np.uint8)
    # Public OGB stores type labels as strings. Only five vector comparisons,
    # never one Python object/string per edge in the output cache.
    out = np.full(size, 255, dtype=np.uint8)
    for type_id, type_name in enumerate(TYPE_NAMES):
        out[array == type_name] = type_id
    if np.any(out == 255):
        raise ValueError(f"{name}: unknown type label")
    return out


def _npy_new(path: Path, dtype, shape: tuple) -> np.memmap:
    # Exclusive reservation prevents accidentally overwriting any cache object.
    with path.open("xb"):
        pass
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _prepare_split(root: Path, cache: Path, split: str, source_hash: str) -> dict:
    import torch  # loading is explicit; no torch import or work at module import

    print(f"Preparing {split}: trusted public .pt -> typed mmap cache", flush=True)

    source = _no_links(root / "split" / "random" / f"{split}.pt")
    # Pickle is required for public OGB NumPy dictionaries. Callers supplying
    # INPUT_DIR explicitly assert these are trusted public OGB split files.
    raw = torch.load(source, weights_only=False, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError(f"{split}.pt is not an OGB split dictionary")
    required = {"head", "relation", "tail", "head_type", "tail_type"}
    if split == "valid":
        required.update(("head_neg", "tail_neg"))
    if not required.issubset(raw):
        raise ValueError(f"{split}: missing keys {sorted(required - set(raw))}")
    n_rows = EXPECTED_COUNTS[split]
    head_type = _type_ids(f"{split}.head_type", raw["head_type"], n_rows)
    tail_type = _type_ids(f"{split}.tail_type", raw["tail_type"], n_rows)
    folder = cache / split
    folder.mkdir(exist_ok=False)
    metadata = {"n_rows": n_rows, "source_sha256": source_hash, "arrays": {}}
    for name, types in (("head", head_type), ("tail", tail_type), ("relation", None)):
        original = _as_array(raw[name])
        upper = N_RELATIONS if types is None else TYPE_SIZES[types]
        _integer(f"{split}.{name}", original, (n_rows,), 0, upper)
        dtype = np.uint16 if types is None else np.uint32
        result = _npy_new(folder / f"{name}.npy", dtype, (n_rows,))
        for begin in tqdm(range(0, n_rows, CHUNK_ROWS), desc=f"{split}.{name}", unit="chunk", leave=False):
            end = min(n_rows, begin + CHUNK_ROWS)
            block = original[begin:end].astype(np.int64)
            if types is not None:
                block += TYPE_OFFSETS[types[begin:end]]
            result[begin:end] = block
            if not np.array_equal(result[begin:end].astype(np.int64), block):
                raise ValueError(f"{split}.{name}: lossy dtype conversion")
        result.flush()
        metadata["arrays"][name] = {"source_dtype": str(original.dtype), "dtype": str(result.dtype), "shape": list(result.shape)}
        del result
    for name, values in (("head_type", head_type), ("tail_type", tail_type),
                         ("row_id", np.arange(n_rows, dtype=np.uint32))):
        result = _npy_new(folder / f"{name}.npy", values.dtype, values.shape)
        result[:] = values
        result.flush()
        metadata["arrays"][name] = {"dtype": str(values.dtype), "shape": list(values.shape)}
        if name != "row_id":
            metadata["arrays"][name]["source_dtype"] = str(_as_array(raw[name]).dtype)
        del result
    if split == "valid":
        for name, types in (("head_neg", head_type), ("tail_neg", tail_type)):
            original = _as_array(raw[name])
            if original.shape != (n_rows, 500) or original.dtype.kind not in "iu":
                raise ValueError(f"{split}.{name}: expected integer [{n_rows},500]")
            result = _npy_new(folder / f"{name}.npy", np.uint32, original.shape)
            for begin in tqdm(range(0, n_rows, CHUNK_ROWS), desc=f"{split}.{name}", unit="chunk", leave=False):
                end = min(n_rows, begin + CHUNK_ROWS)
                block = original[begin:end]
                bounds = TYPE_SIZES[types[begin:end], None]
                _integer(f"{split}.{name}", block, (end - begin, 500), 0, bounds)
                global_block = block.astype(np.int64) + TYPE_OFFSETS[types[begin:end], None]
                result[begin:end] = global_block
                if not np.array_equal(result[begin:end].astype(np.int64), global_block):
                    raise ValueError(f"{split}.{name}: lossy uint32 conversion")
            result.flush()
            metadata["arrays"][name] = {
                "source_dtype": str(original.dtype), "dtype": "uint32", "shape": list(original.shape),
                "source_ids": "type-local, retained unchanged in original .pt",
                "cache_ids": "global; recover local by subtracting row's type offset",
                "order_preserved": True, "duplicates_preserved": True,
            }
            del result
    # Hash cached values too: a completed manifest is the cache commit marker.
    for name, entry in metadata["arrays"].items():
        entry["sha256"] = _sha256(folder / f"{name}.npy")
    del raw
    return metadata


def _schema(train: dict, valid: dict) -> tuple[np.ndarray, np.ndarray]:
    heads = np.full(N_RELATIONS, -1, dtype=np.int64)
    tails = np.full(N_RELATIONS, -1, dtype=np.int64)
    for split in (train, valid):
        for relation in range(N_RELATIONS):
            selected = split["relation"] == relation
            if not np.any(selected):
                continue
            pairs = np.unique(np.stack((split["head_type"][selected], split["tail_type"][selected]), axis=1), axis=0)
            if len(pairs) != 1:
                raise ValueError(f"Relation {relation} has multiple type schemas")
            head, tail = map(int, pairs[0])
            if heads[relation] >= 0 and (heads[relation] != head or tails[relation] != tail):
                raise ValueError(f"Relation {relation}: train/valid schema disagreement")
            heads[relation], tails[relation] = head, tail
    if np.any(heads < 0):
        raise ValueError("Some expected relations have no schema in train/valid")
    return heads, tails


def load_data(workspace: Path, input_dir: Path | None = None, download: bool = False) -> dict:
    """Load public TRAIN+VALID only; downloading requires download=True.

    INPUT_DIR is an explicit trust assertion for these two public pickle files.
    Existing complete caches are verified and mmap-loaded. Incomplete artifacts
    are preserved; the next call creates a fresh UUID cache without cleanup.
    """
    workspace = _no_links(Path(workspace))
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = _no_links(workspace / "data")
    data_dir.mkdir(exist_ok=True)
    pointer = _no_links(data_dir / "public_source.json")
    if input_dir is not None:
        root = _find_public_root(Path(input_dir))
        provenance = {"kind": "explicit_trusted_public_ogb_input", "network_download_this_call": False}
    elif pointer.is_file():
        selection = json.loads(pointer.read_text(encoding="utf-8"))
        relative = PurePosixPath(selection["relative_root"])
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
            raise ValueError("Invalid persisted public source path")
        root = _find_public_root(data_dir.joinpath(*relative.parts))
        provenance = {"kind": "previous_explicit_public_download", "network_download_this_call": False}
    elif download:
        root = _download_public(data_dir)
        _write_json_new(pointer, {"relative_root": root.relative_to(data_dir).as_posix()})
        provenance = {"kind": "official_https_download", "network_download_this_call": True}
    else:
        raise FileNotFoundError("Provide trusted public OGB INPUT_DIR or explicitly set DOWNLOAD_DATA=True. No download was attempted.")
    node_metadata = _node_metadata(root)
    sources = {s: {"sha256": _sha256(root / "split" / "random" / f"{s}.pt"),
                   "bytes": (root / "split" / "random" / f"{s}.pt").stat().st_size}
               for s in ("train", "valid")}
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    # Exact pointer lookup avoids enumerating arbitrary workspace directories.
    cache_pointer = _no_links(data_dir / f"cache_v{CACHE_VERSION}_{fingerprint}.json")
    if cache_pointer.is_file():
        cache_name = json.loads(cache_pointer.read_text(encoding="utf-8"))["cache_directory"]
        if Path(cache_name).name != cache_name:
            raise ValueError("Invalid cache pointer")
        cache = _no_links(data_dir / cache_name)
        manifest = json.loads(_no_links(cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_files"] != sources or manifest["cache_version"] != CACHE_VERSION:
            raise ValueError("Cached source manifest mismatch")
    else:
        cache = _no_links(data_dir / f"cache_v{CACHE_VERSION}_{fingerprint[:16]}_{uuid.uuid4().hex}")
        cache.mkdir(exist_ok=False)
        split_metadata = {s: _prepare_split(root, cache, s, sources[s]["sha256"]) for s in ("train", "valid")}
        mapping = _mapping()
        manifest = {
            "dataset": "ogbl-biokg", "dataset_release_expected": "1",
            "cache_version": CACHE_VERSION, "created_at_utc": _utc(),
            "official_url": OFFICIAL_URL, "source_provenance": provenance,
            "source_files": sources, "source_references": _SOURCE_REFS,
            "packages": {p: _version(p) for p in ("ogb", "torch", "numpy")},
            "node_metadata": node_metadata, "splits": split_metadata,
            "expected_split_counts": EXPECTED_COUNTS,
            "mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
            "n_negative": 500, "test_opened": False,
            "test_overlap": "not_checked", "all_known_negative_filter": "not_verified_test_not_loaded",
            "stage": "preliminary_validation_only",
        }
        download_provenance = root.parent / "download_provenance.json"
        if input_dir is None and download_provenance.is_file():
            manifest["download"] = json.loads(download_provenance.read_text(encoding="utf-8"))
        _write_json_new(cache / "entity_mapping.json", mapping)
        _write_json_new(cache / "manifest.json", manifest)
        _write_json_new(cache_pointer, {"cache_directory": cache.name})
    splits = {}
    for split in ("train", "valid"):
        splits[split] = {}
        columns = _SPLIT_COLUMNS + (("head_neg", "tail_neg") if split == "valid" else ())
        for column in columns:
            path = _no_links(cache / split / f"{column}.npy")
            entry = manifest["splits"][split]["arrays"][column]
            if _sha256(path) != entry["sha256"]:
                raise ValueError(f"Cached array hash mismatch: {split}.{column}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if str(array.dtype) != entry["dtype"] or list(array.shape) != entry["shape"]:
                raise ValueError(f"Cached array metadata mismatch: {split}.{column}")
            splits[split][column] = array
    schema_head, schema_tail = _schema(splits["train"], splits["valid"])
    return {**splits, "n_entities": N_ENTITIES, "n_relations": N_RELATIONS,
            "type_offsets": TYPE_OFFSETS.copy(), "type_sizes": TYPE_SIZES.copy(),
            "type_names": TYPE_NAMES, "schema_head": schema_head, "schema_tail": schema_tail,
            "mapping": _mapping(), "manifest": manifest, "cache_dir": cache}


def load_test_positive_keys(data: dict, archive: Path | None = None) -> np.ndarray:
    """Sorted uint64 keys of TEST positives, for the filtered-negative mask ONLY.

    Reads head/relation/tail of split/random/test.pt from the retained official
    ZIP. Test negatives are never read and test triples are never scored. The
    keys are cached next to the train/valid cache as test_positive_keys.npy.
    """
    import io
    import torch

    cache = Path(data["cache_dir"])
    cached = cache / "test_positive_keys.npy"
    if cached.is_file():
        return np.load(cached, allow_pickle=False)
    if archive is None:
        data_dir = cache.parent
        selection = json.loads((data_dir / "public_source.json").read_text(encoding="utf-8"))
        archive = data_dir.joinpath(*PurePosixPath(selection["relative_root"]).parts).parent / "biokg.zip"
    with zipfile.ZipFile(_no_links(Path(archive))) as bundle:
        raw = torch.load(io.BytesIO(bundle.read("biokg/split/random/test.pt")),
                         weights_only=False, map_location="cpu")
    n_rows = len(_as_array(raw["head"]))
    head_type = _type_ids("test.head_type", raw["head_type"], n_rows).astype(np.int64)
    tail_type = _type_ids("test.tail_type", raw["tail_type"], n_rows).astype(np.int64)
    heads = _as_array(raw["head"]).astype(np.int64) + TYPE_OFFSETS[head_type]
    tails = _as_array(raw["tail"]).astype(np.int64) + TYPE_OFFSETS[tail_type]
    if np.any(heads - TYPE_OFFSETS[head_type] >= TYPE_SIZES[head_type]) or np.any(tails - TYPE_OFFSETS[tail_type] >= TYPE_SIZES[tail_type]):
        raise ValueError("test endpoint ID outside its type range")
    keys = np.unique(triple_keys(heads, _as_array(raw["relation"]).astype(np.int64), tails))
    del raw
    np.save(cached, keys, allow_pickle=False)
    return keys


def triple_keys(heads, relations, tails, n_entities: int = N_ENTITIES, n_relations: int = N_RELATIONS) -> np.ndarray:
    """Collision-free uint64 key, after callers have checked index ranges."""
    if n_entities * n_relations * n_entities - 1 > np.iinfo(np.uint64).max:
        raise OverflowError("Triple keys do not fit uint64")
    return ((np.asarray(heads, dtype=np.uint64) * np.uint64(n_relations)
             + np.asarray(relations, dtype=np.uint64)) * np.uint64(n_entities)
            + np.asarray(tails, dtype=np.uint64))


def contains_keys(sorted_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    """Vectorized membership without Python tuples or sets of graph edges."""
    if not len(sorted_keys):
        return np.zeros(query_keys.shape, dtype=bool)
    positions = np.searchsorted(sorted_keys, query_keys)
    bounded = np.minimum(positions, len(sorted_keys) - 1)
    return (positions < len(sorted_keys)) & (sorted_keys[bounded] == query_keys)


def audit_data(data: dict, output_dir: Path) -> dict:
    """Run required first-stage gates; emit JSON before raising on failure.

    Known positives here mean TRAIN union VALID only. TEST and all-known
    filtering are explicitly unverified; success never means final-test ready.
    """
    output_dir = _no_links(Path(output_dir))
    print("Auditing train/valid IDs, overlap and official negatives on CPU.", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "data_integrity.json"
    if output.exists():
        raise FileExistsError("Choose a fresh run output directory; audit reports are never overwritten")
    report = {"stage": "preliminary_validation_only", "status": "running",
              "test_opened": False, "test_overlap": "not_checked",
              "all_known_negative_filter": "not_verified_test_not_loaded", "splits": {}}
    try:
        if data["n_entities"] != N_ENTITIES or data["n_relations"] != N_RELATIONS:
            raise ValueError("Dataset entity/relation cardinality differs from SPEC")
        if not np.array_equal(data["type_sizes"], TYPE_SIZES) or not np.array_equal(data["type_offsets"], TYPE_OFFSETS):
            raise ValueError("Typed global ID mapping differs from SPEC")
        report["mapping_roundtrip"] = "passed; disjoint contiguous type intervals"
        if not np.array_equal(TYPE_OFFSETS[1:], TYPE_OFFSETS[:-1] + TYPE_SIZES[:-1]) or int(TYPE_OFFSETS[-1] + TYPE_SIZES[-1]) != N_ENTITIES:
            raise ValueError("Global mapping intervals overlap or leave gaps")
        indices = {}
        for split_name in ("train", "valid"):
            split = data[split_name]
            n = EXPECTED_COUNTS[split_name]
            for name in _SPLIT_COLUMNS:
                upper = 5 if name.endswith("type") else N_RELATIONS if name == "relation" else n if name == "row_id" else N_ENTITIES
                _integer(f"{split_name}.{name}", split[name], (n,), 0, upper)
            if not np.array_equal(split["row_id"], np.arange(n, dtype=np.uint32)):
                raise ValueError(f"{split_name}: original row order has changed")
            if split["head_type"].dtype != np.uint8 or split["tail_type"].dtype != np.uint8:
                raise ValueError(f"{split_name}: expected uint8 type IDs")
            for role in ("head", "tail"):
                types = split[f"{role}_type"]
                offsets = TYPE_OFFSETS[types]
                _integer(f"{split_name}.{role}", split[role], (n,), offsets, offsets + TYPE_SIZES[types])
                if not np.array_equal(split[f"{role}_type"], data[f"schema_{role}"][split["relation"]]):
                    raise ValueError(f"{split_name}: relation schema mismatch")
            keys = np.sort(triple_keys(split["head"], split["relation"], split["tail"]))
            duplicate_count = int(np.count_nonzero(keys[1:] == keys[:-1]))
            indices[split_name] = np.unique(keys)
            report["splits"][split_name] = {
                "n_rows": n, "expected_count_gate": "passed", "duplicate_rows": duplicate_count,
                "dtype_and_range_gate": "passed", "relation_counts": np.bincount(split["relation"], minlength=N_RELATIONS).tolist(),
                "dtypes": {k: str(split[k].dtype) for k in _SPLIT_COLUMNS},
            }
        overlap = int(np.intersect1d(indices["train"], indices["valid"], assume_unique=True).size)
        report["direct_train_valid_overlap_unique_triples"] = overlap
        if overlap:
            raise ValueError("Exact train/valid positive overlap: evaluation is on hold")
        # This index stays in this audit function and is never returned to a trainer.
        audit_known = np.union1d(indices["train"], indices["valid"])
        valid = data["valid"]
        for role in ("head", "tail"):
            negative = valid[f"{role}_neg"]
            print(f"Auditing {role} negatives (all 500 per validation row).", flush=True)
            n = EXPECTED_COUNTS["valid"]
            if negative.shape != (n, 500) or negative.dtype.kind not in "iu":
                raise ValueError(f"valid.{role}_neg: expected integer [{n},500]")
            duplicates = 0
            hit_count = 0
            for begin in range(0, n, CHUNK_ROWS):
                end = min(n, begin + CHUNK_ROWS)
                block = negative[begin:end]
                types = valid[f"{role}_type"][begin:end]
                lo = TYPE_OFFSETS[types, None]
                _integer(f"valid.{role}_neg", block, block.shape, lo, lo + TYPE_SIZES[types, None])
                local = block.astype(np.int64) - lo
                if not np.array_equal(local + lo, block):
                    raise ValueError("Negative global/local round-trip failed")
                ordered = np.sort(block, axis=1)
                duplicates += int(np.count_nonzero(ordered[:, 1:] == ordered[:, :-1]))
                relations = valid["relation"][begin:end, None]
                fixed = valid["tail" if role == "head" else "head"][begin:end, None]
                candidate_keys = triple_keys(block, relations, fixed) if role == "head" else triple_keys(fixed, relations, block)
                hit_count += int(np.count_nonzero(contains_keys(audit_known, candidate_keys)))
            report["splits"]["valid"][f"{role}_negative_audit"] = {
                "shape": list(negative.shape), "dtype": str(negative.dtype),
                "duplicate_entries_beyond_first_per_row": duplicates,
                "train_or_valid_positive_collisions": hit_count,
                "order_preserved": True, "duplicates_preserved": True,
            }
            if hit_count:
                raise ValueError(f"Official {role} negatives contain TRAIN/VALID positives; source investigation required")
        report["schema"] = {"head_type": data["schema_head"].tolist(), "tail_type": data["schema_tail"].tolist()}
        report["status"] = "passed_first_stage_only"
        report["required_first_stage_gates_passed"] = True
    except Exception as error:
        report["status"] = "failed"
        report["required_first_stage_gates_passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json_new(output, report)
        raise
    _write_json_new(output, report)
    return report
