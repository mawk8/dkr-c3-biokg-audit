"""Cloud-only FP32 scorers and a small, reproducible C3 pilot trainer.

Nothing trains or downloads on import. Torch must already be installed in the
cloud environment. NumPy train arrays stay on the host; only one sampled batch
is transferred to the model device. ``dim`` counts REAL scalars per entity.

Input queries contain scalar direction (0=head, 1=tail), relation[B], fixed[B],
and optional bookkeeping fields. A scorer never receives/reads the target label.
Output scores[B,K] are float32, with larger meaning more plausible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import nn
from torch.nn import functional as F


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _direction(query: Mapping[str, Any]) -> int:
    value = query["direction"]
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or int(value) not in (0, 1):
        raise ValueError("direction must be scalar 0=head or 1=tail")
    return int(value)


class LearnedScorer(nn.Module):
    """Common candidate scorer; row_id, target_type and split are not features."""

    model_name: str

    def __init__(self, n_entities: int, n_relations: int, dim: int, seed: int):
        super().__init__()
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.dim = int(dim)
        self.seed = int(seed)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _index(self, values: Any) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=self.device)

    def score_torch(self, query: Mapping[str, Any], candidates: Any) -> torch.Tensor:
        raise NotImplementedError

    def score_candidates(self, query: Mapping[str, Any], candidates: np.ndarray) -> np.ndarray:
        """Evaluate one candidate chunk. The evaluator controls B/K for memory."""
        previous_mode = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.score_torch(query, candidates).detach().cpu().numpy()
        finally:
            self.train(previous_mode)

    def model_config(self) -> dict[str, Any]:
        return {"name": self.model_name, "dim": self.dim, "seed": self.seed,
                "n_entities": self.n_entities, "n_relations": self.n_relations,
                "precision": "float32", "gamma": 12.0 if self.model_name == "transe" else None}


class HeadBlind(LearnedScorer):
    model_name = "head_blind"

    def __init__(self, n_entities: int, n_relations: int, dim: int, seed: int = 101):
        super().__init__(n_entities, n_relations, dim, seed)
        self.entity = nn.Embedding(n_entities, dim)
        self.relation_role = nn.Parameter(torch.empty(n_relations, 2, dim))
        self.bias = nn.Parameter(torch.zeros(n_entities, 2))
        nn.init.normal_(self.entity.weight, std=1.0 / math.sqrt(dim))
        nn.init.normal_(self.relation_role, std=1.0 / math.sqrt(dim))

    def score_torch(self, query: Mapping[str, Any], candidates: Any) -> torch.Tensor:
        direction = _direction(query)
        relation = self._index(query["relation"])
        candidate = self._index(candidates)
        role = self.relation_role[relation, direction]
        # Deliberately do not even read query['fixed'].
        return (self.entity(candidate) * role[:, None, :]).sum(-1) + self.bias[candidate, direction]


class TransE(LearnedScorer):
    model_name = "transe"
    gamma = 12.0

    def __init__(self, n_entities: int, n_relations: int, dim: int, seed: int = 101):
        super().__init__(n_entities, n_relations, dim, seed)
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        bound = (self.gamma + 2.0) / dim
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)

    def score_torch(self, query: Mapping[str, Any], candidates: Any) -> torch.Tensor:
        direction = _direction(query)
        fixed = self.entity(self._index(query["fixed"]))[:, None, :]
        relation = self.relation(self._index(query["relation"]))[:, None, :]
        candidate = self.entity(self._index(candidates))
        displacement = fixed + relation - candidate if direction == 1 else candidate + relation - fixed
        return self.gamma - displacement.abs().sum(-1)


class ComplEx(LearnedScorer):
    model_name = "complex"

    def __init__(self, n_entities: int, n_relations: int, dim: int, seed: int = 101):
        if dim % 2:
            raise ValueError("ComplEx dim counts real scalars and must be even")
        super().__init__(n_entities, n_relations, dim, seed)
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        nn.init.normal_(self.entity.weight, std=1.0 / math.sqrt(dim // 2))
        nn.init.normal_(self.relation.weight, std=1.0 / math.sqrt(dim // 2))

    def score_torch(self, query: Mapping[str, Any], candidates: Any) -> torch.Tensor:
        direction = _direction(query)
        fixed = self.entity(self._index(query["fixed"]))[:, None, :]
        candidate = self.entity(self._index(candidates))
        relation = self.relation(self._index(query["relation"]))[:, None, :]
        head, tail = (fixed, candidate) if direction == 1 else (candidate, fixed)
        hr, hi = head.chunk(2, -1)
        rr, ri = relation.chunk(2, -1)
        tr, ti = tail.chunk(2, -1)
        return (hr * rr * tr + hi * rr * ti + hr * ri * ti - hi * ri * tr).sum(-1)


def make_model(name: str, data: Mapping[str, Any], dim: int = 128, seed: int = 101,
               device: str = "cuda") -> LearnedScorer:
    """Initialize in an isolated RNG scope, then move only weights to device."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Enable a Kaggle GPU; CPU requires device='cpu' explicitly.")
    classes = {"head_blind": HeadBlind, "transe": TransE, "complex": ComplEx}
    if name not in classes:
        raise ValueError(f"Unknown learned model: {name!r}")
    with torch.random.fork_rng(devices=[]):
        # Seed CPU initialization only; do not disturb any existing CUDA stream.
        torch.random.default_generator.manual_seed(seed)
        model = classes[name](int(data["n_entities"]), int(data["n_relations"]), dim, seed)
    return model.to(device=requested, dtype=torch.float32)


@dataclass(frozen=True)
class TrainConfig:
    """Short pilot defaults, intentionally smaller than SPEC final-run settings."""

    seed: int = 101
    steps: int = 500
    batch_size: int = 256
    negatives: int = 64
    learning_rate: float = 0.001
    log_every: int = 50
    checkpoint_every: int = 250
    max_minutes: float = 20.0
    validation_every: int = 250
    selection_max_per_relation: int | None = 20
    selection_seed: int = 71

    def validate(self) -> None:
        for key in ("steps", "batch_size", "negatives", "log_every", "checkpoint_every",
                    "validation_every"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        selection = self.selection_max_per_relation
        if selection is not None and (not isinstance(selection, int) or isinstance(selection, bool) or selection <= 0):
            raise ValueError("selection_max_per_relation must be a positive integer or None")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.max_minutes) or self.max_minutes <= 0:
            raise ValueError("max_minutes must be finite and positive")

    def scientific(self) -> dict[str, Any]:
        value = asdict(self)
        # These affect only artifact frequency / this invocation's walltime cap.
        for key in ("log_every", "checkpoint_every", "max_minutes"):
            value.pop(key)
        return value


def _identity(data: Mapping[str, Any]) -> dict[str, str]:
    """Use the prepared manifest, never rehash multi-GB arrays every checkpoint.

    The loader/audit must verify the file hashes declared in this manifest.
    Manifest identity alone cannot detect arrays altered after that verification.
    """
    return {"mapping_sha256": _sha(data["mapping"]),
            "dataset_manifest_sha256": _sha(data["manifest"])}


def _environment(model: LearnedScorer) -> dict[str, Any]:
    device = model.device
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def _source_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sync(model: LearnedScorer) -> None:
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def _json_new(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _unused_path(directory: Path, stem: str, extension: str) -> Path:
    candidate = directory / f"{stem}{extension}"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{suffix:03d}{extension}"
        suffix += 1
    return candidate


def _streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    # Explicit independent streams; model initialization cannot advance either.
    return (np.random.default_rng(np.random.SeedSequence([seed, 1701])),
            np.random.default_rng(np.random.SeedSequence([seed, 1702])))


def _sample_batch(data: Mapping[str, Any], config: TrainConfig, step: int,
                  positive_rng: np.random.Generator,
                  negative_rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray, str]:
    train = data["train"]
    rows = positive_rng.integers(0, len(train["head"]), size=config.batch_size, dtype=np.int64)
    direction = 1 if step % 2 == 1 else 0  # update 1 tail, update 2 head
    target_name, fixed_name = ("tail", "head") if direction else ("head", "tail")
    types = np.asarray(train[f"{target_name}_type"][rows], dtype=np.int64)
    offsets = np.asarray(data["type_offsets"], dtype=np.int64)[types]
    sizes = np.asarray(data["type_sizes"], dtype=np.int64)[types]
    # Integer RNG accepts per-row upper bounds; duplicates/positives are allowed.
    negative = negative_rng.integers(0, sizes[:, None],
                                     size=(config.batch_size, config.negatives), dtype=np.int64)
    negative += offsets[:, None]
    positive = np.asarray(train[target_name][rows], dtype=np.int64)
    candidates = np.concatenate((positive[:, None], negative), axis=1)
    query = {"row_id": rows, "direction": direction,
             "relation": np.asarray(train["relation"][rows], dtype=np.int64),
             "fixed": np.asarray(train[fixed_name][rows], dtype=np.int64), "target_type": types}
    digest = hashlib.sha256()
    digest.update(bytes([direction]))
    for values in (rows, query["relation"], query["fixed"], types, candidates):
        digest.update(np.asarray(values, dtype="<i8").tobytes(order="C"))
    return query, candidates, digest.hexdigest()


def _validate_train(data: Mapping[str, Any], model: LearnedScorer) -> None:
    train = data["train"]
    length = len(train["head"])
    if length == 0:
        raise ValueError("Train split is empty")
    for name in ("head", "relation", "tail", "head_type", "tail_type"):
        array = train[name]
        if array.shape != (length,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"train[{name}] must be an integer vector with common length")
    if (int(data["n_entities"]), int(data["n_relations"])) != (model.n_entities, model.n_relations):
        raise ValueError("Model/data dimensions differ")
    sizes = np.asarray(data["type_sizes"], dtype=np.int64)
    offsets = np.asarray(data["type_offsets"], dtype=np.int64)
    if sizes.ndim != 1 or offsets.shape != sizes.shape or np.any(sizes <= 0):
        raise ValueError("Invalid typed entity universe")
    if not np.array_equal(offsets, np.r_[0, np.cumsum(sizes)[:-1]]) or sizes.sum() != model.n_entities:
        raise ValueError("Type offsets must partition the global entity universe")
    for name, limit in (("head", model.n_entities), ("tail", model.n_entities),
                        ("relation", model.n_relations), ("head_type", len(sizes)), ("tail_type", len(sizes))):
        if int(np.min(train[name])) < 0 or int(np.max(train[name])) >= limit:
            raise ValueError(f"train[{name}] is outside its ID universe")
    # Chunk checks avoid allocating full-length temporary typed-ID arrays.
    for start in range(0, length, 100_000):
        stop = min(start + 100_000, length)
        relation = np.asarray(train["relation"][start:stop], dtype=np.int64)
        for role in ("head", "tail"):
            entity = train[role][start:stop]
            types = np.asarray(train[f"{role}_type"][start:stop], dtype=np.int64)
            schema = np.asarray(data[f"schema_{role}"], dtype=np.int64)
            if schema.shape != (model.n_relations,) or not np.array_equal(types, schema[relation]):
                raise ValueError(f"Train {role} types disagree with relation schema")
            if np.any(entity < offsets[types]) or np.any(entity >= offsets[types] + sizes[types]):
                raise ValueError(f"Train {role} global IDs disagree with their types")


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Load ONLY trusted checkpoints produced by this project (pickle payload)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_checkpoint(model: LearnedScorer, path: str | Path,
                          data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Restore one saved scoring snapshot, not optimizer/training streams.

    Return metadata so tables can label the actual saved step and selection kind.
    The caller must supply a trusted project-created checkpoint.
    """
    checkpoint = _read_checkpoint(Path(path))
    if checkpoint["format_version"] != 1 or checkpoint["model_config"] != model.model_config():
        raise ValueError("Checkpoint/model configuration mismatch")
    if data is not None and checkpoint["data_identity"] != _identity(data):
        raise ValueError("Checkpoint dataset/mapping mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return {"step": checkpoint["step"], "kind": checkpoint["kind"],
            "model_config": checkpoint["model_config"], "best_mrr": checkpoint["best_mrr"],
            "created_at_utc": checkpoint["created_at_utc"]}


def train_model(model: LearnedScorer, data: Mapping[str, Any], config: TrainConfig,
                run_dir: Path, evaluate_fn: Callable[[LearnedScorer, int], Mapping[str, Any]] | None = None,
                resume_path: str | Path | None = None, progress: bool = True) -> dict[str, Any]:
    """Train uniform typed batches and write immutable pilot artifacts.

    evaluate_fn(model, step) must use one fixed train-independent validation ID
    selection and return {'mrr': finite float, ...JSON metadata...}. Higher MRR
    selects the checkpoint; ties retain the earlier checkpoint. No test is read.
    max_minutes is a per-invocation cap checked between updates; a single update,
    validation or checkpoint write may exceed the remaining time. It is not a
    full final-run benchmark. Resume requires identical scientific config, source,
    data identities and recorded device/software environment.
    """
    config.validate()
    if config.seed != model.seed:
        raise ValueError("Training seed must equal model initialization seed")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Only float32 pilot models are supported")
    _validate_train(data, model)
    run_dir = Path(run_dir).resolve()
    identity, environment = _identity(data), _environment(model)
    scientific_sha = _sha(config.scientific())
    source_sha = _source_sha()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    positive_rng, negative_rng = _streams(config.seed)
    step = 0
    best_mrr: float | None = None
    best_checkpoint: str | None = None
    last_checkpoint: str | None = None
    validation_history: list[dict[str, Any]] = []
    batch_hashes: list[dict[str, Any]] = []
    timings: list[float] = []
    previous_wall_seconds = 0.0
    validation_enabled = evaluate_fn is not None
    checkpoint = None
    if resume_path is not None:
        resume = Path(resume_path).resolve()
        # An explicit resume may append only within the same immutable run.
        if resume.parent != run_dir / "checkpoints" or not run_dir.is_dir():
            raise ValueError("Resume checkpoint must belong to run_dir/checkpoints")
        checkpoint = _read_checkpoint(resume)
        expected = {"format_version": 1, "scientific_config_sha256": scientific_sha,
                    "model_config": model.model_config(), "data_identity": identity,
                    "environment": environment, "source_sha256": source_sha,
                    "validation_enabled": validation_enabled, "run_dir": str(run_dir)}
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"Strict resume mismatch: {key}")
        run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        for key in ("scientific_config_sha256", "model_config", "data_identity", "source_sha256",
                    "validation_enabled"):
            if run_metadata.get(key) != expected[key]:
                raise ValueError(f"Run metadata mismatch: {key}")
        step = int(checkpoint["step"])
        if not 0 <= step <= config.steps:
            raise ValueError("Checkpoint step outside requested run")
        # Appending a branch from an older step would create contradictory logs.
        history_path = run_dir / "history.jsonl"
        if history_path.exists():
            with history_path.open(encoding="utf-8") as handle:
                for line in handle:
                    event = json.loads(line)
                    if int(event.get("step", 0)) > step:
                        raise ValueError("Run has later logged steps; resume its latest checkpoint or use a separate run")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        positive_rng.bit_generator.state = checkpoint["positive_rng"]
        negative_rng.bit_generator.state = checkpoint["negative_rng"]
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_global_rng"])
        torch.set_rng_state(checkpoint["torch_cpu_rng"])
        if model.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng"])
        best_mrr = checkpoint["best_mrr"]
        best_checkpoint = checkpoint["best_checkpoint"]
        last_checkpoint = str(resume)
        validation_history = checkpoint["validation_history"]
        batch_hashes = checkpoint["first_batch_hashes"]
        timings = checkpoint["step_seconds"]
        previous_wall_seconds = checkpoint["total_wall_seconds"]
    else:
        if run_dir.exists():
            raise FileExistsError("run_dir already exists; choose a new run ID or explicit resume")
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        _json_new(run_dir / "run.json", {
            "created_at_utc": _utc(), "run_dir": str(run_dir),
            "config": asdict(config), "scientific_config_sha256": scientific_sha,
            "model_config": model.model_config(), "data_identity": identity,
            "environment": environment, "source_sha256": source_sha,
            "validation_enabled": validation_enabled, "scope": "preliminary_validation_pilot",
            "sampler": "uniform_train_rows; alternate_tail_head; typed_with_replacement; no_filter",
            "loss": "balanced_logistic", "selection": "max_validation_micro_MRR; earlier_step_on_tie"})

    started = time.perf_counter()
    history = (run_dir / "history.jsonl").open("a", encoding="utf-8")

    def log(event: Mapping[str, Any]) -> None:
        history.write(_canonical({"utc": _utc(), **event}) + "\n")
        history.flush()

    def save(kind: str) -> str:
        nonlocal last_checkpoint, best_checkpoint
        path = _unused_path(run_dir / "checkpoints", f"{kind}_step_{step:08d}", ".pt")
        if kind == "best":
            best_checkpoint = str(path)
        else:
            last_checkpoint = str(path)
        payload = {
            "format_version": 1, "created_at_utc": _utc(), "kind": kind, "step": step,
            "run_dir": str(run_dir), "model_config": model.model_config(),
            "config": asdict(config), "config_sha256": _sha(asdict(config)),
            "scientific_config_sha256": scientific_sha, "source_sha256": source_sha,
            "data_identity": identity, "environment": environment,
            "validation_enabled": validation_enabled,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "positive_rng": positive_rng.bit_generator.state,
            "negative_rng": negative_rng.bit_generator.state,
            "python_rng": random.getstate(), "numpy_global_rng": np.random.get_state(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else [],
            "best_mrr": best_mrr, "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint, "validation_history": validation_history,
            "first_batch_hashes": batch_hashes, "step_seconds": timings,
            "total_wall_seconds": previous_wall_seconds + time.perf_counter() - started}
        with path.open("xb") as handle:
            torch.save(payload, handle)
        log({"event": "checkpoint", "step": step, "kind": kind, "path": str(path)})
        return str(path)

    stop_reason = "completed_requested_steps"
    try:
        log({"event": "resume" if checkpoint is not None else "start", "step": step,
             "runtime_config": asdict(config)})
        model.train()
        bar = tqdm(total=config.steps, initial=step, desc="train", unit="step", disable=not progress)
        while step < config.steps:
            if time.perf_counter() - started >= config.max_minutes * 60:
                stop_reason = "wall_time_cap"
                break
            next_step = step + 1
            query, candidates, fingerprint = _sample_batch(data, config, next_step, positive_rng, negative_rng)
            _sync(model)
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_torch(query, candidates)
            loss = -0.5 * F.logsigmoid(scores[:, 0]).mean() - 0.5 * F.logsigmoid(-scores[:, 1:]).mean()
            if not bool(torch.isfinite(loss).item()):
                # Do not save an unusable resume stream after sampling a failed batch.
                raise FloatingPointError(f"Nonfinite loss at update {next_step}; use the previous checkpoint")
            loss.backward()
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                   for parameter in model.parameters()):
                raise FloatingPointError(f"Nonfinite gradient at update {next_step}; use the previous checkpoint")
            optimizer.step()
            _sync(model)
            timings.append(time.perf_counter() - began)
            step = next_step
            bar.update(1)
            if len(batch_hashes) < 10:
                batch_hashes.append({"step": step, "sha256": fingerprint, "direction": query["direction"]})
                log({"event": "batch_fingerprint", **batch_hashes[-1]})
            if step % config.log_every == 0 or step == 1 or step == config.steps:
                log({"event": "update", "step": step, "loss": float(loss.detach().cpu()),
                     "step_seconds": timings[-1]})
                bar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}", refresh=False)
            if evaluate_fn is not None and (step % config.validation_every == 0 or step == config.steps):
                model.eval()
                try:
                    with torch.no_grad():
                        metrics = dict(evaluate_fn(model, step))
                finally:
                    model.train()
                mrr = float(metrics["mrr"])
                if not math.isfinite(mrr) or not 0 <= mrr <= 1:
                    raise ValueError("Validation mrr must be finite and in [0,1]")
                record = {"step": step, **metrics}
                _canonical(record)  # Reject non-JSON/nonfinite callback metadata before save.
                validation_history.append(record)
                log({"event": "validation", **record})
                if best_mrr is None or mrr > best_mrr:
                    best_mrr = mrr
                    save("best")
            if step % config.checkpoint_every == 0:
                save("last")
        bar.close()
        # Keep the current snapshot even if the wall-time cap interrupted the pilot.
        save("last")
        elapsed = time.perf_counter() - started
        result = {
            "scope": "preliminary_validation_pilot", "completed_steps": step,
            "requested_steps": config.steps, "completed": step == config.steps,
            "stop_reason": stop_reason, "best_checkpoint": best_checkpoint,
            "last_checkpoint": last_checkpoint, "best_mrr": best_mrr,
            "validation_history": validation_history, "first_batch_hashes": batch_hashes,
            "step_sec": float(np.median(timings)) if timings else None,
            "step_sec_p90": float(np.quantile(timings, 0.9)) if timings else None,
            "measured_update_count": len(timings),
            "invocation_wall_seconds": elapsed, "total_wall_seconds": previous_wall_seconds + elapsed,
            "model_config": model.model_config(), "config": asdict(config),
            "data_identity": identity, "environment": environment,
            "note": "Pilot timing includes gradient-finite checks; no warmup exclusion; not a final benchmark."}
        manifest_path = _unused_path(run_dir, f"result_step_{step:08d}", ".json")
        result["result_manifest"] = str(manifest_path)
        _json_new(manifest_path, result)
        log({"event": "stop", "step": step, "reason": stop_reason, "result_manifest": str(manifest_path)})
        return result
    finally:
        history.close()


def model_gates() -> dict[str, Any]:
    """Future synthetic CPU gate; callable explicitly, never executed on import.

    Compares real-expanded ComplEx against direct complex arithmetic in BOTH
    directions, TransE against L1, and HeadBlind against fixed-ID permutation.
    These are synthetic checks, not ogbl-biokg measurements.
    """
    fixture = {"n_entities": 7, "n_relations": 3}
    candidates = np.array([[0, 2, 2], [1, 5, 6]], dtype=np.int64)
    query = {"relation": np.array([0, 2]), "fixed": np.array([3, 4]), "direction": 1}
    passed: dict[str, Any] = {"scope": "synthetic_model_gates"}
    for name in ("transe", "complex", "head_blind"):
        model = make_model(name, fixture, dim=8, seed=101, device="cpu")
        with torch.no_grad():
            for direction in (0, 1):
                q = {**query, "direction": direction}
                actual = model.score_torch(q, candidates)
                if name == "transe":
                    fixed = model.entity(model._index(q["fixed"]))[:, None, :]
                    entity = model.entity(model._index(candidates))
                    relation = model.relation(model._index(q["relation"]))[:, None, :]
                    head, tail = (fixed, entity) if direction else (entity, fixed)
                    expected = 12.0 - torch.linalg.vector_norm(head + relation - tail, ord=1, dim=-1)
                elif name == "complex":
                    def as_complex(tensor: torch.Tensor) -> torch.Tensor:
                        real, imaginary = tensor.chunk(2, -1)
                        return torch.complex(real, imaginary)
                    fixed = as_complex(model.entity(model._index(q["fixed"])))[:, None, :]
                    entity = as_complex(model.entity(model._index(candidates)))
                    relation = as_complex(model.relation(model._index(q["relation"])))[:, None, :]
                    head, tail = (fixed, entity) if direction else (entity, fixed)
                    expected = (head * relation * torch.conj(tail)).sum(-1).real
                else:
                    changed = {**q, "fixed": np.array([6, 0])}
                    expected = model.score_torch(changed, candidates)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                # Same IDs have same scores, regardless of column position/chunk.
                permutation = np.array([2, 0, 1])
                torch.testing.assert_close(model.score_torch(q, candidates[:, permutation]),
                                           actual[:, permutation], rtol=0, atol=0)
                chunks = torch.cat([model.score_torch(q, candidates[:, i:i + 1]) for i in range(3)], dim=1)
                torch.testing.assert_close(actual, chunks, rtol=2e-5, atol=2e-6)
            if name == "head_blind":
                # Construct a deterministic role difference, rather than relying on chance.
                model.relation_role[:, 0].zero_()
                model.relation_role[:, 1].fill_(1)
                model.entity.weight.fill_(1)
                head = model.score_torch({**query, "direction": 0}, candidates)
                tail = model.score_torch({**query, "direction": 1}, candidates)
                if torch.equal(head, tail):
                    raise AssertionError("HeadBlind must distinguish head and tail roles")
        passed[name] = "passed"
    return passed
