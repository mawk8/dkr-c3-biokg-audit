"""Small cloud workflow utilities; no work is performed at import time."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import time
import zipfile


def environment_report() -> dict:
    import torch
    versions = {}
    for package in ('torch', 'numpy', 'ogb', 'pandas', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  packages=versions, cuda_available=torch.cuda.is_available(),
                  torch_cuda=torch.version.cuda, devices=[])
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result['devices'].append(dict(index=index, name=properties.name,
                                          total_memory_bytes=properties.total_memory))
    return result


def new_run(workspace: Path, config: dict, stage: str) -> Path:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(workspace)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    run = workspace/'runs'/f'{stamp}_{time.time_ns()}_{stage}_{digest}'
    run.mkdir(parents=True, exist_ok=False)
    save_json(run/'config.json', config)
    save_json(run/'environment.json', environment_report())
    save_json(run/'disk_before.json', dict(free_bytes=usage.free, total_bytes=usage.total))
    return run


def save_json(path: Path, value) -> None:
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_comparison(run_dir: Path, results: list[dict]) -> Path:
    if not results:
        raise ValueError('No evaluation results to compare')
    row_hashes = {x['validation_rows_sha256'] for x in results}
    candidate_hashes = {x['candidate_sha256'] for x in results}
    if len(row_hashes) != 1 or len(candidate_hashes) != 1:
        raise ValueError('Scorers used different questions or candidates')
    path = Path(run_dir)/f'comparison_{time.time_ns()}.csv'
    columns = ['model', 'protocol', 'training_seed', 'n_triples', 'n_queries', 'mrr',
               'hits1', 'hits3', 'hits10', 'macro_relation_role_mrr',
               'evaluation_seconds', 'stage', 'split']
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    return path


def export_run(run_dir: Path) -> Path:
    """Zip just this run, retaining all originals. No dataset or private paths."""
    run_dir = Path(run_dir).resolve()
    archive = run_dir.parent/f'{run_dir.name}_export_{time.time_ns()}.zip'
    with zipfile.ZipFile(archive, mode='x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1) as target:
        # All descendants were produced by this project. Do not follow links.
        def add_directory(folder):
            for path in sorted(folder.iterdir()):
                if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                    raise ValueError(f'Linked path in run output: {path.name}')
                if not path.resolve().is_relative_to(run_dir):
                    raise ValueError('Output escapes run directory')
                if path.is_dir():
                    add_directory(path)
                elif path.is_file():
                    target.write(path, path.relative_to(run_dir.parent))
        add_directory(run_dir)
    return archive
