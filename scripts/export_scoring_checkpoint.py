"""Export a trusted training snapshot for validation-only audit scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biokg_audit.data import _no_links  # noqa: E402
from biokg_audit.training import export_scoring_checkpoint  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="trusted project-created training checkpoint")
    parser.add_argument("output", type=Path, help="new scoring export .pt file")
    parser.add_argument("--original-run", type=Path,
                        help="run folder holding original entity_mapping.json and dataset_manifest.json")
    parser.add_argument("--completed-checkpoint", type=Path,
                        help="trusted last training checkpoint from the same completed best-checkpoint run")
    args = parser.parse_args(argv)
    data = None
    if args.original_run is not None:
        mapping = _no_links(args.original_run / "entity_mapping.json")
        manifest = _no_links(args.original_run / "dataset_manifest.json")
        data = {
            "mapping": json.loads(mapping.read_text(encoding="utf-8")),
            "manifest": json.loads(manifest.read_text(encoding="utf-8")),
        }
    completed = _no_links(args.completed_checkpoint) if args.completed_checkpoint is not None else None
    metadata = export_scoring_checkpoint(_no_links(args.checkpoint), _no_links(args.output), data=data,
                                         completed_checkpoint_path=completed)
    print(f"Exported {metadata['kind']} checkpoint at step {metadata['step']} to {args.output}")


if __name__ == "__main__":
    main()
