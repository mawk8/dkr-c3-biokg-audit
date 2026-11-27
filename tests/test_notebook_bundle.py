"""The standalone Kaggle notebook must execute the reviewed pilot modules."""
import ast
import hashlib
import json
from pathlib import Path


def test_kaggle_source_matches_package():
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks/C3_Kaggle_Baselines_and_Pilot_EN.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code"]
    for code in cells:
        compile(code, notebook.name, "exec")
    bundle_cell, = (code for code in cells if code.startswith("SOURCE_BUNDLE = "))
    literals = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(bundle_cell).body if isinstance(node, ast.Assign)
    }
    bundle = literals["SOURCE_BUNDLE"]
    actual = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in bundle.items()}
    assert actual == literals["SOURCE_HASHES"] == document["metadata"]["source_sha256"]
    for name, code in bundle.items():
        assert code == (root / "src/biokg_audit" / name).read_text(encoding="utf-8"), name
