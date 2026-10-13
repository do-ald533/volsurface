"""The committed notebook's code cells run and produce sensible numbers."""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "monte_carlo.ipynb"


def test_notebook_code_cells_execute() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    namespace: dict[str, object] = {}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    smile = namespace["smile"]
    assert len(smile) == 9
    assert smile[0] > smile[-1]
