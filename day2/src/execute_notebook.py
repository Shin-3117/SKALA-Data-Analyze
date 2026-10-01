"""Execute and save the notebook with the invoking Python as its kernel."""
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "day2/day2.ipynb"
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="day2-jupyter-") as temporary:
        directory = Path(temporary)
        kernel = directory / "kernels/day2-ess"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "DAY 2 ESS", "language": "python"}))
        os.environ["JUPYTER_PATH"] = str(directory)
        os.environ["JUPYTER_RUNTIME_DIR"] = str(directory / "runtime")
        os.environ["IPYTHONDIR"] = str(directory / "ipython")

        def starting(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Executing code cell {cell_index}", flush=True)

        client = NotebookClient(notebook, kernel_name="day2-ess", timeout=600,
                                resources={"metadata": {"path": str(root / "day2")}},
                                on_cell_start=starting)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
    print("Notebook execution complete", flush=True)
    print((root / "day2/output/model_performance.csv").read_text(), flush=True)


if __name__ == "__main__":
    main()
