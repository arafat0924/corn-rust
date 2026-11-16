r"""Convert a "# %%" percent-format .py file into a Jupyter notebook.

Run: .\.venv\Scripts\python.exe scripts\py_to_ipynb.py notebooks\train_corn_rust.py
Writes the .ipynb next to the .py. Cells start with "# %%"; "# %% [markdown]" cells have
their leading "# " stripped.
"""
import json
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    cells, cur, kind = [], [], None

    def flush():
        if kind is None:
            return
        lines = cur[:]
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        src_lines = [l + "\n" for l in lines]
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "id": f"cell-{len(cells):03d}", "metadata": {}, "source": src_lines}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            cur, kind = [], "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    flush()

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    print("wrote", convert(Path(sys.argv[1])))
