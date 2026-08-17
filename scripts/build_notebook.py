"""Convert a `# %%` cell-delimited Python file into Kaggle notebook JSON.

Keeps notebooks reviewable as plain Python in git — a .py file diffs cleanly,
a .ipynb does not.

    python scripts/build_notebook.py notebooks/day1_setup.py

Writes the .ipynb beside the source. Pass --stdout to print the JSON instead,
which is what the Kaggle MCP save_notebook tool wants.
"""

import argparse
import json
import sys
from pathlib import Path


def split_cells(text: str):
    """Split on `# %%` markers. `# %% [markdown]` becomes a markdown cell."""
    cells = []
    kind, buf = "code", []

    for line in text.splitlines():
        if line.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    if buf:
        cells.append((kind, buf))

    return cells


def clean(kind: str, lines: list[str]) -> list[str]:
    """Trim blank edges; strip leading '# ' from markdown."""
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if kind == "markdown":
        lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
    return lines


def build(src: Path) -> dict:
    text = src.read_text()

    # Drop the module docstring — it documents the source file, not the notebook.
    if text.lstrip().startswith('"""'):
        body = text.lstrip()[3:]
        end = body.find('"""')
        if end != -1:
            text = body[end + 3:]

    cells = []
    for i, (kind, lines) in enumerate(split_cells(text)):
        lines = clean(kind, lines)
        if not lines:
            continue
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        # Stable ids derived from position — nbformat requires them, and keeping
        # them deterministic means re-builds produce clean diffs.
        cell = {
            "cell_type": kind,
            "id": f"cell-{i:02d}",
            "metadata": {},
            "source": source,
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--stdout", action="store_true", help="print JSON instead of writing")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"error: {args.source} not found")

    nb = build(args.source)

    if args.stdout:
        print(json.dumps(nb, indent=1))
        return 0

    out = args.source.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1))
    counts = {"code": 0, "markdown": 0}
    for c in nb["cells"]:
        counts[c["cell_type"]] += 1
    print(f"{out}  —  {counts['code']} code, {counts['markdown']} markdown cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
