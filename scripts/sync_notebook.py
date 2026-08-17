"""Build a notebook from its .py source and push it to Kaggle.

    python scripts/sync_notebook.py notebooks/day1_setup.py

Notebooks live in git as `# %%`-delimited Python — reviewable, diffable. This
converts one to .ipynb and pushes it as a Kaggle kernel, creating it on first
run and versioning it after.

The pushed notebook is private, GPU-enabled, and has internet on (needed to
clone the repo and pull datasets).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_notebook import build  # noqa: E402

CRED_PATH = Path.home() / ".kaggle" / "kaggle.json"


def resolve_username() -> str:
    if "KAGGLE_USERNAME" in os.environ:
        return os.environ["KAGGLE_USERNAME"]
    if CRED_PATH.exists():
        try:
            return json.loads(CRED_PATH.read_text())["username"]
        except (json.JSONDecodeError, KeyError):
            sys.exit(f"error: {CRED_PATH} is malformed")
    sys.exit(
        "error: no Kaggle credentials.\n"
        "  Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600),\n"
        "  or export KAGGLE_USERNAME and KAGGLE_KEY."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="the .py notebook source")
    ap.add_argument("--slug", help="kernel slug (default: derived from filename)")
    ap.add_argument("--title", help="notebook title (default: derived from filename)")
    ap.add_argument("--no-gpu", action="store_true", help="disable the GPU")
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"error: {args.source} not found")

    username = resolve_username()
    stem = args.source.stem.replace("_", "-")
    slug = args.slug or f"healthcare-{stem}"
    # "day1_setup" -> "Healthcare Day 1 Setup" rather than the bare "Day1 Setup"
    words = args.source.stem.replace("_", " ").replace("day", "day ").split()
    title = args.title or "Healthcare " + " ".join(w.capitalize() for w in words)

    # Kaggle's CLI pushes a directory containing the notebook plus metadata.
    staging = Path(".kaggle-push") / slug
    staging.mkdir(parents=True, exist_ok=True)

    nb_path = staging / f"{slug}.ipynb"
    nb_path.write_text(json.dumps(build(args.source), indent=1))

    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": nb_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not args.no_gpu,
        # The notebook clones the repo and downloads datasets, both of which
        # need network access. Kaggle disables it by default.
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"pushing {username}/{slug}")
    print(f"  title    {title}")
    print(f"  gpu      {'on' if not args.no_gpu else 'off'}, internet on, private")

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(staging)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print(f"\nnotebook built at {nb_path}")
        sys.exit(
            "error: the kaggle CLI is not installed.\n"
            "  pip install kaggle\n"
            "  Or upload the .ipynb above through the Kaggle web UI."
        )

    out = (result.stdout + result.stderr).strip()
    print("  " + out.replace("\n", "\n  "))

    if result.returncode != 0:
        print("\npush failed", file=sys.stderr)
        return 1

    print(f"\nhttps://www.kaggle.com/code/{username}/{slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
