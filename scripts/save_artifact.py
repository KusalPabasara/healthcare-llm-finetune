"""Save a trained adapter to Drive and register it.

Colab's /content is wiped on disconnect. This copies a finished adapter to the
Drive project directory and appends a row to the model registry, so an artifact
is never both finished and unrecorded.

    python scripts/save_artifact.py --name qwen-v1 --path /content/out/qwen-v1

Checkpointing during training is separate — set output_dir to Drive directly so
a mid-run disconnect costs minutes, not the whole run. This script is for the
finished adapter.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/healthcare-llm")


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="artifact name, e.g. qwen-v1")
    ap.add_argument("--path", required=True, help="directory holding the adapter")
    ap.add_argument("--train-time", help="wall-clock training time, e.g. 2h14m")
    ap.add_argument("--note", default="", help="anything worth recording")
    ap.add_argument("--drive", type=Path, default=DRIVE, help="override project root")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.is_dir():
        sys.exit(f"error: {src} is not a directory")

    files = [p for p in src.rglob("*") if p.is_file()]
    if not files:
        sys.exit(f"error: {src} is empty — nothing to save")

    # A LoRA adapter without its config cannot be loaded back. Catch that here
    # rather than on Day 11 when all eight are needed at once.
    names = {p.name for p in files}
    if "adapter_config.json" not in names:
        print("  WARN  no adapter_config.json — is this a PEFT adapter directory?")

    if not args.drive.parent.exists():
        sys.exit(
            f"error: {args.drive.parent} does not exist.\n"
            "  Drive is not mounted. Run:\n"
            "  from google.colab import drive; drive.mount('/content/drive')"
        )

    dest = args.drive / "models" / args.name
    if dest.exists():
        print(f"  {dest} exists — replacing")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    size = dir_size_mb(src)
    print(f"saving {args.name}")
    print(f"  from   {src}")
    print(f"  to     {dest}")
    print(f"  files  {len(files)}  ({size:.1f} MB)")

    shutil.copytree(src, dest)

    # Verify the copy landed — a truncated Drive write is silent otherwise.
    copied = dir_size_mb(dest)
    if abs(copied - size) > 0.5:
        sys.exit(f"error: size mismatch after copy ({size:.1f} -> {copied:.1f} MB)")
    print(f"  verified {copied:.1f} MB on Drive")

    record = {
        "artifact": args.name,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "size_mb": round(copied, 1),
        "path": str(dest),
        "train_time": args.train_time or "",
        "note": args.note,
    }
    ledger = args.drive / "models" / "artifacts.jsonl"
    with ledger.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    print(f"\nrecorded in {ledger}")
    print(f"Add to models/registry.md:  {args.name}  {copied:.0f}MB  {args.train_time or '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
