"""Push a trained adapter to a private Kaggle Dataset.

/kaggle/working is wiped when a session ends, so every training run must push
its adapter before the notebook closes. Run this as the last cell of any
training notebook.

    python scripts/push_artifacts.py --name qwen-v1 --path /kaggle/working/qwen-v1

First push creates the dataset; later pushes add a version. Re-attach it as
notebook input to load the adapter back on a later day.

Credentials are read from ~/.kaggle/kaggle.json or the KAGGLE_USERNAME /
KAGGLE_KEY environment variables. Never hardcode them, and never commit
kaggle.json.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

CRED_PATH = Path.home() / ".kaggle" / "kaggle.json"


def resolve_username():
    """Find the Kaggle username without ever printing the key."""
    if "KAGGLE_USERNAME" in os.environ:
        return os.environ["KAGGLE_USERNAME"]
    if CRED_PATH.exists():
        try:
            return json.loads(CRED_PATH.read_text())["username"]
        except (json.JSONDecodeError, KeyError):
            sys.exit(f"error: {CRED_PATH} is malformed — expected username and key fields")
    sys.exit(
        "error: no Kaggle credentials found.\n"
        "  In a Kaggle notebook: add them via Add-ons > Secrets.\n"
        "  Locally: place kaggle.json at ~/.kaggle/kaggle.json and chmod 600 it."
    )


def check_permissions():
    """Kaggle's CLI refuses world-readable credentials, and it is right to."""
    if CRED_PATH.exists():
        mode = CRED_PATH.stat().st_mode & 0o777
        if mode != 0o600:
            print(f"  fixing permissions on {CRED_PATH} ({oct(mode)} -> 0o600)")
            CRED_PATH.chmod(0o600)


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print("  " + result.stdout.strip().replace("\n", "\n  "))
    if result.returncode != 0:
        print("  " + result.stderr.strip().replace("\n", "\n  "), file=sys.stderr)
    return result.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="artifact name, e.g. qwen-v1")
    ap.add_argument("--path", required=True, help="directory holding the adapter")
    ap.add_argument("--note", default="", help="version note")
    ap.add_argument("--public", action="store_true", help="make public (default private)")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.is_dir():
        sys.exit(f"error: {src} is not a directory")

    files = [p for p in src.rglob("*") if p.is_file()]
    if not files:
        sys.exit(f"error: {src} is empty — nothing to push")

    size_mb = sum(p.stat().st_size for p in files) / 1024**2
    print(f"Pushing {args.name}")
    print(f"  source   {src}")
    print(f"  files    {len(files)}  ({size_mb:.1f} MB)")

    check_permissions()
    username = resolve_username()
    slug = f"{username}/healthcare-{args.name}"
    print(f"  dataset  {slug} ({'public' if args.public else 'private'})")

    meta = src / "dataset-metadata.json"
    meta.write_text(
        json.dumps(
            {
                "title": f"healthcare-{args.name}",
                "id": slug,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
    )

    # Kaggle has no "create or update" verb, so probe for existence first.
    exists = subprocess.run(
        ["kaggle", "datasets", "status", slug],
        capture_output=True,
        text=True,
    ).returncode == 0

    if exists:
        note = args.note or f"{args.name} {date.today().isoformat()}"
        code = run(["kaggle", "datasets", "version", "-p", str(src), "-m", note, "--dir-mode", "zip"])
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(src), "--dir-mode", "zip"]
        if not args.public:
            cmd.append("--private")
        code = run(cmd)

    if code != 0:
        print("\nPush FAILED. The adapter is still in /kaggle/working —")
        print("do not close the session until it is saved somewhere.")
        return 1

    print(f"\nPushed. Record in models/registry.md:  kaggle:{slug}")
    print(f"Re-attach on a later day: Add Input > Datasets > {slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
