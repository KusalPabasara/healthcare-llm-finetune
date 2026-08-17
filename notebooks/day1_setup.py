"""Day 1 — Environment Setup + Data Collection (KAN-11)

Source for the Colab notebook. Build with:
    python scripts/build_notebook.py notebooks/day1_setup.py

Safe to re-run from the top in any session. Cell 0 reports what is already
done and what is left, so a recycled VM or a closed tab never means guessing
where you stopped.
"""

# %% [markdown]
# # Day 1 — Environment Setup + Data Collection
#
# **KAN-11** · Healthcare lane · Qwen + Llama
#
# **Run cell 0 first — it tells you which cells you still need.**
#
# Colab recycles VMs between sessions. Drive data survives; installed packages
# do not. So on a fresh VM you re-run the setup cells but never re-download the
# datasets.
#
# | Cell | Does | Needed when |
# |---|---|---|
# | 0 | Reports status | Always — start here |
# | 1 | Mount Drive | Every session |
# | 2 | Clone/pull repo | Every session |
# | 3 | Install pins | Every fresh VM |
# | 4 | Report + restart | After an install |
# | 5 | Verify | After the restart |
# | 6–8 | Download data | Once, ever |
# | 9 | Summary | When finished |
#
# Set **Runtime → Change runtime type → T4 GPU** before starting.

# %%
# --- 0. Where did I stop? -----------------------------------------------
# Run this first in every session. It inspects the actual environment rather
# than relying on you to remember, and prints the cells still outstanding.

import importlib.metadata as md
import os
import subprocess
from pathlib import Path

PROJECT = Path("/content/healthcare-llm-finetune")
DRIVE = Path("/content/drive/MyDrive/healthcare-llm")
RAW = DRIVE / "data" / "raw"

CRITICAL = ("transformers", "datasets", "accelerate", "peft", "bitsandbytes", "trl")


def load_pins(req: Path) -> dict:
    """Read pins from requirements.txt — never hardcode them here.

    A second copy of the version numbers drifts the moment a pin changes,
    and then this cell reports a mismatch against a version nobody uses.
    """
    pins = {}
    if not req.exists():
        return pins
    for line in req.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        if name.strip().lower() in CRITICAL:
            pins[name.strip().lower()] = version.strip()
    return pins


PINS = load_pins(PROJECT / "requirements.txt")

print("Day 1 status\n" + "=" * 56)

drive_ok = Path("/content/drive/MyDrive").exists()
print(f"  {'ok ' if drive_ok else '-- '} Drive mounted")

repo_ok = PROJECT.exists()
print(f"  {'ok ' if repo_ok else '-- '} Repo cloned")

if not PINS:
    # No repo yet, so requirements.txt is unreadable. Cell 2 fixes that.
    pins_ok, wrong = False, []
    print("  --  Pinned packages  (repo not cloned yet — run cell 2)")
else:
    pins_ok, wrong = True, []
    for pkg, want in PINS.items():
        try:
            if md.version(pkg) != want:
                pins_ok = False
                wrong.append(pkg)
        except md.PackageNotFoundError:
            pins_ok = False
            wrong.append(pkg)
    print(f"  {'ok ' if pins_ok else '-- '} Pinned packages" +
          ("" if pins_ok else f"  ({len(wrong)} wrong/missing: {', '.join(wrong)})"))

gpu_ok = False
try:
    import torch

    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "none"
except ImportError:
    gpu_name = "torch not installed"
print(f"  {'ok ' if gpu_ok else '-- '} GPU  ({gpu_name})")

data_ok = drive_ok and (RAW / "medmcqa").exists() and (RAW / "pubmedqa").exists()
print(f"  {'ok ' if data_ok else '-- '} Datasets on Drive")

print("=" * 56)

todo = []
if not drive_ok:
    todo.append("1  mount Drive")
if not repo_ok:
    todo.append("2  clone the repo")
elif drive_ok:
    todo.append("2  pull latest (fast, always worth it)")
if not pins_ok:
    todo.append("3  install pins, then 4, then RESTART, then 5")
elif not gpu_ok:
    todo.append("5  verify (GPU not detected — check Runtime type)")
else:
    todo.append("5  verify")
if not data_ok:
    todo.append("6-8  download datasets")

if data_ok and pins_ok and gpu_ok:
    print("\nEverything is in place. Run cell 5 to confirm, then cell 9.")
else:
    print("\nRun these cells:")
    for t in todo:
        print(f"    {t}")
    if data_ok:
        print("\n  Skip 6-8 — the datasets are already on Drive.")

# %%
# --- 1. Mount Drive ------------------------------------------------------
# Checks before mounting: an unconditional mount() re-prompts for
# authorisation every session even when Drive is already attached.

from pathlib import Path

if Path("/content/drive/MyDrive").exists():
    print("Drive already mounted — no authorisation needed.")
else:
    from google.colab import drive

    drive.mount("/content/drive")

DRIVE = Path("/content/drive/MyDrive/healthcare-llm")
DRIVE.mkdir(parents=True, exist_ok=True)
print(f"project root: {DRIVE}")

# %%
# --- 2. Repo: clone or pull ---------------------------------------------
# Pulls when the repo exists, so a re-run picks up fixes instead of silently
# keeping whatever was cloned first.

import os
import subprocess
from getpass import getpass
from pathlib import Path

REPO = "KusalPabasara/healthcare-llm-finetune"
PROJECT = Path("/content/healthcare-llm-finetune")


def sh(cmd, **kw):
    """Run a shell command and print its output. Never pass secrets through."""
    print(f"$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, check=False, capture_output=True, text=True, **kw
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result


def clone_private(repo: str, dest: Path, token: str):
    """Clone without the token touching argv, output, or on-disk config."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    helper = f"!f() {{ echo username=x-access-token; echo password={token}; }}; f"
    return subprocess.run(
        ["git", "-c", f"credential.helper={helper}", "clone", "-q",
         f"https://github.com/{repo}.git", str(dest)],
        env=env, capture_output=True, text=True,
    )


if PROJECT.exists():
    os.chdir(PROJECT)
    before = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    pull = subprocess.run(["git", "pull", "--ff-only", "-q"],
                          capture_output=True, text=True)
    after = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
    if pull.returncode != 0:
        print(f"  pull failed: {pull.stderr.strip()[:200]}")
        print("  continuing with the local copy — it may be out of date")
    elif before == after:
        print(f"  already up to date at {after}")
    else:
        print(f"  updated {before} -> {after}")
else:
    tok = getpass("GitHub token (input hidden): ").strip()
    print(f"$ git clone https://github.com/{REPO}.git  (token via credential helper)")
    res = clone_private(REPO, PROJECT, tok)
    del tok
    if res.returncode != 0:
        raise SystemExit(f"clone failed:\n{res.stderr.strip()}")
    print("cloned")
    os.chdir(PROJECT)

print(f"working in {Path.cwd()}")
print(subprocess.run(["git", "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout.strip())

# %%
# --- 3. Install the pins ------------------------------------------------
# Runs in EVERY session on a fresh VM: Colab restores its stock package set,
# so yesterday's pins are gone even though Drive data survives.
#
# Not quiet — -q hides resolver failures, and a partial install that looks
# successful is worse than a loud one that fails.

res = sh("pip install -r requirements.txt 2>&1 | tail -25")

# pip can exit 0 having skipped packages, so check what actually landed.
# Expected versions come from requirements.txt — the same file just installed,
# so this can never disagree with what was asked for.
import importlib.metadata as md

CRITICAL = ("transformers", "datasets", "accelerate", "peft", "bitsandbytes", "trl")
EXPECTED = {}
for line in (PROJECT / "requirements.txt").read_text().splitlines():
    line = line.split("#")[0].strip()
    if "==" in line:
        name, _, version = line.partition("==")
        if name.strip().lower() in CRITICAL:
            EXPECTED[name.strip().lower()] = version.strip()

print("\non disk after install:")
missing = []
for pkg, want in EXPECTED.items():
    try:
        got = md.version(pkg)
        mark = "ok " if got == want else "!! "
        print(f"  {mark} {pkg:<16} {got}" + ("" if got == want else f"  (want {want})"))
        if got != want:
            missing.append(pkg)
    except md.PackageNotFoundError:
        print(f"  !!  {pkg:<16} NOT INSTALLED")
        missing.append(pkg)

if missing:
    print(f"\nInstall did not complete for: {', '.join(missing)}")
    print("Re-run this cell. If it keeps failing, read the pip output above.")
else:
    print("\nAll pins on disk. Next: run cell 4, then RESTART.")

# %%
# --- 4. Confirm on disk, then RESTART -----------------------------------
# pip writes the new version to disk, but the already-imported module stays
# in memory for the life of the interpreter. Verifying in the same cell as
# the install tests the old module and reports a failure already fixed.

import importlib.metadata as md

for pkg in ("bitsandbytes", "transformers", "peft", "trl"):
    try:
        print(f"  {pkg:<16} {md.version(pkg)} on disk")
    except md.PackageNotFoundError:
        print(f"  {pkg:<16} not installed")

print(
    "\n" + "=" * 60
    + "\nNOW: Runtime > Restart session, then run cell 5.\n"
    "Skipping the restart makes the check test stale modules.\n"
    + "=" * 60
)

# %%
# --- 5. Verify (run AFTER restarting) -----------------------------------
# A restart clears the working directory, unmounts nothing but drops every
# import. This cell re-establishes what it needs so it is safe standalone.

import os
import subprocess
from pathlib import Path

PROJECT = Path("/content/healthcare-llm-finetune")
DRIVE = Path("/content/drive/MyDrive/healthcare-llm")

if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive

    drive.mount("/content/drive")

if not PROJECT.exists():
    raise SystemExit("Repo is gone — run cell 2 (a VM reset clears /content).")

os.chdir(PROJECT)

# data/raw symlinks to Drive; re-point it if the restart dropped it.
raw = PROJECT / "data" / "raw"
if not raw.exists():
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.symlink_to(DRIVE / "data" / "raw")


def sh(cmd, **kw):
    """Redefined here: a restart clears every earlier definition."""
    print(f"$ {cmd}")
    result = subprocess.run(
        cmd, shell=True, check=False, capture_output=True, text=True, **kw
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    return result


print(f"cwd: {Path.cwd()}\n")
result = sh("python scripts/verify_env.py")

if result.returncode != 0:
    print(
        "\n" + "=" * 60
        + "\nVerification failed. Read the FAIL / MISS / DRIFT / ERROR line.\n\n"
        "  fresh VM         -> run cell 3, then 4, restart, then 5 again\n"
        "  no CUDA device   -> Runtime > Change runtime type > T4 GPU\n"
        "  ERROR on import  -> installed but broken. For bitsandbytes that\n"
        "                      means no CUDA binary for this torch build:\n"
        "                      pip install -U 'bitsandbytes>=0.46.1'\n"
        + "=" * 60
    )
    raise SystemExit("Environment verification failed — see above.")

print("\n✓ Environment verified.")

# %%
# --- 6. Download the datasets (once, ever) ------------------------------
# Skip if cell 0 said the datasets are already on Drive.
#
# Written straight to Drive so a disconnect does not cost the 140MB download.
# data/raw/ is the reproducibility anchor: read-only for the rest of the sprint.

RAW = DRIVE / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Point the repo's data/raw at Drive, so scripts stay path-agnostic.
local_raw = PROJECT / "data" / "raw"
if local_raw.is_symlink():
    local_raw.unlink()
elif local_raw.exists():
    import shutil

    shutil.rmtree(local_raw)
local_raw.parent.mkdir(parents=True, exist_ok=True)
local_raw.symlink_to(RAW)
print(f"data/raw -> {RAW}")

sh("python scripts/download_data.py")

# %%
# --- 7. Inspect what landed ---------------------------------------------
# A sample row from each dataset — evidence the data is real and correctly
# shaped, rather than an empty success message.

import json

for name in ("medmcqa", "pubmedqa"):
    path = RAW / name / "train.jsonl"
    if not path.exists():
        print(f"{name}: MISSING")
        continue
    with path.open() as fh:
        first = json.loads(fh.readline())
        rows = 1 + sum(1 for _ in fh)
    print(f"\n{'=' * 60}\n{name}  —  {rows:,} rows\n{'=' * 60}")
    for k, v in first.items():
        s = str(v).replace("\n", " ")
        print(f"  {k:<16} {s[:90]}{'...' if len(s) > 90 else ''}")

# %%
# --- 8. Confirm persistence ---------------------------------------------
# The whole point of writing to Drive. If this shows files, a later session
# attaches them without re-downloading.

sh(f"du -sh {RAW}/* 2>/dev/null")
sh("python scripts/download_data.py --verify")

# %%
# --- 9. Day 1 complete ---------------------------------------------------
print(f"""
Day 1 done. Record in PROGRESS.md frozen decisions:
  - GPU type and compute dtype (from cell 5)
  - Dataset row counts and licences ({RAW}/SOURCES.md)
  - Raw data checksums ({RAW}/CHECKSUMS.txt)

Data is on Drive at {RAW} and survives VM resets.

Next: Day 2 (KAN-15) — cleaning, instruction formatting, 80/10/10 split.
Start that session by running cell 0 to see what needs re-doing.
""")
