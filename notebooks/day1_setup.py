"""Day 1 — Environment Setup + Data Collection (KAN-11)

Source for the Colab notebook. Build with:
    python scripts/build_notebook.py notebooks/day1_setup.py

Then open the .ipynb in Colab (or via the GitHub tab) and Run All.
Everything is automated; the only manual steps are authorising the Drive
mount and, for a private repo, pasting a GitHub token when prompted.
"""

# %% [markdown]
# # Day 1 — Environment Setup + Data Collection
#
# **KAN-11** · Healthcare lane · Qwen + Llama
#
# Runs end to end:
#
# 1. Mount Drive (everything persists there — `/content` vanishes on disconnect)
# 2. Clone the repo and install pinned dependencies
# 3. Verify GPU, pins, and 4-bit quantization
# 4. Download MedMCQA (Apache-2.0) and PubMedQA (MIT) straight to Drive
# 5. Checksum the raw data and print samples as evidence
#
# Datasets were chosen for licence as much as content. ChatDoctor (no licence)
# and MedQuAD (CC BY-SA share-alike) were excluded — see `data/raw/SOURCES.md`.
#
# **Runtime → Change runtime type → T4 GPU** before running.

# %%
# --- 1. Mount Drive ------------------------------------------------------
# First cell, every session. Colab can disconnect at any time and /content
# is wiped when it does; Drive is what survives.

from google.colab import drive

drive.mount("/content/drive")

from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/healthcare-llm")
DRIVE.mkdir(parents=True, exist_ok=True)
print(f"project root: {DRIVE}")

# %%
# --- 2. Repo + dependencies ---------------------------------------------
# The repo is private, so the clone needs a token. The token is fed to git
# over stdin via a credential helper — it never appears in a command line,
# in notebook output, or in the resulting .git/config.

import os
import subprocess
from getpass import getpass

REPO = "KusalPabasara/healthcare-llm-finetune"
PROJECT = Path("/content/healthcare-llm-finetune")


def sh(cmd, **kw):
    """Run a shell command and print its output. Never pass secrets through this.

    Subprocess output is captured and re-printed rather than inherited: under
    Colab a child process writing to the real stdout does not reach the cell,
    so a failing command would otherwise report only its exit code.
    """
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
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        # askpass reads the token from an env var the child process sees;
        # it is never part of the command git logs or stores.
        "GIT_ASKPASS": "/bin/echo",
        "GIT_USERNAME": token,
    }
    helper = f"!f() {{ echo username=x-access-token; echo password={token}; }}; f"
    return subprocess.run(
        [
            "git",
            "-c", f"credential.helper={helper}",
            "clone", "-q",
            f"https://github.com/{repo}.git",
            str(dest),
        ],
        env=env,
        capture_output=True,
        text=True,
    )


if PROJECT.exists():
    # Already cloned. Pull instead — otherwise a re-run of this cell keeps
    # running whatever code was cloned the first time, which is how you end
    # up debugging a bug that was fixed hours ago.
    os.chdir(PROJECT)
    before = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    pull = subprocess.run(
        ["git", "pull", "--ff-only", "-q"], capture_output=True, text=True
    )
    after = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    if pull.returncode != 0:
        print(f"  pull failed: {pull.stderr.strip()[:200]}")
        print("  continuing with the local copy — it may be out of date")
    elif before == after:
        print(f"  already up to date at {after}")
    else:
        print(f"  updated {before} -> {after}")
else:
    tok = getpass("GitHub token (input hidden): ").strip()
    print(f"$ git clone https://github.com/{REPO}.git  (token supplied via credential helper)")
    res = clone_private(REPO, PROJECT, tok)
    del tok
    if res.returncode != 0:
        # git's own stderr does not contain the token — safe to surface, and
        # far more useful than a guess at what went wrong.
        raise SystemExit(f"clone failed:\n{res.stderr.strip()}")
    print("cloned")
    os.chdir(PROJECT)

print(f"working in {Path.cwd()}")
print(subprocess.run(["git", "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout.strip())

# %%
# Colab ships older versions of most of these. Installing the pins keeps
# results reproducible across sessions and comparable across the team's lanes.
sh("pip install -q -r requirements.txt")

# %%
# --- 3a. Confirm what is on disk, then RESTART --------------------------
# requirements.txt pins bitsandbytes 0.46.1 for CUDA 12.8. Colab preloads an
# older build, so this reports what is on disk and hands over to a restart.
#
# Verification runs in the NEXT cell, after that restart.
#
# Why the split: pip writes the new version to disk, but the already-imported
# module stays in memory for the life of the interpreter. Verifying in the same
# cell as the install tests the old module and reports a failure that is
# already fixed on disk.

import importlib.metadata as md

for pkg in ("bitsandbytes", "transformers", "peft"):
    try:
        print(f"  {pkg:<16} {md.version(pkg)} on disk")
    except md.PackageNotFoundError:
        print(f"  {pkg:<16} not installed")

print(
    "\n"
    + "=" * 60
    + "\nNOW: Runtime > Restart session, then run the next cell.\n"
    "Skipping the restart makes the check test stale modules.\n"
    + "=" * 60
)

# %%
# --- 3b. Verify the environment (run AFTER restarting) ------------------
# A restart clears the working directory, unmounts Drive, and drops every
# import. This cell re-establishes all of it so it is safe to run on its own.

import os
import subprocess
from pathlib import Path

PROJECT = Path("/content/healthcare-llm-finetune")
DRIVE = Path("/content/drive/MyDrive/healthcare-llm")

if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive

    drive.mount("/content/drive")

if not PROJECT.exists():
    raise SystemExit("Repo is gone — re-run the clone cell (a full VM reset clears /content).")

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
        "\n"
        + "=" * 60
        + "\nVerification failed. Read the FAIL / MISS / DRIFT / ERROR line.\n\n"
        "  no CUDA device   -> Runtime > Change runtime type > T4 GPU\n"
        "  DRIFT on a pin   -> re-run the install cell, restart, retry\n"
        "  MISS a package   -> the install cell did not finish; re-run it\n"
        "  ERROR on import  -> installed but broken. For bitsandbytes this\n"
        "                      means no CUDA binary for this torch build:\n"
        "                      pip install -U 'bitsandbytes>=0.46.1'\n"
        "  4-bit matmul     -> same fix as above, then restart\n"
        + "=" * 60
    )
    raise SystemExit("Environment verification failed — see above.")

print("\n✓ Environment verified.")

# %%
# --- 4. Download the datasets -------------------------------------------
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
# --- 5. Inspect what landed ---------------------------------------------
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
# --- 6. Confirm persistence ---------------------------------------------
# The whole point of writing to Drive. If this shows files, Day 2 can attach
# them without re-downloading.

sh(f"du -sh {RAW}/* 2>/dev/null")
sh(f"python scripts/download_data.py --verify")

# %%
# --- Day 1 complete ------------------------------------------------------
print(f"""
Day 1 done. Record in PROGRESS.md frozen decisions:
  - GPU type and compute dtype (from verify_env.py above)
  - Dataset row counts and licences ({RAW}/SOURCES.md)
  - Raw data checksums ({RAW}/CHECKSUMS.txt)

Data is on Drive at {RAW} and survives disconnects.

Next: Day 2 (KAN-15) — cleaning, instruction formatting, 80/10/10 split.
""")
