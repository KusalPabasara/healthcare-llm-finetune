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


if not PROJECT.exists():
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

# %%
# Colab ships older versions of most of these. Installing the pins keeps
# results reproducible across sessions and comparable across the team's lanes.
sh("pip install -q -r requirements.txt")

# %%
# --- 3. Verify the environment ------------------------------------------
# Fails loudly on version drift, missing GPU, or a broken 4-bit config.
# Do not proceed to Day 2 on a failure here.
#
# Note: pip may warn about needing a restart. If verify_env.py reports version
# drift, use Runtime > Restart session, then re-run from this cell.

result = sh("python scripts/verify_env.py")
if result.returncode != 0:
    print(
        "\n"
        + "=" * 60
        + "\nVerification failed. Read the FAIL / MISS / DRIFT lines above.\n\n"
        "  no CUDA device  -> Runtime > Change runtime type > T4 GPU\n"
        "  DRIFT on a pin  -> Runtime > Restart session, re-run from the\n"
        "                     pip install cell (new versions need a fresh\n"
        "                     interpreter)\n"
        "  MISS a package  -> the pip install cell did not finish; re-run it\n"
        + "=" * 60
    )
    raise SystemExit("Environment verification failed — see above.")

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
