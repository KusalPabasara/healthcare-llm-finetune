"""Day 1 — Environment Setup + Data Collection (KAN-11)

Source for the Kaggle notebook. Pushed via scripts/sync_notebook.py.
Run top to bottom; no manual steps beyond attaching the repo and pressing Run.

Requires: GitHub repo attached as a Kaggle dataset/utility script, or the repo
cloned in the first cell. Secrets KAGGLE_USERNAME and KAGGLE_KEY set via
Add-ons > Secrets for the artifact push.
"""

# %% [markdown]
# # Day 1 — Environment Setup + Data Collection
#
# **KAN-11** · Healthcare lane · Qwen + Llama
#
# Runs end to end without intervention:
#
# 1. Clone the repo and install pinned dependencies
# 2. Verify GPU, pins, and 4-bit quantization
# 3. Download MedMCQA (Apache-2.0) and PubMedQA (MIT)
# 4. Checksum the raw data
# 5. Push everything to a private Kaggle Dataset — `/kaggle/working` is wiped at session end
#
# Datasets were chosen for licence as much as content. ChatDoctor (no licence)
# and MedQuAD (CC BY-SA share-alike) were excluded — see `data/raw/SOURCES.md`.

# %%
# --- 1. Repo + dependencies ---------------------------------------------
# The repo is private, so cloning needs a token. Set GITHUB_TOKEN in
# Add-ons > Secrets, or attach the repo through Kaggle's GitHub import.

import os
import subprocess
from pathlib import Path

REPO = "KusalPabasara/healthcare-llm-finetune"
WORK = Path("/kaggle/working")
PROJECT = WORK / "healthcare-llm-finetune"


def sh(cmd, **kw):
    """Run a shell command, streaming output."""
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False, **kw)


if not PROJECT.exists():
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        sh(f"git clone -q https://{token}@github.com/{REPO}.git {PROJECT}")
    except Exception as exc:  # noqa: BLE001
        print(f"Secret GITHUB_TOKEN unavailable ({type(exc).__name__}).")
        print("Falling back to public clone — will fail if the repo is private.")
        sh(f"git clone -q https://github.com/{REPO}.git {PROJECT}")

os.chdir(PROJECT)
print(f"\nworking in {Path.cwd()}")

# %%
# Kaggle images ship most of this already; installing pinned versions keeps
# results reproducible across sessions and across the team's lanes.
sh("pip install -q -r requirements.txt")

# %%
# --- 2. Verify the environment ------------------------------------------
# Fails loudly on version drift, missing GPU, or a broken 4-bit config.
# Do not proceed to Day 2 on a failure here.

result = sh("python scripts/verify_env.py")
if result.returncode != 0:
    raise SystemExit("Environment verification failed — fix before continuing.")

# %%
# --- 3. Download the datasets -------------------------------------------
# Writes data/raw/ (read-only for the rest of the sprint), regenerates
# SOURCES.md with provenance and licences, and writes CHECKSUMS.txt.

sh("python scripts/download_data.py")

# %%
# --- 4. Inspect what landed ---------------------------------------------
# A sample from each dataset, so the notebook output is evidence the data
# is real and correctly shaped rather than an empty success message.

import json

for name in ("medmcqa", "pubmedqa"):
    path = Path("data/raw") / name / "train.jsonl"
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
# --- 5. Push to a private Kaggle Dataset --------------------------------
# /kaggle/working is wiped when the session ends. This is the step that makes
# Day 1 durable — without it the download must be repeated tomorrow.

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")

sh("python scripts/push_artifacts.py --name raw-data --path data/raw "
   "--note 'Day 1 raw datasets: medmcqa + pubmedqa'")

# %%
# --- Day 1 complete ------------------------------------------------------
print("""
Day 1 done. Recorded for PROGRESS.md frozen decisions:
  - GPU type and compute dtype (from verify_env.py above)
  - Dataset row counts and licences (data/raw/SOURCES.md)
  - Raw data checksums (data/raw/CHECKSUMS.txt)

Next: Day 2 (KAN-15) — cleaning, instruction formatting, 80/10/10 split.
Attach dataset `healthcare-raw-data` as input rather than re-downloading.
""")
