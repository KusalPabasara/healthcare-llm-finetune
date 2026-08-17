# Kaggle scripts (not in use)

Kept from the Day 1 evaluation of Kaggle as the training platform. **Kaggle requires
phone verification for any GPU access**, which is not available on this account, so
training runs on Colab instead — see `docs/platform.md`.

These still work and are worth keeping: if phone verification ever becomes possible,
Kaggle offers 9h sessions against Colab's ~4h and a visible 30 GPU-h/week quota.

| Script | What it does |
|---|---|
| `push_artifacts.py` | Push a trained adapter to a private Kaggle Dataset |
| `sync_notebook.py` | Build a notebook from `.py` source and push it as a Kaggle kernel |

Both need `pip install kaggle` and credentials at `~/.kaggle/kaggle.json`.

Note: Kaggle datasets can still be *read* from Colab without phone verification —
only GPU access is gated. If a Kaggle-hosted dataset is needed later, the API works
fine from a Colab notebook.
