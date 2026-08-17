# Training Platform

Training runs on **Google Colab** (free tier).

## Why not Kaggle

Kaggle was evaluated on Day 1 and rejected: **any GPU access requires phone
verification**, which is not available on this account. Without it Kaggle notebooks
are CPU-only, and QLoRA fine-tuning on CPU is not viable — days per run instead of
hours.

Colab's free tier gives a T4 with no verification, so it is the working platform.

If phone verification ever becomes possible, Kaggle is worth revisiting: 9h sessions
against Colab's ~4h, and a visible 30 GPU-h/week quota instead of silent throttling.
The Kaggle scripts are kept in the repo (`scripts/push_artifacts.py`,
`scripts/sync_notebook.py`) and still work — only the notebooks changed.

## What Colab means for the sprint

| | Reality | Consequence |
|---|---|---|
| Session length | ~4h, can disconnect earlier | **Checkpoint every N steps** — a dropped session must cost minutes, not the day |
| GPU | T4 16GB, availability varies | Run Qwen and Llama **sequentially** — two 4-bit models plus LoRA will OOM |
| Quota | Opaque, throttles silently | Spread training across days; do not batch runs |
| Persistence | Drive mount survives | Simpler than Kaggle — but only if you actually write to Drive, not local disk |
| bf16 | T4 does **not** support bf16 | Use `fp16` compute dtype |

## The rule that replaces "push before session end"

On Kaggle the danger was losing everything at session end. On Colab the danger is
losing everything at a *random* disconnect. The mitigation is the same idea, applied
continuously rather than once:

**Write checkpoints straight to Drive, every N steps.** Never to `/content`, which
vanishes with the session.

```python
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/healthcare-llm/checkpoints/qwen-v1",
    save_steps=100,
    save_total_limit=3,
    ...
)
```

If a session drops mid-run, resume with `resume_from_checkpoint=True` rather than
restarting from zero.

## Drive layout

```
MyDrive/healthcare-llm/
  data/raw/          downloaded once, read-only for the sprint
  data/processed/    cleaned + split
  checkpoints/       in-progress training state
  models/            finished adapters, one dir per artifact
  eval/              evaluation outputs
```

Day 12 (KAN-59) asks for final models on a shared Drive — this layout matches that
directly, which the Kaggle route did not.

## Session hygiene

1. Mount Drive as the **first** cell, every time.
2. Verify the GPU before doing anything expensive — `python scripts/verify_env.py`.
3. Data lives in Drive, so a reconnect does not re-download 140MB.
4. Keep the browser tab open. Colab disconnects idle sessions aggressively.
5. Note the wall-clock time of each run — the master report needs it, and it tells
   you whether a run will fit in a session next time.
