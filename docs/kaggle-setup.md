# Kaggle Workflow

Training for this sprint runs on Kaggle Notebooks rather than Colab.

## Why Kaggle

| | Colab (free) | Kaggle |
|---|---|---|
| Session length | ~4h, disconnects unpredictably | **9h**, predictable |
| GPU | T4 16GB, availability varies | **P100 16GB or 2×T4**, guaranteed |
| Weekly quota | Opaque, silent throttling | **30 GPU-hours**, visible counter |
| Datasets | Re-download each session | Native, versioned, attached as input |
| Persistence | Drive mount survives | **Nothing survives** — must push |

The 9-hour guaranteed session is the main win: both models can train sequentially in one
sitting without babysitting a connection.

## The one real tradeoff

`/kaggle/working` is **wiped when the session ends**. There is no Drive mount. Every
artifact must be pushed to a Kaggle Dataset before the notebook closes, or the run is lost.

This is the single most important operational rule of the sprint.

## Credentials

Never commit `kaggle.json`. It is gitignored, and it must stay that way.

**In a Kaggle notebook** — the API works against your own account automatically for most
operations. For dataset pushes, add credentials via **Add-ons → Secrets**:

```python
from kaggle_secrets import UserSecretsClient
import os
secrets = UserSecretsClient()
os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
```

**Locally:**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

If a key is ever exposed — pasted into a chat, committed, screenshotted — expire it
immediately at kaggle.com/settings and generate a new one.

## Notebook skeleton

Every training notebook follows this shape:

```python
# 1. Pin the environment
!pip install -q -r /kaggle/input/healthcare-repo/requirements.txt

# 2. Verify before doing anything expensive
!python /kaggle/input/healthcare-repo/scripts/verify_env.py

# 3. Train
!python scripts/train.py --config configs/qwen_v1.yaml

# 4. PUSH BEFORE THE SESSION ENDS  <-- never skip
!python scripts/push_artifacts.py --name qwen-v1 --path /kaggle/working/qwen-v1
```

Step 4 is not optional. Make it the last cell of every notebook.

## Session budget

30 GPU-hours per week. The sprint needs 8 training runs (2 models × 4 versions), spread
across two calendar weeks:

| Week | Days | Runs | Budget |
|---|---|---|---|
| 17–21 Aug | 4, 5 | qwen-v1, llama-v1 | 30h available |
| 24–28 Aug | 6, 8, 10 | v2, v3, v4 × 2 models = 6 runs | 30h available, ~5h each |

Week two is the tighter one — six runs. Keep each under ~4h and there is comfortable
headroom. Check remaining quota at kaggle.com/settings before each training day.

## P100 vs T4

Kaggle offers **P100 (single, 16GB)** or **T4 ×2 (16GB each)**.

- **P100** — faster single-GPU, but **no bfloat16**. Use `fp16` compute dtype.
- **T4 ×2** — supports bf16, and two GPUs means Qwen and Llama can train **in parallel**.

For Days 4, 8 and 10 the T4 ×2 option is worth taking: parallel training halves the
wall-clock. `verify_env.py` detects which you have and prints the right compute dtype.

## Loading an adapter back

Trained adapters are Kaggle Datasets. To use one on a later day, attach it as notebook
input (**Add Input → Datasets → `<username>/healthcare-qwen-v1`**), then:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "/kaggle/input/healthcare-qwen-v1")
```

Day 11 needs all eight attached at once for the cross-version comparison.

## Datasets

Kaggle hosts healthcare datasets natively — search and attach rather than downloading.
Record provenance in `data/raw/SOURCES.md` regardless of source; the Day 12 report needs
licences, and attached-dataset licences are easy to forget because you never clicked
through a download page.
