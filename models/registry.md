# Model Registry

Every trained artifact, one row. Weights live in Drive; this file is the index.

Fill a row the moment a training run finishes — reconstructing this on Day 12 from memory
is how details get lost.

| Artifact | Base model | Version | Train data | Trained | Train time | Drive path | BLEU | ROUGE-L | Manual pass |
|---|---|---|---|---|---|---|---|---|---|
| qwen-v1 | | v1 | `data/processed/train_v1.json` | | | | | | |
| llama-v1 | | v1 | `data/processed/train_v1.json` | | | | | | |
| qwen-v2 | | v2 | `data/processed/train_v2.json` | | | | | | |
| llama-v2 | | v2 | `data/processed/train_v2.json` | | | | | | |
| qwen-v3 | | v3 | `data/processed/train_v3.json` | | | | | | |
| llama-v3 | | v3 | `data/processed/train_v3.json` | | | | | | |
| qwen-v4 | | v4 | `data/processed/train_v4.json` | | | | | | |
| llama-v4 | | v4 | `data/processed/train_v4.json` | | | | | | |

## Shared hyperparameters

Constant across all runs unless a row says otherwise. Changing any of these mid-sprint means
the comparison is no longer clean — note it explicitly if you do.

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Quantization | 4-bit (nf4) |
| Epochs | 3 |
| Split | 80 / 10 / 10 |
| Split seed | _set Day 2_ |

## Before closing Day 12

- [ ] Every row filled
- [ ] All v4 artifacts uploaded to shared Drive
- [ ] Uploads verified openable **from a different account**
