# Healthcare Lane — Qwen + Llama Fine-Tuning

Domain-specific fine-tuning of two base models on healthcare data, run as a four-round
controlled experiment ending in retrieval-augmented generation.

**Project:** Via Codos POC (VC4B) ERDP · Jira project KAN
**Owner:** Kusal Pabasara
**Sprint:** 17 Aug – 1 Sep 2026 (12 working days)

---

## What this is

One experiment, four rounds. The same two base models are trained on a progressively
better dataset, and each round is measured against the last.

| Version | Data | Question it answers |
|---|---|---|
| v1 | Cleaned real data | Baseline — what do we get for free? |
| v2 | v1 + 500 synthetic Q&A on v1's weak topics | Does targeted synthetic data patch known gaps? |
| v3 | Restructured with a `context` field | Can the model learn to read retrieved passages? |
| v4 | RAG-aware, evaluated with live retrieval | Does the full pipeline beat a bare model? |

Two models × four versions = **eight trained artifacts**. The deliverable is not a model,
it is the comparison between them.

## Setup

```bash
pip install -r requirements.txt
```

Versions are pinned deliberately. **Do not bump them mid-sprint** — a silent change alters
results with no visible cause and makes v1..v4 non-comparable.

## Layout

```
configs/      per-model, per-version training configs (YAML)
data/raw/     original downloads — READ-ONLY for the whole sprint
data/processed/  cleaned, split, instruction-formatted
data/synthetic/  generated Q&A pairs (Day 5)
scripts/      training, evaluation, data prep
notebooks/    Colab notebooks
models/       adapters — weights live in Drive, see models/registry.md
eval/         evaluation results and comparison tables
reports/      daily notes and the Day 12 master report
```

## Rules that protect the experiment

1. **`data/raw/` is read-only.** Every later version derives from it. Lose it and v1 is not
   reproducible.
2. **The split seed is fixed on Day 2** and recorded in `PROGRESS.md`. A reshuffled test set
   invalidates every comparison.
3. **The test split is not touched until Day 11.** Tuning against it turns the final numbers
   into fiction.
4. **The eval harness is written on Day 3 and frozen.** All eight artifacts must be scored by
   identical code, or Day 11 becomes a re-scoring marathon.
5. **Version changes are config edits, never code edits.**

## Tracking

- `PROGRESS.md` — day-by-day checklist, frozen decisions, blockers, results
- `models/registry.md` — every trained artifact, one row each
- `data/raw/SOURCES.md` — dataset provenance and licences

## Expected surprise

**v3 will likely score flat or slightly worse than v2** on plain BLEU/ROUGE. This is not a
regression — v3 is trained to use context it is not being given during non-RAG evaluation.
The gain shows up in v4 with retrieval attached.
