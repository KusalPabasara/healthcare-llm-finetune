# Healthcare Lane — Progress Tracker

**Project:** Via Codos POC (VC4B) ERDP · Jira project KAN
**Owner:** Kusal Pabasara (also team lead across industry lanes)
**Sprint:** Mon 17 Aug 2026 → Tue 1 Sep 2026 (12 working days)
**Models:** Qwen-Healthcare + Llama-Healthcare · QLoRA r16/α32, 4-bit
**Platform:** Kaggle Notebooks (9h sessions, 30 GPU-h/week) — see `docs/kaggle-setup.md`
**Plan:** https://claude.ai/code/artifact/7334fb01-cae1-4c94-aae5-4c260763865d

> Jira due dates are `12:00 AM` of the named day = **start** of that day.
> Day N's work happens ON the date below and must be done before the next day begins.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Frozen decisions

Record these once, then never change them mid-sprint. Drift here invalidates the final comparison.

| Decision | Value | Set on |
|---|---|---|
| Split seed | _TBD_ | Day 2 |
| Split ratio | 80 / 10 / 10 | Day 2 |
| Eval harness path | _TBD_ | Day 3 |
| Manual question set (fixed) | _TBD_ | Day 3 |
| transformers version | **4.57.6** (not 5.x — see Day 1 notes) | Day 1 ✓ |
| peft / bitsandbytes | **0.14.0** / **0.45.0** | Day 1 ✓ |
| Base model IDs (Qwen / Llama) | _TBD — record from Colab_ | Day 1 |
| GPU + VRAM | _TBD — `verify_env.py` prints it_ | Day 1 |
| Embedding model (RAG) | _TBD_ | Day 9 |
| Kaggle accelerator | _TBD — P100 or T4×2_ | Day 1 |
| Compute dtype | _fp16 on P100 / bf16 on T4_ | Day 1 |

---

## Day 1 — Mon 17 Aug · KAN-11 · Environment Setup + Data Collection

- [ ] **Expire the leaked Kaggle API key** at kaggle.com/settings, generate a new one
- [ ] Kaggle notebook created, accelerator set (prefer **T4 ×2** — enables parallel training)
- [ ] Add `KAGGLE_USERNAME` / `KAGGLE_KEY` via Add-ons → Secrets
- [ ] Install `requirements.txt` → run `python scripts/verify_env.py`
- [ ] Record GPU type + compute dtype in Frozen decisions
- [x] **Pin exact versions → `requirements.txt`** (silent bumps change later numbers)
- [x] Create project folder structure
- [ ] Find/attach healthcare datasets (Kaggle hosts many natively)
- [ ] Save raw copies to `data/raw/` — **treat as read-only for the whole sprint**
- [ ] Record dataset licences/sources → `data/raw/SOURCES.md` (master report needs them)
- [ ] Generate `data/raw/CHECKSUMS.txt` after download (proves raw data never changed)
- [x] Write README + `docs/kaggle-setup.md`
- [ ] Push raw data to a private Kaggle Dataset (**nothing survives the session otherwise**)
- [ ] **Lead:** post exact versions + folder layout to team channel, ask all lanes to match

**Notes:**
- Repo: https://github.com/KusalPabasara/healthcare-llm-finetune (private)
- **Platform switched from Colab to Kaggle** — 9h sessions vs ~4h, guaranteed GPU,
  30 GPU-h/week visible quota. Tradeoff: no Drive mount, `/kaggle/working` is wiped
  at session end, so every artifact must be pushed to a Kaggle Dataset.
- Pinned `transformers` to **4.57.6**, not 5.x. 5.x is current but has breaking API
  changes from 4.x that most QLoRA tutorials/peft paths don't account for yet.
  Deliberate choice — revisit after the sprint, never during.
- `scripts/verify_env.py` checks pins + GPU + 4-bit config + credentials, and prints
  values to paste into Frozen decisions. Exits non-zero on drift.
- `scripts/push_artifacts.py` pushes an adapter to a private Kaggle Dataset. Must be
  the last cell of every training notebook.
- **A Kaggle key was pasted into chat on 17 Aug — expire it before anything else.**

---

## Day 2 — Tue 18 Aug · KAN-15 · Data Cleaning + QLoRA Configuration

- [ ] Clean and normalize raw data
- [ ] Format into instruction–response pairs
- [ ] Split 80/10/10 — **record the seed in Frozen decisions**
- [ ] Quarantine the test split (not touched again until Day 11)
- [ ] Write Qwen config (LoRA r16, α32)
- [ ] Write Llama config (LoRA r16, α32)
- [ ] Verify Qwen loads in 4-bit and generates a token
- [ ] Verify Llama loads in 4-bit and generates a token
- [ ] Resolve chat template + pad-token differences between the two
- [ ] **Lead:** write up the quantization/tokenizer fixes — other lanes hit these today

**Notes:**

---

## Day 3 — Wed 19 Aug · KAN-19 · Writing Training Pipelines ⚑ highest leverage

- [ ] Qwen training script (config → 4-bit load → LoRA → tokenize → 3 epochs)
- [ ] Llama training script
- [ ] **Data path + version tag driven by config, not code** (v2/v3/v4 = config edit only)
- [ ] Weights & Biases wired up
- [ ] Smoke test: 20 examples, 10 steps, loss decreasing, checkpoint written to Drive
- [ ] **Write the eval harness — BLEU/ROUGE + fixed manual question set**
- [ ] **FREEZE the harness.** Record its path in Frozen decisions
- [ ] **Lead:** propose the shared eval harness to all lanes

**Notes:**

---

## Day 4 — Thu 20 Aug · KAN-24 · Running v1 Training 🖥 GPU

- [ ] Check remaining weekly GPU quota at kaggle.com/settings before starting
- [ ] Sequential (P100) or parallel (T4 ×2 — one model per GPU)
- [ ] Train Qwen → save as **qwen-v1**
- [ ] Train Llama → save as **llama-v1**
- [ ] **Push both adapters** — `push_artifacts.py --name qwen-v1 --path ...`
- [ ] Verify pushes landed before closing the session
- [ ] Record wall-clock training time for each (master report needs it)
- [ ] Update `models/registry.md` with dataset slugs
- [ ] Monitor loss + GPU memory
- [ ] **Lead:** teammates on Colab hit disconnects today — share the Kaggle option
- [ ] Use GPU wait time for teammate pipeline reviews

**Notes:**

---

## Day 5 — Fri 21 Aug · KAN-29 · Testing v1 + Synthetic Data (v2 prep) ⚑ heaviest

- [ ] Score v1 (both models) on test set — BLEU/ROUGE
- [ ] Manually probe 20 healthcare questions
- [ ] **Write down specific failure categories** ← the real deliverable, do this before generating
- [ ] Generate 500+ synthetic Q&A pairs targeting those weaknesses
- [ ] Tag synthetic examples as synthetic (keeps v2 composition auditable)
- [ ] **Spot-check synthetic clinical content against a real source; record that you did**
- [ ] Merge → `train_v2.json`
- [ ] Update both configs to v2
- [ ] **Lead:** Friday sweep — anyone not through v1 slips across the weekend gap

**Notes:**

---

## Day 6 — Mon 24 Aug · KAN-33 · v2 Training + Self-Testing

> Weekend gap before this day — expect broken Colab state across the team.

- [ ] Check weekly GPU quota — **week 2 needs 6 runs, budget ~4h each**
- [ ] Train Qwen on `train_v2.json` → **qwen-v2**
- [ ] Train Llama on `train_v2.json` → **llama-v2**
- [ ] **Push both adapters before the session ends**
- [ ] Full BLEU/ROUGE via the frozen harness
- [ ] Manually test 30 questions
- [ ] Document v1→v2 delta **the same day**
- [ ] If v2 did NOT improve: say so plainly, note which weaknesses the synthetic data missed
- [ ] **Lead:** morning sync — catch people who lost environment state over the weekend

**Notes:**

---

## Day 7 — Tue 25 Aug · KAN-37 · Preparing v3 Data (RAG-Aware) ⚑ lightest day

- [ ] Merge synthetic + original → `train_v3.json`
- [ ] Add `context` field to every example
- [ ] **Make context resemble real ChromaDB output** (messy retrieved chunks, not clean summaries)
- [ ] Include examples where context is irrelevant/absent (teaches not to blind-trust retrieval)
- [ ] Update configs → v3
- [ ] **Prep Day 9:** read Chroma docs, decide embedding model + chunking strategy NOW
- [ ] **Decide what `train_v4.json` will be** (board never assigns a day to build it)
- [ ] **Lead:** start the cross-lane comparison scaffold; raise the v4 gap with the team

**Notes:**

---

## Day 8 — Wed 26 Aug · KAN-41 · v3 Training + Self-Testing 🖥 GPU

- [ ] Train Qwen on `train_v3.json` → **qwen-v3**
- [ ] Train Llama on `train_v3.json` → **llama-v3**
- [ ] **Push both adapters before the session ends**
- [ ] Test set + 30 manual questions
- [ ] Compare against v2
- [ ] **Expect flat/slightly worse plain scores** — v3 is trained to use context it isn't given yet
- [ ] Optional: score with a stub context to preview the coming gain
- [ ] Note the expectation in writing so Day 11 doesn't read this as regression
- [ ] **Lead:** explain the v3 dip publicly before other lanes "fix" a non-broken result

**Notes:**

---

## Day 9 — Thu 27 Aug · KAN-46 · RAG Integration ⚠ highest risk

- [ ] Stand up ChromaDB
- [ ] Index healthcare documents as embeddings
- [ ] **Push the Chroma index as a Kaggle Dataset** — re-embedding costs hours, and
      `/kaggle/working` is wiped at session end. Day 10 needs this index.
- [ ] Sanity-check retrieval alone — junk in, junk out regardless of model quality
- [ ] **Get ONE correct end-to-end answer before optimizing anything**
- [ ] Full pipeline: retrieve → context → fine-tuned model → answer
- [ ] Evaluate on 200 test queries
- [ ] **Build `train_v4.json`** (v3 + real retrieved context) — Day 10 depends on it
- [ ] **If slipping, say so TODAY** — weekend buffer still exists before Day 11
- [ ] **Lead:** propose one shared Chroma setup pattern instead of 5 people debugging separately

**Notes:**

---

## Day 10 — Fri 28 Aug · KAN-51 · v4 Training + RAG Testing 🖥 GPU

- [ ] Attach the Day 9 Chroma index dataset as notebook input
- [ ] Train Qwen on `train_v4.json` → **qwen-v4**
- [ ] Train Llama on `train_v4.json` → **llama-v4**
- [ ] **Push both adapters** — these are the artifacts the recommendation rests on
- [ ] Test both **with RAG attached** on 100 queries
- [ ] Document results
- [ ] Clean, clearly-named checkpoints (last GPU day before final validation)
- [ ] **Lead:** confirm every lane has their v4 defined and running

**Notes:**

---

## Day 11 — Mon 31 Aug · KAN-55 · Final Validation Across All Versions

> Weekend gap before this day.

- [ ] **Attach all 8 adapter datasets as notebook inputs** (Kaggle allows this)
- [ ] Load all 8 artifacts: qwen v1–v4, llama v1–v4
- [ ] Run the **frozen** harness across every version (start the batch early — it's slow)
- [ ] Watch the 9h session limit — 8 evaluations is a long batch; split across two sessions if needed
- [ ] Build comparison chart: version × metric, two model series
- [ ] Verify chart is readable by someone who wasn't in the sprint
- [ ] **Lead:** collect other lanes' final numbers TODAY, not tomorrow

**Notes:**

---

## Day 12 — Tue 1 Sep · KAN-59 · Final Save + Master Report

- [ ] 100 final test questions, manually checked
- [ ] Master report: Qwen vs Llama scores, training times, dataset sizes, final metrics
- [ ] **Recommendation written as a decision** — which model for healthcare, and why (incl. cost + inference speed, not just BLEU)
- [ ] Include what didn't work (failed synthetic categories, the v3 dip)
- [ ] Upload all v4 models to shared Drive — **KAN-59 explicitly asks for Drive**, so
      download from Kaggle and upload, or share the Kaggle Dataset and confirm that satisfies it
- [ ] Master registry file: name, version, base model, training data, date, metrics, path
- [ ] **Verify uploads open from a different account** before closing
- [ ] **Lead:** share your report structure early so all lanes arrive consistent

**Notes:**

---

## Blockers log

| Date | Day | Blocker | Status | Raised to team? |
|---|---|---|---|---|
| | | | | |

## Results log

Fill as you go — Day 11 assembles from this.

| Artifact | BLEU | ROUGE | Manual pass rate | Train time | Notes |
|---|---|---|---|---|---|
| qwen-v1 | | | | | |
| llama-v1 | | | | | |
| qwen-v2 | | | | | |
| llama-v2 | | | | | |
| qwen-v3 | | | | | |
| llama-v3 | | | | | |
| qwen-v4 | | | | | |
| llama-v4 | | | | | |
