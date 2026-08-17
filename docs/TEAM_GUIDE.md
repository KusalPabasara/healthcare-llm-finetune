# Domain LLM Fine-Tuning — Team Guide

| | |
|---|---|
| **Project** | Via Codos Proof of Concept (VC4B) ERDP · Jira project `KAN` |
| **Sprint** | Mon 17 Aug – Tue 1 Sep 2026 (12 working days) |
| **Team lead** | Kusal Pabasara |
| **Version** | 2.0 · 17 Aug 2026 · aligned to the company scope document |

---

## Before anything else: the board and the scope document disagree

Management issued *AI Model Development Project — Scope, Expectations & Specialization
Guide*. It is the authority on **what each model should do**. Our Jira board is the
authority on **what work is scheduled**. On several points they do not match, and the
differences are not cosmetic.

| | Company scope document | Jira board (`KAN`) |
|---|---|---|
| Engineers | 7 | 5 |
| Industries | 7 separate tracks | 4 lanes |
| **Deepana** | **Legal** | **SME Daily Business** |
| **Thisal** | **Manufacturing** — own track, 3 days late | **Support** on Rimaz's lane |
| **Rimaz** | Retail | Retail, E-commerce **and Manufacturing** |
| Oshan, Gevindu | Education, Hospitality tracks | Not on this board |
| Duration | 14 days | 12 days |

**Work to the Jira board for scheduling. Work to the scope document for what the model
should be capable of.** Where they conflict on your industry — Deepana especially — stop
and ask before Day 2, because building the wrong domain is not recoverable inside this
sprint.

Kusal is raising all of the above with management. Flagged items are marked
**[NEEDS CONFIRMATION]** throughout this guide.

---

## Read this first

None of us have built a model like this before. This guide is the shared map: what
we're building, why, and what to do on each of the twelve days. Jira has the tickets —
this has the reasoning behind them.

**If you read only one section, read [What we are actually building](#what-we-are-actually-building),
[The capability bar](#the-capability-bar-every-model-must-reach) and
[The five rules](#the-five-rules-that-protect-the-experiment).** Everything else you can
come back to on the day you need it.

### Contents

1. [What we are actually building](#what-we-are-actually-building)
2. [The capability bar every model must reach](#the-capability-bar-every-model-must-reach)
3. [Who owns what](#who-owns-what)
4. [The five rules that protect the experiment](#the-five-rules-that-protect-the-experiment)
5. [Non-negotiables from management](#non-negotiables-from-management)
6. [Concepts you need](#concepts-you-need-in-plain-terms)
7. [Choosing your datasets](#choosing-your-datasets)
8. [The twelve days](#the-twelve-days)
9. [Your lane](#your-lane-day-by-day)
10. [Known traps](#known-traps-found-the-hard-way)
11. [Open questions](#open-questions)

---

## What we are actually building

Strip away the day labels and this is **one controlled experiment, run four times in
parallel.**

Each of us takes two base models — **Qwen** and **Llama** — and fine-tunes them on our own
industry's data, four times over, with each round using better data than the last. Then we
measure whether the data changes actually made the models better.

| Version | Data it trains on | The question it answers |
|---|---|---|
| **v1** | Real collected data, cleaned | Baseline. What do we get for free? |
| **v2** | v1 + ~500 synthetic examples targeting v1's weak spots | Does targeted synthetic data fix known gaps? |
| **v3** | Same data restructured with a `context` field | Can the model learn to read retrieved passages? |
| **v4** | RAG-aware, evaluated with live retrieval | Does the full pipeline beat a bare model? |

Two models × four versions = **8 trained artifacts per person**, 32 across the team.

**The deliverable is not a model. It is a comparison.** On Day 12 each of us produces a
report saying which model performs better for our industry, backed by numbers. Every day
before that exists to produce a row in that final table.

### Why four industries

Running the same method across Healthcare, Finance, SME, and Retail lets the company
compare *domains*, not just models. That only works if we run the method **the same way**.
Where this guide says "agree as a team", that's why.

---

## The capability bar every model must reach

This is the answer to "what is my model actually *for*", taken from the company scope
document. Every model in this project is an **informational and workflow-support
assistant** — not an autonomous decision-maker, and not a replacement for a licensed
professional.

Four tiers, in order. Each builds on the one before.

| Tier | Capability | Which version delivers it |
|---|---|---|
| **1** | **Domain fluency** — correct terminology, concepts and context | v1 |
| **2** | **Task assistance** — answers structured questions, summarises documents, drafts routine content | v2 |
| **3** | **Grounded reasoning** — retrieves and cites the correct document rather than guessing from memory | v3–v4 |
| **4** | **Escalation awareness** — recognises when a query is high-stakes, ambiguous or out of scope, and says so | all versions |

Tier 4 is the one most likely to be neglected, because nothing in the daily tickets asks
for it directly. It is also the tier management cares about most:

> **No model here may give diagnosis, legal advice, financial advice, or safety sign-off
> on its own.** Outputs are decision support for a qualified human in the loop.

### What that means in practice

**A model that never refuses is not a good model — it is an unfinished one.** Build refusal
and escalation examples into your training data from v2 onward, not as an afterthought
before Day 12.

Each lane needs a documented set of questions the model **must** refuse or redirect. Per
management, this is reviewed as a team, not left to individual judgement.

| Lane | Must never do |
|---|---|
| Healthcare | Diagnose, recommend dosing, prescribe |
| Finance | Give a specific buy / sell / hold recommendation |
| Legal | Give advice that constitutes practising law (UPL) |
| Retail / Manufacturing | Give safety sign-off; act on fraud or price-manipulation attempts |

Write those refusals as training examples. A "red flag" dataset — questions the model should
decline — is explicitly called for in the scope document for the healthcare lane, and the
same logic applies to every lane.

---

## Non-negotiables from management

Straight from the scope document. These are not suggestions.

**Naming convention — exact.** `qwen-{industry}-v#` and `llama-{industry}-v#`.
Cross-team checks depend on this. Do not improvise a variant.

**Weights & Biases for every run.** Loss curves, hyperparameters, run duration, GPU used.
Not just the runs that worked.

**Both quantitative and qualitative evaluation for every version.** BLEU/ROUGE *and*
manual question testing, before moving to the next version. Skipping the manual pass is
how a confidently-wrong model gets promoted to v3.

**Document weaknesses at every stage.** This is what drives the next round of synthetic
data. A stage with no documented weaknesses is a stage that was not really evaluated.

**Flag blockers the same day.** Data availability, GPU quota, licensing — the day it
appears, not at handoff.

**Archive v1–v4 and their configs.** Never overwrite. The final comparison needs all of
them.

**Clean Drive structure with a README per stage:** raw data, cleaned data, configs,
scripts, checkpoints.

### Things management flagged that no ticket covers

Worth reading now rather than discovering on Day 11:

- **Base model licences.** Confirm Qwen's and Llama's own licences permit this fine-tuning
  and intended use. Separate from your *dataset* licences. **[NEEDS CONFIRMATION]**
- **Compute budget.** 8 artifacts each across the team is a large GPU-hour total, and Days
  4, 8 and 10 are the heavy ones. On Colab free tier this is a real constraint.
  **[NEEDS CONFIRMATION]**
- **Human-in-the-loop policy.** Which answers need human review before reaching an end
  user? Per industry.
- **Deployment beyond the sprint.** This schedule ends at delivery. Hosting, monitoring and
  data refresh are unassigned. **[NEEDS CONFIRMATION]**

---

## Who owns what

Everyone runs the same 12-day shape, on the same dates, with the same due dates.

| Lane | Owner | Model names | Jira range |
|---|---|---|---|
| Healthcare | **Kusal Pabasara** | `qwen-healthcare-v#`, `llama-healthcare-v#` | KAN-11 → KAN-59 |
| Finance | **Navodya Dissanayake** | `qwen-finance-v#`, `llama-finance-v#` | KAN-12 → KAN-60 |
| SME Daily Business | **Deepana Nirmal** | `qwen-sme-v#`, `llama-sme-v#` **[NEEDS CONFIRMATION]** | KAN-13 → KAN-61 |
| Retail / E-commerce / Manufacturing | **Rimaz Nowfel** | `qwen-retail-v#`, `llama-retail-v#` | KAN-14 → KAN-62 |
| Retail lane support | **Thisal Sooriyanayaka** | — | Days 3, 4, 8, 9 |

Because the lanes are synchronised, **any problem you hit on Day N, three other people hit
the same day.** Post fixes in the team channel — ten minutes of writing saves the team hours.
This is the single highest-value habit for this sprint.

### Your mission, per lane

From the company scope document. This is what your model is being built to do — read your
own section carefully before Day 2, because it determines what data you collect and what
"weak area" means on Day 5.

#### Kusal — Healthcare

A healthcare information and clinical-reference assistant for clinicians, administrative
staff, and patients. Fast, correctly-sourced answers, **never a diagnosis or a treatment
prescription.**

*Cover in training data:* general medicine and clinical reference · pharmacology basics
(drug classes and interactions, **not dosing**) · medical exam reasoning (USMLE-style) ·
research-literature summarisation · patient education in plain language · hospital
administration and billing · emergency triage information that always defers to emergency
services.

*Before v4:* confirm no real patient data ever enters training · bias testing across age,
gender and ethnicity · a hard rule set for what must not be answered · a citation format
a clinician can verify · decide whether English-only is acceptable.

#### Navodya — Finance

A finance research and analysis assistant covering corporate finance, markets and personal
finance literacy — **informational analysis, never individualised investment advice.**

*Cover in training data:* SEC filings comprehension (10-K/10-Q) · earnings-call
summarisation · market and news interpretation · personal finance literacy · accounting
fundamentals · basic risk and compliance awareness.

*Before v4:* clarify market scope (US-only or global) · build refusal examples so the model
never phrases output as buy/sell/hold · document the training data's as-of date, since
market data ages fast · pick GAAP or IFRS and note the gap · confirm outputs cannot be read
as investment advice under securities law.

#### Deepana — SME Daily Business **[NEEDS CONFIRMATION]**

> **The scope document assigns Deepana to Legal; the Jira board assigns SME Daily Business.
> These are different models. Resolve this before Day 2.**

If **SME Daily Business** is correct: a general small-business operations assistant —
bookkeeping basics, customer communication, scheduling, supplier and inventory questions,
simple compliance. Note that this lane has no standard public benchmark, so expect a harder
Day 1 and heavier reliance on synthetic generation from Day 5.

If **Legal** is correct: a legal research and drafting-support assistant — case law
summarisation, contract understanding, statutory interpretation, employment law, IP
fundamentals. Positioned as **legal information, not legal advice**. Critically, every
training example needs a **jurisdiction tag**, since laws vary by region and an untagged
legal model gives confidently wrong answers across borders. Also needs
unauthorised-practice-of-law refusal examples and an as-of date on every statute.

#### Rimaz — Retail / E-commerce / Manufacturing

The board combines what the scope document treats as two tracks: **Retail** and
**Manufacturing**. That makes this the broadest lane, which is presumably why Thisal
supports it on the four heaviest days.

*Retail side:* product discovery · customer service (orders, returns, complaints) ·
inventory and supply chain · policy explanation · product copywriting · review sentiment.

*Manufacturing side:* safety procedures · equipment operation and maintenance · quality
control · troubleshooting · production planning · regulatory awareness (OSHA-style).

*Before v4:* **manufacturing is safety-critical** — wrong equipment or safety guidance can
injure someone, so it needs a stricter evaluation bar than any other lane here. Also: decide
brand voice · plan for catalogue freshness · guardrails against return-fraud and
price-manipulation attempts · confirm which regulatory framework applies.

> **[NEEDS CONFIRMATION]** Combining a customer-service domain with a safety-critical one in
> a single model is a real design question, not just extra scope. Ask whether these should
> be separate models before Day 2.

#### Thisal — Support (Retail lane)

Days 3, 4, 8 and 9: the pipeline-writing day, two GPU-heavy training days, and the RAG day.
Rimaz and Thisal should agree who runs which model to avoid duplicating work.

> **[NEEDS CONFIRMATION]** The scope document has Thisal owning Manufacturing as his own
> track, starting three days behind. The board has him supporting Rimaz. Very different jobs.

### Reading the Jira dates correctly

Due dates are set to `12:00 AM` of the named day, which means the **start** of that day.
Day 1's "due 17/Aug 12:00 AM" means the work happens **on the 17th** and must be done before
the 18th begins. Not "due at the end of the 17th".

---

## The five rules that protect the experiment

Break any of these and the final comparison stops being trustworthy. They cost nothing to
follow on Day 1 and are expensive to fix later.

**1. `data/raw/` is read-only for the whole sprint.**
Everything else derives from it. If you lose or modify your raw data, v1 can never be
reproduced and the whole version comparison collapses.

**2. Fix the split seed on Day 2 and write it down.**
Your 80/10/10 train/validation/test split must be identical across all four versions. A
reshuffled test set makes v1 and v4 scores incomparable — you'd be measuring different exams.

**3. Do not touch the test split until Day 11.**
If you look at test results and tune against them, your final numbers describe how well you
tuned, not how good the model is. Use the validation split during development.

**4. Write the evaluation harness on Day 3, then freeze it.**
All 8 of your artifacts must be scored by *identical* code. If you improve the harness on
Day 8, every earlier score is invalid and you re-run everything on Day 11.

**5. Version changes are config edits, never code edits.**
`train_v2.json` → `train_v3.json` should be a line in a config file. If you're editing the
training script between versions, you've changed two things at once and can't attribute the
difference.

---

## Concepts you need, in plain terms

Skip anything you already know.

### Fine-tuning
Taking a model that already understands language and teaching it your domain. We are not
training from scratch — that costs millions. We're adjusting an existing model.

### LoRA and QLoRA
Fine-tuning normally means updating billions of parameters, which needs far more GPU memory
than we have.

**LoRA** (Low-Rank Adaptation) freezes the original model and trains a small set of extra
weights — an "adapter" — that sits alongside it. You train maybe 0.1% as many parameters.

**QLoRA** adds quantisation: the frozen base model is compressed to 4 bits, cutting memory
further. This is what makes a 7B model trainable on a free Colab T4.

Our settings, the same for everyone: **rank 16, alpha 32, 4-bit, 3 epochs.**

> **What you actually save** is the adapter — a few hundred MB — not the full model. To use
> it later you load the base model and apply the adapter on top.

### Instruction–response pairs
The format models learn from. Raw data becomes:

```json
{"instruction": "What is the first-line treatment for X?",
 "input": "",
 "output": "The first-line treatment is..."}
```

Day 2 is mostly converting your raw data into this shape.

### Epochs, loss, checkpoints
- **Epoch** — one full pass through your training data. We use 3.
- **Loss** — how wrong the model is. It should go *down*. If it doesn't, something's broken.
- **Checkpoint** — a saved snapshot mid-training. Essential on Colab, which disconnects.

### RAG (Retrieval-Augmented Generation)
Instead of relying on what the model memorised, you *look up* relevant documents at question
time and hand them to the model as context. Days 7–10 are about making our models good at
using retrieved context.

**ChromaDB** is the vector database we use to store and search documents by meaning.

### BLEU and ROUGE
Metrics comparing generated text to a reference answer by word overlap.

> **Important limitation:** they measure *wording*, not *correctness*. A confidently wrong
> answer phrased like the reference scores well. This is why manual testing is in every
> ticket — do not skip it.

---

## Choosing your datasets

**Do this before you download anything.** Two of the four best-known healthcare datasets
turned out to be unusable for a commercial project, and the same trap exists in every lane.

### The licence rule

This is a **commercial** proof of concept. That rules out:

| Licence type | Verdict | Why |
|---|---|---|
| MIT, Apache-2.0, CC-BY-4.0 | **Safe** | Commercial use permitted, attribution only |
| CDLA-Sharing, CC-BY-SA | **Ask first** | Share-alike may encumber derived model weights |
| **CC-BY-NC** (any) | **Do not use** | Non-commercial. Explicitly excludes this project |
| **No licence stated** | **Do not use** | No licence means no permission |

Check the licence on the dataset's own page. **Do not trust a blog post or a tutorial** —
several widely-recommended datasets are non-commercial and the tutorials never mention it.

### Verified starting points per lane

Checked against dataset pages on 17 Aug 2026. Verify again before you use them.

**Healthcare — Kusal**
- `openlifescienceai/medmcqa` — Apache-2.0 — 182k medical MCQs, 21 subjects, expert explanations
- `qiaojin/PubMedQA` (pqa_labeled) — MIT — 1k expert-labeled QA **with a `context` field**
- **Excluded.** ChatDoctor-HealthCareMagic-100k — no stated licence
- **Excluded.** MedQuAD — CC BY-SA 4.0 share-alike

**Finance — Navodya**
- `ibm-research/finqa` — CC-BY-4.0 — 8k QA over financial tables, numerical reasoning
- `gbharti/finance-alpaca` — MIT — 68k instruction pairs
- **Excluded — read this.** **Financial PhraseBank — CC-BY-NC-SA-3.0.** This is the most commonly recommended
  finance dataset and it is **non-commercial**. The authors will license it commercially on
  request — ask early if you want it.

**Retail / E-commerce / Manufacturing — Rimaz & Thisal**
- `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset` — CDLA-Sharing-1.0, share-alike —
  44.8k instruction/response pairs. Commercial use permitted **but share-alike** — confirm
  with whoever owns licensing before building v1 on it.

**SME Daily Business — Deepana**
This is the hardest lane to source, because "SME daily business" has no standard benchmark.
Expect to combine general business-instruction data with customer-service data, and to lean
more heavily on synthetic generation from Day 5. Budget extra time on Day 1 for the search,
and ask the team if you're stuck — this is a genuine difficulty, not a personal one.

### Record everything as you download

Fill in a `SOURCES.md` with dataset name, URL, licence, row count, and date. The Day 12
report needs it, and reconstructing licences two weeks later is miserable.

---

## The twelve days

The same shape for everyone. Dates are when the work happens.

### Day 1 — Mon 17 Aug · Environment Setup + Data Collection

Get Colab running with a GPU, install pinned library versions, download your datasets,
save raw copies, back up to Drive.

- **Set Runtime → Change runtime type → T4 GPU** before anything else.
- **Pin your library versions** and share them with the team. A silent version bump changes
  results with no visible cause.
- Keep raw data on **Google Drive**, not `/content` — Colab wipes local storage.

**Done when:** GPU verified, datasets on Drive, licences recorded.

### Day 2 — Tue 18 Aug · Data Cleaning + QLoRA Configuration

Clean and normalise, format into instruction–response pairs, split 80/10/10, write configs
for both models, confirm both load in 4-bit.

- **Record your split seed.** ([Rule 2](#the-five-rules-that-protect-the-experiment))
- Qwen and Llama want **different chat templates and pad-token handling**. This is the most
  common Day 2 blocker — expect it, and post your fix.
- Proving both models load and generate one token is the real goal here.
- **Format toward your lane's mission, not generically.** A numerical-reasoning lane keeps
  the calculation steps; a clinical lane keeps the explanation; a legal lane needs a
  jurisdiction tag on every example.

**Done when:** `train.json` / `val.json` / `test.json` exist and both models load in 4-bit.

### Day 3 — Wed 19 Aug · Writing Training Pipelines

*Highest-leverage day of the sprint.*

Write training scripts for both models, smoke-test on a small subset, set up Weights & Biases.

**This day determines whether Days 4–12 are calm or painful.** Everything after this re-runs
what you write today with a different data path.

- Make the **data path and version tag config-driven** ([Rule 5](#the-five-rules-that-protect-the-experiment)).
- **Write the evaluation harness now**, alongside training — BLEU/ROUGE plus a fixed set of
  manual questions. Then freeze it ([Rule 4](#the-five-rules-that-protect-the-experiment)).
- Smoke test means 20 examples and 10 steps: does loss decrease, does a checkpoint save?

**Done when:** both scripts train on a tiny subset without error, and the harness exists.

### Day 4 — Thu 20 Aug · Running v1 Training

Run both trainings, monitor loss and GPU memory, save as v1.

- **Run sequentially, not simultaneously.** Two 4-bit models plus LoRA will exhaust a single
  T4. Qwen first, then Llama.
- **Checkpoint to Drive every ~100 steps.** Colab disconnects; a drop should cost minutes.
- Record **wall-clock training time** — the Day 12 report needs it.
- Long GPU waits are the natural time to review a teammate's pipeline.

**Done when:** `qwen-v1` and `llama-v1` are saved to Drive with training times recorded.

### Day 5 — Fri 21 Aug · Testing v1 and Synthetic Data

*The heaviest day.*

Score v1, manually probe 20 domain questions, identify weak areas, generate 500+ synthetic
Q&A pairs targeting those weaknesses, merge into `train_v2.json`.

Two distinct jobs in one day. Start the manual probing early — the synthetic generation
depends on knowing what's weak.

- **The weakness analysis is the real deliverable**, not the 500 pairs. Random synthetic
  data will not move v2. Write down specific failure categories first.
- **Tag synthetic examples as synthetic** so v2's composition stays auditable.
- **Write your refusal set now and include it in v2.** Tier 4 (escalation awareness) is not
  delivered by any ticket — it has to be trained in. Generate examples where the right
  answer is "I can't answer that, here's who can". Your lane's must-never-do list is in
  [Who owns what](#who-owns-what).
- Spot-check generated content against a real source, and record that you did.

**Done when:** you can name your model's specific weaknesses, and `train_v2.json` exists.

### Day 6 — Mon 24 Aug · v2 Training + Self-Testing

> Weekend gap before this day. Colab environments break over gaps — budget setup time.

Train both on `train_v2.json`, evaluate, manually test 30 questions, document the v1→v2 change.

- **First real evidence the method works.** If v2 hasn't improved, the synthetic data missed
  the weaknesses. **Say so plainly** — a negative result documented is worth more than a
  vague positive.
- Write up the delta the same day, while you remember what the outputs looked like.

**Done when:** v2 artifacts saved and the v1→v2 comparison is written down.

### Day 7 — Tue 25 Aug · Preparing v3 Data (RAG-Aware)

*The lightest day — use the slack deliberately.*

Merge data into `train_v3.json` and add a `context` field to every example.

- **No training runs today.** Use the slack for Day 9 preparation and helping teammates.
- **Context must resemble what ChromaDB will actually return** — messy retrieved chunks, not
  clean hand-written summaries. Train on tidy context and the model breaks on real retrieval.
- Include examples where context is **irrelevant or absent**, so the model learns not to
  blindly trust retrieval.
- **Read the ChromaDB docs today** and decide your embedding model and chunking strategy.

**Done when:** `train_v3.json` has context on every example, and you know your Day 9 plan.

### Day 8 — Wed 26 Aug · v3 Training and Self-Testing

Train both on `train_v3.json`, evaluate, compare against v2.

> **Expect v3 to score flat or slightly worse** on plain BLEU/ROUGE without retrieval.
> This is **not** a regression — v3 is trained to use context it isn't being given yet. Note
> the expectation now so nobody "fixes" a result that isn't broken.

**Done when:** v3 artifacts saved, and the expected dip is documented as expected.

### Day 9 — Thu 27 Aug · RAG Integration

*Highest risk. The day most likely to slip.*

Stand up ChromaDB, index your documents as embeddings, wire up
retrieve → context → model → answer, evaluate on 200 queries.

**The most likely day to slip.** New infrastructure, embedding choices, chunking strategy,
and 200 evaluation queries — with Day 10 depending on all of it.

- **Get one correct end-to-end answer before optimising anything.** Working beats tuned.
- **Persist the Chroma index to Drive.** Re-embedding after a disconnect costs hours.
- **Sanity-check retrieval on its own.** If the retriever returns junk, no model quality saves
  the answer.
- **Build `train_v4.json` today** (v3 + real retrieved context) — see [Known traps](#known-traps-found-the-hard-way).
- **If you're going to slip, say so today**, while a weekend of buffer still exists.

**Done when:** the RAG pipeline answers correctly end to end, and the index is on Drive.

### Day 10 — Fri 28 Aug · v4 Training and RAG Testing

Train both on `train_v4.json`, test with RAG attached on 100 queries.

This is the payoff run. v4 + RAG is the configuration your recommendation rests on.

**Done when:** v4 artifacts saved and RAG-attached results documented.

### Day 11 — Mon 31 Aug · Final Validation

> Weekend gap before this day.

Load all 8 artifacts, run the frozen harness across every version, build the comparison chart.

- **This day only works if the harness stayed frozen** ([Rule 4](#the-five-rules-that-protect-the-experiment)).
- Loading 8 adapters takes real time. Start the batch early.
- The chart is the report's centrepiece: version on one axis, metric on the other, two series
  for the two models. Make it readable by someone who wasn't in the sprint.

**Done when:** all 8 artifacts scored by identical code, chart built.

### Day 12 — Tue 1 Sep · Final Save + Master Report

100 final test questions checked manually. Master report with scores, training times, dataset
sizes, and a recommendation. All v4 models uploaded to shared Drive with a registry file.

- **Write the recommendation as a decision, not a summary.** Which model, for your industry,
  and why — including cost and inference speed, not only BLEU.
- **Include what didn't work.** Failed synthetic categories and the v3 dip are genuinely
  useful findings. A report with only good news is less trustworthy.
- **Verify uploads open from a different account** before closing the ticket.
- **Check the naming convention one last time:** `qwen-{industry}-v#` and
  `llama-{industry}-v#`. Cross-team comparison depends on it.
- **Confirm v1–v4 and their configs are archived, not overwritten.** Management asks for
  this explicitly.

**Done when:** report delivered, models uploaded and verified, registry complete.

---

## Your lane, day by day

Exact Jira issue keys, generated from the board.

### Kusal Pabasara — Healthcare

| Day | Date | Issue | Task |
|---|---|---|---|
| 1 | 17/Aug/26 | `KAN-11` | Environment Setup + Data Collection |
| 2 | 18/Aug/26 | `KAN-15` | Data Cleaning + QLoRA Configuration |
| 3 | 19/Aug/26 | `KAN-19` | Writing Training Pipelines |
| 4 | 20/Aug/26 | `KAN-24` | Running v1 Training (Qwen + Llama) |
| 5 | 21/Aug/26 | `KAN-29` | Testing v1 + Synthetic Data Generation (v2 prep) |
| 6 | 24/Aug/26 | `KAN-33` | Running v2 Training + Self-Testing v2 |
| 7 | 25/Aug/26 | `KAN-37` | Preparing v3 Data (RAG-Aware Dataset) |
| 8 | 26/Aug/26 | `KAN-41` | Running v3 Training + Self-Testing v3 |
| 9 | 27/Aug/26 | `KAN-46` | RAG Integration + Testing |
| 10 | 28/Aug/26 | `KAN-51` | Running v4 Training + Testing (RAG-Aware) |
| 11 | 31/Aug/26 | `KAN-55` | Final Validation Across All Versions |
| 12 | 01/Sep/26 | `KAN-59` | Final Save + Master Report |

### Navodya Dissanayake — Finance

| Day | Date | Issue | Task |
|---|---|---|---|
| 1 | 17/Aug/26 | `KAN-12` | Environment Setup + Data Collection |
| 2 | 18/Aug/26 | `KAN-16` | Data Cleaning + QLoRA Configuration |
| 3 | 19/Aug/26 | `KAN-20` | Writing Training Pipelines |
| 4 | 20/Aug/26 | `KAN-25` | Running v1 Training (Qwen + Llama) |
| 5 | 21/Aug/26 | `KAN-30` | Testing v1 + Synthetic Data Generation (v2 prep) |
| 6 | 24/Aug/26 | `KAN-34` | Running v2 Training + Self-Testing v2 |
| 7 | 25/Aug/26 | `KAN-38` | Preparing v3 Data (RAG-Aware Dataset) |
| 8 | 26/Aug/26 | `KAN-42` | Running v3 Training + Self-Testing v3 |
| 9 | 27/Aug/26 | `KAN-47` | RAG Integration + Testing |
| 10 | 28/Aug/26 | `KAN-52` | Running v4 Training + Testing (RAG-Aware) |
| 11 | 31/Aug/26 | `KAN-56` | Final Validation Across All Versions |
| 12 | 01/Sep/26 | `KAN-60` | Final Save + Master Report |

### Deepana Nirmal — SME Daily Business

| Day | Date | Issue | Task |
|---|---|---|---|
| 1 | 17/Aug/26 | `KAN-13` | Environment Setup + Data Collection |
| 2 | 18/Aug/26 | `KAN-17` | Data Cleaning + QLoRA Configuration |
| 3 | 19/Aug/26 | `KAN-21` | Writing Training Pipelines |
| 4 | 20/Aug/26 | `KAN-26` | Running v1 Training (Qwen + Llama) |
| 5 | 21/Aug/26 | `KAN-31` | Testing v1 + Synthetic Data Generation (v2 prep) |
| 6 | 24/Aug/26 | `KAN-35` | Running v2 Training + Self-Testing v2 |
| 7 | 25/Aug/26 | `KAN-39` | Preparing v3 Data (RAG-Aware Dataset) |
| 8 | 26/Aug/26 | `KAN-43` | Running v3 Training + Self-Testing v3 |
| 9 | 27/Aug/26 | `KAN-48` | RAG Integration + Testing |
| 10 | 28/Aug/26 | `KAN-53` | Running v4 Training + Testing (RAG-Aware) |
| 11 | 31/Aug/26 | `KAN-57` | Final Validation Across All Versions |
| 12 | 01/Sep/26 | `KAN-61` | Final Save + Master Report |

### Rimaz Nowfel — Retail / E-commerce / Manufacturing

| Day | Date | Issue | Task |
|---|---|---|---|
| 1 | 17/Aug/26 | `KAN-14` | Environment Setup + Data Collection |
| 2 | 18/Aug/26 | `KAN-18` | Data Cleaning + QLoRA Configuration |
| 3 | 19/Aug/26 | `KAN-22` | Writing Training Pipelines |
| 4 | 20/Aug/26 | `KAN-27` | Running v1 Training (Qwen + Llama) |
| 5 | 21/Aug/26 | `KAN-32` | Testing v1 + Synthetic Data Generation (v2 prep) |
| 6 | 24/Aug/26 | `KAN-36` | Running v2 Training + Self-Testing v2 |
| 7 | 25/Aug/26 | `KAN-40` | Preparing v3 Data (RAG-Aware Dataset) |
| 8 | 26/Aug/26 | `KAN-44` | Running v3 Training + Self-Testing v3 |
| 9 | 27/Aug/26 | `KAN-49` | RAG Integration + Testing |
| 10 | 28/Aug/26 | `KAN-54` | Running v4 Training + Testing (RAG-Aware) |
| 11 | 31/Aug/26 | `KAN-58` | Final Validation Across All Versions |
| 12 | 01/Sep/26 | `KAN-62` | Final Save + Master Report |

### Thisal Sooriyanayaka — Support (Retail lane)

| Day | Date | Issue | Task |
|---|---|---|---|
| 3 | 19/Aug/26 | `KAN-23` | Writing Training Pipelines (Support) |
| 4 | 20/Aug/26 | `KAN-28` | Running v1 Training (Support) |
| 8 | 26/Aug/26 | `KAN-45` | Running v3 Training (Support) |
| 9 | 27/Aug/26 | `KAN-50` | RAG Integration (Support) |

Thisal covers Days 3, 4, 8 and 9 — the pipeline-writing day, two GPU-heavy days, and the
RAG day. Rimaz and Thisal should agree who runs which model to avoid duplicating work.

---

## Known traps, found the hard way

These were hit during Day 1 in the Healthcare lane. All four apply to every lane.

### 1. `train_v4.json` has no day assigned to build it

Day 7 produces v3. Day 10 says "train on `train_v4.json`" — but **nothing creates it**.
This is a gap in the board, not a misreading.

**Do:** build it on Day 9 as v3 plus real retrieved context from your Chroma index.

### 2. bitsandbytes must be 0.46.1 or newer on Colab

Colab ships PyTorch built for CUDA 12.8. **bitsandbytes 0.45.x has no CUDA 12.8 binary** — it
imports without error, reports no problem, and silently has no GPU support. You find out when
training fails.

```bash
pip install -U 'bitsandbytes>=0.46.1'
```

Then **restart the runtime.**

**The nasty part:** constructing a `BitsAndBytesConfig` succeeds even when this is broken,
because it's just a config object. The real test is running an `nf4 Linear4bit` matmul on the
GPU.

### 3. Colab recycles VMs between sessions

Drive data survives. **Installed packages do not.** Coming back to a wall of version-mismatch
errors usually means a fresh VM, not a broken setup — re-run your install cell and restart.

Budget ~5 minutes at the start of every session for: **pull → install → restart → verify**.

### 4. `pip install` then verify in the same cell will mislead you

pip writes the new version to disk, but the already-imported module stays in memory for the
life of the interpreter. Checking straight after installing tests the **old** module and
reports a failure that's already fixed.

**Always restart between installing and verifying.**

### 5. T4 does not support bf16

Use `fp16` as your compute dtype. A bf16 config will fail on Colab's T4.

---

## Open questions

The scope document answered the biggest one — what the models are *for*. These remain.

### Must be resolved before Day 2

**1. Deepana's track: Legal or SME Daily Business?**
The scope document and the board disagree. These are entirely different models with
different data, different risks, and different refusal rules. Building the wrong one is not
recoverable inside twelve days.

**2. Is Thisal supporting Rimaz, or running Manufacturing himself?**
Same conflict. It also determines whether the Retail lane is one broad model or two.

**3. Should Retail and Manufacturing be one model or two?**
Customer-service retail and safety-critical manufacturing have very different accuracy bars.
Merging them means the safety content sets the standard for everything.

### Must be resolved before Day 3

**4. One shared evaluation rubric, or four?**
Management explicitly calls for a shared rubric so all models are judged the same way. Day 3
is when the harness gets written and frozen — after that, changing it means re-scoring
everything.

**5. Is BLEU/ROUGE enough?**
They measure word overlap, not correctness. A confidently wrong answer phrased like the
reference scores well. Each lane likely needs one correctness-based measure suited to its
domain — exact match where answers are checkable, structured manual review otherwise.

**6. What is the shared refusal standard?**
Management wants refusal sets reviewed as a team rather than left to individual judgement.
Agree the format on Day 3 so refusal examples can go into v2 data on Day 5.

### Longer-lived

**7. Do the Qwen and Llama licences permit this use?** Base model licences, separate from
dataset licences.

**8. Does the GPU budget cover Days 4, 8 and 10 across the whole team?** On Colab free tier
this is a live constraint, not a formality.

**9. Who owns hosting, monitoring and data refresh after delivery?** The schedule ends at
Day 12.

**10. English-only, or multilingual?** Called out for healthcare, education and hospitality
in the scope document. Affects data collection from Day 1, so worth answering early even if
the answer is "English-only for the proof of concept".

---

## Getting help

- **Post blockers in the team channel the same day.** Synchronised lanes mean your Day-N
  problem is three other people's Day-N problem.
- **Kusal is available for reviews**, particularly during GPU waits on Days 4, 8 and 10.
- **A negative result is a real result.** If v2 doesn't beat v1, that's a finding — document
  it rather than hiding it.
- **Nobody here has done this before.** Asking early is cheaper than debugging alone for a day.

---

*Reference implementation: `github.com/KusalPabasara/healthcare-llm-finetune` — the Healthcare
lane's scripts, Colab notebook, and environment checks. Reusable across lanes; the only
lane-specific parts are the dataset choices.*
