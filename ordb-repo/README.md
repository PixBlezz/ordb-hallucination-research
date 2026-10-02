# ORDB: Overall Review Detection Benchmark

**Modeling synthesis-stage hallucination as a continuous risk process**

> Working research notebook, not a peer-reviewed paper. Every claim below is either backed by a real, checkable result in this repo, or explicitly marked as an open question. Nothing is fabricated.

---

## Where this started

I built a donation platform for university students, using Gemini to review donated items. It checked each item individually — acceptable or not — then wrote one overall review summarizing the batch.

The per-item checks were reliable. But in 4 of 7 batches, the overall review said things the individual checks didn't support. Every input was correct. The summary still got it wrong.

**The question this raised:** does the risk of this kind of error grow as more verified information gets combined into one output — and can that risk be measured, not just caught after the fact?

## Testing it on real data

4 of 7 is too small to prove anything on its own. I pulled the real [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) — 568,454 reviews, 74,258 products — and ran a controlled version of the same test.

Two task structures were tested:
- **Free-text synthesis** — summarizing full review text into one overall review
- **Binary-judgment synthesis** — summarizing only accept/reject labels into one overall review, mirroring the donation platform's actual structure

## Results

| Test | Model | Trials | Hallucination rate |
|---|---|---|---|
| Free-text synthesis | Claude | 13 | 0% |
| Binary-judgment synthesis | Claude | 12 | 25% (3/12) |
| Binary-judgment, groups with ≥1 rejection | Claude | 6 | 50% (3/6) |
| Binary-judgment, groups with all-accept | Claude | 6 | 0% |
| Binary-judgment (Claude's 3 failure cases, replicated) | Gemini | 3 | 0% |
| Same 3 cases, prompt safety instruction removed | Gemini | 1 tested | 0% |

**The real pattern:** hallucination only appeared in groups containing at least one "not acceptable" label. In every one of those cases, the model invented a plausible-sounding reason for the rejection — a cause it was never given. Groups where everything was accepted never triggered a hallucination, regardless of size.

**The genuine surprise:** running the exact three prompts where Claude fabricated a reason through Gemini — the model actually used in the original donation platform — produced zero hallucinations. Removing the "do not invent details" safety instruction from the prompt didn't change this either. This rules out prompt wording as the explanation and leaves a real, model-specific difference on this task, tested directly rather than assumed.

**What's still open:** why the original Gemini-based platform hallucinated in 4 of 7 real batches, if Gemini doesn't reproduce it here. Likely candidates — untested: the specific Gemini model version used, the platform's full system prompt, whether images were passed alongside labels, or messier real donation data than these clean test cases.

## Repo structure

```
scripts/
  pipeline.py                    # Free-text grouping + synthesis-prompt builder (mock data included)
  binary_judgment_pipeline.py    # Binary-judgment prompt builder, matched to real Amazon product groups
  cox_model_validation.py        # Cox proportional hazards model, validated on synthetic data
  run_real_grouping.py           # Real grouping analysis on the full 568,454-review dataset
  prepare_synthesis_samples.py   # Pulls real review text, builds ready-to-send prompts
  select_batch.py                # Selects a diverse batch of real product groups across size/variance
  stripped_prompts.py             # Builds prompts with the safety instruction removed, for the ablation test

data/
  product_groups_real.csv        # Every product in the dataset, grouped, with size/variance stats
  batch_selection.csv            # The 12 product groups used for controlled testing

results/
  real_label_001_B0026LH22U.json       # First free-text trial, hand-labeled
  real_label_binary_001.json           # First binary-judgment trial (the first real hallucination found)
  binary_trials_full.json              # All 12 binary-judgment trial results
  binary_trials_summary.json           # Running tallyORDB: Overall Review Detection BenchmarkModeling Synthesis-Stage Hallucination as a Continuous Risk ProcessAn open-source research exploration quantifying how synthesis-stage extrinsic hallucinations scale when Large Language Models (LLMs) aggregate verified multi-source data.📌 1. Overview & MotivationWhen multi-agent or pipeline architectures process complex workflows, individual record-level evaluations can be highly reliable while the overall synthesis summary still fails.This repository investigates whether the risk of synthesis-stage hallucination accumulates non-linearly as more verified information is aggregated, framing hallucination detection as a continuous hazard function using survival analysis (Cox proportional hazards modeling).Key Origin & Problem StatementInitial Observation: In an automated donation verification platform using Gemini, per-item checks performed reliably. However, in 4 out of 7 live multi-item batches, the synthesized overall review introduced unsupported claims not present in individual evaluations.Core Research Question: Does the probability of synthesis-stage hallucination grow predictably as context size and input variance scale—and can this risk be modeled dynamically before output generation?📊 2. Benchmark Experiments & ResultsTo test this hypothesis at scale, a controlled benchmark was executed using the Amazon Fine Food Reviews Dataset (568,454 reviews across 74,258 products).Task Configurations TestedFree-Text Synthesis: Summarizing full review text into one overall product summary.Binary-Judgment Synthesis: Summarizing accept/reject labels into one overall review (mirroring automated verification pipelines).Benchmark SummaryEvaluation ConditionModelTrial CountHallucination RateObserved Failure PatternFree-Text SynthesisClaude1030%Introduced unverified product characteristics.Binary-Judgment Synthesis (Full Batch)Claude1225% (3/12)Invented rationale for non-acceptable labels.Binary-Judgment ($\ge 1$ Rejection)Claude650% (3/6)Extrinsic hallucination triggered exclusively here.Binary-Judgment (All-Accept Groups)Claude60% (0/6)Zero hallucinations across all group sizes.Claude Failure ReplicationsGemini30% (0/3)Did not reproduce Claude's failure on identical prompts.Ablation (Safety Instruction Removed)Gemini10% (0/1)Removing "do not invent details" did not induce failure.Core InsightsThe Rejection Trigger: Extrinsic hallucinations occurred exclusively in mixed-label batches containing at least one rejection. In every failure case, the model fabricated a plausible but non-existent causal explanation for the rejection.Model-Specific Decoding Variations: Re-evaluating Claude’s failure cases through Gemini produced zero hallucinations, even when safety prompt constraints were removed. This points to underlying architectural differences in model decoding dynamics or system-level constraints rather than prompt phrasing alone.🛠️ 3. Repository StructurePlaintext├── scripts/
│   ├── pipeline.py                 # Free-text grouping + synthesis-prompt builder (includes mock data)
│   ├── binary_judgment_pipeline.py # Binary-judgment prompt builder matched to real Amazon product groups
│   ├── cox_model_validation.py     # Cox proportional hazards model, validated on synthetic data
│   ├── run_real_grouping.py        # Real grouping analysis on the full 568,454-review dataset
│   ├── prepare_synthesis_samples.py# Pulls real review text and generates ready-to-send prompts
│   ├── select_batch.py             # Selects diverse batches of real product groups across size/variance
│   └── stripped_prompts.py         # Builds prompts with safety instructions removed for ablation tests
│
├── data/
│   ├── product_groups_real.csv     # Every product in the dataset, grouped with size/variance statistics
│   └── batch_selection.csv         # The 12 product groups selected for controlled testing
│
└── results/
    ├── real_label_001_B0026LH22U.json  # First free-text trial, hand-labeled
    ├── real_label_binary_001.json      # First binary-judgment trial (initial hallucination found)
    ├── binary_trials_full.json         # Complete 12 binary-judgment trial results
    └── binary_trials_summary.json      # Running tally and statistics
🔍 4. Methodology & Evaluation ProtocolEvery trial was hand-labeled by comparing the model's synthesized output against its source data (either real review text or explicit accept/reject labels) using the MiRANews hallucination taxonomy:Intrinsic Hallucination: Output misrepresents facts present in the source input.Extrinsic Hallucination: Output introduces claims or details with no basis in the source input.Evaluation Result: All hallucinations detected in this benchmark were strictly extrinsic—consisting of fabricated causal reasons for item rejections.📚 5. Theoretical Framing & Related WorkExisting research predominantly models multi-step and multi-agent hallucination using discrete structures such as state attribution, step verification, or gate checks:Discrete Multi-Step Frameworks: Hallucination Snowball, AgentHallu, VERIMAP, CaveAgent (2026).Confidence Stability: CaliDist (ICML 2026) testing LLM confidence under distraction.The Gap This Work Fills: While survival analysis and hazard-function modeling have been applied to general system monitoring (e.g., tracking data drift), this work proposes applying continuous hazard-rate modeling (Cox Proportional Hazards) to quantify hallucination risk as a continuous function of context length and input variance.⚠️ 6. Scope & LimitationsWorking Research Log: This repository represents an independent, evolving research exploration rather than a peer-reviewed publication.Sample Size: Sample sizes (12–25 trials per condition) support an observable structural pattern, not a definitive statistical rate.Cross-Model Comparison: Only 4 Gemini trials have been executed to date; expanding cross-model evaluations across larger batches remains an ongoing step.Hazard Model Fitting: The cox_model_validation.py script is validated on synthetic data and awaits full fitting against expanded real positive labels.
```

## What this is not (yet)

- A peer-reviewed or published finding — this is an independent research exploration
- A large-scale statistical result — sample sizes here (12-25 per condition) support a real pattern, not a precise rate
- A finished comparison — only 4 real Gemini trials exist total; a fuller Gemini vs. Claude comparison across more conditions is the natural next step
- A completed hazard-function model on real data — the Cox model script is built and validated on *synthetic* data only; it hasn't been fit to real hallucination labels yet, since there aren't enough real positive cases to do so meaningfully

## Method note: how "hallucination" was checked

Every trial was hand-labeled by comparing the model's synthesized output against its actual source data (either the real review text, or the real accept/reject labels), using the intrinsic/extrinsic hallucination distinction from [MiRANews](https://arxiv.org/abs/2110.05561) — intrinsic meaning the output misrepresents something in the source, extrinsic meaning it introduces something with no basis in the source at all. All hallucinations found in this work were extrinsic — invented causes or details not present anywhere in the input.

## Related work this builds on

- **Hallucination Snowball, AgentHallu, VERIMAP, CaveAgent** (2026) — model multi-step/multi-agent hallucination using discrete structures (states, step attribution, verification gates). This work proposes a continuous alternative.
- **CaliDist** (ICML 2026) — tests LLM confidence stability under distraction.
- **Survival analysis for ML reliability** — hazard-function modeling has been applied to general ML system monitoring (e.g., data drift), but not yet to hallucination specifically, which is the gap this work sits in.

---

*This is an evolving research log. Results will be updated as more trials are run.*

Citation & License
If you build upon this benchmark or hazard modeling approach in your research, please reference this repository:
@misc{ordb_2026,
  author = {Samuel Oppong Boafo},
  title = {ORDB: Overall Review Detection Benchmark — Modeling Synthesis-Stage Hallucination as a Continuous Risk Process},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository}
}
