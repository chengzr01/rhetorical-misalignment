# Pipeline

Pre-processing and analysis scripts that support the main experiment loop
(`core/`). Scripts are grouped below by the stage they belong to.

---

## Dataset Preparation

### `parse_usmle_sample.py`
Parses raw USMLE plain-text question/answer files into structured JSON.
Extracts question stems, multiple-choice options (A–E), correct answer,
exam level (Step 1/2/3), and auto-generated MetaMap phrases.

- **Input:** raw `.txt` question and answer files
- **Output:** structured JSON list consumed by `generate_questions.py`

### `verify_usmle_parsing.py`
Quality-check script for parsed USMLE data. Reports duplicate question
numbers, missing required fields, and answer/option mismatches.

- **Input:** parsed USMLE JSON from `parse_usmle_sample.py`
- **Output:** console report only (no file written)

### `generate_questions.py`
Converts a parsed USMLE JSON into the standard clinical-questions format
used by all downstream scripts (`clinical_questions_usmle_sample.json`).
Adds a stable `id` field and filters out questions with missing answers.

- **Input:** parsed USMLE JSON
- **Output:** `experiments/questions/clinical_questions_*.json`

---

## Model Evaluation (Baseline)

### `test_usmle_sample.py`
Evaluates one LLM directly on USMLE questions: sends each question +
options to the model and collects its answer choice and (optionally) its
confidence/belief score. Supports OpenRouter, NVIDIA, and SGLang backends.
Results go to `experiments/tests/`.

- **Input:** `experiments/questions/clinical_questions_usmle_sample.json`
- **Output:** `experiments/tests/test_usmle_sample_<model>_belief.json`
- **Invoked by:** `scripts/run_test.sh`

### `annotate_no_answer.py`
Post-processes a test-results file for cases where the model's response
could not be parsed into an A–E letter. Uses DeepSeek-V3.1 as a fallback
judge to infer the intended answer from the raw response. Updates the
results file in-place.

- **Input/Output:** `experiments/tests/test_usmle_sample_<model>_belief.json` (updated in-place)

---

## Factualness Analysis

### `analyze_information_factualness.py`
LLM-judge analysis of agent information quality. For each case in an
`agent_*.json` file, an LLM (default: DeepSeek-V3.1) breaks the agent's
free-text response into individual claims and labels each as **factual**,
**unfactual**, or **uncertain**, with a one-sentence explanation.

- **Input:** `experiments/agents/usmle_sample/agent_<model>.json`
- **Output:** `experiments/information/factualness_agent_<model>.json`
- **Default dirs:** `experiments/agents/usmle_sample/`, `experiments/information/`
- **Modes:** `analyze` · `check` · `summarize`

### `aggregate_information.py`
Combines factualness results from one or more agents, filtering to the
desired claim labels (default: `factual`). Cases can be the union or
intersection across agents. Produces the `aggregated_*.json` format used
by framing and information-design experiments.

- **Input:** `experiments/information/factualness_agent_<model>.json` (one per agent)
- **Output:** `experiments/aggregation/aggregated_<label>.json`
- **Default dirs:** `experiments/information/`, `experiments/aggregation/`

### `synthesize_factual_information.py`
Converts factualness results into ready-to-use agent-cache files for
principal inference. For each model, produces three files:

| File | Content |
|------|---------|
| `agent_<model>_factual.json` | only factual claims |
| `agent_<model>_unfactual.json` | only unfactual + uncertain claims |
| `agent_<model>_all_claims.json` | all claims combined |

The `information` field in each record contains claims formatted as bullet
points (or numbered list / plain, configurable). These files can be passed
directly to `core/principal_inference.py` via `--agent-cache`.

- **Input:** `experiments/information/factualness_agent_<model>.json` + `experiments/agents/usmle_sample/agent_<model>.json`
- **Output:** `experiments/agents/usmle_sample/agent_<model>_{factual,unfactual,all_claims}.json`
- **Invoked by:** `scripts/run_comparison.sh`

---

## Claim Preparation for Framing / Information-Design Experiments

### `neutralize_claims.py`
Rewrites claims in an `aggregated_*.json` file in neutral, model-agnostic
clinical language. Removes source-model style, hedging, and attribution
phrasing while preserving all factual content. Adds an `original_claim`
field to each claim dict. Useful when using a specific model's claims
(e.g. Gemini-2.5-Pro) as ground truth: neutralizing removes stylistic
confounds so that experimental effects reflect *what* is said, not *how*
the source model said it.

- **Input:** `aggregated_*.json` (from `aggregate_information.py` or `extract_agent_claims.py`)
- **Output:** same schema with claims rewritten (e.g. `aggregated_gemini_factual_neutral.json`)
- **Resumes** automatically from partial output
- **Invoked by:** `scripts/prepare_ground_truth.sh`

### `extract_agent_claims.py`
Splits a framing response file into individual verbatim claims using an
LLM. The LLM only segments — it does not rewrite — so claims appear exactly
as the framing agent wrote them. Outputs in `aggregated_*.json` format for
use in the martingale framing test.

- **Input:** `experiments/agents/usmle_sample/framing_<agent>_gt_factual_agg.json`
- **Output:** `experiments/aggregation/aggregated_<agent>_framing.json`
- **Invoked by:** `scripts/run_martingale_framing.sh`

### `information_aggregation.py`
Converts an `aggregated_*.json` file directly into the flat agent-cache
list format (`agent_inference.py` output schema) so it can be used as
`--ground-truth` for `agent_presentation.py` or `--agent-cache` for
`principal_inference.py` without going through an agent framing step.

- **Input:** `aggregated_*.json` + `clinical_questions_*.json` (for metadata)
- **Output:** `experiments/agents/usmle_sample/agent_<name>.json`

---

## Bayesian Reliability Tests

### `generate_permutations.py`
Generates K random orderings of each case's claims for the martingale
reliability test. Produces one record per (case, permutation), all in
agent-cache format ready for `core/principal_inference.py`.

- **Input:** `aggregated_*.json` + `clinical_questions_*.json`
- **Output:** `experiments/agents/usmle_sample/martingale_permutations_k<K>.json`
- **Invoked by:** `scripts/run_martingale.sh` (Stage 1)

### `generate_paraphrases.py`
Generates K LLM-rewritten paraphrases of each case's claims for the
sufficient-statistics test. Each paraphrase is an independent fluent-prose
rewrite that preserves all clinical facts. Produces K+1 records per case
(1 original + K paraphrases), all in agent-cache format.

- **Input:** `aggregated_*.json` + `clinical_questions_*.json`
- **Output:** `experiments/agents/usmle_sample/paraphrase_records_<model>_k<K>.json`
- **Invoked by:** `scripts/run_paraphrase.sh` (Stage 1)

### `compute_martingale.py`
Analyzes principal inference results from a permutation or paraphrase
experiment. Groups results by `case_id` and computes: majority-vote answer,
answer consistency across samples, mean/std of confidence beliefs, and
whether the majority answer is correct. Reports aggregate accuracy
improvement from majority voting vs. single-run.

- **Input:** `experiments/principals/usmle_sample/principal_martingale_*_bayesian_martingale_choices.json`
- **Output:** `experiments/principals/usmle_sample/martingale_analysis_*.json`
- **Invoked by:** `scripts/run_martingale.sh` (Stage 3), `scripts/run_paraphrase.sh` (Stage 3)

---

## Data Flow Summary

```
Raw USMLE text
  └─ parse_usmle_sample.py
  └─ verify_usmle_parsing.py
  └─ generate_questions.py
        └─► clinical_questions_usmle_sample.json

clinical_questions_*.json
  ├─ test_usmle_sample.py ──────────────────────────────────────────► tests/test_*.json
  │     └─ annotate_no_answer.py  (post-process unparseable answers)
  │
  └─ [core/agent_inference.py]
        └─► agents/usmle_sample/agent_<model>.json
              │
              ├─ analyze_information_factualness.py
              │     └─► information/factualness_agent_<model>.json
              │               │
              │               ├─ aggregate_information.py
              │               │     └─► aggregation/aggregated_<label>.json
              │               │               │
              │               │               ├─ neutralize_claims.py
              │               │               │     └─► aggregation/aggregated_*_neutral.json
              │               │               │
              │               │               ├─ generate_permutations.py ──► martingale test
              │               │               ├─ generate_paraphrases.py  ──► paraphrase test
              │               │               └─ information_aggregation.py ─► agent cache
              │               │
              │               └─ synthesize_factual_information.py
              │                     └─► agents/usmle_sample/agent_<model>_{factual,unfactual,all_claims}.json
              │
              └─ extract_agent_claims.py  (from framing outputs)
                    └─► aggregation/aggregated_<agent>_framing.json
                              └─ generate_permutations.py ──► martingale framing test
```
