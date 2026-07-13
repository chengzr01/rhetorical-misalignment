#!/bin/bash
#
# End-to-end pipeline to curate decision problems, generate bias-aligned
# representations, and evaluate rational vs behavioral principals.
#
# Usage:
#   bash scripts/run_decision_validation.sh
#
# Environment customisation (optional):
#   CURATION_QUESTIONS_PATH   – input clinical questions JSON
#   CURATION_OUTPUT           – curated decision problem cache
#   CURATION_MODEL            – OpenRouter model for curation (default anthropic/claude-haiku-4-5)
#   CURATION_TEMPERATURE      – sampling temperature (default 0.2)
#   CURATION_MAX_PROBLEMS     – limit number of cases (empty = all)
#   CURATION_START_INDEX      – 0-based offset into questions (default 0)
#   CURATION_SAVE_INTERVAL    – save every N cases (default 5)
#   CURATION_OVERWRITE        – set to "true" to discard existing cache
#
#   REPRESENTATION_OUTPUT     – bias representation cache
#   REPRESENTATION_MODEL      – OpenRouter model for representations (default deepseek/deepseek-chat-v3.1)
#   REPRESENTATION_TEMPERATURE
#   REPRESENTATION_MAX_CASES
#   REPRESENTATION_START_INDEX
#   REPRESENTATION_SAVE_INTERVAL
#   REPRESENTATION_OVERWRITE
#   REPRESENTATION_BIASES     – space-separated list of bias names to generate (optional)
#
#   VALIDATION_OUTPUT         – evaluation results JSON
#   VALIDATION_MODEL          – OpenRouter model for principals (default deepseek/deepseek-chat-v3.1)
#   VALIDATION_TEMPERATURE
#   VALIDATION_MAX_CASES
#   VALIDATION_START_INDEX
#   VALIDATION_SAVE_INTERVAL
#   VALIDATION_OVERWRITE
#   VALIDATION_BIASES         – space-separated style labels to score (optional)
#
# All steps require OPENROUTER_API_KEY to be set in the environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── Defaults ──────────────────────────────────────────────────────────────

CURATION_QUESTIONS_PATH="${CURATION_QUESTIONS_PATH:-experiments/questions/clinical_questions_usmle_sample.json}"
CURATION_OUTPUT="${CURATION_OUTPUT:-experiments/decision_problems/usmle_rhetorical_decisions.json}"
CURATION_MODEL="${CURATION_MODEL:-anthropic/claude-haiku-4-5}"
CURATION_TEMPERATURE="${CURATION_TEMPERATURE:-0.2}"
CURATION_SAVE_INTERVAL="${CURATION_SAVE_INTERVAL:-5}"
CURATION_THREADS="${CURATION_THREADS:-16}"

REPRESENTATION_OUTPUT="${REPRESENTATION_OUTPUT:-experiments/decision_problems/usmle_bias_representations.json}"
REPRESENTATION_MODEL="${REPRESENTATION_MODEL:-deepseek/deepseek-chat-v3.1}"
REPRESENTATION_TEMPERATURE="${REPRESENTATION_TEMPERATURE:-0.2}"
REPRESENTATION_SAVE_INTERVAL="${REPRESENTATION_SAVE_INTERVAL:-5}"
REPRESENTATION_THREADS="${REPRESENTATION_THREADS:-16}"

SENSITIVITY_OUTPUT="${SENSITIVITY_OUTPUT:-experiments/analysis/bias_sensitivity_summary.json}"
SENSITIVITY_FILTERED_REPRESENTATIONS="${SENSITIVITY_FILTERED_REPRESENTATIONS:-experiments/decision_problems/usmle_bias_representations_filtered.json}"
SENSITIVITY_MODEL="${SENSITIVITY_MODEL:-deepseek/deepseek-chat-v3.1}"
SENSITIVITY_TEMPERATURE="${SENSITIVITY_TEMPERATURE:-0.2}"
SENSITIVITY_TOP_CASES="${SENSITIVITY_TOP_CASES:-60}"
SENSITIVITY_THREADS="${SENSITIVITY_THREADS:-16}"

VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-experiments/analysis/decision_maker_validation_openended.json}"
VALIDATION_MODEL="${VALIDATION_MODEL:-deepseek/deepseek-chat-v3.1}"
VALIDATION_TEMPERATURE="${VALIDATION_TEMPERATURE:-0.2}"
VALIDATION_SAVE_INTERVAL="${VALIDATION_SAVE_INTERVAL:-5}"
VALIDATION_REPRESENTATIONS="${VALIDATION_REPRESENTATIONS:-$SENSITIVITY_FILTERED_REPRESENTATIONS}"
VALIDATION_THREADS="${VALIDATION_THREADS:-16}"
VALIDATION_RATIONAL_PROMPT="${VALIDATION_RATIONAL_PROMPT:-prompts/principal/bayesian_belief_openended.yaml}"
VALIDATION_BEHAVIORAL_PROMPT="${VALIDATION_BEHAVIORAL_PROMPT:-prompts/principal/behavioral_belief_openended.yaml}"

ANNOTATION_INPUT="${ANNOTATION_INPUT:-$VALIDATION_OUTPUT}"
ANNOTATION_OUTPUT="${ANNOTATION_OUTPUT:-experiments/analysis/principal_response_annotations.json}"
ANNOTATION_MODEL="${ANNOTATION_MODEL:-deepseek/deepseek-chat-v3.1}"
ANNOTATION_TEMPERATURE="${ANNOTATION_TEMPERATURE:-0.2}"
ANNOTATION_THREADS="${ANNOTATION_THREADS:-16}"
ANNOTATION_CACHE="${ANNOTATION_CACHE:-experiments/analysis/principal_response_annotations_cache.json}"

# ── Step 1: Curate decision problems ─────────────────────────────────────

log "Step 1/5: Curating decision problems"

declare -a CURATE_ARGS=(
  "--questions-path" "$CURATION_QUESTIONS_PATH"
  "--output" "$CURATION_OUTPUT"
  "--model" "$CURATION_MODEL"
  "--temperature" "$CURATION_TEMPERATURE"
  "--save-interval" "$CURATION_SAVE_INTERVAL"
  "--threads" "$CURATION_THREADS"
)

if [ -n "${CURATION_MAX_PROBLEMS:-}" ]; then
  CURATE_ARGS+=("--max-problems" "$CURATION_MAX_PROBLEMS")
fi
if [ -n "${CURATION_START_INDEX:-}" ]; then
  CURATE_ARGS+=("--start-index" "$CURATION_START_INDEX")
fi
if [ "${CURATION_OVERWRITE:-false}" = "true" ]; then
  CURATE_ARGS+=("--overwrite")
fi
if [ -n "${CURATION_CACHE:-}" ]; then
  CURATE_ARGS+=("--cache-path" "$CURATION_CACHE")
fi

python pipeline/curate_decision_problems.py "${CURATE_ARGS[@]}"

# ── Step 2: Generate bias representations ────────────────────────────────

log "Step 2/5: Generating bias-aligned representations"

declare -a REPRESENT_ARGS=(
  "--decision-problems" "$CURATION_OUTPUT"
  "--output" "$REPRESENTATION_OUTPUT"
  "--prompt-path" "prompts/experiments/generate_bias_representations.yaml"
  "--bias-prompt" "prompts/principal/behavioral_belief.yaml"
  "--model" "$REPRESENTATION_MODEL"
  "--temperature" "$REPRESENTATION_TEMPERATURE"
  "--save-interval" "$REPRESENTATION_SAVE_INTERVAL"
  "--threads" "$REPRESENTATION_THREADS"
)

if [ -n "${REPRESENTATION_MAX_CASES:-}" ]; then
  REPRESENT_ARGS+=("--max-cases" "$REPRESENTATION_MAX_CASES")
fi
if [ -n "${REPRESENTATION_START_INDEX:-}" ]; then
  REPRESENT_ARGS+=("--start-index" "$REPRESENTATION_START_INDEX")
fi
if [ "${REPRESENTATION_OVERWRITE:-false}" = "true" ]; then
  REPRESENT_ARGS+=("--overwrite")
fi
if [ -n "${REPRESENTATION_BIASES:-}" ]; then
  # shellcheck disable=SC2206
  REPRESENT_ARGS+=("--biases" ${REPRESENTATION_BIASES})
fi
if [ -n "${REPRESENTATION_CACHE:-}" ]; then
  REPRESENT_ARGS+=("--cache-path" "$REPRESENTATION_CACHE")
fi

python pipeline/generate_bias_representations.py "${REPRESENT_ARGS[@]}"

# ── Step 3: Score bias sensitivity and filter cases ───────────────────────

log "Step 3/5: Scoring bias sensitivity for bias representations"

declare -a SENSITIVITY_ARGS=(
  "--decision-problems" "$CURATION_OUTPUT"
  "--representations" "$REPRESENTATION_OUTPUT"
  "--summary-output" "$SENSITIVITY_OUTPUT"
  "--filtered-representations" "$SENSITIVITY_FILTERED_REPRESENTATIONS"
  "--model" "$SENSITIVITY_MODEL"
  "--temperature" "$SENSITIVITY_TEMPERATURE"
  "--top-cases" "$SENSITIVITY_TOP_CASES"
  "--threads" "$SENSITIVITY_THREADS"
)

if [ -n "${SENSITIVITY_MAX_CASES:-}" ]; then
  SENSITIVITY_ARGS+=("--max-cases" "$SENSITIVITY_MAX_CASES")
fi
if [ -n "${SENSITIVITY_START_INDEX:-}" ]; then
  SENSITIVITY_ARGS+=("--start-index" "$SENSITIVITY_START_INDEX")
fi
if [ "${SENSITIVITY_OVERWRITE:-false}" = "true" ]; then
  SENSITIVITY_ARGS+=("--overwrite")
fi
if [ -n "${SENSITIVITY_CACHE:-}" ]; then
  SENSITIVITY_ARGS+=("--cache-path" "$SENSITIVITY_CACHE")
fi

python pipeline/evaluate_bias_sensitivity.py "${SENSITIVITY_ARGS[@]}"

# ── Step 4: Validate decision makers ─────────────────────────────────────

log "Step 4/5: Evaluating rational vs behavioral principals"

declare -a VALIDATE_ARGS=(
  "--decision-problems" "$CURATION_OUTPUT"
  "--representations" "$VALIDATION_REPRESENTATIONS"
  "--output" "$VALIDATION_OUTPUT"
  "--rational-prompt" "$VALIDATION_RATIONAL_PROMPT"
  "--behavioral-prompt" "$VALIDATION_BEHAVIORAL_PROMPT"
  "--model" "$VALIDATION_MODEL"
  "--temperature" "$VALIDATION_TEMPERATURE"
  "--save-interval" "$VALIDATION_SAVE_INTERVAL"
  "--threads" "$VALIDATION_THREADS"
)

if [ -n "${VALIDATION_MAX_CASES:-}" ]; then
  VALIDATE_ARGS+=("--max-cases" "$VALIDATION_MAX_CASES")
fi
if [ -n "${VALIDATION_START_INDEX:-}" ]; then
  VALIDATE_ARGS+=("--start-index" "$VALIDATION_START_INDEX")
fi
if [ "${VALIDATION_OVERWRITE:-false}" = "true" ]; then
  VALIDATE_ARGS+=("--overwrite")
fi
if [ -n "${VALIDATION_BIASES:-}" ]; then
  # shellcheck disable=SC2206
  VALIDATE_ARGS+=("--biases" ${VALIDATION_BIASES})
fi
if [ -n "${VALIDATION_CACHE:-}" ]; then
  VALIDATE_ARGS+=("--cache-path" "$VALIDATION_CACHE")
fi

python pipeline/validate_decision_makers.py "${VALIDATE_ARGS[@]}"

log "Step 5/5: Annotating principal responses"

declare -a ANNOTATE_ARGS=(
  "--input" "$ANNOTATION_INPUT"
  "--output" "$ANNOTATION_OUTPUT"
  "--model" "$ANNOTATION_MODEL"
  "--temperature" "$ANNOTATION_TEMPERATURE"
  "--threads" "$ANNOTATION_THREADS"
)

if [ -n "${ANNOTATION_START_INDEX:-}" ]; then
  ANNOTATE_ARGS+=("--start-index" "$ANNOTATION_START_INDEX")
fi
if [ -n "${ANNOTATION_MAX_RECORDS:-}" ]; then
  ANNOTATE_ARGS+=("--max-records" "$ANNOTATION_MAX_RECORDS")
fi
if [ "${ANNOTATION_OVERWRITE:-false}" = "true" ]; then
  ANNOTATE_ARGS+=("--overwrite")
fi
if [ -n "${ANNOTATION_CACHE:-}" ]; then
  ANNOTATE_ARGS+=("--cache-path" "$ANNOTATION_CACHE")
fi

python pipeline/annotate_principal_responses.py "${ANNOTATE_ARGS[@]}"

log "Pipeline complete. Validation results: ${VALIDATION_OUTPUT} | Bias sensitivity summary: ${SENSITIVITY_OUTPUT} | Annotation file: ${ANNOTATION_OUTPUT}"
