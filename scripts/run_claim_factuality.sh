#!/bin/bash
#
# Run claim factuality validation experiments on curated USMLE claim sets.
# This script optionally curates claims via an OpenRouter verifier and then
# evaluates a target model's ability to classify those claims as factual or
# unfactual.
#
# Usage:
#   bash scripts/run_claim_factuality.sh [model_key] [aggregated_key]
#
#   model_key:       entry in scripts/model_config.sh (e.g., deepseek, gpt)
#   aggregated_key:  identifier for aggregated claims.
#                    - If a file exists at experiments/aggregation/aggregated_<key>.json,
#                      it will be used automatically.
#                    - Otherwise the argument is treated as a path to a JSON file.
#
# Environment variables:
#   CURATION_MODEL             OpenRouter model for curation (default: $CLAIM_CURATION_MODEL or deepseek/deepseek-chat-v3.1)
#   CURATION_PROMPT            Prompt YAML for curation (default: prompts/experiments/claim_curation.yaml)
#   CURATION_TEMPERATURE       Sampling temperature for curation (default: 0.0)
#   CURATION_MAX_WORKERS       Parallel workers for curation (default: 4)
#   CURATION_MAX_CASES         Limit number of cases (default: none)
#   CURATION_MAX_CLAIMS        Limit number of claims per case (default: none)
#   SKIP_CURATION              If true, skip curation step
#
#   BACKEND                    Evaluation backend (openrouter|nvidia|sglang, default: openrouter)
#   TEMPERATURE                Evaluation sampling temperature (default: 0.0)
#   MAX_WORKERS                Evaluation parallel workers (default: 8)
#   PROMPT_FILE                Evaluation prompt file (default: prompts/experiments/claim_factuality.yaml)
#   SGLANG_PORT                Port for sglang backend (default: 30000)
#   SGLANG_BASE_URL            Base URL for sglang backend (default: http://127.0.0.1)
#   SKIP_EVALUATION            If true, skip evaluation step
#   OUTPUT_DIR                 Directory for evaluation outputs (default: experiments/tests/claim_factuality/<aggregated_key>)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/model_config.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_claim_factuality.sh [model_key] [aggregated_key]" >&2
  exit 1
fi

MODEL_KEY="$1"
AGGREGATED_KEY="${2:-gemini_factual}"  # default aggregated key

MODEL="${MODEL_MAP[$MODEL_KEY]:-}"
if [[ -z "$MODEL" ]]; then
  echo "Error: Unknown model key '$MODEL_KEY'" >&2
  echo "Available keys: ${!MODEL_MAP[@]}" >&2
  exit 1
fi

# Resolve aggregated file path
AGGREGATED_FILE=""
if [[ -f "$REPO_ROOT/experiments/aggregation/aggregated_${AGGREGATED_KEY}.json" ]]; then
  AGGREGATED_FILE="$REPO_ROOT/experiments/aggregation/aggregated_${AGGREGATED_KEY}.json"
elif [[ -f "$AGGREGATED_KEY" ]]; then
  AGGREGATED_FILE="$AGGREGATED_KEY"
elif [[ -f "$REPO_ROOT/$AGGREGATED_KEY" ]]; then
  AGGREGATED_FILE="$REPO_ROOT/$AGGREGATED_KEY"
else
  echo "Error: Could not locate aggregated claims for key '$AGGREGATED_KEY'" >&2
  exit 1
fi

QUESTIONS_FILE="$REPO_ROOT/experiments/questions/clinical_questions_usmle_sample.json"
if [[ ! -f "$QUESTIONS_FILE" ]]; then
  echo "Error: Questions file not found at $QUESTIONS_FILE" >&2
  exit 1
fi

CLAIMS_DIR="$REPO_ROOT/experiments/claims/usmle_sample"
EVAL_BASE_DIR="$REPO_ROOT/experiments/tests/claim_factuality/${AGGREGATED_KEY}"
mkdir -p "$CLAIMS_DIR" "$EVAL_BASE_DIR"

CURATED_FILE="$CLAIMS_DIR/curated_${AGGREGATED_KEY}.json"
CURATION_MODEL="${CURATION_MODEL:-${CLAIM_CURATION_MODEL:-deepseek/deepseek-chat-v3.1}}"
CURATION_PROMPT="${CURATION_PROMPT:-prompts/experiments/claim_curation.yaml}"
CURATION_TEMPERATURE="${CURATION_TEMPERATURE:-0.0}"
CURATION_MAX_WORKERS="${CURATION_MAX_WORKERS:-4}"
CURATION_MAX_CASES="${CURATION_MAX_CASES:-}"  # optional
CURATION_MAX_CLAIMS="${CURATION_MAX_CLAIMS:-}"  # optional
SKIP_CURATION="${SKIP_CURATION:-false}"

BACKEND="${BACKEND:-openrouter}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PROMPT_FILE="${PROMPT_FILE:-prompts/experiments/claim_factuality.yaml}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1}"
SKIP_EVALUATION="${SKIP_EVALUATION:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_BASE_DIR}"

echo "============================================="
echo "Claim Factuality Experiment"
echo "============================================="
echo "Model key:          $MODEL_KEY"
echo "Model identifier:   $MODEL"
echo "Aggregated source:  $AGGREGATED_FILE"
echo "Curated file:       $CURATED_FILE"
echo "Evaluation backend: $BACKEND"
echo "Evaluation prompt:  $PROMPT_FILE"
echo "============================================="

if [[ "$SKIP_CURATION" != "true" ]]; then
  echo "-- Curating claims via $CURATION_MODEL --"
  CURATION_ARGS=(
    "pipeline/curate_claims_via_llm.py"
    "--aggregated-info" "$AGGREGATED_FILE"
    "--questions" "$QUESTIONS_FILE"
    "--output" "$CURATED_FILE"
    "--model" "$CURATION_MODEL"
    "--prompt" "$CURATION_PROMPT"
    "--temperature" "$CURATION_TEMPERATURE"
    "--max-workers" "$CURATION_MAX_WORKERS"
    "--resume"
  )
  if [[ -n "$CURATION_MAX_CASES" ]]; then
    CURATION_ARGS+=("--max-cases" "$CURATION_MAX_CASES")
  fi
  if [[ -n "$CURATION_MAX_CLAIMS" ]]; then
    CURATION_ARGS+=("--max-claims-per-case" "$CURATION_MAX_CLAIMS")
  fi
  (cd "$REPO_ROOT" && python "${CURATION_ARGS[@]}")
else
  echo "Skipping curation (SKIP_CURATION=true)"
fi

if [[ "$SKIP_EVALUATION" = "true" ]]; then
  echo "Skipping evaluation (SKIP_EVALUATION=true)"
  exit 0
fi

if [[ ! -f "$CURATED_FILE" ]]; then
  echo "Error: Curated file not found at $CURATED_FILE" >&2
  exit 1
fi

echo "-- Evaluating $MODEL_KEY on curated claims --"
EVAL_ARGS=(
  "pipeline/test_claim_factuality.py"
  "--claims" "$CURATED_FILE"
  "--model" "$MODEL"
  "--backend" "$BACKEND"
  "--prompt" "$PROMPT_FILE"
  "--temperature" "$TEMPERATURE"
  "--max-workers" "$MAX_WORKERS"
  "--output-dir" "$OUTPUT_DIR"
)
if [[ "$BACKEND" == "sglang" ]]; then
  EVAL_ARGS+=("--sglang-port" "$SGLANG_PORT" "--sglang-base-url" "$SGLANG_BASE_URL")
fi

(cd "$REPO_ROOT" && python "${EVAL_ARGS[@]}")

echo "============================================="
echo "Factuality experiment complete."
RESULT_PATH=$(ls -t "$OUTPUT_DIR"/test_claim_factuality_* 2>/dev/null | head -n 1 || true)
if [[ -n "$RESULT_PATH" ]]; then
  echo "Latest results: $RESULT_PATH"
fi
echo "============================================="
