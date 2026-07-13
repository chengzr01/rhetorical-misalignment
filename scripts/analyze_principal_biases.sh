#!/bin/bash
#
# Annotate cognitive biases in baseline principal responses.
# Requires OPENROUTER_API_KEY and optional BIAS_ANNOTATOR_MODEL.
#
# Usage:
#   bash scripts/analyze_principal_biases.sh [model_key ...] [-- python args]
#
# By default the script annotates DeepSeek principals.  Provide one or more
# model keys (matching scripts/run_baseline.sh) to target other principal
# outputs, or export MODEL_KEYS="deepseek claude".  Any remaining arguments
# (or anything following a literal --) are forwarded to
# pipeline/analyze_principal_biases.py.
#
# Examples:
#   bash scripts/analyze_principal_biases.sh deepseek claude -- --limit 5
#   MODEL_KEYS="llama llama-small" bash scripts/analyze_principal_biases.sh --resume
#   BIAS_ANNOTATOR_MODEL=anthropic/claude-haiku-4.5 bash scripts/analyze_principal_biases.sh --limit 10
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load shared model aliases so callers can reuse MODEL_KEYS from run_baseline.sh
source "$SCRIPT_DIR/model_config.sh"

ANNOTATOR_MODEL="${BIAS_ANNOTATOR_MODEL:-deepseek/deepseek-chat-v3.1}"
ANNOTATION_MAX_WORKERS="${ANNOTATION_MAX_WORKERS:-4}"
RESUME_ANNOTATIONS="${RESUME_ANNOTATIONS:-true}"

# Split positional arguments into model keys and passthrough Python args.
declare -a SELECTED_KEYS=()
declare -a PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      PYTHON_ARGS=("$@")
      break
      ;;
    -*)
      PYTHON_ARGS=("$@")
      break
      ;;
    *)
      SELECTED_KEYS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SELECTED_KEYS[@]} -eq 0 ]]; then
  if [[ -n "${MODEL_KEYS:-}" ]]; then
    read -r -a SELECTED_KEYS <<< "${MODEL_KEYS}"
  else
    SELECTED_KEYS=(
      deepseek
      gpt
      claude
      gemini
      llama
      llama-small
      llama-sft
      llama-dpo
    )
  fi
fi

# Validate model keys against the shared map.
for key in "${SELECTED_KEYS[@]}"; do
  if [[ -z "${MODEL_MAP[$key]:-}" ]]; then
    echo "[ERROR] Unknown model key '${key}'. Available keys: ${!MODEL_MAP[@]}" >&2
    exit 1
  fi
done

# Remove duplicates while preserving order.
declare -A SEEN_KEYS=()
declare -a UNIQUE_KEYS=()
for key in "${SELECTED_KEYS[@]}"; do
  if [[ -z "${SEEN_KEYS[$key]:-}" ]]; then
    SEEN_KEYS[$key]=1
    UNIQUE_KEYS+=("$key")
  fi
done

BASE_INPUT_DIR="$REPO_ROOT/experiments/principals/usmle_sample/baseline"
BASE_OUTPUT_DIR="$REPO_ROOT/experiments/analysis/bias_annotations/usmle_sample/baseline"

for key in "${UNIQUE_KEYS[@]}"; do
  glob="principal_${key}_*.json"
  echo "→ Annotating principals for model '${key}' (glob: ${glob}) using ${ANNOTATOR_MODEL}" >&2
  args=(
    "$REPO_ROOT/pipeline/analyze_principal_biases.py"
    --input-dir "$BASE_INPUT_DIR"
    --output-dir "$BASE_OUTPUT_DIR"
    --glob "$glob"
    --model "$ANNOTATOR_MODEL"
    --max-workers "$ANNOTATION_MAX_WORKERS"
  )
  if [[ "$RESUME_ANNOTATIONS" == "true" ]]; then
    args+=(--resume)
  fi
  if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    args+=("${PYTHON_ARGS[@]}")
  fi
  python "${args[@]}"
done
