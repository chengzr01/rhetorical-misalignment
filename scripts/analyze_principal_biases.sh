#!/bin/bash
#
# Annotate cognitive biases in baseline principal responses.
# Requires OPENROUTER_API_KEY and optional BIAS_ANNOTATOR_MODEL.
#
# Usage:
#   bash scripts/analyze_principal_biases.sh [additional python args]
#
# Examples:
#   bash scripts/analyze_principal_biases.sh --glob 'principal_*_behavioral*.json'
#   BIAS_ANNOTATOR_MODEL=anthropic/claude-haiku-4.5 bash scripts/analyze_principal_biases.sh --limit 10
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python "$REPO_ROOT/pipeline/analyze_principal_biases.py" \
  --input-dir "$REPO_ROOT/experiments/principals/usmle_sample/baseline" \
  --output-dir "$REPO_ROOT/experiments/analysis/bias_annotations/usmle_sample/baseline" \
  "$@"
