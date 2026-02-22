#!/bin/bash

# Quick start script for cross-model comparison analysis
# Usage: ./run_cross_model_comparison.sh [MAX_CASES]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="../outputs"
MAX_CASES="${1:-}"

# Check for API key in environment
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable not set"
    echo ""
    echo "Set it with:"
    echo "  export OPENROUTER_API_KEY=your_key_here"
    echo ""
    exit 1
fi

echo "=================================="
echo "Cross-Model Comparison Analysis"
echo "=================================="
echo ""

# Check if persuasion examples exist
if [ ! -f "$OUTPUT_DIR/persuasion_examples.json" ]; then
    echo "Error: persuasion_examples.json not found in $OUTPUT_DIR/"
    echo "Run: python collect_persuasion_examples.py"
    exit 1
fi

# Build command
CMD="python llm_analysis.py --mode compare-reactions --persuasion-file $OUTPUT_DIR/persuasion_examples.json --output-dir $OUTPUT_DIR"

if [ -n "$MAX_CASES" ]; then
    CMD="$CMD --max-cases $MAX_CASES"
    echo "Running with max $MAX_CASES cases..."
else
    echo "Running on all available cases..."
fi

echo ""

# Run the analysis
eval $CMD

echo ""
echo "=================================="
echo "Done!"
echo "=================================="
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/cross_model_comparison_report.txt"
echo ""
