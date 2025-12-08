#!/bin/bash
#
# Test script for USMLE sample experiments
# Usage: ./test.sh [model_key]
#
# Model keys:
#   deepseek     - deepseek/deepseek-chat-v3.1 (default)
#   llama-small  - meta-llama/llama-3.1-8b-instruct
#   llama        - meta-llama/llama-3.3-70b-instruct
#   llama-large  - meta-llama/llama-3.1-405b-instruct
#
# Examples:
#   bash scripts/test.sh                # Run with deepseek (default)
#   bash scripts/test.sh llama          # Run with llama-3.3-70b
#   bash scripts/test.sh llama-small    # Run with llama-3.1-8b
#
# Customize parameters:
#   BACKEND=nvidia bash scripts/test.sh deepseek
#   MAX_WORKERS=16 bash scripts/test.sh llama-small
#   ELICIT_BELIEF=false bash scripts/test.sh llama
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
BACKEND="${BACKEND:-openrouter}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_WORKERS="${MAX_WORKERS:-8}"
ELICIT_BELIEF="${ELICIT_BELIEF:-true}"
PROMPT_FILE="${PROMPT_FILE:-prompts/experiments/elicit.yaml}"

# Model configurations
declare -A MODEL_MAP=(
    ["deepseek"]="deepseek/deepseek-chat-v3.1"
    ["llama"]="meta-llama/llama-3.3-70b-instruct"
    ["llama-small"]="meta-llama/llama-3.1-8b-instruct"
    ["llama-large"]="meta-llama/llama-3.1-405b-instruct"
)

# Worker configurations per model (adjust based on model size)
declare -A WORKER_MAP=(
    ["deepseek"]="8"
    ["llama"]="8"
    ["llama-small"]="32"
    ["llama-large"]="4"
)

# Parse arguments
MODEL_KEY="${1:-deepseek}"

# Resolve model
MODEL="${MODEL_MAP[$MODEL_KEY]}"
if [ -z "$MODEL" ]; then
    echo -e "${RED}Error: Unknown model '${MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi

# Use model-specific worker count if MAX_WORKERS not explicitly set
if [ -z "${MAX_WORKERS_SET}" ]; then
    MAX_WORKERS="${WORKER_MAP[$MODEL_KEY]:-8}"
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model:        ${GREEN}${MODEL}${NC}"
echo -e "Backend:      ${GREEN}${BACKEND}${NC}"
echo -e "Temperature:  ${GREEN}${TEMPERATURE}${NC}"
echo -e "Max Workers:  ${GREEN}${MAX_WORKERS}${NC}"
echo -e "Elicit Belief: ${GREEN}${ELICIT_BELIEF}${NC}"
echo -e "Prompt File:  ${GREEN}${PROMPT_FILE}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Build command arguments
CMD_ARGS=(
    "experiments/test_usmle_sample.py"
    "--model" "${MODEL}"
    "--backend" "${BACKEND}"
    "--temperature" "${TEMPERATURE}"
    "--max-workers" "${MAX_WORKERS}"
)

# Add elicit-belief flag if enabled
if [ "$ELICIT_BELIEF" = "true" ]; then
    CMD_ARGS+=("--elicit-belief")
fi

# Add prompt file
CMD_ARGS+=("--prompt" "${PROMPT_FILE}")

# Run the test
echo -e "${BLUE}Running test...${NC}\n"
python "${CMD_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Test completed successfully${NC}"
else
    echo -e "\n${RED}✗ Test failed${NC}"
    exit 1
fi
