#!/bin/bash
#
# Test script for USMLE sample experiments
# Usage: ./test.sh [model_key]
#
# Model keys:
#   deepseek        - deepseek/deepseek-chat-v3.1 (default)
#   gemini          - google/gemini-2.5-pro
#   gpt             - openai/gpt-5.1
#   claude          - anthropic/claude-haiku-4.5
#   deepseek-llama  - deepseek/deepseek-r1-distill-llama-70b
#   llama-small     - meta-llama/llama-3.1-8b-instruct
#   llama           - meta-llama/llama-3.3-70b-instruct
#   llama-large     - meta-llama/llama-3.1-405b-instruct
#   llama-dpo       - allenai/Llama-3.1-Tulu-3-8B-DPO
#   llama-sft       - allenai/Llama-3.1-Tulu-3-8B-SFT
#   qwen            - qwen/qwen-2.5-7b-instruct
#   mistral         - mistralai/mistral-7b-instruct
#
# Examples:
#   bash scripts/test.sh                    # Run with deepseek (default)
#   bash scripts/test.sh llama              # Run with llama-3.3-70b
#   bash scripts/test.sh llama-small        # Run with llama-3.1-8b
#   bash scripts/test.sh deepseek-llama     # Run with deepseek-llama
#   bash scripts/test.sh llama-dpo          # Run with llama-dpo
#   bash scripts/test.sh llama-sft          # Run with llama-sft
#   bash scripts/test.sh qwen               # Run with qwen-2.5-7b
#   bash scripts/test.sh mistral            # Run with mistral-7b
#
# Customize parameters:
#   BACKEND=nvidia bash scripts/test.sh deepseek
#   MAX_WORKERS=16 bash scripts/test.sh llama-small
#   ELICIT_BELIEF=false bash scripts/test.sh llama
#   SGLANG_PORT=30000 bash scripts/test.sh llama-sft
#

set -e  # Exit on error

# Load shared model configuration
source "$(dirname "$0")/model_config.sh"

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

# Worker configurations per model (adjust based on model size)
declare -A WORKER_MAP=(
    ["deepseek"]="8"
    ["gemini"]="8"
    ["gpt"]="8"
    ["claude"]="8"
    ["deepseek-llama"]="8"
    ["llama"]="8"
    ["llama-small"]="32"
    ["llama-large"]="4"
    ["llama-dpo"]="32"
    ["llama-sft"]="32"
    ["qwen"]="32"
    ["mistral"]="32"
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

# Configure backend and port using model config
SELECTED_BACKEND="${BACKEND:-$(get_model_server $MODEL_KEY)}"
SGLANG_PORT="${SGLANG_PORT:-$(get_agent_sglang_port $MODEL_KEY)}"

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "Backend:       ${GREEN}${SELECTED_BACKEND}${NC}"
echo -e "Temperature:   ${GREEN}${TEMPERATURE}${NC}"
echo -e "Max Workers:   ${GREEN}${MAX_WORKERS}${NC}"
echo -e "Elicit Belief: ${GREEN}${ELICIT_BELIEF}${NC}"
echo -e "Prompt File:   ${GREEN}${PROMPT_FILE}${NC}"
if [[ "$SELECTED_BACKEND" == "sglang" && -n "$SGLANG_PORT" ]]; then
    echo -e "SGLang Port:   ${GREEN}${SGLANG_PORT}${NC}"
fi
echo -e "${BLUE}========================================${NC}\n"

# Build command arguments
CMD_ARGS=(
    "experiments/test_usmle_sample.py"
    "--model" "${MODEL}"
    "--backend" "${SELECTED_BACKEND}"
    "--temperature" "${TEMPERATURE}"
    "--max-workers" "${MAX_WORKERS}"
)

# Add elicit-belief flag if enabled
if [ "$ELICIT_BELIEF" = "true" ]; then
    CMD_ARGS+=("--elicit-belief")
fi

# Add prompt file
CMD_ARGS+=("--prompt" "${PROMPT_FILE}")

# Add port for sglang/dpo/sft models
if [[ "$SELECTED_BACKEND" == "sglang" && -n "$SGLANG_PORT" ]]; then
    CMD_ARGS+=("--sglang-port" "${SGLANG_PORT}")
fi

# Run the test
echo -e "${BLUE}Running test...${NC}\n"
python "${CMD_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Test completed successfully${NC}"
else
    echo -e "\n${RED}✗ Test failed${NC}"
    exit 1
fi
