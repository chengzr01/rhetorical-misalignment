#!/bin/bash
#
# Main inference runner script
# Usage: ./run_inference.sh <agent_model> <dataset> <principal_types>
#
# Examples:
#   bash scripts/run_inference.sh llama-large mimiciv_demo bayesian
#   bash scripts/run_inference.sh deepseek usmle behavioral
#   bash scripts/run_inference.sh deepseek usmle all
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
AGENT_SERVER="${AGENT_SERVER:-nvidia}"
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-nvidia}"
PRINCIPAL_MODEL="${PRINCIPAL_MODEL:-deepseek-ai/deepseek-v3.1}"
MAX_WORKERS="${MAX_WORKERS:-8}"
PRINCIPAL_WORKERS="${PRINCIPAL_WORKERS:-4}"

# Model configurations
declare -A MODEL_MAP=(
    ["deepseek"]="deepseek-ai/deepseek-v3.1"
    ["llama"]="meta/llama-3.3-70b-instruct"
    ["llama-small"]="meta/llama-3.1-8b-instruct"
    ["llama-large"]="meta/llama-3.1-405b-instruct"
    ["llama-dpo"]="allenai/Llama-3.1-Tulu-3-8B-DPO"
    ["llama-sft"]="allenai/Llama-3.1-Tulu-3-8B-SFT"
)

# Dataset configurations
declare -A DATASET_MAP=(
    ["mimiciv_demo"]="experiments/input/clinical_questions_mimiciv_demo.json"
    ["usmle"]="experiments/input/clinical_questions_usmle.json"
)

# Parse arguments
AGENT_MODEL_KEY="${1:-llama}"
DATASET_KEY="${2:-mimiciv_demo}"
PRINCIPAL_TYPES="${3:-bayesian}"

# Resolve model
AGENT_MODEL="${MODEL_MAP[$AGENT_MODEL_KEY]}"
if [ -z "$AGENT_MODEL" ]; then
    echo -e "${RED}Error: Unknown agent model '${AGENT_MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi

# Override AGENT_SERVER for sglang models
if [ "$AGENT_MODEL_KEY" == "llama-dpo" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30001"
elif [ "$AGENT_MODEL_KEY" == "llama-sft" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30002"
fi

# Resolve dataset
INPUT_FILE="${DATASET_MAP[$DATASET_KEY]}"
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Unknown dataset '${DATASET_KEY}'${NC}"
    echo "Available datasets: ${!DATASET_MAP[@]}"
    exit 1
fi

# Set output paths
AGENT_CACHE="experiments/cache/${DATASET_KEY}/agent_${AGENT_MODEL_KEY}.json"
if [ "$PRINCIPAL_TYPES" == "all" ]; then
    PRINCIPAL_OUTPUT="experiments/output/${DATASET_KEY}/principal_${AGENT_MODEL_KEY}_all.json"
else
    PRINCIPAL_OUTPUT="experiments/output/${DATASET_KEY}/principal_${AGENT_MODEL_KEY}_${PRINCIPAL_TYPES}.json"
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Inference Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Agent Model:      ${GREEN}${AGENT_MODEL}${NC}"
echo -e "Agent Server:     ${GREEN}${AGENT_SERVER}${NC}"
if [ -n "$SGLANG_PORT" ]; then
    echo -e "SGLang Port:      ${GREEN}${SGLANG_PORT}${NC}"
fi
echo -e "Dataset:          ${GREEN}${DATASET_KEY}${NC} (${INPUT_FILE})"
echo -e "Principal Types:  ${GREEN}${PRINCIPAL_TYPES}${NC}"
echo -e "Agent Cache:      ${YELLOW}${AGENT_CACHE}${NC}"
echo -e "Principal Output: ${YELLOW}${PRINCIPAL_OUTPUT}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Stage 1: Agent Inference
echo -e "${BLUE}[STAGE 1]${NC} Running agent inference..."
if [ -n "$SGLANG_PORT" ]; then
    python agent_inference.py \
        --agent-server "${AGENT_SERVER}" \
        --agent-model "${AGENT_MODEL}" \
        --agent-sglang-port "${SGLANG_PORT}" \
        --input "${INPUT_FILE}" \
        --output "${AGENT_CACHE}" \
        --max-workers "${MAX_WORKERS}"
else
    python agent_inference.py \
        --agent-server "${AGENT_SERVER}" \
        --agent-model "${AGENT_MODEL}" \
        --input "${INPUT_FILE}" \
        --output "${AGENT_CACHE}" \
        --max-workers "${MAX_WORKERS}"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Agent inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Agent inference complete${NC}\n"

# Stage 2: Principal Inference
echo -e "${BLUE}[STAGE 2]${NC} Running principal inference..."
python principal_inference.py \
    --principal-server "${PRINCIPAL_SERVER}" \
    --principal-model "${PRINCIPAL_MODEL}" \
    --agent-cache "${AGENT_CACHE}" \
    --output "${PRINCIPAL_OUTPUT}" \
    --principal-types ${PRINCIPAL_TYPES} \
    --max-workers "${PRINCIPAL_WORKERS}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Principal inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Principal inference complete${NC}\n"

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Inference Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Agent results:     ${AGENT_CACHE}"
echo -e "Principal results: ${PRINCIPAL_OUTPUT}"
echo -e "${GREEN}========================================${NC}"
