#!/bin/bash
#
# Inference Runner Script for Clinical Questions (Agent + Principal)
#
# Usage:
#   bash scripts/run.sh [agent_model_key] [dataset_key] [inference_mode]
#
#   agent_model_key:  deepseek | deepseek-llama | llama | llama-small | llama-large | llama-dpo | llama-sft | llama3-dpo | llama3-kto | mistral-dpo | mistral-kto | qwen | mistral
#   dataset_key:      mimiciv_demo | usmle | usmle_sample
#   inference_mode:   agent | full        # agent = agent only, full = agent + principal (default: agent)
#
# Examples:
#   bash scripts/run.sh llama usmle_sample agent
#   bash scripts/run.sh deepseek mimiciv_demo full
#   AGENT_SERVER=sglang PRINCIPAL_MODEL=llama-dpo bash scripts/run.sh llama-dpo usmle full
#
# Environment variables (optional overrides):
#   AGENT_SERVER, PRINCIPAL_SERVER, PRINCIPAL_MODEL, MAX_WORKERS, PRINCIPAL_TYPES, FORCE_RERUN, PRINCIPAL_WORKERS, etc.
#   (See below for all configurables.)
#
# This script configures and runs agent inference and optionally principal inference, supporting multiple model/server options.

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
AGENT_SERVER="${AGENT_SERVER:-openrouter}"
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-openrouter}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-false}"
INFERENCE_MODE="${3:-agent}"  # agent or full
PRINCIPAL_TYPES="${PRINCIPAL_TYPES:-bayesian behavioral}"  # Space-separated list
PRINCIPAL_WORKERS="${PRINCIPAL_WORKERS:-${MAX_WORKERS}}"

# Model configurations
declare -A MODEL_MAP=(
    ["deepseek"]="deepseek/deepseek-chat-v3.1"
    ["deepseek-llama"]="deepseek/deepseek-r1-distill-llama-70b"
    ["llama"]="meta-llama/llama-3.3-70b-instruct"
    ["llama-small"]="meta-llama/llama-3.1-8b-instruct"
    ["llama-large"]="meta-llama/llama-3.1-405b-instruct"
    ["llama-dpo"]="allenai/Llama-3.1-Tulu-3-8B-DPO"
    ["llama-sft"]="allenai/Llama-3.1-Tulu-3-8B-SFT"
    ["llama3-dpo"]="princeton-nlp/Llama-3-Instruct-8B-DPO"
    ["llama3-kto"]="princeton-nlp/Llama-3-Instruct-8B-KTO"
    ["mistral-dpo"]="princeton-nlp/Mistral-7B-Instruct-DPO"
    ["mistral-kto"]="princeton-nlp/Mistral-7B-Instruct-KTO"
    ["qwen"]="qwen/qwen-2.5-7b-instruct"
    ["mistral"]="mistralai/mistral-7b-instruct"
)


# Dataset configurations
declare -A DATASET_MAP=(
    ["mimiciv_demo"]="experiments/input/clinical_questions_mimiciv_demo.json"
    ["usmle"]="experiments/input/clinical_questions_usmle.json"
    ["usmle_sample"]="experiments/input/clinical_questions_usmle_sample.json"
)

# Parse arguments
AGENT_MODEL_KEY="${1:-llama}"
DATASET_KEY="${2:-mimiciv_demo}"

# Validate inference mode
if [ "$INFERENCE_MODE" != "agent" ] && [ "$INFERENCE_MODE" != "full" ]; then
    echo -e "${RED}Error: Invalid inference mode '${INFERENCE_MODE}'${NC}"
    echo "Valid modes: agent, full"
    exit 1
fi

# Resolve agent model
AGENT_MODEL="${MODEL_MAP[$AGENT_MODEL_KEY]}"
if [ -z "$AGENT_MODEL" ]; then
    echo -e "${RED}Error: Unknown agent model '${AGENT_MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi

# Configure principal model (use deepseek by default, or override)
PRINCIPAL_MODEL_KEY="${PRINCIPAL_MODEL:-deepseek}"
PRINCIPAL_MODEL="${MODEL_MAP[$PRINCIPAL_MODEL_KEY]}"
if [ -z "$PRINCIPAL_MODEL" ]; then
    echo -e "${RED}Error: Unknown principal model '${PRINCIPAL_MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-${AGENT_SERVER}}"

# Override AGENT_SERVER for sglang models
if [ "$AGENT_MODEL_KEY" == "llama-dpo" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30000"
elif [ "$AGENT_MODEL_KEY" == "llama-sft" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30000"
elif [ "$AGENT_MODEL_KEY" == "llama3-dpo" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30000"
elif [ "$AGENT_MODEL_KEY" == "llama3-kto" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30000"
elif [ "$AGENT_MODEL_KEY" == "mistral-dpo" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30000"
elif [ "$AGENT_MODEL_KEY" == "mistral-kto" ]; then
    AGENT_SERVER="sglang"
    SGLANG_PORT="30000"
fi

# Override PRINCIPAL_SERVER for sglang models (if using different principal model)
if [ "$PRINCIPAL_MODEL_KEY" == "llama-dpo" ]; then
    PRINCIPAL_SERVER="sglang"
    PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-30001}"
elif [ "$PRINCIPAL_MODEL_KEY" == "llama-sft" ]; then
    PRINCIPAL_SERVER="sglang"
    PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-30002}"
elif [ "$PRINCIPAL_MODEL_KEY" == "llama3-dpo" ]; then
    PRINCIPAL_SERVER="sglang"
    PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-30001}"
elif [ "$PRINCIPAL_MODEL_KEY" == "llama3-kto" ]; then
    PRINCIPAL_SERVER="sglang"
    PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-30002}"
elif [ "$PRINCIPAL_MODEL_KEY" == "mistral-dpo" ]; then
    PRINCIPAL_SERVER="sglang"
    PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-30003}"
elif [ "$PRINCIPAL_MODEL_KEY" == "mistral-kto" ]; then
    PRINCIPAL_SERVER="sglang"
    PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-30004}"
fi

# Resolve dataset
INPUT_FILE="${DATASET_MAP[$DATASET_KEY]}"
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Unknown dataset '${DATASET_KEY}'${NC}"
    echo "Available datasets: ${!DATASET_MAP[@]}"
    exit 1
fi

# Set output path
AGENT_OUTPUT="experiments/cache/${DATASET_KEY}/agent_${AGENT_MODEL_KEY}.json"

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Inference Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Mode:             ${GREEN}${INFERENCE_MODE}${NC}"
echo -e "Agent Model:      ${GREEN}${AGENT_MODEL}${NC}"
echo -e "Agent Server:     ${GREEN}${AGENT_SERVER}${NC}"
if [ -n "$SGLANG_PORT" ]; then
    echo -e "SGLang Port:      ${GREEN}${SGLANG_PORT}${NC}"
fi
echo -e "Dataset:          ${GREEN}${DATASET_KEY}${NC} (${INPUT_FILE})"
echo -e "Agent Output:     ${YELLOW}${AGENT_OUTPUT}${NC}"

if [ "$INFERENCE_MODE" = "full" ]; then
    echo -e "\n${BLUE}Principal Inference:${NC}"
    echo -e "Principal Model:  ${GREEN}${PRINCIPAL_MODEL}${NC}"
    echo -e "Principal Server: ${GREEN}${PRINCIPAL_SERVER}${NC}"
    if [ -n "$PRINCIPAL_SGLANG_PORT" ]; then
        echo -e "Principal Port:   ${GREEN}${PRINCIPAL_SGLANG_PORT}${NC}"
    fi
    echo -e "Principal Types:  ${GREEN}${PRINCIPAL_TYPES}${NC}"
fi
echo -e "${BLUE}========================================${NC}\n"

# Run Agent Inference
echo -e "${BLUE}Running agent inference...${NC}\n"

# Build force flag if needed
FORCE_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi

if [ -n "$SGLANG_PORT" ]; then
    python agent_inference.py \
        --agent-server "${AGENT_SERVER}" \
        --agent-model "${AGENT_MODEL}" \
        --agent-sglang-port "${SGLANG_PORT}" \
        --input "${INPUT_FILE}" \
        --output "${AGENT_OUTPUT}" \
        --max-workers "${MAX_WORKERS}" \
        ${FORCE_FLAG}
else
    python agent_inference.py \
        --agent-server "${AGENT_SERVER}" \
        --agent-model "${AGENT_MODEL}" \
        --input "${INPUT_FILE}" \
        --output "${AGENT_OUTPUT}" \
        --max-workers "${MAX_WORKERS}" \
        ${FORCE_FLAG}
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Agent inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Agent inference complete${NC}\n"

# Stage 2: Principal Inference (only if mode is 'full')
if [ "$INFERENCE_MODE" = "full" ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[STAGE 2] Principal Inference${NC}"
    echo -e "${BLUE}========================================${NC}\n"

    # Set principal output path
    PRINCIPAL_OUTPUT_BASE="experiments/output/${DATASET_KEY}/principal_${AGENT_MODEL_KEY}.json"

    echo -e "${BLUE}Running principal inference...${NC}\n"

    if [ -n "$PRINCIPAL_SGLANG_PORT" ]; then
        python principal_inference.py \
            --principal-server "${PRINCIPAL_SERVER}" \
            --principal-model "${PRINCIPAL_MODEL}" \
            --principal-sglang-port "${PRINCIPAL_SGLANG_PORT}" \
            --agent-cache "${AGENT_OUTPUT}" \
            --output "${PRINCIPAL_OUTPUT_BASE}" \
            --principal-types ${PRINCIPAL_TYPES} \
            --max-workers "${PRINCIPAL_WORKERS}" \
            ${FORCE_FLAG}
    else
        python principal_inference.py \
            --principal-server "${PRINCIPAL_SERVER}" \
            --principal-model "${PRINCIPAL_MODEL}" \
            --agent-cache "${AGENT_OUTPUT}" \
            --output "${PRINCIPAL_OUTPUT_BASE}" \
            --principal-types ${PRINCIPAL_TYPES} \
            --max-workers "${PRINCIPAL_WORKERS}" \
            ${FORCE_FLAG}
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Principal inference failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Principal inference complete${NC}\n"
fi

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Inference Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Mode:              ${INFERENCE_MODE}"
echo -e "Agent results:     ${AGENT_OUTPUT}"
if [ "$INFERENCE_MODE" = "full" ]; then
    echo -e "Principal results: ${PRINCIPAL_OUTPUT_BASE}_*.json"
fi
echo -e "${GREEN}========================================${NC}"
