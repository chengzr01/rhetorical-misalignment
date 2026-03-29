#!/bin/bash
#
# Framing Inference Runner Script for Clinical Questions (Agent Presentation + Principal)
#
# Usage:
#   bash scripts/run_framing.sh [agent_model_key] [dataset_key] [ground_truth_key] [inference_mode]
#
#   agent_model_key:    deepseek | gemini | gpt | claude | deepseek-llama | llama | llama-small | llama-large | llama-dpo | llama-sft | llama-base | llama-medium-sft | llama-medium-dpo | qwen | mistral | olmo | olmo-sft | olmo-dpo | olmo-base
#   dataset_key:        mimiciv_demo | usmle | usmle_sample
#   ground_truth_key:   deepseek | deepseek-llama | llama | etc.  (model-generated info)
#                       factual_agg | unfactual_agg               (aggregated llama claims)
#   inference_mode:     presentation | full  # presentation = agent presentation only, full = presentation + principal (default: presentation)
#
# Examples:
#   bash scripts/run_framing.sh llama usmle_sample deepseek presentation
#   bash scripts/run_framing.sh llama-dpo usmle_sample deepseek full
#   bash scripts/run_framing.sh llama usmle_sample factual_agg presentation
#   bash scripts/run_framing.sh llama usmle_sample unfactual_agg full
#   CLAIM_FORMAT=numbered bash scripts/run_framing.sh llama usmle_sample factual_agg full
#   AGENT_SERVER=sglang PRINCIPAL_MODEL=llama-dpo bash scripts/run_framing.sh llama-dpo usmle deepseek full
#
# Environment variables (optional overrides):
#   AGENT_SERVER, PRINCIPAL_SERVER, PRINCIPAL_MODEL, MAX_WORKERS, PRINCIPAL_TYPES, FORCE_RERUN, PRINCIPAL_WORKERS, CONTENT_FILTER, CLAIM_FORMAT, etc.
#   (See below for all configurables.)
#
# This script configures and runs agent presentation (using pre-generated information as ground truth) and optionally principal inference.

set -e  # Exit on error

# Load shared model configuration
source "$(dirname "$0")/model_config.sh"

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
INFERENCE_MODE="${4:-presentation}"  # presentation or full
PRINCIPAL_TYPES="${PRINCIPAL_TYPES:-bayesian_choices behavioral_choices}"  # Space-separated list
PRINCIPAL_WORKERS="${PRINCIPAL_WORKERS:-${MAX_WORKERS}}"
CONTENT_FILTER="${CONTENT_FILTER:-full}"  # full, key_points, diagnosis_only, treatment_only, facts_only
INCLUDE_OPTIONS="${INCLUDE_OPTIONS:-true}"  # true or false
CLAIM_FORMAT="${CLAIM_FORMAT:-bullets}"     # bullets | numbered | plain  (aggregated-info mode only)
MAX_CASES="${MAX_CASES:-0}"               # max cases to process (0 = no limit)


# Dataset configurations
declare -A DATASET_MAP=(
    ["mimiciv_demo"]="experiments/questions/clinical_questions_mimiciv_demo.json"
    ["usmle"]="experiments/questions/clinical_questions_usmle.json"
    ["usmle_sample"]="experiments/questions/clinical_questions_usmle_sample.json"
)

# Aggregated-info ground truth sources (bypass MODEL_MAP lookup)
declare -A AGGREGATED_MAP=(
    ["factual_agg"]="experiments/aggregation/aggregated_factual.json"
    ["unfactual_agg"]="experiments/aggregation/aggregated_unfactual.json"
)

# Parse arguments
AGENT_MODEL_KEY="${1:-llama}"
DATASET_KEY="${2:-usmle_sample}"
GROUND_TRUTH_KEY="${3:-deepseek}"

# Validate inference mode
if [ "$INFERENCE_MODE" != "presentation" ] && [ "$INFERENCE_MODE" != "full" ]; then
    echo -e "${RED}Error: Invalid inference mode '${INFERENCE_MODE}'${NC}"
    echo "Valid modes: presentation, full"
    exit 1
fi

# Resolve agent model
AGENT_MODEL="${MODEL_MAP[$AGENT_MODEL_KEY]}"
if [ -z "$AGENT_MODEL" ]; then
    echo -e "${RED}Error: Unknown agent model '${AGENT_MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi

# Resolve ground truth source: aggregated claims or model-generated
IS_AGGREGATED=false
if [ -n "${AGGREGATED_MAP[$GROUND_TRUTH_KEY]+x}" ]; then
    IS_AGGREGATED=true
    GROUND_TRUTH_MODEL="(aggregated llama claims)"
else
    GROUND_TRUTH_MODEL="${MODEL_MAP[$GROUND_TRUTH_KEY]}"
    if [ -z "$GROUND_TRUTH_MODEL" ]; then
        echo -e "${RED}Error: Unknown ground truth key '${GROUND_TRUTH_KEY}'${NC}"
        echo "Model keys: ${!MODEL_MAP[@]}"
        echo "Aggregated keys: ${!AGGREGATED_MAP[@]}"
        exit 1
    fi
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

# Configure AGENT_SERVER and SGLANG_PORT using model config (env vars take precedence)
AGENT_SERVER="${AGENT_SERVER:-$(get_model_server $AGENT_MODEL_KEY)}"
SGLANG_PORT="${SGLANG_PORT:-$(get_agent_sglang_port $AGENT_MODEL_KEY)}"

# Configure PRINCIPAL_SERVER and port
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-$(get_model_server $PRINCIPAL_MODEL_KEY)}"
PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-$(get_principal_sglang_port $PRINCIPAL_MODEL_KEY)}"

# Resolve dataset
INPUT_FILE="${DATASET_MAP[$DATASET_KEY]}"
if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Unknown dataset '${DATASET_KEY}'${NC}"
    echo "Available datasets: ${!DATASET_MAP[@]}"
    exit 1
fi

# Set ground truth path and output path
if [ "$IS_AGGREGATED" = "true" ]; then
    GROUND_TRUTH_PATH="${AGGREGATED_MAP[$GROUND_TRUTH_KEY]}"
    QUESTIONS_FILE="${DATASET_MAP[$DATASET_KEY]}"
else
    GROUND_TRUTH_PATH="experiments/agents/${DATASET_KEY}/agent_${GROUND_TRUTH_KEY}.json"
fi
AGENT_OUTPUT="experiments/agents/${DATASET_KEY}/framing_${AGENT_MODEL_KEY}_gt_${GROUND_TRUTH_KEY}.json"

# Check if ground truth file exists
if [ ! -f "$GROUND_TRUTH_PATH" ]; then
    echo -e "${RED}Error: Ground truth file not found: ${GROUND_TRUTH_PATH}${NC}"
    if [ "$IS_AGGREGATED" = "true" ]; then
        echo -e "${YELLOW}Please run experiments/aggregate_information.py first to generate the aggregated claims file.${NC}"
    else
        echo -e "${YELLOW}Please run agent inference first to generate ground truth:${NC}"
        echo -e "${YELLOW}  bash scripts/run_baseline.sh ${GROUND_TRUTH_KEY} ${DATASET_KEY} agent${NC}"
    fi
    exit 1
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Framing Inference Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Mode:             ${GREEN}${INFERENCE_MODE}${NC}"
echo -e "Agent Model:      ${GREEN}${AGENT_MODEL}${NC}"
echo -e "Agent Server:     ${GREEN}${AGENT_SERVER}${NC}"
if [ -n "$SGLANG_PORT" ]; then
    echo -e "SGLang Port:      ${GREEN}${SGLANG_PORT}${NC}"
fi
echo -e "Ground Truth:     ${GREEN}${GROUND_TRUTH_KEY}${NC} (${GROUND_TRUTH_MODEL})"
echo -e "Ground Truth Path:${YELLOW}${GROUND_TRUTH_PATH}${NC}"
if [ "$IS_AGGREGATED" = "true" ]; then
    echo -e "Claim Format:     ${GREEN}${CLAIM_FORMAT}${NC}"
fi
echo -e "Dataset:          ${GREEN}${DATASET_KEY}${NC} (${INPUT_FILE})"
echo -e "Content Filter:   ${GREEN}${CONTENT_FILTER}${NC}"
echo -e "Include Options:  ${GREEN}${INCLUDE_OPTIONS}${NC}"
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

# Run Agent Presentation
echo -e "${BLUE}Running agent presentation...${NC}\n"

# Build force flag if needed
FORCE_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi

# Build options flag if needed
OPTIONS_FLAG=""
if [ "$INCLUDE_OPTIONS" = "false" ]; then
    OPTIONS_FLAG="--no-options"
fi

if [ "$IS_AGGREGATED" = "true" ]; then
    # Aggregated-info mode: pass raw claims JSON + questions file directly
    SGLANG_FLAG=""
    if [ -n "$SGLANG_PORT" ]; then
        SGLANG_FLAG="--agent-sglang-port ${SGLANG_PORT}"
    fi
    python agent_presentation.py \
        --aggregated-info "${GROUND_TRUTH_PATH}" \
        --questions       "${QUESTIONS_FILE}" \
        --claim-format    "${CLAIM_FORMAT}" \
        --agent-server    "${AGENT_SERVER}" \
        --agent-model     "${AGENT_MODEL}" \
        --content-filter  "${CONTENT_FILTER}" \
        --output          "${AGENT_OUTPUT}" \
        --max-workers     "${MAX_WORKERS}" \
        --max-cases       "${MAX_CASES}" \
        ${SGLANG_FLAG} \
        ${OPTIONS_FLAG} \
        ${FORCE_FLAG}
elif [ -n "$SGLANG_PORT" ]; then
    python agent_presentation.py \
        --ground-truth "${GROUND_TRUTH_PATH}" \
        --agent-server "${AGENT_SERVER}" \
        --agent-model "${AGENT_MODEL}" \
        --agent-sglang-port "${SGLANG_PORT}" \
        --content-filter "${CONTENT_FILTER}" \
        --output "${AGENT_OUTPUT}" \
        --max-workers "${MAX_WORKERS}" \
        --max-cases "${MAX_CASES}" \
        ${OPTIONS_FLAG} \
        ${FORCE_FLAG}
else
    python agent_presentation.py \
        --ground-truth "${GROUND_TRUTH_PATH}" \
        --agent-server "${AGENT_SERVER}" \
        --agent-model "${AGENT_MODEL}" \
        --content-filter "${CONTENT_FILTER}" \
        --output "${AGENT_OUTPUT}" \
        --max-workers "${MAX_WORKERS}" \
        --max-cases "${MAX_CASES}" \
        ${OPTIONS_FLAG} \
        ${FORCE_FLAG}
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Agent presentation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Agent presentation complete${NC}\n"

# Stage 2: Principal Inference (only if mode is 'full')
if [ "$INFERENCE_MODE" = "full" ]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[STAGE 2] Principal Inference${NC}"
    echo -e "${BLUE}========================================${NC}\n"

    # Set principal output path
    PRINCIPAL_OUTPUT_BASE="experiments/principals/${DATASET_KEY}/principal_framing_${AGENT_MODEL_KEY}_gt_${GROUND_TRUTH_KEY}.json"

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
echo -e "${GREEN}Framing Inference Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Mode:              ${INFERENCE_MODE}"
echo -e "Ground Truth:      ${GROUND_TRUTH_KEY}"
echo -e "Agent results:     ${AGENT_OUTPUT}"
if [ "$INFERENCE_MODE" = "full" ]; then
    echo -e "Principal results: ${PRINCIPAL_OUTPUT_BASE}_*.json"
fi
echo -e "${GREEN}========================================${NC}"
