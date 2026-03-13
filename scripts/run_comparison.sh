#!/bin/bash
#
# Comparison Inference Runner Script (Principal only, no agent LLM call)
#
# Runs principal inference directly on the three synthesized claim files produced by
# experiments/synthesize_factual_information.py, allowing a clean comparison of how
# principals respond to purely factual, purely unfactual/uncertain, or all claims.
#
# Usage:
#   bash scripts/run_comparison.sh [agent_model_key] [dataset_key]
#
#   agent_model_key:  deepseek | gemini | gpt | claude | deepseek-llama | llama | llama-small | llama-dpo | llama-sft | ...
#   dataset_key:      mimiciv_demo | usmle | usmle_sample
#
# Examples:
#   bash scripts/run_comparison.sh claude usmle_sample
#   bash scripts/run_comparison.sh gpt usmle_sample
#   CLAIM_VARIANTS="factual unfactual" bash scripts/run_comparison.sh claude usmle_sample
#   PRINCIPAL_MODEL=llama-dpo bash scripts/run_comparison.sh gpt usmle_sample
#   FORCE_RERUN=true bash scripts/run_comparison.sh claude usmle_sample
#
# Environment variables (optional overrides):
#   PRINCIPAL_SERVER, PRINCIPAL_MODEL, PRINCIPAL_TYPES, PRINCIPAL_WORKERS,
#   CLAIM_VARIANTS, FORCE_RERUN, MAX_WORKERS, PRINCIPAL_SGLANG_PORT
#
# Claim variants (controlled by CLAIM_VARIANTS, default: all three):
#   factual      -- only "factual" claims from factualness analysis
#   unfactual    -- only "unfactual" + "uncertain" claims
#   all_claims   -- all claims from factualness analysis (combined baseline)
#
# Input files (must exist):
#   experiments/agents/{dataset_key}/agent_{model_key}_{variant}.json
# Output files:
#   experiments/principals/{dataset_key}/principal_{model_key}_{variant}_{bayesian|behavioral}.json

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
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-openrouter}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-false}"
PRINCIPAL_TYPES="${PRINCIPAL_TYPES:-bayesian behavioral}"
PRINCIPAL_WORKERS="${PRINCIPAL_WORKERS:-${MAX_WORKERS}}"
CLAIM_VARIANTS="${CLAIM_VARIANTS:-factual unfactual all_claims}"

# Dataset configurations
declare -A DATASET_MAP=(
    ["mimiciv_demo"]="experiments/questions/clinical_questions_mimiciv_demo.json"
    ["usmle"]="experiments/questions/clinical_questions_usmle.json"
    ["usmle_sample"]="experiments/questions/clinical_questions_usmle_sample.json"
)

# Parse arguments
AGENT_MODEL_KEY="${1:-claude}"
DATASET_KEY="${2:-usmle_sample}"

# Validate agent model key (used only for file paths, not for LLM calls)
if [ -z "${MODEL_MAP[$AGENT_MODEL_KEY]+x}" ]; then
    echo -e "${RED}Error: Unknown agent model key '${AGENT_MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi

# Validate dataset
if [ -z "${DATASET_MAP[$DATASET_KEY]+x}" ]; then
    echo -e "${RED}Error: Unknown dataset '${DATASET_KEY}'${NC}"
    echo "Available datasets: ${!DATASET_MAP[@]}"
    exit 1
fi

# Configure principal model
PRINCIPAL_MODEL_KEY="${PRINCIPAL_MODEL:-deepseek}"
PRINCIPAL_MODEL_NAME="${MODEL_MAP[$PRINCIPAL_MODEL_KEY]}"
if [ -z "$PRINCIPAL_MODEL_NAME" ]; then
    echo -e "${RED}Error: Unknown principal model '${PRINCIPAL_MODEL_KEY}'${NC}"
    echo "Available models: ${!MODEL_MAP[@]}"
    exit 1
fi

# Configure principal server and port
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-$(get_model_server $PRINCIPAL_MODEL_KEY)}"
PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-$(get_principal_sglang_port $PRINCIPAL_MODEL_KEY)}"

# Build force flag if needed
FORCE_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Comparison Inference Configuration${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Agent Model Key:  ${GREEN}${AGENT_MODEL_KEY}${NC} (input files only, no LLM call)"
echo -e "Dataset:          ${GREEN}${DATASET_KEY}${NC}"
echo -e "Claim Variants:   ${GREEN}${CLAIM_VARIANTS}${NC}"
echo -e "Principal Model:  ${GREEN}${PRINCIPAL_MODEL_NAME}${NC}"
echo -e "Principal Server: ${GREEN}${PRINCIPAL_SERVER}${NC}"
if [ -n "$PRINCIPAL_SGLANG_PORT" ]; then
    echo -e "Principal Port:   ${GREEN}${PRINCIPAL_SGLANG_PORT}${NC}"
fi
echo -e "Principal Types:  ${GREEN}${PRINCIPAL_TYPES}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Run principal inference for each claim variant
for VARIANT in $CLAIM_VARIANTS; do
    AGENT_CACHE="experiments/agents/${DATASET_KEY}/agent_${AGENT_MODEL_KEY}_${VARIANT}.json"
    PRINCIPAL_OUTPUT="experiments/principals/${DATASET_KEY}/principal_${AGENT_MODEL_KEY}_${VARIANT}.json"

    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Variant: ${VARIANT}${NC}"
    echo -e "Input:   ${YELLOW}${AGENT_CACHE}${NC}"
    echo -e "Output:  ${YELLOW}${PRINCIPAL_OUTPUT}${NC}"
    echo -e "${BLUE}----------------------------------------${NC}\n"

    # Verify input file exists
    if [ ! -f "$AGENT_CACHE" ]; then
        echo -e "${RED}Error: Synthesized agent file not found: ${AGENT_CACHE}${NC}"
        echo -e "${YELLOW}Run experiments/synthesize_factual_information.py first:${NC}"
        echo -e "${YELLOW}  python experiments/synthesize_factual_information.py --models ${AGENT_MODEL_KEY}${NC}"
        exit 1
    fi

    if [ -n "$PRINCIPAL_SGLANG_PORT" ]; then
        python principal_inference.py \
            --principal-server      "${PRINCIPAL_SERVER}" \
            --principal-model       "${PRINCIPAL_MODEL_NAME}" \
            --principal-sglang-port "${PRINCIPAL_SGLANG_PORT}" \
            --agent-cache           "${AGENT_CACHE}" \
            --output                "${PRINCIPAL_OUTPUT}" \
            --principal-types       ${PRINCIPAL_TYPES} \
            --max-workers           "${PRINCIPAL_WORKERS}" \
            ${FORCE_FLAG}
    else
        python principal_inference.py \
            --principal-server  "${PRINCIPAL_SERVER}" \
            --principal-model   "${PRINCIPAL_MODEL_NAME}" \
            --agent-cache       "${AGENT_CACHE}" \
            --output            "${PRINCIPAL_OUTPUT}" \
            --principal-types   ${PRINCIPAL_TYPES} \
            --max-workers       "${PRINCIPAL_WORKERS}" \
            ${FORCE_FLAG}
    fi

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Principal inference failed for variant '${VARIANT}'${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Variant '${VARIANT}' complete${NC}\n"
done

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Comparison Inference Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Agent model key: ${AGENT_MODEL_KEY}"
echo -e "Dataset:         ${DATASET_KEY}"
echo -e "Variants run:    ${CLAIM_VARIANTS}"
echo -e ""
echo -e "Principal results:"
for VARIANT in $CLAIM_VARIANTS; do
    for PTYPE in ${PRINCIPAL_TYPES}; do
        echo -e "  experiments/principals/${DATASET_KEY}/principal_${AGENT_MODEL_KEY}_${VARIANT}_${PTYPE}.json"
    done
done
echo -e "${GREEN}========================================${NC}"
