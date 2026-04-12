#!/bin/bash
#
# Prepare Neutral Ground-Truth Claims from a Single Model's Factualness Analysis
#
# Runs two pipeline stages to convert per-model factualness analysis results into
# a neutralized aggregated-claims file ready for framing / information-design experiments:
#
#   STAGE 1: Aggregate — filter to factual claims from the specified model
#   STAGE 2: Neutralize — rewrite claim language to be model-agnostic
#
# The output can be used in place of aggregated_factual.json for any script that
# accepts --aggregated-info (agent_presentation.py, agent_selection.py) or the
# AGGREGATED_MAP keys in run_framing.sh / run_information.sh.
#
# Usage:
#   bash scripts/prepare_ground_truth.sh [model_key] [dataset_key]
#
#   model_key:    gemini | gpt | claude | deepseek | llama | llama-small | ...
#                 Must have a corresponding factualness_agent_<model_key>.json in
#                 experiments/information/
#   dataset_key:  usmle_sample | usmle | mimiciv_demo  (default: usmle_sample)
#
# Prerequisites:
#   factualness_agent_<model_key>.json must exist in experiments/information/.
#   Run pipeline/analyze_information_factualness.py first if it does not.
#
# Examples:
#   bash scripts/prepare_ground_truth.sh gemini usmle_sample
#   bash scripts/prepare_ground_truth.sh gpt    usmle_sample
#   FORCE_RERUN=true bash scripts/prepare_ground_truth.sh gemini usmle_sample
#   NEUTRALIZE_MODEL=anthropic/claude-haiku-4.5 bash scripts/prepare_ground_truth.sh gemini
#
# Outputs:
#   experiments/aggregation/aggregated_<model_key>_factual.json        (Stage 1)
#   experiments/aggregation/aggregated_<model_key>_factual_neutral.json (Stage 2)
#
# Environment variables (optional overrides):
#   NEUTRALIZE_MODEL    OpenRouter model for claim neutralization
#   MAX_WORKERS         Parallel API calls (default: 8)
#   FORCE_RERUN         Re-run all stages even if outputs exist (default: false)
#   MAX_CASES           Limit cases for testing, 0 = no limit (default: 0)

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Arguments
MODEL_KEY="${1:-gemini}"
DATASET_KEY="${2:-usmle_sample}"

# Configurable overrides
NEUTRALIZE_MODEL="${NEUTRALIZE_MODEL:-deepseek/deepseek-chat-v3.1}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-false}"
MAX_CASES="${MAX_CASES:-0}"

# Dataset question file
declare -A DATASET_MAP=(
    ["mimiciv_demo"]="experiments/questions/clinical_questions_mimiciv_demo.json"
    ["usmle"]="experiments/questions/clinical_questions_usmle.json"
    ["usmle_sample"]="experiments/questions/clinical_questions_usmle_sample.json"
)

QUESTIONS_FILE="${DATASET_MAP[$DATASET_KEY]}"
if [ -z "$QUESTIONS_FILE" ]; then
    echo -e "${RED}Error: Unknown dataset '${DATASET_KEY}'${NC}"
    echo "Available: ${!DATASET_MAP[@]}"
    exit 1
fi

# Paths
INFORMATION_DIR="experiments/information"
FACTUALNESS_FILE="${INFORMATION_DIR}/factualness_agent_${MODEL_KEY}.json"
AGGREGATED_OUT="experiments/aggregation/aggregated_${MODEL_KEY}_factual.json"
NEUTRAL_OUT="experiments/aggregation/aggregated_${MODEL_KEY}_factual_neutral.json"

# Validate prerequisite
if [ ! -f "$FACTUALNESS_FILE" ]; then
    echo -e "${RED}Error: Factualness file not found: ${FACTUALNESS_FILE}${NC}"
    echo -e "${YELLOW}Run pipeline/analyze_information_factualness.py first:${NC}"
    echo -e "${YELLOW}  python pipeline/analyze_information_factualness.py analyze --agents ${MODEL_KEY}${NC}"
    exit 1
fi

FORCE_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi

MAX_CASES_FLAG=""
if [ "${MAX_CASES}" -gt 0 ]; then
    MAX_CASES_FLAG="--max-cases ${MAX_CASES}"
fi

# Print configuration
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Prepare Neutral Ground-Truth Claims${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Model key:          ${GREEN}${MODEL_KEY}${NC}"
echo -e "Dataset:            ${GREEN}${DATASET_KEY}${NC}"
echo -e "Factualness input:  ${YELLOW}${FACTUALNESS_FILE}${NC}"
echo -e "Aggregated output:  ${YELLOW}${AGGREGATED_OUT}${NC}"
echo -e "Neutral output:     ${YELLOW}${NEUTRAL_OUT}${NC}"
echo -e "Neutralize model:   ${GREEN}${NEUTRALIZE_MODEL}${NC}"
echo -e "Workers:            ${GREEN}${MAX_WORKERS}${NC}"
echo -e "${BLUE}================================================${NC}\n"

# ------------------------------------------------------------------
# STAGE 1: Aggregate factual claims
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 1] Aggregating factual claims for ${MODEL_KEY}...${NC}\n"

if [ -f "$AGGREGATED_OUT" ] && [ "$FORCE_RERUN" != "true" ]; then
    echo -e "${YELLOW}  Skipping — output already exists: ${AGGREGATED_OUT}${NC}"
    echo -e "${YELLOW}  Set FORCE_RERUN=true to re-aggregate.${NC}\n"
else
    python pipeline/aggregate_information.py \
        --agents           "${MODEL_KEY}" \
        --information-dir  "${INFORMATION_DIR}" \
        --labels           factual \
        --output           "${AGGREGATED_OUT}"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Aggregation failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Aggregated factual claims → ${AGGREGATED_OUT}${NC}\n"
fi

# ------------------------------------------------------------------
# STAGE 2: Neutralize claim language
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 2] Neutralizing claim language...${NC}\n"

NEUTRALIZE_MODEL="${NEUTRALIZE_MODEL}" \
python pipeline/neutralize_claims.py \
    --input       "${AGGREGATED_OUT}" \
    --questions   "${QUESTIONS_FILE}" \
    --output      "${NEUTRAL_OUT}" \
    --model       "${NEUTRALIZE_MODEL}" \
    --max-workers "${MAX_WORKERS}" \
    ${MAX_CASES_FLAG} \
    ${FORCE_FLAG}

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Neutralization failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Neutralized claims → ${NEUTRAL_OUT}${NC}\n"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Ground-Truth Preparation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "Factual claims:     ${AGGREGATED_OUT}"
echo -e "Neutral claims:     ${NEUTRAL_OUT}"
echo -e ""
echo -e "Use in framing experiments:"
echo -e "  bash scripts/run_framing.sh <agent_key> ${DATASET_KEY} ${MODEL_KEY}_factual_neutral full"
echo -e ""
echo -e "Use in information-design experiments:"
echo -e "  bash scripts/run_information.sh <agent_key> ${DATASET_KEY} ${MODEL_KEY}_factual_neutral full"
echo -e "${GREEN}================================================${NC}"
