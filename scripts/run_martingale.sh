#!/bin/bash
#
# Martingale Reliability Test for Bayesian Benchmark
#
# Usage:
#   bash scripts/run_martingale.sh [dataset_key] [num_permutations] [principal_model_key]
#
#   dataset_key:          usmle_sample | usmle | mimiciv_demo  (default: usmle_sample)
#   num_permutations:     number of random claim orderings per case (default: 10)
#   principal_model_key:  deepseek | gpt | claude | gemini | ...  (default: deepseek)
#
# Examples:
#   bash scripts/run_martingale.sh usmle_sample 10 deepseek
#   bash scripts/run_martingale.sh usmle 20 gpt
#   FORCE_RERUN=true bash scripts/run_martingale.sh usmle_sample 10 deepseek
#
# Environment variables (optional overrides):
#   PRINCIPAL_SERVER, MAX_WORKERS, FORCE_RERUN, CLAIM_FORMAT, SEED, MAX_CASES
#
# Pipeline:
#   STAGE 1: Generate K random permutations of claims per case
#   STAGE 2: Run Bayesian principal inference on all permutations
#   STAGE 3: Analyze martingale reliability (majority vote, belief variance)

set -e

source "$(dirname "$0")/model_config.sh"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Arguments
DATASET_KEY="${1:-usmle_sample}"
NUM_PERMUTATIONS="${2:-10}"
PRINCIPAL_MODEL_KEY="${3:-deepseek}"
AGENT_KEY="${4:-}"          # optional: tag for per-agent runs (e.g. claude, deepseek)

# Configurable overrides
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-openrouter}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-false}"
CLAIM_FORMAT="${CLAIM_FORMAT:-bullets}"
SEED="${SEED:-42}"
MAX_CASES="${MAX_CASES:-0}"  # 0 = no limit

# Dataset configurations
declare -A DATASET_MAP=(
    ["mimiciv_demo"]="experiments/questions/clinical_questions_mimiciv_demo.json"
    ["usmle"]="experiments/questions/clinical_questions_usmle.json"
    ["usmle_sample"]="experiments/questions/clinical_questions_usmle_sample.json"
)

# Resolve dataset
QUESTIONS_FILE="${DATASET_MAP[$DATASET_KEY]}"
if [ -z "$QUESTIONS_FILE" ]; then
    echo -e "${RED}Error: Unknown dataset '${DATASET_KEY}'${NC}"
    echo "Available: ${!DATASET_MAP[@]}"
    exit 1
fi

# Resolve principal model
PRINCIPAL_MODEL="${MODEL_MAP[$PRINCIPAL_MODEL_KEY]}"
if [ -z "$PRINCIPAL_MODEL" ]; then
    echo -e "${RED}Error: Unknown principal model '${PRINCIPAL_MODEL_KEY}'${NC}"
    echo "Available: ${!MODEL_MAP[@]}"
    exit 1
fi

PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-$(get_model_server $PRINCIPAL_MODEL_KEY)}"
PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-$(get_principal_sglang_port $PRINCIPAL_MODEL_KEY)}"

# Paths
# AGGREGATED_INFO can be overridden via env var for per-agent runs
AGGREGATED_INFO="${AGGREGATED_INFO:-experiments/aggregation/aggregated_factual.json}"

# Optional suffix to distinguish per-agent output files (e.g. "_claude", "_deepseek")
_AGENT_SUFFIX=""
if [ -n "$AGENT_KEY" ]; then
    _AGENT_SUFFIX="_${AGENT_KEY}"
fi

PERMUTATIONS_FILE="experiments/agents/${DATASET_KEY}/martingale_permutations${_AGENT_SUFFIX}_k${NUM_PERMUTATIONS}.json"
PRINCIPAL_OUTPUT="experiments/principals/${DATASET_KEY}/principal_martingale${_AGENT_SUFFIX}_k${NUM_PERMUTATIONS}.json"
ANALYSIS_OUTPUT="experiments/principals/${DATASET_KEY}/martingale_analysis${_AGENT_SUFFIX}_k${NUM_PERMUTATIONS}.json"

# Validate aggregated info exists
if [ ! -f "$AGGREGATED_INFO" ]; then
    echo -e "${RED}Error: Aggregated claims not found: ${AGGREGATED_INFO}${NC}"
    echo -e "${YELLOW}Please run experiments/pipeline/aggregate_information.py first.${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Martingale Reliability Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Dataset:              ${GREEN}${DATASET_KEY}${NC}"
echo -e "Agent key:            ${GREEN}${AGENT_KEY:-<default>}${NC}"
echo -e "Num permutations (K): ${GREEN}${NUM_PERMUTATIONS}${NC}"
echo -e "Claim format:         ${GREEN}${CLAIM_FORMAT}${NC}"
echo -e "Random seed:          ${GREEN}${SEED}${NC}"
echo -e "Principal model:      ${GREEN}${PRINCIPAL_MODEL}${NC}"
echo -e "Principal server:     ${GREEN}${PRINCIPAL_SERVER}${NC}"
echo -e "Aggregated info:      ${YELLOW}${AGGREGATED_INFO}${NC}"
echo -e "Permutations output:  ${YELLOW}${PERMUTATIONS_FILE}${NC}"
echo -e "Principal output:     ${YELLOW}${PRINCIPAL_OUTPUT}${NC}"
echo -e "Analysis output:      ${YELLOW}${ANALYSIS_OUTPUT}${NC}"
echo -e "${BLUE}========================================${NC}\n"

FORCE_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi

# ------------------------------------------------------------------
# STAGE 1: Generate permutations
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 1] Generating permutations...${NC}\n"

MAX_CASES_FLAG=""
if [ "${MAX_CASES}" -gt 0 ]; then
    MAX_CASES_FLAG="--max-cases ${MAX_CASES}"
fi

python pipeline/generate_permutations.py \
    --aggregated-info "${AGGREGATED_INFO}" \
    --questions       "${QUESTIONS_FILE}" \
    --num-permutations "${NUM_PERMUTATIONS}" \
    --claim-format    "${CLAIM_FORMAT}" \
    --seed            "${SEED}" \
    --output          "${PERMUTATIONS_FILE}" \
    ${MAX_CASES_FLAG} \
    ${FORCE_FLAG}

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Permutation generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Permutations generated${NC}\n"

# ------------------------------------------------------------------
# STAGE 2: Principal inference on all permutations
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 2] Running principal inference on permutations...${NC}\n"

if [ -n "$PRINCIPAL_SGLANG_PORT" ]; then
    python core/principal_inference.py \
        --principal-server     "${PRINCIPAL_SERVER}" \
        --principal-model      "${PRINCIPAL_MODEL}" \
        --principal-sglang-port "${PRINCIPAL_SGLANG_PORT}" \
        --agent-cache          "${PERMUTATIONS_FILE}" \
        --output               "${PRINCIPAL_OUTPUT}" \
        --principal-types      bayesian_martingale_choices \
        --max-workers          "${MAX_WORKERS}" \
        ${FORCE_FLAG}
else
    python core/principal_inference.py \
        --principal-server  "${PRINCIPAL_SERVER}" \
        --principal-model   "${PRINCIPAL_MODEL}" \
        --agent-cache       "${PERMUTATIONS_FILE}" \
        --output            "${PRINCIPAL_OUTPUT}" \
        --principal-types   bayesian_martingale_choices \
        --max-workers       "${MAX_WORKERS}" \
        ${FORCE_FLAG}
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Principal inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Principal inference complete${NC}\n"

# ------------------------------------------------------------------
# STAGE 3: Analyze martingale reliability
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 3] Analyzing martingale reliability...${NC}\n"

# principal_inference.py appends the principal type to the output filename
PRINCIPAL_RESULTS="${PRINCIPAL_OUTPUT%.json}_bayesian_martingale_choices.json"

python pipeline/compute_martingale.py \
    --input  "${PRINCIPAL_RESULTS}" \
    --output "${ANALYSIS_OUTPUT}" \
    ${FORCE_FLAG}

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Analysis failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Analysis complete${NC}\n"

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Martingale Test Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Permutations: ${PERMUTATIONS_FILE}"
echo -e "Results:      ${PRINCIPAL_RESULTS}"
echo -e "Analysis:     ${ANALYSIS_OUTPUT}"
echo -e "${GREEN}========================================${NC}"
