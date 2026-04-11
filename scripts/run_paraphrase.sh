#!/bin/bash
#
# Sufficient Statistics (Paraphrase Invariance) Test for Bayesian Benchmark
#
# Tests whether LLM inference is invariant to semantics-preserving paraphrase of
# the input evidence — a necessary property of coherent Bayesian reasoning.
# A cheap LLM independently paraphrases the aggregated claims K times; the majority
# vote across the K paraphrases approximates the model's "true" Bayesian posterior.
# Agreement between the original (bullet-point) result and this majority estimate
# indicates presentation-invariant inference.
#
# Usage:
#   bash scripts/run_paraphrase.sh [dataset_key] [num_paraphrases] [principal_model_key] [paraphrase_model_key]
#
#   dataset_key:            usmle_sample | usmle | mimiciv_demo  (default: usmle_sample)
#   num_paraphrases:        number of independent paraphrases per case  (default: 10)
#   principal_model_key:    deepseek | gpt | claude | gemini | ...  (default: deepseek)
#   paraphrase_model_key:   claude | gpt | llama-small | ...  (default: claude)
#
# Examples:
#   bash scripts/run_paraphrase.sh usmle_sample 10 deepseek claude
#   bash scripts/run_paraphrase.sh usmle 5 gpt claude
#   FORCE_RERUN=true bash scripts/run_paraphrase.sh usmle_sample 10 deepseek claude
#
# Environment variables (optional overrides):
#   PRINCIPAL_SERVER, MAX_WORKERS, FORCE_RERUN, CLAIM_FORMAT, MAX_CASES,
#   PARAPHRASE_TEMPERATURE
#
# Pipeline:
#   STAGE 1: Generate K paraphrases per case using the paraphrase model
#   STAGE 2: Run Bayesian principal inference on all (K+1) records per case
#   STAGE 3: Analyze agreement — original vs. paraphrase majority vote,
#            inner reliability across K paraphrases, belief delta

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
NUM_PARAPHRASES="${2:-10}"
PRINCIPAL_MODEL_KEY="${3:-deepseek}"
PARAPHRASE_MODEL_KEY="${4:-claude}"

# Configurable overrides
PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-openrouter}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-false}"
CLAIM_FORMAT="${CLAIM_FORMAT:-bullets}"
MAX_CASES="${MAX_CASES:-0}"  # 0 = no limit
PARAPHRASE_TEMPERATURE="${PARAPHRASE_TEMPERATURE:-0.7}"

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

# Resolve paraphrase model
PARAPHRASE_MODEL="${MODEL_MAP[$PARAPHRASE_MODEL_KEY]}"
if [ -z "$PARAPHRASE_MODEL" ]; then
    echo -e "${RED}Error: Unknown paraphrase model '${PARAPHRASE_MODEL_KEY}'${NC}"
    echo "Available: ${!MODEL_MAP[@]}"
    exit 1
fi

PRINCIPAL_SERVER="${PRINCIPAL_SERVER:-$(get_model_server $PRINCIPAL_MODEL_KEY)}"
PRINCIPAL_SGLANG_PORT="${PRINCIPAL_SGLANG_PORT:-$(get_principal_sglang_port $PRINCIPAL_MODEL_KEY)}"

# Paths — K and paraphrase model key are embedded so multiple runs coexist
PARAPHRASE_FILE="experiments/agents/${DATASET_KEY}/paraphrase_records_${PARAPHRASE_MODEL_KEY}_k${NUM_PARAPHRASES}.json"
PRINCIPAL_OUTPUT="experiments/principals/${DATASET_KEY}/principal_paraphrase_${PARAPHRASE_MODEL_KEY}_k${NUM_PARAPHRASES}.json"
ANALYSIS_OUTPUT="experiments/principals/${DATASET_KEY}/paraphrase_analysis_${PARAPHRASE_MODEL_KEY}_k${NUM_PARAPHRASES}.json"
AGGREGATED_INFO="experiments/aggregation/aggregated_factual.json"

# Validate aggregated info exists
if [ ! -f "$AGGREGATED_INFO" ]; then
    echo -e "${RED}Error: Aggregated claims not found: ${AGGREGATED_INFO}${NC}"
    echo -e "${YELLOW}Please run experiments/pipeline/aggregate_information.py first.${NC}"
    exit 1
fi

# Print configuration
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Sufficient Statistics (Paraphrase) Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "Dataset:              ${GREEN}${DATASET_KEY}${NC}"
echo -e "Num paraphrases (K):  ${GREEN}${NUM_PARAPHRASES}${NC}"
echo -e "Paraphrase model:     ${GREEN}${PARAPHRASE_MODEL}${NC}"
echo -e "Paraphrase temp:      ${GREEN}${PARAPHRASE_TEMPERATURE}${NC}"
echo -e "Claim format:         ${GREEN}${CLAIM_FORMAT}${NC}"
echo -e "Principal model:      ${GREEN}${PRINCIPAL_MODEL}${NC}"
echo -e "Principal server:     ${GREEN}${PRINCIPAL_SERVER}${NC}"
echo -e "Aggregated info:      ${YELLOW}${AGGREGATED_INFO}${NC}"
echo -e "Paraphrase records:   ${YELLOW}${PARAPHRASE_FILE}${NC}"
echo -e "Principal output:     ${YELLOW}${PRINCIPAL_OUTPUT}${NC}"
echo -e "Analysis output:      ${YELLOW}${ANALYSIS_OUTPUT}${NC}"
echo -e "${BLUE}================================================${NC}\n"

FORCE_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi

# ------------------------------------------------------------------
# STAGE 1: Generate K paraphrases per case
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 1] Generating ${NUM_PARAPHRASES} paraphrases per case...${NC}\n"

MAX_CASES_FLAG=""
if [ "${MAX_CASES}" -gt 0 ]; then
    MAX_CASES_FLAG="--max-cases ${MAX_CASES}"
fi

python pipeline/generate_paraphrases.py \
    --aggregated-info   "${AGGREGATED_INFO}" \
    --questions         "${QUESTIONS_FILE}" \
    --num-paraphrases   "${NUM_PARAPHRASES}" \
    --paraphrase-model  "${PARAPHRASE_MODEL}" \
    --claim-format      "${CLAIM_FORMAT}" \
    --temperature       "${PARAPHRASE_TEMPERATURE}" \
    --max-workers       "${MAX_WORKERS}" \
    --output            "${PARAPHRASE_FILE}" \
    ${MAX_CASES_FLAG} \
    ${FORCE_FLAG}

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Paraphrase generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Paraphrases generated${NC}\n"

# ------------------------------------------------------------------
# STAGE 2: Principal inference on all (K+1) records per case
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 2] Running principal inference on original + ${NUM_PARAPHRASES} paraphrases...${NC}\n"

if [ -n "$PRINCIPAL_SGLANG_PORT" ]; then
    python core/principal_inference.py \
        --principal-server      "${PRINCIPAL_SERVER}" \
        --principal-model       "${PRINCIPAL_MODEL}" \
        --principal-sglang-port "${PRINCIPAL_SGLANG_PORT}" \
        --agent-cache           "${PARAPHRASE_FILE}" \
        --output                "${PRINCIPAL_OUTPUT}" \
        --principal-types       bayesian_choices \
        --max-workers           "${MAX_WORKERS}" \
        ${FORCE_FLAG}
else
    python core/principal_inference.py \
        --principal-server  "${PRINCIPAL_SERVER}" \
        --principal-model   "${PRINCIPAL_MODEL}" \
        --agent-cache       "${PARAPHRASE_FILE}" \
        --output            "${PRINCIPAL_OUTPUT}" \
        --principal-types   bayesian_choices \
        --max-workers       "${MAX_WORKERS}" \
        ${FORCE_FLAG}
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Principal inference failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Principal inference complete${NC}\n"

# ------------------------------------------------------------------
# STAGE 3: Analyze paraphrase invariance
# ------------------------------------------------------------------
echo -e "${BLUE}[STAGE 3] Analyzing sufficient-statistics agreement...${NC}\n"

# principal_inference.py appends the principal type to the output filename
PRINCIPAL_RESULTS="${PRINCIPAL_OUTPUT%.json}_bayesian_choices.json"

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
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Sufficient Statistics Test Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "Paraphrase records: ${PARAPHRASE_FILE}"
echo -e "Results:            ${PRINCIPAL_RESULTS}"
echo -e "Analysis:           ${ANALYSIS_OUTPUT}"
echo -e "${GREEN}================================================${NC}"
