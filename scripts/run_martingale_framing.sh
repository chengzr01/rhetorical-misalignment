#!/bin/bash
#
# Martingale Reliability Test — Per-Agent Framing Responses
#
# Runs two stages for each agent model:
#   STAGE 1: Extract individual claims from each agent's framing response
#            (framing_<agent>_gt_factual_agg.json) using an LLM.
#            Claims are copied verbatim — the LLM only splits, not rewrites.
#   STAGE 2: Run the martingale permutation test on those per-agent claims.
#
# Usage:
#   bash scripts/run_martingale_framing.sh [dataset_key] [num_permutations] [principal_model_key]
#
#   dataset_key:         usmle_sample | usmle | mimiciv_demo  (default: usmle_sample)
#   num_permutations:    number of random claim orderings per case (default: 10)
#   principal_model_key: deepseek | gpt | claude | gemini | ...  (default: deepseek)
#
# Examples:
#   bash scripts/run_martingale_framing.sh
#   bash scripts/run_martingale_framing.sh usmle_sample 10 deepseek
#   AGENTS="gpt" bash scripts/run_martingale_framing.sh usmle_sample 3 deepseek
#   FORCE_RERUN=true bash scripts/run_martingale_framing.sh usmle_sample 3 deepseek
#
# Environment variables (optional overrides):
#   AGENTS           space-separated list of agent keys to run (default: all available)
#   EXTRACT_MODEL    OpenRouter model used for claim extraction
#   EXTRACT_WORKERS  parallel API calls for extraction  (default: 8)
#   MAX_WORKERS      parallel API calls for principal inference (default: 8)
#   FORCE_RERUN      re-run even if output files already exist (default: false)
#   PRINCIPAL_SERVER openrouter | sglang  (default: openrouter)
#   MAX_CASES        limit cases for quick testing, 0 = no limit (default: 0)

set -e

source "$(dirname "$0")/model_config.sh"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Arguments
DATASET_KEY="${1:-usmle_sample}"
NUM_PERMUTATIONS="${2:-10}"
PRINCIPAL_MODEL_KEY="${3:-deepseek}"

# Configurable overrides
EXTRACT_MODEL="${EXTRACT_MODEL:-deepseek/deepseek-chat-v3.1}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-8}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_RERUN="${FORCE_RERUN:-false}"
MAX_CASES="${MAX_CASES:-0}"

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

# Determine which agents to run
FRAMING_DIR="experiments/agents/${DATASET_KEY}"
if [ -n "$AGENTS" ]; then
    # User-specified subset
    AGENT_LIST=($AGENTS)
else
    # Auto-discover from framing files present in the dataset directory
    AGENT_LIST=()
    for f in "${FRAMING_DIR}"/framing_*_gt_factual_agg.json; do
        [ -f "$f" ] || continue
        basename "$f" | sed 's/framing_\(.*\)_gt_factual_agg\.json/\1/'
    done | while read -r agent; do
        AGENT_LIST+=("$agent")
    done
    # Re-read since the subshell above doesn't persist the array
    mapfile -t AGENT_LIST < <(
        for f in "${FRAMING_DIR}"/framing_*_gt_factual_agg.json; do
            [ -f "$f" ] || continue
            basename "$f" | sed 's/framing_\(.*\)_gt_factual_agg\.json/\1/'
        done
    )
fi

if [ ${#AGENT_LIST[@]} -eq 0 ]; then
    echo -e "${RED}Error: No framing files found in ${FRAMING_DIR}${NC}"
    exit 1
fi

FORCE_FLAG=""
MAX_CASES_FLAG=""
if [ "$FORCE_RERUN" = "true" ]; then
    FORCE_FLAG="--force"
fi
if [ "${MAX_CASES}" -gt 0 ]; then
    MAX_CASES_FLAG="--max-cases ${MAX_CASES}"
fi

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Martingale Test — Per-Agent Framing${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Dataset:          ${GREEN}${DATASET_KEY}${NC}"
echo -e "Agents:           ${GREEN}${AGENT_LIST[*]}${NC}"
echo -e "Permutations (K): ${GREEN}${NUM_PERMUTATIONS}${NC}"
echo -e "Principal model:  ${GREEN}${PRINCIPAL_MODEL}${NC}"
echo -e "Extract model:    ${GREEN}${EXTRACT_MODEL}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# ------------------------------------------------------------------
# Per-agent loop
# ------------------------------------------------------------------
FAILED_AGENTS=()

for AGENT in "${AGENT_LIST[@]}"; do
    echo -e "${CYAN}########################################${NC}"
    echo -e "${CYAN}Agent: ${AGENT}${NC}"
    echo -e "${CYAN}########################################${NC}\n"

    FRAMING_FILE="${FRAMING_DIR}/framing_${AGENT}_gt_factual_agg.json"
    AGGREGATED_OUT="experiments/aggregation/aggregated_${AGENT}_framing.json"

    # Validate framing file exists
    if [ ! -f "$FRAMING_FILE" ]; then
        echo -e "${RED}  Skipping ${AGENT}: framing file not found: ${FRAMING_FILE}${NC}\n"
        FAILED_AGENTS+=("${AGENT}(missing_framing_file)")
        continue
    fi

    # ---- STAGE 1: Extract claims ----------------------------------------
    echo -e "${BLUE}  [STAGE 1] Extracting claims from agent response...${NC}"

    if [ -f "$AGGREGATED_OUT" ] && [ "$FORCE_RERUN" != "true" ]; then
        echo -e "${YELLOW}  Skipping extraction — output already exists: ${AGGREGATED_OUT}${NC}"
        echo -e "${YELLOW}  Set FORCE_RERUN=true to re-extract.${NC}\n"
    else
        python pipeline/extract_agent_claims.py \
            --input      "${FRAMING_FILE}" \
            --output     "${AGGREGATED_OUT}" \
            --agent-key  "${AGENT}" \
            --model      "${EXTRACT_MODEL}" \
            --max-workers "${EXTRACT_WORKERS}" \
            ${MAX_CASES_FLAG} \
            ${FORCE_FLAG}

        if [ $? -ne 0 ]; then
            echo -e "${RED}  Error: Claim extraction failed for ${AGENT}${NC}\n"
            FAILED_AGENTS+=("${AGENT}(extraction_failed)")
            continue
        fi
        echo -e "${GREEN}  ✓ Claims extracted → ${AGGREGATED_OUT}${NC}\n"
    fi

    # ---- STAGE 2: Martingale test ----------------------------------------
    echo -e "${BLUE}  [STAGE 2] Running martingale test...${NC}\n"

    AGGREGATED_INFO="${AGGREGATED_OUT}" \
    FORCE_RERUN="${FORCE_RERUN}" \
    MAX_WORKERS="${MAX_WORKERS}" \
    MAX_CASES="${MAX_CASES}" \
    bash scripts/run_martingale.sh \
        "${DATASET_KEY}" \
        "${NUM_PERMUTATIONS}" \
        "${PRINCIPAL_MODEL_KEY}" \
        "${AGENT}"

    if [ $? -ne 0 ]; then
        echo -e "${RED}  Error: Martingale test failed for ${AGENT}${NC}\n"
        FAILED_AGENTS+=("${AGENT}(martingale_failed)")
        continue
    fi

    echo -e "${GREEN}  ✓ Martingale complete for ${AGENT}${NC}\n"
done

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}All Agents Complete${NC}"
echo -e "${BLUE}========================================${NC}"

SUCCEEDED=()
for AGENT in "${AGENT_LIST[@]}"; do
    FAILED=false
    for F in "${FAILED_AGENTS[@]}"; do
        [[ "$F" == "${AGENT}("* ]] && FAILED=true && break
    done
    if [ "$FAILED" = "false" ]; then
        SUCCEEDED+=("$AGENT")
        ANALYSIS="experiments/principals/${DATASET_KEY}/martingale_analysis_${AGENT}_k${NUM_PERMUTATIONS}.json"
        echo -e "  ${GREEN}✓${NC} ${AGENT} → ${ANALYSIS}"
    fi
done

if [ ${#FAILED_AGENTS[@]} -gt 0 ]; then
    echo -e "\n  ${RED}Failed:${NC}"
    for F in "${FAILED_AGENTS[@]}"; do
        echo -e "  ${RED}✗ ${F}${NC}"
    done
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
