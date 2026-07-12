#!/bin/bash
#
# Run USMLE data-contamination probes using paraphrased questions and permuted options.
# Paraphrases require OpenRouter access (OPENROUTER_API_KEY).
# Usage: bash scripts/check_data_contamination.sh [model_key]
#
# Environment overrides (defaults shown):
#   BACKEND=openrouter
#   TEMPERATURE=0.0
#   MAX_WORKERS=<auto per model>
#   ELICIT_BELIEF=true
#   PROMPT_FILE=prompts/experiments/elicit.yaml
#   PARAPHRASE_MODEL=${PARAPHRASE_QUESTION_MODEL:-anthropic/claude-haiku-4.5}
#   PARAPHRASE_TEMPERATURE=0.5
#   PARAPHRASE_MAX_WORKERS=4
#   PARAPHRASE_RESUME=true
#   PARAPHRASE_FORCE=false
#   PERMUTE_SEEDS="13"
#   PERMUTE_OUTPUT_PREFIX=clinical_questions_usmle_sample_permuted
#   SKIP_PARAPHRASE=false
#   SKIP_PERMUTE=false
#
# Examples:
#   bash scripts/check_data_contamination.sh deepseek
#   PARAPHRASE_MODEL=google/gemini-1.5-flash bash scripts/check_data_contamination.sh llama
#   PERMUTE_SEEDS="13 27 99" bash scripts/check_data_contamination.sh gpt
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load shared model configuration
source "$SCRIPT_DIR/model_config.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_KEY="${1:-deepseek}"
MODEL="${MODEL_MAP[$MODEL_KEY]:-}"
if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Unknown model key '${MODEL_KEY}'. Available keys: ${!MODEL_MAP[@]}${NC}" >&2
    exit 1
fi

BACKEND="${BACKEND:-$(get_model_server "$MODEL_KEY")}" 
TEMPERATURE="${TEMPERATURE:-0.0}"
ELICIT_BELIEF="${ELICIT_BELIEF:-true}"
PROMPT_FILE="${PROMPT_FILE:-prompts/experiments/elicit.yaml}"

# Worker configuration per model (mirrors run_test.sh defaults)
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
    ["llama-base"]="32"
    ["olmo"]="32"
    ["olmo-sft"]="32"
    ["olmo-dpo"]="32"
    ["olmo-base"]="32"
    ["olmo-large"]="8"
    ["olmo-large-sft"]="8"
    ["olmo-large-dpo"]="8"
    ["llama-medium-sft"]="8"
    ["llama-medium-dpo"]="8"
)

MAX_WORKERS="${MAX_WORKERS:-${WORKER_MAP[$MODEL_KEY]:-8}}"

SELECTED_BACKEND="$BACKEND"
SGLANG_PORT="${SGLANG_PORT:-$(get_agent_sglang_port "$MODEL_KEY")}"

PARAPHRASE_MODEL="${PARAPHRASE_MODEL:-${PARAPHRASE_QUESTION_MODEL:-anthropic/claude-haiku-4.5}}"
PARAPHRASE_TEMPERATURE="${PARAPHRASE_TEMPERATURE:-0.5}"
PARAPHRASE_MAX_WORKERS="${PARAPHRASE_MAX_WORKERS:-4}"
PARAPHRASE_RESUME="${PARAPHRASE_RESUME:-true}"
PARAPHRASE_FORCE="${PARAPHRASE_FORCE:-false}"
SKIP_PARAPHRASE="${SKIP_PARAPHRASE:-false}"
SKIP_PERMUTE="${SKIP_PERMUTE:-false}"
PERMUTE_SEEDS_RAW="${PERMUTE_SEEDS:-13}"
PERMUTE_OUTPUT_PREFIX="${PERMUTE_OUTPUT_PREFIX:-clinical_questions_usmle_sample_permuted}"

# Resolve prompt path relative to repo root
PROMPT_FILE_PATH="$REPO_ROOT/$PROMPT_FILE"
if [[ ! -f "$PROMPT_FILE_PATH" ]]; then
    echo -e "${RED}Prompt file not found: $PROMPT_FILE${NC}" >&2
    exit 1
fi

# Common paths
INPUT_DATASET="$REPO_ROOT/experiments/questions/clinical_questions_usmle_sample.json"
if [[ ! -f "$INPUT_DATASET" ]]; then
    echo -e "${RED}Base USMLE dataset not found at $INPUT_DATASET${NC}" >&2
    exit 1
fi

QUESTIONS_VARIANT_DIR="$REPO_ROOT/experiments/questions/data_contamination"
PARAPHRASED_DIR="$QUESTIONS_VARIANT_DIR/paraphrased"
PERMUTED_DIR="$QUESTIONS_VARIANT_DIR/permuted"

TESTS_BASE_DIR="$REPO_ROOT/experiments/tests/data_contamination"
PARAPHRASED_TEST_DIR="$TESTS_BASE_DIR/paraphrased"
PERMUTED_TEST_DIR="$TESTS_BASE_DIR/permuted"

mkdir -p "$PARAPHRASED_DIR" "$PERMUTED_DIR" "$PARAPHRASED_TEST_DIR" "$PERMUTED_TEST_DIR"

SAFE_MODEL="${MODEL//\//-}"
SAFE_MODEL="${SAFE_MODEL//:/-}"
OUTPUT_FILENAME="test_usmle_sample_${SAFE_MODEL}"
if [[ "$ELICIT_BELIEF" == "true" ]]; then
    OUTPUT_FILENAME="${OUTPUT_FILENAME}_belief"
fi
OUTPUT_FILENAME="${OUTPUT_FILENAME}.json"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_header "Data Contamination Check"
echo -e "Model:         ${GREEN}${MODEL}${NC}"
echo -e "Backend:       ${GREEN}${SELECTED_BACKEND}${NC}"
echo -e "Temperature:   ${GREEN}${TEMPERATURE}${NC}"
echo -e "Max Workers:   ${GREEN}${MAX_WORKERS}${NC}"
echo -e "Elicit Belief: ${GREEN}${ELICIT_BELIEF}${NC}"
echo -e "Prompt File:   ${GREEN}${PROMPT_FILE}${NC}"
if [[ "$SELECTED_BACKEND" == "sglang" && -n "${SGLANG_PORT:-}" ]]; then
    echo -e "SGLang Port:   ${GREEN}${SGLANG_PORT}${NC}"
fi

RESULT_PATHS=()

# ------------------------------------------------------------------
# Paraphrased question variant
# ------------------------------------------------------------------
if [[ "$SKIP_PARAPHRASE" != "true" ]]; then
    PARAPHRASED_DATASET="$PARAPHRASED_DIR/clinical_questions_usmle_sample_paraphrased.json"
    echo -e "\n${BLUE}Generating paraphrased question dataset...${NC}"
    PARAPHRASE_ARGS=(
        "pipeline/paraphrase_usmle_questions.py"
        "--input" "$INPUT_DATASET"
        "--output" "$PARAPHRASED_DATASET"
        "--model" "$PARAPHRASE_MODEL"
        "--temperature" "$PARAPHRASE_TEMPERATURE"
        "--max-workers" "$PARAPHRASE_MAX_WORKERS"
    )
    if [[ "$PARAPHRASE_RESUME" == "true" ]]; then
        PARAPHRASE_ARGS+=("--resume")
    fi
    if [[ "$PARAPHRASE_FORCE" == "true" ]]; then
        PARAPHRASE_ARGS+=("--force")
    fi

    OPENROUTER_KEY_PRESENT="${OPENROUTER_API_KEY+x}"
    if [[ -z "$OPENROUTER_KEY_PRESENT" ]]; then
        echo -e "${YELLOW}Warning: OPENROUTER_API_KEY not set. Paraphrasing will fail unless provided.${NC}"
    fi

    (cd "$REPO_ROOT" && python "${PARAPHRASE_ARGS[@]}")

    echo -e "\n${BLUE}Evaluating model on paraphrased dataset...${NC}"
    TEST_ARGS=(
        "pipeline/test_usmle_sample.py"
        "--input" "$PARAPHRASED_DATASET"
        "--model" "$MODEL"
        "--backend" "$SELECTED_BACKEND"
        "--temperature" "$TEMPERATURE"
        "--max-workers" "$MAX_WORKERS"
        "--prompt" "$PROMPT_FILE"
        "--output-dir" "$PARAPHRASED_TEST_DIR"
    )
    if [[ "$ELICIT_BELIEF" == "true" ]]; then
        TEST_ARGS+=("--elicit-belief")
    fi
    if [[ "$SELECTED_BACKEND" == "sglang" && -n "${SGLANG_PORT:-}" ]]; then
        TEST_ARGS+=("--sglang-port" "$SGLANG_PORT")
    fi
    (cd "$REPO_ROOT" && python "${TEST_ARGS[@]}")
    RESULT_PATHS+=("Paraphrased -> experiments/tests/data_contamination/paraphrased/${OUTPUT_FILENAME}")
else
    echo -e "\n${YELLOW}Skipping paraphrased variant (SKIP_PARAPHRASE=true).${NC}"
fi

# ------------------------------------------------------------------
# Permuted option variants
# ------------------------------------------------------------------
if [[ "$SKIP_PERMUTE" != "true" ]]; then
    read -r -a PERMUTE_SEEDS <<< "$PERMUTE_SEEDS_RAW"
    echo -e "\n${BLUE}Generating permuted option datasets (seeds: ${PERMUTE_SEEDS[*]})...${NC}"
    for seed in "${PERMUTE_SEEDS[@]}"; do
        if [[ -z "$seed" ]]; then
            continue
        fi
        PERMUTED_DATASET="$PERMUTED_DIR/${PERMUTE_OUTPUT_PREFIX}_seed${seed}.json"
        (cd "$REPO_ROOT" && python pipeline/permute_usmle_options.py \
            --input "$INPUT_DATASET" \
            --output "$PERMUTED_DATASET" \
            --seed "$seed" \
            --overwrite)

        VARIANT_TEST_DIR="$PERMUTED_TEST_DIR/seed_${seed}"
        mkdir -p "$VARIANT_TEST_DIR"

        echo -e "${BLUE}Evaluating model on permuted dataset (seed ${seed})...${NC}"
        TEST_ARGS=(
            "pipeline/test_usmle_sample.py"
            "--input" "$PERMUTED_DATASET"
            "--model" "$MODEL"
            "--backend" "$SELECTED_BACKEND"
            "--temperature" "$TEMPERATURE"
            "--max-workers" "$MAX_WORKERS"
            "--prompt" "$PROMPT_FILE"
            "--output-dir" "$VARIANT_TEST_DIR"
        )
        if [[ "$ELICIT_BELIEF" == "true" ]]; then
            TEST_ARGS+=("--elicit-belief")
        fi
        if [[ "$SELECTED_BACKEND" == "sglang" && -n "${SGLANG_PORT:-}" ]]; then
            TEST_ARGS+=("--sglang-port" "$SGLANG_PORT")
        fi
        (cd "$REPO_ROOT" && python "${TEST_ARGS[@]}")
        RESULT_PATHS+=("Permuted (seed ${seed}) -> experiments/tests/data_contamination/permuted/seed_${seed}/${OUTPUT_FILENAME}")
    done
else
    echo -e "\n${YELLOW}Skipping permuted variants (SKIP_PERMUTE=true).${NC}"
fi

print_header "Finished"
for entry in "${RESULT_PATHS[@]}"; do
    echo -e "${GREEN}${entry}${NC}"
done
