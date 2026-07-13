#!/bin/bash
#
# Run USMLE data-contamination probes using paraphrased question stems and paraphrased options.
# Generates both datasets (unless skipped) and evaluates one or more models on each variant.
# Paraphrasing requires OpenRouter access (OPENROUTER_API_KEY).
#
# Usage: bash scripts/check_data_contamination.sh [model_key ...]
#   - Without arguments evaluates default set: deepseek
#   - You can also set EVAL_MODEL_KEYS="deepseek llama" to override.
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

# Determine evaluation model keys
if [[ $# -gt 0 ]]; then
    MODEL_KEYS=("$@")
elif [[ -n "${EVAL_MODEL_KEYS:-}" ]]; then
    read -r -a MODEL_KEYS <<< "${EVAL_MODEL_KEYS}"
else
    MODEL_KEYS=(deepseek)
fi

if [[ ${#MODEL_KEYS[@]} -eq 0 ]]; then
    echo -e "${RED}No model keys provided for evaluation.${NC}" >&2
    exit 1
fi

for key in "${MODEL_KEYS[@]}"; do
    if [[ -z "${MODEL_MAP[$key]:-}" ]]; then
        echo -e "${RED}Unknown model key '${key}'. Available keys: ${!MODEL_MAP[@]}${NC}" >&2
        exit 1
    fi
done

# Global overrides & defaults
BACKEND_OVERRIDE="${BACKEND:-}"
TEMPERATURE="${TEMPERATURE:-0.0}"
ELICIT_BELIEF="${ELICIT_BELIEF:-true}"
PROMPT_FILE="${PROMPT_FILE:-prompts/experiments/elicit.yaml}"
MAX_WORKERS_OVERRIDE="${MAX_WORKERS:-}"
SGLANG_PORT_OVERRIDE="${SGLANG_PORT:-}"

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

# Paraphrase configuration
PARAPHRASE_MODEL="${PARAPHRASE_MODEL:-${PARAPHRASE_QUESTION_MODEL:-deepseek/deepseek-chat-v3.1}}"
PARAPHRASE_TEMPERATURE="${PARAPHRASE_TEMPERATURE:-0.5}"
PARAPHRASE_MAX_WORKERS="${PARAPHRASE_MAX_WORKERS:-4}"
PARAPHRASE_RESUME="${PARAPHRASE_RESUME:-true}"
PARAPHRASE_FORCE="${PARAPHRASE_FORCE:-false}"
PARAPHRASE_CHECKPOINT_INTERVAL="${PARAPHRASE_CHECKPOINT_INTERVAL:-10}"

PARAPHRASE_OPTIONS_MODEL="${PARAPHRASE_OPTIONS_MODEL:-$PARAPHRASE_MODEL}"
PARAPHRASE_OPTIONS_PROMPT="${PARAPHRASE_OPTIONS_PROMPT:-prompts/paraphrase/paraphrase_option.yaml}"
PARAPHRASE_OPTIONS_TEMPERATURE="${PARAPHRASE_OPTIONS_TEMPERATURE:-$PARAPHRASE_TEMPERATURE}"
PARAPHRASE_OPTIONS_MAX_WORKERS="${PARAPHRASE_OPTIONS_MAX_WORKERS:-$PARAPHRASE_MAX_WORKERS}"
PARAPHRASE_OPTIONS_RESUME="${PARAPHRASE_OPTIONS_RESUME:-true}"
PARAPHRASE_OPTIONS_FORCE="${PARAPHRASE_OPTIONS_FORCE:-false}"
PARAPHRASE_OPTIONS_CHECKPOINT_INTERVAL="${PARAPHRASE_OPTIONS_CHECKPOINT_INTERVAL:-10}"

SKIP_QUESTION_PARAPHRASE="${SKIP_PARAPHRASE:-false}"
SKIP_OPTION_PARAPHRASE="${SKIP_OPTION_PARAPHRASE:-${SKIP_PERMUTE:-false}}"

# Resolve prompt path relative to repo root
PROMPT_FILE_PATH="$REPO_ROOT/$PROMPT_FILE"
if [[ ! -f "$PROMPT_FILE_PATH" ]]; then
    echo -e "${RED}Prompt file not found: $PROMPT_FILE${NC}" >&2
    exit 1
fi

INPUT_DATASET="$REPO_ROOT/experiments/questions/clinical_questions_usmle_sample.json"
if [[ ! -f "$INPUT_DATASET" ]]; then
    echo -e "${RED}Base USMLE dataset not found at $INPUT_DATASET${NC}" >&2
    exit 1
fi

QUESTIONS_VARIANT_DIR="$REPO_ROOT/experiments/questions/data_contamination"
PARAPHRASED_DIR="$QUESTIONS_VARIANT_DIR/paraphrased_questions"
OPTIONS_PARAPHRASED_DIR="$QUESTIONS_VARIANT_DIR/paraphrased_options"

PARAPHRASED_DATASET="$PARAPHRASED_DIR/clinical_questions_usmle_sample_paraphrased.json"
OPTIONS_PARAPHRASED_DATASET="$OPTIONS_PARAPHRASED_DIR/clinical_questions_usmle_sample_options_paraphrased.json"

TESTS_BASE_DIR="$REPO_ROOT/experiments/tests/data_contamination"
PARAPHRASED_TEST_DIR="$TESTS_BASE_DIR/paraphrased_questions"
OPTIONS_TEST_DIR="$TESTS_BASE_DIR/paraphrased_options"

mkdir -p "$PARAPHRASED_DIR" "$OPTIONS_PARAPHRASED_DIR" "$PARAPHRASED_TEST_DIR" "$OPTIONS_TEST_DIR"

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_header "Data Contamination Setup"
echo -e "Question paraphrase model: ${GREEN}${PARAPHRASE_MODEL}${NC}"
echo -e "Option paraphrase model:   ${GREEN}${PARAPHRASE_OPTIONS_MODEL}${NC}"
echo -e "Evaluation models:        ${GREEN}${MODEL_KEYS[*]}${NC}"
echo -e "Prompt file:              ${GREEN}${PROMPT_FILE}${NC}"
echo -e "Temperature:              ${GREEN}${TEMPERATURE}${NC}"

OPENROUTER_KEY_PRESENT="${OPENROUTER_API_KEY+x}"
if [[ -z "$OPENROUTER_KEY_PRESENT" ]]; then
    echo -e "${YELLOW}Warning: OPENROUTER_API_KEY not set. Paraphrasing requests will fail unless provided.${NC}"
fi

# ------------------------------------------------------------------
# Paraphrased question variant
# ------------------------------------------------------------------
if [[ "$SKIP_QUESTION_PARAPHRASE" != "true" ]]; then
    if [[ -f "$PARAPHRASED_DATASET" && "$PARAPHRASE_RESUME" == "true" && "$PARAPHRASE_FORCE" != "true" ]]; then
        echo -e "\n${YELLOW}Paraphrased question dataset already exists; skipping regeneration (set PARAPHRASE_FORCE=true to rebuild).${NC}"
    else
        echo -e "\n${BLUE}Generating paraphrased question dataset...${NC}"
        PARAPHRASE_ARGS=(
            "pipeline/paraphrase_usmle_questions.py"
            "--input" "$INPUT_DATASET"
            "--output" "$PARAPHRASED_DATASET"
            "--model" "$PARAPHRASE_MODEL"
            "--temperature" "$PARAPHRASE_TEMPERATURE"
            "--max-workers" "$PARAPHRASE_MAX_WORKERS"
        )
        if [[ -n "${PARAPHRASE_PROMPT:-}" ]]; then
            PARAPHRASE_ARGS+=("--prompt" "$PARAPHRASE_PROMPT")
        fi
        if [[ "$PARAPHRASE_RESUME" == "true" ]]; then
            PARAPHRASE_ARGS+=("--resume")
        fi
        if [[ "$PARAPHRASE_FORCE" == "true" ]]; then
            PARAPHRASE_ARGS+=("--force")
        fi
        if [[ -n "$PARAPHRASE_CHECKPOINT_INTERVAL" ]]; then
            PARAPHRASE_ARGS+=("--checkpoint-interval" "$PARAPHRASE_CHECKPOINT_INTERVAL")
        fi
        (cd "$REPO_ROOT" && python3 "${PARAPHRASE_ARGS[@]}")
    fi
else
    echo -e "\n${YELLOW}Skipping question paraphrasing (SKIP_PARAPHRASE=true).${NC}"
fi

if [[ ! -f "$PARAPHRASED_DATASET" ]]; then
    echo -e "${YELLOW}Question paraphrased dataset missing at $PARAPHRASED_DATASET${NC}"
fi

# ------------------------------------------------------------------
# Paraphrased option variant (no shuffling)
# ------------------------------------------------------------------
if [[ "$SKIP_OPTION_PARAPHRASE" != "true" ]]; then
    if [[ -f "$OPTIONS_PARAPHRASED_DATASET" && "$PARAPHRASE_OPTIONS_RESUME" == "true" && "$PARAPHRASE_OPTIONS_FORCE" != "true" ]]; then
        echo -e "\n${YELLOW}Paraphrased option dataset already exists; skipping regeneration (set PARAPHRASE_OPTIONS_FORCE=true to rebuild).${NC}"
    else
        echo -e "\n${BLUE}Generating paraphrased option dataset...${NC}"
        OPTION_ARGS=(
            "pipeline/paraphrase_usmle_options.py"
            "--input" "$INPUT_DATASET"
            "--output" "$OPTIONS_PARAPHRASED_DATASET"
            "--model" "$PARAPHRASE_OPTIONS_MODEL"
            "--prompt" "$PARAPHRASE_OPTIONS_PROMPT"
            "--temperature" "$PARAPHRASE_OPTIONS_TEMPERATURE"
            "--max-workers" "$PARAPHRASE_OPTIONS_MAX_WORKERS"
        )
        if [[ "$PARAPHRASE_OPTIONS_RESUME" == "true" ]]; then
            OPTION_ARGS+=("--resume")
        fi
        if [[ "$PARAPHRASE_OPTIONS_FORCE" == "true" ]]; then
            OPTION_ARGS+=("--force")
        fi
        if [[ -n "$PARAPHRASE_OPTIONS_CHECKPOINT_INTERVAL" ]]; then
            OPTION_ARGS+=("--checkpoint-interval" "$PARAPHRASE_OPTIONS_CHECKPOINT_INTERVAL")
        fi
        (cd "$REPO_ROOT" && python3 "${OPTION_ARGS[@]}")
    fi
else
    echo -e "\n${YELLOW}Skipping option paraphrasing (SKIP_OPTION_PARAPHRASE=true).${NC}"
fi

if [[ ! -f "$OPTIONS_PARAPHRASED_DATASET" ]]; then
    echo -e "${YELLOW}Option paraphrased dataset missing at $OPTIONS_PARAPHRASED_DATASET${NC}"
fi

# Collect available datasets
declare -a DATASET_LABELS=()
declare -a DATASET_FILES=()
declare -a DATASET_TEST_BASE=()

declare -A DATASET_DESCRIPTIONS=(
    ["question_paraphrase"]="Paraphrased question stems"
    ["option_paraphrase"]="Paraphrased answer options"
)

if [[ -f "$PARAPHRASED_DATASET" ]]; then
    DATASET_LABELS+=("question_paraphrase")
    DATASET_FILES+=("$PARAPHRASED_DATASET")
    DATASET_TEST_BASE+=("$PARAPHRASED_TEST_DIR")
fi

if [[ -f "$OPTIONS_PARAPHRASED_DATASET" ]]; then
    DATASET_LABELS+=("option_paraphrase")
    DATASET_FILES+=("$OPTIONS_PARAPHRASED_DATASET")
    DATASET_TEST_BASE+=("$OPTIONS_TEST_DIR")
fi

if [[ ${#DATASET_LABELS[@]} -eq 0 ]]; then
    echo -e "${RED}No datasets available for evaluation. Exiting.${NC}" >&2
    exit 1
fi

print_header "Evaluation"
RESULT_PATHS=()

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
    MODEL="${MODEL_MAP[$MODEL_KEY]}"
    BACKEND_TO_USE="$BACKEND_OVERRIDE"
    if [[ -z "$BACKEND_TO_USE" ]]; then
        BACKEND_TO_USE="$(get_model_server "$MODEL_KEY")"
    fi

    MODEL_MAX_WORKERS="$MAX_WORKERS_OVERRIDE"
    if [[ -z "$MODEL_MAX_WORKERS" ]]; then
        MODEL_MAX_WORKERS="${WORKER_MAP[$MODEL_KEY]:-8}"
    fi

    MODEL_SGLANG_PORT="$SGLANG_PORT_OVERRIDE"
    if [[ -z "$MODEL_SGLANG_PORT" && "$BACKEND_TO_USE" == "sglang" ]]; then
        MODEL_SGLANG_PORT="$(get_agent_sglang_port "$MODEL_KEY")"
    fi

    SAFE_MODEL="${MODEL//\//-}"
    SAFE_MODEL="${SAFE_MODEL//:/-}"
    OUTPUT_FILENAME="test_usmle_sample_${SAFE_MODEL}"
    if [[ "$ELICIT_BELIEF" == "true" ]]; then
        OUTPUT_FILENAME="${OUTPUT_FILENAME}_belief"
    fi
    OUTPUT_FILENAME="${OUTPUT_FILENAME}.json"

    echo -e "\n${BLUE}Model: ${GREEN}${MODEL_KEY}${NC} (${MODEL})"
    echo -e "Backend: ${GREEN}${BACKEND_TO_USE}${NC} | Max workers: ${GREEN}${MODEL_MAX_WORKERS}${NC} | Elicit belief: ${GREEN}${ELICIT_BELIEF}${NC}"
    if [[ "$BACKEND_TO_USE" == "sglang" && -n "$MODEL_SGLANG_PORT" ]]; then
        echo -e "SGLang Port: ${GREEN}${MODEL_SGLANG_PORT}${NC}"
    fi

    for idx in "${!DATASET_LABELS[@]}"; do
        VARIANT_KEY="${DATASET_LABELS[$idx]}"
        DATASET_PATH="${DATASET_FILES[$idx]}"
        BASE_TEST_DIR="${DATASET_TEST_BASE[$idx]}"
        DESCRIPTION="${DATASET_DESCRIPTIONS[$VARIANT_KEY]}"

        MODEL_TEST_DIR="${BASE_TEST_DIR}/${MODEL_KEY}"
        mkdir -p "$MODEL_TEST_DIR"

        echo -e "${BLUE}  → Evaluating ${DESCRIPTION}${NC}"
        TEST_ARGS=(
            "pipeline/test_usmle_sample.py"
            "--input" "$DATASET_PATH"
            "--model" "$MODEL"
            "--backend" "$BACKEND_TO_USE"
            "--temperature" "$TEMPERATURE"
            "--max-workers" "$MODEL_MAX_WORKERS"
            "--prompt" "$PROMPT_FILE"
            "--output-dir" "$MODEL_TEST_DIR"
        )
        if [[ "$ELICIT_BELIEF" == "true" ]]; then
            TEST_ARGS+=("--elicit-belief")
        fi
        if [[ "$BACKEND_TO_USE" == "sglang" && -n "$MODEL_SGLANG_PORT" ]]; then
            TEST_ARGS+=("--sglang-port" "$MODEL_SGLANG_PORT")
        fi

        (cd "$REPO_ROOT" && python3 "${TEST_ARGS[@]}")

        RESULT_PATH="$MODEL_TEST_DIR/$OUTPUT_FILENAME"
        if [[ -f "$RESULT_PATH" ]]; then
            RELATIVE_RESULT="${RESULT_PATH#$REPO_ROOT/}"
            RESULT_PATHS+=("${VARIANT_KEY} | ${MODEL_KEY} -> ${RELATIVE_RESULT}")
        else
            echo -e "${YELLOW}Warning: Expected output missing at $RESULT_PATH${NC}"
        fi
    done

    echo -e "${GREEN}Completed evaluations for ${MODEL_KEY}.${NC}"

    if [[ "$BACKEND_TO_USE" == "sglang" ]]; then
        sleep 1  # brief pause between heavy models
    fi

done

print_header "Evaluation Outputs"
for entry in "${RESULT_PATHS[@]}"; do
    echo -e "${GREEN}${entry}${NC}"
done

print_header "Next Steps"
echo -e "Inspect the JSON outputs above or summarise results with auxiliary scripts as needed."
