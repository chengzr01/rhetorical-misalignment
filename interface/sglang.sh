#!/bin/bash

# Model alias to HuggingFace model ID mapping (matches scripts/model_config.sh)
declare -A MODEL_MAP=(
    ["llama-dpo"]="allenai/Llama-3.1-Tulu-3-8B-DPO"
    ["llama-sft"]="allenai/Llama-3.1-Tulu-3-8B-SFT"
    ["llama-base"]="meta-llama/Llama-3.1-8B"
    ["olmo"]="allenai/Olmo-3-7B-Instruct"
    ["olmo-sft"]="allenai/Olmo-3-7B-Instruct-SFT"
    ["olmo-dpo"]="allenai/Olmo-3-7B-Instruct-DPO"
    ["olmo-base"]="allenai/Olmo-3-1025-7B"
    ["olmo-large"]="allenai/Olmo-3.1-32B-Instruct"
    ["olmo-large-sft"]="allenai/Olmo-3.1-32B-Instruct-SFT"
    ["olmo-large-dpo"]="allenai/Olmo-3.1-32B-Instruct-DPO"
)

# Fixed port per model — each model always runs on the same port regardless of role
declare -A MODEL_PORT=(
    ["llama-dpo"]="60000"
    ["llama-sft"]="60001"
    ["llama-base"]="60002"
    ["olmo"]="60003"
    ["olmo-sft"]="60004"
    ["olmo-dpo"]="60005"
    ["olmo-base"]="60006"
    ["olmo-large"]="60007"
    ["olmo-large-sft"]="60008"
    ["olmo-large-dpo"]="60009"
)

MODEL_KEY=${1:-llama-sft}
PORT=${2:-${MODEL_PORT[$MODEL_KEY]}}

MODEL_ID="${MODEL_MAP[$MODEL_KEY]}"
if [ -z "$MODEL_ID" ]; then
    echo "Unknown model key: $MODEL_KEY"
    echo "Available keys: ${!MODEL_MAP[@]}"
    exit 1
fi

echo "Launching server: $MODEL_KEY ($MODEL_ID) on port $PORT"
python3 -m sglang.launch_server --model "$MODEL_ID" --host 0.0.0.0 --port "$PORT" --tp 2
