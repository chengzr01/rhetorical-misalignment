#!/bin/bash
#
# Model Configuration File
# This file defines model aliases and their corresponding keys for OpenRouter/HuggingFace
# Can be sourced by different experiment scripts
#
# Usage in other scripts:
#   source scripts/model_config.sh
#   MODEL="${MODEL_MAP[$MODEL_KEY]}"
#

# Model alias to OpenRouter/HuggingFace key mapping
declare -A MODEL_MAP=(
    # OpenRouter models
    ["deepseek"]="deepseek/deepseek-chat-v3.1"
    ["gemini"]="google/gemini-2.5-pro"
    ["gpt"]="openai/gpt-5.1"
    ["claude"]="anthropic/claude-haiku-4.5"
    ["deepseek-llama"]="deepseek/deepseek-r1-distill-llama-70b"
    ["llama"]="meta-llama/llama-3.3-70b-instruct"
    ["llama-small"]="meta-llama/llama-3.1-8b-instruct"
    ["llama-large"]="meta-llama/llama-3.1-405b-instruct"
    ["qwen"]="qwen/qwen-2.5-7b-instruct"
    ["mistral"]="mistralai/mistral-7b-instruct"

    # HuggingFace models (require sglang server)
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
    ["llama-medium-sft"]="allenai/Llama-3.1-Tulu-3-70B-SFT"
    ["llama-medium-dpo"]="allenai/Llama-3.1-Tulu-3-70B-DPO"
)

# Server configuration for each model
# Models that require sglang server (HuggingFace models)
declare -A MODEL_SERVER=(
    ["llama-dpo"]="sglang"
    ["llama-sft"]="sglang"
    ["llama-base"]="sglang"
    ["olmo"]="sglang"
    ["olmo-sft"]="sglang"
    ["olmo-dpo"]="sglang"
    ["olmo-base"]="sglang"
    ["olmo-large"]="sglang"
    ["olmo-large-sft"]="sglang"
    ["olmo-large-dpo"]="sglang"
    ["llama-medium-sft"]="sglang"
    ["llama-medium-dpo"]="sglang"
)

# Fixed sglang port per model — each model always runs on the same port regardless of role
# (matches interface/sglang.sh MODEL_PORT assignments)
declare -A AGENT_SGLANG_PORT=(
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
    ["llama-medium-sft"]="60010"
    ["llama-medium-dpo"]="60011"
)

declare -A PRINCIPAL_SGLANG_PORT=(
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
    ["llama-medium-sft"]="60010"
    ["llama-medium-dpo"]="60011"
)

# Helper function to get server for a model
get_model_server() {
    local model_key="$1"
    echo "${MODEL_SERVER[$model_key]:-openrouter}"
}

# Helper function to get agent sglang port
get_agent_sglang_port() {
    local model_key="$1"
    echo "${AGENT_SGLANG_PORT[$model_key]}"
}

# Helper function to get principal sglang port
get_principal_sglang_port() {
    local model_key="$1"
    echo "${PRINCIPAL_SGLANG_PORT[$model_key]}"
}
