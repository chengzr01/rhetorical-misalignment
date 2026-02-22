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
)

# Server configuration for each model
# Models that require sglang server (HuggingFace models)
declare -A MODEL_SERVER=(
    ["llama-dpo"]="sglang"
    ["llama-sft"]="sglang"
)

# Default sglang ports for agent models
declare -A AGENT_SGLANG_PORT=(
    ["llama-dpo"]="30000"
    ["llama-sft"]="30000"
)

# Default sglang ports for principal models (different from agent to avoid conflicts)
declare -A PRINCIPAL_SGLANG_PORT=(
    ["llama-dpo"]="30001"
    ["llama-sft"]="30002"
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
