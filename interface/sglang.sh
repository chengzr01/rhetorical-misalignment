#!/bin/bash

TULU_3_8B_SFT_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/snapshots/f2a0b46b0cfda21003c6141b1ff837b7e165524d"
TULU_3_8B_DPO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-DPO/snapshots/a7beb67e33ffd01cc87ac3b46cadc1000985b8db"
# LLAMA_3_8B_DPO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--princeton-nlp--Llama-3-Instruct-8B-DPO/snapshots/56dd2caf03cbe9f0a7c1c1024fe5e80c4ca3f9e8"
# LLAMA_3_8B_KTO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--princeton-nlp--Llama-3-Instruct-8B-KTO/snapshots/fdf57c5ad8fa80d60ff8326f645f127cae2ffd39"
# MISTRAL_7B_DPO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--princeton-nlp--Mistral-7B-Instruct-DPO/snapshots/3ebb003172fb8c4c098b570915adf91be362731b"
# MISTRAL_7B_KTO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--princeton-nlp--Mistral-7B-Instruct-KTO/snapshots/e2f0b88d0b0608921bf64a89fb84f374037fbe9e"

MODEL_TYPE=${1:-SFT}

case $MODEL_TYPE in
  DPO)
    MODEL_PATH=$TULU_3_8B_DPO_PATH
    ;;
  SFT)
    MODEL_PATH=$TULU_3_8B_SFT_PATH
    ;;
  LLAMA_DPO)
    MODEL_PATH=$LLAMA_3_8B_DPO_PATH
    ;;
  LLAMA_KTO)
    MODEL_PATH=$LLAMA_3_8B_KTO_PATH
    ;;
  MISTRAL_DPO)
    MODEL_PATH=$MISTRAL_7B_DPO_PATH
    ;;
  MISTRAL_KTO)
    MODEL_PATH=$MISTRAL_7B_KTO_PATH
    ;;
  *)
    echo "Unknown model type: $MODEL_TYPE"
    echo "Available model types: DPO, SFT, LLAMA_DPO, LLAMA_KTO, MISTRAL_DPO, MISTRAL_KTO"
    exit 1
    ;;
esac

echo "Launching server with model: $MODEL_TYPE"
python3 -m sglang.launch_server --model-path $MODEL_PATH --host 0.0.0.0 --port 30000
