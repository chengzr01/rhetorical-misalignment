#!/bin/bash

TULU_3_8B_SFT_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/snapshots/f2a0b46b0cfda21003c6141b1ff837b7e165524d"
TULU_3_8B_DPO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-DPO/snapshots/a7beb67e33ffd01cc87ac3b46cadc1000985b8db"

MODEL_TYPE=${1:-SFT}

case $MODEL_TYPE in
  DPO)
    MODEL_PATH=$TULU_3_8B_DPO_PATH
    ;;
  SFT)
    MODEL_PATH=$TULU_3_8B_SFT_PATH
    ;;
  *)
    echo "Unknown model type: $MODEL_TYPE"
    MODEL_PATH=$TULU_3_8B_SFT_PATH
    ;;
esac

echo "Launching server with model: $MODEL_TYPE"
python3 -m sglang.launch_server --model-path $MODEL_PATH --host 0.0.0.0 --port 30000
