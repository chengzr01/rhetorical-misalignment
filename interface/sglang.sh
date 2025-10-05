#!/bin/bash

LLAMA_3_1_8B_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LLAMA_3_1_8B_INSTRUCT_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

conda activate misalignment
python3 -m sglang.launch_server --model-path $LLAMA_3_1_8B_PATH --host 0.0.0.0 --port 30000
# python3 -m sglang.launch_server --model-path $LLAMA_8B_INSTRUCT_PATH --host 0.0.0.0 --port 30001