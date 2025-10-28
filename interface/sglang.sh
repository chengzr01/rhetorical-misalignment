LLAMA_3_1_8B_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LLAMA_3_1_8B_INSTRUCT_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
TULU_3_8B_SFT_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/snapshots/f2a0b46b0cfda21003c6141b1ff837b7e165524d"
TULU_3_8B_DPO_PATH="/work/hdd/bdhh/ziruic4/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-DPO/snapshots/a7beb67e33ffd01cc87ac3b46cadc1000985b8db"
python3 -m sglang.launch_server --model-path $TULU_3_8B_SFT_PATH --host 0.0.0.0 --port 30001
# python3 -m sglang.launch_server --model-path $TULU_3_8B_DPO_PATH --host 0.0.0.0 --port 30000
