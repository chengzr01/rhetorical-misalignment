python experiments/test_usmle_sample.py \
        --model deepseek-ai/deepseek-v3.1 \
        --backend nvidia \
        --temperature 0.0 \
        --max-workers 8 \
        --elicit-belief \
        --prompt prompts/experiments/elicit.yaml

python experiments/test_usmle_sample.py \
        --model meta/llama-3.1-8b-instruct \
        --backend nvidia \
        --temperature 0.0 \
        --max-workers 32 \
        --elicit-belief \
        --prompt prompts/experiments/elicit.yaml

python experiments/test_usmle_sample.py \
        --model meta/llama-3.3-70b-instruct \
        --backend nvidia \
        --temperature 0.0 \
        --max-workers 8 \
        --elicit-belief \
        --prompt prompts/experiments/elicit.yaml

python experiments/test_usmle_sample.py \
        --model meta/llama-3.1-405b-instruct \
        --backend nvidia \
        --temperature 0.0 \
        --max-workers 4 \
        --elicit-belief \
        --prompt prompts/experiments/elicit.yaml