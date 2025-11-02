python main.py --agent-server nvidia \
  --agent-model deepseek-ai/deepseek-v3.1 \
  --input experiments/input/clinical_questions.json \
  --agent-cache experiments/cache/agent_large_dpo.json \
  --output experiments/output/large_dpo_bayesian.json

# python main.py --agent-server sglang --agent-sglang-port 30002 \
#   --input experiments/input/clinical_questions.json \
#   --agent-cache experiments/cache/agent_small_sft.json \
#   --output experiments/output/small_sft_bayesian.json