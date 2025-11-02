python main.py --agent-server sglang --agent-sglang-port 30001 \
  --input experiments/input/clinical_questions.json \
  --principal-types all \
  --agent-cache experiments/cache/agent_small_dpo.json \
  --output experiments/output/small_dpo_all.json