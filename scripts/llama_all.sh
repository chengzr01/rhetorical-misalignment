python main.py --agent-server nvidia \
  --agent-model meta/llama-3.3-70b-instruct \
  --input experiments/input/clinical_questions.json \
  --agent-cache experiments/cache/agent_llama.json \
  --output experiments/output/principal_llama_all.json \
  --principal-types all \
  --principal-workers 4