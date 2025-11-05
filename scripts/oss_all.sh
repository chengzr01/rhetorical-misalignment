python main.py --agent-server nvidia \
  --agent-model openai/gpt-oss-120b \
  --input experiments/input/clinical_questions.json \
  --agent-cache experiments/cache/agent_oss.json \
  --output experiments/output/principal_oss_all.json \
  --principal-types all \
  --principal-workers 4 