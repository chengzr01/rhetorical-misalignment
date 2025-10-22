# python -m experiments.generate \
#   --data-dir datasets/processed \
#   --output experiments/input/hypothesis.json \
#   --n-hypotheses 100 \
#   --age-window 10

# python main.py \
#   --agent-server sglang \
#   --agent-sglang-port 30002 \
#   --principal-model deepseek-ai/deepseek-v3.1 \
#   --input experiments/input/hypothesis.json \
#   --output experiments/output_sft/results.json

python analyze.py \
  --input experiments/output_sft \
  --output-dir experiments/analysis_sft

python analyze.py \
  --input experiments/output_dpo \
  --output-dir experiments/analysis_dpo