python main.py --agent-server sglang --agent-sglang-port 30001 \
  --input experiments/input/hypothesis_pmc.json \
  --hypothesis-type pmc \
  --output experiments/output/pmc_results_dpo.json

python main.py --agent-server sglang --agent-sglang-port 30002 \
  --input experiments/input/hypothesis_pmc.json \
  --hypothesis-type pmc \
  --output experiments/output/pmc_results_sft.json

python main.py --agent-server sglang --agent-sglang-port 30001 \
  --input experiments/input/hypothesis_mimic.json \
  --hypothesis-type mimic \
  --output experiments/output/mimic_results_dpo.json

python main.py --agent-server sglang --agent-sglang-port 30002 \
  --input experiments/input/hypothesis_mimic.json \
  --hypothesis-type mimic \
  --output experiments/output/mimic_results_sft.json