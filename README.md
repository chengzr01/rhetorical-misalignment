# Persuasive Misalignment

LLM-driven principal-agent framework for clinical decision-making.

## Setup

```bash
pip install polars pyarrow openai pyyaml
export NVIDIA_API_KEY="your-key-here"  # For NVIDIA backend (default)
# OR
export OPENROUTER_API_KEY="your-key-here"  # For OpenRouter backend
```

## Usage

### 1. Generate Hypotheses

```bash
python -m experiments.generate \
  --data-dir datasets/processed \
  --output experiments/input/hypothesis.json \
  --n-hypotheses 5 \
  --age-window 10
```

Parameters:
- `--server`: `openrouter` or `sglang`
- `--model`: Model identifier
- `--data-dir`: Path to datasets
- `--output`: Output file path
- `--n-hypotheses`: Number to generate
- `--age-window`: Age range (±N years)

### 2. Run Experiments

```bash
python main.py \
  --server nvidia \
  --agent-model meta/llama-3.3-70b-instruct \
  --principal-model deepseek-ai/deepseek-r1 \
  --input experiments/input/hypothesis.json \
  --output experiments/output/results.json
```

Parameters:
- `--server`: LLM backend (`nvidia`, `openrouter`, or `sglang`, default: `nvidia`)
- `--agent-model`: Model for agents (default: `deepseek-ai/deepseek-r1`)
- `--principal-model`: Model for principals (default: `deepseek-ai/deepseek-r1`)
- `--input`: Input hypothesis file
- `--output`: Output results file

## Structure

```
agents/         # Base, agent, principal classes
interface/      # LLM clients
prompts/        # Agent and principal prompts
experiments/    # Hypothesis generation and verification
datasets/       # Patient data
main.py         # Simulation script
```

## Data Sources

+ Dataset: `/projects/bdhh/haopeng/physionet.org`
+ Tables: `hosp.patients`, `hosp.admissions`, `hosp.emar`, `hosp.emar_detail`, `hosp.labevents`, `hosp.d_labitems`
+ Quota: `/work/hdd/bdhh/physionet.org`
