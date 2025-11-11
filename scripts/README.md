# Inference Scripts

Clean and minimal script for running agent and principal inferences.

## Architecture

```
scripts/
├── run_inference.sh       # Main runner script (handles all logic)
├── README.md              # This file
└── QUICK_REFERENCE.md     # Quick command reference
```

## Quick Start

The main script handles all inference combinations with a simple syntax:

```bash
# Syntax: ./scripts/run_inference.sh <model> <dataset> <principal_types>

# MIMIC dataset examples
./scripts/run_inference.sh llama mimic bayesian        # Llama 70B, Bayesian
./scripts/run_inference.sh llama mimic all             # Llama 70B, all principals
./scripts/run_inference.sh deepseek mimic bayesian     # DeepSeek, Bayesian
./scripts/run_inference.sh llama-small mimic bayesian  # Llama 8B, Bayesian

# USMLE dataset examples
./scripts/run_inference.sh llama usmle bayesian        # Llama 70B, Bayesian
./scripts/run_inference.sh llama usmle all             # Llama 70B, all principals
./scripts/run_inference.sh deepseek usmle all          # DeepSeek, all principals
```

## Available Options

### Models
- `deepseek` - DeepSeek-V3.1
- `llama` - Llama 3.3 70B Instruct
- `llama-small` - Llama 3.1 8B Instruct
- `oss` - OpenAI GPT OSS 120B

### Datasets
- `mimic` - MIMIC-IV clinical decision-making
- `usmle` - USMLE medical exam questions

### Principal Types
- `bayesian` - Bayesian reasoning principal
- `all` - All principal types (anchoring, availability, confirmation, conservatism, overconfidence, prospect)

## Environment Variables

Customize behavior with environment variables:

```bash
# Server backends (default: nvidia)
export AGENT_SERVER=nvidia
export PRINCIPAL_SERVER=nvidia

# Principal model (default: deepseek-ai/deepseek-v3.1)
export PRINCIPAL_MODEL=deepseek-ai/deepseek-v3.1

# Parallelization
export MAX_WORKERS=8              # Agent inference workers
export PRINCIPAL_WORKERS=4        # Principal inference workers

# Run with custom settings
MAX_WORKERS=16 bash scripts/llama.sh
```

## Output Files

Results are automatically saved to:

```
experiments/
├── cache/
│   ├── agent_<model>_<dataset>.json      # Agent inference cache
│   └── ...
└── output/
    ├── principal_<model>_<dataset>.json       # Results (Bayesian)
    ├── principal_<model>_<dataset>_all.json   # Results (all principals)
    └── ...
```

## Examples

### Run Different Models on MIMIC

```bash
./scripts/run_inference.sh llama mimic bayesian        # Llama 70B
./scripts/run_inference.sh llama-small mimic bayesian  # Llama 8B (faster)
./scripts/run_inference.sh deepseek mimic bayesian     # DeepSeek
./scripts/run_inference.sh oss mimic bayesian          # GPT OSS
```

### Run with All Principal Types

```bash
./scripts/run_inference.sh llama mimic all       # Takes longer, tests all bias types
./scripts/run_inference.sh deepseek mimic all    # DeepSeek with all principals
```

### Test on USMLE Dataset

```bash
./scripts/run_inference.sh llama usmle bayesian  # Single principal
./scripts/run_inference.sh llama usmle all       # All principals
```

### Custom Configuration

```bash
# Use different principal model
PRINCIPAL_MODEL=meta/llama-3.3-70b-instruct ./scripts/run_inference.sh llama mimic bayesian

# Increase parallelization
MAX_WORKERS=16 PRINCIPAL_WORKERS=8 ./scripts/run_inference.sh llama mimic all

# Multiple custom settings
AGENT_SERVER=sglang MAX_WORKERS=16 ./scripts/run_inference.sh deepseek usmle bayesian
```

## Two-Stage Pipeline

Each script runs a two-stage pipeline:

1. **Stage 1: Agent Inference**
   - Analyzes clinical cases
   - Generates recommendations
   - Caches results (reused if exists)

2. **Stage 2: Principal Inference**
   - Evaluates agent recommendations
   - Applies decision-making biases
   - Produces final results

## Adding New Configurations

### Add a New Model

Edit `run_inference.sh` and add to the MODEL_MAP:

```bash
declare -A MODEL_MAP=(
    ["my-model"]="organization/model-name"
    ...
)
```

Then use it:

```bash
./scripts/run_inference.sh my-model mimic bayesian
```

### Add a New Dataset

Edit `run_inference.sh`:

```bash
declare -A DATASET_MAP=(
    ["my-dataset"]="experiments/input/my_dataset.json"
    ...
)
```

## Troubleshooting

### Script Not Found

```bash
# Make sure you're in the project root
cd /path/to/persuasive-misalignment
./scripts/run_inference.sh llama mimic bayesian
```

### Permission Denied

```bash
chmod +x scripts/run_inference.sh
```

### Agent Cache Not Loading

```bash
# Force re-run agent inference
rm experiments/cache/agent_*.json
./scripts/run_inference.sh llama mimic bayesian
```

### Python Module Errors

```bash
# Make sure you're in the right environment
poetry install
poetry shell
```

## Performance Tips

1. **Reuse Agent Cache**: Agent inference is expensive. Run once, then test different principals.
2. **Parallel Processing**: Increase `MAX_WORKERS` for faster agent inference.
3. **Subset Testing**: Use a smaller dataset or model (llama-small) for quick tests.
4. **Background Execution**: Run long jobs with `nohup`:
   ```bash
   nohup ./scripts/run_inference.sh llama mimic all > output.log 2>&1 &
   ```
