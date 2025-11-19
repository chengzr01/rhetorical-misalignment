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

# Default behavior (runs agent, bayesian, and behavioral inference)
./scripts/run_inference.sh                    # Default: llama + mimiciv_demo
./scripts/run_inference.sh llama              # Llama 70B on mimiciv_demo
./scripts/run_inference.sh deepseek usmle     # DeepSeek on USMLE

# Custom principal types
./scripts/run_inference.sh llama mimic bayesian          # Only Bayesian
./scripts/run_inference.sh llama mimic "behavioral"      # Only Behavioral
./scripts/run_inference.sh llama mimic all               # All principals
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
- `bayesian` - Bayesian reasoning principal (default)
- `behavioral` - Behavioral economics principal (default)
- `"bayesian behavioral"` - Both Bayesian and Behavioral (default)
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

# Force re-run even if results exist (default: false)
export FORCE_RERUN=true           # Skip existing result checks and re-run everything

# Run with custom settings
MAX_WORKERS=16 bash scripts/run_inference.sh llama mimic
FORCE_RERUN=true bash scripts/run_inference.sh llama mimic
```

## Output Files

Results are automatically saved to separate files for each principal type:

```
experiments/
├── cache/
│   └── <dataset>/
│       └── agent_<model>.json                    # Agent inference cache
└── output/
    └── <dataset>/
        ├── principal_<model>_bayesian.json       # Bayesian principal results
        ├── principal_<model>_behavioral.json     # Behavioral principal results
        ├── principal_<model>_anchoring.json      # Anchoring bias results
        ├── principal_<model>_availability.json   # Availability bias results
        └── ...                                   # One file per principal type
```

**Note**: Each principal type is saved to its own file, making it easy to analyze results independently or compare across different principal types.

## Examples

### Run with Default Settings (Bayesian + Behavioral)

```bash
./scripts/run_inference.sh llama mimic           # Llama 70B, default principals
./scripts/run_inference.sh llama-small mimic     # Llama 8B (faster), default principals
./scripts/run_inference.sh deepseek mimic        # DeepSeek, default principals
./scripts/run_inference.sh oss mimic             # GPT OSS, default principals
```

### Run with Specific Principal Types

```bash
./scripts/run_inference.sh llama mimic bayesian      # Only Bayesian
./scripts/run_inference.sh llama mimic behavioral    # Only Behavioral
./scripts/run_inference.sh llama mimic all           # All principals (takes longer)
```

### Test on USMLE Dataset

```bash
./scripts/run_inference.sh llama usmle               # Default principals
./scripts/run_inference.sh llama usmle bayesian      # Only Bayesian
./scripts/run_inference.sh llama usmle all           # All principals
```

### Custom Configuration

```bash
# Use different principal model
PRINCIPAL_MODEL=meta/llama-3.3-70b-instruct ./scripts/run_inference.sh llama mimic

# Increase parallelization
MAX_WORKERS=16 PRINCIPAL_WORKERS=8 ./scripts/run_inference.sh llama mimic

# Multiple custom settings
AGENT_SERVER=sglang MAX_WORKERS=16 ./scripts/run_inference.sh deepseek usmle
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

### Smart Result Caching

Both stages automatically check for existing results and skip re-running inference if the output files already exist:

- **Agent Inference**: Checks if agent cache file exists with expected number of results
- **Principal Inference**: Checks each principal type's output file separately
  - Only runs inference for principal types that don't have complete results
  - Example: If `principal_llama_bayesian.json` exists but `principal_llama_behavioral.json` doesn't, only behavioral inference runs

This saves time and API costs when re-running experiments. To force re-run even if results exist:

```bash
FORCE_RERUN=true ./scripts/run_inference.sh llama mimic
```

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
./scripts/run_inference.sh my-model mimic  # Uses default principals
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
./scripts/run_inference.sh llama mimic
```

### Permission Denied

```bash
chmod +x scripts/run_inference.sh
```

### Results Already Exist (Want to Re-run)

By default, the script skips re-running if results already exist. To force re-run:

```bash
# Force re-run both agent and principal inference
FORCE_RERUN=true ./scripts/run_inference.sh llama mimic

# Or manually delete specific principal type results
rm experiments/output/mimiciv_demo/principal_llama_bayesian.json
rm experiments/output/mimiciv_demo/principal_llama_behavioral.json
./scripts/run_inference.sh llama mimic

# Or delete all output files
rm experiments/cache/*/agent_*.json
rm experiments/output/*/principal_*.json
./scripts/run_inference.sh llama mimic
```

### Python Module Errors

```bash
# Make sure you're in the right environment
poetry install
poetry shell
```

## Performance Tips

1. **Smart Caching (NEW)**: Both agent and principal inference now check for existing results and skip re-running by default. This saves time and API costs when running experiments multiple times.
2. **Reuse Agent Cache**: Agent inference is expensive. The cache is automatically reused across different principal type runs.
3. **Parallel Processing**: Increase `MAX_WORKERS` for faster agent inference.
4. **Subset Testing**: Use a smaller dataset or model (llama-small) for quick tests.
5. **Start Simple**: Default runs both Bayesian and Behavioral. For testing, run single principals first:
   ```bash
   ./scripts/run_inference.sh llama mimic bayesian  # Test with one principal first
   ```
6. **Background Execution**: Run long jobs with `nohup`:
   ```bash
   nohup ./scripts/run_inference.sh llama mimic > output.log 2>&1 &
   ```
