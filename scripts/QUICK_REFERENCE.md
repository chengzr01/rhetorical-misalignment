# Quick Reference Guide

## Most Common Commands

```bash
# Syntax: ./scripts/run_inference.sh <model> <dataset> <principal_types>

# Standard workflow: Llama 70B on MIMIC with Bayesian principal
./scripts/run_inference.sh llama mimic bayesian

# Full analysis: Llama 70B on MIMIC with all principals
./scripts/run_inference.sh llama mimic all

# USMLE testing: Llama 70B on USMLE questions
./scripts/run_inference.sh llama usmle bayesian

# Quick test: Small model for fast iteration
./scripts/run_inference.sh llama-small mimic bayesian
```

## Common Workflows

```bash
# MIMIC dataset - different models
./scripts/run_inference.sh llama mimic bayesian        # Llama 70B + Bayesian
./scripts/run_inference.sh deepseek mimic bayesian     # DeepSeek + Bayesian
./scripts/run_inference.sh llama-small mimic bayesian  # Llama 8B + Bayesian (fast)
./scripts/run_inference.sh oss mimic bayesian          # GPT OSS + Bayesian

# USMLE dataset
./scripts/run_inference.sh llama usmle bayesian        # Llama 70B + Bayesian
./scripts/run_inference.sh llama usmle all             # Llama 70B + All principals

# All principals (slower, comprehensive)
./scripts/run_inference.sh llama mimic all             # Llama 70B + All
./scripts/run_inference.sh deepseek mimic all          # DeepSeek + All
```

## Custom Settings

```bash
# With environment variables
MAX_WORKERS=16 ./scripts/run_inference.sh llama mimic bayesian
PRINCIPAL_MODEL=meta/llama-3.3-70b-instruct ./scripts/run_inference.sh llama mimic bayesian

# Multiple custom settings
AGENT_SERVER=sglang MAX_WORKERS=16 ./scripts/run_inference.sh deepseek usmle all
```

## File Locations

| Type | Pattern | Example |
|------|---------|---------|
| Agent Cache | `experiments/cache/agent_<model>_<dataset>.json` | `agent_llama_mimic.json` |
| Results (Bayesian) | `experiments/output/principal_<model>_<dataset>.json` | `principal_llama_mimic.json` |
| Results (All) | `experiments/output/principal_<model>_<dataset>_all.json` | `principal_llama_mimic_all.json` |

## Model Keys

- `deepseek` → `deepseek-ai/deepseek-v3.1`
- `llama` → `meta/llama-3.3-70b-instruct`
- `llama-small` → `meta/llama-3.1-8b-instruct`
- `oss` → `openai/gpt-oss-120b`

## Principal Types

- `bayesian` - Bayesian reasoning (single, fast)
- `all` - All bias types: anchoring, availability, confirmation, conservatism, overconfidence, prospect

## Environment Variables

```bash
export AGENT_SERVER=nvidia              # or: openrouter, sglang
export PRINCIPAL_SERVER=nvidia          # or: openrouter, sglang
export PRINCIPAL_MODEL=deepseek-ai/deepseek-v3.1
export MAX_WORKERS=8                    # Agent parallelization
export PRINCIPAL_WORKERS=4              # Principal parallelization
```

## Quick Debugging

```bash
# Check if cache exists
ls -lh experiments/cache/

# View recent results
ls -lt experiments/output/ | head

# Clear cache and re-run
rm experiments/cache/agent_llama_mimic.json
bash scripts/llama.sh

# Run only agent inference (for caching)
python agent_inference.py --input experiments/input/clinical_questions.json \
  --output experiments/cache/agent_llama_mimic.json \
  --agent-model meta/llama-3.3-70b-instruct

# Run only principal inference (using cache)
python principal_inference.py --agent-cache experiments/cache/agent_llama_mimic.json \
  --output experiments/output/principal_llama_mimic.json \
  --principal-types bayesian
```

## Common Workflows

### Experiment Workflow
```bash
# 1. Run agent inference once (slow, expensive)
./scripts/run_inference.sh llama mimic bayesian

# 2. Test different principal configurations (fast, uses cache)
python principal_inference.py \
  --agent-cache experiments/cache/agent_llama_mimic.json \
  --output experiments/output/test_anchoring.json \
  --principal-types anchoring

python principal_inference.py \
  --agent-cache experiments/cache/agent_llama_mimic.json \
  --output experiments/output/test_confirmation.json \
  --principal-types confirmation
```

### Comparison Workflow
```bash
# Run different models on same dataset
./scripts/run_inference.sh llama mimic bayesian
./scripts/run_inference.sh deepseek mimic bayesian
./scripts/run_inference.sh llama-small mimic bayesian

# Compare results
python analyze.py \
  experiments/output/principal_llama_mimic.json \
  experiments/output/principal_deepseek_mimic.json \
  experiments/output/principal_llama_small_mimic.json
```

### Dataset Comparison
```bash
# Same model on different datasets
./scripts/run_inference.sh llama mimic bayesian   # MIMIC
./scripts/run_inference.sh llama usmle bayesian   # USMLE

# Compare performance
python compare.py \
  experiments/output/principal_llama_mimic.json \
  experiments/output/principal_llama_usmle.json
```
