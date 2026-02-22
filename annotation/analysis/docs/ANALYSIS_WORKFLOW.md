# Complete Analysis Workflow

## Overview

This document describes the complete workflow for analyzing persuasion patterns in AI-human interactions, from individual case analysis to model comparison.

## Two-Stage Analysis Pipeline

```
Stage 1: Individual Case Analysis (characterization.py)
├── Input: persuasion_examples.json (117 cases)
├── Filter: Cases with user reasoning (114 cases)
├── Process: Analyze each case individually with LLM
└── Output: characterization_results_{timestamp}.json
     └── For each case: WHY did the user change their mind?
         ├── Key persuasive elements
         ├── Reasoning patterns
         ├── Cognitive factors
         └── Quality assessment

Stage 2: Model Comparison Analysis (model_comparison_analysis.py)
├── Input: characterization_results_{timestamp}.json
├── Group: Cases by model (6 models)
├── Process:
│   ├── Analyze each model's patterns (6 LLM calls)
│   └── Compare across models (1 LLM call)
└── Output: model_comparison_results_{timestamp}.json
     └── For each model:
         ├── Distinctive persuasion style
         ├── Harmful patterns
         ├── Helpful patterns
         ├── Risk profile
         └── Comparative insights
```

## Files and Their Purposes

### Data Files
- `persuasion_examples.json` - All cases where users changed answers
- `characterization_results_{timestamp}.json` - Individual case analyses
- `model_comparison_results_{timestamp}.json` - Model-level analyses

### Analysis Scripts
- `characterization.py` - Stage 1: Analyze individual cases
- `model_comparison_analysis.py` - Stage 2: Compare models

### Testing Scripts
- `test_characterization_setup.py` - Verify Stage 1 setup
- `test_model_comparison_setup.py` - Verify Stage 2 setup

### Documentation
- `CHARACTERIZATION_GUIDE.md` - Stage 1 guide
- `MODEL_COMPARISON_GUIDE.md` - Stage 2 guide
- `EXAMPLE_OUTPUT.md` - Sample outputs
- `ANALYSIS_WORKFLOW.md` - This file

## Model Distribution

Based on current data (114 cases with reasoning):

| Rank | Model | Total | Harmful | Helpful | Net | H-Rate |
|------|-------|-------|---------|---------|-----|--------|
| 1 | llama-3.1-405b-instruct | 11 | 5 | 6 | +1 | **45.5%** ⚠️ |
| 2 | Llama-3.1-Tulu-3-8B-SFT | 33 | 10 | 23 | +13 | **30.3%** |
| 3 | llama-3.1-8b-instruct | 19 | 4 | 15 | +11 | **21.1%** |
| 4 | Llama-3.1-Tulu-3-8B-DPO | 19 | 3 | 16 | +13 | **15.8%** ✓ |
| 5 | deepseek-chat-v3.1 | 19 | 3 | 16 | +13 | **15.8%** ✓ |
| 6 | llama-3.3-70b-instruct | 13 | 2 | 11 | +9 | **15.4%** ✓ |

**Key Observations:**
- 📊 **Large model risk**: The largest model (405B) has the highest harmful rate
- 📊 **Training method matters**: DPO shows lower harmful rate than SFT (15.8% vs 30.3%)
- 📊 **Newer is safer**: llama-3.3 (newer) safer than llama-3.1 (45.5% → 15.4%)
- 📊 **DeepSeek competitive**: Comparable safety to best Llama models

## Research Questions This Enables

### Model Safety
1. Which models are safest for medical AI assistance?
2. How does model size affect persuasion risk?
3. Does training method (SFT vs DPO) impact safety?

### Persuasion Mechanisms
4. What persuasion tactics do different models use?
5. Are harmful vs helpful persuasion mechanisms different?
6. What cognitive biases are most commonly triggered?

### Model Development
7. What makes some models safer than others?
8. Can we design interventions to reduce harmful persuasion?
9. How do model families (Llama, DeepSeek, Tulu) differ?

### User Behavior
10. What makes users susceptible to harmful persuasion?
11. Are there patterns in user reasoning quality?
12. Can we predict which cases are high-risk?

## Quick Start

### First Time Setup
```bash
# Install dependencies
pip install openai

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Test setup
python test_characterization_setup.py
```

### Run Stage 1 (if needed)
```bash
# Run on 5 cases for testing
python characterization.py --max-cases 5

# Run on all 114 cases (~$0.15-0.35, ~30-60 minutes)
python characterization.py
```

### Run Stage 2
```bash
# Test setup
python test_model_comparison_setup.py

# Run full analysis (~$0.10-0.15, ~5-10 minutes)
python model_comparison_analysis.py
```

## Output Structure

```
annotation/analysis/
├── persuasion_examples.json                  # Input data
├── characterization_results/
│   ├── characterization_results_*.json       # Stage 1 JSON
│   ├── characterization_report_*.txt         # Stage 1 report
│   └── partial_results_*.json                # Checkpoints
└── model_comparison_results/
    ├── model_comparison_analysis_*.json      # Stage 2 JSON
    └── model_comparison_report_*.txt         # Stage 2 report
```

## Cost Summary

| Stage | API Calls | Estimated Cost | Time |
|-------|-----------|----------------|------|
| Stage 1 (characterization) | 114 | $0.15-0.35 | 30-60 min |
| Stage 2 (model comparison) | 7 | $0.10-0.15 | 5-10 min |
| **Total** | **121** | **$0.25-0.50** | **35-70 min** |

Using `deepseek/deepseek-v3.2` (very cost-effective model).

## Integration with Paper

### Quantitative Results
- Use model statistics table (from Stage 2)
- Plot harmful rates by model
- Show correlation between model size and risk
- Compare training methods (SFT vs DPO)

### Qualitative Results
- Quote user reasoning examples (from Stage 1)
- Describe model-specific persuasion patterns (from Stage 2)
- Identify cognitive biases (from both stages)
- Show comparative insights (from Stage 2)

### Mixed Methods
- Combine statistics with qualitative quotes
- Use LLM analyses to interpret quantitative patterns
- Support claims with specific examples

## Advanced Usage

### Custom Analysis Models
```bash
# Use Claude instead of DeepSeek
python model_comparison_analysis.py --model "anthropic/claude-3.5-sonnet"

# Use GPT-4
python model_comparison_analysis.py --model "openai/gpt-4-turbo"
```

### Rate Limiting
```bash
# Slower, more reliable
python model_comparison_analysis.py --rate-limit 2.0
```

### Resuming After Errors
```bash
# Stage 1: Resume from case 50
python characterization.py --start-from 50

# Stage 2: No resume needed (only 7 calls)
```

## Troubleshooting

### Common Issues

**1. "File not found: characterization_results_*.json"**
- Run Stage 1 first: `python characterization.py`

**2. API rate limit errors**
- Increase rate limit: `--rate-limit 2.0`
- Check OpenRouter dashboard for limits

**3. "openai package not found"**
- Install: `pip install openai`

**4. Results look incomplete**
- Check for partial_results_*.json files
- Resume from last checkpoint with `--start-from`

**5. Analysis seems wrong**
- Verify input data: `python test_characterization_setup.py`
- Check API key is correct
- Try different analysis model

## Next Steps

After completing both stages:

1. **Read the reports** - Start with model_comparison_report_*.txt
2. **Identify key findings** - Note surprising patterns
3. **Deep dive** - Examine specific interesting cases
4. **Visualize** - Create plots from JSON data
5. **Write paper sections** - Use analyses for qualitative sections
6. **Design interventions** - Based on identified patterns
7. **Further research** - Generate new hypotheses

## Support

- **Documentation**: See individual GUIDE.md files
- **Examples**: See EXAMPLE_OUTPUT.md
- **Testing**: Use test_*_setup.py scripts
- **Issues**: Check GitHub issues or contact authors

---

**Happy Analyzing! 🔬**
