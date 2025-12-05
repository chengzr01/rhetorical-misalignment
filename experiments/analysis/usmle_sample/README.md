# USMLE Sample Dataset Analysis

This directory contains comprehensive statistical analysis and visualizations comparing bayesian and behavioral principals' decision-making patterns, along with DeepSeek-v3.1 belief distributions.

## Generated Files

### 1. Visualizations

#### `belief_distribution.png` (4 subplots)
- **Overall Belief Distribution**: Histogram showing the distribution of DeepSeek-v3.1's confidence scores
- **Correct vs Incorrect**: Overlaid histograms comparing belief scores for correct vs incorrect answers
- **Box Plot Comparison**: Side-by-side comparison of belief distributions
- **Cumulative Distribution Function**: Shows percentile rankings of belief scores

#### `principal_comparison.png` (4 subplots)
- **Acceptance Rates**: Bar chart comparing bayesian vs behavioral acceptance rates by model
- **Disagreement Rates**: Bar chart showing the rate of principal disagreement for each model
- **Scatter Plot**: Relationship between bayesian and behavioral acceptance rates
- **Difference Plot**: Horizontal bar chart showing the magnitude and direction of differences

#### `belief_vs_disagreement.png` (2 subplots)
- **Belief Distribution by Agreement**: Histogram comparing beliefs when principals agree vs disagree
- **Box Plot**: Statistical comparison of belief scores for agreement vs disagreement cases

### 2. Data Files

#### `summary.md`
- Quick reference table of key statistics
- DeepSeek belief statistics (mean, median, std, quantiles)
- Principal comparison table across all models

#### `detailed_statistics.json`
- Complete statistical breakdown
- Belief statistics with quantiles
- Model-by-model comparison data
- Case-level details for all models

## Key Findings

### DeepSeek-v3.1 Belief Patterns

1. **High Overall Confidence**
   - Mean belief: 0.9011
   - Median belief: 0.95
   - 75% of cases have belief ≥ 0.95

2. **Belief Correlates with Correctness**
   - Correct answers: mean belief = 0.9317
   - Incorrect answers: mean belief = 0.8217
   - Difference: 0.11 (statistically significant)

3. **Low Confidence Cases**
   - ~10% of cases have belief < 0.85
   - These cases show higher uncertainty and are valuable for further analysis

### Bayesian vs Behavioral Principal Comparison

1. **Consistent Pattern Across Models**
   - Bayesian principals are more accepting than behavioral in ALL models
   - Difference ranges from +5.04% (llama-dpo) to +26.16% (deepseek)
   - Average difference: ~15-20 percentage points

2. **Disagreement Rates**
   - Lowest: llama-sft (27.20%)
   - Highest: deepseek (40.73%)
   - Average: ~35-40% disagreement across models

3. **Model-Specific Observations**

   | Model | Bayesian Accept | Behavioral Accept | Gap | Interpretation |
   |-------|----------------|------------------|-----|----------------|
   | deepseek | 80.46% | 54.30% | +26.16% | Largest divergence |
   | llama-large | 76.82% | 60.26% | +16.56% | High acceptance, moderate gap |
   | llama | 74.83% | 59.93% | +14.90% | Balanced performance |
   | llama-sft | 70.40% | 59.20% | +11.20% | Most agreement |
   | llama-small | 63.83% | 57.45% | +6.38% | Moderate acceptance |
   | llama-dpo | 61.34% | 56.30% | +5.04% | Smallest gap |

### Belief and Disagreement Relationship

- Cases where principals **disagree** tend to have **lower average beliefs**
- This suggests that uncertain/difficult cases are more likely to trigger behavioral biases
- Low confidence cases (belief < 0.8) show higher disagreement rates

## Usage

To regenerate these analyses:

```bash
# Generate statistics and visualizations
python analyze_usmle_sample_stats.py

# With custom parameters
python analyze_usmle_sample_stats.py \
    --input-dir output/usmle_sample \
    --belief-file tests/test_usmle_sample_deepseek-ai-deepseek-v3.1_belief.json \
    --output-dir analysis/usmle_sample
```

## Interpretation

### Why Bayesian is More Accepting

The consistent pattern of higher acceptance rates for bayesian principals suggests:
1. **Rational updating**: Bayesian principals update beliefs based on evidence without behavioral biases
2. **Loss aversion in behavioral**: Behavioral principals may be more conservative due to loss aversion
3. **Anchoring effects**: Behavioral principals may anchor on initial conservative positions

### Clinical Implications

1. **High Disagreement on Uncertain Cases**: When model confidence is low, principal disagreement increases
2. **Systematic Bias**: Behavioral principals consistently more conservative across all models
3. **Model Dependency**: The magnitude of disagreement varies by model, suggesting different sensitivity to behavioral biases

## Related Files

- Main analysis script: `../../analyze_usmle_sample.py`
- Statistics script: `../../analyze_usmle_sample_stats.py`
- Case files: `../../cases/usmle_sample/decision_making_analysis_*_max_diff.json`
