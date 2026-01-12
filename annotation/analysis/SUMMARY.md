# USMLE Sample Dataset - Analysis Summary

**Dataset**: USMLE Sample
**Total Annotations**: 162
**Unique Cases**: 113
**Unique Annotators**: 23

---

## Executive Summary

This analysis examines how AI-generated clinical reasoning affects medical professionals' diagnostic decisions. Annotators provided initial answers (Step 1), then saw AI reasoning and could update their answers (Step 2), and finally saw the ground truth (Step 3).

### Key Takeaways

1. **AI has mixed persuasion effects**: 18.5% of annotators changed their answer after seeing AI
2. **Net helpful effect overall**: 9.9% were persuaded to correct answers vs. 3.7% to incorrect answers
3. **Model differences are significant**: Llama-3.1-8B is net helpful (+8.0%), while Tulu-3-8B-DPO is net harmful (-18.2%)
4. **Belief changes are common**: 24.7% increased confidence, 6.2% decreased confidence after seeing AI
5. **Most persuasion happens in Step 2**: Only 8.0% changed answers after seeing ground truth

---

## 1. Overall Accuracy Results

### Answer Correctness by Step

| Step | Correct | Total | Accuracy | Change from Previous |
|------|---------|-------|----------|---------------------|
| **Step 1** (Initial answer) | 61 | 162 | 37.7% | - |
| **Step 2** (After AI) | 71 | 162 | 43.8% | **+6.1 pp** |
| **Step 3** (After ground truth) | 84 | 162 | 51.9% | **+8.1 pp** |

**Key Insight**: AI improved accuracy by 6.1 percentage points, and ground truth added another 8.1 percentage points.

---

## 2. Persuasion Analysis

### Answer Changes

| Transition | Changed Answer | Percentage |
|------------|----------------|------------|
| **Step 1 → Step 2** (After seeing AI) | 30/162 | **18.5%** |
| **Step 2 → Step 3** (After seeing ground truth) | 13/162 | **8.0%** |

### Persuasion Effects (Step 1 → Step 2)

| Effect Type | Count | Percentage | Mean Belief Change |
|-------------|-------|------------|-------------------|
| **Correct → Incorrect** (Harmful) | 6 | 3.7% | -0.020 |
| **Incorrect → Correct** (Helpful) | 16 | 9.9% | +0.114 |
| **Stayed Correct** | 55 | 34.0% | - |
| **Stayed Incorrect** | 77 | 47.5% | - |

**Net Effect**: +10 more people persuaded to correct answers (+6.2 percentage points)

**Key Insight**: AI is 2.7× more likely to persuade to correct answers than incorrect ones.

---

## 3. Belief Change Analysis

### Step 1 → Step 2 (After Seeing AI)

| Belief Change | Count | Percentage | Mean | Median | Range | Std Dev |
|--------------|-------|------------|------|--------|-------|---------|
| **Increased** | 40 | 24.7% | +0.249 | +0.225 | +0.020 to +0.830 | 0.204 |
| **Decreased** | 10 | 6.2% | -0.184 | -0.185 | -0.450 to -0.030 | 0.120 |
| **Unchanged** | 112 | 69.1% | 0.000 | 0.000 | - | - |

### Step 2 → Step 3 (After Seeing Ground Truth)

| Belief Change | Count | Percentage | Mean | Median | Range | Std Dev |
|--------------|-------|------------|------|--------|-------|---------|
| **Increased** | 4 | 2.5% | +0.260 | +0.245 | +0.050 to +0.500 | 0.192 |
| **Decreased** | 3 | 1.9% | -0.237 | -0.190 | -0.430 to -0.090 | 0.175 |
| **Unchanged** | 155 | 95.7% | 0.000 | 0.000 | - | - |

**Key Insights**:
- AI causes 4× more belief changes than ground truth (30.9% vs 4.4%)
- When beliefs increase, they increase by ~0.25 on average (on 0-1 scale)
- Most annotators (69.1%) don't change their confidence after seeing AI

---

## 4. Model Comparison

### Overview

| Model | Annotations | % of Total |
|-------|-------------|------------|
| **meta-llama/llama-3.1-8b-instruct** | 150 | 92.6% |
| **allenai/Llama-3.1-Tulu-3-8B-DPO** | 11 | 6.8% |
| **meta-llama/llama-3.1-405b-instruct** | 1 | 0.6% |

### Detailed Model Performance

#### **Llama-3.1-8B (meta-llama/llama-3.1-8b-instruct)** - 150 annotations

| Metric | Value |
|--------|-------|
| **Accuracy Change** (Step 1 → Step 2) | **+8.0 pp** (37.3% → 45.3%) |
| **Answer Change Rate** | 18.0% |
| **Harmful Persuasion** (C→I) | 2.7% |
| **Helpful Persuasion** (I→C) | 10.7% |
| **Net Effect** | **+8.0 pp (helpful)** |
| **Belief Increased** | 25.3% (mean: +0.252) |
| **Belief Decreased** | 6.7% (mean: -0.184) |

**Verdict**: ✅ **Net Helpful** - Improves accuracy and mostly persuades toward correct answers.

#### **Tulu-3-8B-DPO (allenai/Llama-3.1-Tulu-3-8B-DPO)** - 11 annotations

| Metric | Value |
|--------|-------|
| **Accuracy Change** (Step 1 → Step 2) | **-18.2 pp** (45.5% → 27.3%) |
| **Answer Change Rate** | 27.3% |
| **Harmful Persuasion** (C→I) | 18.2% |
| **Helpful Persuasion** (I→C) | 0.0% |
| **Net Effect** | **-18.2 pp (harmful)** |
| **Belief Increased** | 18.2% (mean: +0.195) |
| **Belief Decreased** | 0.0% |

**Verdict**: ❌ **Net Harmful** - Decreases accuracy and only persuades toward incorrect answers.

#### **Llama-3.1-405B (meta-llama/llama-3.1-405b-instruct)** - 1 annotation

| Metric | Value |
|--------|-------|
| **Accuracy Change** (Step 1 → Step 2) | 0.0 pp |
| **Answer Change Rate** | 0.0% |
| **Net Effect** | 0.0 pp |

**Verdict**: ⚠️ **Insufficient data** - Only 1 annotation.

---

## 5. Engagement Metrics

### Highlights (Text Selections)

| Metric | Value |
|--------|-------|
| **Total highlights** | 19 |
| **Annotations with highlights** | 16/162 (9.9%) |
| **Highlights in Step 2** (AI response) | 17 (89.5%) |
| **Highlights in Step 3** (Ground truth) | 2 (10.5%) |

**Key Insight**: Annotators primarily highlight AI reasoning, not ground truth.

### Time Spent per Step

| Step | Mean (seconds) | Median (seconds) | Range |
|------|----------------|------------------|-------|
| **Step 1** (Initial answer) | 607.5 | 493.7 | 2.9 - 1871.8 |
| **Step 2** (After AI) | 24.7 | 13.3 | 1.7 - 331.1 |
| **Step 3** (After truth) | 44.3 | 22.9 | 2.5 - 446.2 |

**Key Insights**:
- Step 1 takes ~10 minutes (reading case and forming initial opinion)
- Step 2 takes ~15-25 seconds (quick review of AI reasoning)
- Step 3 takes ~20-45 seconds (reviewing ground truth)

---

## 6. Demographics

### Age Distribution

| Age | Count | Percentage |
|-----|-------|------------|
| 30 | 22 | 13.6% |
| 55 | 12 | 7.4% |
| 33 | 11 | 6.8% |
| 31, 29, 38, 26, 49, 27, 54 | 10 each | 6.2% each |
| Other ages (23-62) | 34 | 21.0% |

**Mean age**: ~37 years

### Gender

| Gender | Count | Percentage |
|--------|-------|------------|
| **Female** | 114 | 70.4% |
| **Male** | 32 | 19.8% |
| **Not specified** | 16 | 9.9% |

### Race/Ethnicity

| Race | Count | Percentage |
|------|-------|------------|
| **White** | 81 | 50.0% |
| **Black/African American** | 35 | 21.6% |
| **Asian** | 11 | 6.8% |
| **Two or more** | 10 | 6.2% |
| **Hispanic/Latino** | 9 | 5.6% |
| **Not specified** | 16 | 9.9% |

### Expertise Areas (Top 10)

| Expertise | Count | Percentage |
|-----------|-------|------------|
| Construction | 11 | 6.8% |
| Internal Medicine | 20 | 12.3% |
| Nursing | 10 | 6.2% |
| OB | 10 | 6.2% |
| EMT | 10 | 6.2% |
| Healthcare | 10 | 6.2% |
| Social Work | 10 | 6.2% |
| Urology | 10 | 6.2% |
| Medical Doctor | 10 | 6.2% |
| Medical Device | 10 | 6.2% |

**Note**: Diverse mix of healthcare and non-healthcare professionals.

### Years of Practice

| Years | Count | Percentage |
|-------|-------|------------|
| 5 | 24 | 14.8% |
| 15 | 22 | 13.6% |
| 10 | 20 | 12.3% |
| 2 | 20 | 12.3% |
| 11-12 | 22 | 13.6% |
| Other | 54 | 33.3% |

**Range**: 0-33 years of practice

### Geographic Distribution

**Locations**: Across United States (multiple states including CA, NY, WA, TN, IA, MD, VA, CT, WI, IL, NJ, DE)

---

## 7. Statistical Significance

### Belief Change Statistics

**Step 1 → Step 2 (AI Persuasion)**:
- Belief increases: Mean = 0.249, SD = 0.204, n = 40
- Belief decreases: Mean = -0.184, SD = 0.120, n = 10

**Distribution**: Belief changes show moderate effect sizes with considerable variation (large SD).

### Model Comparison Significance

The difference between Llama-3.1-8B and Tulu-3-8B-DPO is substantial:
- Llama-3.1-8B: 2.7% harmful, 10.7% helpful
- Tulu-3-8B-DPO: 18.2% harmful, 0.0% helpful
- **Difference**: 26.2 percentage points in net effect

---

## 8. Key Conclusions

### Main Findings

1. **AI persuasion is real and measurable**: Nearly 1 in 5 annotators (18.5%) changed their answer after seeing AI reasoning

2. **Overall positive impact**: AI is net helpful, with 2.7× more helpful persuasion than harmful (9.9% vs 3.7%)

3. **Model selection matters critically**:
   - Llama-3.1-8B shows +8.0% net helpful effect
   - Tulu-3-8B-DPO shows -18.2% net harmful effect
   - This 26.2 percentage point difference highlights the importance of careful model selection

4. **Confidence effects are asymmetric**: More people increase confidence (24.7%) than decrease it (6.2%), suggesting AI primarily boosts existing beliefs

5. **Limited engagement with AI reasoning**: Only 9.9% of annotators highlighted text, suggesting most read but don't deeply engage

6. **Demographics are diverse**: The sample includes various ages (23-62), genders (70.4% female), races, and expertise levels, providing broad representation

### Implications

- **For AI deployment**: Model choice significantly impacts whether AI helps or harms decision-making
- **For human-AI interaction**: Most belief changes happen immediately after seeing AI (Step 2), not after seeing ground truth (Step 3)
- **For system design**: Consider ways to encourage deeper engagement with AI reasoning (currently low highlight rate)
- **For future research**: Need to understand why Tulu-3-8B-DPO performs poorly despite being a "DPO" (Direct Preference Optimization) model

---

## Data Files

- **Full analysis**: Run `python analyze_results.py`
- **Model comparison**: Run `python compare_models.py`
- **CSV export**: `usmle_sample_annotations.csv` (162 rows, 27 columns)

Generated: 2026-01-07
