# Annotation Results Analysis - USMLE Sample

This directory contains scripts for analyzing the annotation results from the USMLE Sample dataset persuasive misalignment study.

## Available Scripts

### 1. `analyze_results.py`
Comprehensive analysis of all annotation results, including:
- Answer correctness at each step (Step 1 → Step 2 → Step 3)
- Belief change analysis with detailed statistics
- Persuasion effectiveness metrics (correct→incorrect, incorrect→correct)
- Highlights analysis
- Time analysis (how long annotators spent on each step)
- Annotator demographics breakdown

**Usage:**
```bash
python analyze_results.py
```

**Key Metrics:**
- **Step 1 → Step 2 Analysis**: Shows AI persuasion effects
- **Step 2 → Step 3 Analysis**: Shows ground truth correction effects
- **Persuasion Rates**: Percentage of annotators persuaded to correct vs. incorrect answers
- **Belief Changes**: Mean, median, min, max, and standard deviation of belief changes

### 2. `compare_models.py`
Compares persuasion effectiveness across different AI models (Llama-3.1-8B, Llama-3.1-405B, Tulu-3-8B-DPO).

**Usage:**
```bash
python compare_models.py
```

**Output:**
- Detailed analysis for each model
- Accuracy changes after seeing AI
- Persuasion effects (C→I vs I→C)
- Net helpful/harmful effect
- Belief change statistics by model
- Summary comparison table

**Key Insights:**
- Which models are most persuasive
- Which models cause harmful persuasion (correct→incorrect)
- Which models provide helpful corrections (incorrect→correct)

### 3. `export_to_csv.py`
Exports all annotation data to CSV file for further analysis in spreadsheet tools or statistical software.

**Usage:**
```bash
python export_to_csv.py
```

**Output File:**
- `usmle_sample_annotations.csv` - All annotations with calculated metrics

**CSV Columns:**
- Annotator ID, case ID, model information
- All step data (answers, beliefs, correctness)
- Change indicators between steps
- Calculated belief changes
- Persuasion indicators (correct→incorrect, incorrect→correct)
- Highlight counts
- Time metrics

### 4. `run_all_analyses.sh`
Convenience script to run all analyses and save results to files.

**Usage:**
```bash
./run_all_analyses.sh
```

**Output Files:**
- `analysis_report.txt` - Full comprehensive analysis
- `model_comparison.txt` - Model comparison results
- `usmle_sample_annotations.csv` - CSV export

## Data Structure

### Input Data Location
- Results directory: `../results/usmle_sample/*.json`

### Annotation JSON Format

```json
{
  "annotator_id": "...",
  "case_id": "usmle_sample_XXX",
  "agent_model": "meta-llama/llama-3.1-8b-instruct",
  "correct_answer": "Drug Name",
  "correct_answer_idx": "A|B|C|D",
  "step1": {
    "answer": "A|B|C|D",
    "answer_belief": 0.0-1.0,
    "is_correct": true|false
  },
  "step2": {
    "answer": "A|B|C|D",
    "answer_belief": 0.0-1.0,
    "is_correct": true|false
  },
  "step3": {
    "answer": "A|B|C|D",
    "answer_belief": 0.0-1.0,
    "is_correct": true|false
  },
  "step1_to_step2_changes": {
    "answer_changed": true|false,
    "answer_belief_changed": true|false
  },
  "step2_to_step3_changes": {
    "answer_changed": true|false,
    "answer_belief_changed": true|false
  },
  "highlights": [
    {
      "context": "...",
      "text": "highlighted text",
      "timestamp": "ISO timestamp",
      "step": "step1|step2|step3"
    }
  ],
  "demographics": {
    "age": "...",
    "expertise": "...",
    "practice_location": "...",
    "race": "...",
    "sex": "...",
    "years_of_practice": "..."
  },
  "session_start": "ISO timestamp",
  "step1_time": "ISO timestamp",
  "step2_time": "ISO timestamp",
  "step3_time": "ISO timestamp"
}
```

## Key Metrics Explained

### Belief Changes
- **Positive change** (e.g., +0.25): Annotator became MORE confident in their answer
- **Negative change** (e.g., -0.18): Annotator became LESS confident
- **Zero**: No change in confidence
- **Calculated as**: `belief_after - belief_before`

### Persuasion Effectiveness
- **Correct → Incorrect (C→I)**: Annotator changed from correct to incorrect answer after seeing AI (harmful)
- **Incorrect → Correct (I→C)**: Annotator changed from incorrect to correct answer after seeing AI (helpful)
- **Net Effect**: I→C - C→I (positive means net helpful, negative means net harmful)

### Step Transitions
- **Step 1 → Step 2**: After seeing AI response (measures AI persuasion)
- **Step 2 → Step 3**: After seeing ground truth (measures correction from ground truth)

### Accuracy Metrics
- **Step 1 Accuracy**: Initial answer correctness (baseline)
- **Step 2 Accuracy**: Correctness after seeing AI
- **Step 3 Accuracy**: Correctness after seeing ground truth
- **Change Step1→Step2**: Percentage point change in accuracy (positive = improvement)

## Current Dataset Statistics

Based on 162 annotations across 113 unique cases from 23 annotators:

### Overall Results
- **Total annotations**: 162
- **Unique cases**: 113
- **Unique annotators**: 23

### Model Distribution
- **meta-llama/llama-3.1-8b-instruct**: 150 (92.6%)
- **allenai/Llama-3.1-Tulu-3-8B-DPO**: 11 (6.8%)
- **meta-llama/llama-3.1-405b-instruct**: 1 (0.6%)

### Key Findings
- **18.5%** of annotators changed their answer after seeing AI
- **3.7%** were persuaded from correct → incorrect (harmful)
- **9.9%** were persuaded from incorrect → correct (helpful)
- **24.7%** had increased belief confidence after seeing AI
- **6.2%** had decreased belief confidence after seeing AI

### Model Comparison
- **Llama-3.1-8B**: +8.0% net helpful effect (10.7% I→C - 2.7% C→I)
- **Tulu-3-8B-DPO**: -18.2% net harmful effect (0.0% I→C - 18.2% C→I)

## Example Analyses

### Quick Summary
```bash
python analyze_results.py | grep -A 10 "PERSUASION ANALYSIS"
```

### Compare Model Effects
```bash
python compare_models.py | grep -A 5 "SUMMARY"
```

### Export for Statistical Analysis
```bash
python export_to_csv.py
# Then open usmle_sample_annotations.csv in R, Python pandas, or Excel
```

## Requirements

All scripts use only Python standard library:
- `json` - for reading annotation files
- `os` - for file operations
- `csv` - for CSV export
- `statistics` - for mean/median/stdev calculations
- `collections` - for defaultdict
- `datetime` - for time calculations

No external dependencies required!

## Demographics

The dataset includes diverse annotator demographics:
- **Age range**: 23-62 years (most common: 30, 55, 33)
- **Gender**: 70.4% female, 19.8% male
- **Race**: 50.0% white, 21.6% Black/African American, 6.8% Asian
- **Expertise**: Various healthcare roles (nursing, internal medicine, EMT, social work, etc.)
- **Experience**: 0-33 years of practice
- **Locations**: Across United States

## Tips

- CSV file is best for custom analyses in R, Python pandas, or Excel
- Use grep/search to find specific metrics in text reports
- Compare results across different models to understand model differences
- Look at net effect (I→C - C→I) to determine if a model is helpful or harmful overall
