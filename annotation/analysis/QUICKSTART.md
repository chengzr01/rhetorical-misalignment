# Quick Start Guide - USMLE Sample Analysis

Get started analyzing your USMLE Sample annotation results!

## Quick Commands

### Run Everything at Once
```bash
./run_all_analyses.sh
```
This will generate:
- `analysis_report.txt` - Full statistical analysis
- `model_comparison.txt` - Model-by-model comparison
- `usmle_sample_annotations.csv` - CSV for further analysis

### View Summary
```bash
cat SUMMARY.md
```
See all key findings in an easy-to-read format!

---

## Individual Analyses

**1. Comprehensive Statistics**
```bash
python analyze_results.py
```
Shows:
- Answer correctness at each step
- Belief changes (mean, median, std dev)
- Persuasion metrics (correct→incorrect, incorrect→correct)
- Time spent per step
- Demographics breakdown

**2. Compare Models**
```bash
python compare_models.py
```
See which AI models are most/least persuasive:
- Llama-3.1-8B vs Tulu-3-8B-DPO vs Llama-3.1-405B
- Net helpful vs harmful effects
- Accuracy changes per model

**3. Export to CSV**
```bash
python export_to_csv.py
```
Creates `usmle_sample_annotations.csv` for analysis in Excel, R, or Python.

---

## Key Findings (162 Annotations)

### Overall Results
- **18.5%** changed answer after seeing AI
- **3.7%** persuaded correct → incorrect (harmful)
- **9.9%** persuaded incorrect → correct (helpful)
- **Net effect**: AI is 2.7× more helpful than harmful

### Model Comparison
- **Llama-3.1-8B**: +8.0% net helpful (✅ Good choice)
- **Tulu-3-8B-DPO**: -18.2% net harmful (❌ Avoid)
- **26.2 percentage point difference** between best and worst model

### Belief Changes
- **24.7%** increased confidence (mean: +0.25)
- **6.2%** decreased confidence (mean: -0.18)
- **69.1%** unchanged confidence

### Demographics
- 23 unique annotators
- Age range: 23-62 years
- 70.4% female, 19.8% male
- 50.0% white, 21.6% Black/African American
- Diverse expertise: nursing, internal medicine, EMT, social work, etc.
- 0-33 years of practice experience

---

## Understanding the Metrics

### Persuasion Types
- **C→I (Correct to Incorrect)**: AI misled annotator ❌
- **I→C (Incorrect to Correct)**: AI helped annotator ✅
- **Net Effect**: I→C - C→I (positive = helpful overall)

### Belief Changes
- **Positive** (+0.25): Annotator became MORE confident
- **Negative** (-0.18): Annotator became LESS confident
- **Zero** (0.00): No change in confidence

### Step Transitions
- **Step 1**: Initial answer (baseline)
- **Step 2**: After seeing AI response
- **Step 3**: After seeing ground truth

---

## Example Analyses

### 1. See persuasion effects
```bash
python analyze_results.py | grep -A 15 "PERSUASION ANALYSIS"
```

### 2. Compare model performance
```bash
python compare_models.py | grep -A 10 "SUMMARY"
```

### 3. Check demographics
```bash
python analyze_results.py | grep -A 50 "DEMOGRAPHICS"
```

### 4. Export for custom analysis
```bash
python export_to_csv.py
# Open usmle_sample_annotations.csv in Excel, R, or pandas
```

---

## File Structure

```
analysis/
├── analyze_results.py          # Main analysis script
├── compare_models.py            # Model comparison
├── export_to_csv.py             # CSV export
├── run_all_analyses.sh          # Run everything
├── README.md                    # Full documentation
├── SUMMARY.md                   # Key findings summary
├── QUICKSTART.md                # This file
└── usmle_sample_annotations.csv # Generated data file
```

---

## Next Steps

1. **Quick overview**: Read `SUMMARY.md`
2. **Run analyses**: Execute `./run_all_analyses.sh`
3. **Deep dive**: Open `usmle_sample_annotations.csv` in your preferred tool
4. **Custom analysis**: Use the CSV with R, Python pandas, or statistical software

---

## Tips

✅ **Do**:
- Look at net effect (I→C - C→I) to evaluate model usefulness
- Compare belief changes across different demographics
- Check which cases had most persuasion effects
- Use CSV for statistical tests and visualizations

❌ **Don't**:
- Ignore model differences (26.2 pp difference is huge!)
- Only look at harmful persuasion (helpful persuasion matters too)
- Forget about unchanged confidence (69.1% didn't change)

---

## Questions?

See `README.md` for detailed documentation on:
- Data formats and structure
- All available metrics
- Script parameters and options
- Statistical methods used
