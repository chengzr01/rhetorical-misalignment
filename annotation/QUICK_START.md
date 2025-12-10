# Quick Start Guide - Improved Annotation Interface

## What's New? 🎉

Your annotation interface now has **intelligent case assignment** that:
- Automatically ensures every case gets annotated by at least 3 people
- Prioritizes cases that haven't been annotated yet
- **Tracks each model separately** - llama_small, llama, llama_large, and deepseek each have independent annotation tracking

## How to Use

### 1. Start the Application

```bash
conda activate alignment
python app.py
```

Then open your browser to: `http://localhost:8000`

### 2. Select Your Model

On the demographics page:
- Choose your dataset (e.g., USMLE Sample)
- **Select the model** you want to annotate (e.g., Llama-3.1-8B-Instruct)
- Each model has its own pool of manipulative cases tracked independently

### 3. Start Annotating

Click "Start Annotation" and the system will:
- ✅ Assign you **10 random cases**
- ✅ Prioritize cases with **0 annotations** first
- ✅ Ensure fair distribution across all cases
- ✅ Track progress automatically

### 4. Monitor Progress

Run the analysis script anytime:
```bash
python analyze_coverage.py
```

Example output:
```
================================================================================
Dataset: USMLE Sample, Model: llama_small
================================================================================
  Total manipulative cases: 108

  Annotation Coverage:
    ✗ Not annotated yet:          108 cases (100.0%)
    ◐ Annotated once:               0 cases (0.0%)
    ◑ Annotated twice:              0 cases (0.0%)
    ✓ Goal reached (3+ annot.):     0 cases (0.0%)

  Progress:
    Current annotations: 0/324
    Overall progress: 0.0%

  🔴 Priority: 108 cases need first annotation
```

## The Smart Assignment Algorithm

### How It Works

The system uses a **priority-based random selection**:

```
Priority 1: Cases with 0 annotations ← START HERE
    ↓ (when all cases have 1+ annotations)
Priority 2: Cases with 1 annotation
    ↓ (when all cases have 2+ annotations)
Priority 3: Cases with 2 annotations
    ↓ (when all cases have 3+ annotations)
Priority 4: Cases with 3+ annotations ← GOAL REACHED
```

### Why This Approach?

1. **Full Coverage First**: Every case gets at least 1 annotation before any case gets a 4th
2. **Random Within Priority**: Prevents ordering bias, ensures fair distribution
3. **Efficient Progress**: Optimally reaches the 3-annotation goal
4. **Automatic Balancing**: No manual tracking or assignment needed

## Example Workflow

### Participant 1 arrives:
- System: "Here are 10 cases with 0 annotations"
- Participant annotates all 10
- **Status**: 10 cases now have 1 annotation each

### Participant 2 arrives:
- System: "Here are 10 different cases with 0 annotations"
- Participant annotates all 10
- **Status**: 20 cases now have 1 annotation each

### After 10 participants:
- **100 cases** have 1 annotation each
- **8 cases** still have 0 annotations (from 108 total)

### Participant 11 arrives:
- System: "Here are 8 cases with 0 annotations + 2 cases with 1 annotation"
- **Status**: Now ALL 108 cases have at least 1 annotation! ✅

### The process continues:
- Next phase: All cases get 2nd annotation
- Final phase: All cases get 3rd annotation
- **Goal achieved**: 108 cases × 3 annotations = 324 total annotations

## Current Status

**Dataset**: USMLE Sample (4 models available)

Each model is tracked **independently**:

| Model | Total Cases | Target Annotations | Progress |
|-------|-------------|-------------------|----------|
| Llama-3.1-8B | 108 | 324 | 0.0% |
| Llama-3.3-70B | 122 | 366 | 0.0% |
| Llama-3.1-405B | 127 | 381 | 0.0% |
| DeepSeek-V3 | 190 | 570 | 0.0% |

**Total across all models**: 547 cases, 1,641 target annotations

## Testing

All systems verified and working:
- ✅ Flask application loads correctly
- ✅ Coverage API returns accurate statistics
- ✅ Smart assignment prioritizes correctly
- ✅ Demographics page displays real-time data
- ✅ Random selection works within priority groups

## Need Help?

- **View detailed docs**: See `ANNOTATION_IMPROVEMENTS.md`
- **Check coverage**: Run `python analyze_coverage.py`
- **Test the app**: `conda activate alignment && python app.py`

## Key Files

- **app.py**: Main Flask application with smart assignment logic
- **templates/demographics.html**: User interface with coverage statistics
- **analyze_coverage.py**: Command-line analysis tool
- **annotations/{dataset}/**: Annotation results stored here

---

**Ready to start?** Just run `python app.py` and navigate to the demographics page! 🚀
