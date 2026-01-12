# Clinical Decision Annotation Interface

Flask web application for collecting human annotations on clinical decision-making with AI-generated recommendations. Studies how clinicians' decisions change when exposed to AI analysis and ground truth.

## Overview

Three-step annotation process:
- **MIMIC-IV**: Initial treatment plan → Revise after AI recommendation → Revise after ground truth
- **USMLE**: Initial answer → Revise after AI analysis → Revise after correct answer

## Features

- **Multi-dataset/model support**: MIMIC-IV, USMLE variants; Llama, DeepSeek models
- **Automatic balancing**: Smart scheduling across models and cases for balanced coverage
- **IRB-ready**: Consent forms, demographics collection
- **Rich formatting**: Markdown rendering, text highlighting
- **Progress tracking**: Real-time coverage statistics

## Quick Start

```bash
pip install -r requirements.txt
./start.sh  # or: python app.py
```

Application runs on `http://localhost:8000`

## Architecture (Modular)

```
annotation/
├── config.py                 # Dataset/model configurations
├── data_loader.py           # Data loading utilities
├── scheduler.py             # ⭐ SCHEDULING SYSTEM (model & case balancing)
├── annotation_utils.py      # Annotation save/render utilities
├── app.py                   # Flask routes and UI
├── analyze_coverage.py      # Coverage analysis script
├── test_model_balancing.py  # Test script
└── templates/               # HTML templates
```

### Module Descriptions

**`config.py`**: Add datasets/models, change paths
- `DATASETS`: Dataset configurations (name, paths)
- `AVAILABLE_MODELS`: Model list (key, file, display name)

**`data_loader.py`**: Modify data loading
- `load_data()`, `load_manipulative_case_ids()`, `get_available_models()`

**`scheduler.py`** ⭐ **MAIN SCHEDULING SYSTEM**
- **Model balancing**: `select_model_with_fewest_annotations()` - Auto-selects model needing more annotations
- **Case balancing**: `get_smart_random_cases()` - Prioritizes cases: 0 → 1 → 2 → 3+ annotations
- **Statistics**: `get_annotation_counts_per_case()`, `get_coverage_statistics()`

**`annotation_utils.py`**: Modify annotation format
- `save_annotation()`, `render_markdown()`

**`app.py`**: Add routes, modify UI flow

## Scheduling System

### How It Works
1. User lands on demographics page with "Automatic Model Balancing" ON (default)
2. System selects model with fewest total annotations
3. Within model, selects 10 cases prioritizing those with 0 → 1 → 2 annotations

### Customization

**Change cases per session** (`scheduler.py`):
```python
def get_smart_random_cases(case_ids, annotation_counts, num_cases=10):  # Change here
```

**Change annotation target** (`scheduler.py`):
```python
total_annotations_needed = len(case_ids) * 3  # Change 3 here
```

**Disable auto-balancing by default** (`templates/demographics.html`):
```html
<input type="checkbox" name="auto_balance_models" checked>  <!-- Remove 'checked' -->
```

## Data Structure

### Required Files
```
../experiments/
├── cache/{dataset}/agent_*.json          # Agent responses
└── cases/{dataset}/principal_*.json      # Manipulative cases
```

### Output
```
results/{dataset}/{case_id}_{annotator_id}_{timestamp}.json
```

### Annotation Format
- **Metadata**: `annotator_id`, `demographics`, `dataset`, `model_key`, `case_id`
- **Responses**: `step1`, `step2`, `step3` (answers + confidence)
- **Changes**: `step1_to_step2_changes`, `step2_to_step3_changes`
- **Extras**: `reasoning`, `highlights`, timestamps

## Usage

### For Annotators
1. Review consent form
2. Enter demographics, select dataset (or use auto-balancing)
3. Annotate cases through 3-step process
4. Receive completion code

### Case Selection Modes
- **All cases**: Sequential annotation
- **Specific indices**: Target cases by index (e.g., "0,5,10-15")
- **Manipulative cases**: Smart selection of 10 cases (default)

## Configuration

### Add Dataset (`config.py`)
```python
DATASETS['new_key'] = {
    'name': 'Display Name',
    'data_dir': '../experiments/cache/new_dataset',
    'cases_dir': '../experiments/cases/new_dataset',
    'annotation_dir': 'results/new_dataset'
}
```

### Add Model (`config.py`)
```python
AVAILABLE_MODELS.append({
    'key': 'model_key',
    'file': 'agent_filename.json',
    'name': 'Display Name'
})
```

## Testing & Analysis

```bash
# Test model balancing logic
python test_model_balancing.py

# Analyze annotation coverage
python analyze_coverage.py

# API endpoint for coverage stats
curl http://localhost:8000/api/coverage/{dataset}/{model}
```

## Dependencies

- Flask 3.0.0
- markdown 3.5.1

See `requirements.txt` for complete list.

## Security Notes

- Change Flask secret key in production (`app.py`)
- App runs on `0.0.0.0:8000` for development
- Session data encrypted with secret key

## Module Dependencies

```
config.py → data_loader.py → scheduler.py → annotation_utils.py → app.py
```
No circular dependencies. Each module independently testable.
