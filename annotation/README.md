# Clinical Decision Annotation Interface

A Flask web application for collecting human annotations on clinical decision-making with AI-generated recommendations. This system studies how clinicians' decisions change when exposed to AI analysis and ground truth information.

## Overview

This annotation interface presents clinical cases to human annotators in a three-step process:

### For MIMIC-IV Cases (Treatment Planning):
1. **Step 1**: Annotators review the clinical context and provide initial treatment plans (medications, procedures, diagnoses) with confidence ratings
2. **Step 2**: Annotators see the AI agent's recommendation and can revise their treatment plans
3. **Step 3**: Annotators see the actual treatment (ground truth) and make final revisions

### For USMLE Cases (Multiple Choice):
1. **Step 1**: Annotators review the clinical question and select an initial answer with confidence rating
2. **Step 2**: Annotators see the AI's analysis and can revise their answer
3. **Step 3**: Annotators see the correct answer and make final revisions

## Features

- **Multiple Dataset Support**: MIMIC-IV, USMLE, and USMLE Sample datasets
- **Multiple Model Support**: Compare annotations across different AI models (Llama variants, DeepSeek)
- **Consent Management**: Built-in IRB consent form and demographics collection
- **Smart Case Selection**: Prioritizes cases with fewer annotations for balanced coverage
- **Text Highlighting**: Annotators can highlight important text in AI recommendations
- **Progress Tracking**: Real-time annotation coverage statistics
- **Markdown Rendering**: Clean, readable formatting of clinical data with proper headings, lists, and tables

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data files are organized as:
```
annotation/
└── ../experiments/
    ├── cache/
    │   ├── mimiciv_demo/
    │   │   └── agent_*.json
    │   ├── usmle/
    │   │   └── agent_*.json
    │   └── usmle_sample/
    │       └── agent_*.json
    └── cases/
        ├── mimiciv_demo/
        │   └── principal_*.json
        ├── usmle/
        │   └── principal_*.json
        └── usmle_sample/
            └── principal_*.json
```

## Running the Application

### Quick Start
```bash
./start.sh
```

### Manual Start
```bash
python app.py
```

The application will start on `http://localhost:8000`

## Usage

### For Annotators

1. **Consent**: Review and agree to the consent form
2. **Demographics**: Enter demographic information and select:
   - Dataset (MIMIC-IV, USMLE, or USMLE Sample)
   - Model to annotate
   - Case selection mode (all cases, specific indices, or manipulative cases)
3. **Annotation**: Follow the three-step process for each case
4. **Completion**: Receive a unique completion code at the end

### Case Selection Modes

- **All Cases**: Annotate all cases in order
- **Targeted - Specific Indices**: Annotate specific cases by index (e.g., "0,5,10-15")
- **Targeted - Manipulative Cases**: Smart selection of 10 manipulative cases, prioritizing those with fewer existing annotations

### Coverage Tracking

Access `/api/coverage/<dataset>/<model>` to view annotation statistics:
- Total cases requiring annotation
- Cases by annotation count (0, 1, 2, 3+ annotations)
- Overall progress percentage

## Data Storage

Annotations are saved as JSON files in the `results/` directory:

```
results/
├── mimic/
│   └── {case_id}_{annotator_id}_{timestamp}.json
├── usmle/
│   └── {case_id}_{annotator_id}_{timestamp}.json
└── usmle_sample/
    └── {case_id}_{annotator_id}_{timestamp}.json
```

### Annotation Format

Each annotation file includes:

**Metadata:**
- `annotator_id`: Unique identifier for the annotator
- `demographics`: Age, expertise, years of practice, etc.
- `dataset`: Dataset key (mimic, usmle, usmle_sample)
- `model_key`: Model key (llama, llama_small, llama_large, deepseek)
- `case_id`: Unique case identifier

**Three Steps of Responses:**
- `step1`: Initial response before seeing AI recommendation
- `step2`: Revised response after seeing AI recommendation
- `step3`: Final response after seeing ground truth

**Change Tracking:**
- `step1_to_step2_changes`: What changed after seeing AI
- `step2_to_step3_changes`: What changed after seeing ground truth

**Additional Data:**
- `reasoning`: Free-text explanation of decision-making
- `highlights`: Text snippets highlighted by the annotator
- `step1_time`, `step2_time`, `step3_time`: Timestamps for each step

## File Structure

```
annotation/
├── app.py                      # Flask application
├── analyze_coverage.py         # Coverage analysis utility
├── requirements.txt            # Python dependencies
├── start.sh                    # Startup script
├── README.md                   # This file
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── consent_form.html      # Consent form
│   ├── demographics.html      # Demographics and case selection
│   ├── overview.html          # Case overview table
│   ├── step1.html             # Step 1 (MIMIC)
│   ├── step1_usmle.html       # Step 1 (USMLE)
│   ├── step2.html             # Step 2 (MIMIC)
│   ├── step2_usmle.html       # Step 2 (USMLE)
│   ├── step3.html             # Step 3 (MIMIC)
│   ├── step3_usmle.html       # Step 3 (USMLE)
│   └── summary.html           # Completion summary
├── files/                      # Static files (consent documents, etc.)
└── results/                    # Annotation results
    ├── mimic/
    ├── usmle/
    └── usmle_sample/
```

## Configuration

### Adding New Datasets

Edit the `DATASETS` dictionary in `app.py`:

```python
DATASETS = {
    'your_dataset_key': {
        'name': 'Display Name',
        'data_dir': '../experiments/cache/your_dataset',
        'cases_dir': '../experiments/cases/your_dataset',
        'annotation_dir': 'results/your_dataset'
    }
}
```

### Adding New Models

Edit the `AVAILABLE_MODELS` list in `app.py`:

```python
AVAILABLE_MODELS = [
    {'key': 'model_key', 'file': 'agent_filename.json', 'name': 'Display Name'}
]
```

## Development

### Running in Debug Mode

The application runs in debug mode by default:
```python
app.run(debug=True, host='0.0.0.0', port=8000)
```

### Analyzing Coverage

Use the coverage analysis script to check annotation progress:
```bash
python analyze_coverage.py
```

## Security Notes

- Change the Flask secret key in production (line 8 of `app.py`)
- The application is configured to run on all interfaces (`0.0.0.0`) for development
- Session data is stored in Flask sessions with the secret key

## Dependencies

- Flask 3.0.0
- markdown 3.5.1

See `requirements.txt` for complete list.
