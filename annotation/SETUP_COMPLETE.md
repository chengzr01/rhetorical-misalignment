# Setup Complete - Clinical Decision Annotation System

## What Was Created

A complete Flask web application for collecting human annotations on clinical decision-making with AI-generated treatment recommendations.

### Key Components

1. **Flask Backend** (`app.py`)
   - Session management for tracking annotator progress
   - Markdown rendering for clinical content
   - Automatic data persistence
   - 6 routes handling the complete workflow

2. **HTML Templates** (`templates/`)
   - `base.html` - Base template with comprehensive CSS styling
   - `index.html` - Landing page for annotator registration
   - `step1.html` - Initial decision page (limited context)
   - `step2.html` - Revised decision page (full context + ground truth)
   - `complete.html` - Completion page

3. **Markdown Rendering**
   - AI recommendations properly formatted with headings, lists, bold/italic
   - Clinical context displayed with clear structure
   - Tables, code blocks, and blockquotes supported
   - Improved readability for complex medical information

4. **Data Management**
   - Reads from: `../experiments/cache/agent_deepseek.json` (275 cases)
   - Saves to: `annotations/{case_id}_{annotator_id}_{timestamp}.json`
   - Each annotation includes: initial decision, final decision, reasoning, timestamps

## Features

### Two-Step Annotation Process

**Step 1: Initial Decision**
- Shows: Principal context (basic patient info)
- Shows: AI agent recommendation (with markdown rendering)
- Collects: Accept or Reject decision

**Step 2: Revised Decision**
- Shows: Full clinical context (detailed patient data)
- Shows: Ground truth (actual treatments, procedures, diagnoses)
- Shows: AI agent recommendation again for reference
- Collects: Final Accept or Reject decision + optional reasoning

### Markdown Rendering Benefits

The AI recommendations contain rich formatting:
- `### Headings` → Properly styled section headers
- `**Bold text**` → Highlighted important medications and terms
- `* Lists` → Organized treatment plans and monitoring parameters
- Tables → Structured monitoring schedules
- `*Italics*` → Emphasized clinical notes

All of this is now properly rendered in HTML for easy reading!

## How to Run

### Quick Start
```bash
cd annotation
./start.sh
```

### Manual Start
```bash
conda activate alignment
python app.py
```

Access at: **http://localhost:5000**

## Testing

Run the test suite:
```bash
conda activate alignment
python test_app.py         # Test data loading and structure
python test_markdown.py    # Test markdown rendering
```

## Data Flow

```
User Registration
    ↓
Step 1: Limited Context + AI Recommendation
    ↓ (Accept/Reject)
Session Saved
    ↓
Step 2: Full Context + Ground Truth + AI Recommendation
    ↓ (Accept/Reject + Reasoning)
Annotation Saved to File
    ↓
Next Case (or Complete)
```

## Saved Annotation Format

```json
{
  "annotator_id": "user123",
  "case_id": "21606243_10031404",
  "hadm_id": 21606243,
  "subject_id": 10031404,
  "agent_name": "default_agent",
  "agent_model": "deepseek-ai/deepseek-v3.1",
  "initial_decision": "accept",
  "final_decision": "reject",
  "decision_changed": true,
  "reasoning": "After seeing full labs, the recommendation doesn't account for renal impairment.",
  "session_start": "2025-11-04T12:00:00",
  "step1_time": "2025-11-04T12:05:00",
  "step2_time": "2025-11-04T12:10:00",
  "timestamp": "2025-11-04T12:10:00"
}
```

## Files Created

```
annotation/
├── app.py                              # Flask application
├── requirements.txt                    # Python dependencies
├── start.sh                            # Startup script
├── test_app.py                         # Application tests
├── test_markdown.py                    # Markdown rendering tests
├── README.md                           # Technical documentation
├── USAGE.md                            # User guide
├── SETUP_COMPLETE.md                   # This file
├── templates/
│   ├── base.html                       # Base template + CSS
│   ├── index.html                      # Landing page
│   ├── step1.html                      # Step 1 (initial decision)
│   ├── step2.html                      # Step 2 (revised decision)
│   └── complete.html                   # Completion page
└── annotations/                        # Annotation storage (auto-created)
    └── SAMPLE_annotation_format.json   # Example annotation

Total: 275 clinical cases ready for annotation
```

## Next Steps

1. **Start the application**: Run `./start.sh`
2. **Test the workflow**: Annotate a few sample cases
3. **Verify data saving**: Check `annotations/` directory for saved files
4. **Begin real annotations**: Enter your annotator ID and start

## Troubleshooting

If the app won't start:
```bash
# Ensure markdown is installed
conda activate alignment
pip install markdown==3.5.1

# Verify data file exists
ls -l ../experiments/cache/agent_deepseek.json

# Check for errors
python app.py
```

## Research Applications

This system enables study of:
1. Information availability effects on clinical decision-making
2. AI recommendation acceptance rates
3. Decision revision patterns when ground truth is revealed
4. Factors influencing clinician trust in AI systems
5. Differences between AI models' recommendation quality
