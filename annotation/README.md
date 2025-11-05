# Clinical Decision Annotation System

A Flask web application for collecting human annotations on clinical decision-making with AI-generated treatment recommendations.

## Overview

This system presents clinical cases to human annotators in a two-step process:

1. **Step 1**: Annotators see minimal patient information (principal context) and the AI agent's recommendation, then make an initial accept/reject decision
2. **Step 2**: Annotators see the full clinical context and ground truth, then can revise their decision

Both decisions are saved for comparison to study how additional information affects clinical decision-making.

**Key Features:**
- Markdown rendering for AI recommendations and clinical context
- Clean, readable formatting with proper headings, lists, tables, and emphasis
- Responsive web interface optimized for clinical data review

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the data file exists at: `../experiments/cache/agent_deepseek.json`

## Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. Open the application in a web browser
2. Enter your annotator ID
3. Optionally choose a starting case index
4. Follow the two-step annotation process for each case
5. Annotations are automatically saved to the `annotations/` directory

## Data Storage

Annotations are saved as JSON files in the `annotations/` directory with the following naming convention:
```
{case_id}_{annotator_id}_{timestamp}.json
```

Each annotation includes:
- Case identifiers (case_id, hadm_id, subject_id)
- Agent information (name, model)
- Initial decision (from Step 1)
- Final decision (from Step 2)
- Whether the decision changed
- Optional reasoning text
- Timestamps for each step

## File Structure

```
annotation/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── step1.html
│   ├── step2.html
│   └── complete.html
└── annotations/           # Saved annotations (created automatically)
```
