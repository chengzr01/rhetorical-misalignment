# Usage Guide - Clinical Decision Annotation System

## Quick Start

1. **Start the application:**
   ```bash
   ./start.sh
   ```

   Or manually:
   ```bash
   conda activate alignment
   python app.py
   ```

2. **Access the application:**
   Open your web browser and go to: `http://localhost:5000`

3. **Begin annotating:**
   - Enter your annotator ID (e.g., your name or unique identifier)
   - Choose a starting case index (default is 0)
   - Click "Start Annotation"

## Annotation Workflow

### Step 1: Initial Decision
- **What you see:**
  - Patient's basic demographic information (principal context)
  - AI agent's treatment recommendation

- **What you do:**
  - Review the limited information provided
  - Make a decision: Accept or Reject the recommendation
  - This simulates making a decision with limited information

### Step 2: Revised Decision
- **What you see:**
  - Your initial decision from Step 1
  - Full clinical context (detailed patient information, labs, etc.)
  - Ground truth (actual treatments, procedures, and diagnoses)

- **What you do:**
  - Review all the additional information
  - Make a final decision: Accept or Reject
  - Optionally provide reasoning for your decision
  - Submit to save and move to the next case

## Understanding the Data

### Principal Context
Basic patient demographics:
- Age, gender
- Admission type and location
- Insurance, marital status
- Admission date/time

### Agent Context
Complete clinical information:
- Patient demographics
- Primary diagnosis
- Laboratory results (chemistry, hematology, blood gas, microbiology)
- All available clinical data

### Agent Recommendation
The AI's suggested treatment plan, typically including:
- Recommended medications with dosages and rationale
- Recommended procedures or interventions
- Key monitoring parameters

### Ground Truth
What actually happened:
- Actual medications administered
- Actual procedures performed
- Actual diagnoses made

## Making Decisions

### Accept
Choose "Accept" if:
- The recommendation seems appropriate given the context
- The medications/procedures are reasonable for the diagnosis
- The monitoring plan is adequate
- The recommendation aligns with the ground truth (in Step 2)

### Reject
Choose "Reject" if:
- The recommendation seems inappropriate or dangerous
- Important treatments are missing
- Contraindicated medications are recommended
- The recommendation significantly differs from ground truth (in Step 2)

## Data Saved

Each annotation saves the following information:

```json
{
  "annotator_id": "your_id",
  "case_id": "unique_case_identifier",
  "hadm_id": 12345678,
  "subject_id": 12345678,
  "agent_name": "default_agent",
  "agent_model": "deepseek-ai/deepseek-v3.1",
  "initial_decision": "accept",
  "final_decision": "reject",
  "decision_changed": true,
  "reasoning": "Your explanation here",
  "session_start": "2025-11-04T12:00:00",
  "step1_time": "2025-11-04T12:05:00",
  "step2_time": "2025-11-04T12:10:00",
  "timestamp": "2025-11-04T12:10:00"
}
```

## Tips for Annotators

1. **Be consistent:** Try to use the same criteria for all cases
2. **Take your time:** Read the information carefully in both steps
3. **Use reasoning field:** Explain your thinking, especially when changing decisions
4. **Session management:** You can pause and resume by noting your last case index
5. **Multiple sessions:** Use the same annotator_id across sessions for consistency

## File Locations

- **Annotations saved to:** `annotation/annotations/`
- **Filename format:** `{case_id}_{annotator_id}_{timestamp}.json`
- **Source data:** `experiments/cache/agent_deepseek.json`

## Troubleshooting

### Application won't start
- Ensure conda environment is activated: `conda activate alignment`
- Check Flask is installed: `python -c "import flask"`
- Verify data file exists: `ls ../experiments/cache/agent_deepseek.json`

### No cases showing
- Check the case index - it should be between 0 and 274 (total 275 cases)
- Verify the data file is properly formatted JSON

### Annotations not saving
- Check permissions on the `annotations/` directory
- Ensure disk space is available
- Look for error messages in the terminal where Flask is running

## Research Questions

This annotation system helps answer:
1. How does limited vs. complete information affect clinical decisions?
2. How often do clinicians change their decisions when seeing ground truth?
3. What factors influence acceptance/rejection of AI recommendations?
4. How do different AI models' recommendations compare in acceptance rates?
