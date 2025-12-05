# USMLE Sample Question Parser

This directory contains scripts to parse USMLE sample questions from plain text files into structured JSON format.

## Overview

The parsing workflow consists of two scripts:

1. **`parse_usmle_sample.py`**: Parses plain text question and answer files into JSON
2. **`generate_usmle_sample.py`**: Converts the parsed JSON into the format used for agent/principal inference

## Sanity Checks

Both scripts include comprehensive sanity checks to ensure data quality:

### Parse Script Checks
- Verifies each answer letter (A, B, C, etc.) corresponds to an actual option
- Checks for missing answers
- Warns about empty option text

### Generate Script Checks
- Verifies answer text exists in the options dictionary
- Confirms answer_idx exists as an option key
- Ensures answer text exactly matches the option at answer_idx
- Reports any mismatches with detailed error messages

All 338 parsed questions pass these sanity checks, ensuring that every answer correctly maps to its corresponding option.

### Running Verification

To verify the quality of the parsed data, run:

```bash
python verify_usmle_parsing.py
```

This comprehensive verification script checks:
- No duplicate question numbers or IDs
- All required fields are present
- All answers match their corresponding options
- Answer text consistency between parsed and generated formats
- Meta_info distribution across steps

## Usage

### Step 1: Parse Plain Text Files

```bash
python parse_usmle_sample.py
```

This script reads the plain text files from `../datasets/usmle_sample/` and creates parsed JSON files:
- `Step1_Questions.txt` + `Step1_Answers.txt` → `Step1_questions_parsed.json`
- `Step2_Questions.txt` + `Step2_Answers.txt` → `Step2_CK_questions_parsed.json`
- `Step3_Sample_Items.txt` + `Step3_Answers.txt` → `Step3_questions_parsed.json`

**Options:**
- `--input-dir`: Directory containing the plain text files (default: `../datasets/usmle_sample`)
- `--output-dir`: Directory to save the parsed JSON files (default: `../datasets/usmle_sample`)

### Step 2: Generate Final Format

```bash
python generate_usmle_sample.py
```

This script converts the parsed JSON files into the format used for experiments:
- Output: `input/clinical_questions_usmle_sample.json`

**Options:**
- `--datasets-dir`: Directory containing the parsed JSON files (default: `../datasets/usmle_sample`)
- `--output`: Output file path (default: `input/clinical_questions_usmle_sample.json`)
- `--limit`: Limit number of questions per step for testing (optional)

## Question Format

### Input (Plain Text)
Questions follow these patterns:
```
1. A 67-year-old woman comes to the office...
(A) Option A text
(B) Option B text
(C) Option C text
...

2. Patient Information
Age: 6 years
...
Question: Which of the following...
(A) Option A
...
```

Answer files contain:
```
1. B
2. A
3. C
...
```

### Output (Parsed JSON)
```json
{
  "exam": "USMLE Step 1",
  "question_number": 1,
  "stem": "A 67-year-old woman comes to...",
  "options": [
    {"label": "A", "text": "Option A text"},
    {"label": "B", "text": "Option B text"}
  ],
  "answer": "B"
}
```

### Final Format (Generated JSON)
```json
{
  "id": "usmle_sample_0",
  "question": "A 67-year-old woman comes to...",
  "options": {
    "A": "Option A text",
    "B": "Option B text"
  },
  "answer": "Option B text",
  "answer_idx": "B",
  "meta_info": "step1",  // or "step2" or "step3" based on exam type
  "metamap_phrases": [...]
}
```

## Parsing Details

### Recognized Question Patterns
The parser recognizes questions that start with many different patterns:
- **Clinical presentations**: `A/An [age]-year-old...`, `A/An [age] -year-old...` (handles spacing variations)
- **Patient information format**: `Patient Information`
- **Study questions**: `During an experiment`, `During a study`, `Over [time], a study`, `A randomized [trial/study]`
- **Special populations**: `Six healthy subjects`, `An asymptomatic`, `An otherwise healthy`, `A healthy`, `A sexually active`
- **Time-based**: `After being`, `After undergoing`, `Two/Three/... weeks/days/months after`
- **Research contexts**: `A phase [number]`, `A trial`, `Researchers`, `A researcher`
- **Shared backgrounds**: `Which of the following...` (for questions >= 10)
- **Item references**: `Items #` (for linked question sets)

### Known Limitations
Some questions cannot be parsed due to:
- **Image-based options**: Questions where options are labeled points in diagrams (e.g., "labeled sites in the photograph")
- **Table-based options**: Questions with options formatted as tables with arrows and special characters
- **Formatting issues**: Unusual text formatting in the source files

In the current dataset:
- **Step 1**: 110/119 questions parsed (92.4%)
- **Step 2 CK**: 102/120 questions parsed (85.0%)
- **Step 3**: 130/137 questions parsed (94.9%)
- **Total**: 342/376 questions successfully parsed (91.0%)

### Special Handling

#### Shared Background in Item Sets
The parser automatically detects and handles questions in item sets that share a common clinical background:

- **Detection**: Questions starting with "Which of the following", "Appropriate treatment", "After", "Following", "In addition to", or very short stems (< 200 chars)
- **Extraction**: The shared background is extracted from the clinical scenario paragraph that appears before the first question in the set
- **Propagation**: For continuation questions (2nd, 3rd in a set), the background is copied from the previous question if direct extraction fails
- **Result**: Each question in an item set includes the full clinical context, even if it wasn't explicitly repeated in the source text

**Example (Step 2 Questions 56-57)**:
- Both questions now include the complete background: "A 57-year-old woman comes to the emergency department..."
- Q56 adds: "In addition to intravenous administration..."
- Q57 adds: "After intravenous administration..."

**Example (Step 3 Questions 62-63)**:
- Both questions include: "A 16-year-old boy comes to the office..."
- Q62 adds: "Which of the following is the most likely causal agent..."
- Q63 adds: "Appropriate treatment is prescribed..."

#### Other Special Handling
- Options with irregular spacing or formatting (blank lines between label and text)
- Questions with "Patient Information" format
- Avoiding false positives from text like "mm³" split across lines
- Skipping "Items #X-Y" markers and instruction text that appears in item sets

## Troubleshooting

**Issue**: Some questions are not parsed
- Check if the question references images/diagrams for options
- Check if the question has table-based options
- Verify the question follows one of the recognized patterns

**Issue**: Wrong options extracted
- The parser uses 3+ consecutive blank lines as a signal that options have ended
- Check for unusual formatting in the source text

**Issue**: Duplicate question numbers
- The parser tracks seen question numbers and skips duplicates
- This handles cases like "mm³" being split across lines

## Examples

Parse with custom directories:
```bash
python parse_usmle_sample.py --input-dir /path/to/text/files --output-dir /path/to/output
```

Generate a sample for testing:
```bash
python generate_usmle_sample.py --limit 10
```

Generate to a custom location:
```bash
python generate_usmle_sample.py --output my_questions.json
```
