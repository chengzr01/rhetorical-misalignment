"""
Parse USMLE sample questions from plain text files.

This script reads plain text question and answer files from the USMLE sample
datasets and converts them to structured JSON format for use with generate_usmle_sample.py.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple


def parse_answers(answer_file: Path) -> Dict[int, str]:
    """
    Parse the answer file to get a mapping of question number to answer letter.

    Args:
        answer_file: Path to the answer text file

    Returns:
        Dictionary mapping question number to answer letter
    """
    answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse lines like "1. B" or "10. A"
            match = re.match(r'^(\d+)\.\s*([A-Z])$', line)
            if match:
                question_num = int(match.group(1))
                answer_letter = match.group(2)
                answers[question_num] = answer_letter
    return answers


def extract_options(text: str, start_pos: int) -> Tuple[List[Dict[str, str]], int]:
    """
    Extract options from the question text starting at start_pos.

    Args:
        text: Full question text
        start_pos: Position to start looking for options

    Returns:
        Tuple of (list of options, position where options end)
    """
    options = []
    lines = text[start_pos:].split('\n')

    current_option = None
    option_text = []
    end_line_idx = 0
    empty_line_count = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Check if this line starts a new option (e.g., "(A) Some text" or "(A)Some text")
        option_match = re.match(r'^\(([A-Z])\)\s*(.*)$', line)

        if option_match:
            # Save previous option if it exists
            if current_option is not None:
                options.append({
                    'label': current_option,
                    'text': ' '.join(option_text).strip()
                })
                option_text = []

            # Start new option
            current_option = option_match.group(1)
            option_text_part = option_match.group(2).strip()
            if option_text_part:  # Only add if there's text
                option_text.append(option_text_part)
            empty_line_count = 0
        elif current_option is not None and line:
            # This is a continuation of the current option
            # But check if it's the start of a new question
            if re.match(r'^\d+\.\s', line):
                # This is the start of a new question, stop here
                end_line_idx = i
                break
            option_text.append(line)
            empty_line_count = 0
        elif current_option is not None and not line:
            # Empty line - could be between option label and text, or end of options
            empty_line_count += 1
            # If we see 3+ consecutive empty lines, assume options have ended
            if empty_line_count >= 3:
                end_line_idx = i + 1
                break

    # Save the last option
    if current_option is not None:
        options.append({
            'label': current_option,
            'text': ' '.join(option_text).strip()
        })

    if end_line_idx == 0:
        end_line_idx = len(lines)

    # Calculate actual position in original text
    end_pos = start_pos + sum(len(line) + 1 for line in lines[:end_line_idx])

    return options, end_pos


def extract_shared_background(content: str, question_start: int) -> str:
    """
    Extract shared background text that appears before a question.

    This is used for questions in item sets where multiple questions share a
    common clinical scenario.

    Args:
        content: Full text content
        question_start: Position where the question number starts

    Returns:
        Background text (empty string if none found)
    """
    # Look backward from question_start to find the background
    # The background is typically a clinical scenario paragraph that appears
    # immediately before the question

    search_start = max(0, question_start - 3000)  # Look back up to 3000 chars
    preceding_text = content[search_start:question_start]

    # Split into lines and work backward
    lines_before = preceding_text.split('\n')

    # Remove trailing empty lines
    while lines_before and not lines_before[-1].strip():
        lines_before.pop()

    if not lines_before:
        return ''

    # Collect the background paragraph
    # Start from the end and go backward until we hit:
    # - A question number pattern
    # - An "END OF SET" marker
    # - An option pattern like "(A)"
    # BUT: Skip "Items #" markers and instruction text - the background comes after them

    background_lines = []
    found_content = False
    skip_items_marker = False  # Whether we've encountered and passed the "Items #" marker

    for line in reversed(lines_before):
        stripped = line.strip()

        # Stop if we hit a question number
        if re.match(r'^\d+\.\s+', stripped):
            break

        # Stop if we hit "END OF SET"
        if 'END OF SET' in stripped:
            break

        # Skip "Items #" markers and the instruction text that follows
        # The background comes AFTER these
        if re.match(r'^Items\s+#', stripped):
            skip_items_marker = True
            continue

        # Skip instruction text (appears after "Items #")
        if ('In the actual examination environment' in stripped or
            'Proceed to Next Item' in stripped or
            'first item' in stripped or
            'second item' in stripped):
            skip_items_marker = True
            continue

        # Stop if we hit an option
        if re.match(r'^\([A-Z]\)', stripped):
            break

        # Add the line
        if stripped:
            background_lines.insert(0, line)
            found_content = True
        elif found_content:
            # Empty line in the middle of content - might be paragraph break
            # Only include if it's not too many empty lines
            if len(background_lines) > 0:
                background_lines.insert(0, line)

    if not background_lines:
        return ''

    background_text = '\n'.join(background_lines).strip()

    # Only return if it's substantial and starts with a clinical description
    # (typically starts with "A [age]-year-old" or similar)
    if len(background_text) > 100:
        # Check if it looks like a clinical scenario
        if re.match(r'^(?:A|An)\s+\d+', background_text, re.IGNORECASE):
            return background_text

    return ''


def parse_questions(question_file: Path, answers: Dict[int, str], exam_name: str) -> List[Dict[str, Any]]:
    """
    Parse questions from a question text file.

    Args:
        question_file: Path to the question text file
        answers: Dictionary mapping question number to answer letter
        exam_name: Name of the exam (e.g., "USMLE Step 1")

    Returns:
        List of parsed questions
    """
    with open(question_file, 'r', encoding='utf-8') as f:
        content = f.read()

    questions = []

    # Find all question starts with multiple patterns
    # Questions can start with many different patterns. We'll use multiple regex patterns:

    patterns = [
        # Pattern 1: "number. A/An [age]-year-old" or "number. A/An [age] -year-old" (note space variations)
        r'(?:^|\n)(\d{1,3})\.\s+(?:A|An)\s+\d+\s*[\s-](?:year|month|day|week)',

        # Pattern 2: "number. Patient Information"
        r'(?:^|\n)(\d{1,3})\.\s+Patient\s+Information',

        # Pattern 3: Research/study questions
        r'(?:^|\n)(\d{1,3})\.\s+(?:During an experiment|Over \d+|A randomized|Six healthy|[A-Z]\w+ healthy)',

        # Pattern 4: Clinical presentations with specific adjectives
        r'(?:^|\n)(\d{1,3})\.\s+An? (?:asymptomatic|otherwise healthy)',

        # Pattern 5: "After being..."
        r'(?:^|\n)(\d{1,3})\.\s+After (?:being|undergoing)',

        # Pattern 6: Questions starting with items reference
        r'(?:^|\n)(\d{1,3})\.\s+Items #',

        # Pattern 7: Questions starting with "A phase" or research designs
        r'(?:^|\n)(\d{1,3})\.\s+A (?:phase|study|trial)',

        # Pattern 8: Questions starting with "Researchers" or "A researcher"
        r'(?:^|\n)(\d{1,3})\.\s+(?:A researcher|Researchers)',

        # Pattern 9: Questions starting with "Two weeks after" or similar time references
        r'(?:^|\n)(\d{1,3})\.\s+(?:Two|Three|Four|Five|Six) (?:weeks?|days?|months?|hours?) (?:after|later)',

        # Pattern 10: Questions starting with test/device/procedure descriptions
        r'(?:^|\n)(\d{1,3})\.\s+A (?:new|sexually active|healthy|study)',

        # Pattern 11: Questions starting with procedure descriptions
        r'(?:^|\n)(\d{1,3})\.\s+During a (?:study|trial|routine)',

        # Pattern 12: Item set questions that start with "Which of the following" (any number)
        r'(?:^|\n)(\d{1,3})\.\s+Which\s+of\s+the\s+following',

        # Pattern 13: Item set continuation questions
        r'(?:^|\n)(\d{1,3})\.\s+(?:Appropriate|After|Following|Based on|In addition to)',
    ]

    all_matches = []
    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.MULTILINE | re.IGNORECASE)
        all_matches.extend(pattern.finditer(content))

    # Sort by position and remove duplicates
    matches = sorted(all_matches, key=lambda m: m.start())

    # Track seen question numbers to avoid duplicates (e.g., from "mm³" split across lines)
    seen_questions = set()

    for i, match in enumerate(matches):
        question_num = int(match.group(1))

        # Skip if we've already seen this question number
        if question_num in seen_questions:
            continue
        seen_questions.add(question_num)

        # Start from the beginning of the match (after the number and period)
        # to preserve the full question text
        # If match started with \n, skip it
        match_start = match.start()
        if match_start > 0 and content[match_start] == '\n':
            match_start += 1

        question_start = match_start + len(str(question_num)) + 1  # +1 for the period
        while question_start < len(content) and content[question_start].isspace():
            question_start += 1

        # Find where this question ends (start of next question or end of file)
        if i + 1 < len(matches):
            next_match_start = matches[i + 1].start()
            if next_match_start > 0 and content[next_match_start] == '\n':
                next_match_start += 1
            question_end = next_match_start
        else:
            question_end = len(content)

        question_text = content[question_start:question_end].strip()

        # Find where the options start
        # Options typically start with (A), (B), etc.
        option_pattern = re.compile(r'\(([A-Z])\)\s')
        option_matches = list(option_pattern.finditer(question_text))

        if not option_matches:
            print(f"Warning: No options found for question {question_num} in {exam_name}")
            continue

        # The stem is everything before the first option
        first_option_pos = option_matches[0].start()
        stem = question_text[:first_option_pos].strip()

        # Check if this question might have a shared background
        # This happens in item sets where the stem is very short (just the question)
        # Look for shared background if stem is short or starts with certain patterns
        if len(stem) < 200 or re.match(r'^\s*(?:Which|Appropriate|After|Following|Based on|In addition to)', stem, re.IGNORECASE):
            background = extract_shared_background(content, match_start)

            # If we couldn't find a background directly (e.g., for continuation questions),
            # check if the previous question has a background we can reuse
            if not background and question_num > 1:
                # Look for the previous question in the already-parsed list
                prev_q = next((q for q in questions if q['question_number'] == question_num - 1), None)
                if prev_q:
                    # Check if the previous question's stem has a clinical background
                    # (long stem starting with "A [age]-year-old")
                    prev_stem = prev_q['stem']
                    if len(prev_stem) > 500 and re.match(r'^(?:A|An)\s+\d+', prev_stem, re.IGNORECASE):
                        # Extract the background portion (before the actual question)
                        # The background typically ends with the clinical findings,
                        # before "Which of the following" or similar question starts
                        match_q = re.search(r'\n\n(?:Which of the following|What is|In addition to)', prev_stem, re.IGNORECASE)
                        if match_q:
                            background = prev_stem[:match_q.start()].strip()

            if background:
                # Prepend the background to the stem
                stem = background + '\n\n' + stem

        # Extract options
        options, _ = extract_options(question_text, first_option_pos)

        if not options:
            print(f"Warning: Could not parse options for question {question_num} in {exam_name}")
            continue

        # Get the answer
        answer = answers.get(question_num)
        if answer is None:
            print(f"Warning: No answer found for question {question_num} in {exam_name}")

        questions.append({
            'exam': exam_name,
            'question_number': question_num,
            'stem': stem,
            'options': options,
            'answer': answer
        })

    return questions


def sanity_check_questions(questions: List[Dict[str, Any]], exam_name: str) -> None:
    """
    Perform sanity checks on parsed questions.

    Args:
        questions: List of parsed questions
        exam_name: Name of the exam for error reporting
    """
    issues_found = 0

    for question in questions:
        q_num = question['question_number']
        answer = question.get('answer')
        options = question.get('options', [])

        # Check if answer is provided
        if answer is None:
            print(f"  WARNING: Question {q_num} has no answer")
            issues_found += 1
            continue

        # Check if answer corresponds to an actual option
        option_labels = [opt['label'] for opt in options]
        if answer not in option_labels:
            print(f"  ERROR: Question {q_num} answer '{answer}' not found in options {option_labels}")
            issues_found += 1
            continue

        # Check if answer text can be found (optional - just for verification)
        answer_option = next((opt for opt in options if opt['label'] == answer), None)
        if answer_option and not answer_option['text']:
            print(f"  WARNING: Question {q_num} answer option '{answer}' has empty text")
            issues_found += 1

    if issues_found > 0:
        print(f"  Found {issues_found} issue(s) in {exam_name}")
    else:
        print(f"  Sanity check passed: All answers match options")


def parse_step(
    question_file: Path,
    answer_file: Path,
    exam_name: str
) -> List[Dict[str, Any]]:
    """
    Parse a complete USMLE step (questions + answers).

    Args:
        question_file: Path to the question text file
        answer_file: Path to the answer text file
        exam_name: Name of the exam (e.g., "USMLE Step 1")

    Returns:
        List of parsed questions
    """
    print(f"Parsing {exam_name}...")

    # Parse answers first
    answers = parse_answers(answer_file)
    print(f"  Found {len(answers)} answers")

    # Parse questions
    questions = parse_questions(question_file, answers, exam_name)
    print(f"  Parsed {len(questions)} questions")

    # Sanity check
    sanity_check_questions(questions, exam_name)

    return questions


def main():
    """Main function to parse all USMLE sample datasets."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse USMLE sample questions from plain text files'
    )
    parser.add_argument(
        '--input-dir',
        default='../datasets/usmle_sample',
        help='Directory containing the plain text question and answer files'
    )
    parser.add_argument(
        '--output-dir',
        default='../datasets/usmle_sample',
        help='Directory to save the parsed JSON files'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the steps to process
    steps = [
        {
            'question_file': 'Step1_Questions.txt',
            'answer_file': 'Step1_Answers.txt',
            'exam_name': 'USMLE Step 1',
            'output_file': 'Step1_questions_parsed.json'
        },
        {
            'question_file': 'Step2_Questions.txt',
            'answer_file': 'Step2_Answers.txt',
            'exam_name': 'USMLE Step 2 CK',
            'output_file': 'Step2_CK_questions_parsed.json'
        },
        {
            'question_file': 'Step3_Sample_Items.txt',
            'answer_file': 'Step3_Answers.txt',
            'exam_name': 'USMLE Step 3',
            'output_file': 'Step3_questions_parsed.json'
        }
    ]

    all_questions_count = 0

    for step in steps:
        question_file = input_dir / step['question_file']
        answer_file = input_dir / step['answer_file']
        output_file = output_dir / step['output_file']

        if not question_file.exists():
            print(f"Warning: {question_file} not found, skipping...")
            continue

        if not answer_file.exists():
            print(f"Warning: {answer_file} not found, skipping...")
            continue

        # Parse the step
        questions = parse_step(question_file, answer_file, step['exam_name'])

        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        print(f"  Saved to {output_file}")
        all_questions_count += len(questions)
        print()

    print(f"Total questions parsed: {all_questions_count}")
    print("\nNote: Some questions may not be parsed due to:")
    print("  - Questions that reference images/diagrams for options (e.g., labeled anatomical sites)")
    print("  - Questions with table-based options")
    print("  - Formatting issues in the source text")
    print("\nDone!")


if __name__ == '__main__':
    main()
