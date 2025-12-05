"""
Generate USMLE sample questions in the format used for agent and principal inference.

This script reads the parsed USMLE questions from datasets/usmle_sample and converts
them to the format used in experiments/input/clinical_questions_usmle.json.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def extract_metamap_phrases(text: str) -> List[str]:
    """
    Extract key phrases from the question text.
    This is a simple implementation that extracts noun phrases and key terms.
    For production use, you might want to use a proper NLP library like MetaMap.
    """
    # Remove special characters and split into phrases
    # Simple heuristic: split by punctuation and extract meaningful phrases
    phrases = []

    # Split by common punctuation
    sentences = re.split(r'[.?;]', text)

    for sentence in sentences:
        # Extract phrases separated by commas or other delimiters
        parts = re.split(r',|\(|\)', sentence)
        for part in parts:
            part = part.strip()
            if part and len(part.split()) <= 5:  # Keep shorter phrases
                phrases.append(part)
            elif part:
                # For longer parts, extract individual meaningful terms
                words = part.split()
                for i, word in enumerate(words):
                    # Skip common articles and short words
                    if word.lower() not in ['the', 'a', 'an', 'of', 'to', 'in', 'is', 'are', 'was', 'were']:
                        if len(word) > 2:
                            phrases.append(word)

    return phrases[:50]  # Limit to 50 phrases


def convert_question(question: Dict[str, Any], question_id: str) -> Dict[str, Any]:
    """
    Convert a question from the parsed format to the target format.

    Args:
        question: Question in the parsed format
        question_id: Unique ID for the question (e.g., "usmle_train_0")

    Returns:
        Question in the target format
    """
    # Convert options from array to dict
    options = {}
    for option in question['options']:
        options[option['label']] = option['text']

    # Get the answer text
    answer_idx = question['answer']

    # Skip questions with missing answers
    if answer_idx is None:
        return None

    try:
        answer_text = next(
            opt['text'] for opt in question['options']
            if opt['label'] == answer_idx
        )
    except StopIteration:
        # If answer not found in options, skip this question
        print(f"Warning: Answer '{answer_idx}' not found in options for question {question_id}")
        print(f"  Available options: {[opt['label'] for opt in question['options']]}")
        return None

    # Determine meta_info based on exam type
    exam = question['exam']
    if 'Step 1' in exam:
        meta_info = 'step1'
    elif 'Step 2' in exam:
        meta_info = 'step2'
    elif 'Step 3' in exam:
        meta_info = 'step3'
    else:
        meta_info = 'unknown'

    # Extract metamap phrases from the question stem
    metamap_phrases = extract_metamap_phrases(question['stem'])

    return {
        'id': question_id,
        'question': question['stem'],
        'options': options,
        'answer': answer_text,
        'answer_idx': answer_idx,
        'meta_info': meta_info,
        'metamap_phrases': metamap_phrases
    }


def load_parsed_questions(file_path: Path) -> List[Dict[str, Any]]:
    """Load questions from a parsed JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_usmle_sample(
    datasets_dir: str = '../datasets/usmle_sample',
    output_file: str = 'input/clinical_questions_usmle_sample.json',
    limit_per_step: int = None
) -> None:
    """
    Generate USMLE sample questions from the parsed datasets.

    Args:
        datasets_dir: Directory containing the parsed USMLE questions
        output_file: Output file path for the generated questions
        limit_per_step: Optional limit on number of questions per step (for testing)
    """
    datasets_path = Path(datasets_dir)

    # Define the files to process
    step_files = [
        'Step1_questions_parsed.json',
        'Step2_CK_questions_parsed.json',
        'Step3_questions_parsed.json'
    ]

    all_converted_questions = []
    question_counter = 0

    for step_file in step_files:
        file_path = datasets_path / step_file

        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping...")
            continue

        print(f"Processing {step_file}...")
        questions = load_parsed_questions(file_path)

        # Apply limit if specified
        if limit_per_step:
            questions = questions[:limit_per_step]

        # Convert each question
        converted_count = 0
        skipped_count = 0
        for question in questions:
            question_id = f"usmle_sample_{question_counter}"
            converted = convert_question(question, question_id)
            if converted is not None:
                all_converted_questions.append(converted)
                converted_count += 1
            else:
                skipped_count += 1
            question_counter += 1

        print(f"  Converted {converted_count} questions from {step_file} (skipped {skipped_count} with missing answers)")

    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_converted_questions, f, indent=2, ensure_ascii=False)

    print(f"\nTotal questions converted: {len(all_converted_questions)}")
    print(f"Output saved to: {output_path.absolute()}")

    # Sanity check: verify all answers match their options
    print("\nPerforming sanity check...")
    issues_found = 0
    for q in all_converted_questions:
        # Check if answer text exists in options
        if q['answer'] not in q['options'].values():
            print(f"  ERROR: Question {q['id']} - answer text '{q['answer'][:50]}...' not found in options")
            issues_found += 1
        # Check if answer_idx exists in options
        if q['answer_idx'] not in q['options']:
            print(f"  ERROR: Question {q['id']} - answer_idx '{q['answer_idx']}' not found in options keys")
            issues_found += 1
        # Check if answer matches the option at answer_idx
        elif q['options'][q['answer_idx']] != q['answer']:
            print(f"  ERROR: Question {q['id']} - answer mismatch:")
            print(f"    Expected: '{q['options'][q['answer_idx']]}'")
            print(f"    Got:      '{q['answer']}'")
            issues_found += 1

    if issues_found == 0:
        print("  ✓ Sanity check passed: All answers correctly match their options")
    else:
        print(f"  ✗ Found {issues_found} issue(s)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate USMLE sample questions for agent/principal inference'
    )
    parser.add_argument(
        '--datasets-dir',
        default='../datasets/usmle_sample',
        help='Directory containing parsed USMLE questions (default: ../datasets/usmle_sample)'
    )
    parser.add_argument(
        '--output',
        default='input/clinical_questions_usmle_sample.json',
        help='Output file path (default: input/clinical_questions_usmle_sample.json)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions per step (for testing)'
    )

    args = parser.parse_args()

    generate_usmle_sample(
        datasets_dir=args.datasets_dir,
        output_file=args.output,
        limit_per_step=args.limit
    )
