"""
Verification script to check the quality of parsed USMLE questions.

This script performs comprehensive checks on the parsed data to ensure:
1. All answers match their corresponding options
2. No duplicate question numbers within each step
3. All required fields are present
4. Data consistency between parsed and generated formats
"""

import json
from pathlib import Path
from collections import Counter


def verify_parsed_file(filepath: Path, exam_name: str, total_expected: int):
    """Verify a parsed JSON file."""
    print(f"\n{'='*60}")
    print(f"Verifying {exam_name}")
    print(f"{'='*60}")

    with open(filepath, 'r') as f:
        questions = json.load(f)

    print(f"Total questions: {len(questions)}/{total_expected} ({100*len(questions)/total_expected:.1f}%)")

    # Check for duplicates
    q_numbers = [q['question_number'] for q in questions]
    duplicates = [num for num, count in Counter(q_numbers).items() if count > 1]
    if duplicates:
        print(f"  ✗ Found duplicate question numbers: {duplicates}")
    else:
        print(f"  ✓ No duplicate question numbers")

    # Check required fields
    required_fields = ['exam', 'question_number', 'stem', 'options', 'answer']
    missing_fields = []
    for q in questions:
        for field in required_fields:
            if field not in q:
                missing_fields.append((q['question_number'], field))

    if missing_fields:
        print(f"  ✗ Missing fields: {missing_fields[:5]}...")
    else:
        print(f"  ✓ All questions have required fields")

    # Check answer validity
    invalid_answers = []
    empty_options = []

    for q in questions:
        q_num = q['question_number']
        answer = q.get('answer')
        options = q.get('options', [])
        option_labels = [opt['label'] for opt in options]

        # Check if answer exists in options
        if answer and answer not in option_labels:
            invalid_answers.append((q_num, answer, option_labels))

        # Check for empty option text
        for opt in options:
            if not opt.get('text', '').strip():
                empty_options.append((q_num, opt['label']))

    if invalid_answers:
        print(f"  ✗ Invalid answers: {len(invalid_answers)} questions")
        for q_num, ans, opts in invalid_answers[:3]:
            print(f"    Q{q_num}: answer '{ans}' not in {opts}")
    else:
        print(f"  ✓ All answers match available options")

    if empty_options:
        print(f"  ✗ Empty option text: {len(empty_options)} options")
        for q_num, label in empty_options[:3]:
            print(f"    Q{q_num}: option ({label})")
    else:
        print(f"  ✓ All options have text")

    # Check stem length
    short_stems = [(q['question_number'], len(q['stem'])) for q in questions if len(q['stem']) < 50]
    if short_stems:
        print(f"  ⚠ {len(short_stems)} questions have very short stems (< 50 chars)")

    return len(questions), len(invalid_answers) == 0 and len(empty_options) == 0


def verify_generated_file(filepath: Path):
    """Verify the generated JSON file."""
    print(f"\n{'='*60}")
    print(f"Verifying Generated Dataset")
    print(f"{'='*60}")

    with open(filepath, 'r') as f:
        questions = json.load(f)

    print(f"Total questions: {len(questions)}")

    # Check for duplicates
    q_ids = [q['id'] for q in questions]
    duplicates = [qid for qid, count in Counter(q_ids).items() if count > 1]
    if duplicates:
        print(f"  ✗ Found duplicate IDs: {duplicates}")
    else:
        print(f"  ✓ No duplicate IDs")

    # Check required fields
    required_fields = ['id', 'question', 'options', 'answer', 'answer_idx', 'meta_info']
    missing_fields = []
    for q in questions:
        for field in required_fields:
            if field not in q:
                missing_fields.append((q['id'], field))

    if missing_fields:
        print(f"  ✗ Missing fields: {missing_fields[:5]}...")
    else:
        print(f"  ✓ All questions have required fields")

    # Check answer consistency
    inconsistent_answers = []

    for q in questions:
        qid = q['id']
        answer_idx = q.get('answer_idx')
        answer_text = q.get('answer')
        options = q.get('options', {})

        # Check if answer_idx exists in options
        if answer_idx not in options:
            inconsistent_answers.append((qid, f"answer_idx '{answer_idx}' not in options"))
            continue

        # Check if answer text matches option at answer_idx
        if options[answer_idx] != answer_text:
            inconsistent_answers.append((qid, f"mismatch: '{options[answer_idx][:30]}...' != '{answer_text[:30]}...'"))

        # Check if answer text exists in options values
        if answer_text not in options.values():
            inconsistent_answers.append((qid, f"answer text not in options"))

    if inconsistent_answers:
        print(f"  ✗ Inconsistent answers: {len(inconsistent_answers)} questions")
        for qid, issue in inconsistent_answers[:3]:
            print(f"    {qid}: {issue}")
    else:
        print(f"  ✓ All answers are consistent")

    # Check meta_info distribution
    meta_counts = Counter(q['meta_info'] for q in questions)
    print(f"\n  Meta_info distribution:")
    for meta, count in sorted(meta_counts.items()):
        print(f"    {meta}: {count}")

    return len(questions), len(inconsistent_answers) == 0


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("USMLE Question Parsing Verification")
    print("="*60)

    base_path = Path('../datasets/usmle_sample')

    # Verify parsed files
    parsed_files = [
        (base_path / 'Step1_questions_parsed.json', 'USMLE Step 1', 119),
        (base_path / 'Step2_CK_questions_parsed.json', 'USMLE Step 2 CK', 120),
        (base_path / 'Step3_questions_parsed.json', 'USMLE Step 3', 137),
    ]

    total_parsed = 0
    all_passed = True

    for filepath, exam_name, total_expected in parsed_files:
        if filepath.exists():
            count, passed = verify_parsed_file(filepath, exam_name, total_expected)
            total_parsed += count
            all_passed = all_passed and passed
        else:
            print(f"\n✗ File not found: {filepath}")
            all_passed = False

    # Verify generated file
    generated_file = Path('questions/clinical_questions_usmle_sample.json')
    if generated_file.exists():
        count, passed = verify_generated_file(generated_file)
        all_passed = all_passed and passed

        # Check if parsed count matches generated count
        if count == total_parsed:
            print(f"\n  ✓ Generated count ({count}) matches parsed count ({total_parsed})")
        else:
            print(f"\n  ✗ Count mismatch: generated={count}, parsed={total_parsed}")
            all_passed = False
    else:
        print(f"\n✗ Generated file not found: {generated_file}")
        all_passed = False

    # Final summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print(f"✓ Total questions: {total_parsed}")
        print("✓ All answers match their options")
        print("✓ All data is consistent")
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please review the errors above")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
