#!/usr/bin/env python3
"""
Collect detailed examples of helpful and harmful persuasion cases.
Shows highlights, reasoning, and AI recommendations for qualitative analysis.
"""

import json
import os
from collections import defaultdict


# Results directory
RESULTS_DIR = '../../results/usmle_sample'


def load_all_annotations():
    """Load all annotation files"""
    if not os.path.exists(RESULTS_DIR):
        return []

    annotations = []
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                annotation = json.load(f)
                annotations.append(annotation)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return annotations


def load_all_cache_files():
    """Load all agent cache files that contain questions and model analysis"""
    cache_dir = '../../experiments/cache/usmle_sample'
    cache_data = {}

    if not os.path.exists(cache_dir):
        print(f"Warning: Cache directory not found: {cache_dir}")
        return cache_data

    # Load each agent cache file
    agent_files = [
        'agent_llama-small.json',
        'agent_llama.json',
        'agent_llama-large.json',
        'agent_llama-dpo.json',
        'agent_llama-sft.json',
        'agent_deepseek.json',
        'agent_deepseek-llama.json'
    ]

    for filename in agent_files:
        filepath = os.path.join(cache_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # data is an array of cases
                    for case in data:
                        case_id = case.get('case_id')
                        agent_model = case.get('agent_model')
                        if case_id and agent_model:
                            # Index by (case_id, agent_model) to handle different models for same case
                            key = (case_id, agent_model)
                            cache_data[key] = {
                                'question': case.get('agent_context') or case.get('principal_context'),
                                'options': case.get('options', {}),
                                'model_analysis': case.get('information', ''),
                                'agent_model': agent_model,
                                'correct_answer': case.get('correct_answer'),
                                'correct_answer_idx': case.get('correct_answer_idx')
                            }
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return cache_data


def collect_persuasion_examples(annotations, cache_data):
    """Collect examples of AI persuasion with details

    Collects cases where AI persuaded humans to CHANGE their answer:
    1. correct_to_incorrect: Changed answer from correct to incorrect (harmful persuasion)
    2. incorrect_to_correct: Changed answer from incorrect to correct (helpful persuasion)

    Note: This function ensures ALL persuasion cases are collected by:
    - Iterating through every single annotation
    - Checking answer_changed flag from step 1 to step 2
    - Verifying correctness change matches answer change
    """

    correct_to_incorrect = []  # Harmful: answer changed from correct to incorrect
    incorrect_to_correct = []  # Helpful: answer changed from incorrect to correct
    no_answer_change = 0  # For verification
    stayed_correct = 0
    stayed_incorrect = 0
    wrong_to_wrong = 0  # Changed answer but both wrong

    for annotation in annotations:
        step1 = annotation.get('step1', {})
        step2 = annotation.get('step2', {})
        changes_1_2 = annotation.get('step1_to_step2_changes', {})

        step1_correct = step1.get('is_correct', False)
        step2_correct = step2.get('is_correct', False)
        answer_changed = changes_1_2.get('answer_changed', False)

        # Track all cases for verification
        if not answer_changed:
            no_answer_change += 1
            if step1_correct:
                stayed_correct += 1
            else:
                stayed_incorrect += 1
            continue

        # Calculate belief change
        belief1 = step1.get('answer_belief')
        belief2 = step2.get('answer_belief')
        belief_change = None
        if belief1 is not None and belief2 is not None:
            belief_change = belief2 - belief1

        # Look up source question and model analysis from cache
        case_id = annotation.get('case_id')
        agent_model = annotation.get('agent_model')
        cache_key = (case_id, agent_model)

        source_data = cache_data.get(cache_key, {})
        question = source_data.get('question', 'N/A')
        options = source_data.get('options', {})
        model_analysis = source_data.get('model_analysis', 'N/A')

        example = {
            'annotation': annotation,
            'case_id': case_id,
            'annotator_id': annotation.get('annotator_id'),
            'model': agent_model,
            'correct_answer': annotation.get('correct_answer'),
            'correct_answer_idx': annotation.get('correct_answer_idx'),
            'question': question,
            'options': options,
            'model_analysis': model_analysis,
            'step1_answer': step1.get('answer'),
            'step1_belief': step1.get('answer_belief'),
            'step1_correct': step1_correct,
            'step2_answer': step2.get('answer'),
            'step2_belief': step2.get('answer_belief'),
            'step2_correct': step2_correct,
            'answer_changed': answer_changed,
            'belief_change': belief_change,
            'highlights': annotation.get('highlights', []),
            'reasoning': annotation.get('reasoning', ''),
            'demographics': annotation.get('demographics', {})
        }

        # Categorize based on correctness trajectory
        if step1_correct and not step2_correct:
            # Harmful: AI persuaded from correct to incorrect answer
            correct_to_incorrect.append(example)
        elif not step1_correct and step2_correct:
            # Helpful: AI persuaded from incorrect to correct answer
            incorrect_to_correct.append(example)
        else:
            # Answer changed but correctness didn't change
            # This occurs when they changed from one wrong answer to another wrong answer
            # (or theoretically, one correct to another correct if multiple answers were correct)
            wrong_to_wrong += 1

    # Print verification summary
    total = len(annotations)
    persuaded = len(correct_to_incorrect) + len(incorrect_to_correct)
    answer_changed_total = persuaded + wrong_to_wrong
    print(f"\nCollection verification:")
    print(f"  Total annotations: {total}")
    print(f"  Answer changed: {answer_changed_total}")
    print(f"    - Harmful persuasion (C→I): {len(correct_to_incorrect)}")
    print(f"    - Helpful persuasion (I→C): {len(incorrect_to_correct)}")
    print(f"    - Wrong to wrong (W→W): {wrong_to_wrong}")
    print(f"  Answer unchanged: {no_answer_change}")
    print(f"    - Stayed correct: {stayed_correct}")
    print(f"    - Stayed incorrect: {stayed_incorrect}")

    sum_check = answer_changed_total + no_answer_change
    if sum_check == total:
        print(f"  Verification: {answer_changed_total} + {no_answer_change} = {total} ✓")
        print(f"  → ALL {total} annotations accounted for!")
    else:
        print(f"  ERROR: {sum_check} ≠ {total} - counts don't match!")

    return {
        'correct_to_incorrect': correct_to_incorrect,
        'incorrect_to_correct': incorrect_to_correct
    }


def format_example(example, index, persuasion_type):
    """Format a single example for output"""
    output = []

    output.append(f"\n{'='*80}")
    output.append(f"{persuasion_type} EXAMPLE #{index}")
    output.append(f"{'='*80}")

    # Basic information
    output.append(f"\nCase ID: {example['case_id']}")
    output.append(f"Annotator ID: {example['annotator_id']}")
    output.append(f"Model: {example['model']}")
    output.append(f"Correct Answer: {example['correct_answer']} (Option {example['correct_answer_idx']})")

    # Source question
    output.append(f"\n{'-'*80}")
    output.append(f"SOURCE QUESTION")
    output.append(f"{'-'*80}")
    output.append(f"\n{example.get('question', 'N/A')}")

    # Options
    if example.get('options'):
        output.append(f"\nOptions:")
        for key, value in sorted(example['options'].items()):
            marker = " ← CORRECT" if key == example['correct_answer_idx'] else ""
            output.append(f"  {key}. {value}{marker}")

    # Annotator demographics
    demo = example['demographics']
    if any(demo.values()):
        output.append(f"\nAnnotator Demographics:")
        if demo.get('age'):
            output.append(f"  Age: {demo['age']}")
        if demo.get('sex'):
            output.append(f"  Gender: {demo['sex']}")
        if demo.get('race'):
            output.append(f"  Race: {demo['race']}")
        if demo.get('expertise'):
            output.append(f"  Expertise: {demo['expertise']}")
        if demo.get('years_of_practice'):
            output.append(f"  Years of Practice: {demo['years_of_practice']}")
        if demo.get('practice_location'):
            output.append(f"  Location: {demo['practice_location']}")

    # AI Model's Analysis
    output.append(f"\n{'-'*80}")
    output.append(f"AI MODEL'S ANALYSIS")
    output.append(f"{'-'*80}")
    model_analysis = example.get('model_analysis', 'N/A')
    if len(model_analysis) > 3000:
        # Truncate very long analyses but show beginning and end
        output.append(f"\n{model_analysis[:2000]}")
        output.append(f"\n... [truncated {len(model_analysis) - 3000} characters] ...")
        output.append(f"\n{model_analysis[-1000:]}")
    else:
        output.append(f"\n{model_analysis}")

    # Answer progression
    output.append(f"\n{'-'*80}")
    output.append(f"ANSWER PROGRESSION")
    output.append(f"{'-'*80}")

    step1_marker = "✓ CORRECT" if example['step1_answer'] == example['correct_answer_idx'] else "✗ INCORRECT"
    step2_marker = "✓ CORRECT" if example['step2_answer'] == example['correct_answer_idx'] else "✗ INCORRECT"

    output.append(f"\nStep 1 (Initial):")
    output.append(f"  Answer: {example['step1_answer']} {step1_marker}")
    output.append(f"  Confidence: {example['step1_belief']:.2f}" if example['step1_belief'] is not None else "  Confidence: N/A")

    output.append(f"\nStep 2 (After AI):")
    output.append(f"  Answer: {example['step2_answer']} {step2_marker}")
    output.append(f"  Confidence: {example['step2_belief']:.2f}" if example['step2_belief'] is not None else "  Confidence: N/A")

    answer_changed_text = "Yes" if example.get('answer_changed', False) else "No"
    output.append(f"  Answer Changed: {answer_changed_text}")

    if example['belief_change'] is not None:
        change_direction = "increased" if example['belief_change'] > 0 else "decreased" if example['belief_change'] < 0 else "unchanged"
        output.append(f"  Belief Change: {example['belief_change']:+.3f} ({change_direction})")

    # Highlights
    if example['highlights']:
        output.append(f"\n{'-'*80}")
        output.append(f"HIGHLIGHTS (What annotator found notable)")
        output.append(f"{'-'*80}")

        step2_highlights = [h for h in example['highlights'] if h.get('step') == 'step2']

        if step2_highlights:
            output.append(f"\nAnnotator highlighted {len(step2_highlights)} passage(s) from AI response:")
            for i, highlight in enumerate(step2_highlights, 1):
                output.append(f"\n  Highlight #{i}:")
                output.append(f"    Context: {highlight.get('context', 'N/A')}")
                highlighted_text = highlight.get('text', 'N/A')
                # Wrap long text
                if len(highlighted_text) > 70:
                    words = highlighted_text.split()
                    lines = []
                    current_line = "    Text: \""
                    for word in words:
                        if len(current_line) + len(word) + 1 > 80:
                            lines.append(current_line)
                            current_line = "           " + word
                        else:
                            current_line += (" " if current_line.endswith('"') else " ") + word
                    lines.append(current_line + "\"")
                    output.extend(lines)
                else:
                    output.append(f"    Text: \"{highlighted_text}\"")
        else:
            output.append(f"\nNo highlights in Step 2 (AI response)")
    else:
        output.append(f"\n{'-'*80}")
        output.append(f"HIGHLIGHTS")
        output.append(f"{'-'*80}")
        output.append(f"\nNo highlights made by annotator")

    # Reasoning
    if example['reasoning']:
        output.append(f"\n{'-'*80}")
        output.append(f"ANNOTATOR'S REASONING/COMMENTS")
        output.append(f"{'-'*80}")
        output.append(f"\n{example['reasoning']}")
    else:
        output.append(f"\n{'-'*80}")
        output.append(f"ANNOTATOR'S REASONING/COMMENTS")
        output.append(f"{'-'*80}")
        output.append(f"\nNo reasoning provided by annotator")

    return "\n".join(output)


def generate_report(cases_dict):
    """Generate a comprehensive report"""

    report = []

    report.append("="*80)
    report.append("AI PERSUASION EXAMPLES REPORT - USMLE SAMPLE")
    report.append("="*80)
    report.append("")
    report.append("This report contains detailed examples of AI persuading humans to change")
    report.append("their answers from correct to incorrect (harmful) or incorrect to correct (helpful).")
    report.append("")

    # Extract case lists
    correct_to_incorrect = cases_dict['correct_to_incorrect']
    incorrect_to_correct = cases_dict['incorrect_to_correct']

    # Summary statistics
    report.append("="*80)
    report.append("SUMMARY")
    report.append("="*80)
    report.append(f"\nTotal cases where AI persuaded humans to change their answer: {len(correct_to_incorrect) + len(incorrect_to_correct)}")
    report.append("")
    report.append(f"Breakdown by outcome:")
    report.append(f"  1. Correct → Incorrect (harmful persuasion):   {len(correct_to_incorrect):3d} cases")
    report.append(f"  2. Incorrect → Correct (helpful persuasion):   {len(incorrect_to_correct):3d} cases")
    report.append("")
    report.append(f"Net helpful persuasions: {len(incorrect_to_correct) - len(correct_to_incorrect):+d}")

    # Model breakdown
    def count_by_model(cases):
        by_model = defaultdict(int)
        for case in cases:
            model_name = case['model'].split('/')[-1] if case['model'] and '/' in case['model'] else (case['model'] or 'unknown')
            by_model[model_name] += 1
        return by_model

    report.append(f"\nBreakdown by Model:")
    all_models = set()
    c2i_models = count_by_model(correct_to_incorrect)
    i2c_models = count_by_model(incorrect_to_correct)

    all_models.update(c2i_models.keys())
    all_models.update(i2c_models.keys())

    report.append(f"  {'Model':<50} {'Harmful (C→I)':>15} {'Helpful (I→C)':>15} {'Net':>6}")
    report.append(f"  {'-'*90}")
    for model in sorted(all_models):
        c2i = c2i_models.get(model, 0)
        i2c = i2c_models.get(model, 0)
        net = i2c - c2i
        report.append(f"  {model:<50} {c2i:>15} {i2c:>15} {net:>+6}")
    report.append(f"\n  Legend: C→I = Correct to Incorrect, I→C = Incorrect to Correct")

    # Highlights and reasoning statistics
    def count_metadata(cases):
        with_highlights = sum(1 for c in cases if c['highlights'])
        with_reasoning = sum(1 for c in cases if c['reasoning'])
        return with_highlights, with_reasoning

    report.append(f"\nAnnotator Feedback:")
    for category_name, cases in [
        ("Harmful (Correct → Incorrect)", correct_to_incorrect),
        ("Helpful (Incorrect → Correct)", incorrect_to_correct)
    ]:
        h, r = count_metadata(cases)
        report.append(f"  {category_name:<30}: {h}/{len(cases)} with highlights, {r}/{len(cases)} with reasoning")

    # Section 1: Harmful (Correct → Incorrect)
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 1: HARMFUL PERSUASION (CORRECT → INCORRECT)")
    report.append("="*80)
    report.append("\nCases where AI persuaded annotators to change from")
    report.append("a CORRECT answer to an INCORRECT answer.")
    report.append("")

    if correct_to_incorrect:
        sorted_cases = sorted(correct_to_incorrect, key=lambda x: abs(x['belief_change'] or 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            report.append(format_example(case, i, "HARMFUL PERSUASION: Correct → Incorrect"))
    else:
        report.append("\nNo harmful persuasion cases found.")

    # Section 2: Helpful (Incorrect → Correct)
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 2: HELPFUL PERSUASION (INCORRECT → CORRECT)")
    report.append("="*80)
    report.append("\nCases where AI persuaded annotators to change from")
    report.append("an INCORRECT answer to a CORRECT answer.")
    report.append("")

    if incorrect_to_correct:
        sorted_cases = sorted(incorrect_to_correct, key=lambda x: abs(x['belief_change'] or 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            report.append(format_example(case, i, "HELPFUL PERSUASION: Incorrect → Correct"))
    else:
        report.append("\nNo helpful persuasion cases found.")

    return "\n".join(report)


def generate_json_export(cases_dict):
    """Generate organized JSON export of all persuasion cases for further analysis"""

    def format_case_for_export(case):
        """Format a single case for JSON export"""
        return {
            'case_id': case['case_id'],
            'annotator_id': case['annotator_id'],
            'model': case['model'],
            'correct_answer': case['correct_answer'],
            'correct_answer_idx': case['correct_answer_idx'],
            'question': case.get('question', 'N/A'),
            'options': case.get('options', {}),
            'model_analysis': case.get('model_analysis', 'N/A'),
            'step1_answer': case['step1_answer'],
            'step1_belief': case['step1_belief'],
            'step1_correct': case['step1_correct'],
            'step2_answer': case['step2_answer'],
            'step2_belief': case['step2_belief'],
            'step2_correct': case['step2_correct'],
            'answer_changed': case['answer_changed'],
            'belief_change': case['belief_change'],
            'num_highlights': len(case['highlights']),
            'highlights': case['highlights'],
            'has_reasoning': bool(case['reasoning']),
            'reasoning': case['reasoning'],
            'demographics': case['demographics']
        }

    # Extract case lists
    correct_to_incorrect = cases_dict['correct_to_incorrect']
    incorrect_to_correct = cases_dict['incorrect_to_correct']

    output = {
        'metadata': {
            'description': 'AI persuasion cases where humans changed their answer',
            'dataset': 'USMLE Sample',
            'total_cases': len(correct_to_incorrect) + len(incorrect_to_correct)
        },
        'summary': {
            'harmful_persuasion': {
                'count': len(correct_to_incorrect),
                'description': 'Cases where AI persuaded from correct to incorrect answer'
            },
            'helpful_persuasion': {
                'count': len(incorrect_to_correct),
                'description': 'Cases where AI persuaded from incorrect to correct answer'
            },
            'net_helpful_persuasions': len(incorrect_to_correct) - len(correct_to_incorrect)
        },
        'cases': {
            'harmful_persuasion': [format_case_for_export(case) for case in correct_to_incorrect],
            'helpful_persuasion': [format_case_for_export(case) for case in incorrect_to_correct]
        }
    }

    return output


def main():
    """Main function"""
    print("\n" + "="*80)
    print("COLLECTING AI PERSUASION EXAMPLES")
    print("="*80)

    print("\nStep 1: Loading source questions and model analyses from cache...")
    cache_data = load_all_cache_files()
    print(f"  Loaded {len(cache_data)} case-model combinations from cache")

    print("\nStep 2: Loading annotation results...")
    annotations = load_all_annotations()
    if not annotations:
        print("\nNo annotations found in", RESULTS_DIR)
        return

    print(f"  Loaded {len(annotations)} annotations")

    print("\nStep 3: Analyzing persuasion cases where humans changed their answer...")
    cases_dict = collect_persuasion_examples(annotations, cache_data)

    # Print summary
    harmful_count = len(cases_dict['correct_to_incorrect'])
    helpful_count = len(cases_dict['incorrect_to_correct'])
    total = harmful_count + helpful_count

    print(f"\n{'='*80}")
    print(f"SUMMARY OF PERSUASION CASES")
    print(f"{'='*80}")
    print(f"  - Harmful (Correct → Incorrect):  {harmful_count:3d} cases")
    print(f"  - Helpful (Incorrect → Correct):  {helpful_count:3d} cases")
    print(f"  - Total persuasion cases:         {total:3d} cases")
    print(f"  - Net helpful:                    {helpful_count - harmful_count:+4d} cases")

    # Generate text report
    print(f"\n{'='*80}")
    print("Step 4: Generating detailed report with questions and model analyses...")
    report = generate_report(cases_dict)

    output_file = 'persuasion_examples.txt'
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"  ✓ Text report saved to: {output_file}")

    # Generate JSON export
    print("\nStep 5: Generating JSON export...")
    json_data = generate_json_export(cases_dict)

    json_file = 'persuasion_examples.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"  ✓ JSON export saved to: {json_file}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_file} (Detailed human-readable report)")
    print(f"  - {json_file} (Structured data for analysis)")
    print(f"\nEach case now includes:")
    print(f"  ✓ Source clinical question")
    print(f"  ✓ Answer options")
    print(f"  ✓ AI model's detailed analysis/reasoning")
    print(f"  ✓ Annotator's answer progression (Step 1 → Step 2)")
    print(f"  ✓ Annotator's highlights and comments")
    print(f"  ✓ Demographics (when available)")
    print("")


if __name__ == '__main__':
    main()
