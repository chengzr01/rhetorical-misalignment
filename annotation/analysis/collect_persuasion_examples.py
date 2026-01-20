#!/usr/bin/env python3
"""
Collect detailed examples of helpful and harmful persuasion cases.
Shows highlights, reasoning, and AI recommendations for qualitative analysis.
"""

import json
import os
from collections import defaultdict


# Results directory
RESULTS_DIR = '../results/usmle_sample'


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


def load_case_details(case_id):
    """Load the original case details including AI response"""
    # Try to find the case file
    cases_dir = '../experiments/cases/usmle_sample'

    # Look for agent output files
    for model_name in ['llama-small', 'llama', 'llama-large', 'deepseek']:
        agent_file = os.path.join(cases_dir, f'agent_{model_name}.json')
        if os.path.exists(agent_file):
            try:
                with open(agent_file, 'r') as f:
                    data = json.load(f)
                    # Find the specific case
                    for case in data.get('cases', []):
                        if case.get('case_id') == case_id:
                            return case
            except:
                pass

    return None


def collect_persuasion_examples(annotations):
    """Collect examples of AI persuasion/influence with details

    Collects ALL cases where belief changed, categorized by correctness trajectory:
    1. correct_to_incorrect: Started correct, ended incorrect (harmful)
    2. incorrect_to_correct: Started incorrect, ended correct (helpful)
    3. stayed_correct_belief_changed: Stayed correct but belief changed
    4. stayed_incorrect_belief_changed: Stayed incorrect but belief changed
    """

    correct_to_incorrect = []  # Harmful: answer changed from correct to incorrect
    incorrect_to_correct = []  # Helpful: answer changed from incorrect to correct
    stayed_correct_belief_changed = []  # Correct answer maintained, belief influenced
    stayed_incorrect_belief_changed = []  # Incorrect answer maintained, belief influenced

    for annotation in annotations:
        step1 = annotation.get('step1', {})
        step2 = annotation.get('step2', {})
        changes_1_2 = annotation.get('step1_to_step2_changes', {})

        step1_correct = step1.get('is_correct', False)
        step2_correct = step2.get('is_correct', False)
        answer_changed = changes_1_2.get('answer_changed', False)

        # Calculate belief change
        belief1 = step1.get('answer_belief')
        belief2 = step2.get('answer_belief')
        belief_change = None
        if belief1 is not None and belief2 is not None:
            belief_change = belief2 - belief1

        # Skip if no belief change
        if belief_change is None or belief_change == 0:
            continue

        example = {
            'annotation': annotation,
            'case_id': annotation.get('case_id'),
            'annotator_id': annotation.get('annotator_id'),
            'model': annotation.get('agent_model'),
            'correct_answer': annotation.get('correct_answer'),
            'correct_answer_idx': annotation.get('correct_answer_idx'),
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
            # Harmful: persuaded from correct to incorrect answer
            correct_to_incorrect.append(example)
        elif not step1_correct and step2_correct:
            # Helpful: persuaded from incorrect to correct answer
            incorrect_to_correct.append(example)
        elif step1_correct and step2_correct:
            # Stayed correct but belief was influenced
            stayed_correct_belief_changed.append(example)
        elif not step1_correct and not step2_correct:
            # Stayed incorrect but belief was influenced
            stayed_incorrect_belief_changed.append(example)

    return {
        'correct_to_incorrect': correct_to_incorrect,
        'incorrect_to_correct': incorrect_to_correct,
        'stayed_correct_belief_changed': stayed_correct_belief_changed,
        'stayed_incorrect_belief_changed': stayed_incorrect_belief_changed
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
    report.append("AI BELIEF INFLUENCE REPORT - USMLE SAMPLE")
    report.append("="*80)
    report.append("")
    report.append("This report contains detailed examples of AI influence on human beliefs,")
    report.append("including cases where beliefs changed with or without answer changes.")
    report.append("")

    # Extract case lists
    correct_to_incorrect = cases_dict['correct_to_incorrect']
    incorrect_to_correct = cases_dict['incorrect_to_correct']
    stayed_correct = cases_dict['stayed_correct_belief_changed']
    stayed_incorrect = cases_dict['stayed_incorrect_belief_changed']

    # Summary statistics
    report.append("="*80)
    report.append("SUMMARY")
    report.append("="*80)
    report.append(f"\nTotal cases where AI influenced beliefs: {len(correct_to_incorrect) + len(incorrect_to_correct) + len(stayed_correct) + len(stayed_incorrect)}")
    report.append("")
    report.append(f"Breakdown by outcome:")
    report.append(f"  1. Correct → Incorrect (harmful):             {len(correct_to_incorrect):3d} cases")
    report.append(f"  2. Incorrect → Correct (helpful):             {len(incorrect_to_correct):3d} cases")
    report.append(f"  3. Stayed correct, belief changed:            {len(stayed_correct):3d} cases")
    report.append(f"  4. Stayed incorrect, belief changed:          {len(stayed_incorrect):3d} cases")
    report.append("")
    report.append(f"Net helpful answer changes: {len(incorrect_to_correct) - len(correct_to_incorrect):+d}")

    # Model breakdown for all categories
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
    sc_models = count_by_model(stayed_correct)
    si_models = count_by_model(stayed_incorrect)

    all_models.update(c2i_models.keys())
    all_models.update(i2c_models.keys())
    all_models.update(sc_models.keys())
    all_models.update(si_models.keys())

    report.append(f"  {'Model':<40} {'C→I':>6} {'I→C':>6} {'SC':>6} {'SI':>6} {'Total':>6}")
    report.append(f"  {'-'*76}")
    for model in sorted(all_models):
        c2i = c2i_models.get(model, 0)
        i2c = i2c_models.get(model, 0)
        sc = sc_models.get(model, 0)
        si = si_models.get(model, 0)
        total = c2i + i2c + sc + si
        report.append(f"  {model:<40} {c2i:>6} {i2c:>6} {sc:>6} {si:>6} {total:>6}")
    report.append(f"\n  Legend: C→I=Correct to Incorrect, I→C=Incorrect to Correct, SC=Stayed Correct, SI=Stayed Incorrect")

    # Highlights and reasoning statistics
    def count_metadata(cases):
        with_highlights = sum(1 for c in cases if c['highlights'])
        with_reasoning = sum(1 for c in cases if c['reasoning'])
        return with_highlights, with_reasoning

    report.append(f"\nCases with Highlights:")
    for category_name, cases in [
        ("Correct → Incorrect", correct_to_incorrect),
        ("Incorrect → Correct", incorrect_to_correct),
        ("Stayed Correct", stayed_correct),
        ("Stayed Incorrect", stayed_incorrect)
    ]:
        h, r = count_metadata(cases)
        report.append(f"  {category_name:<25}: {h}/{len(cases)} with highlights, {r}/{len(cases)} with reasoning")

    # Section 1: Harmful (Correct → Incorrect)
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 1: CORRECT → INCORRECT (HARMFUL)")
    report.append("="*80)
    report.append("\nCases where AI persuaded annotators to change from")
    report.append("a CORRECT answer to an INCORRECT answer.")
    report.append("")

    if correct_to_incorrect:
        sorted_cases = sorted(correct_to_incorrect, key=lambda x: abs(x['belief_change'] or 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            report.append(format_example(case, i, "HARMFUL: Correct → Incorrect"))
    else:
        report.append("\nNo cases found in this category.")

    # Section 2: Helpful (Incorrect → Correct)
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 2: INCORRECT → CORRECT (HELPFUL)")
    report.append("="*80)
    report.append("\nCases where AI persuaded annotators to change from")
    report.append("an INCORRECT answer to a CORRECT answer.")
    report.append("")

    if incorrect_to_correct:
        sorted_cases = sorted(incorrect_to_correct, key=lambda x: abs(x['belief_change'] or 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            report.append(format_example(case, i, "HELPFUL: Incorrect → Correct"))
    else:
        report.append("\nNo cases found in this category.")

    # Section 3: Stayed Correct (Belief Changed)
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 3: STAYED CORRECT (BELIEF INFLUENCED)")
    report.append("="*80)
    report.append("\nCases where annotator maintained CORRECT answer but")
    report.append("AI influenced their confidence/belief.")
    report.append("")

    if stayed_correct:
        sorted_cases = sorted(stayed_correct, key=lambda x: abs(x['belief_change'] or 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            report.append(format_example(case, i, "STAYED CORRECT: Belief Influenced"))
    else:
        report.append("\nNo cases found in this category.")

    # Section 4: Stayed Incorrect (Belief Changed)
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 4: STAYED INCORRECT (BELIEF INFLUENCED)")
    report.append("="*80)
    report.append("\nCases where annotator maintained INCORRECT answer but")
    report.append("AI influenced their confidence/belief.")
    report.append("")

    if stayed_incorrect:
        sorted_cases = sorted(stayed_incorrect, key=lambda x: abs(x['belief_change'] or 0), reverse=True)
        for i, case in enumerate(sorted_cases, 1):
            report.append(format_example(case, i, "STAYED INCORRECT: Belief Influenced"))
    else:
        report.append("\nNo cases found in this category.")

    return "\n".join(report)


def generate_json_export(cases_dict):
    """Generate organized JSON export of all cases for further analysis"""

    def format_case_for_export(case):
        """Format a single case for JSON export"""
        return {
            'case_id': case['case_id'],
            'annotator_id': case['annotator_id'],
            'model': case['model'],
            'correct_answer': case['correct_answer'],
            'correct_answer_idx': case['correct_answer_idx'],
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
    stayed_correct = cases_dict['stayed_correct_belief_changed']
    stayed_incorrect = cases_dict['stayed_incorrect_belief_changed']

    output = {
        'metadata': {
            'description': 'AI influence on human beliefs in medical diagnosis',
            'dataset': 'USMLE Sample',
            'total_cases': len(correct_to_incorrect) + len(incorrect_to_correct) + len(stayed_correct) + len(stayed_incorrect)
        },
        'summary': {
            'correct_to_incorrect': {
                'count': len(correct_to_incorrect),
                'description': 'Cases where AI persuaded from correct to incorrect answer'
            },
            'incorrect_to_correct': {
                'count': len(incorrect_to_correct),
                'description': 'Cases where AI persuaded from incorrect to correct answer'
            },
            'stayed_correct_belief_changed': {
                'count': len(stayed_correct),
                'description': 'Cases where correct answer was maintained but belief changed'
            },
            'stayed_incorrect_belief_changed': {
                'count': len(stayed_incorrect),
                'description': 'Cases where incorrect answer was maintained but belief changed'
            },
            'net_helpful_answer_changes': len(incorrect_to_correct) - len(correct_to_incorrect)
        },
        'cases': {
            'correct_to_incorrect': [format_case_for_export(case) for case in correct_to_incorrect],
            'incorrect_to_correct': [format_case_for_export(case) for case in incorrect_to_correct],
            'stayed_correct_belief_changed': [format_case_for_export(case) for case in stayed_correct],
            'stayed_incorrect_belief_changed': [format_case_for_export(case) for case in stayed_incorrect]
        }
    }

    return output


def main():
    """Main function"""
    print("\n" + "="*80)
    print("COLLECTING AI BELIEF INFLUENCE EXAMPLES")
    print("="*80)

    annotations = load_all_annotations()
    if not annotations:
        print("\nNo annotations found in", RESULTS_DIR)
        return

    print(f"\nLoaded {len(annotations)} annotations")
    print("Analyzing belief influence cases...")

    cases_dict = collect_persuasion_examples(annotations)

    # Print summary
    print(f"\nFound belief influence cases:")
    print(f"  - Correct → Incorrect:           {len(cases_dict['correct_to_incorrect']):3d} cases (harmful)")
    print(f"  - Incorrect → Correct:           {len(cases_dict['incorrect_to_correct']):3d} cases (helpful)")
    print(f"  - Stayed Correct (belief Δ):     {len(cases_dict['stayed_correct_belief_changed']):3d} cases")
    print(f"  - Stayed Incorrect (belief Δ):   {len(cases_dict['stayed_incorrect_belief_changed']):3d} cases")
    total = sum(len(v) for v in cases_dict.values())
    print(f"  - Total:                         {total:3d} cases")

    # Generate text report
    print("\nGenerating detailed report...")
    report = generate_report(cases_dict)

    output_file = 'persuasion_examples.txt'
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Text report saved to: {output_file}")

    # Generate JSON export
    print("Generating JSON export...")
    json_data = generate_json_export(cases_dict)

    json_file = 'persuasion_examples.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ JSON export saved to: {json_file}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_file} (Detailed human-readable report)")
    print(f"  - {json_file} (Structured data with organized categories)")
    print("")


if __name__ == '__main__':
    main()
