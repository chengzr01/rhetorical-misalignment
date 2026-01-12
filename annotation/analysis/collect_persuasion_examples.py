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
    """Collect examples of persuasion with details"""

    harmful_cases = []  # Correct → Incorrect
    helpful_cases = []  # Incorrect → Correct

    for annotation in annotations:
        step1 = annotation.get('step1', {})
        step2 = annotation.get('step2', {})
        changes_1_2 = annotation.get('step1_to_step2_changes', {})

        step1_correct = step1.get('is_correct', False)
        step2_correct = step2.get('is_correct', False)
        answer_changed = changes_1_2.get('answer_changed', False)

        if not answer_changed:
            continue

        # Calculate belief change
        belief1 = step1.get('answer_belief')
        belief2 = step2.get('answer_belief')
        belief_change = None
        if belief1 is not None and belief2 is not None:
            belief_change = belief2 - belief1

        example = {
            'annotation': annotation,
            'case_id': annotation.get('case_id'),
            'annotator_id': annotation.get('annotator_id'),
            'model': annotation.get('agent_model'),
            'correct_answer': annotation.get('correct_answer'),
            'correct_answer_idx': annotation.get('correct_answer_idx'),
            'step1_answer': step1.get('answer'),
            'step1_belief': step1.get('answer_belief'),
            'step2_answer': step2.get('answer'),
            'step2_belief': step2.get('answer_belief'),
            'belief_change': belief_change,
            'highlights': annotation.get('highlights', []),
            'reasoning': annotation.get('reasoning', ''),
            'demographics': annotation.get('demographics', {})
        }

        if step1_correct and not step2_correct:
            harmful_cases.append(example)
        elif not step1_correct and step2_correct:
            helpful_cases.append(example)

    return harmful_cases, helpful_cases


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


def generate_report(harmful_cases, helpful_cases):
    """Generate a comprehensive report"""

    report = []

    report.append("="*80)
    report.append("PERSUASION EXAMPLES REPORT - USMLE SAMPLE")
    report.append("="*80)
    report.append("")
    report.append("This report contains detailed examples of AI persuasion effects,")
    report.append("including annotator highlights and reasoning.")
    report.append("")

    # Summary statistics
    report.append("="*80)
    report.append("SUMMARY")
    report.append("="*80)
    report.append(f"\nTotal Harmful Persuasion Cases (Correct → Incorrect): {len(harmful_cases)}")
    report.append(f"Total Helpful Persuasion Cases (Incorrect → Correct): {len(helpful_cases)}")
    report.append(f"Total Persuasion Cases: {len(harmful_cases) + len(helpful_cases)}")

    # Model breakdown
    harmful_by_model = defaultdict(int)
    helpful_by_model = defaultdict(int)

    for case in harmful_cases:
        harmful_by_model[case['model']] += 1
    for case in helpful_cases:
        helpful_by_model[case['model']] += 1

    if harmful_by_model:
        report.append(f"\nHarmful Persuasion by Model:")
        for model, count in sorted(harmful_by_model.items()):
            model_name = model.split('/')[-1] if '/' in model else model
            report.append(f"  {model_name}: {count}")

    if helpful_by_model:
        report.append(f"\nHelpful Persuasion by Model:")
        for model, count in sorted(helpful_by_model.items()):
            model_name = model.split('/')[-1] if '/' in model else model
            report.append(f"  {model_name}: {count}")

    # Highlights statistics
    harmful_with_highlights = sum(1 for c in harmful_cases if c['highlights'])
    helpful_with_highlights = sum(1 for c in helpful_cases if c['highlights'])

    report.append(f"\nCases with Highlights:")
    report.append(f"  Harmful cases: {harmful_with_highlights}/{len(harmful_cases)}")
    report.append(f"  Helpful cases: {helpful_with_highlights}/{len(helpful_cases)}")

    # Reasoning statistics
    harmful_with_reasoning = sum(1 for c in harmful_cases if c['reasoning'])
    helpful_with_reasoning = sum(1 for c in helpful_cases if c['reasoning'])

    report.append(f"\nCases with Annotator Reasoning:")
    report.append(f"  Harmful cases: {harmful_with_reasoning}/{len(harmful_cases)}")
    report.append(f"  Helpful cases: {helpful_with_reasoning}/{len(helpful_cases)}")

    # Harmful cases
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 1: HARMFUL PERSUASION CASES")
    report.append("="*80)
    report.append("\nThese are cases where the AI persuaded annotators to change from")
    report.append("a CORRECT answer to an INCORRECT answer.")
    report.append("")

    if harmful_cases:
        # Sort by belief change (most confident first)
        harmful_sorted = sorted(harmful_cases,
                               key=lambda x: (x['step2_belief'] or 0),
                               reverse=True)

        for i, case in enumerate(harmful_sorted, 1):
            report.append(format_example(case, i, "HARMFUL PERSUASION"))
    else:
        report.append("\nNo harmful persuasion cases found.")

    # Helpful cases
    report.append("\n\n")
    report.append("="*80)
    report.append("SECTION 2: HELPFUL PERSUASION CASES")
    report.append("="*80)
    report.append("\nThese are cases where the AI persuaded annotators to change from")
    report.append("an INCORRECT answer to a CORRECT answer.")
    report.append("")

    if helpful_cases:
        # Sort by belief change (most confident first)
        helpful_sorted = sorted(helpful_cases,
                               key=lambda x: (x['step2_belief'] or 0),
                               reverse=True)

        for i, case in enumerate(helpful_sorted, 1):
            report.append(format_example(case, i, "HELPFUL PERSUASION"))
    else:
        report.append("\nNo helpful persuasion cases found.")

    return "\n".join(report)


def generate_json_export(harmful_cases, helpful_cases):
    """Generate JSON export of all cases for further analysis"""

    output = {
        'summary': {
            'total_harmful': len(harmful_cases),
            'total_helpful': len(helpful_cases),
            'total_persuasion': len(harmful_cases) + len(helpful_cases)
        },
        'harmful_cases': [],
        'helpful_cases': []
    }

    for case in harmful_cases:
        output['harmful_cases'].append({
            'case_id': case['case_id'],
            'annotator_id': case['annotator_id'],
            'model': case['model'],
            'correct_answer': case['correct_answer'],
            'correct_answer_idx': case['correct_answer_idx'],
            'step1_answer': case['step1_answer'],
            'step1_belief': case['step1_belief'],
            'step2_answer': case['step2_answer'],
            'step2_belief': case['step2_belief'],
            'belief_change': case['belief_change'],
            'highlights': case['highlights'],
            'reasoning': case['reasoning'],
            'demographics': case['demographics']
        })

    for case in helpful_cases:
        output['helpful_cases'].append({
            'case_id': case['case_id'],
            'annotator_id': case['annotator_id'],
            'model': case['model'],
            'correct_answer': case['correct_answer'],
            'correct_answer_idx': case['correct_answer_idx'],
            'step1_answer': case['step1_answer'],
            'step1_belief': case['step1_belief'],
            'step2_answer': case['step2_answer'],
            'step2_belief': case['step2_belief'],
            'belief_change': case['belief_change'],
            'highlights': case['highlights'],
            'reasoning': case['reasoning'],
            'demographics': case['demographics']
        })

    return output


def main():
    """Main function"""
    print("\n" + "="*80)
    print("COLLECTING PERSUASION EXAMPLES")
    print("="*80)

    annotations = load_all_annotations()
    if not annotations:
        print("\nNo annotations found in", RESULTS_DIR)
        return

    print(f"\nLoaded {len(annotations)} annotations")
    print("Analyzing persuasion cases...")

    harmful_cases, helpful_cases = collect_persuasion_examples(annotations)

    print(f"\nFound {len(harmful_cases)} harmful persuasion cases (correct → incorrect)")
    print(f"Found {len(helpful_cases)} helpful persuasion cases (incorrect → correct)")

    # Generate text report
    print("\nGenerating detailed report...")
    report = generate_report(harmful_cases, helpful_cases)

    output_file = 'persuasion_examples.txt'
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"✓ Text report saved to: {output_file}")

    # Generate JSON export
    print("Generating JSON export...")
    json_data = generate_json_export(harmful_cases, helpful_cases)

    json_file = 'persuasion_examples.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ JSON export saved to: {json_file}")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_file} (Detailed human-readable report)")
    print(f"  - {json_file} (Structured data for analysis)")
    print("")


if __name__ == '__main__':
    main()
