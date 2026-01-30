#!/usr/bin/env python3
"""
Generate a report showing annotation counts for each case in the dataset.
Shows how many annotations each case has received, broken down by model.
"""

import json
import os
from collections import defaultdict

# Results directory
RESULTS_DIR = '../results/usmle_sample'

# Model mapping to identify which model each annotation belongs to
MODEL_PATTERNS = {
    'llama_small': ['llama-3.1-8b-instruct'],
    'llama': ['llama-3.3', '70b-instruct'],
    'llama_large': ['405b', 'llama-3.1-405b'],
    'llama_dpo': ['tulu-3-8b-dpo', '8b-dpo'],
    'llama_sft': ['tulu-3-8b-sft', '8b-sft'],
    'deepseek': ['deepseek'],
}


def identify_model(agent_model_str):
    """Identify which model an agent_model string belongs to"""
    if not agent_model_str:
        return 'unknown'

    agent_model_lower = agent_model_str.lower()

    for model_key, patterns in MODEL_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in agent_model_lower:
                return model_key

    return 'unknown'


def load_all_annotations():
    """Load all annotation files and organize by case_id and model"""
    if not os.path.exists(RESULTS_DIR):
        return {}

    # Structure: {case_id: {model_key: [annotation1, annotation2, ...]}}
    annotations_by_case_and_model = defaultdict(lambda: defaultdict(list))

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                annotation = json.load(f)
                case_id = annotation.get('case_id')
                agent_model = annotation.get('agent_model', '')

                if case_id:
                    model_key = identify_model(agent_model)
                    annotations_by_case_and_model[case_id][model_key].append(annotation)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return annotations_by_case_and_model


def generate_report():
    """Generate annotation count report"""
    print("\n" + "="*100)
    print("ANNOTATION COUNT REPORT - USMLE Sample Dataset")
    print("="*100)

    annotations_data = load_all_annotations()

    if not annotations_data:
        print("\nNo annotations found.")
        return

    # Get all case IDs sorted
    all_case_ids = sorted(annotations_data.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

    # Get all models that appear in the data
    all_models = set()
    for case_data in annotations_data.values():
        all_models.update(case_data.keys())
    all_models = sorted(all_models)

    print(f"\nTotal unique cases with annotations: {len(all_case_ids)}")
    print(f"Models found: {', '.join(all_models)}")
    print(f"\nGoal: 3 annotations per case per model")

    # Table header
    print("\n" + "="*100)
    header = f"{'Case ID':<20}"
    for model in all_models:
        header += f"{model:>15}"
    header += f"{'Total':>15}"
    print(header)
    print("-"*100)

    # Statistics
    total_annotations = 0
    cases_by_count = defaultdict(int)
    model_totals = defaultdict(int)

    # Print each case
    for case_id in all_case_ids:
        case_data = annotations_data[case_id]

        row = f"{case_id:<20}"
        case_total = 0

        for model in all_models:
            count = len(case_data.get(model, []))
            row += f"{count:>15}"
            case_total += count
            model_totals[model] += count

        row += f"{case_total:>15}"
        print(row)

        total_annotations += case_total
        cases_by_count[case_total] += 1

    # Summary footer
    print("-"*100)
    footer = f"{'TOTAL':<20}"
    for model in all_models:
        footer += f"{model_totals[model]:>15}"
    footer += f"{total_annotations:>15}"
    print(footer)
    print("="*100)

    # Statistics summary
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    print(f"\nTotal annotations across all cases: {total_annotations}")
    print(f"Average annotations per case: {total_annotations / len(all_case_ids):.2f}")

    print(f"\nAnnotation count distribution:")
    for count in sorted(cases_by_count.keys()):
        num_cases = cases_by_count[count]
        pct = (num_cases / len(all_case_ids) * 100) if len(all_case_ids) > 0 else 0
        print(f"  Cases with {count:2d} annotations: {num_cases:3d} ({pct:5.1f}%)")

    print(f"\nPer-model statistics:")
    for model in all_models:
        count = model_totals[model]
        avg = count / len(all_case_ids) if len(all_case_ids) > 0 else 0
        print(f"  {model:15s}: {count:4d} annotations (avg {avg:.2f} per case)")

    # Cases needing more annotations
    print("\n" + "="*100)
    print("CASES NEEDING MORE ANNOTATIONS")
    print("="*100)

    cases_0 = [cid for cid in all_case_ids if sum(len(annotations_data[cid].get(m, [])) for m in all_models) == 0]
    cases_1_2 = [cid for cid in all_case_ids if 1 <= sum(len(annotations_data[cid].get(m, [])) for m in all_models) <= 2]

    if cases_0:
        print(f"\n⚠️  Cases with 0 annotations ({len(cases_0)} cases):")
        for i in range(0, len(cases_0), 5):
            print(f"    {', '.join(cases_0[i:i+5])}")

    if cases_1_2:
        print(f"\n⚠️  Cases with 1-2 annotations ({len(cases_1_2)} cases):")
        for i in range(0, min(20, len(cases_1_2)), 5):
            cid = cases_1_2[i:i+5]
            counts = [sum(len(annotations_data[c].get(m, [])) for m in all_models) for c in cid]
            display = [f"{c} ({n})" for c, n in zip(cid, counts)]
            print(f"    {', '.join(display)}")
        if len(cases_1_2) > 20:
            print(f"    ... and {len(cases_1_2) - 20} more cases")

    print("\n" + "="*100)


def main():
    generate_report()


if __name__ == '__main__':
    main()
