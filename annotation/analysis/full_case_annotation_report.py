#!/usr/bin/env python3
"""
Generate a complete report showing annotation counts for ALL cases (including unannotated ones).
Loads case lists from principal files and shows annotation status for each model.
"""

import json
import os
from collections import defaultdict

# Directories
CASES_DIR = '../../experiments/cases/usmle_sample'
RESULTS_DIR = '../results/usmle_sample'

# Model configurations
MODELS = [
    {'key': 'llama_small', 'name': 'Llama-3.1-8B', 'patterns': ['llama-3.1-8b-instruct']},
    {'key': 'llama', 'name': 'Llama-3.3-70B', 'patterns': ['llama-3.3', '70b-instruct']},
    {'key': 'llama_large', 'name': 'Llama-3.1-405B', 'patterns': ['405b', 'llama-3.1-405b']},
    {'key': 'llama_dpo', 'name': 'Llama-3.1-Tulu-3-8B-DPO', 'patterns': ['tulu-3-8b-dpo', '8b-dpo']},
    {'key': 'llama_sft', 'name': 'Llama-3.1-Tulu-3-8B-SFT', 'patterns': ['tulu-3-8b-sft', '8b-sft']},
    {'key': 'deepseek', 'name': 'DeepSeek-V3', 'patterns': ['deepseek']},
]


def identify_model(agent_model_str):
    """Identify which model an agent_model string belongs to"""
    if not agent_model_str:
        return None

    agent_model_lower = agent_model_str.lower()

    for model_info in MODELS:
        for pattern in model_info['patterns']:
            if pattern.lower() in agent_model_lower:
                return model_info['key']

    return None


def load_principal_file(model_key):
    """Load principal file for a given model"""
    filepath = os.path.join(CASES_DIR, f'principal_{model_key}.json')
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_all_case_ids_per_model():
    """Get the complete list of case IDs for each model from principal files"""
    case_ids_by_model = {}

    for model_info in MODELS:
        model_key = model_info['key']
        principal_data = load_principal_file(model_key)

        if principal_data and 'cases' in principal_data:
            case_ids = [case['case_id'] for case in principal_data['cases']]
            case_ids_by_model[model_key] = sorted(case_ids)

    return case_ids_by_model


def load_all_annotations():
    """Load all annotation files and organize by case_id and model"""
    if not os.path.exists(RESULTS_DIR):
        return {}

    # Structure: {case_id: {model_key: count}}
    annotations_by_case_and_model = defaultdict(lambda: defaultdict(int))

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
                    if model_key:
                        annotations_by_case_and_model[case_id][model_key] += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return annotations_by_case_and_model


def generate_report_by_model(model_key, model_name, case_ids, annotations_data):
    """Generate a report for a specific model"""
    print(f"\n{'='*100}")
    print(f"MODEL: {model_name} ({model_key})")
    print(f"{'='*100}")

    if not case_ids:
        print("  No principal file found for this model.")
        return

    print(f"\nTotal cases in dataset: {len(case_ids)}")

    # Count annotations for each case
    case_annotation_counts = {}
    for case_id in case_ids:
        count = annotations_data.get(case_id, {}).get(model_key, 0)
        case_annotation_counts[case_id] = count

    # Statistics
    count_distribution = defaultdict(int)
    for count in case_annotation_counts.values():
        if count >= 3:
            count_distribution['3+'] += 1
        else:
            count_distribution[count] += 1

    total_annotations = sum(case_annotation_counts.values())
    target_annotations = len(case_ids) * 3  # Goal: 3 per case

    print(f"\nAnnotation Status:")
    print(f"  Not annotated (0):     {count_distribution[0]:4d} cases ({count_distribution[0]/len(case_ids)*100:5.1f}%)")
    print(f"  Annotated once (1):    {count_distribution[1]:4d} cases ({count_distribution[1]/len(case_ids)*100:5.1f}%)")
    print(f"  Annotated twice (2):   {count_distribution[2]:4d} cases ({count_distribution[2]/len(case_ids)*100:5.1f}%)")
    print(f"  Goal reached (3+):     {count_distribution['3+']:4d} cases ({count_distribution.get('3+', 0)/len(case_ids)*100:5.1f}%)")

    print(f"\nProgress:")
    print(f"  Total annotations: {total_annotations}/{target_annotations} ({total_annotations/target_annotations*100:.1f}%)")
    print(f"  Average per case: {total_annotations/len(case_ids):.2f}")
    print(f"  Remaining to goal: {target_annotations - total_annotations}")

    # Show specific cases by annotation count
    cases_by_count = defaultdict(list)
    for case_id, count in case_annotation_counts.items():
        if count >= 3:
            cases_by_count[3].append(case_id)
        else:
            cases_by_count[count].append(case_id)

    # Cases with 0 annotations (priority)
    if cases_by_count[0]:
        print(f"\n🔴 Cases with 0 annotations ({len(cases_by_count[0])} cases):")
        for i in range(0, min(30, len(cases_by_count[0])), 5):
            print(f"    {', '.join(cases_by_count[0][i:i+5])}")
        if len(cases_by_count[0]) > 30:
            print(f"    ... and {len(cases_by_count[0]) - 30} more cases")

    # Cases with 1-2 annotations
    cases_need_more = cases_by_count[1] + cases_by_count[2]
    if cases_need_more:
        print(f"\n⚠️  Cases with 1-2 annotations ({len(cases_need_more)} cases):")
        for i in range(0, min(20, len(cases_need_more))):
            case_id = cases_need_more[i]
            count = case_annotation_counts[case_id]
            print(f"    {case_id} ({count} annotation{'s' if count > 1 else ''})", end='')
            if (i + 1) % 3 == 0 or i == len(cases_need_more) - 1:
                print()
        if len(cases_need_more) > 20:
            print(f"    ... and {len(cases_need_more) - 20} more cases")

    # Cases with 3+ annotations (completed)
    if cases_by_count[3]:
        print(f"\n✅ Cases with 3+ annotations ({len(cases_by_count[3])} cases):")
        for i in range(0, min(15, len(cases_by_count[3])), 5):
            print(f"    {', '.join(cases_by_count[3][i:i+5])}")


def generate_summary_table(case_ids_by_model, annotations_data):
    """Generate a summary comparison table across all models"""
    print(f"\n\n{'='*100}")
    print("SUMMARY - ANNOTATION STATUS ACROSS ALL MODELS")
    print(f"{'='*100}")

    print(f"\n{'Model':<25} {'Total Cases':>12} {'Not Ann':>10} {'1 Ann':>10} {'2 Ann':>10} {'3+ Ann':>10} {'Progress':>10}")
    print('-'*100)

    for model_info in MODELS:
        model_key = model_info['key']
        model_name = model_info['name']

        case_ids = case_ids_by_model.get(model_key, [])
        if not case_ids:
            continue

        # Count annotations for each case
        count_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        total_annotations = 0

        for case_id in case_ids:
            count = annotations_data.get(case_id, {}).get(model_key, 0)
            total_annotations += count

            if count >= 3:
                count_distribution[3] += 1
            else:
                count_distribution[count] += 1

        total_cases = len(case_ids)
        target_annotations = total_cases * 3
        progress_pct = (total_annotations / target_annotations * 100) if target_annotations > 0 else 0

        print(f"{model_name:<25} {total_cases:>12} "
              f"{count_distribution[0]:>9} ({count_distribution[0]/total_cases*100:4.0f}%) "
              f"{count_distribution[1]:>9} ({count_distribution[1]/total_cases*100:4.0f}%) "
              f"{count_distribution[2]:>9} ({count_distribution[2]/total_cases*100:4.0f}%) "
              f"{count_distribution[3]:>9} ({count_distribution[3]/total_cases*100:4.0f}%) "
              f"{progress_pct:>9.1f}%")

    print('='*100)


def main():
    """Main analysis function"""
    print("\n" + "="*100)
    print("COMPLETE ANNOTATION STATUS REPORT")
    print("USMLE Sample Dataset - All Cases")
    print("="*100)
    print("\nThis report shows annotation status for ALL cases (including unannotated ones)")
    print("Goal: 3 annotations per case per model")

    # Load case IDs from principal files
    case_ids_by_model = get_all_case_ids_per_model()

    # Load all annotations
    annotations_data = load_all_annotations()

    # Generate report for each model
    for model_info in MODELS:
        model_key = model_info['key']
        model_name = model_info['name']
        case_ids = case_ids_by_model.get(model_key, [])

        generate_report_by_model(model_key, model_name, case_ids, annotations_data)

    # Generate summary table
    generate_summary_table(case_ids_by_model, annotations_data)

    print("\n" + "="*100)
    print("END OF REPORT")
    print("="*100)
    print()


if __name__ == '__main__':
    main()
