#!/usr/bin/env python3
"""
Analyze annotation status by case type (bayesian_decision vs behavioral_decision)
for each model in the USMLE Sample dataset.

Reports annotation coverage for each combination of:
- bayesian_decision: accept/reject
- behavioral_decision: accept/reject
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# Directories
CASES_DIR = '../experiments/cases/usmle_sample'
RESULTS_DIR = '../results/usmle_sample'

# Model configurations
MODELS = [
    {'key': 'llama_small', 'name': 'Llama-3.1-8B-Instruct'},
    {'key': 'llama', 'name': 'Llama-3.3-70B-Instruct'},
    {'key': 'llama_large', 'name': 'Llama-3.1-405B-Instruct'},
    {'key': 'deepseek', 'name': 'DeepSeek-V3.1'},
]


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


def categorize_cases_by_decision_type(principal_data):
    """Categorize cases by their bayesian and behavioral decisions

    Returns a dict with keys like 'accept_accept', 'accept_reject', etc.
    Each value is a list of case_ids
    """
    categories = defaultdict(list)

    for case in principal_data.get('cases', []):
        case_id = case.get('case_id')
        bayesian = case.get('bayesian_decision', 'unknown')
        behavioral = case.get('behavioral_decision', 'unknown')

        category_key = f"{bayesian}_{behavioral}"
        categories[category_key].append(case_id)

    return categories


def load_all_annotations():
    """Load all annotation files and index by case_id"""
    if not os.path.exists(RESULTS_DIR):
        return {}

    annotations_by_case = defaultdict(list)

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(RESULTS_DIR, filename)
            with open(filepath, 'r') as f:
                annotation = json.load(f)
                case_id = annotation.get('case_id')
                agent_model = annotation.get('agent_model')
                if case_id:
                    annotations_by_case[case_id].append({
                        'agent_model': agent_model,
                        'annotator_id': annotation.get('annotator_id'),
                        'annotation': annotation
                    })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return annotations_by_case


def get_annotation_counts_by_model(annotations_by_case, case_ids, target_model_key):
    """Count annotations for cases in case_ids for a specific model

    Returns a dict mapping each annotation count (0, 1, 2, 3+) to number of cases
    """
    model_name_mapping = {
        'llama_small': 'allenai/Llama-3.1-Tulu-3-8B-DPO',
        'llama': 'meta-llama/Llama-3.3-70B-Instruct',
        'llama_large': 'meta-llama/Llama-3.1-405B-Instruct-Turbo',
        'deepseek': 'deepseek/deepseek-chat',
    }

    # Get the model name(s) that correspond to this model key
    target_model_names = []
    if target_model_key in model_name_mapping:
        target_model_names.append(model_name_mapping[target_model_key])

    # Also match by partial string for flexibility
    key_patterns = {
        'llama_small': ['tulu', '8b'],
        'llama': ['llama-3.3', '70b'],
        'llama_large': ['405b'],
        'deepseek': ['deepseek'],
    }

    counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 0 = not annotated, 1 = once, 2 = twice, 3 = 3+ times

    for case_id in case_ids:
        annotations = annotations_by_case.get(case_id, [])

        # Count annotations that match this model
        model_annotation_count = 0
        for ann_info in annotations:
            agent_model = ann_info['agent_model'] or ''

            # Check if this annotation is for our target model
            is_match = False

            # Exact match
            if agent_model in target_model_names:
                is_match = True

            # Pattern match
            patterns = key_patterns.get(target_model_key, [])
            for pattern in patterns:
                if pattern.lower() in agent_model.lower():
                    is_match = True
                    break

            if is_match:
                model_annotation_count += 1

        # Categorize by count
        if model_annotation_count >= 3:
            counts[3] += 1
        else:
            counts[model_annotation_count] += 1

    return counts


def analyze_model(model_key, model_name):
    """Analyze annotation status by case type for a specific model"""
    print(f"\n{'='*90}")
    print(f"MODEL: {model_name} ({model_key})")
    print(f"{'='*90}")

    # Load principal file
    principal_data = load_principal_file(model_key)
    if not principal_data:
        print(f"  ⚠️  No principal file found for {model_key}")
        return None

    # Categorize cases by decision type
    categories = categorize_cases_by_decision_type(principal_data)

    total_cases = sum(len(cases) for cases in categories.values())
    print(f"\nTotal manipulative cases: {total_cases}")

    # Load all annotations
    annotations_by_case = load_all_annotations()

    # Analyze each category
    category_display_names = {
        'accept_accept': 'Both Accept (Bayesian ✓, Behavioral ✓)',
        'accept_reject': 'Bayesian Accepts, Behavioral Rejects (Bayesian ✓, Behavioral ✗)',
        'reject_accept': 'Bayesian Rejects, Behavioral Accepts (Bayesian ✗, Behavioral ✓)',
        'reject_reject': 'Both Reject (Bayesian ✗, Behavioral ✗)',
    }

    results = {}

    for category_key in sorted(categories.keys()):
        case_ids = categories[category_key]
        display_name = category_display_names.get(category_key, category_key)

        print(f"\n{'-'*90}")
        print(f"Decision Type: {display_name}")
        print(f"{'-'*90}")
        print(f"  Number of cases: {len(case_ids)}")

        # Get annotation counts for this model
        counts = get_annotation_counts_by_model(annotations_by_case, case_ids, model_key)

        # Calculate statistics
        total = len(case_ids)
        if total == 0:
            continue

        print(f"\n  Annotation Coverage:")
        print(f"    ✗ Not annotated yet:          {counts[0]:3d} cases ({counts[0]/total*100:5.1f}%)")
        print(f"    ◐ Annotated once:             {counts[1]:3d} cases ({counts[1]/total*100:5.1f}%)")
        print(f"    ◑ Annotated twice:            {counts[2]:3d} cases ({counts[2]/total*100:5.1f}%)")
        print(f"    ✓ Goal reached (3+ annot.):   {counts[3]:3d} cases ({counts[3]/total*100:5.1f}%)")

        # Calculate progress
        target_annotations = total * 3
        current_annotations = 0
        for case_id in case_ids:
            annotations = annotations_by_case.get(case_id, [])
            # Count annotations for this model
            model_name_mapping = {
                'llama_small': 'allenai/Llama-3.1-Tulu-3-8B-DPO',
                'llama': 'meta-llama/Llama-3.3-70B-Instruct',
                'llama_large': 'meta-llama/Llama-3.1-405B-Instruct-Turbo',
                'deepseek': 'deepseek/deepseek-chat',
            }
            key_patterns = {
                'llama_small': ['tulu', '8b'],
                'llama': ['llama-3.3', '70b'],
                'llama_large': ['405b'],
                'deepseek': ['deepseek'],
            }
            for ann_info in annotations:
                agent_model = ann_info['agent_model'] or ''
                is_match = False
                if model_key in model_name_mapping and agent_model == model_name_mapping[model_key]:
                    is_match = True
                patterns = key_patterns.get(model_key, [])
                for pattern in patterns:
                    if pattern.lower() in agent_model.lower():
                        is_match = True
                        break
                if is_match:
                    current_annotations += 1

        progress_pct = (current_annotations / target_annotations * 100) if target_annotations > 0 else 0

        print(f"\n  Progress:")
        print(f"    Current annotations: {current_annotations}/{target_annotations}")
        print(f"    Overall progress: {progress_pct:.1f}%")
        print(f"    Annotations remaining: {target_annotations - current_annotations}")

        results[category_key] = {
            'total_cases': total,
            'counts': counts,
            'current_annotations': current_annotations,
            'target_annotations': target_annotations,
            'progress_pct': progress_pct
        }

    return results


def generate_summary_table(all_results):
    """Generate a summary table across all models and case types"""
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE - ANNOTATION STATUS BY CASE TYPE AND MODEL")
    print(f"{'='*90}")

    category_names = {
        'accept_accept': 'Both Accept',
        'accept_reject': 'Bayes Accept, Behav Reject',
        'reject_accept': 'Bayes Reject, Behav Accept',
        'reject_reject': 'Both Reject',
    }

    for category_key in ['accept_accept', 'accept_reject', 'reject_accept', 'reject_reject']:
        print(f"\n{'-'*90}")
        print(f"DECISION TYPE: {category_names.get(category_key, category_key)}")
        print(f"{'-'*90}")
        print(f"{'Model':<30} {'Cases':>6} {'Not Ann':>8} {'1 Ann':>8} {'2 Ann':>8} {'3+ Ann':>8} {'Progress':>10}")
        print(f"{'-'*90}")

        for model_info in MODELS:
            model_key = model_info['key']
            model_name = model_info['name']

            if model_key not in all_results or category_key not in all_results[model_key]:
                continue

            data = all_results[model_key][category_key]
            counts = data['counts']
            total = data['total_cases']
            progress = data['progress_pct']

            if total == 0:
                continue

            print(f"{model_name:<30} {total:>6} "
                  f"{counts[0]:>7} ({counts[0]/total*100:4.0f}%) "
                  f"{counts[1]:>7} ({counts[1]/total*100:4.0f}%) "
                  f"{counts[2]:>7} ({counts[2]/total*100:4.0f}%) "
                  f"{counts[3]:>7} ({counts[3]/total*100:4.0f}%) "
                  f"{progress:>9.1f}%")


def main():
    """Main analysis function"""
    print("\n" + "="*90)
    print("ANNOTATION STATUS BY CASE TYPE AND MODEL")
    print("USMLE Sample Dataset")
    print("="*90)
    print("\nCase Types:")
    print("  - Both Accept: Bayesian decision = accept AND Behavioral decision = accept")
    print("  - Bayesian Accept, Behavioral Reject: Bayesian = accept, Behavioral = reject")
    print("  - Bayesian Reject, Behavioral Accept: Bayesian = reject, Behavioral = accept")
    print("  - Both Reject: Bayesian decision = reject AND Behavioral decision = reject")

    all_results = {}

    for model_info in MODELS:
        model_key = model_info['key']
        model_name = model_info['name']
        results = analyze_model(model_key, model_name)
        if results:
            all_results[model_key] = results

    # Generate summary table
    generate_summary_table(all_results)

    print("\n" + "="*90)
    print("REPORT COMPLETE")
    print("="*90)
    print("\nGoal: 3 annotations per case")
    print("Legend: ✗ = 0 annotations, ◐ = 1 annotation, ◑ = 2 annotations, ✓ = 3+ annotations")
    print("\n")


if __name__ == '__main__':
    main()
