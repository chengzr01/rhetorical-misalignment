#!/usr/bin/env python3
"""
Analyze annotation coverage across all datasets.
Shows which cases need more annotations to reach the 3-annotation goal.
"""

import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASETS, AVAILABLE_MODELS
from data_loader import load_manipulative_case_ids
from scheduler import get_annotation_counts_per_case

MODELS = ['llama_small', 'llama', 'llama_large', 'deepseek']

def analyze_dataset_coverage(dataset_key, model_key):
    """Analyze annotation coverage for a specific dataset and model"""
    print(f"\n{'='*80}")
    print(f"Dataset: {DATASETS[dataset_key]['name']}, Model: {model_key}")
    print(f"{'='*80}")

    # Load manipulative cases
    case_ids = load_manipulative_case_ids(model_key, dataset_key)

    if not case_ids:
        print(f"  ⚠️  No manipulative cases found (principal file may not exist)")
        return

    print(f"  Total manipulative cases: {len(case_ids)}")

    # Get annotation counts for this specific model
    annotation_counts = get_annotation_counts_per_case(dataset_key, model_key)

    # Categorize cases by annotation count
    cases_by_count = {
        0: [],  # Never annotated
        1: [],  # Annotated once
        2: [],  # Annotated twice
        3: []   # Annotated 3+ times (goal reached)
    }

    for case_id in case_ids:
        count = annotation_counts.get(case_id, 0)
        if count >= 3:
            cases_by_count[3].append(case_id)
        else:
            cases_by_count[count].append(case_id)

    # Print statistics
    print(f"\n  Annotation Coverage:")
    print(f"    ✗ Not annotated yet:          {len(cases_by_count[0]):3d} cases ({len(cases_by_count[0])/len(case_ids)*100:.1f}%)")
    print(f"    ◐ Annotated once:             {len(cases_by_count[1]):3d} cases ({len(cases_by_count[1])/len(case_ids)*100:.1f}%)")
    print(f"    ◑ Annotated twice:            {len(cases_by_count[2]):3d} cases ({len(cases_by_count[2])/len(case_ids)*100:.1f}%)")
    print(f"    ✓ Goal reached (3+ annot.):   {len(cases_by_count[3]):3d} cases ({len(cases_by_count[3])/len(case_ids)*100:.1f}%)")

    # Calculate progress metrics
    total_annotations_needed = len(case_ids) * 3  # Goal: 3 annotations per case
    total_annotations_current = sum(annotation_counts.get(cid, 0) for cid in case_ids)

    print(f"\n  Progress:")
    print(f"    Current annotations: {total_annotations_current}/{total_annotations_needed}")
    print(f"    Overall progress: {total_annotations_current/total_annotations_needed*100:.1f}%")

    # Cases that need immediate attention (0 annotations)
    if cases_by_count[0]:
        print(f"\n  🔴 Priority: {len(cases_by_count[0])} cases need first annotation")
        if len(cases_by_count[0]) <= 10:
            print(f"     Case IDs: {', '.join(cases_by_count[0])}")

    # Cases that need more annotations
    cases_need_more = len(cases_by_count[0]) + len(cases_by_count[1]) + len(cases_by_count[2])
    if cases_need_more > 0:
        print(f"\n  📊 Next assignment will prioritize:")
        if cases_by_count[0]:
            print(f"     • {len(cases_by_count[0])} cases with 0 annotations (highest priority)")
        if cases_by_count[1]:
            print(f"     • {len(cases_by_count[1])} cases with 1 annotation")
        if cases_by_count[2]:
            print(f"     • {len(cases_by_count[2])} cases with 2 annotations")

    return {
        'total_cases': len(case_ids),
        'cases_by_count': {k: len(v) for k, v in cases_by_count.items()},
        'total_annotations': total_annotations_current,
        'target_annotations': total_annotations_needed,
        'progress_percent': total_annotations_current/total_annotations_needed*100 if total_annotations_needed > 0 else 0
    }

def main():
    print("\n" + "="*80)
    print("ANNOTATION COVERAGE ANALYSIS")
    print("="*80)
    print("\nGoal: Every manipulative case should be annotated by at least 3 annotators")
    print("Priority: First ensure all cases have at least 1 annotation, then work towards 3")

    all_stats = {}

    for dataset_key in DATASETS.keys():
        dataset_stats = {}
        for model_key in MODELS:
            stats = analyze_dataset_coverage(dataset_key, model_key)
            if stats:
                dataset_stats[model_key] = stats
        if dataset_stats:
            all_stats[dataset_key] = dataset_stats

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    total_cases = 0
    total_annotations = 0
    total_target = 0

    for dataset_key, dataset_stats in all_stats.items():
        for model_key, stats in dataset_stats.items():
            total_cases += stats['total_cases']
            total_annotations += stats['total_annotations']
            total_target += stats['target_annotations']

    if total_target > 0:
        print(f"\n  Total manipulative cases across all datasets/models: {total_cases}")
        print(f"  Total annotations collected: {total_annotations}")
        print(f"  Target annotations (3 per case): {total_target}")
        print(f"  Overall progress: {total_annotations/total_target*100:.1f}%")
        print(f"\n  Annotations remaining to reach goal: {total_target - total_annotations}")
    else:
        print("\n  No data found. Make sure the dataset and case files exist.")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
