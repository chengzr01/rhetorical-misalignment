"""
Scheduling system for balanced annotation coverage.

This module handles:
1. Model-level balancing: Selecting which model needs more annotations
2. Case-level balancing: Within a model, prioritizing cases with fewer annotations
"""

import os
import json
import random
from collections import defaultdict
from data_loader import (
    get_dataset_config,
    get_available_models,
    load_manipulative_case_ids
)


def get_annotation_counts_per_case(dataset_key, model_key=None):
    """Count how many times each case has been annotated

    Args:
        dataset_key: The dataset to count annotations for
        model_key: Optional model key to filter by (e.g., 'llama_small')
                   If None, counts all annotations regardless of model

    Returns:
        dict: case_id -> count of annotations
    """
    dataset_config = get_dataset_config(dataset_key)
    annotation_dir = dataset_config['annotation_dir']

    annotation_counts = {}

    if not os.path.exists(annotation_dir):
        return annotation_counts

    # Scan all annotation files
    for filename in os.listdir(annotation_dir):
        if not filename.endswith('.json'):
            continue

        try:
            filepath = os.path.join(annotation_dir, filename)
            with open(filepath, 'r') as f:
                annotation = json.load(f)

            # Filter by model_key if specified
            if model_key is not None:
                annotation_model = annotation.get('model_key')
                if annotation_model != model_key:
                    continue

            case_id = annotation.get('case_id')
            if case_id:
                annotation_counts[case_id] = annotation_counts.get(case_id, 0) + 1
        except Exception as e:
            print(f"Error reading annotation file {filename}: {e}")
            continue

    return annotation_counts


def get_smart_random_cases(case_ids, annotation_counts, num_cases=10):
    """Select cases intelligently based on annotation coverage

    Prioritizes cases with fewer annotations:
    1. First fill with cases that have 0 annotations
    2. Then cases with 1 annotation
    3. Then cases with 2 annotations
    4. Finally cases with 3+ annotations

    Within each group, randomly shuffle.

    Args:
        case_ids: List of all available case IDs
        annotation_counts: Dict of case_id -> annotation count
        num_cases: Number of cases to select (default 10)

    Returns:
        List of selected case_ids
    """
    # Group cases by annotation count
    grouped_cases = {
        0: [],  # Never annotated
        1: [],  # Annotated once
        2: [],  # Annotated twice
        3: []   # Annotated 3+ times
    }

    for case_id in case_ids:
        count = annotation_counts.get(case_id, 0)
        if count >= 3:
            grouped_cases[3].append(case_id)
        else:
            grouped_cases[count].append(case_id)

    # Shuffle each group
    for group in grouped_cases.values():
        random.shuffle(group)

    # Select cases prioritizing lower annotation counts
    selected = []
    for priority in [0, 1, 2, 3]:
        available = grouped_cases[priority]
        needed = num_cases - len(selected)

        if needed <= 0:
            break

        # Take as many as we need from this group
        selected.extend(available[:needed])

    # If we still don't have enough cases (shouldn't happen), pad with random cases
    if len(selected) < num_cases:
        remaining = [c for c in case_ids if c not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:num_cases - len(selected)])

    return selected


def select_model_with_fewest_annotations(dataset_key='mimic'):
    """Select the model with the fewest annotations for load balancing

    This function counts total annotations across all cases for each model
    and returns the model key with the lowest annotation count.

    Strategy:
    1. Count total annotations for each available model
    2. Find model(s) with minimum annotation count
    3. If multiple models have the same minimum, randomly select one

    Args:
        dataset_key: The dataset to check annotations for

    Returns:
        str: The model key with fewest annotations, or the first available model
    """
    # Get available models for this dataset
    available_models = get_available_models(dataset_key)
    available_model_keys = [m['key'] for m in available_models if m.get('available', False)]

    if not available_model_keys:
        # No models available, return default
        return 'llama_small'

    # Count total annotations per model
    model_annotation_counts = {}

    for model_key in available_model_keys:
        # Load manipulative cases for this model
        case_ids = load_manipulative_case_ids(model_key, dataset_key)

        if not case_ids:
            # If no manipulative cases, assign 0 annotations
            model_annotation_counts[model_key] = 0
            continue

        # Get annotation counts for this model
        annotation_counts = get_annotation_counts_per_case(dataset_key, model_key)

        # Sum up total annotations for this model
        total_annotations = sum(annotation_counts.get(case_id, 0) for case_id in case_ids)
        model_annotation_counts[model_key] = total_annotations

    # Find model(s) with minimum annotations
    if not model_annotation_counts:
        # Fallback to first available model
        return available_model_keys[0]

    min_annotations = min(model_annotation_counts.values())
    models_with_min = [model_key for model_key, count in model_annotation_counts.items()
                       if count == min_annotations]

    # If multiple models have the same minimum count, randomly select one
    selected_model = random.choice(models_with_min)

    print(f"Model selection for {dataset_key}:")
    for model_key, count in sorted(model_annotation_counts.items(), key=lambda x: x[1]):
        marker = "→ SELECTED" if model_key == selected_model else ""
        print(f"  {model_key}: {count} annotations {marker}")

    return selected_model


def get_coverage_statistics(dataset_key, model_key):
    """Get annotation coverage statistics for a dataset and model

    Args:
        dataset_key: The dataset to analyze
        model_key: The model to analyze

    Returns:
        dict: Coverage statistics including:
            - total_cases: Total number of manipulative cases
            - cases_by_count: Distribution of cases by annotation count
            - total_annotations: Current total annotations
            - target_annotations: Target annotations (3 per case)
            - progress_percent: Percentage of target reached
    """
    # Load manipulative cases
    case_ids = load_manipulative_case_ids(model_key, dataset_key)

    if not case_ids:
        return {
            'error': 'No manipulative cases found',
            'total_cases': 0,
            'cases_by_count': {0: 0, 1: 0, 2: 0, 3: 0},
            'total_annotations': 0,
            'target_annotations': 0,
            'progress_percent': 0
        }

    # Get annotation counts for this specific model
    annotation_counts = get_annotation_counts_per_case(dataset_key, model_key)

    # Categorize cases by annotation count
    cases_by_count = {0: 0, 1: 0, 2: 0, 3: 0}

    for case_id in case_ids:
        count = annotation_counts.get(case_id, 0)
        if count >= 3:
            cases_by_count[3] += 1
        else:
            cases_by_count[count] += 1

    # Calculate progress metrics
    total_annotations_needed = len(case_ids) * 3
    total_annotations_current = sum(annotation_counts.get(cid, 0) for cid in case_ids)

    return {
        'total_cases': len(case_ids),
        'cases_by_count': cases_by_count,
        'total_annotations': total_annotations_current,
        'target_annotations': total_annotations_needed,
        'progress_percent': (total_annotations_current / total_annotations_needed * 100) if total_annotations_needed > 0 else 0
    }
