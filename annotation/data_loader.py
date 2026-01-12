"""
Data loading utilities for the annotation interface.
"""

import json
import os
from config import DATASETS, AVAILABLE_MODELS


def get_dataset_config(dataset_key):
    """Get dataset configuration"""
    return DATASETS.get(dataset_key, DATASETS['mimic'])


def get_model_info(model_key):
    """Get model info by key"""
    for model in AVAILABLE_MODELS:
        if model['key'] == model_key:
            return model
    return AVAILABLE_MODELS[0]  # Default to first model


def get_available_models(dataset_key='mimic'):
    """Get list of available model files in order for given dataset"""
    dataset_config = get_dataset_config(dataset_key)
    available = []
    for model in AVAILABLE_MODELS:
        filepath = os.path.join(dataset_config['data_dir'], model['file'])
        model_copy = model.copy()
        model_copy['available'] = os.path.exists(filepath)
        available.append(model_copy)
    return available


def check_dataset_availability():
    """Check which datasets have available data"""
    dataset_availability = {}
    for dataset_key, dataset_config in DATASETS.items():
        # Check if data directory exists and has at least one model file
        data_dir = dataset_config['data_dir']
        has_data = False
        if os.path.exists(data_dir):
            for model in AVAILABLE_MODELS:
                filepath = os.path.join(data_dir, model['file'])
                if os.path.exists(filepath):
                    has_data = True
                    break
        dataset_availability[dataset_key] = has_data
    return dataset_availability


def load_data(model_key='llama_small', dataset_key='mimic'):
    """Load data for specified model and dataset"""
    dataset_config = get_dataset_config(dataset_key)
    model_info = get_model_info(model_key)
    filepath = os.path.join(dataset_config['data_dir'], model_info['file'])
    with open(filepath, 'r') as f:
        return json.load(f)


def get_case_file_model_name(model_key):
    """Map model key to the model name used in case files"""
    # Mapping for models where the key differs from case file naming
    model_mapping = {
        'llama_small': 'llama_small',
    }
    return model_mapping.get(model_key, model_key)


def load_manipulative_case_ids(model_key, dataset_key='mimic'):
    """Load manipulative case IDs from principal file"""
    dataset_config = get_dataset_config(dataset_key)
    # Map model key to case file model name
    case_model_name = get_case_file_model_name(model_key)
    analysis_file = f'principal_{case_model_name}.json'
    filepath = os.path.join(dataset_config['cases_dir'], analysis_file)

    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        case_ids = set()

        # New format: Extract case_ids from the cases array
        if 'cases' in data:
            cases = data.get('cases', [])
            for case in cases:
                case_ids.add(case['case_id'])
        else:
            print(f"Unknown format in {filepath}")
            return []

        return list(case_ids)
    except Exception as e:
        print(f"Error loading manipulative cases: {e}")
        return []


def get_indices_for_case_ids(data, case_ids):
    """Find indices in data array that match the given case_ids"""
    indices = []
    for idx, case in enumerate(data):
        if case.get('case_id') in case_ids:
            indices.append(idx)
    return sorted(indices)


def parse_indices(indices_string, total_cases):
    """Parse a string of indices into a list of integers

    Supports formats:
    - Comma-separated: "0,5,10,15"
    - Range: "0-20"
    - Mixed: "0,5,10-15,20"
    """
    indices = []

    if not indices_string or not indices_string.strip():
        return []

    parts = indices_string.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range format
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                indices.extend(range(start, end + 1))
            except ValueError:
                continue
        else:
            # Single index
            try:
                indices.append(int(part))
            except ValueError:
                continue

    # Filter to valid indices
    indices = [i for i in indices if 0 <= i < total_cases]
    # Remove duplicates and sort
    indices = sorted(set(indices))

    return indices


def get_random_indices(total_cases, count):
    """Get random case indices"""
    import random
    count = min(count, total_cases)
    indices = list(range(total_cases))
    random.shuffle(indices)
    return sorted(indices[:count])
