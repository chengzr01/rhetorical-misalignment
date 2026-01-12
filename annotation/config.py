"""
Configuration constants for the annotation interface.
"""

# Dataset configuration
DATASETS = {
    'mimic': {
        'name': 'MIMIC-IV',
        'data_dir': '../experiments/cache/mimiciv_demo',
        'cases_dir': '../experiments/cases/mimiciv_demo',
        'annotation_dir': 'results/mimic'
    },
    'usmle': {
        'name': 'USMLE',
        'data_dir': '../experiments/cache/usmle',
        'cases_dir': '../experiments/cases/usmle',
        'annotation_dir': 'results/usmle'
    },
    'usmle_sample': {
        'name': 'USMLE Sample',
        'data_dir': '../experiments/cache/usmle_sample',
        'cases_dir': '../experiments/cases/usmle_sample',
        'annotation_dir': 'results/usmle_sample'
    }
}

# Available model files (in order as per README.md)
AVAILABLE_MODELS = [
    {'key': 'llama_small', 'file': 'agent_llama-small.json', 'name': 'Llama-3.1-8B-Instruct'},
    {'key': 'llama', 'file': 'agent_llama.json', 'name': 'Llama-3.3-70B-Instruct'},
    {'key': 'llama_large', 'file': 'agent_llama-large.json', 'name': 'Llama-3.1-405B-Instruct'},
    {'key': 'deepseek', 'file': 'agent_deepseek.json', 'name': 'DeepSeek-V3.1'},
]
