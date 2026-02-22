"""
Configuration constants for the annotation interface.
"""

# Dataset configuration
DATASETS = {
    'usmle_sample': {
        'name': 'USMLE Sample',
        'data_dir': '../experiments/agents/usmle_sample',
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
    {'key': 'llama_dpo', 'file': 'agent_llama-dpo.json', 'name': 'Llama-3.1-Tulu-3-8B-DPO'},
    {'key': 'llama_sft', 'file': 'agent_llama-sft.json', 'name': 'Llama-3.1-Tulu-3-8B-SFT'},
]

# Default number of cases to assign to each annotator
DEFAULT_NUM_CASES_PER_ANNOTATOR = 5
