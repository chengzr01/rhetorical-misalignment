"""
Annotation utilities for saving and managing annotations.
"""

import json
import os
from datetime import datetime
import markdown as md
from data_loader import get_dataset_config


# Configure markdown converter
markdown_converter = md.Markdown(extensions=['extra', 'nl2br', 'sane_lists'])


def render_markdown(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    # Reset markdown converter for each use
    markdown_converter.reset()
    return markdown_converter.convert(text)


def save_annotation(annotation, dataset_key='usmle_sample'):
    """Save annotation to a file in dataset-specific directory"""
    dataset_config = get_dataset_config(dataset_key)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{annotation['case_id']}_{annotation['annotator_id']}_{timestamp}.json"
    filepath = os.path.join(dataset_config['annotation_dir'], filename)

    with open(filepath, 'w') as f:
        json.dump(annotation, f, indent=2)

    return filepath
