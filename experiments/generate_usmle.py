#!/usr/bin/env python3
"""
Generate clinical questions dataset from USMLE MedQA dataset.
Loads the GBaker/MedQA-USMLE-4-options dataset and converts it to a structured format.
"""

import argparse
import json
import os
from typing import Dict, List, Optional
from datasets import load_dataset


def convert_record_to_entry(record: Dict, split_name: str, idx: int) -> Dict:
    """
    Convert a single USMLE record to structured format.

    Args:
        record: Raw record from the dataset
        split_name: Name of the split (train/test)
        idx: Index of the record in the split

    Returns:
        Dictionary with all original keys and values preserved
    """
    entry = {
        "id": f"usmle_{split_name}_{idx}",
        "question": record["question"],
        "options": record["options"],
        "answer": record["answer"],
        "answer_idx": record["answer_idx"],
        "meta_info": record["meta_info"],
        "metamap_phrases": record["metamap_phrases"]
    }
    return entry


def process_dataset_split(dataset, split_name: str, max_records: Optional[int] = None) -> List[Dict]:
    """
    Process a dataset split and convert all records to structured format.

    Args:
        dataset: HuggingFace dataset split
        split_name: Name of the split (train/test)
        max_records: Maximum number of records to process (None for all)

    Returns:
        List of structured entries
    """
    structured_data = []
    total_to_process = len(dataset) if max_records is None else min(max_records, len(dataset))

    for idx, record in enumerate(dataset):
        if max_records is not None and idx >= max_records:
            break

        entry = convert_record_to_entry(record, split_name, idx)
        structured_data.append(entry)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total_to_process} records...")

    return structured_data


def generate_usmle_dataset(output_file: str, n: Optional[int] = None):
    """
    Generate USMLE clinical questions dataset.

    Args:
        output_file: Path to output JSON file
        n: Number of questions to generate (None for all)
    """
    print("Loading USMLE MedQA dataset from HuggingFace...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")

    print(f"\nDataset info:")
    print(f"  Train split: {len(ds['train'])} records")
    print(f"  Test split: {len(ds['test'])} records")

    if n is not None:
        print(f"  Limiting to {n} total questions")

    # Calculate how many records to take from each split
    if n is None:
        train_limit = None
        test_limit = None
    else:
        # Take proportionally from train and test, but prioritize train
        train_limit = min(n, len(ds['train']))
        test_limit = max(0, n - train_limit)

    # Process train split
    print("\nProcessing train split...")
    train_data = process_dataset_split(ds["train"], "train", max_records=train_limit)
    print(f"  Completed: {len(train_data)} train records")

    # Process test split
    print("\nProcessing test split...")
    test_data = process_dataset_split(ds["test"], "test", max_records=test_limit)
    print(f"  Completed: {len(test_data)} test records")

    # Combine all data
    all_data = train_data + test_data
    print(f"\nTotal records: {len(all_data)}")

    # Save to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Generated {len(all_data)} clinical questions.")

    # Print sample
    print("\n" + "="*80)
    print("SAMPLE RECORD:")
    print("="*80)
    sample = all_data[0]
    print(f"ID: {sample['id']}")
    print(f"Meta Info: {sample['meta_info']}")
    print(f"\nQuestion:\n{sample['question'][:300]}...")
    print(f"\nOptions:")
    for key, value in sample['options'].items():
        print(f"  {key}: {value}")
    print(f"\nAnswer: {sample['answer_idx']} - {sample['answer']}")
    print(f"\nMetamap Phrases (first 5): {sample['metamap_phrases'][:5]}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate clinical questions dataset from USMLE MedQA.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='input/clinical_questions_usmle.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '-n', '--num-questions',
        type=int,
        default=1024,
        help='Number of questions to generate (default: all)'
    )

    args = parser.parse_args()

    generate_usmle_dataset(args.output, n=args.num_questions)


if __name__ == '__main__':
    main()
