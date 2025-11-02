#!/usr/bin/env python3
"""
MIMIC-IV Dataset Column Lister

Simple script to list only column names for each processed parquet file.

Usage:
    python demonstrate.py
    python demonstrate.py --output columns_info.json
"""

import json
from pathlib import Path
import polars as pl
import argparse

def list_parquet_columns(file_path: Path) -> dict:
    """Return just the list of columns for a parquet file."""
    print(f"\nListing columns in {file_path.name}...")
    df = pl.read_parquet(file_path)
    return {
        "file": file_path.name,
        "columns": list(df.columns)
    }

def list_all_datasets(input_dir: Path, output_file: Path) -> None:
    """List columns for all parquet files in the directory and save to JSON."""
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    parquet_files = sorted(input_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files")
    print("=" * 80)

    all_datasets_info = {}

    for parquet_file in parquet_files:
        dataset_name = parquet_file.stem  # Filename without extension
        try:
            dataset_info = list_parquet_columns(parquet_file)
            all_datasets_info[dataset_name] = dataset_info
            print(f"  ✓ {dataset_name}: {len(dataset_info['columns'])} columns")
        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")

    print("\n" + "=" * 80)
    print(f"Saving column information to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(all_datasets_info, f, indent=2)

    print(f"✓ Done! Column information saved to {output_file}")
    print(f"  Total datasets: {len(all_datasets_info)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List columns in MIMIC-IV datasets and generate JSON summary")
    parser.add_argument(
        '--input_dir',
        type=str,
        default="/u/ziruic4/persuasive-misalignment/datasets/mimiciv_demo/processed/",
        help='Input directory containing processed parquet files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="columns_info.json",
        help='Output JSON file name (default: columns_info.json)'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MIMIC-IV Dataset Column Lister")
    print("=" * 80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output file:      {args.output}")
    print("=" * 80)

    list_all_datasets(Path(args.input_dir), Path(args.output))
