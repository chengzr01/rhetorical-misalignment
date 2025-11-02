#!/usr/bin/env python3
"""
MIMIC-IV Dataset Column Explorer

Generates a JSON file containing all columns, their data types, and sample statistics
from the processed MIMIC-IV parquet files.

Input directory: /work/hdd/bdhh/ziruic4/datasets/mimiciv/processed/

Usage:
    python demonstrate.py
    python demonstrate.py --output columns_info.json
"""

import json
from pathlib import Path
from typing import Any
import polars as pl
import argparse


def explore_parquet_file(file_path: Path) -> dict[str, Any]:
    """Explore a single parquet file and extract column information."""
    print(f"\nExploring {file_path.name}...")

    df = pl.read_parquet(file_path)

    columns_info = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].null_count()
        total_rows = len(df)
        null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

        col_info = {
            "dtype": dtype,
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
            "total_rows": total_rows
        }

        # Add unique count for categorical-like columns
        if df[col].dtype in [pl.Utf8, pl.Categorical, pl.Boolean]:
            unique_count = df[col].n_unique()
            col_info["unique_count"] = unique_count

            # Add sample values for string columns (up to 10)
            if df[col].dtype == pl.Utf8 and unique_count <= 100:
                sample_values = df[col].drop_nulls().unique().sort().head(10).to_list()
                col_info["sample_values"] = sample_values

        # Add statistics for numeric columns
        elif df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                pl.Float32, pl.Float64]:
            non_null = df[col].drop_nulls()
            if len(non_null) > 0:
                col_info["min"] = float(non_null.min())
                col_info["max"] = float(non_null.max())
                col_info["mean"] = float(non_null.mean())
                col_info["median"] = float(non_null.median())

        columns_info[col] = col_info

    return {
        "file": file_path.name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": columns_info
    }


def explore_all_datasets(input_dir: Path, output_file: Path) -> None:
    """Explore all parquet files in the directory and save column information to JSON."""
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
            dataset_info = explore_parquet_file(parquet_file)
            all_datasets_info[dataset_name] = dataset_info
            print(f"  ✓ {dataset_name}: {dataset_info['total_rows']:,} rows, {dataset_info['total_columns']} columns")
        except Exception as e:
            print(f"  ✗ Error processing {dataset_name}: {e}")

    # Save to JSON
    print("\n" + "=" * 80)
    print(f"Saving column information to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(all_datasets_info, f, indent=2)

    print(f"✓ Done! Column information saved to {output_file}")
    print(f"  Total datasets: {len(all_datasets_info)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore MIMIC-IV dataset columns and generate JSON summary")
    parser.add_argument(
        '--input_dir',
        type=str,
        default="/work/hdd/bdhh/ziruic4/datasets/mimiciv/processed/",
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
    print("MIMIC-IV Dataset Column Explorer")
    print("=" * 80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output file:      {args.output}")
    print("=" * 80)

    explore_all_datasets(Path(args.input_dir), Path(args.output))
