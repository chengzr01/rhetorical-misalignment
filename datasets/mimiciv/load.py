#!/usr/bin/env python3
"""
MIMIC-IV Full Dataset Processing Script

Input directory: /work/hdd/bdhh/ziruic4/physionet.org/files/mimiciv/3.1/
Output directory: /work/hdd/bdhh/ziruic4/datasets/mimiciv/processed/

Usage:
    python load.py --prop 0.05
"""

import sys
from pathlib import Path
import polars as pl
from tqdm import tqdm

import argparse

def limit_rows(df: pl.DataFrame, proportion: float) -> pl.DataFrame:
    if not (0 < proportion <= 1):
        raise ValueError(f"Proportion must be between 0 and 1. Got {proportion}")
    if len(df) == 0:
        return df
    n = int(len(df) * proportion)
    n = max(1, n)  # Always keep at least one row if not empty
    return df.slice(0, n)

def load_tables(csv_dir: Path) -> dict[str, pl.DataFrame | pl.LazyFrame]:
    """Load all required tables from CSV files.

    Large tables (labevents, emar, emar_detail) are loaded as LazyFrames
    to avoid memory issues with 40+ million rows.
    """
    print("\nLoading tables from full MIMIC-IV dataset...")

    # Ensure csv_dir is a Path object
    if not isinstance(csv_dir, Path):
        csv_dir = Path(csv_dir)

    table_configs = {
        "patients": {"path": csv_dir / "hosp" / "patients.csv", "rename": {"gender": "sex"}},
        "admissions": {"path": csv_dir / "hosp" / "admissions.csv"},
        "emar": {"path": csv_dir / "hosp" / "emar.csv", "lazy": True},
        "emar_detail": {
            "path": csv_dir / "hosp" / "emar_detail.csv",
            "null_values": ["___"],
            "schema_overrides": {
                "infusion_rate": pl.Float64,
                "product_amount_given": pl.Float64,
                "dose_given": pl.Float64
            },
            "lazy": True
        },
        "labevents": {"path": csv_dir / "hosp" / "labevents.csv", "lazy": True},
        "d_labitems": {"path": csv_dir / "hosp" / "d_labitems.csv"}
    }

    tables = {}
    for name, config in tqdm(table_configs.items(), desc="Loading tables"):
        if config.get("lazy"):
            # Use scan_csv for lazy loading of large tables
            df = pl.scan_csv(
                config["path"],
                null_values=config.get("null_values", []),
                schema_overrides=config.get("schema_overrides", {}),
                ignore_errors=True    # Added to ignore errors and allow parsing
            )
            print(f"  {name.title()}: Loaded lazily (streaming mode)")
        else:
            df = pl.read_csv(config["path"]).rename(config.get("rename", {}))
            print(f"  {name.title()}: {len(df):,} rows")
        tables[name] = df

    return tables

def create_medication_dataset(emar: pl.LazyFrame, emar_detail: pl.LazyFrame, patients: pl.DataFrame) -> pl.DataFrame:
    """Create medication dataset with patient demographics.

    Uses an updated execution method to handle large tables efficiently.
    """
    print("\nCreating medications dataset...")
    meds = (
        emar
        .join(emar_detail, on=["subject_id", "emar_id"], how="inner")
        .select([
            "subject_id",
            "hadm_id",
            "charttime",
            pl.col("product_description").alias("drug"),
            pl.col("dose_given").alias("dose"),
            pl.col("dose_given_unit").alias("unit")
        ])
        .join(
            patients.select(["subject_id", "sex", "anchor_age"]).lazy(),
            on="subject_id",
            how="left"
        )
        .collect(engine="streaming")  # Updated to use `engine="streaming"`
    )
    print(f"  Medications: {len(meds):,} rows")
    return meds

def create_labs_dataset(labevents: pl.LazyFrame, d_labitems: pl.DataFrame, patients: pl.DataFrame) -> pl.DataFrame:
    """Create lab results dataset with patient demographics.

    Uses an updated execution method to handle large tables efficiently (42M+ rows).
    """
    print("\nCreating labs dataset...")
    labs = (
        labevents
        .join(d_labitems.lazy(), on="itemid", how="inner")
        .select([
            "subject_id",
            "hadm_id",
            "charttime",
            pl.col("label").alias("lab_test"),
            "valuenum",
            "valueuom"
        ])
        .filter(pl.col("valuenum").is_not_null())
        .join(
            patients.select(["subject_id", "sex", "anchor_age"]).lazy(),
            on="subject_id",
            how="left"
        )
        .collect(engine="streaming")  # Updated to use `engine="streaming"`
    )
    print(f"  Labs: {len(labs):,} rows")
    return labs

def save_datasets(datasets: dict[str, pl.DataFrame], out_dir: Path, prop: float = 1.0) -> None:
    """Save all datasets to parquet files with chunking for large files (optionally proportionally)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving files...")
    for name, df in tqdm(datasets.items(), desc="Saving datasets"):
        output_path = out_dir / f"{name}.parquet"
        
        # Check if lazy or regular DataFrame
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        
        # Limit to a sampled proportion, except for prop==1.0 (full save)
        if 0 < prop < 1.0:
            orig_len = len(df)
            df = limit_rows(df, prop)
            print(f"  - Proportionally saving {len(df):,}/{orig_len:,} rows to {name}.parquet")
        else:
            print(f"  - Saving full {len(df):,} rows to {name}.parquet")

        df.write_parquet(output_path)
        print(f"  - {name}.parquet ({len(df):,} rows)")

def load_and_transform(csv_dir: Path, out_dir: Path, prop: float = 1.0) -> tuple[pl.DataFrame, ...]:
    """Load tables and create treatment effectiveness datasets from full MIMIC-IV.
    Will only save a sampled proportion of the datasets if specified.
    """
    # Ensure csv_dir and out_dir are Path objects
    if not isinstance(csv_dir, Path):
        csv_dir = Path(csv_dir)
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)

    # Load raw tables
    tables = load_tables(csv_dir)

    print("\nCreating treatment-outcome datasets...")

    # Transform into analysis datasets
    meds_full = create_medication_dataset(
        tables["emar"],
        tables["emar_detail"],
        tables["patients"]
    )

    labs_full = create_labs_dataset(
        tables["labevents"],
        tables["d_labitems"],
        tables["patients"]
    )

    # Save all datasets (save only a subset if prop < 1)
    datasets = {
        "patients": tables["patients"],
        "admissions": tables["admissions"],
        "medications": meds_full,
        "labs": labs_full
    }
    save_datasets(datasets, out_dir, prop=prop)

    print(f"\nDone! Files saved to {out_dir}")

    return tuple(datasets.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save a subset of MIMIC-IV dataset.")
    parser.add_argument('--csv_dir', type=str, default="/work/hdd/bdhh/ziruic4/physionet.org/files/mimiciv/3.1/", help='Input directory containing CSV files')
    parser.add_argument('--out_dir', type=str, default="/work/hdd/bdhh/ziruic4/datasets/mimiciv/processed/", help='Output directory for parquet files')
    parser.add_argument('--prop', type=float, default=0.05, help='Proportion of each dataset to save (0 < prop <= 1, default: 0.05)')
    args = parser.parse_args()

    print("=" * 80)
    print("MIMIC-IV Full Dataset Processing")
    print("=" * 80)
    print(f"Input directory:  {args.csv_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Saving proportion: {args.prop}")
    print("=" * 80)

    load_and_transform(args.csv_dir, args.out_dir, args.prop)
