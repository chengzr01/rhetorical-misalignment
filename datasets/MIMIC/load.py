#!/usr/bin/env python3
"""
Usage:
    python load.py --csv_dir /path/to/csv/files --out_dir /path/to/output
"""

import sys
import argparse
from pathlib import Path
import polars as pl


def load_tables(csv_dir: Path) -> dict[str, pl.DataFrame]:
    """Load all required tables from CSV files."""
    print("Loading tables...")
    
    tables = {
        "patients": pl.read_csv(csv_dir / "hosp" / "patients.csv").rename({"gender": "sex"}),
        "admissions": pl.read_csv(csv_dir / "hosp" / "admissions.csv"),
        "emar": pl.read_csv(csv_dir / "hosp" / "emar.csv"),
        "emar_detail": pl.read_csv(
            csv_dir / "hosp" / "emar_detail.csv", 
            schema_overrides={
                "infusion_rate": pl.Float64,
                "product_amount_given": pl.Float64
            }
        ),
        "labevents": pl.read_csv(csv_dir / "hosp" / "labevents.csv"),
        "d_labitems": pl.read_csv(csv_dir / "hosp" / "d_labitems.csv")
    }

    for name, df in tables.items():
        print(f"  {name.title()}: {len(df)} rows")

    return tables


def create_medication_dataset(emar: pl.DataFrame, emar_detail: pl.DataFrame, patients: pl.DataFrame) -> pl.DataFrame:
    """Create medication dataset with patient demographics."""
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
            patients.select(["subject_id", "sex", "anchor_age"]),
            on="subject_id",
            how="left"
        )
    )
    return meds


def create_labs_dataset(labevents: pl.DataFrame, d_labitems: pl.DataFrame, patients: pl.DataFrame) -> pl.DataFrame:
    """Create lab results dataset with patient demographics."""
    labs = (
        labevents
        .join(d_labitems, on="itemid", how="inner")
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
            patients.select(["subject_id", "sex", "anchor_age"]),
            on="subject_id", 
            how="left"
        )
    )
    return labs


def save_datasets(datasets: dict[str, pl.DataFrame], out_dir: Path) -> None:
    """Save all datasets to parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving files...")
    for name, df in datasets.items():
        df.write_parquet(out_dir / f"{name}.parquet")
        print(f"  - {name}.parquet ({len(df)} rows)")


def load_and_transform(csv_dir: Path, out_dir: Path) -> tuple[pl.DataFrame, ...]:
    """Load tables and create a simple treatment effectiveness dataset."""
    # Load raw tables
    tables = load_tables(csv_dir)
    
    print("\nCreating treatment-outcome dataset...")
    
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

    # Save all datasets
    datasets = {
        "patients": tables["patients"],
        "admissions": tables["admissions"],
        "medications": meds_full,
        "labs": labs_full
    }
    save_datasets(datasets, out_dir)
    
    print(f"\nDone! Files saved to {out_dir}")
    
    return tuple(datasets.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and transform MIMIC-IV data into parquet files')
    parser.add_argument('--csv_dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--out_dir', type=str, help='Output directory for parquet files')
    
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)

    load_and_transform(csv_dir, out_dir)
