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
    """
    Load all tables (CSVs) present in the relevant folders and subfolders.
    Handles special dtype issues for d_icd_diagnoses, d_icd_procedures, pharmacy, and prescriptions.
    """
    print("Loading tables...")

    tables = {}

    # Fix: These tables have mixed-format ICD codes, must be loaded as string
    force_str_schema_tables = {
        ("hosp", "d_icd_diagnoses"): {"icd_code": pl.String},
        ("hosp", "d_icd_procedures"): {"icd_code": pl.String}
    }

    # Additional hard-coded overrides for tables with known issues
    force_schema_overrides = {
        ("hosp", "pharmacy"): {
            # dispense times can be weird strings when scheduled, so treat as string or allow nulls
            "disp_sched": pl.String,  # Accept "08, 20" as a string instead of int
        },
        ("hosp", "prescriptions"): {
            # dose_val_rx mixed-type; treat as string if it can't be int/float
            "dose_val_rx": pl.String,
        }
    }
    # Also, treat these as null values to avoid parse errors (for pharmacy disp_sched)
    additional_null_values = {
        ("hosp", "pharmacy"): ["", "NULL"]
    }

    for subfolder in ["hosp", "icu"]:
        folder = csv_dir / subfolder
        if not folder.exists():
            continue
        for csv_file in folder.glob("*.csv"):
            table_name = csv_file.stem
            key_name = f"{subfolder}_{table_name}" if subfolder == "icu" else table_name
            try:
                print(f"Loading table: {key_name}")  # Print the name of the table being dealt with

                # Special handling for 'patients' and 'emar_detail'
                if subfolder == "hosp" and table_name == "patients":
                    tables[table_name] = pl.read_csv(csv_file).rename({"gender": "sex"})
                elif subfolder == "hosp" and table_name == "emar_detail":
                    tables[table_name] = pl.read_csv(
                        csv_file,
                        schema_overrides={
                            "infusion_rate": pl.Float64,
                            "product_amount_given": pl.Float64
                        }
                    )
                elif (subfolder, table_name) in force_str_schema_tables:
                    # Fix: For ICD tables, force icd_code as string
                    overrides = force_str_schema_tables[(subfolder, table_name)]
                    tables[table_name] = pl.read_csv(csv_file, schema_overrides=overrides)
                elif (subfolder, table_name) in force_schema_overrides:
                    overrides = force_schema_overrides[(subfolder, table_name)]
                    # Use extra null values if set for this table
                    null_values = additional_null_values.get((subfolder, table_name), None)
                    tables[table_name] = pl.read_csv(
                        csv_file,
                        schema_overrides=overrides,
                        null_values=null_values,
                        ignore_errors=True  # for problematic columns, ignore type errors and continue
                    )
                else:
                    tables[key_name] = pl.read_csv(csv_file)
                print(f"  {subfolder}/{table_name}: {len(tables[key_name if subfolder == 'icu' else table_name])} rows")
            except Exception as e:
                print(f"  Warning: Failed to load {subfolder}/{table_name}: {e}")
    return tables


def create_medication_dataset(emar: pl.DataFrame, emar_detail: pl.DataFrame, patients: pl.DataFrame) -> pl.DataFrame:
    """Create medication dataset with patient demographics."""
    print("Dealing with table: medications")
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
    print("Dealing with table: labs")
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
        print(f"Saving table: {name}")  # Print the name of the table being saved
        df.write_parquet(out_dir / f"{name}.parquet")
        print(f"  - {name}.parquet ({len(df)} rows)")


def load_and_transform(csv_dir: Path, out_dir: Path) -> dict[str, pl.DataFrame]:
    """Load all tables and create enriched analysis datasets."""
    # Load raw tables
    tables = load_tables(csv_dir)

    print("\nCreating enriched datasets...")

    # Create enriched datasets (only if required tables exist)
    datasets = {}

    # Known table key mapping for derived datasets (because ICU tables are prefixed)
    hosp_patients_key = "patients"
    hosp_emar_key = "emar"
    hosp_emar_detail_key = "emar_detail"
    hosp_labevents_key = "labevents"
    hosp_d_labitems_key = "d_labitems"

    # Add medication dataset if components exist
    if all(k in tables for k in [hosp_emar_key, hosp_emar_detail_key, hosp_patients_key]):
        print("Creating table: medications")
        datasets["medications"] = create_medication_dataset(
            tables[hosp_emar_key],
            tables[hosp_emar_detail_key],
            tables[hosp_patients_key]
        )
        print(f"  Created medications dataset: {len(datasets['medications'])} rows")

    # Add labs dataset if components exist
    if all(k in tables for k in [hosp_labevents_key, hosp_d_labitems_key, hosp_patients_key]):
        print("Creating table: labs")
        datasets["labs"] = create_labs_dataset(
            tables[hosp_labevents_key],
            tables[hosp_d_labitems_key],
            tables[hosp_patients_key]
        )
        print(f"  Created labs dataset: {len(datasets['labs'])} rows")

    # Add all raw tables to output (do not overwrite derived datasets)
    for key, value in tables.items():
        if key not in datasets:
            print(f"Dealing with table: {key}")
            datasets[key] = value

    # Save all datasets
    save_datasets(datasets, out_dir)

    print(f"\nDone! {len(datasets)} files saved to {out_dir}")

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and transform MIMIC-IV data into parquet files')
    parser.add_argument('--csv_dir', type=str, default="/u/ziruic4/persuasive-misalignment/datasets/mimiciv_demo/raw", help='Directory containing CSV files')
    parser.add_argument('--out_dir', type=str, default="/u/ziruic4/persuasive-misalignment/datasets/mimiciv_demo/processed", help='Output directory for parquet files')
    
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)

    load_and_transform(csv_dir, out_dir)
