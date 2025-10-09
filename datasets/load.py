#!/usr/bin/env python3
"""
Usage:
    python load.py /path/to/csv/files /path/to/output
"""

import sys
from pathlib import Path
import polars as pl


def load_and_transform(csv_dir: Path, out_dir: Path):
    """Load 6 tables and create a simple treatment effectiveness dataset."""

    print("Loading tables...")

    # Load patients (rename gender -> sex for consistency)
    patients = pl.read_csv(csv_dir / "hosp" / "patients.csv").rename({"gender": "sex"})

    # Load admissions
    admissions = pl.read_csv(csv_dir / "hosp" / "admissions.csv")

    # Load medications
    emar = pl.read_csv(csv_dir / "hosp" / "emar.csv")
    emar_detail = pl.read_csv(csv_dir / "hosp" / "emar_detail.csv")

    # Load lab results
    labevents = pl.read_csv(csv_dir / "hosp" / "labevents.csv")
    d_labitems = pl.read_csv(csv_dir / "hosp" / "d_labitems.csv")

    print(f"  Patients: {len(patients)} rows")
    print(f"  Admissions: {len(admissions)} rows")
    print(f"  Medications: {len(emar)} rows")
    print(f"  Med details: {len(emar_detail)} rows")
    print(f"  Lab events: {len(labevents)} rows")
    print(f"  Lab items: {len(d_labitems)} rows")

    # Create simple treatment-outcome dataset
    print("\nCreating treatment-outcome dataset...")

    # Join medications with details to get drug names
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
    )

    # Join lab results with item names
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
    )

    # Combine with patient demographics
    meds_full = meds.join(
        patients.select(["subject_id", "sex", "anchor_age"]),
        on="subject_id",
        how="left"
    )

    labs_full = labs.join(
        patients.select(["subject_id", "sex", "anchor_age"]),
        on="subject_id",
        how="left"
    )

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving files...")
    patients.write_parquet(out_dir / "patients.parquet")
    admissions.write_parquet(out_dir / "admissions.parquet")
    meds_full.write_parquet(out_dir / "medications.parquet")
    labs_full.write_parquet(out_dir / "labs.parquet")

    print(f"\nDone! Files saved to {out_dir}")
    print(f"  - patients.parquet ({len(patients)} rows)")
    print(f"  - admissions.parquet ({len(admissions)} rows)")
    print(f"  - medications.parquet ({len(meds_full)} rows)")
    print(f"  - labs.parquet ({len(labs_full)} rows)")

    return patients, admissions, meds_full, labs_full


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python load.py <csv_dir> <out_dir>")
        print("Example: python load.py /projects/bdhh/haopeng/physionet.org ./output")
        sys.exit(1)

    csv_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    load_and_transform(csv_dir, out_dir)
