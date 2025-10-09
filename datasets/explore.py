#!/usr/bin/env python3
"""
Explore treatment effectiveness from the simple dataset.

Usage:
    python explore.py /path/to/output
"""

import sys
from pathlib import Path
import polars as pl


def analyze_treatment_effect(data_dir: Path, drug_name: str = None, lab_test: str = None):
    """
    Analyze treatment effectiveness by comparing lab values before and after medication.

    Args:
        data_dir: Directory containing parquet files
        drug_name: Filter by drug name (substring match)
        lab_test: Filter by lab test name (substring match)
    """

    print("Loading data...")
    meds = pl.read_parquet(data_dir / "medications.parquet")
    labs = pl.read_parquet(data_dir / "labs.parquet")

    # Filter if specified
    if drug_name:
        meds = meds.filter(pl.col("drug").str.contains(f"(?i){drug_name}"))
        print(f"Filtered to drugs containing '{drug_name}': {len(meds)} administrations")

    if lab_test:
        labs = labs.filter(pl.col("lab_test").str.contains(f"(?i){lab_test}"))
        print(f"Filtered to labs containing '{lab_test}': {len(labs)} measurements")

    # For each medication administration, find lab values before and after
    print("\nMatching medications with lab results...")

    results = []

    # Join meds with labs on same admission
    combined = meds.join(labs, on=["subject_id", "hadm_id"], how="inner", suffix="_lab")

    # Calculate time difference (lab time - med time)
    combined = combined.with_columns([
        (pl.col("charttime_lab").cast(pl.Int64) - pl.col("charttime").cast(pl.Int64)).alias("time_diff_ns")
    ])

    # Find lab before (closest negative time_diff)
    labs_before = (
        combined
        .filter(pl.col("time_diff_ns") < 0)
        .sort(["subject_id", "hadm_id", "drug", "charttime", "lab_test", "time_diff_ns"], descending=[False, False, False, False, False, True])
        .group_by(["subject_id", "hadm_id", "drug", "charttime", "lab_test"])
        .agg([
            pl.col("valuenum").first().alias("value_before"),
            pl.col("time_diff_ns").first().alias("time_before")
        ])
    )

    # Find lab after (closest positive time_diff)
    labs_after = (
        combined
        .filter(pl.col("time_diff_ns") > 0)
        .sort(["subject_id", "hadm_id", "drug", "charttime", "lab_test", "time_diff_ns"])
        .group_by(["subject_id", "hadm_id", "drug", "charttime", "lab_test"])
        .agg([
            pl.col("valuenum").first().alias("value_after"),
            pl.col("time_diff_ns").first().alias("time_after")
        ])
    )

    # Join before and after
    effects = labs_before.join(
        labs_after,
        on=["subject_id", "hadm_id", "drug", "charttime", "lab_test"],
        how="inner"
    )

    # Calculate change
    effects = effects.with_columns([
        (pl.col("value_after") - pl.col("value_before")).alias("value_change"),
        ((pl.col("value_after") - pl.col("value_before")) / pl.col("value_before") * 100).alias("pct_change"),
        (pl.col("time_before").abs() / 1e9 / 3600).alias("hours_before"),
        (pl.col("time_after") / 1e9 / 3600).alias("hours_after")
    ])

    print(f"\nFound {len(effects)} medication-lab pairs with before/after measurements")

    if len(effects) > 0:
        # Summary by drug and lab
        summary = (
            effects
            .group_by(["drug", "lab_test"])
            .agg([
                pl.count().alias("n_episodes"),
                pl.col("value_change").mean().alias("avg_change"),
                pl.col("pct_change").mean().alias("avg_pct_change"),
                pl.col("value_change").std().alias("std_change"),
                pl.col("hours_before").mean().alias("avg_hours_before"),
                pl.col("hours_after").mean().alias("avg_hours_after")
            ])
            .sort("n_episodes", descending=True)
        )

        print("\nTreatment Effectiveness Summary:")
        print(summary)

        return effects, summary

    return None, None


def show_drug_stats(data_dir: Path):
    """Show top drugs and lab tests."""
    meds = pl.read_parquet(data_dir / "medications.parquet")
    labs = pl.read_parquet(data_dir / "labs.parquet")

    print("\nTop 20 drugs by frequency:")
    top_drugs = (
        meds
        .group_by("drug")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(20)
    )
    print(top_drugs)

    print("\nTop 20 lab tests by frequency:")
    top_labs = (
        labs
        .group_by("lab_test")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(20)
    )
    print(top_labs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore.py <data_dir> [drug_name] [lab_test]")
        print("Example: python explore.py ./output")
        print("         python explore.py ./output insulin glucose")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    drug_name = sys.argv[2] if len(sys.argv) > 2 else None
    lab_test = sys.argv[3] if len(sys.argv) > 3 else None

    if not drug_name and not lab_test:
        show_drug_stats(data_dir)
    else:
        analyze_treatment_effect(data_dir, drug_name, lab_test)
