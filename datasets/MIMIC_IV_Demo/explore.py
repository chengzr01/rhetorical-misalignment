#!/usr/bin/env python3
"""
Explore treatment effectiveness from the simple dataset.

Usage:
    python explore.py --data_dir /path/to/output
"""

import argparse
from pathlib import Path
import polars as pl
import datetime


def analyze_treatment_effect(data_dir: Path, drug_name: str = None, lab_test: str = None):
    """
    Analyze treatment effectiveness by comparing lab values before and after medication.

    Args:
        data_dir: Directory containing parquet files
        drug_name: Filter by drug name (substring match)
        lab_test: Filter by lab test name (substring match)
    """
    # Load and filter data
    print("Loading data...")
    meds = pl.read_parquet(data_dir / "medications.parquet")
    labs = pl.read_parquet(data_dir / "labs.parquet")

    if drug_name:
        meds = meds.filter(pl.col("drug").str.contains(f"(?i){drug_name}"))
        print(f"Filtered to drugs containing '{drug_name}': {len(meds)} administrations")

    if lab_test:
        labs = labs.filter(pl.col("lab_test").str.contains(f"(?i){lab_test}"))
        print(f"Filtered to labs containing '{lab_test}': {len(labs)} measurements")

    # Match medications with lab results
    print("\nMatching medications with lab results...")

    # Join and calculate time differences
    combined = (
        meds.join(labs, on=["subject_id", "hadm_id"], how="inner", suffix="_lab")
        .with_columns([
            pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col("charttime_lab").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        ])
        .with_columns([
            (pl.col("charttime_lab") - pl.col("charttime")).alias("time_diff_ns")
        ])
    )

    # Common columns for grouping
    group_cols = ["subject_id", "hadm_id", "drug", "charttime", "lab_test"]

    # Find closest lab values before and after medication
    labs_before = (
        combined
        .filter(pl.col("time_diff_ns") < datetime.timedelta(0))
        .sort([*group_cols, "time_diff_ns"], descending=[False]*5 + [True])
        .group_by(group_cols)
        .agg([
            pl.col("valuenum").first().alias("value_before"),
            pl.col("time_diff_ns").first().alias("time_before")
        ])
    )

    labs_after = (
        combined
        .filter(pl.col("time_diff_ns") > datetime.timedelta(0))
        .sort([*group_cols, "time_diff_ns"])
        .group_by(group_cols)
        .agg([
            pl.col("valuenum").first().alias("value_after"),
            pl.col("time_diff_ns").first().alias("time_after")
        ])
    )

    # Join and calculate changes
    effects = (
        labs_before.join(labs_after, on=group_cols, how="inner")
        .with_columns([
            (pl.col("value_after") - pl.col("value_before")).alias("value_change"),
            ((pl.col("value_after") - pl.col("value_before")) / pl.col("value_before") * 100).alias("pct_change"),
            (pl.col("time_before").cast(pl.Duration).dt.total_hours() * -1).alias("hours_before"),
            (pl.col("time_after").cast(pl.Duration).dt.total_hours()).alias("hours_after")
        ])
    )

    print(f"\nFound {len(effects)} medication-lab pairs with before/after measurements")

    if len(effects) > 0:
        # Generate summary statistics
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

    def get_top_items(df: pl.DataFrame, group_col: str, n: int = 20) -> pl.DataFrame:
        return (
            df
            .group_by(group_col)
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
            .head(n)
        )

    print("\nTop 20 drugs by frequency:")
    print(get_top_items(meds, "drug"))

    print("\nTop 20 lab tests by frequency:")
    print(get_top_items(labs, "lab_test"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explore treatment effectiveness from MIMIC-IV data')
    parser.add_argument('--data_dir', type=str, help='Directory containing parquet files')
    parser.add_argument('--drug', type=str, help='Filter by drug name (substring match)', default=None)
    parser.add_argument('--lab', type=str, help='Filter by lab test name (substring match)', default=None)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    
    if not args.drug and not args.lab:
        show_drug_stats(data_dir)
    else:
        analyze_treatment_effect(data_dir, args.drug, args.lab)
