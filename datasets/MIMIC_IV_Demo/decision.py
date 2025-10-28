#!/usr/bin/env python3

import polars as pl
import datetime
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings('ignore')

# Data directory
DATA_DIR = Path("processed")

# Load datasets
print("Loading datasets")
patients = pl.read_parquet(DATA_DIR / "patients.parquet")
admissions = pl.read_parquet(DATA_DIR / "admissions.parquet")
medications = pl.read_parquet(DATA_DIR / "medications.parquet")
labs = pl.read_parquet(DATA_DIR / "labs.parquet")

print(f"Patients: {len(patients):,}")
print(f"Admissions: {len(admissions):,}")
print(f"Medication records: {len(medications):,}")
print(f"Lab records: {len(labs):,}")
print("Datasets loaded successfully!\n")


def get_patient_profile(patient_id: int, n_examples: int = 3):
    """Get comprehensive patient information."""
    patient_info = patients.filter(pl.col("subject_id") == patient_id)

    if len(patient_info) == 0:
        print(f"Patient {patient_id} not found!")
        return None

    patient_admissions = admissions.filter(pl.col("subject_id") == patient_id).sort("admittime")
    patient_meds = medications.filter(pl.col("subject_id") == patient_id).sort("charttime")
    patient_labs = labs.filter(pl.col("subject_id") == patient_id).sort("charttime")

    return {
        "demographics": patient_info,
        "admissions": patient_admissions,
        "medications": patient_meds,
        "labs": patient_labs
    }


def get_lab_summary(patient_id: int, lab_test_name: str = None):
    """Get lab value summary for a patient, optionally filtered by test name."""
    patient_labs = labs.filter(pl.col("subject_id") == patient_id)

    if lab_test_name:
        patient_labs = patient_labs.filter(
            pl.col("lab_test").str.contains(f"(?i){lab_test_name}")
        )

    if len(patient_labs) == 0:
        print("No lab results found with those criteria.")
        return None, None

    summary = (
        patient_labs
        .group_by(["lab_test", "valueuom"])
        .agg([
            pl.count().alias("n_measurements"),
            pl.col("valuenum").min().alias("min_value"),
            pl.col("valuenum").max().alias("max_value"),
            pl.col("valuenum").mean().alias("mean_value"),
            pl.col("charttime").min().alias("first_test"),
            pl.col("charttime").max().alias("last_test")
        ])
        .sort("n_measurements", descending=True)
    )

    return patient_labs, summary


def get_medication_summary(patient_id: int, drug_name: str = None):
    """Get medication summary for a patient, optionally filtered by drug name."""
    patient_meds = medications.filter(pl.col("subject_id") == patient_id)

    if drug_name:
        patient_meds = patient_meds.filter(
            pl.col("drug").str.contains(f"(?i){drug_name}")
        )

    if len(patient_meds) == 0:
        print("No medication records found with those criteria.")
        return None, None

    summary = (
        patient_meds
        .group_by(["drug", "unit"])
        .agg([
            pl.count().alias("n_administrations"),
            pl.col("charttime").min().alias("first_admin"),
            pl.col("charttime").max().alias("last_admin")
        ])
        .sort("n_administrations", descending=True)
    )

    return patient_meds, summary


def should_give_treatment(patient_id: int, drug_name: str, lab_test: str,
                          current_lab_value: float = None):
    """
    Decision support for whether to administer a treatment.
    Analyzes historical outcomes for similar patients.
    """
    patient_info = patients.filter(pl.col("subject_id") == patient_id)
    if len(patient_info) == 0:
        return {"error": "Patient not found"}

    patient_sex = patient_info['sex'][0]
    patient_age = patient_info['anchor_age'][0]

    # Find similar patients (same sex, age ±10 years)
    age_range = (patient_age - 10, patient_age + 10)

    meds_filtered = medications.filter(
        pl.col("drug").str.contains(f"(?i){drug_name}")
    ).filter(
        (pl.col("sex") == patient_sex) &
        (pl.col("anchor_age") >= age_range[0]) &
        (pl.col("anchor_age") <= age_range[1])
    )

    labs_filtered = labs.filter(
        pl.col("lab_test").str.contains(f"(?i){lab_test}")
    ).filter(
        (pl.col("sex") == patient_sex) &
        (pl.col("anchor_age") >= age_range[0]) &
        (pl.col("anchor_age") <= age_range[1])
    )

    if len(meds_filtered) == 0 or len(labs_filtered) == 0:
        return {
            "recommendation": "INSUFFICIENT DATA",
            "reason": "Not enough historical data for similar patients"
        }

    # Join and calculate treatment effects
    combined = (
        meds_filtered.join(labs_filtered, on=["subject_id", "hadm_id"], how="inner", suffix="_lab")
        .with_columns([
            pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col("charttime_lab").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        ])
        .with_columns([
            (pl.col("charttime_lab") - pl.col("charttime")).alias("time_diff_ns")
        ])
    )

    group_cols = ["subject_id", "hadm_id", "drug", "charttime", "lab_test"]

    # Labs before treatment
    labs_before = (
        combined
        .filter(pl.col("time_diff_ns") < datetime.timedelta(0))
        .sort([*group_cols, "time_diff_ns"], descending=[False]*5 + [True])
        .group_by(group_cols)
        .agg([pl.col("valuenum").first().alias("value_before")])
    )

    # Labs after treatment
    labs_after = (
        combined
        .filter(pl.col("time_diff_ns") > datetime.timedelta(0))
        .sort([*group_cols, "time_diff_ns"])
        .group_by(group_cols)
        .agg([pl.col("valuenum").first().alias("value_after")])
    )

    # Calculate effects
    effects = (
        labs_before.join(labs_after, on=group_cols, how="inner")
        .with_columns([
            (pl.col("value_after") - pl.col("value_before")).alias("value_change"),
            ((pl.col("value_after") - pl.col("value_before")) / pl.col("value_before") * 100).alias("pct_change")
        ])
    )

    if len(effects) == 0:
        return {
            "recommendation": "INSUFFICIENT DATA",
            "reason": "No before/after measurements found for similar patients"
        }

    # Analyze outcomes
    avg_change = effects.select(pl.col("value_change").mean())[0,0]
    avg_pct_change = effects.select(pl.col("pct_change").mean())[0,0]
    n_episodes = len(effects)

    # Count positive outcomes (assuming decrease is good)
    positive = len(effects.filter(pl.col("value_change") < 0))
    success_rate = positive / n_episodes * 100

    # Make recommendation
    if success_rate >= 60 and avg_change < 0:
        recommendation = "RECOMMEND"
    elif success_rate >= 40:
        recommendation = "CONSIDER"
    else:
        recommendation = "NOT RECOMMENDED"

    return {
        "recommendation": recommendation,
        "patient_demographics": f"{patient_sex}, age {patient_age}",
        "similar_patients_analyzed": n_episodes,
        "average_value_change": round(avg_change, 2),
        "average_percent_change": round(avg_pct_change, 2),
        "success_rate_percent": round(success_rate, 1),
        "current_value": current_lab_value,
        "historical_effects": effects
    }


def get_latest_labs(patient_id: int, n_recent: int = 10):
    """Get the most recent lab values for a patient."""
    return (
        labs
        .filter(pl.col("subject_id") == patient_id)
        .sort("charttime", descending=True)
        .select(["charttime", "lab_test", "valuenum", "valueuom"])
        .head(n_recent)
    )


def get_latest_medications(patient_id: int, n_recent: int = 10):
    """Get the most recent medications for a patient."""
    return (
        medications
        .filter(pl.col("subject_id") == patient_id)
        .sort("charttime", descending=True)
        .select(["charttime", "drug", "dose", "unit"])
        .head(n_recent)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze patient data from MIMIC dataset')
    parser.add_argument('--patient_id', type=int, help='Patient ID to analyze')
    parser.add_argument('--n_examples', type=int, default=10, help='Number of example records to show')
    args = parser.parse_args()

    print(f"Analyzing Patient {args.patient_id}")
    print("=" * 80)

    # Get profile
    profile = get_patient_profile(args.patient_id)
    if profile:
        print(f"\nPatient has {len(profile['medications'])} medication records")
        print(f"Patient has {len(profile['labs'])} lab records")
        
        if len(profile['medications']) > 0:
            print(f"\n{min(args.n_examples, len(profile['medications']))} example medication records:")
            example_meds = profile['medications'].head(args.n_examples)
            for i in range(min(args.n_examples, len(example_meds))):
                print(f"\nRecord {i+1}:")
                for col in example_meds.columns:
                    print(f"{col}: {example_meds[col][i]}")
            
        if len(profile['labs']) > 0:
            print(f"\n{min(args.n_examples, len(profile['labs']))} example lab records:")
            example_labs = profile['labs'].head(args.n_examples)
            for i in range(min(args.n_examples, len(example_labs))):
                print(f"\nRecord {i+1}:")
                for col in example_labs.columns:
                    print(f"{col}: {example_labs[col][i]}")
