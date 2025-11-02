#!/usr/bin/env python3
"""
Generate test case episodes from MIMIC-IV demo data.

This script creates episode dictionaries where each episode represents a patient admission
identified by hadm_id. Each episode contains core patient demographics and admission details
that can be used to generate realistic test scenarios.

Usage:
    python episodes.py --data_dir processed --out_dir test
"""

import argparse
import json
from pathlib import Path
import polars as pl


def create_episode_dictionaries(data_dir: Path) -> list[dict]:
    """
    Create episode dictionaries for each admission.

    Each episode is identified by hadm_id and contains:
    - Patient demographics (from patients table)
    - Admission details (from admissions table)
    - Diagnosis codes (from diagnoses_icd table)
    - DRG codes (from drgcodes table)

    Args:
        data_dir: Directory containing processed parquet files

    Returns:
        List of episode dictionaries
    """
    print("Loading data...")
    patients = pl.read_parquet(data_dir / "patients.parquet")
    admissions = pl.read_parquet(data_dir / "admissions.parquet")
    diagnoses_icd = pl.read_parquet(data_dir / "diagnoses_icd.parquet")
    drgcodes = pl.read_parquet(data_dir / "drgcodes.parquet")
    d_icd_diagnoses = pl.read_parquet(data_dir / "d_icd_diagnoses.parquet")
    procedures_icd = pl.read_parquet(data_dir / "procedures_icd.parquet")
    d_icd_procedures = pl.read_parquet(data_dir / "d_icd_procedures.parquet")
    labevents = pl.read_parquet(data_dir / "labevents.parquet")
    d_labitems = pl.read_parquet(data_dir / "d_labitems.parquet")
    microbiologyevents = pl.read_parquet(data_dir / "microbiologyevents.parquet")
    hcpcsevents = pl.read_parquet(data_dir / "hcpcsevents.parquet")
    d_hcpcs = pl.read_parquet(data_dir / "d_hcpcs.parquet")
    emar = pl.read_parquet(data_dir / "emar.parquet")
    emar_detail = pl.read_parquet(data_dir / "emar_detail.parquet")

    print(f"Loaded {len(patients)} patients, {len(admissions)} admissions")
    print(f"  {len(diagnoses_icd)} ICD diagnoses, {len(procedures_icd)} ICD procedures")
    print(f"  {len(drgcodes)} DRG codes")
    print(f"  {len(labevents)} lab events, {len(microbiologyevents)} microbiology events")
    print(f"  {len(hcpcsevents)} HCPCS events")
    print(f"  {len(emar)} medication administrations (EMAR)")

    # Join patients with admissions to create base episodes
    episodes_df = admissions.join(
        patients,
        on="subject_id",
        how="left"
    )

    print(f"\nCreating {len(episodes_df)} episode dictionaries...")

    # Convert to list of dictionaries
    episodes = []

    for row in episodes_df.iter_rows(named=True):
        hadm_id = row["hadm_id"]

        # Get ICD diagnoses for this admission with descriptions
        icd_dx = (
            diagnoses_icd
            .filter(pl.col("hadm_id") == hadm_id)
            .join(
                d_icd_diagnoses,
                on=["icd_code", "icd_version"],
                how="left"
            )
            .sort("seq_num")
        )
        diagnoses_list = []
        for dx_row in icd_dx.iter_rows(named=True):
            diagnoses_list.append({
                "seq_num": dx_row["seq_num"],
                "icd_code": dx_row["icd_code"],
                "icd_version": dx_row["icd_version"],
                "long_title": dx_row["long_title"]
            })

        # Get ICD procedures for this admission with descriptions
        icd_proc = (
            procedures_icd
            .filter(pl.col("hadm_id") == hadm_id)
            .join(
                d_icd_procedures,
                on=["icd_code", "icd_version"],
                how="left"
            )
            .sort("seq_num")
        )
        procedures_list = []
        for proc_row in icd_proc.iter_rows(named=True):
            procedures_list.append({
                "seq_num": proc_row["seq_num"],
                "chartdate": proc_row["chartdate"],
                "icd_code": proc_row["icd_code"],
                "icd_version": proc_row["icd_version"],
                "long_title": proc_row["long_title"]
            })

        # Get DRG codes for this admission
        drg_data = drgcodes.filter(pl.col("hadm_id") == hadm_id)
        drg_list = []
        for drg_row in drg_data.iter_rows(named=True):
            drg_list.append({
                "drg_type": drg_row["drg_type"],
                "drg_code": drg_row["drg_code"],
                "description": drg_row["description"],
                "drg_severity": drg_row["drg_severity"],
                "drg_mortality": drg_row["drg_mortality"]
            })

        # Get lab events for this admission with item labels
        lab_data = (
            labevents
            .filter(pl.col("hadm_id") == hadm_id)
            .join(
                d_labitems,
                on="itemid",
                how="left"
            )
            .sort("charttime")
        )
        lab_list = []
        for lab_row in lab_data.iter_rows(named=True):
            lab_list.append({
                "labevent_id": lab_row["labevent_id"],
                "charttime": lab_row["charttime"],
                "itemid": lab_row["itemid"],
                "label": lab_row["label"],
                "fluid": lab_row["fluid"],
                "category": lab_row["category"],
                "value": lab_row["value"],
                "valuenum": lab_row["valuenum"],
                "valueuom": lab_row["valueuom"],
                "ref_range_lower": lab_row["ref_range_lower"],
                "ref_range_upper": lab_row["ref_range_upper"],
                "flag": lab_row["flag"]
            })

        # Get microbiology events for this admission
        micro_data = microbiologyevents.filter(pl.col("hadm_id") == hadm_id).sort("charttime")
        micro_list = []
        for micro_row in micro_data.iter_rows(named=True):
            micro_list.append({
                "microevent_id": micro_row["microevent_id"],
                "chartdate": micro_row["chartdate"],
                "charttime": micro_row["charttime"],
                "spec_type_desc": micro_row["spec_type_desc"],
                "test_name": micro_row["test_name"],
                "org_name": micro_row["org_name"],
                "isolate_num": micro_row["isolate_num"],
                "ab_name": micro_row["ab_name"],
                "dilution_text": micro_row["dilution_text"],
                "interpretation": micro_row["interpretation"],
                "comments": micro_row["comments"]
            })

        # Get HCPCS events for this admission with code descriptions
        hcpcs_data = (
            hcpcsevents
            .filter(pl.col("hadm_id") == hadm_id)
            .join(
                d_hcpcs,
                left_on="hcpcs_cd",
                right_on="code",
                how="left"
            )
            .sort("chartdate")
        )
        hcpcs_list = []
        for hcpcs_row in hcpcs_data.iter_rows(named=True):
            hcpcs_list.append({
                "hcpcs_cd": hcpcs_row["hcpcs_cd"],
                "chartdate": hcpcs_row["chartdate"],
                "seq_num": hcpcs_row["seq_num"],
                "short_description": hcpcs_row["short_description"],
                "long_description": hcpcs_row["long_description"],
                "category": hcpcs_row["category"]
            })

        # Get medication administrations (EMAR) for this admission with details
        emar_data = (
            emar
            .filter(pl.col("hadm_id") == hadm_id)
            .join(
                emar_detail,
                on=["subject_id", "emar_id"],
                how="left"
            )
            .sort("charttime")
        )
        emar_list = []
        for emar_row in emar_data.iter_rows(named=True):
            emar_list.append({
                "emar_id": emar_row["emar_id"],
                "charttime": emar_row["charttime"],
                "medication": emar_row["medication"],
                "event_txt": emar_row["event_txt"],
                "product_description": emar_row["product_description"],
                "dose_given": emar_row["dose_given"],
                "dose_given_unit": emar_row["dose_given_unit"],
                "route": emar_row["route"],
                "administration_type": emar_row["administration_type"],
                "infusion_rate": emar_row["infusion_rate"],
                "infusion_rate_unit": emar_row["infusion_rate_unit"]
            })

        episode = {
            # Episode identifier
            "hadm_id": hadm_id,
            "subject_id": row["subject_id"],

            # Patient demographics
            "patient": {
                "sex": row["sex"],
                "anchor_age": row["anchor_age"],
                "anchor_year": row["anchor_year"],
                "anchor_year_group": row["anchor_year_group"],
                "dod": row["dod"]
            },

            # Admission details
            "admission": {
                "admittime": row["admittime"],
                "dischtime": row["dischtime"],
                "deathtime": row["deathtime"],
                "admission_type": row["admission_type"],
                "admit_provider_id": row["admit_provider_id"],
                "admission_location": row["admission_location"],
                "discharge_location": row["discharge_location"],
                "insurance": row["insurance"],
                "language": row["language"],
                "marital_status": row["marital_status"],
                "race": row["race"],
                "edregtime": row["edregtime"],
                "edouttime": row["edouttime"],
                "hospital_expire_flag": row["hospital_expire_flag"]
            },

            # Diagnosis and procedure data
            "diagnoses_icd": diagnoses_list,
            "procedures_icd": procedures_list,
            "drg_codes": drg_list,

            # Lab and microbiology data
            "labevents": lab_list,
            "microbiologyevents": micro_list,

            # HCPCS events (procedures/imaging)
            "hcpcsevents": hcpcs_list,

            # Medication administrations
            "medication_admin": emar_list
        }

        episodes.append(episode)

    return episodes


def save_episodes(episodes: list[dict], out_dir: Path, filename: str = "episodes.json") -> None:
    """
    Save episode dictionaries to JSON file.

    Args:
        episodes: List of episode dictionaries
        out_dir: Output directory
        filename: Output filename
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    with open(out_path, 'w') as f:
        json.dump(episodes, f, indent=2)

    print(f"\nSaved {len(episodes)} episodes to {out_path}")

    # Print summary statistics
    print("\nEpisode Summary:")
    print(f"  Total episodes: {len(episodes)}")

    # Count unique patients
    unique_patients = len(set(ep["subject_id"] for ep in episodes))
    print(f"  Unique patients: {unique_patients}")

    # Count diagnosis and procedure coverage
    episodes_with_icd = sum(1 for ep in episodes if len(ep["diagnoses_icd"]) > 0)
    episodes_with_procedures = sum(1 for ep in episodes if len(ep["procedures_icd"]) > 0)
    episodes_with_drg = sum(1 for ep in episodes if len(ep["drg_codes"]) > 0)
    episodes_with_labs = sum(1 for ep in episodes if len(ep["labevents"]) > 0)
    episodes_with_micro = sum(1 for ep in episodes if len(ep["microbiologyevents"]) > 0)
    episodes_with_hcpcs = sum(1 for ep in episodes if len(ep["hcpcsevents"]) > 0)
    episodes_with_meds = sum(1 for ep in episodes if len(ep["medication_admin"]) > 0)
    total_icd_diagnoses = sum(len(ep["diagnoses_icd"]) for ep in episodes)
    total_procedures = sum(len(ep["procedures_icd"]) for ep in episodes)
    total_drg_codes = sum(len(ep["drg_codes"]) for ep in episodes)
    total_labs = sum(len(ep["labevents"]) for ep in episodes)
    total_micro = sum(len(ep["microbiologyevents"]) for ep in episodes)
    total_hcpcs = sum(len(ep["hcpcsevents"]) for ep in episodes)
    total_meds = sum(len(ep["medication_admin"]) for ep in episodes)

    print(f"\n  Diagnosis and procedure coverage:")
    print(f"    Episodes with ICD diagnoses: {episodes_with_icd} ({episodes_with_icd/len(episodes)*100:.1f}%)")
    print(f"    Episodes with ICD procedures: {episodes_with_procedures} ({episodes_with_procedures/len(episodes)*100:.1f}%)")
    print(f"    Episodes with DRG codes: {episodes_with_drg} ({episodes_with_drg/len(episodes)*100:.1f}%)")
    print(f"    Total ICD diagnoses: {total_icd_diagnoses}")
    print(f"    Total ICD procedures: {total_procedures}")
    print(f"    Total DRG codes: {total_drg_codes}")
    print(f"    Avg ICD diagnoses per episode: {total_icd_diagnoses/len(episodes):.1f}")
    print(f"    Avg ICD procedures per episode: {total_procedures/len(episodes):.1f}")

    print(f"\n  Lab and microbiology coverage:")
    print(f"    Episodes with lab events: {episodes_with_labs} ({episodes_with_labs/len(episodes)*100:.1f}%)")
    print(f"    Episodes with microbiology events: {episodes_with_micro} ({episodes_with_micro/len(episodes)*100:.1f}%)")
    print(f"    Episodes with HCPCS events: {episodes_with_hcpcs} ({episodes_with_hcpcs/len(episodes)*100:.1f}%)")
    print(f"    Episodes with medication administrations: {episodes_with_meds} ({episodes_with_meds/len(episodes)*100:.1f}%)")
    print(f"    Total lab events: {total_labs:,}")
    print(f"    Total microbiology events: {total_micro:,}")
    print(f"    Total HCPCS events: {total_hcpcs:,}")
    print(f"    Total medication administrations: {total_meds:,}")
    print(f"    Avg lab events per episode: {total_labs/len(episodes):.1f}")
    print(f"    Avg microbiology events per episode: {total_micro/len(episodes):.1f}")
    print(f"    Avg HCPCS events per episode: {total_hcpcs/len(episodes):.1f}")
    print(f"    Avg medication administrations per episode: {total_meds/len(episodes):.1f}")

    # Count admission types
    admission_types = {}
    for ep in episodes:
        adm_type = ep["admission"]["admission_type"]
        admission_types[adm_type] = admission_types.get(adm_type, 0) + 1

    print("\n  Admission types:")
    for adm_type, count in sorted(admission_types.items(), key=lambda x: -x[1]):
        print(f"    {adm_type}: {count}")

    # Count by sex
    sex_counts = {}
    for ep in episodes:
        sex = ep["patient"]["sex"]
        sex_counts[sex] = sex_counts.get(sex, 0) + 1

    print("\n  Patient sex distribution:")
    for sex, count in sorted(sex_counts.items()):
        print(f"    {sex}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate test case episodes from MIMIC-IV demo data'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='processed',
        help='Directory containing processed parquet files'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='test',
        help='Output directory for episode JSON files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='episodes.json',
        help='Output filename for episodes JSON'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # Create episode dictionaries
    episodes = create_episode_dictionaries(data_dir)

    # Save to file
    save_episodes(episodes, out_dir, args.output_file)

    print("\nDone!")


if __name__ == "__main__":
    main()
