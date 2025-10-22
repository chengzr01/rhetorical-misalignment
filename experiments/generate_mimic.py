from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from tqdm import tqdm

import polars as pl
import yaml

from interface.client import OpenRouterChatClient, SGLangChatClient, NvidiaChatClient


def setup_client(backend: str) -> OpenRouterChatClient | SGLangChatClient | NvidiaChatClient:
    if backend == "nvidia":
        return NvidiaChatClient()
    elif backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=30000, base_url="http://127.0.0.1")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def load_prompt_template(template_path: str | Path) -> str:
    """Load prompt template from YAML file."""
    with open(template_path, "r") as f:
        template_config = yaml.safe_load(f)
        return template_config["prompt"]


def select_patient_admission(
    medications_df: pl.DataFrame,
    labs_df: pl.DataFrame,
    admissions_df: pl.DataFrame,
    patient_id: int
):
    """
    Select a hospital admission for a patient that has both medication and lab records.
    Returns (hadm_id, decision_time, admission_context) or (None, None, None) if no suitable admission found.

    The decision_time is set at ~60% through the admission timeline to ensure we have
    historical data to work with and potentially future data for validation.

    admission_context is a dict with contextual information about the admission.
    """
    # Get admissions with both medications and labs
    patient_meds = medications_df.filter(pl.col("subject_id") == patient_id)
    patient_labs = labs_df.filter(pl.col("subject_id") == patient_id)

    if len(patient_meds) == 0 or len(patient_labs) == 0:
        return None, None, None

    # Find admissions that have both meds and labs
    admissions_with_meds = set(patient_meds["hadm_id"].unique().to_list())
    admissions_with_labs = set(patient_labs["hadm_id"].unique().to_list())
    valid_admissions = admissions_with_meds & admissions_with_labs

    if not valid_admissions:
        return None, None, None

    # Select a random admission from valid ones
    hadm_id = random.choice(list(valid_admissions))

    # Get all records for this admission
    admission_meds = patient_meds.filter(pl.col("hadm_id") == hadm_id)
    admission_labs = patient_labs.filter(pl.col("hadm_id") == hadm_id)

    # Parse charttimes to find temporal range
    med_times = admission_meds.with_columns(
        pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )["charttime"]

    lab_times = admission_labs.with_columns(
        pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    )["charttime"]

    # Combine all times and sort
    all_times = pl.concat([med_times, lab_times]).sort()

    if len(all_times) < 4:  # Need at least a few records before decision point
        return None, None, None

    # Set decision point at ~60% through the admission (leaves some future for potential validation)
    decision_idx = int(len(all_times) * 0.6)
    decision_time = all_times[decision_idx]

    # Get admission context information
    admission_info = admissions_df.filter(pl.col("hadm_id") == hadm_id)
    if len(admission_info) == 0:
        return None, None, None

    admission_row = admission_info.row(0, named=True)
    admission_context = {
        "admission_type": admission_row.get("admission_type", "Unknown"),
        "admission_location": admission_row.get("admission_location", "Unknown"),
        "discharge_location": admission_row.get("discharge_location", "Unknown"),
        "insurance": admission_row.get("insurance", "Unknown"),
        "language": admission_row.get("language", "Unknown"),
        "marital_status": admission_row.get("marital_status", "Unknown"),
        "race": admission_row.get("race", "Unknown"),
        "hospital_expire_flag": admission_row.get("hospital_expire_flag", 0),
    }

    return hadm_id, decision_time, admission_context


def get_patient_medication_history(
    medications_df: pl.DataFrame,
    patient_id: int,
    hadm_id: int,
    decision_time,
    limit: int = 20
) -> str:
    """Get formatted medication history for a patient BEFORE decision time within specific admission."""
    patient_meds = (
        medications_df
        .filter(pl.col("subject_id") == patient_id)
        .filter(pl.col("hadm_id") == hadm_id)
        .filter(pl.col("drug").is_not_null())
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
        .filter(pl.col("charttime") < decision_time)
        .select(["drug", "charttime"])
        .sort("charttime", descending=False)  # Chronological order
        .tail(limit)  # Get most recent before decision time
    )

    if len(patient_meds) == 0:
        return "  No medication history available"

    med_lines = []
    for row in patient_meds.iter_rows(named=True):
        med_lines.append(f"  - {row['drug']} (administered: {row['charttime']})")

    return "\n".join(med_lines)


def get_patient_lab_history(
    labs_df: pl.DataFrame,
    patient_id: int,
    hadm_id: int,
    decision_time,
    limit: int = 20
) -> str:
    """Get formatted lab history for a patient BEFORE decision time within specific admission."""
    patient_labs = (
        labs_df
        .filter(pl.col("subject_id") == patient_id)
        .filter(pl.col("hadm_id") == hadm_id)
        .with_columns(
            pl.col("charttime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
        )
        .filter(pl.col("charttime") < decision_time)
        .select(["lab_test", "valuenum", "valueuom", "charttime"])
        .sort("charttime", descending=False)  # Chronological order
        .tail(limit)  # Get most recent before decision time
    )

    if len(patient_labs) == 0:
        return "  No lab history available"

    lab_lines = []
    for row in patient_labs.iter_rows(named=True):
        lab_lines.append(
            f"  - {row['lab_test']}: {row['valuenum']} {row['valueuom']} (date: {row['charttime']})"
        )

    return "\n".join(lab_lines)


def generate_hypothesis(
    client: OpenRouterChatClient | SGLangChatClient | NvidiaChatClient,
    model: str,
    prompt_template: str,
    patient_id: int,
    sex: str,
    age: int,
    medication_history: str,
    lab_history: str,
    admission_context: dict,
    temperature: float = 0.7,
) -> str | None:
    """Generate treatment hypothesis using LLM."""

    prompt = prompt_template.format(
        patient_id=patient_id,
        sex=sex,
        age=age,
        race=admission_context["race"],
        marital_status=admission_context["marital_status"],
        language=admission_context["language"],
        insurance=admission_context["insurance"],
        admission_type=admission_context["admission_type"],
        admission_location=admission_context["admission_location"],
        medication_history=medication_history,
        lab_history=lab_history,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.create_completion(
            model=model, messages=messages, temperature=temperature
        )
        return response
    except Exception as e:
        print(f"Error generating hypothesis: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate treatment hypotheses from MIMIC-IV patient histories"
    )
    parser.add_argument("--server", type=str, default="nvidia",
                       help="LLM backend: nvidia, openrouter, or sglang")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-v3.1",
                       help="Model name to use for hypothesis generation")
    parser.add_argument("--data-dir", type=str, default="datasets/MIMIC/processed",
                       help="Directory containing MIMIC-IV parquet files")
    parser.add_argument("--output", type=str, default="experiments/input/hypothesis_mimic.json",
                       help="Output path for hypothesis JSON")
    parser.add_argument("--n-hypotheses", type=int, default=100,
                       help="Number of hypotheses to generate")
    parser.add_argument("--med-history-limit", type=int, default=25,
                       help="Number of medications to include in history")
    parser.add_argument("--lab-history-limit", type=int, default=25,
                       help="Number of lab results to include in history")
    parser.add_argument("--prompt-template", type=str,
                       default="prompts/experiments/mimic_history_hypothesis.yaml",
                       help="Path to prompt template YAML")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for patient sampling")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.random_seed)

    # Setup LLM client
    print(f"Setting up {args.server} client with model {args.model}")
    client = setup_client(args.server)

    # Load prompt template
    print(f"Loading prompt template from {args.prompt_template}")
    prompt_template = load_prompt_template(args.prompt_template)

    # Load MIMIC-IV data
    data_dir = Path(args.data_dir)
    print(f"Loading MIMIC-IV data from {data_dir}")

    patients_df = pl.read_parquet(data_dir / "patients.parquet")
    admissions_df = pl.read_parquet(data_dir / "admissions.parquet")
    medications_df = pl.read_parquet(data_dir / "medications.parquet")
    labs_df = pl.read_parquet(data_dir / "labs.parquet")

    print(f"Total patients: {len(patients_df)}")
    print(f"Total admissions: {len(admissions_df)}")

    # Get patients who have both medication and lab records
    patients_with_meds = medications_df.select("subject_id").unique()
    patients_with_labs = labs_df.select("subject_id").unique()

    valid_patients = (
        patients_df
        .join(patients_with_meds, on="subject_id", how="inner")
        .join(patients_with_labs, on="subject_id", how="inner")
    )

    print(f"Patients with both medication and lab records: {len(valid_patients)}")

    # Sample patients
    if len(valid_patients) > args.n_hypotheses:
        sampled_patients = valid_patients.sample(n=args.n_hypotheses, seed=args.random_seed)
    else:
        sampled_patients = valid_patients

    print(f"Processing {len(sampled_patients)} patients")

    hypotheses = []
    successful = 0

    for row in tqdm(sampled_patients.iter_rows(named=True), total=len(sampled_patients)):
        try:
            patient_id = row["subject_id"]
            sex = row["sex"]
            age = row["anchor_age"]

            print(f"\nProcessing patient {patient_id}")
            print(f"Demographics: {sex}, {age} years old")

            # Select a specific admission and decision timepoint
            hadm_id, decision_time, admission_context = select_patient_admission(
                medications_df, labs_df, admissions_df, patient_id
            )

            if hadm_id is None or decision_time is None or admission_context is None:
                print(f"  No valid admission found for patient {patient_id}")
                continue

            print(f"  Selected admission: {hadm_id}")
            print(f"  Decision timepoint: {decision_time}")
            print(f"  Admission type: {admission_context['admission_type']}")
            print(f"  Race: {admission_context['race']}")

            # Get patient's medication history BEFORE decision time
            medication_history = get_patient_medication_history(
                medications_df, patient_id, hadm_id, decision_time,
                limit=args.med_history_limit
            )

            # Get patient's lab history BEFORE decision time
            lab_history = get_patient_lab_history(
                labs_df, patient_id, hadm_id, decision_time,
                limit=args.lab_history_limit
            )

            print(f"  Medication history: {len(medication_history.split(chr(10)))} entries")
            print(f"  Lab history: {len(lab_history.split(chr(10)))} entries")

            # Skip if insufficient historical data
            if "No medication history" in medication_history and "No lab history" in lab_history:
                print(f"  Insufficient historical data before decision point")
                continue

            # Generate hypothesis
            print("  Generating hypothesis...")
            hypothesis = generate_hypothesis(
                client=client,
                model=args.model,
                prompt_template=prompt_template,
                patient_id=patient_id,
                sex=sex,
                age=age,
                medication_history=medication_history,
                lab_history=lab_history,
                admission_context=admission_context,
            )

            if not hypothesis:
                print("  Failed to generate hypothesis")
                continue

            print(f"  Generated hypothesis: {hypothesis[:100]}...")

            # Create hypothesis entry
            hypotheses.append({
                "patient_id": patient_id,
                "hadm_id": hadm_id,
                "decision_time": str(decision_time),
                "sex": sex,
                "age": age,
                "race": admission_context["race"],
                "marital_status": admission_context["marital_status"],
                "language": admission_context["language"],
                "insurance": admission_context["insurance"],
                "admission_type": admission_context["admission_type"],
                "admission_location": admission_context["admission_location"],
                "discharge_location": admission_context["discharge_location"],
                "hospital_expire_flag": admission_context["hospital_expire_flag"],
                "medication_history": medication_history,
                "lab_history": lab_history,
                "hypothesis": hypothesis,
            })

            successful += 1
            print(f"Successfully processed {successful}/{args.n_hypotheses} patients")

            if successful >= args.n_hypotheses:
                break

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(hypotheses, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Successfully generated {len(hypotheses)} hypotheses")
    print(f"Saved to {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
