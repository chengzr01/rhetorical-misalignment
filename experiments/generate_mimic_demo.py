#!/usr/bin/env python3
"""
Generate clinical decision-making questions from MIMIC-IV episodes data.
"""

import argparse
import json
from typing import Dict


def format_patient_info(episode: Dict) -> str:
    """Format patient demographic information."""
    patient = episode['patient']
    info = [
        f"Age: {patient['anchor_age']} years",
        f"Sex: {patient['sex']}"
    ]
    return "\n".join(info)


def format_admission_info(episode: Dict) -> str:
    """Format admission information."""
    admission = episode['admission']
    info = [
        f"Admission Type: {admission['admission_type']}",
        f"Admission Location: {admission['admission_location']}",
        f"Admission Time: {admission['admittime']}"
    ]
    return "\n".join(info)


def get_primary_diagnosis(episode: Dict) -> str:
    """Get the primary diagnosis (seq_num = 1)."""
    diagnoses = episode.get('diagnoses_icd', [])
    for diag in diagnoses:
        if diag['seq_num'] == 1:
            return f"{diag['long_title']} (ICD-{diag['icd_version']}: {diag['icd_code']})"
    return "No primary diagnosis recorded"


def format_lab_results(episode: Dict) -> str:
    """Format lab results, emphasizing abnormal values."""
    labevents = episode.get('labevents', [])
    if not labevents:
        return "No lab results available"

    # Group lab events by category
    labs_by_category = {}
    for lab in labevents:
        category = lab.get('category', 'Other')
        if category not in labs_by_category:
            labs_by_category[category] = []
        labs_by_category[category].append(lab)

    # Format labs, prioritizing abnormal values
    lines = []
    for category in sorted(labs_by_category.keys()):
        category_labs = labs_by_category[category]
        # Show abnormal labs first
        abnormal_labs = [lab for lab in category_labs if lab.get('flag') == 'abnormal']
        normal_labs = [lab for lab in category_labs if lab.get('flag') != 'abnormal']

        if abnormal_labs or normal_labs:
            lines.append(f"\n{category}:")

            # Show abnormal values with flag
            for lab in abnormal_labs[:10]:  # Limit to first 10 abnormal per category
                value = lab.get('value', lab.get('valuenum', 'N/A'))
                unit = lab.get('valueuom', '')
                charttime = lab.get('charttime', 'Unknown time')
                ref_range = ""
                if lab.get('ref_range_lower') and lab.get('ref_range_upper'):
                    ref_range = f" (ref: {lab['ref_range_lower']}-{lab['ref_range_upper']} {unit})"
                lines.append(f"  - {lab['label']}: {value} {unit}{ref_range} [ABNORMAL] (at {charttime})")

            # Show some normal values for context (limit to 5)
            for lab in normal_labs[:5]:
                value = lab.get('value', lab.get('valuenum', 'N/A'))
                unit = lab.get('valueuom', '')
                charttime = lab.get('charttime', 'Unknown time')
                if value and value != 'N/A':
                    lines.append(f"  - {lab['label']}: {value} {unit} (at {charttime})")

    return "\n".join(lines) if lines else "No lab results available"


def format_microbiology_results(episode: Dict) -> str:
    """Format microbiology test results."""
    micro_events = episode.get('microbiologyevents', [])
    if not micro_events:
        return "No microbiology results available"

    lines = []
    for event in micro_events:
        charttime = event.get('charttime', 'Unknown time')
        test_info = f"- {event.get('test_name', 'Unknown test')}"
        if event.get('spec_type_desc'):
            test_info += f" ({event['spec_type_desc']})"
        test_info += f" - collected at {charttime}"

        if event.get('org_name'):
            test_info += f"\n  Organism: {event['org_name']}"

        if event.get('interpretation'):
            test_info += f"\n  Result: {event['interpretation']}"

        if event.get('comments'):
            test_info += f"\n  Comments: {event['comments'].strip()}"

        lines.append(test_info)

    return "\n".join(lines) if lines else "No microbiology results available"


def generate_context(episode: Dict) -> str:
    """Generate context with just patient and admission information (no clinical data)."""
    patient = episode['patient']
    admission = episode['admission']

    # Format patient demographics
    sex_text = "male" if patient['sex'] == 'M' else "female"
    patient_text = f"A {patient['anchor_age']}-year-old {sex_text} patient"

    # Format admission details
    admission_parts = []
    admission_parts.append(f"admitted via {admission['admission_location'].lower()}")
    admission_parts.append(f"admission type: {admission['admission_type'].lower()}")
    admission_parts.append(f"admitted on {admission['admittime']}")

    if admission.get('insurance'):
        admission_parts.append(f"insurance: {admission['insurance']}")

    if admission.get('marital_status'):
        admission_parts.append(f"marital status: {admission['marital_status'].lower()}")

    # Combine into paragraph
    context = f"{patient_text}, {', '.join(admission_parts)}."

    return context


def generate_clinical_question(episode: Dict) -> str:
    """Generate a clinical question prompt for the episode."""
    sections = [
        "# Clinical Case",
        "",
        "## Patient Information",
        format_patient_info(episode),
        "",
        "## Admission Details",
        format_admission_info(episode),
        "",
        "## Primary Diagnosis",
        get_primary_diagnosis(episode),
        "",
        "## Laboratory Results",
        format_lab_results(episode),
        "",
        "## Microbiology Results",
        format_microbiology_results(episode),
        "",
    ]

    return "\n".join(sections)


def format_medications_text(episode: Dict) -> str:
    """Format medications as a paragraph."""
    meds_admin = episode.get('medication_admin', [])
    seen_meds = set()
    med_descriptions = []

    for med in meds_admin:
        med_name = med.get('medication', '')
        # Filter out basic maintenance items
        if med_name and 'Flush' not in med_name and med_name not in seen_meds:
            desc_parts = [med_name]

            dose = f"{med.get('dose_given', '')} {med.get('dose_given_unit', '')}".strip()
            if dose:
                desc_parts.append(f"{dose}")

            route = med.get('route')
            if route:
                desc_parts.append(f"via {route}")

            admin_type = med.get('administration_type')
            if admin_type and admin_type not in ['Standard Medication', 'Standard Maintenance Medication']:
                desc_parts.append(f"({admin_type})")

            med_descriptions.append(" ".join(desc_parts))
            seen_meds.add(med_name)

    if not med_descriptions:
        return "No medications administered during this admission."

    return "Medications: " + "; ".join(med_descriptions) + "."


def format_procedures_text(episode: Dict) -> str:
    """Format procedures as a paragraph."""
    procedures = episode.get('procedures_icd', [])

    if not procedures:
        return "No procedures performed during this admission."

    proc_descriptions = []
    for proc in procedures:
        proc_desc = f"{proc['long_title']}"
        if proc.get('chartdate'):
            proc_desc += f" (performed on {proc['chartdate']})"
        proc_descriptions.append(proc_desc)

    return "Procedures: " + "; ".join(proc_descriptions) + "."


def format_diagnoses_text(episode: Dict) -> str:
    """Format all diagnoses as a paragraph with sequence numbers."""
    diagnoses = episode.get('diagnoses_icd', [])

    if not diagnoses:
        return "No diagnoses recorded."

    diag_descriptions = []
    for diag in diagnoses:
        seq = diag['seq_num']
        title = diag['long_title']
        icd = diag['icd_code']
        prefix = "Primary:" if seq == 1 else f"{seq}."
        diag_descriptions.append(f"{prefix} {title} (ICD-{diag['icd_version']}: {icd})")

    return "Diagnoses: " + " ".join(diag_descriptions)


def extract_ground_truth_dict(episode: Dict) -> dict:
    """Extract ground truth as a dictionary with medications, procedures, and diagnoses."""
    medications = format_medications_text(episode)
    procedures = format_procedures_text(episode)
    diagnoses = format_diagnoses_text(episode)
    return {
        "medications": medications,
        "procedures": procedures,
        "diagnoses": diagnoses
    }


def generate_questions_dataset(input_file: str, output_file: str):
    """Generate clinical questions dataset from episodes data."""
    print(f"Loading episodes from {input_file}...")
    with open(input_file, 'r') as f:
        episodes = json.load(f)

    print(f"Processing {len(episodes)} episodes...")

    dataset = []
    for i, episode in enumerate(episodes):
        hadm_id = episode['hadm_id']
        subject_id = episode['subject_id']

        # Generate context (patient and admission only)
        context = generate_context(episode)

        # Generate question
        question = generate_clinical_question(episode)

        # Extract ground truth as dict
        ground_truth = extract_ground_truth_dict(episode)

        # Create dataset entry
        entry = {
            'id': f"{hadm_id}_{subject_id}",
            'hadm_id': hadm_id,
            'subject_id': subject_id,
            'context': context,
            'question': question,
            'ground_truth': ground_truth
        }

        dataset.append(entry)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(episodes)} episodes...")

    print(f"\nSaving {len(dataset)} questions to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Done! Generated {len(dataset)} clinical questions.")

    # Print a sample
    print("\n" + "="*80)
    print("SAMPLE CONTEXT:")
    print("="*80)
    print(dataset[0]['context'])
    print("\n" + "="*80)
    print("SAMPLE QUESTION:")
    print("="*80)
    print(dataset[0]['question'])
    print("\n" + "="*80)
    print("SAMPLE GROUND TRUTH:")
    print("="*80)
    # Pretty print the sample ground_truth dictionary
    import pprint
    pprint.pprint(dataset[0]['ground_truth'])


def main():
    parser = argparse.ArgumentParser(
        description='Generate clinical decision-making questions from MIMIC-IV episodes data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='../datasets/mimiciv_demo/test/episodes.json',
        help='Path to input episodes JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='../datasets/mimiciv_demo/test/clinical_questions.json',
        help='Path to output clinical questions JSON file'
    )

    args = parser.parse_args()

    generate_questions_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
