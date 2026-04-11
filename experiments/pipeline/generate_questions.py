#!/usr/bin/env python3
import argparse
import json
import pprint
import re
from pathlib import Path


# ── USMLE ──────────────────────────────────────────────────────────────────

def _extract_metamap_phrases(text: str) -> list[str]:
    phrases = []
    for sentence in re.split(r'[.?;]', text):
        for part in re.split(r',|\(|\)', sentence):
            part = part.strip()
            if part and len(part.split()) <= 5:
                phrases.append(part)
            elif part:
                for word in part.split():
                    if word.lower() not in {'the','a','an','of','to','in','is','are','was','were'} and len(word) > 2:
                        phrases.append(word)
    return phrases[:50]


def _convert_usmle_question(question: dict, question_id: str) -> dict | None:
    options = {opt['label']: opt['text'] for opt in question['options']}
    answer_idx = question['answer']
    if answer_idx is None:
        return None
    try:
        answer_text = next(opt['text'] for opt in question['options'] if opt['label'] == answer_idx)
    except StopIteration:
        print(f"Warning: answer '{answer_idx}' not found for {question_id}")
        return None
    exam = question['exam']
    meta_info = 'step1' if 'Step 1' in exam else 'step2' if 'Step 2' in exam else 'step3' if 'Step 3' in exam else 'unknown'
    return {
        'id': question_id,
        'question': question['stem'],
        'options': options,
        'answer': answer_text,
        'answer_idx': answer_idx,
        'meta_info': meta_info,
        'metamap_phrases': _extract_metamap_phrases(question['stem']),
    }


def generate_usmle(datasets_dir: str, output_file: str, limit: int | None = None):
    datasets_path = Path(datasets_dir)
    step_files = ['Step1_questions_parsed.json', 'Step2_CK_questions_parsed.json', 'Step3_questions_parsed.json']
    all_questions = []
    counter = 0
    for step_file in step_files:
        fp = datasets_path / step_file
        if not fp.exists():
            print(f"Warning: {fp} not found, skipping")
            continue
        questions = json.loads(fp.read_text(encoding='utf-8'))
        if limit:
            questions = questions[:limit]
        ok = skip = 0
        for q in questions:
            converted = _convert_usmle_question(q, f"usmle_sample_{counter}")
            if converted:
                all_questions.append(converted)
                ok += 1
            else:
                skip += 1
            counter += 1
        print(f"{step_file}: {ok} converted, {skip} skipped")
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_questions, indent=2, ensure_ascii=False))
    print(f"\nTotal: {len(all_questions)} questions → {out.absolute()}")
    # Sanity check
    issues = sum(
        1 for q in all_questions
        if q['answer'] not in q['options'].values()
        or q['answer_idx'] not in q['options']
        or q['options'].get(q['answer_idx']) != q['answer']
    )
    print("✓ Sanity check passed" if issues == 0 else f"✗ {issues} issue(s) found")


# ── MIMIC ───────────────────────────────────────────────────────────────────

def _fmt_patient(ep): return f"Age: {ep['patient']['anchor_age']} years\nSex: {ep['patient']['sex']}"
def _fmt_admission(ep):
    a = ep['admission']
    return f"Admission Type: {a['admission_type']}\nLocation: {a['admission_location']}\nTime: {a['admittime']}"

def _primary_diagnosis(ep):
    for d in ep.get('diagnoses_icd', []):
        if d['seq_num'] == 1:
            return f"{d['long_title']} (ICD-{d['icd_version']}: {d['icd_code']})"
    return "No primary diagnosis recorded"

def _fmt_labs(ep):
    labs = ep.get('labevents', [])
    if not labs:
        return "No lab results available"
    by_cat = {}
    for lab in labs:
        by_cat.setdefault(lab.get('category', 'Other'), []).append(lab)
    lines = []
    for cat in sorted(by_cat):
        abnormal = [l for l in by_cat[cat] if l.get('flag') == 'abnormal']
        normal = [l for l in by_cat[cat] if l.get('flag') != 'abnormal']
        if abnormal or normal:
            lines.append(f"\n{cat}:")
            for l in abnormal[:10]:
                v, u, t = l.get('value', l.get('valuenum', 'N/A')), l.get('valueuom', ''), l.get('charttime', '?')
                ref = f" (ref: {l['ref_range_lower']}-{l['ref_range_upper']} {u})" if l.get('ref_range_lower') and l.get('ref_range_upper') else ""
                lines.append(f"  - {l['label']}: {v} {u}{ref} [ABNORMAL] (at {t})")
            for l in normal[:5]:
                v, u, t = l.get('value', l.get('valuenum', 'N/A')), l.get('valueuom', ''), l.get('charttime', '?')
                if v and v != 'N/A':
                    lines.append(f"  - {l['label']}: {v} {u} (at {t})")
    return "\n".join(lines) if lines else "No lab results available"

def _fmt_micro(ep):
    events = ep.get('microbiologyevents', [])
    if not events:
        return "No microbiology results available"
    lines = []
    for e in events:
        info = f"- {e.get('test_name','?')}"
        if e.get('spec_type_desc'): info += f" ({e['spec_type_desc']})"
        info += f" - {e.get('charttime','?')}"
        if e.get('org_name'): info += f"\n  Organism: {e['org_name']}"
        if e.get('interpretation'): info += f"\n  Result: {e['interpretation']}"
        lines.append(info)
    return "\n".join(lines)

def _generate_context(ep):
    p, a = ep['patient'], ep['admission']
    sex = "male" if p['sex'] == 'M' else "female"
    parts = [f"admitted via {a['admission_location'].lower()}", f"admission type: {a['admission_type'].lower()}", f"admitted on {a['admittime']}"]
    if a.get('insurance'): parts.append(f"insurance: {a['insurance']}")
    if a.get('marital_status'): parts.append(f"marital status: {a['marital_status'].lower()}")
    return f"A {p['anchor_age']}-year-old {sex} patient, {', '.join(parts)}."

def _generate_clinical_question(ep):
    return "\n".join([
        "# Clinical Case", "", "## Patient Information", _fmt_patient(ep), "",
        "## Admission Details", _fmt_admission(ep), "",
        "## Primary Diagnosis", _primary_diagnosis(ep), "",
        "## Laboratory Results", _fmt_labs(ep), "",
        "## Microbiology Results", _fmt_micro(ep), "",
    ])

def _fmt_medications(ep):
    seen, descs = set(), []
    for med in ep.get('medication_admin', []):
        name = med.get('medication', '')
        if name and 'Flush' not in name and name not in seen:
            parts = [name]
            dose = f"{med.get('dose_given','')} {med.get('dose_given_unit','')}".strip()
            if dose: parts.append(dose)
            if med.get('route'): parts.append(f"via {med['route']}")
            at = med.get('administration_type')
            if at and at not in {'Standard Medication', 'Standard Maintenance Medication'}: parts.append(f"({at})")
            descs.append(" ".join(parts))
            seen.add(name)
    return ("Medications: " + "; ".join(descs) + ".") if descs else "No medications administered."

def _fmt_procedures(ep):
    procs = ep.get('procedures_icd', [])
    if not procs: return "No procedures performed."
    return "Procedures: " + "; ".join(f"{p['long_title']}" + (f" ({p['chartdate']})" if p.get('chartdate') else "") for p in procs) + "."

def _fmt_diagnoses(ep):
    diags = ep.get('diagnoses_icd', [])
    if not diags: return "No diagnoses recorded."
    parts = []
    for d in diags:
        prefix = "Primary:" if d['seq_num'] == 1 else f"{d['seq_num']}."
        parts.append(f"{prefix} {d['long_title']} (ICD-{d['icd_version']}: {d['icd_code']})")
    return "Diagnoses: " + " ".join(parts)


def generate_mimic(input_file: str, output_file: str):
    print(f"Loading episodes from {input_file}...")
    episodes = json.loads(Path(input_file).read_text())
    print(f"Processing {len(episodes)} episodes...")
    dataset = []
    for i, ep in enumerate(episodes):
        dataset.append({
            'id': f"{ep['hadm_id']}_{ep['subject_id']}",
            'hadm_id': ep['hadm_id'],
            'subject_id': ep['subject_id'],
            'context': _generate_context(ep),
            'question': _generate_clinical_question(ep),
            'ground_truth': {
                'medications': _fmt_medications(ep),
                'procedures': _fmt_procedures(ep),
                'diagnoses': _fmt_diagnoses(ep),
            },
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(episodes)} processed...")
    Path(output_file).write_text(json.dumps(dataset, indent=2))
    print(f"Done! {len(dataset)} questions → {output_file}")
    if dataset:
        print("\nSample context:"); print(dataset[0]['context'])
        print("\nSample ground truth:"); pprint.pprint(dataset[0]['ground_truth'])


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='dataset', required=True)

    p_usmle = sub.add_parser('usmle', help='Generate USMLE sample questions')
    p_usmle.add_argument('--datasets-dir', default='../datasets/usmle_sample')
    p_usmle.add_argument('--output', default='questions/clinical_questions_usmle_sample.json')
    p_usmle.add_argument('--limit', type=int, default=None, help='Questions per step (for testing)')

    p_mimic = sub.add_parser('mimic', help='Generate MIMIC clinical questions')
    p_mimic.add_argument('--input', default='../datasets/mimiciv_demo/test/episodes.json')
    p_mimic.add_argument('--output', default='../datasets/mimiciv_demo/test/clinical_questions.json')

    args = parser.parse_args()
    if args.dataset == 'usmle':
        generate_usmle(args.datasets_dir, args.output, args.limit)
    else:
        generate_mimic(args.input, args.output)


if __name__ == '__main__':
    main()
