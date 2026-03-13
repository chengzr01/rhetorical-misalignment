#!/usr/bin/env python3
"""
Annotation coverage and case distribution reports for the USMLE Sample dataset.

Modes:
  by-case-type  - Annotation coverage broken down by Bayesian/Behavioral
                  decision type (accept/reject) for each model.
  full-report   - Per-case annotation status for all models, including
                  unannotated cases (sourced from principal files).
  case-counts   - Comprehensive report: original vs annotated cases, step
                  distribution, persuasion breakdown, and cross-model case
                  overlap.

Usage:
  python coverage_report.py --mode by-case-type
  python coverage_report.py --mode full-report
  python coverage_report.py --mode case-counts --persuasion-file persuasion_examples.json
  python coverage_report.py --cases-dir /path/to/cases --results-dir /path/to/results
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_CASES_DIR = '../../../experiments/principals/usmle_sample'
DEFAULT_RESULTS_DIR = '../../results/usmle_sample'

# Unified model configuration used across all modes
MODELS = [
    {'key': 'llama_small', 'name': 'Llama-3.1-8B',
     'patterns': ['llama-3.1-8b-instruct']},
    {'key': 'llama', 'name': 'Llama-3.3-70B',
     'patterns': ['llama-3.3', '70b-instruct']},
    {'key': 'llama_large', 'name': 'Llama-3.1-405B',
     'patterns': ['405b', 'llama-3.1-405b']},
    {'key': 'llama_dpo', 'name': 'Llama-3.1-Tulu-3-8B-DPO',
     'patterns': ['tulu-3-8b-dpo', '8b-dpo']},
    {'key': 'llama_sft', 'name': 'Llama-3.1-Tulu-3-8B-SFT',
     'patterns': ['tulu-3-8b-sft', '8b-sft']},
    {'key': 'deepseek', 'name': 'DeepSeek-V3',
     'patterns': ['deepseek']},
]


# ─── Shared utilities ─────────────────────────────────────────────────────────

def identify_model(agent_model_str: str):
    """Map a raw agent_model string to a model key from MODELS, or None."""
    if not agent_model_str:
        return None
    lower = agent_model_str.lower()
    for m in MODELS:
        if any(p.lower() in lower for p in m['patterns']):
            return m['key']
    return None


def load_principal_file(model_key: str, cases_dir: str):
    """Load the principal JSON file for a given model key."""
    filepath = os.path.join(cases_dir, f'principal_{model_key}.json')
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_annotations_by_case_and_model(results_dir: str) -> dict:
    """
    Load all annotation files and index them.
    Returns {case_id: {model_key: count}} dict.
    """
    result: dict = defaultdict(lambda: defaultdict(int))
    if not os.path.exists(results_dir):
        return result
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(results_dir, filename)) as f:
                ann = json.load(f)
            case_id = ann.get('case_id')
            model_key = identify_model(ann.get('agent_model', ''))
            if case_id and model_key:
                result[case_id][model_key] += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return result


def load_annotations_with_full_data(results_dir: str):
    """
    Load all annotations and return two structures:
    - annotations_by_case: {case_id: {model_key: [annotation, ...]}}
    - cases_by_model: {model_key: set(case_ids)}
    """
    ann_by_case: dict = defaultdict(lambda: defaultdict(list))
    cases_by_model: dict = defaultdict(set)
    if not os.path.exists(results_dir):
        return ann_by_case, cases_by_model
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(results_dir, filename)) as f:
                ann = json.load(f)
            case_id = ann.get('case_id')
            model_key = identify_model(ann.get('agent_model', ''))
            if case_id and model_key:
                ann_by_case[case_id][model_key].append(ann)
                cases_by_model[model_key].add(case_id)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return ann_by_case, cases_by_model


def _coverage_stats(case_ids: list, ann_counts: dict, model_key: str) -> dict:
    """Compute annotation coverage statistics for a list of cases."""
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0
    for cid in case_ids:
        cnt = ann_counts.get(cid, {}).get(model_key, 0)
        total += cnt
        dist[min(cnt, 3)] += 1
    return {
        'total_cases': len(case_ids),
        'distribution': dist,
        'total_annotations': total,
        'target_annotations': len(case_ids) * 3,
        'progress_pct': total / (len(case_ids) * 3) * 100 if case_ids else 0,
    }


def _print_coverage(stats: dict) -> None:
    d = stats['distribution']
    total = stats['total_cases']
    print(f"    ✗ Not annotated yet:        {d[0]:3d} cases ({d[0]/total*100:5.1f}%)")
    print(f"    ◐ Annotated once:           {d[1]:3d} cases ({d[1]/total*100:5.1f}%)")
    print(f"    ◑ Annotated twice:          {d[2]:3d} cases ({d[2]/total*100:5.1f}%)")
    print(f"    ✓ Goal reached (3+ annot.): {d[3]:3d} cases ({d[3]/total*100:5.1f}%)")
    print(f"    Progress: {stats['total_annotations']}/{stats['target_annotations']} "
          f"({stats['progress_pct']:.1f}%)")


# ─── Mode: by-case-type ───────────────────────────────────────────────────────

def run_by_case_type(cases_dir: str, results_dir: str) -> None:
    """Report annotation coverage broken down by Bayesian/Behavioral decision type."""
    print("\n" + "=" * 90)
    print("ANNOTATION STATUS BY CASE TYPE AND MODEL")
    print("USMLE Sample Dataset")
    print("=" * 90)
    print("\nDecision type combinations: accept_accept | accept_reject | reject_accept | reject_reject")

    ann_counts = load_annotations_by_case_and_model(results_dir)
    all_results = {}

    CATEGORY_NAMES = {
        'accept_accept': 'Both Accept (Bayesian ✓, Behavioral ✓)',
        'accept_reject': 'Bayesian Accepts, Behavioral Rejects',
        'reject_accept': 'Bayesian Rejects, Behavioral Accepts',
        'reject_reject': 'Both Reject (Bayesian ✗, Behavioral ✗)',
    }

    for model_info in MODELS:
        key, name = model_info['key'], model_info['name']
        principal = load_principal_file(key, cases_dir)
        if not principal:
            print(f"\n  ⚠  No principal file for {key}")
            continue

        print(f"\n{'=' * 90}")
        print(f"MODEL: {name} ({key})")
        print(f"{'=' * 90}")

        # Categorise cases by decision type
        categories: dict = defaultdict(list)
        for case in principal.get('cases', []):
            cat = f"{case.get('bayesian_decision', 'unknown')}_{case.get('behavioral_decision', 'unknown')}"
            categories[cat].append(case['case_id'])

        total = sum(len(v) for v in categories.values())
        print(f"\nTotal manipulative cases: {total}")

        model_results = {}
        for cat in sorted(categories):
            cids = categories[cat]
            stats = _coverage_stats(cids, ann_counts, key)
            print(f"\n  Decision type: {CATEGORY_NAMES.get(cat, cat)}")
            print(f"  Cases: {len(cids)}")
            _print_coverage(stats)
            model_results[cat] = stats

        all_results[key] = model_results

    # Summary table
    print(f"\n\n{'=' * 90}")
    print("SUMMARY TABLE - ANNOTATION STATUS BY CASE TYPE AND MODEL")
    print(f"{'=' * 90}")
    for cat, cat_name in CATEGORY_NAMES.items():
        print(f"\n  DECISION TYPE: {cat_name}")
        print(f"  {'Model':<30} {'Cases':>6} {'0-ann':>8} {'1-ann':>8} {'2-ann':>8} {'3+ann':>8} {'Progress':>10}")
        print(f"  {'-' * 85}")
        for model_info in MODELS:
            k = model_info['key']
            if k not in all_results or cat not in all_results[k]:
                continue
            s = all_results[k][cat]
            d, n = s['distribution'], s['total_cases']
            if n == 0:
                continue
            print(f"  {model_info['name']:<30} {n:>6} "
                  f"{d[0]:>5}({d[0]/n*100:4.0f}%) "
                  f"{d[1]:>5}({d[1]/n*100:4.0f}%) "
                  f"{d[2]:>5}({d[2]/n*100:4.0f}%) "
                  f"{d[3]:>5}({d[3]/n*100:4.0f}%) "
                  f"{s['progress_pct']:>9.1f}%")

    print("\n" + "=" * 90)
    print("Legend: ✗=0  ◐=1  ◑=2  ✓=3+ annotations  |  Goal: 3 annotations per case")


# ─── Mode: full-report ────────────────────────────────────────────────────────

def run_full_report(cases_dir: str, results_dir: str) -> None:
    """Detailed per-model, per-case annotation status report."""
    print("\n" + "=" * 100)
    print("COMPLETE ANNOTATION STATUS REPORT")
    print("USMLE Sample Dataset - All Cases (including unannotated)")
    print("=" * 100)
    print("\nGoal: 3 annotations per case per model")

    ann_counts = load_annotations_by_case_and_model(results_dir)

    # Load case IDs from principal files
    case_ids_by_model = {}
    for model_info in MODELS:
        key = model_info['key']
        principal = load_principal_file(key, cases_dir)
        if principal and 'cases' in principal:
            case_ids_by_model[key] = sorted(c['case_id'] for c in principal['cases'])

    for model_info in MODELS:
        key, name = model_info['key'], model_info['name']
        case_ids = case_ids_by_model.get(key, [])
        if not case_ids:
            continue

        print(f"\n{'=' * 100}\nMODEL: {name} ({key})\n{'=' * 100}")
        print(f"\nTotal cases in dataset: {len(case_ids)}")

        per_case = {cid: ann_counts.get(cid, {}).get(key, 0) for cid in case_ids}
        dist: dict = defaultdict(int)
        for cnt in per_case.values():
            dist['3+' if cnt >= 3 else cnt] += 1
        total = sum(per_case.values())
        target = len(case_ids) * 3

        print(f"\nAnnotation Status:")
        print(f"  Not annotated (0)  : {dist[0]:4d} ({dist[0]/len(case_ids)*100:5.1f}%)")
        print(f"  Annotated once (1) : {dist[1]:4d} ({dist[1]/len(case_ids)*100:5.1f}%)")
        print(f"  Annotated twice (2): {dist[2]:4d} ({dist[2]/len(case_ids)*100:5.1f}%)")
        print(f"  Goal reached (3+)  : {dist['3+']:4d} ({dist.get('3+',0)/len(case_ids)*100:5.1f}%)")
        print(f"\nProgress: {total}/{target} ({total/target*100:.1f}%) | avg/case: {total/len(case_ids):.2f}")

        # Group by count for sample listings
        by_cnt: dict = defaultdict(list)
        for cid, cnt in per_case.items():
            by_cnt[min(cnt, 3)].append(cid)

        if by_cnt[0]:
            n0 = len(by_cnt[0])
            print(f"\n  🔴 {n0} cases with 0 annotations:")
            for i in range(0, min(30, n0), 5):
                print(f"    {', '.join(by_cnt[0][i:i+5])}")
            if n0 > 30:
                print(f"    ... and {n0 - 30} more")

        need_more = by_cnt[1] + by_cnt[2]
        if need_more:
            print(f"\n  ⚠  {len(need_more)} cases need more annotations (1 or 2 so far)")
            for i in range(min(20, len(need_more))):
                cid = need_more[i]
                print(f"    {cid} ({per_case[cid]} ann.)", end=('  ' if (i + 1) % 3 else '\n'))
            if len(need_more) % 3:
                print()
            if len(need_more) > 20:
                print(f"    ... and {len(need_more) - 20} more")

    # Summary table across all models
    print(f"\n\n{'=' * 100}")
    print("SUMMARY - ANNOTATION STATUS ACROSS ALL MODELS")
    print(f"{'=' * 100}")
    print(f"\n{'Model':<25} {'Total Cases':>12} {'0-ann':>10} {'1-ann':>10} "
          f"{'2-ann':>10} {'3+-ann':>10} {'Progress':>10}")
    print('-' * 100)
    for model_info in MODELS:
        key, name = model_info['key'], model_info['name']
        case_ids = case_ids_by_model.get(key, [])
        if not case_ids:
            continue
        dist2 = {0: 0, 1: 0, 2: 0, 3: 0}
        total2 = 0
        for cid in case_ids:
            cnt = ann_counts.get(cid, {}).get(key, 0)
            total2 += cnt
            dist2[min(cnt, 3)] += 1
        n2 = len(case_ids)
        pct = total2 / (n2 * 3) * 100 if n2 else 0
        print(f"{name:<25} {n2:>12} "
              f"{dist2[0]:>7}({dist2[0]/n2*100:4.0f}%) "
              f"{dist2[1]:>7}({dist2[1]/n2*100:4.0f}%) "
              f"{dist2[2]:>7}({dist2[2]/n2*100:4.0f}%) "
              f"{dist2[3]:>7}({dist2[3]/n2*100:4.0f}%) "
              f"{pct:>9.1f}%")
    print('=' * 100)


# ─── Mode: case-counts ────────────────────────────────────────────────────────

def _extract_step(case_id: str) -> str:
    """Map a case_id index to Step1/Step2/Step3 based on known index ranges."""
    try:
        idx = int(case_id.split('_')[-1])
        if idx <= 117:
            return 'Step1'
        if idx <= 251:
            return 'Step2'
        return 'Step3'
    except Exception:
        return 'Unknown'


def run_case_counts(results_dir: str, persuasion_file: str) -> None:
    """
    Comprehensive report: original vs annotated cases, step distribution,
    persuasion breakdown, and cross-model case overlap.
    """
    print("\n" + "=" * 100)
    print("COMPREHENSIVE CASE ANNOTATION ANALYSIS REPORT")
    print("Dataset: USMLE Sample")
    print("=" * 100)

    ann_by_case, cases_by_model = load_annotations_with_full_data(results_dir)
    all_models = sorted(cases_by_model.keys())

    # Section 1: Original vs annotated
    print("\n" + "=" * 100)
    print("SECTION 1: ANNOTATED CASES BY MODEL")
    print("=" * 100)
    print(f"\n{'Model':<20} {'Cases with annot.':>18} {'Avg annot./case':>18}")
    print("-" * 60)
    for model in all_models:
        n_cases = len(cases_by_model[model])
        n_anns = sum(len(ann_by_case[cid][model]) for cid in cases_by_model[model])
        avg = n_anns / n_cases if n_cases else 0
        print(f"{model:<20} {n_cases:>18} {avg:>18.2f}")

    # Section 2: Step distribution
    print("\n" + "=" * 100)
    print("SECTION 2: DISTRIBUTION OF CASES ACROSS USMLE STEPS")
    print("=" * 100)
    step_dist: dict = defaultdict(lambda: defaultdict(int))
    for model in all_models:
        for cid in cases_by_model[model]:
            step_dist[model][_extract_step(cid)] += 1

    steps = ['Step1', 'Step2', 'Step3']
    print(f"\n{'Model':<20} {'Step1':<15} {'Step2':<15} {'Step3':<15} {'Total':<10}")
    print("-" * 75)
    for model in all_models:
        total = sum(step_dist[model].values())
        row = f"{model:<20}"
        for s in steps:
            cnt = step_dist[model][s]
            pct = cnt / total * 100 if total else 0
            row += f"{cnt} ({pct:.1f}%)     "[:15]
        row += str(total)
        print(row)

    # Section 3: Persuasion distribution
    print("\n" + "=" * 100)
    print("SECTION 3: PERSUASION DISTRIBUTION (Harmful vs Helpful)")
    print("=" * 100)
    if persuasion_file and os.path.exists(persuasion_file):
        try:
            with open(persuasion_file) as f:
                persuasion_data = json.load(f)
            summary = persuasion_data.get('summary', {})
            harmful = summary.get('harmful_persuasion', {}).get('count', 0)
            helpful = summary.get('helpful_persuasion', {}).get('count', 0)
            total_p = harmful + helpful
            if total_p:
                print(f"\n  Total persuasion cases: {total_p}")
                print(f"    Harmful (correct→incorrect): {harmful} ({harmful/total_p*100:.1f}%)")
                print(f"    Helpful (incorrect→correct): {helpful} ({helpful/total_p*100:.1f}%)")
                print(f"    Net helpful: {helpful - harmful:+d}")

            # By model
            model_p: dict = defaultdict(lambda: {'harmful': 0, 'helpful': 0})
            cases_p = persuasion_data.get('cases', {})
            for ptype in ['harmful_persuasion', 'helpful_persuasion']:
                for case in cases_p.get(ptype, []):
                    mk = identify_model(case.get('model', '')) or 'unknown'
                    model_p[mk]['harmful' if ptype == 'harmful_persuasion' else 'helpful'] += 1

            print(f"\n  By model:")
            print(f"  {'Model':<20} {'Harmful':>10} {'Helpful':>10} {'Total':>10} {'Net':>10}")
            print("  " + "-" * 55)
            for mk in sorted(model_p):
                h = model_p[mk]['harmful']
                hp = model_p[mk]['helpful']
                print(f"  {mk:<20} {h:>10} {hp:>10} {h+hp:>10} {hp-h:>+10}")

            # By step
            step_p: dict = defaultdict(lambda: {'harmful': 0, 'helpful': 0})
            for ptype in ['harmful_persuasion', 'helpful_persuasion']:
                for case in cases_p.get(ptype, []):
                    s = _extract_step(case.get('case_id', ''))
                    step_p[s]['harmful' if ptype == 'harmful_persuasion' else 'helpful'] += 1

            print(f"\n  By USMLE step:")
            for s in ['Step1', 'Step2', 'Step3']:
                if s in step_p:
                    h, hp = step_p[s]['harmful'], step_p[s]['helpful']
                    print(f"    {s}: harmful={h}  helpful={hp}  net={hp-h:+d}")
        except Exception as e:
            print(f"  Could not load persuasion data: {e}")
    else:
        print(f"\n  Persuasion file not found ({persuasion_file}). "
              "Use --persuasion-file to specify its path.")

    # Section 4: Cross-model case overlap
    print("\n" + "=" * 100)
    print("SECTION 4: CASE OVERLAP ACROSS MODELS")
    print("=" * 100)
    print(f"\n  {'Model 1':<20} {'Model 2':<20} {'Overlap':>10} "
          f"{'Unique M1':>12} {'Unique M2':>12}")
    print("  " + "-" * 80)
    for i, m1 in enumerate(all_models):
        for m2 in all_models[i + 1:]:
            c1, c2 = cases_by_model[m1], cases_by_model[m2]
            ov = len(c1 & c2)
            print(f"  {m1:<20} {m2:<20} {ov:>10} {len(c1-c2):>12} {len(c2-c1):>12}")

    # Three-way overlap for the small 8B models if present
    target = ['llama_small', 'llama_dpo', 'llama_sft']
    if all(m in cases_by_model for m in target):
        s, d, f = (cases_by_model[m] for m in target)
        print(f"\n  Three-way analysis (llama_small / llama_dpo / llama_sft):")
        print(f"    Common to all three         : {len(s & d & f)}")
        print(f"    Common to small & dpo only  : {len((s & d) - f)}")
        print(f"    Common to small & sft only  : {len((s & f) - d)}")
        print(f"    Common to dpo & sft only    : {len((d & f) - s)}")
        print(f"    Unique to llama_small        : {len(s - d - f)}")
        print(f"    Unique to llama_dpo          : {len(d - s - f)}")
        print(f"    Unique to llama_sft          : {len(f - s - d)}")
        print(f"    Total unique across three    : {len(s | d | f)}")

    all_cases: set = set()
    for cases in cases_by_model.values():
        all_cases.update(cases)
    print(f"\n  Total unique cases across ALL models: {len(all_cases)}")
    model_freq: dict = defaultdict(int)
    for cid in all_cases:
        model_freq[cid] = sum(1 for m in cases_by_model.values() if cid in m)
    freq_dist: dict = defaultdict(int)
    for freq in model_freq.values():
        freq_dist[freq] += 1
    print("  Distribution by number of models covering the case:")
    for freq in sorted(freq_dist):
        n = freq_dist[freq]
        print(f"    In {freq} model(s): {n} cases ({n / len(all_cases) * 100:.1f}%)")

    # Section 5: Detailed per-model annotation statistics
    print("\n" + "=" * 100)
    print("SECTION 5: DETAILED ANNOTATION STATISTICS PER MODEL")
    print("=" * 100)
    total_anns = sum(
        len(ann_by_case[cid][m])
        for cid in ann_by_case
        for m in all_models
    )
    print(f"\n  Total unique annotated cases : {len(ann_by_case)}")
    print(f"  Total annotation files       : {total_anns}")
    print(f"\n  {'Model':<20} {'Annotations':>12} {'Cases':>8} {'Avg/case':>10}")
    print("  " + "-" * 55)
    for model in all_models:
        n_cases = len(cases_by_model[model])
        n_anns = sum(len(ann_by_case[cid][model]) for cid in cases_by_model[model])
        avg = n_anns / n_cases if n_cases else 0
        print(f"  {model:<20} {n_anns:>12} {n_cases:>8} {avg:>10.2f}")

    print("\n" + "=" * 100)
    print("END OF REPORT")
    print("=" * 100)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Annotation coverage and case distribution reports.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes:
  by-case-type  - Coverage broken down by Bayesian/Behavioral decision type
  full-report   - Per-case status for all models (including unannotated cases)
  case-counts   - Original vs annotated cases, step distribution, persuasion,
                  and cross-model overlap""",
    )
    parser.add_argument(
        '--mode', choices=['by-case-type', 'full-report', 'case-counts'],
        default='full-report',
        help='Report mode (default: full-report)',
    )
    parser.add_argument(
        '--cases-dir', default=DEFAULT_CASES_DIR,
        help=f'Path to cases directory (default: {DEFAULT_CASES_DIR})',
    )
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help=f'Path to annotation results directory (default: {DEFAULT_RESULTS_DIR})',
    )
    parser.add_argument(
        '--persuasion-file', default='persuasion_examples.json',
        help='Path to persuasion_examples.json for case-counts mode',
    )
    args = parser.parse_args()

    if args.mode == 'by-case-type':
        run_by_case_type(args.cases_dir, args.results_dir)
    elif args.mode == 'full-report':
        run_full_report(args.cases_dir, args.results_dir)
    elif args.mode == 'case-counts':
        run_case_counts(args.results_dir, args.persuasion_file)


if __name__ == '__main__':
    main()
