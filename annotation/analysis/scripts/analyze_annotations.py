#!/usr/bin/env python3
"""
Analyze annotation results from the USMLE Sample dataset.

Modes:
  stats   - Comprehensive statistics: belief changes, persuasion effects,
            answer correctness, highlights, time spent, and demographics.
  compare - Compare persuasion effectiveness across AI models, including
            detailed belief-change tables and conditional analyses.
  export  - Export all annotations to CSV for downstream analysis.

Usage:
  python analyze_annotations.py --mode stats
  python analyze_annotations.py --mode compare
  python analyze_annotations.py --mode export --output my_data.csv
  python analyze_annotations.py --results-dir /path/to/results --mode stats
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
from typing import Optional
from collections import defaultdict
from datetime import datetime


DEFAULT_RESULTS_DIR = '../../results/usmle_sample'


# ─── Shared utilities ─────────────────────────────────────────────────────────

def load_all_annotations(results_dir: str) -> list:
    """Load all annotation JSON files from the results directory."""
    if not os.path.exists(results_dir):
        return []
    annotations = []
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
        try:
            with open(os.path.join(results_dir, filename)) as f:
                annotations.append(json.load(f))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return annotations


def calculate_belief_change(before, after):
    """Return after - before, or None if either value is missing."""
    if before is None or after is None:
        return None
    return after - before


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + (z ** 2) / n
    center = phat + (z ** 2) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + (z ** 2) / (4 * n)) / n)
    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    return (lower, upper)


def _wald_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wald interval and standard error for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = successes / n
    se = math.sqrt(phat * (1 - phat) / n)
    lower = max(0.0, phat - z * se)
    upper = min(1.0, phat + z * se)
    return (lower, upper, se)


def _cohens_h(p1: float, p2: float) -> float:
    """Effect size for difference between two proportions."""
    eps = 1e-12
    p1 = min(max(p1, eps), 1 - eps)
    p2 = min(max(p2, eps), 1 - eps)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def _parse_years_of_practice(value) -> Optional[float]:
    """Extract a numeric years-of-practice value from free-form text."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _pearson_correlation(xs, ys):
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _section(title: str, width: int = 80) -> None:
    print(f"\n{'-' * width}\n{title}\n{'-' * width}")


def _header(title: str, width: int = 80) -> None:
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


# ─── Mode: stats ──────────────────────────────────────────────────────────────

def run_stats(annotations: list) -> None:
    """Print comprehensive statistics on the annotation dataset."""
    _header("ANNOTATION RESULTS ANALYSIS")

    n = len(annotations)
    unique_cases = {a.get('case_id') for a in annotations}
    unique_annotators = {a.get('annotator_id') for a in annotations}
    print(f"\nTotal annotations : {n}")
    print(f"Unique cases      : {len(unique_cases)}")
    print(f"Unique annotators : {len(unique_annotators)}")

    model_counts: dict = defaultdict(int)
    for a in annotations:
        model_counts[a.get('agent_model', 'unknown')] += 1
    print("\nAnnotations by model:")
    for model, cnt in sorted(model_counts.items()):
        print(f"  {model}: {cnt} ({cnt / n * 100:.1f}%)")

    # Answer correctness
    _section("ANSWER CORRECTNESS ANALYSIS")
    for step, label in [('step1', 'Initial answer'), ('step2', 'After AI'), ('step3', 'After truth')]:
        correct = sum(1 for a in annotations if a.get(step, {}).get('is_correct'))
        print(f"  {label:<22}: {correct}/{n} correct ({correct / n * 100:.1f}%)")

    # Answer changes
    _section("ANSWER CHANGE ANALYSIS")
    for key, label in [
        ('step1_to_step2_changes', 'Step 1 → Step 2 (after AI)   '),
        ('step2_to_step3_changes', 'Step 2 → Step 3 (after truth)'),
    ]:
        changed = sum(1 for a in annotations if a.get(key, {}).get('answer_changed'))
        print(f"  {label}: {changed}/{n} changed ({changed / n * 100:.1f}%)")

    # Belief changes
    _section("BELIEF CHANGE ANALYSIS")
    for (s1_key, s2_key), label in [
        (('step1', 'step2'), "Step 1 → Step 2 (after seeing AI response)"),
        (('step2', 'step3'), "Step 2 → Step 3 (after seeing ground truth)"),
    ]:
        deltas = [
            calculate_belief_change(
                a.get(s1_key, {}).get('answer_belief'),
                a.get(s2_key, {}).get('answer_belief'),
            )
            for a in annotations
        ]
        deltas = [d for d in deltas if d is not None]
        if deltas:
            _print_belief_summary(deltas, label)

    # Persuasion analysis
    _section("PERSUASION ANALYSIS")
    c_to_i = i_to_c = stayed_correct = stayed_incorrect = 0
    c_to_i_bc, i_to_c_bc = [], []
    for a in annotations:
        s1, s2 = a.get('step1', {}), a.get('step2', {})
        changed = a.get('step1_to_step2_changes', {}).get('answer_changed', False)
        bc = calculate_belief_change(s1.get('answer_belief'), s2.get('answer_belief'))
        s1c, s2c = s1.get('is_correct', False), s2.get('is_correct', False)
        if changed:
            if s1c and not s2c:
                c_to_i += 1
                if bc is not None:
                    c_to_i_bc.append(bc)
            elif not s1c and s2c:
                i_to_c += 1
                if bc is not None:
                    i_to_c_bc.append(bc)
        else:
            (stayed_correct if s1c else stayed_incorrect).__class__  # just counts
            if s1c:
                stayed_correct += 1
            else:
                stayed_incorrect += 1

    print(f"\n  From Step 1 to Step 2 (AI persuasion):")
    print(f"    Persuaded correct → incorrect : {c_to_i}/{n} ({c_to_i / n * 100:.1f}%)"
          + (f"  |  mean belief Δ: {statistics.mean(c_to_i_bc):.3f}" if c_to_i_bc else ""))
    print(f"    Persuaded incorrect → correct : {i_to_c}/{n} ({i_to_c / n * 100:.1f}%)"
          + (f"  |  mean belief Δ: {statistics.mean(i_to_c_bc):.3f}" if i_to_c_bc else ""))
    print(f"    Stayed correct                : {stayed_correct}/{n} ({stayed_correct / n * 100:.1f}%)")
    print(f"    Stayed incorrect              : {stayed_incorrect}/{n} ({stayed_incorrect / n * 100:.1f}%)")

    # Highlights
    _section("HIGHLIGHTS ANALYSIS")
    total_hl = sum(len(a.get('highlights', [])) for a in annotations)
    ann_with_hl = sum(1 for a in annotations if a.get('highlights'))
    print(f"\n  Total highlights: {total_hl}")
    print(f"  Annotations with highlights: {ann_with_hl}/{n} ({ann_with_hl / n * 100:.1f}%)")
    step_hl: dict = defaultdict(int)
    for a in annotations:
        for h in a.get('highlights', []):
            step_hl[h.get('step', 'unknown')] += 1
    if step_hl:
        print("  Highlights by step:")
        for step, count in sorted(step_hl.items()):
            print(f"    {step}: {count}")

    # Time analysis
    _section("TIME ANALYSIS")
    times: dict = defaultdict(list)
    for a in annotations:
        try:
            t0 = datetime.fromisoformat(a['session_start'])
            t1 = datetime.fromisoformat(a['step1_time'])
            t2 = datetime.fromisoformat(a['step2_time'])
            t3 = datetime.fromisoformat(a['step3_time'])
            times['step1'].append((t1 - t0).total_seconds())
            times['step2'].append((t2 - t1).total_seconds())
            times['step3'].append((t3 - t2).total_seconds())
        except Exception:
            pass
    for step_label, vals in times.items():
        if vals:
            print(f"\n  Time on {step_label} (seconds): "
                  f"mean={statistics.mean(vals):.1f}  median={statistics.median(vals):.1f}  "
                  f"min={min(vals):.1f}  max={max(vals):.1f}")

    # Demographics
    _section("ANNOTATOR DEMOGRAPHICS")
    demo: dict = defaultdict(lambda: defaultdict(int))
    for a in annotations:
        for k, v in a.get('demographics', {}).items():
            if k != 'submitted_at' and v and str(v).strip():
                demo[k][str(v)] += 1
    if demo:
        for key, values in sorted(demo.items()):
            print(f"\n  {key.replace('_', ' ').title()}:")
            for v, cnt in sorted(values.items(), key=lambda x: x[1], reverse=True):
                print(f"    {v}: {cnt} ({cnt / n * 100:.1f}%)")
    else:
        print("\n  No demographic information available (all anonymous).")

    _print_career_years_analysis(annotations)

    print("\n" + "=" * 80)


def _print_belief_summary(deltas: list, label: str) -> None:
    inc = [x for x in deltas if x > 0]
    dec = [x for x in deltas if x < 0]
    unc = [x for x in deltas if x == 0]
    total = len(deltas)
    print(f"\n  {label}:")
    for tag, lst in [("Increased", inc), ("Decreased", dec), ("Unchanged", unc)]:
        line = f"    {tag}: {len(lst)}/{total} ({len(lst) / total * 100:.1f}%)"
        if lst and tag != "Unchanged":
            line += (f"  |  mean={statistics.mean(lst):.3f}"
                     f"  median={statistics.median(lst):.3f}"
                     f"  min={min(lst):.3f}  max={max(lst):.3f}")
        print(line)


# ─── Mode: compare ────────────────────────────────────────────────────────────

def run_compare(annotations: list) -> None:
    """Compare persuasion effectiveness and belief-change patterns across models."""
    _header("MODEL COMPARISON ANALYSIS")

    n = len(annotations)
    by_model: dict = defaultdict(list)
    for a in annotations:
        by_model[a.get('agent_model', 'unknown')].append(a)

    print(f"\nTotal annotations: {n}   |   Models: {len(by_model)}")

    model_stats = {k: _compute_model_stats(v) for k, v in by_model.items()}

    # Per-model detail
    _section("DETAILED MODEL ANALYSIS")
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['n'], reverse=True):
        _print_model_detail(model, stats)

    # Summary table
    _section("SUMMARY COMPARISON TABLE")
    print(f"\n{'Model':<50} {'N':>5} {'Ans Chg%':>9} {'C→I%':>6} {'I→C%':>6} {'Net':>6}")
    print("-" * 85)
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['n'], reverse=True):
        m = stats['n']
        chg = stats['answer_changed'] / m * 100 if m else 0
        c2i = stats['correct_to_incorrect'] / m * 100 if m else 0
        i2c = stats['incorrect_to_correct'] / m * 100 if m else 0
        name = (model[:47] + '...') if len(model) > 50 else model
        print(f"{name:<50} {m:>5} {chg:>8.1f}% {c2i:>5.1f}% {i2c:>5.1f}% {i2c - c2i:>+6.1f}%")
    print("\n  C→I = Correct → Incorrect (harmful)   I→C = Incorrect → Correct (helpful)")

    _print_rate_statistics(model_stats)

    _print_belief_change_tables(model_stats)


def _compute_model_stats(anns: list) -> dict:
    stats: dict = {
        'n': len(anns),
        'answer_changed': 0,
        'belief_changes': [],
        'belief_changes_when_correct': [],
        'belief_changes_when_incorrect': [],
        'belief_changes_persuaded': [],
        'belief_changes_not_persuaded': [],
        'correct_to_incorrect': 0,
        'incorrect_to_correct': 0,
        'step1_correct': 0,
        'step2_correct': 0,
        'step3_correct': 0,
    }
    for a in anns:
        s1, s2, s3 = a.get('step1', {}), a.get('step2', {}), a.get('step3', {})
        changed = a.get('step1_to_step2_changes', {}).get('answer_changed', False)
        s1c = s1.get('is_correct', False)
        s2c = s2.get('is_correct', False)
        s3c = s3.get('is_correct', False)
        bc = calculate_belief_change(s1.get('answer_belief'), s2.get('answer_belief'))

        if changed:
            stats['answer_changed'] += 1
        if s1c:
            stats['step1_correct'] += 1
        if s2c:
            stats['step2_correct'] += 1
        if s3c:
            stats['step3_correct'] += 1
        if bc is not None:
            stats['belief_changes'].append(bc)
            if s1c:
                stats['belief_changes_when_correct'].append(bc)
            else:
                stats['belief_changes_when_incorrect'].append(bc)
            if changed:
                stats['belief_changes_persuaded'].append(bc)
            else:
                stats['belief_changes_not_persuaded'].append(bc)
        if changed:
            if s1c and not s2c:
                stats['correct_to_incorrect'] += 1
            elif not s1c and s2c:
                stats['incorrect_to_correct'] += 1
    return stats


def _print_model_detail(model: str, stats: dict) -> None:
    m = stats['n']
    print(f"\n  {'─' * 78}")
    print(f"  {model}   (N={m})")
    print(f"  {'─' * 78}")
    for step, label in [('step1', 'Initial'), ('step2', 'After AI'), ('step3', 'After truth')]:
        acc = stats[f'{step}_correct'] / m * 100 if m else 0
        print(f"    Accuracy {label:<12}: {stats[f'{step}_correct']}/{m} ({acc:.1f}%)")
    chg = stats['answer_changed'] / m * 100 if m else 0
    c2i = stats['correct_to_incorrect'] / m * 100 if m else 0
    i2c = stats['incorrect_to_correct'] / m * 100 if m else 0
    print(f"    Answer change rate : {chg:.1f}%")
    print(f"    Persuasion         : C→I={c2i:.1f}%  I→C={i2c:.1f}%  Net={i2c - c2i:+.1f}%")
    bcs = stats['belief_changes']
    if bcs:
        inc = [x for x in bcs if x > 0]
        dec = [x for x in bcs if x < 0]
        unc = [x for x in bcs if x == 0]
        print(f"    Belief changes     : ↑{len(inc) / len(bcs) * 100:.1f}%"
              f"  ↓{len(dec) / len(bcs) * 100:.1f}%"
              f"  ={len(unc) / len(bcs) * 100:.1f}%"
              + (f"  (mean↑={statistics.mean(inc):.3f}" if inc else "")
              + (f"  mean↓={statistics.mean(dec):.3f})" if dec else ""))


def _aggregate_rate_totals(model_stats: dict, keys: list) -> dict:
    totals = {k: 0 for k in keys}
    totals['n'] = 0
    for stats in model_stats.values():
        totals['n'] += stats['n']
        for key in keys:
            totals[key] += stats.get(key, 0)
    return totals


def _print_rate_statistics(model_stats: dict) -> None:
    metrics = [
        ('answer_changed', 'Answer change'),
        ('correct_to_incorrect', 'C→I (harmful)'),
        ('incorrect_to_correct', 'I→C (helpful)'),
    ]
    totals = _aggregate_rate_totals(model_stats, [m[0] for m in metrics])
    if totals['n'] == 0:
        return

    _section("DECISION CHANGE RATES: CONFIDENCE INTERVALS AND EFFECT SIZES")
    print(f"\n{'Model':<45} {'Metric':<20} {'Rate':>7} {'95% CI':>20} {'h vs rest':>11}")
    print("-" * 105)

    per_model_rows = {}
    model_order = []

    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['n'], reverse=True):
        n = stats['n']
        if n == 0:
            continue
        rows = []
        for key, label in metrics:
            count = stats.get(key, 0)
            rate = count / n
            ci_low, ci_high = _wilson_ci(count, n)
            wald_low, wald_high, se = _wald_ci(count, n)
            rest_n = totals['n'] - n
            rest_count = totals[key] - count
            if rest_n > 0 and rest_count >= 0:
                rest_rate = rest_count / rest_n if rest_n else 0.0
                effect = _cohens_h(rate, rest_rate)
                effect_str = f"{effect:+.3f}"
            else:
                effect_str = "n/a"
            rows.append({
                'label': label,
                'rate': rate,
                'wilson': (ci_low, ci_high),
                'wald': (wald_low, wald_high),
                'se': se,
                'effect': effect_str,
            })

        name = (model[:42] + '...') if len(model) > 45 else model
        for idx, row in enumerate(rows):
            model_col = name if idx == 0 else ''
            ci_low, ci_high = row['wilson']
            ci_str = f"[{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]"
            print(f"{model_col:<45} {row['label']:<20} {row['rate'] * 100:>6.1f}% {ci_str:>20} {row['effect']:>11}")

        per_model_rows[model] = rows
        model_order.append(model)

    overall_rows = []
    for key, label in metrics:
        count = totals[key]
        rate = count / totals['n']
        ci_low, ci_high = _wilson_ci(count, totals['n'])
        wald_low, wald_high, se = _wald_ci(count, totals['n'])
        overall_rows.append({
            'label': label,
            'rate': rate,
            'wilson': (ci_low, ci_high),
            'wald': (wald_low, wald_high),
            'se': se,
        })

    print("-" * 105)
    for row in overall_rows:
        ci_low, ci_high = row['wilson']
        ci_str = f"[{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]"
        print(f"{'Overall':<45} {row['label']:<20} {row['rate'] * 100:>6.1f}% {ci_str:>20} {'—':>11}")

    _section("DECISION CHANGE RATES: WALD MEAN ± STANDARD ERROR")
    print(f"\n{'Model':<45} {'Metric':<20} {'Mean ± SE':>18} {'Wald 95% CI':>20}")
    print("-" * 105)

    for model in model_order:
        rows = per_model_rows.get(model, [])
        if not rows:
            continue
        name = (model[:42] + '...') if len(model) > 45 else model
        for idx, row in enumerate(rows):
            model_col = name if idx == 0 else ''
            mean_se = f"{row['rate'] * 100:5.1f}% ± {row['se'] * 100:4.1f}%"
            wald_low, wald_high = row['wald']
            wald_ci = f"[{wald_low * 100:.1f}%, {wald_high * 100:.1f}%]"
            print(f"{model_col:<45} {row['label']:<20} {mean_se:>18} {wald_ci:>20}")

    print("-" * 105)
    for row in overall_rows:
        mean_se = f"{row['rate'] * 100:5.1f}% ± {row['se'] * 100:4.1f}%"
        wald_low, wald_high = row['wald']
        wald_ci = f"[{wald_low * 100:.1f}%, {wald_high * 100:.1f}%]"
        print(f"{'Overall':<45} {row['label']:<20} {mean_se:>18} {wald_ci:>20}")


def _print_career_years_analysis(annotations: list) -> None:
    annotated_years = []
    annotator_years = {}
    for a in annotations:
        years = _parse_years_of_practice(a.get('demographics', {}).get('years_of_practice'))
        if years is None:
            continue
        annotated_years.append((a, years))
        annotator_id = a.get('annotator_id')
        if annotator_id and annotator_id not in annotator_years:
            annotator_years[annotator_id] = years

    if not annotated_years and not annotator_years:
        _section("CAREER EXPERIENCE ANALYSIS")
        print("\n  No years-of-practice data found in demographics.")
        return

    _section("CAREER EXPERIENCE ANALYSIS")

    def _summary(values):
        if not values:
            return None
        return {
            'n': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
        }

    annotator_summary = _summary(list(annotator_years.values()))
    annotation_summary = _summary([years for _, years in annotated_years])

    if annotator_summary:
        s = annotator_summary
        print(f"\n  Unique annotators with years reported : {s['n']}")
        print(f"    mean={s['mean']:.1f}  median={s['median']:.1f}"
              f"  stdev={s['stdev']:.1f}  min={s['min']:.1f}  max={s['max']:.1f}")
    if annotation_summary:
        s = annotation_summary
        print(f"\n  Annotation-level entries (with repeats): {s['n']}")
        print(f"    mean={s['mean']:.1f}  median={s['median']:.1f}"
              f"  stdev={s['stdev']:.1f}  min={s['min']:.1f}  max={s['max']:.1f}")

    bins = [
        ("<2 yrs", lambda y: y < 2),
        ("2–5 yrs", lambda y: 2 <= y < 5),
        ("5–10 yrs", lambda y: 5 <= y < 10),
        ("10–20 yrs", lambda y: 10 <= y < 20),
        ("20+ yrs", lambda y: y >= 20),
    ]

    change_data = {
        'answer_change': [],
        'harmful': [],
        'helpful': [],
    }
    for ann, yrs in annotated_years:
        change = ann.get('step1_to_step2_changes', {}).get('answer_changed', False)
        s1 = ann.get('step1', {})
        s2 = ann.get('step2', {})
        s1c = s1.get('is_correct')
        s2c = s2.get('is_correct')
        harmful = bool(change and (s1c is True) and (s2c is False))
        helpful = bool(change and (s1c is False) and (s2c is True))
        change_data['answer_change'].append((yrs, 1 if change else 0))
        change_data['harmful'].append((yrs, 1 if harmful else 0))
        change_data['helpful'].append((yrs, 1 if helpful else 0))

    band_results = []
    for label, predicate in bins:
        subset = [(ann, yrs) for ann, yrs in annotated_years if predicate(yrs)]
        row = {'label': label, 'n': len(subset)}
        if subset:
            row['mean_years'] = statistics.mean([yrs for _, yrs in subset])
            acc_rows = {}
            for step in ['step1', 'step2', 'step3']:
                total = 0
                correct = 0
                for ann, _ in subset:
                    is_correct = ann.get(step, {}).get('is_correct')
                    if is_correct is None:
                        continue
                    total += 1
                    if is_correct:
                        correct += 1
                acc = (correct / total * 100) if total else float('nan')
                acc_rows[step] = (acc, total)
            change_count = 0
            harmful_count = 0
            helpful_count = 0
            for ann, _ in subset:
                change_flag = ann.get('step1_to_step2_changes', {}).get('answer_changed', False)
                if change_flag:
                    change_count += 1
                    s1c = ann.get('step1', {}).get('is_correct')
                    s2c = ann.get('step2', {}).get('is_correct')
                    if s1c is True and s2c is False:
                        harmful_count += 1
                    elif s1c is False and s2c is True:
                        helpful_count += 1
            row['accuracy'] = acc_rows
            row['change'] = {
                'answer_change': (change_count, len(subset)),
                'harmful': (harmful_count, len(subset)),
                'helpful': (helpful_count, len(subset)),
            }
        else:
            row['mean_years'] = None
            row['accuracy'] = {step: (float('nan'), 0) for step in ['step1', 'step2', 'step3']}
            row['change'] = {k: (0, 0) for k in ['answer_change', 'harmful', 'helpful']}
        band_results.append(row)

    print("\n  Accuracy by experience band (annotation-level):")
    header = ("    Band", "n", "Step1 acc", "Step2 acc", "Step3 acc", "Mean yrs")
    print(f"  {header[0]:<12} {header[1]:>4} {header[2]:>11} {header[3]:>11} {header[4]:>11} {header[5]:>10}")
    print(f"  {'-' * 64}")
    for row in band_results:
        label = row['label']
        n = row['n']
        mean_years = row['mean_years']
        s1, s2, s3 = (row['accuracy'][step] for step in ['step1', 'step2', 'step3'])
        if n == 0:
            print(f"  {label:<12}    0 {'-' * 42}")
        else:
            print(
                f"  {label:<12} {n:>4} "
                f"{(f'{s1[0]:5.1f}%/{s1[1]}' if s1[1] else '   n/a '):>11} "
                f"{(f'{s2[0]:5.1f}%/{s2[1]}' if s2[1] else '   n/a '):>11} "
                f"{(f'{s3[0]:5.1f}%/{s3[1]}' if s3[1] else '   n/a '):>11} "
                f"{mean_years:>10.1f}"
            )

    print("\n  Decision-change rates by experience band (Step 1 → Step 2):")
    header_change = ("    Band", "n", "Ans change", "C→I", "I→C")
    print(f"  {header_change[0]:<12} {header_change[1]:>4} {header_change[2]:>13} {header_change[3]:>8} {header_change[4]:>8}")
    print(f"  {'-' * 58}")
    for row in band_results:
        label = row['label']
        n = row['n']
        chg = row['change']['answer_change']
        harmful = row['change']['harmful']
        helpful = row['change']['helpful']
        if n == 0:
            print(f"  {label:<12}    0 {'-' * 41}")
            continue
        count_chg, total_chg = chg
        count_harm, total_harm = harmful
        count_help, total_help = helpful
        chg_rate = count_chg / total_chg * 100 if total_chg else float('nan')
        harm_rate = count_harm / total_harm * 100 if total_harm else float('nan')
        help_rate = count_help / total_help * 100 if total_help else float('nan')
        chg_str = f"{chg_rate:5.1f}%/{count_chg}" if not math.isnan(chg_rate) else "   n/a "
        harm_str = f"{harm_rate:5.1f}%/{count_harm}" if not math.isnan(harm_rate) else "   n/a "
        help_str = f"{help_rate:5.1f}%/{count_help}" if not math.isnan(help_rate) else "   n/a "
        print(f"  {label:<12} {n:>4} {chg_str:>13} {harm_str:>8} {help_str:>8}")

    corr_results = {}
    for step in ['step1', 'step2', 'step3']:
        xs = []
        ys = []
        for ann, yrs in annotated_years:
            is_correct = ann.get(step, {}).get('is_correct')
            if is_correct is None:
                continue
            xs.append(yrs)
            ys.append(1 if is_correct else 0)
        corr = _pearson_correlation(xs, ys)
        corr_results[step] = (corr, len(xs))

    print("\n  Pearson correlation between years of practice and accuracy:")
    for step, (corr, n) in corr_results.items():
        label = {'step1': 'Initial', 'step2': 'After AI', 'step3': 'After truth'}[step]
        if corr is None:
            print(f"    {label:<12}: insufficient data (n={n})")
        else:
            print(f"    {label:<12}: r={corr:+.3f}  (n={n})")

    change_corr_labels = {
        'answer_change': 'Answer change',
        'harmful': 'C→I (harmful)',
        'helpful': 'I→C (helpful)',
    }
    print("\n  Pearson correlation between years of practice and decision changes:")
    for key, label in change_corr_labels.items():
        xs = [yrs for yrs, _ in change_data[key]]
        ys = [flag for _, flag in change_data[key]]
        corr = _pearson_correlation(xs, ys)
        n = len(xs)
        if corr is None:
            print(f"    {label:<16}: insufficient data (n={n})")
        else:
            print(f"    {label:<16}: r={corr:+.3f}  (n={n})")


def _print_belief_change_tables(model_stats: dict) -> None:
    W = 120
    _header("DETAILED BELIEF CHANGE PATTERNS BY MODEL", width=W)

    # Compute per-model summary stats
    belief_data = {}
    for model, stats in model_stats.items():
        bcs = stats['belief_changes']
        if not bcs:
            continue
        inc = [x for x in bcs if x > 0]
        dec = [x for x in bcs if x < 0]
        unc = [x for x in bcs if x == 0]
        abs_c = [abs(x) for x in bcs if x != 0]
        belief_data[model] = {
            'n': len(bcs),
            'mean': statistics.mean(bcs),
            'median': statistics.median(bcs),
            'inc_pct': len(inc) / len(bcs) * 100,
            'dec_pct': len(dec) / len(bcs) * 100,
            'unc_pct': len(unc) / len(bcs) * 100,
            'mean_inc': statistics.mean(inc) if inc else 0,
            'mean_dec': statistics.mean(dec) if dec else 0,
            'mean_abs': statistics.mean(abs_c) if abs_c else 0,
            'max_inc': max(inc) if inc else 0,
            'max_dec': min(dec) if dec else 0,
        }

    # Overview table
    _section("BELIEF CHANGE OVERVIEW", width=W)
    print(f"{'Model':<45} {'N':>5} {'Mean Δ':>8} {'Med Δ':>8} {'↑%':>7} {'↓%':>7} {'=%':>7}")
    print("-" * W)
    for model in sorted(belief_data, key=lambda x: belief_data[x]['mean'], reverse=True):
        d = belief_data[model]
        name = (model[:42] + '...') if len(model) > 45 else model
        print(f"{name:<45} {d['n']:>5} {d['mean']:>+7.3f} {d['median']:>+7.3f} "
              f"{d['inc_pct']:>6.1f}% {d['dec_pct']:>6.1f}% {d['unc_pct']:>6.1f}%")

    # Magnitude table
    _section("DETAILED BELIEF CHANGE MAGNITUDES", width=W)
    print(f"{'Model':<45} {'Mean↑':>8} {'Max↑':>8} {'Mean↓':>8} {'Max↓':>8} {'Mean|Δ|':>9}")
    print("-" * W)
    for model in sorted(belief_data, key=lambda x: belief_data[x]['mean_abs'], reverse=True):
        d = belief_data[model]
        name = (model[:42] + '...') if len(model) > 45 else model
        print(f"{name:<45} {d['mean_inc']:>+7.3f} {d['max_inc']:>+7.3f} "
              f"{d['mean_dec']:>+7.3f} {d['max_dec']:>+7.3f} {d['mean_abs']:>8.3f}")

    # Distribution by magnitude bins
    _section("BELIEF CHANGE DISTRIBUTION (count by magnitude)", width=W)
    bins = [
        ("Large decrease (Δ ≤ -0.3)",       lambda x: x <= -0.3),
        ("Moderate decrease (-0.3 < Δ ≤ -0.1)", lambda x: -0.3 < x <= -0.1),
        ("Small decrease (-0.1 < Δ < 0)",   lambda x: -0.1 < x < 0),
        ("No change (Δ = 0)",               lambda x: x == 0),
        ("Small increase (0 < Δ < 0.1)",    lambda x: 0 < x < 0.1),
        ("Moderate increase (0.1 ≤ Δ < 0.3)", lambda x: 0.1 <= x < 0.3),
        ("Large increase (Δ ≥ 0.3)",        lambda x: x >= 0.3),
    ]
    header_row = f"{'Model':<45}"
    for name, _ in bins:
        header_row += f" {name.split('(')[0].strip()[:10]:>10}"
    print(header_row)
    print("-" * W)
    for model in sorted(model_stats, key=lambda x: model_stats[x]['n'], reverse=True):
        bcs = model_stats[model]['belief_changes']
        if not bcs:
            continue
        name = (model[:42] + '...') if len(model) > 45 else model
        row = f"{name:<45}"
        for _, fn in bins:
            cnt = sum(1 for x in bcs if fn(x))
            row += f" {cnt:>3}({cnt / len(bcs) * 100:>4.1f}%)"
        print(row)

    # Conditional analysis: by initial correctness and persuasion
    _section("BELIEF CHANGES BY INITIAL CORRECTNESS AND PERSUASION", width=W)
    for (field_a, field_b), col_a, col_b in [
        (('belief_changes_when_correct', 'belief_changes_when_incorrect'),
         "Initially Correct", "Initially Incorrect"),
        (('belief_changes_persuaded', 'belief_changes_not_persuaded'),
         "Persuaded (answer changed)", "Not Persuaded"),
    ]:
        print(f"\n  {col_a.upper()} vs {col_b.upper()}")
        print(f"  {'Model':<45} {col_a:>25} {col_b:>25} {'Difference':>20}")
        print(f"  {'':45} {'N':>8} {'Mean Δ':>9} {'Med Δ':>7} {'N':>8} {'Mean Δ':>9} {'Med Δ':>7} "
              f"{'Mean':>9} {'Med':>9}")
        print(f"  {'-' * 115}")
        for model in sorted(model_stats, key=lambda x: model_stats[x]['n'], reverse=True):
            a = model_stats[model][field_a]
            b = model_stats[model][field_b]
            ma = statistics.mean(a) if a else 0
            meda = statistics.median(a) if a else 0
            mb = statistics.mean(b) if b else 0
            medb = statistics.median(b) if b else 0
            name = (model[:42] + '...') if len(model) > 45 else model
            print(f"  {name:<45} {len(a):>8} {ma:>+8.3f} {meda:>+6.3f} "
                  f"{len(b):>8} {mb:>+8.3f} {medb:>+6.3f} "
                  f"{mb - ma:>+8.3f} {medb - meda:>+8.3f}")


# ─── Mode: export ─────────────────────────────────────────────────────────────

def run_export(annotations: list, output_file: str) -> None:
    """Export annotations to a flat CSV file."""
    if not annotations:
        print("No annotations to export.")
        return

    fieldnames = [
        'annotator_id', 'case_id', 'agent_name', 'agent_model',
        'correct_answer', 'correct_answer_idx',
        'step1_answer', 'step1_answer_belief', 'step1_is_correct',
        'step2_answer', 'step2_answer_belief', 'step2_is_correct',
        'step3_answer', 'step3_answer_belief', 'step3_is_correct',
        'answer_changed_1_to_2', 'answer_belief_changed_1_to_2',
        'answer_changed_2_to_3', 'answer_belief_changed_2_to_3',
        'belief_change_1_to_2', 'belief_change_2_to_3',
        'correct_to_incorrect', 'incorrect_to_correct',
        'num_highlights', 'num_highlights_step2',
        'session_start', 'step1_time', 'step2_time', 'step3_time',
        'time_step1_seconds', 'time_step2_seconds', 'time_step3_seconds',
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in annotations:
            s1 = a.get('step1', {})
            s2 = a.get('step2', {})
            s3 = a.get('step3', {})
            c12 = a.get('step1_to_step2_changes', {})
            c23 = a.get('step2_to_step3_changes', {})
            bc12 = calculate_belief_change(s1.get('answer_belief'), s2.get('answer_belief'))
            bc23 = calculate_belief_change(s2.get('answer_belief'), s3.get('answer_belief'))
            ans_chg = c12.get('answer_changed', False)
            s1c, s2c = s1.get('is_correct', False), s2.get('is_correct', False)
            highlights = a.get('highlights', [])
            try:
                t0 = datetime.fromisoformat(a['session_start'])
                t1 = datetime.fromisoformat(a['step1_time'])
                t2 = datetime.fromisoformat(a['step2_time'])
                t3 = datetime.fromisoformat(a['step3_time'])
                ts1 = (t1 - t0).total_seconds()
                ts2 = (t2 - t1).total_seconds()
                ts3 = (t3 - t2).total_seconds()
            except Exception:
                ts1 = ts2 = ts3 = None
            writer.writerow({
                'annotator_id': a.get('annotator_id'),
                'case_id': a.get('case_id'),
                'agent_name': a.get('agent_name'),
                'agent_model': a.get('agent_model'),
                'correct_answer': a.get('correct_answer'),
                'correct_answer_idx': a.get('correct_answer_idx'),
                'step1_answer': s1.get('answer'),
                'step1_answer_belief': s1.get('answer_belief'),
                'step1_is_correct': s1.get('is_correct'),
                'step2_answer': s2.get('answer'),
                'step2_answer_belief': s2.get('answer_belief'),
                'step2_is_correct': s2.get('is_correct'),
                'step3_answer': s3.get('answer'),
                'step3_answer_belief': s3.get('answer_belief'),
                'step3_is_correct': s3.get('is_correct'),
                'answer_changed_1_to_2': c12.get('answer_changed'),
                'answer_belief_changed_1_to_2': c12.get('answer_belief_changed'),
                'answer_changed_2_to_3': c23.get('answer_changed'),
                'answer_belief_changed_2_to_3': c23.get('answer_belief_changed'),
                'belief_change_1_to_2': bc12,
                'belief_change_2_to_3': bc23,
                'correct_to_incorrect': ans_chg and s1c and not s2c,
                'incorrect_to_correct': ans_chg and not s1c and s2c,
                'num_highlights': len(highlights),
                'num_highlights_step2': sum(1 for h in highlights if h.get('step') == 'step2'),
                'session_start': a.get('session_start'),
                'step1_time': a.get('step1_time'),
                'step2_time': a.get('step2_time'),
                'step3_time': a.get('step3_time'),
                'time_step1_seconds': ts1,
                'time_step2_seconds': ts2,
                'time_step3_seconds': ts3,
            })

    print(f"Exported {len(annotations)} annotations to {output_file}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze USMLE Sample annotation results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes:
  stats   - Comprehensive statistics (belief changes, persuasion, highlights,
            time spent, demographics)
  compare - Per-model comparison with detailed belief-change tables
  export  - Export annotations to CSV""",
    )
    parser.add_argument(
        '--mode', choices=['stats', 'compare', 'export'], default='stats',
        help='Analysis mode (default: stats)',
    )
    parser.add_argument(
        '--results-dir', default=DEFAULT_RESULTS_DIR,
        help=f'Path to annotation results directory (default: {DEFAULT_RESULTS_DIR})',
    )
    parser.add_argument(
        '--output', default='annotations.csv',
        help='Output CSV path for export mode (default: annotations.csv)',
    )
    args = parser.parse_args()

    annotations = load_all_annotations(args.results_dir)
    if not annotations:
        print(f"No annotations found in {args.results_dir}")
        return

    print(f"Loaded {len(annotations)} annotations from {args.results_dir}")

    if args.mode == 'stats':
        run_stats(annotations)
    elif args.mode == 'compare':
        run_compare(annotations)
    elif args.mode == 'export':
        run_export(annotations, args.output)


if __name__ == '__main__':
    main()
