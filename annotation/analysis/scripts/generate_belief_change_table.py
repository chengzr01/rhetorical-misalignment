#!/usr/bin/env python3
"""
Generate detailed belief-change tables from persuasion_examples.json.

Outputs:
  belief_change_table.csv   — one row per case, all fields
  belief_change_table.txt   — human-readable formatted tables

Usage:
  python generate_belief_change_table.py
  python generate_belief_change_table.py --input ../outputs/persuasion_examples.json
  python generate_belief_change_table.py --no-txt   (CSV only)
  python generate_belief_change_table.py --no-csv   (text only)
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

DEFAULT_INPUT = '../outputs/persuasion_examples.json'
OUTPUT_DIR = '../outputs'


def short_model(name):
    return (name or 'unknown').split('/')[-1]


def direction(bc):
    if bc is None:
        return '?'
    if bc > 0:
        return 'increase'
    if bc < 0:
        return 'decrease'
    return 'unchanged'


def belief_category(bc):
    """Label the magnitude of the belief change."""
    if bc is None:
        return 'N/A'
    if bc <= -0.30:
        return 'large decrease'
    if bc <= -0.10:
        return 'moderate decrease'
    if bc < 0:
        return 'small decrease'
    if bc == 0:
        return 'unchanged'
    if bc < 0.10:
        return 'small increase'
    if bc < 0.30:
        return 'moderate increase'
    return 'large increase'


def initial_confidence_band(b):
    if b is None:
        return 'N/A'
    if b <= 0.40:
        return 'low (≤0.40)'
    if b <= 0.70:
        return 'medium (0.41–0.70)'
    return 'high (>0.70)'


def load_cases(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for case in data['cases']['harmful_persuasion']:
        rows.append((case, 'harmful'))
    for case in data['cases']['helpful_persuasion']:
        rows.append((case, 'helpful'))
    return rows, data


def build_rows(cases):
    """Convert raw cases to flat dicts for the table."""
    records = []
    for case, ptype in cases:
        demo = case.get('demographics', {})
        bc = case.get('belief_change')
        s1b = case.get('step1_belief')
        s2b = case.get('step2_belief')
        reasoning_text = (case.get('reasoning') or '').strip()
        records.append({
            'persuasion_type':       ptype,
            'case_id':               case.get('case_id', ''),
            'annotator_id':          case.get('annotator_id', ''),
            'model':                 short_model(case.get('model')),
            'correct_answer_idx':    case.get('correct_answer_idx', ''),
            'step1_answer':          case.get('step1_answer', ''),
            'step1_belief':          f"{s1b:.3f}" if s1b is not None else '',
            'initial_confidence_band': initial_confidence_band(s1b),
            'step2_answer':          case.get('step2_answer', ''),
            'step2_belief':          f"{s2b:.3f}" if s2b is not None else '',
            'belief_change':         f"{bc:+.3f}" if bc is not None else '',
            'belief_direction':      direction(bc),
            'belief_category':       belief_category(bc),
            'num_highlights':        str(case.get('num_highlights', 0)),
            'has_reasoning':         'yes' if case.get('has_reasoning') and reasoning_text else 'no',
            'reasoning_snippet':     reasoning_text[:120].replace('\n', ' ') if reasoning_text else '',
            'expertise':             demo.get('expertise', ''),
            'years_of_practice':     demo.get('years_of_practice', ''),
            'age':                   demo.get('age', ''),
            'sex':                   demo.get('sex', ''),
        })
    return records


# ── CSV ────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    'persuasion_type', 'case_id', 'annotator_id', 'model',
    'correct_answer_idx', 'step1_answer', 'step1_belief', 'initial_confidence_band',
    'step2_answer', 'step2_belief', 'belief_change', 'belief_direction', 'belief_category',
    'num_highlights', 'has_reasoning', 'reasoning_snippet',
    'expertise', 'years_of_practice', 'age', 'sex',
]


def write_csv(records, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV saved: {path}  ({len(records)} rows)")


# ── Text tables ────────────────────────────────────────────────────────────────

def _header(title, width=120):
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def _section(title, width=120):
    print(f"\n{'-' * width}\n{title}\n{'-' * width}")


def _stats_row(values, label, indent=2):
    pad = ' ' * indent
    if not values:
        print(f"{pad}{label}: (no data)")
        return
    mean = statistics.mean(values)
    med  = statistics.median(values)
    sd   = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{pad}{label}:  n={len(values):3d}  mean={mean:+.3f}  median={med:+.3f}"
          f"  stdev={sd:.3f}  min={min(values):+.3f}  max={max(values):+.3f}")


def write_txt(cases_raw, data, path):
    import sys, io
    buf = io.StringIO()
    orig = sys.stdout
    sys.stdout = buf

    harmful_raw = [(c, 'harmful') for c, t in cases_raw if t == 'harmful']
    helpful_raw = [(c, 'helpful') for c, t in cases_raw if t == 'helpful']

    _header("DETAILED BELIEF CHANGE TABLES — HUMAN STUDY (USMLE SAMPLE)")

    # ── Table 1: Full per-case table ──────────────────────────────────────────
    _section("TABLE 1: PER-CASE BELIEF CHANGES (sorted by persuasion type, then |Δbelief|)")

    COL = {
        'Case':      24,
        'Model':     32,
        'S1':         6,
        'S2':         6,
        'Δ':          7,
        'Dir':       10,
        'Category':  17,
        'HL':         3,
        'Rsn':        4,
        'Expertise': 28,
        'YoP':        5,
    }

    def hdr():
        print(f"  {'Case':<{COL['Case']}} {'Model':<{COL['Model']}} "
              f"{'S1Bel':>{COL['S1']}} {'S2Bel':>{COL['S2']}} {'ΔBel':>{COL['Δ']}} "
              f"{'Direction':<{COL['Dir']}} {'Category':<{COL['Category']}} "
              f"{'HL':>{COL['HL']}} {'Rsn':<{COL['Rsn']}} "
              f"{'Expertise':<{COL['Expertise']}} {'YoP':>{COL['YoP']}}")
        print(f"  {'-'*118}")

    def row(case, ptype):
        demo = case.get('demographics', {})
        bc = case.get('belief_change')
        s1b = case.get('step1_belief')
        s2b = case.get('step2_belief')
        s1s = f"{s1b:.3f}" if s1b is not None else 'N/A'
        s2s = f"{s2b:.3f}" if s2b is not None else 'N/A'
        bcs = f"{bc:+.3f}" if bc is not None else 'N/A'
        cat = belief_category(bc)
        dir_ = direction(bc)
        hl  = str(case.get('num_highlights', 0))
        rsn = 'yes' if case.get('has_reasoning') and (case.get('reasoning') or '').strip() else 'no'
        exp = (demo.get('expertise') or '')[:27]
        yop = (demo.get('years_of_practice') or '')[:5]
        case_label = f"[{ptype[0].upper()}] {case.get('case_id', '')}"[:24]
        model_s = short_model(case.get('model', ''))[:32]
        print(f"  {case_label:<{COL['Case']}} {model_s:<{COL['Model']}} "
              f"{s1s:>{COL['S1']}} {s2s:>{COL['S2']}} {bcs:>{COL['Δ']}} "
              f"{dir_:<{COL['Dir']}} {cat:<{COL['Category']}} "
              f"{hl:>{COL['HL']}} {rsn:<{COL['Rsn']}} "
              f"{exp:<{COL['Expertise']}} {yop:>{COL['YoP']}}")

    print()
    print("  Legend: [H] = Harmful (correct→incorrect), [P] = Helpful (incorrect→correct)")
    print("  S1Bel = initial belief, S2Bel = post-AI belief, ΔBel = belief change")
    print("  HL = # highlights, Rsn = provided reasoning, YoP = years of practice")
    print()
    hdr()

    for section_label, subset, tag in [
        ("HARMFUL PERSUASION (correct → incorrect)", harmful_raw, 'harmful'),
        ("HELPFUL PERSUASION (incorrect → correct)", helpful_raw, 'helpful'),
    ]:
        print(f"\n  ── {section_label} ──")
        sorted_cases = sorted(
            subset,
            key=lambda x: abs(x[0].get('belief_change') or 0),
            reverse=True
        )
        for case, ptype in sorted_cases:
            row(case, ptype)

    # ── Table 2: Summary by model ─────────────────────────────────────────────
    _section("TABLE 2: BELIEF CHANGE SUMMARY BY MODEL")

    model_stats = defaultdict(lambda: {'harmful': [], 'helpful': []})
    for case, ptype in cases_raw:
        bc = case.get('belief_change')
        if bc is not None:
            model_stats[short_model(case.get('model'))][ptype].append(bc)

    all_models = sorted(model_stats.keys())

    print()
    print(f"  {'Model':<38} "
          f"{'── Harmful (C→I) ──':^30}  "
          f"{'── Helpful (I→C) ──':^30}")
    print(f"  {'':38} "
          f"{'N':>4} {'Mean':>7} {'Med':>7} {'Stdev':>6}  "
          f"{'N':>4} {'Mean':>7} {'Med':>7} {'Stdev':>6}")
    print(f"  {'-'*108}")

    for m in all_models:
        hm = model_stats[m]['harmful']
        hp = model_stats[m]['helpful']
        hm_mean = statistics.mean(hm) if hm else float('nan')
        hm_med  = statistics.median(hm) if hm else float('nan')
        hm_sd   = statistics.stdev(hm) if len(hm) > 1 else 0.0
        hp_mean = statistics.mean(hp) if hp else float('nan')
        hp_med  = statistics.median(hp) if hp else float('nan')
        hp_sd   = statistics.stdev(hp) if len(hp) > 1 else 0.0
        print(f"  {m:<38} "
              f"{len(hm):>4} {hm_mean:>+6.3f} {hm_med:>+6.3f} {hm_sd:>5.3f}  "
              f"{len(hp):>4} {hp_mean:>+6.3f} {hp_med:>+6.3f} {hp_sd:>5.3f}")

    # ── Table 3: Summary by initial confidence band ───────────────────────────
    _section("TABLE 3: BELIEF CHANGE BY INITIAL CONFIDENCE BAND")

    conf_bands = [
        ('Low   (step1 ≤ 0.40)', lambda b: b is not None and b <= 0.40),
        ('Medium(0.40 < step1 ≤ 0.70)', lambda b: b is not None and 0.40 < b <= 0.70),
        ('High  (step1 > 0.70)', lambda b: b is not None and b > 0.70),
    ]

    print()
    print(f"  {'Confidence Band':<30} "
          f"{'── Harmful (C→I) ──':^38}  "
          f"{'── Helpful (I→C) ──':^38}")
    print(f"  {'':30} "
          f"{'N':>4} {'Mean Δ':>8} {'Med Δ':>7} {'↑%':>6} {'↓%':>6}  "
          f"{'N':>4} {'Mean Δ':>8} {'Med Δ':>7} {'↑%':>6} {'↓%':>6}")
    print(f"  {'-'*116}")

    for label, fn in conf_bands:
        def band_stats(subset):
            filtered = [(c.get('belief_change'), c.get('step1_belief'))
                        for c, _ in subset
                        if fn(c.get('step1_belief'))]
            bcs = [bc for bc, _ in filtered if bc is not None]
            if not bcs:
                return 0, float('nan'), float('nan'), 0.0, 0.0
            inc = sum(1 for x in bcs if x > 0)
            dec = sum(1 for x in bcs if x < 0)
            return (len(bcs),
                    statistics.mean(bcs),
                    statistics.median(bcs),
                    inc / len(bcs) * 100,
                    dec / len(bcs) * 100)

        nh, hm, hmed, hinc, hdec = band_stats(harmful_raw)
        np_, pm, pmed, pinc, pdec = band_stats(helpful_raw)
        print(f"  {label:<30} "
              f"{nh:>4} {hm:>+7.3f} {hmed:>+6.3f} {hinc:>5.1f}% {hdec:>5.1f}%  "
              f"{np_:>4} {pm:>+7.3f} {pmed:>+6.3f} {pinc:>5.1f}% {pdec:>5.1f}%")

    # ── Table 4: Belief change magnitude distribution ─────────────────────────
    _section("TABLE 4: BELIEF CHANGE MAGNITUDE DISTRIBUTION")

    mag_bins = [
        ('≤ −0.30  (large decrease)',      lambda x: x <= -0.30),
        ('−0.30–−0.10 (moderate decrease)', lambda x: -0.30 < x <= -0.10),
        ('−0.10– 0.00 (small decrease)',    lambda x: -0.10 < x < 0),
        (' 0.00       (unchanged)',         lambda x: x == 0),
        (' 0.00– 0.10 (small increase)',    lambda x: 0 < x < 0.10),
        (' 0.10– 0.30 (moderate increase)', lambda x: 0.10 <= x < 0.30),
        ('≥  0.30    (large increase)',     lambda x: x >= 0.30),
    ]

    harm_bcs = [c.get('belief_change') for c, _ in harmful_raw if c.get('belief_change') is not None]
    help_bcs = [c.get('belief_change') for c, _ in helpful_raw if c.get('belief_change') is not None]
    all_bcs  = harm_bcs + help_bcs

    print()
    print(f"  {'Magnitude Category':<36} "
          f"{'All':>5} {'All%':>6}  "
          f"{'Harmful':>7} {'H%':>6}  "
          f"{'Helpful':>7} {'P%':>6}")
    print(f"  {'-'*82}")

    for label, fn in mag_bins:
        na = sum(1 for x in all_bcs if fn(x))
        nh = sum(1 for x in harm_bcs if fn(x))
        np_ = sum(1 for x in help_bcs if fn(x))
        pct_a = na / len(all_bcs)  * 100 if all_bcs  else 0
        pct_h = nh / len(harm_bcs) * 100 if harm_bcs else 0
        pct_p = np_ / len(help_bcs) * 100 if help_bcs else 0
        print(f"  {label:<36} "
              f"{na:>5} {pct_a:>5.1f}%  "
              f"{nh:>7} {pct_h:>5.1f}%  "
              f"{np_:>7} {pct_p:>5.1f}%")

    # Row totals
    print(f"  {'─'*82}")
    print(f"  {'TOTAL':<36} {len(all_bcs):>5}          "
          f"{len(harm_bcs):>7}          {len(help_bcs):>7}")

    # ── Table 5: Belief change by years of practice ───────────────────────────
    _section("TABLE 5: BELIEF CHANGE BY YEARS OF PRACTICE")

    def parse_years(val):
        try:
            return float(str(val).split()[0])
        except Exception:
            return None

    yop_bins = [
        ('< 5 years',   lambda y: y < 5),
        ('5–14 years',  lambda y: 5 <= y < 15),
        ('15–29 years', lambda y: 15 <= y < 30),
        ('30+ years',   lambda y: y >= 30),
        ('Unknown',     None),
    ]

    print()
    print(f"  {'Years of Practice':<18} "
          f"{'── Harmful (C→I) ──':^38}  "
          f"{'── Helpful (I→C) ──':^38}")
    print(f"  {'':18} "
          f"{'N':>4} {'Mean Δ':>8} {'Med Δ':>7} {'↑%':>6} {'↓%':>6}  "
          f"{'N':>4} {'Mean Δ':>8} {'Med Δ':>7} {'↑%':>6} {'↓%':>6}")
    print(f"  {'-'*104}")

    for label, fn in yop_bins:
        def yop_stats(subset):
            filtered = []
            for c, _ in subset:
                bc = c.get('belief_change')
                if bc is None:
                    continue
                y = parse_years(c.get('demographics', {}).get('years_of_practice'))
                if fn is None:
                    if y is None:
                        filtered.append(bc)
                elif y is not None and fn(y):
                    filtered.append(bc)
            if not filtered:
                return 0, float('nan'), float('nan'), 0.0, 0.0
            inc = sum(1 for x in filtered if x > 0)
            dec = sum(1 for x in filtered if x < 0)
            return (len(filtered),
                    statistics.mean(filtered),
                    statistics.median(filtered),
                    inc / len(filtered) * 100,
                    dec / len(filtered) * 100)

        nh, hm, hmed, hinc, hdec = yop_stats(harmful_raw)
        np_, pm, pmed, pinc, pdec = yop_stats(helpful_raw)
        print(f"  {label:<18} "
              f"{nh:>4} {hm:>+7.3f} {hmed:>+6.3f} {hinc:>5.1f}% {hdec:>5.1f}%  "
              f"{np_:>4} {pm:>+7.3f} {pmed:>+6.3f} {pinc:>5.1f}% {pdec:>5.1f}%")

    # ── Table 6: Highlights breakdown ─────────────────────────────────────────
    _section("TABLE 6: BELIEF CHANGE BY HIGHLIGHT COUNT")

    hl_bins = [
        ('0 highlights',   lambda h: h == 0),
        ('1 highlight',    lambda h: h == 1),
        ('2–3 highlights', lambda h: 2 <= h <= 3),
        ('4+ highlights',  lambda h: h >= 4),
    ]

    print()
    print(f"  {'Highlights':<18} "
          f"{'── Harmful (C→I) ──':^38}  "
          f"{'── Helpful (I→C) ──':^38}")
    print(f"  {'':18} "
          f"{'N':>4} {'Mean Δ':>8} {'Med Δ':>7} {'↑%':>6} {'↓%':>6}  "
          f"{'N':>4} {'Mean Δ':>8} {'Med Δ':>7} {'↑%':>6} {'↓%':>6}")
    print(f"  {'-'*104}")

    for label, fn in hl_bins:
        def hl_stats(subset):
            bcs = [c.get('belief_change') for c, _ in subset
                   if fn(c.get('num_highlights', 0)) and c.get('belief_change') is not None]
            if not bcs:
                return 0, float('nan'), float('nan'), 0.0, 0.0
            inc = sum(1 for x in bcs if x > 0)
            dec = sum(1 for x in bcs if x < 0)
            return (len(bcs), statistics.mean(bcs), statistics.median(bcs),
                    inc / len(bcs) * 100, dec / len(bcs) * 100)

        nh, hm, hmed, hinc, hdec = hl_stats(harmful_raw)
        np_, pm, pmed, pinc, pdec = hl_stats(helpful_raw)
        print(f"  {label:<18} "
              f"{nh:>4} {hm:>+7.3f} {hmed:>+6.3f} {hinc:>5.1f}% {hdec:>5.1f}%  "
              f"{np_:>4} {pm:>+7.3f} {pmed:>+6.3f} {pinc:>5.1f}% {pdec:>5.1f}%")

    # ── Table 7: Aggregate summary ─────────────────────────────────────────────
    _section("TABLE 7: AGGREGATE SUMMARY")

    print()
    print(f"  {'Metric':<50} {'Harmful (C→I)':>18} {'Helpful (I→C)':>18} {'All':>10}")
    print(f"  {'-'*98}")

    rows_data = [
        ('Total cases (persuasion)',         len(harmful_raw), len(helpful_raw),
         len(harmful_raw) + len(helpful_raw)),
    ]

    def agg(subset, fn):
        vals = [fn(c) for c, _ in subset if fn(c) is not None]
        return vals

    hm_bc = [c.get('belief_change') for c, _ in harmful_raw if c.get('belief_change') is not None]
    hp_bc = [c.get('belief_change') for c, _ in helpful_raw if c.get('belief_change') is not None]
    all_bc2 = hm_bc + hp_bc

    def fmt(v, is_pct=False):
        if isinstance(v, float) and (v != v):
            return 'N/A'
        if is_pct:
            return f"{v:.1f}%"
        if isinstance(v, float):
            return f"{v:+.3f}"
        return str(v)

    summary_rows = [
        ('N (with belief data)',
         len(hm_bc), len(hp_bc), len(all_bc2)),
        ('Mean Δbelief',
         statistics.mean(hm_bc) if hm_bc else float('nan'),
         statistics.mean(hp_bc) if hp_bc else float('nan'),
         statistics.mean(all_bc2) if all_bc2 else float('nan')),
        ('Median Δbelief',
         statistics.median(hm_bc) if hm_bc else float('nan'),
         statistics.median(hp_bc) if hp_bc else float('nan'),
         statistics.median(all_bc2) if all_bc2 else float('nan')),
        ('Stdev Δbelief',
         statistics.stdev(hm_bc) if len(hm_bc) > 1 else 0.0,
         statistics.stdev(hp_bc) if len(hp_bc) > 1 else 0.0,
         statistics.stdev(all_bc2) if len(all_bc2) > 1 else 0.0),
        ('Min Δbelief',  min(hm_bc) if hm_bc else float('nan'),
         min(hp_bc) if hp_bc else float('nan'),
         min(all_bc2) if all_bc2 else float('nan')),
        ('Max Δbelief',  max(hm_bc) if hm_bc else float('nan'),
         max(hp_bc) if hp_bc else float('nan'),
         max(all_bc2) if all_bc2 else float('nan')),
        ('% with increased belief',
         sum(1 for x in hm_bc if x > 0) / len(hm_bc) * 100 if hm_bc else float('nan'),
         sum(1 for x in hp_bc if x > 0) / len(hp_bc) * 100 if hp_bc else float('nan'),
         sum(1 for x in all_bc2 if x > 0) / len(all_bc2) * 100 if all_bc2 else float('nan')),
        ('% with decreased belief',
         sum(1 for x in hm_bc if x < 0) / len(hm_bc) * 100 if hm_bc else float('nan'),
         sum(1 for x in hp_bc if x < 0) / len(hp_bc) * 100 if hp_bc else float('nan'),
         sum(1 for x in all_bc2 if x < 0) / len(all_bc2) * 100 if all_bc2 else float('nan')),
        ('% with unchanged belief',
         sum(1 for x in hm_bc if x == 0) / len(hm_bc) * 100 if hm_bc else float('nan'),
         sum(1 for x in hp_bc if x == 0) / len(hp_bc) * 100 if hp_bc else float('nan'),
         sum(1 for x in all_bc2 if x == 0) / len(all_bc2) * 100 if all_bc2 else float('nan')),
        ('Mean initial belief (step1)',
         statistics.mean(c.get('step1_belief') for c, _ in harmful_raw if c.get('step1_belief') is not None),
         statistics.mean(c.get('step1_belief') for c, _ in helpful_raw if c.get('step1_belief') is not None),
         statistics.mean(c.get('step1_belief') for c, _ in cases_raw if c.get('step1_belief') is not None)),
        ('Mean post-AI belief (step2)',
         statistics.mean(c.get('step2_belief') for c, _ in harmful_raw if c.get('step2_belief') is not None),
         statistics.mean(c.get('step2_belief') for c, _ in helpful_raw if c.get('step2_belief') is not None),
         statistics.mean(c.get('step2_belief') for c, _ in cases_raw if c.get('step2_belief') is not None)),
        ('% with highlights',
         sum(1 for c, _ in harmful_raw if c.get('num_highlights', 0) > 0) / len(harmful_raw) * 100,
         sum(1 for c, _ in helpful_raw if c.get('num_highlights', 0) > 0) / len(helpful_raw) * 100,
         sum(1 for c, _ in cases_raw if c.get('num_highlights', 0) > 0) / len(cases_raw) * 100),
        ('% with annotator reasoning',
         sum(1 for c, _ in harmful_raw if c.get('has_reasoning') and (c.get('reasoning') or '').strip()) / len(harmful_raw) * 100,
         sum(1 for c, _ in helpful_raw if c.get('has_reasoning') and (c.get('reasoning') or '').strip()) / len(helpful_raw) * 100,
         sum(1 for c, _ in cases_raw if c.get('has_reasoning') and (c.get('reasoning') or '').strip()) / len(cases_raw) * 100),
    ]

    for label, hv, pv, av in summary_rows:
        is_pct = '%' in label
        print(f"  {label:<50} {fmt(hv, is_pct):>18} {fmt(pv, is_pct):>18} {fmt(av, is_pct):>10}")

    sys.stdout = orig
    content = buf.getvalue()
    print(content)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Text tables saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate belief-change tables.')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--no-csv', action='store_true')
    parser.add_argument('--no-txt', action='store_true')
    args = parser.parse_args()

    cases_raw, data = load_cases(args.input)
    print(f"Loaded {len(cases_raw)} persuasion cases")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.no_csv:
        records = build_rows(cases_raw)
        write_csv(records, os.path.join(OUTPUT_DIR, 'belief_change_table.csv'))

    if not args.no_txt:
        write_txt(cases_raw, data, os.path.join(OUTPUT_DIR, 'belief_change_table.txt'))


if __name__ == '__main__':
    main()
