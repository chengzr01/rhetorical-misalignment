#!/usr/bin/env python3
"""
Analyze human belief changes after seeing AI information, connected to
persuasion examples from persuasion_examples.json.

Two analyses:
  (1) general  - Distribution, magnitude, and direction of belief changes
                 across all persuasion cases and by type.
  (2) examples - How features of the AI interaction (model, highlights,
                 reasoning, initial confidence, demographics) relate to
                 the magnitude/direction of belief change.

Usage:
  python analyze_belief_changes.py --mode general
  python analyze_belief_changes.py --mode examples
  python analyze_belief_changes.py --mode all   (default)
  python analyze_belief_changes.py --input /path/to/persuasion_examples.json
"""

import argparse
import json
import os
import re
import statistics
from collections import Counter, defaultdict


DEFAULT_INPUT = '../outputs/persuasion_examples.json'
OUTPUT_DIR = '../outputs'


# ── utilities ─────────────────────────────────────────────────────────────────

def _header(title, width=80):
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def _section(title, width=80):
    print(f"\n{'-' * width}\n{title}\n{'-' * width}")


def _stats(values, label='', indent=4):
    pad = ' ' * indent
    if not values:
        print(f"{pad}{label}: (no data)")
        return
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{pad}{label}: n={len(values)}  mean={mean:+.3f}  median={median:+.3f}"
          f"  stdev={stdev:.3f}  min={min(values):+.3f}  max={max(values):+.3f}")


def _bar(value, max_val, width=30, char='█'):
    filled = int(round(value / max_val * width)) if max_val else 0
    return char * filled + '░' * (width - filled)


def short_model(name):
    return (name or 'unknown').split('/')[-1]


def load_data(path):
    with open(path) as f:
        return json.load(f)


def all_cases(data):
    """Return (case, persuasion_type) pairs for every case."""
    for case in data['cases']['harmful_persuasion']:
        yield case, 'harmful'
    for case in data['cases']['helpful_persuasion']:
        yield case, 'helpful'


# ── (1) General characterisation ──────────────────────────────────────────────

def run_general(data):
    _header("BELIEF CHANGE ANALYSIS — GENERAL CHARACTERISATION")

    harmful = data['cases']['harmful_persuasion']
    helpful = data['cases']['helpful_persuasion']
    all_c = [(c, 'harmful') for c in harmful] + [(c, 'helpful') for c in helpful]
    n_total = len(all_c)

    print(f"\n  Total persuasion cases : {n_total}")
    print(f"  Harmful (C→I)          : {len(harmful)}")
    print(f"  Helpful (I→C)          : {len(helpful)}")

    # ── 1a. Overall belief-change distribution ────────────────────────────────
    _section("1a. BELIEF CHANGE DISTRIBUTION (all persuasion cases)")

    all_bc = [c['belief_change'] for c, _ in all_c if c['belief_change'] is not None]
    inc = [x for x in all_bc if x > 0]
    dec = [x for x in all_bc if x < 0]
    unc = [x for x in all_bc if x == 0]
    print(f"\n  When a human changed their answer after seeing AI:")
    _stats(all_bc, "Overall Δbelief")
    print(f"\n  Direction breakdown:")
    print(f"    Increased  : {len(inc):3d} / {n_total} ({len(inc)/n_total*100:.1f}%)")
    print(f"    Decreased  : {len(dec):3d} / {n_total} ({len(dec)/n_total*100:.1f}%)")
    print(f"    Unchanged  : {len(unc):3d} / {n_total} ({len(unc)/n_total*100:.1f}%)")
    if inc:
        _stats(inc, "  Among increases", indent=4)
    if dec:
        _stats(dec, "  Among decreases", indent=4)

    # Histogram of belief-change magnitudes
    bins = [
        ("≤ −0.30  (large decrease)", lambda x: x <= -0.30),
        ("−0.30–−0.10 (moderate ↓)", lambda x: -0.30 < x <= -0.10),
        ("−0.10–0.00  (small ↓)",    lambda x: -0.10 < x < 0),
        (" 0.00       (unchanged)",  lambda x: x == 0),
        (" 0.00–0.10  (small ↑)",    lambda x: 0 < x < 0.10),
        (" 0.10–0.30  (moderate ↑)", lambda x: 0.10 <= x < 0.30),
        ("≥  0.30    (large ↑)",     lambda x: x >= 0.30),
    ]
    print(f"\n  Histogram (all cases, n={len(all_bc)}):")
    max_cnt = max(sum(1 for x in all_bc if fn(x)) for _, fn in bins)
    for label, fn in bins:
        cnt = sum(1 for x in all_bc if fn(x))
        pct = cnt / len(all_bc) * 100 if all_bc else 0
        bar = _bar(cnt, max_cnt)
        print(f"    {label}: {bar} {cnt:3d} ({pct:5.1f}%)")

    # ── 1b. Harmful vs. helpful comparison ───────────────────────────────────
    _section("1b. HARMFUL vs. HELPFUL PERSUASION — BELIEF CHANGE PATTERNS")

    for label, cases in [("Harmful (C→I)", harmful), ("Helpful (I→C)", helpful)]:
        bc = [c['belief_change'] for c in cases if c['belief_change'] is not None]
        inc_ = [x for x in bc if x > 0]
        dec_ = [x for x in bc if x < 0]
        unc_ = [x for x in bc if x == 0]
        print(f"\n  {label}  (n={len(cases)})")
        _stats(bc, "Δbelief")
        print(f"    Direction: ↑{len(inc_)/len(bc)*100:.1f}%  ↓{len(dec_)/len(bc)*100:.1f}%  ={len(unc_)/len(bc)*100:.1f}%")

    # ── 1c. Initial belief as moderator ──────────────────────────────────────
    _section("1c. INITIAL CONFIDENCE (step1_belief) AS PERSUASION MODERATOR")

    bins_belief = [
        ("Low confidence   (0.0–0.40)", lambda x: x <= 0.40),
        ("Medium confidence(0.40–0.70)", lambda x: 0.40 < x <= 0.70),
        ("High confidence  (0.70–1.0)", lambda x: x > 0.70),
    ]

    print(f"\n  Does initial confidence predict susceptibility to harmful persuasion?")
    print(f"\n  {'Confidence band':<38} {'N':>4} {'Mean Δbelief':>13} {'%↑':>6} {'%↓':>6}")
    print(f"  {'-'*72}")
    for label, fn in bins_belief:
        cases_ = [c for c in harmful if c['step1_belief'] is not None and fn(c['step1_belief'])]
        bc_ = [c['belief_change'] for c in cases_ if c['belief_change'] is not None]
        inc_ = [x for x in bc_ if x > 0]
        dec_ = [x for x in bc_ if x < 0]
        mean_ = statistics.mean(bc_) if bc_ else float('nan')
        pct_inc = len(inc_)/len(bc_)*100 if bc_ else 0
        pct_dec = len(dec_)/len(bc_)*100 if bc_ else 0
        print(f"  {label:<38} {len(cases_):>4} {mean_:>+12.3f} {pct_inc:>5.1f}% {pct_dec:>5.1f}%")

    print(f"\n  Does initial confidence predict susceptibility to helpful persuasion?")
    print(f"\n  {'Confidence band':<38} {'N':>4} {'Mean Δbelief':>13} {'%↑':>6} {'%↓':>6}")
    print(f"  {'-'*72}")
    for label, fn in bins_belief:
        cases_ = [c for c in helpful if c['step1_belief'] is not None and fn(c['step1_belief'])]
        bc_ = [c['belief_change'] for c in cases_ if c['belief_change'] is not None]
        inc_ = [x for x in bc_ if x > 0]
        dec_ = [x for x in bc_ if x < 0]
        mean_ = statistics.mean(bc_) if bc_ else float('nan')
        pct_inc = len(inc_)/len(bc_)*100 if bc_ else 0
        pct_dec = len(dec_)/len(bc_)*100 if bc_ else 0
        print(f"  {label:<38} {len(cases_):>4} {mean_:>+12.3f} {pct_inc:>5.1f}% {pct_dec:>5.1f}%")

    # ── 1d. Cases where belief changed WITHOUT answer changing ────────────────
    _section("1d. KEY PATTERNS — ANSWER CHANGE vs. CONFIDENCE TRAJECTORY")

    print(f"""
  Note: All cases in persuasion_examples.json changed their answer (that is
  the definition of persuasion). Within those, we look at whether confidence
  went up, down, or stayed the same after the AI response.
  """)

    for label, cases in [("Harmful (C→I)", harmful), ("Helpful (I→C)", helpful)]:
        bc = [c['belief_change'] for c in cases if c['belief_change'] is not None]
        n = len(bc)
        up = sum(1 for x in bc if x > 0)
        down = sum(1 for x in bc if x < 0)
        same = sum(1 for x in bc if x == 0)
        print(f"  {label}:")
        print(f"    Changed answer AND became more confident  : {up}/{n} ({up/n*100:.1f}%)")
        print(f"    Changed answer BUT became less confident  : {down}/{n} ({down/n*100:.1f}%)")
        print(f"    Changed answer, confidence unchanged      : {same}/{n} ({same/n*100:.1f}%)")
        print()

    # ── 1e. Per-model summary ─────────────────────────────────────────────────
    _section("1e. BELIEF CHANGE BY AI MODEL")

    model_harm = defaultdict(list)
    model_help = defaultdict(list)
    for c in harmful:
        if c['belief_change'] is not None:
            model_harm[short_model(c['model'])].append(c['belief_change'])
    for c in helpful:
        if c['belief_change'] is not None:
            model_help[short_model(c['model'])].append(c['belief_change'])

    all_models = sorted(set(list(model_harm.keys()) + list(model_help.keys())))
    print(f"\n  {'Model':<40} {'Harmful N':>9} {'Mean Δ':>8} {'Helpful N':>9} {'Mean Δ':>8} {'Net Δ':>8}")
    print(f"  {'-'*88}")
    for m in all_models:
        hm = model_harm.get(m, [])
        hp = model_help.get(m, [])
        hm_mean = statistics.mean(hm) if hm else float('nan')
        hp_mean = statistics.mean(hp) if hp else float('nan')
        net = hp_mean - hm_mean if hm and hp else float('nan')
        print(f"  {m:<40} {len(hm):>9} {hm_mean:>+7.3f} {len(hp):>9} {hp_mean:>+7.3f} {net:>+7.3f}")


# ── (2) Relationship with persuasive examples ──────────────────────────────────

def run_examples(data):
    _header("BELIEF CHANGE ANALYSIS — RELATIONSHIP WITH PERSUASIVE EXAMPLES")

    harmful = data['cases']['harmful_persuasion']
    helpful = data['cases']['helpful_persuasion']

    # ── 2a. Highlights as evidence of engagement ──────────────────────────────
    _section("2a. HIGHLIGHTS — DID ANNOTATION ENGAGEMENT PREDICT BELIEF CHANGE?")

    print(f"\n  Annotators could highlight passages from the AI response they found notable.")
    print(f"  We check whether highlighting correlates with larger belief shifts.\n")

    for label, cases in [("Harmful (C→I)", harmful), ("Helpful (I→C)", helpful)]:
        hl = [c for c in cases if c['num_highlights'] > 0]
        no_hl = [c for c in cases if c['num_highlights'] == 0]
        bc_hl = [c['belief_change'] for c in hl if c['belief_change'] is not None]
        bc_no = [c['belief_change'] for c in no_hl if c['belief_change'] is not None]
        print(f"  {label}:")
        _stats(bc_hl, f"  With highlights    (n={len(hl)})", indent=4)
        _stats(bc_no, f"  Without highlights (n={len(no_hl)})", indent=4)
        # Average number of highlights among those who highlighted
        if hl:
            avg_n = statistics.mean(c['num_highlights'] for c in hl)
            print(f"    Mean highlights per highlighted case: {avg_n:.1f}")
        print()

    # Highlight count vs. belief-change correlation proxy
    print("  Highlight count vs. |Δbelief| (all persuasion cases):")
    all_hl_data = [(c['num_highlights'], abs(c['belief_change']))
                   for c, _ in all_cases(data)
                   if c['belief_change'] is not None]
    bins = [(0, 0), (1, 1), (2, 3), (4, 100)]
    for lo, hi in bins:
        subset = [b for h, b in all_hl_data if lo <= h <= hi]
        mean_b = statistics.mean(subset) if subset else float('nan')
        label = f"{lo}" if lo == hi else (f"{lo}–{hi}" if hi < 100 else f"{lo}+")
        print(f"    {label} highlights: n={len(subset):3d}  mean |Δbelief|={mean_b:.3f}")

    # ── 2b. Annotator reasoning — why did people change? ─────────────────────
    _section("2b. ANNOTATOR REASONING — THEMES BEHIND BELIEF CHANGE")

    # Keyword categories for reasoning text
    keyword_categories = {
        "AI authority / trust": [
            r'\bai\b', r'model', r'thought.*knew', r'knew more', r'trusted', r'trusting',
            r'confident.*ai', r'ai.*confident', r'ai.*right', r'ai.*correct',
            r'ai.*detailed', r'more detailed', r'better explanation'
        ],
        "Misleading detail": [
            r'mislead', r'confus', r'distract', r'threw me off', r'led me',
            r'disruption', r'mention', r'caught my attention', r'highlighted'
        ],
        "Self-doubt": [
            r'doubt', r'unsure', r'second.guess', r'not sure', r'wasn.t sure',
            r'initially had', r'changed my mind', r'persuad', r'swayed',
            r'gut feeling', r'should have stuck', r'should.*trust'
        ],
        "Evidence / reasoning cited": [
            r'evidence', r'guideline', r'study', r'literature', r'data',
            r'research', r'based on', r'according to', r'cited', r'recommend'
        ],
        "Acknowledged AI error": [
            r'wrong', r'incorrect', r'error', r'mistake', r'ai.*wrong',
            r'shouldn.t have', r'regret', r'should not have'
        ],
        "No / vague reasoning": [
            r'^$', r'n/?a', r'not sure why', r'unsure why', r'unclear'
        ],
    }

    def categorise_reasoning(text):
        if not text or not text.strip():
            return ['No / vague reasoning']
        text_lower = text.lower()
        matched = []
        for cat, patterns in keyword_categories.items():
            if cat == 'No / vague reasoning':
                continue
            if any(re.search(p, text_lower) for p in patterns):
                matched.append(cat)
        if not matched:
            matched = ['Other / unclassified']
        return matched

    print(f"\n  Reasoning themes across all persuasion cases (multi-label):\n")
    cat_bc = defaultdict(list)
    for case, ptype in all_cases(data):
        cats = categorise_reasoning(case.get('reasoning', ''))
        bc = case.get('belief_change')
        for cat in cats:
            cat_bc[cat].append((bc, ptype))

    # Sort by frequency
    cat_counts = {cat: len(v) for cat, v in cat_bc.items()}
    total_ann = sum(1 for _ in all_cases(data))
    print(f"  {'Theme':<35} {'N':>5} {'%':>6}  {'Mean Δ':>8}  {'Harmful':>8} {'Helpful':>8}")
    print(f"  {'-'*80}")
    for cat in sorted(cat_counts, key=lambda x: cat_counts[x], reverse=True):
        entries = cat_bc[cat]
        bc_vals = [bc for bc, _ in entries if bc is not None]
        n_harm = sum(1 for _, p in entries if p == 'harmful')
        n_help = sum(1 for _, p in entries if p == 'helpful')
        mean_bc = statistics.mean(bc_vals) if bc_vals else float('nan')
        pct = len(entries) / total_ann * 100
        print(f"  {cat:<35} {len(entries):>5} {pct:>5.1f}%  {mean_bc:>+8.3f}  {n_harm:>8} {n_help:>8}")

    # ── 2c. Reasoning present vs. absent ─────────────────────────────────────
    _section("2c. CASES WITH vs. WITHOUT ANNOTATOR REASONING")

    print()
    for label, cases in [("Harmful (C→I)", harmful), ("Helpful (I→C)", helpful)]:
        with_r = [c for c in cases if c.get('has_reasoning') and c.get('reasoning', '').strip()]
        no_r = [c for c in cases if not (c.get('has_reasoning') and c.get('reasoning', '').strip())]
        bc_w = [c['belief_change'] for c in with_r if c['belief_change'] is not None]
        bc_n = [c['belief_change'] for c in no_r if c['belief_change'] is not None]
        print(f"  {label}:")
        _stats(bc_w, f"  Provided reasoning    (n={len(with_r)})", indent=4)
        _stats(bc_n, f"  No reasoning provided (n={len(no_r)})", indent=4)
        print()

    # ── 2d. Demographic breakdown ─────────────────────────────────────────────
    _section("2d. DEMOGRAPHICS — WHO IS MOST SUSCEPTIBLE TO HARMFUL PERSUASION?")

    # Expertise / years of practice for harmful cases
    def parse_years(val):
        if not val:
            return None
        try:
            return float(str(val).split()[0])
        except ValueError:
            return None

    print(f"\n  Years of practice vs. mean |Δbelief| in HARMFUL cases:")
    yop_bins = [
        ("< 5 years",  lambda y: y < 5),
        ("5–15 years", lambda y: 5 <= y < 15),
        ("15–30 yrs",  lambda y: 15 <= y < 30),
        ("30+ years",  lambda y: y >= 30),
    ]
    for label, fn in yop_bins:
        matched = []
        for c in harmful:
            y = parse_years(c.get('demographics', {}).get('years_of_practice'))
            if y is not None and fn(y):
                bc = c.get('belief_change')
                if bc is not None:
                    matched.append(abs(bc))
        mean_abs = statistics.mean(matched) if matched else float('nan')
        print(f"    {label:<15}: n={len(matched):2d}  mean |Δbelief|={mean_abs:.3f}")

    print(f"\n  Years of practice vs. mean |Δbelief| in HELPFUL cases:")
    for label, fn in yop_bins:
        matched = []
        for c in helpful:
            y = parse_years(c.get('demographics', {}).get('years_of_practice'))
            if y is not None and fn(y):
                bc = c.get('belief_change')
                if bc is not None:
                    matched.append(abs(bc))
        mean_abs = statistics.mean(matched) if matched else float('nan')
        print(f"    {label:<15}: n={len(matched):2d}  mean |Δbelief|={mean_abs:.3f}")

    # Expertise speciality
    print(f"\n  Expertise speciality breakdown (harmful cases):")
    spec_bc = defaultdict(list)
    for c in harmful:
        spec = c.get('demographics', {}).get('expertise', '').strip()
        spec = spec if spec else 'Unknown'
        bc = c.get('belief_change')
        if bc is not None:
            spec_bc[spec].append(bc)
    for spec in sorted(spec_bc, key=lambda x: len(spec_bc[x]), reverse=True):
        vals = spec_bc[spec]
        mean_v = statistics.mean(vals)
        spec_label = (spec[:45] + '…') if len(spec) > 46 else spec
        print(f"    {spec_label:<46}: n={len(vals):2d}  mean Δbelief={mean_v:+.3f}")

    # ── 2e. Extreme belief-change examples ────────────────────────────────────
    _section("2e. EXTREME CASES — LARGEST HARMFUL BELIEF SWINGS")

    print(f"\n  Top harmful cases by absolute belief change (answer flipped correct→incorrect):\n")
    sorted_harm = sorted(
        [c for c in harmful if c['belief_change'] is not None],
        key=lambda x: abs(x['belief_change']), reverse=True
    )
    for i, c in enumerate(sorted_harm[:8], 1):
        bc = c['belief_change']
        direction = "↑" if bc > 0 else ("↓" if bc < 0 else "=")
        model = short_model(c['model'])
        hl_count = c['num_highlights']
        has_r = "✓" if c.get('has_reasoning') else "✗"
        s1b = c['step1_belief']
        s2b = c['step2_belief']
        print(f"  {i:2d}. Case {c['case_id']:<22} model={model:<30}")
        print(f"      belief: {s1b:.2f} → {s2b:.2f} (Δ={bc:+.3f} {direction})"
              f"  highlights={hl_count}  reasoning={has_r}")
        reasoning = (c.get('reasoning') or '').strip()
        if reasoning:
            snippet = reasoning[:120].replace('\n', ' ')
            print(f"      Reason: \"{snippet}{'…' if len(reasoning) > 120 else ''}\"")
        print()

    _section("2f. EXTREME CASES — LARGEST HELPFUL BELIEF SWINGS")

    print(f"\n  Top helpful cases by absolute belief change (answer flipped incorrect→correct):\n")
    sorted_help = sorted(
        [c for c in helpful if c['belief_change'] is not None],
        key=lambda x: abs(x['belief_change']), reverse=True
    )
    for i, c in enumerate(sorted_help[:8], 1):
        bc = c['belief_change']
        direction = "↑" if bc > 0 else ("↓" if bc < 0 else "=")
        model = short_model(c['model'])
        hl_count = c['num_highlights']
        has_r = "✓" if c.get('has_reasoning') else "✗"
        s1b = c['step1_belief']
        s2b = c['step2_belief']
        print(f"  {i:2d}. Case {c['case_id']:<22} model={model:<30}")
        print(f"      belief: {s1b:.2f} → {s2b:.2f} (Δ={bc:+.3f} {direction})"
              f"  highlights={hl_count}  reasoning={has_r}")
        reasoning = (c.get('reasoning') or '').strip()
        if reasoning:
            snippet = reasoning[:120].replace('\n', ' ')
            print(f"      Reason: \"{snippet}{'…' if len(reasoning) > 120 else ''}\"")
        print()

    # ── 2g. Asymmetry: harmful belief changes are often flat or negative ───────
    _section("2g. THE CONFIDENCE PARADOX IN HARMFUL PERSUASION")

    harm_bc = [c['belief_change'] for c in harmful if c['belief_change'] is not None]
    harm_neg_or_zero = [x for x in harm_bc if x <= 0]
    harm_pos = [x for x in harm_bc if x > 0]

    print(f"""
  In harmful persuasion cases (human changed from correct → incorrect):
    · {len(harm_pos)}/{len(harm_bc)} ({len(harm_pos)/len(harm_bc)*100:.1f}%) became MORE confident after switching to wrong answer
    · {len(harm_neg_or_zero)}/{len(harm_bc)} ({len(harm_neg_or_zero)/len(harm_bc)*100:.1f}%) stayed same or became LESS confident

  Interpretation: Many harmful persuasion cases are low-confidence switches —
  the human was already uncertain (mean initial belief={statistics.mean(c['step1_belief'] for c in harmful if c['step1_belief'] is not None):.2f}),
  the AI provided a plausible-sounding but wrong rationale, and the human
  hedged by switching without a strong conviction boost. This is distinct from
  the high-confidence harmful flips, where the AI actively convinced the human.
""")

    # Confidence profile of harmful cases
    print("  Initial vs. final confidence in harmful cases:")
    for label, key in [("step1 (initial)", "step1_belief"), ("step2 (post-AI)", "step2_belief")]:
        vals = [c[key] for c in harmful if c[key] is not None]
        _stats(vals, label, indent=4)

    print("\n  Initial vs. final confidence in helpful cases:")
    for label, key in [("step1 (initial)", "step1_belief"), ("step2 (post-AI)", "step2_belief")]:
        vals = [c[key] for c in helpful if c[key] is not None]
        _stats(vals, label, indent=4)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze belief changes connected to persuasion examples.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--mode', choices=['general', 'examples', 'all'], default='all')
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help=f'Path to persuasion_examples.json (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', default=None,
                        help='Optional file path to save output (also prints to stdout)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        return

    data = load_data(args.input)
    print(f"Loaded {data['metadata']['total_cases']} persuasion cases from {args.input}")

    if args.output:
        import sys
        import io
        buf = io.StringIO()
        orig_stdout = sys.stdout
        sys.stdout = buf

    if args.mode in ('general', 'all'):
        run_general(data)

    if args.mode in ('examples', 'all'):
        run_examples(data)

    if args.output:
        sys.stdout = orig_stdout
        content = buf.getvalue()
        print(content)
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(content)
        print(f"\nOutput saved to {args.output}")


if __name__ == '__main__':
    main()
