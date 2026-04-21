#!/usr/bin/env python3
"""Analyze principal decision accuracy and shifts across baseline, framing, and information conditions.

Produces two tables:
  1. Decision accuracy per agent × condition × prompt type
  2. Decision shifts vs baseline per agent × condition × prompt type

Usage:
    python experiments/analyze_principal_accuracy.py
    python experiments/analyze_principal_accuracy.py --principals-dir experiments/principals/usmle_sample
    python experiments/analyze_principal_accuracy.py --output experiments/analysis/accuracy_shifts.json
    python experiments/analyze_principal_accuracy.py --detail
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR    = Path(__file__).parent
_PROJECT_DIR   = _SCRIPT_DIR.parent
PRINCIPALS_DIR = str(_PROJECT_DIR / "experiments/principals/usmle_sample")
DEFAULT_OUTPUT = str(_PROJECT_DIR / "experiments/analysis/accuracy_shifts.json")

AGENTS      = [
    "claude", "deepseek", "gemini", "gpt",
    "llama-small", "llama-sft", "llama-dpo",           # llama 7B family
    "llama", "llama-medium-sft", "llama-medium-dpo",   # llama 70B family
    "olmo", "olmo-sft", "olmo-dpo",                    # olmo 7B family
    "olmo-large", "olmo-large-sft", "olmo-large-dpo",  # olmo-large family
]
CONDITIONS  = ["baseline", "framing", "information"]
PROMPT_TYPES = ["bayesian", "behavioral"]

_FILE_PATTERNS = {
    "baseline":    "baseline/principal_{a}_{pt}_choices.json",
    "framing":     "framing/principal_framing_{a}_gt_factual_agg_{pt}_choices.json",
    "information": "information/principal_information_{a}_gt_factual_agg_{pt}_choices.json",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records(principals_dir: str, agent: str, condition: str, prompt_type: str) -> dict[str, dict] | None:
    pattern = _FILE_PATTERNS[condition]
    path = Path(principals_dir) / pattern.format(a=agent, pt=prompt_type)
    if not path.exists():
        return None
    entries = json.loads(path.read_text())
    return {e["case_id"]: e for e in entries}


def load_all(principals_dir: str) -> dict:
    """Return data[agent][condition][prompt_type] = {case_id: entry} | None."""
    data: dict = {}
    for agent in AGENTS:
        data[agent] = {}
        for cond in CONDITIONS:
            data[agent][cond] = {}
            for pt in PROMPT_TYPES:
                data[agent][cond][pt] = load_records(principals_dir, agent, cond, pt)
    return data


# ── Decision parsing ──────────────────────────────────────────────────────────

def parse_decision(raw: str | None, options: dict) -> str | None:
    """Normalise a principal decision string to a single option letter.

    Handles:
      - Plain letters already matching an option key  ("B", "F")
      - Full-text answers prefixed with the letter    ("C. Pulmonary Hypoplasia")
    Returns None for empty / truly unparseable responses.
    """
    if not raw:
        return None
    d = raw.strip().upper()
    if not d:
        return None
    if d in options:
        return d
    m = re.match(r'^([A-Z])[.\s:]', d)
    if m and m.group(1) in options:
        return m.group(1)
    return None


def get_decision(entry: dict) -> str | None:
    return parse_decision(entry.get("decision"), entry.get("options") or {})


def get_correct(entry: dict) -> str | None:
    v = (entry.get("correct_answer_idx") or "").strip().upper()
    return v if v else None


# ── Accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(records: dict[str, dict] | None) -> dict | None:
    if records is None:
        return None
    total   = len(records)
    correct = sum(
        1 for e in records.values()
        if get_decision(e) is not None and get_decision(e) == get_correct(e)
    )
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


# ── Decision shifts ───────────────────────────────────────────────────────────

def compute_shifts(
    baseline: dict[str, dict] | None,
    experiment: dict[str, dict] | None,
) -> dict | None:
    if baseline is None or experiment is None:
        return None
    common = set(baseline) & set(experiment)
    shifted = [
        (cid,
         get_decision(baseline[cid]),
         get_decision(experiment[cid]))
        for cid in common
        if get_decision(baseline[cid]) != get_decision(experiment[cid])
    ]
    plus_correct = sum(
        1 for cid, _, ed in shifted
        if ed is not None and ed == get_correct(experiment[cid])
    )
    minus_correct = sum(
        1 for cid, bd, _ in shifted
        if bd is not None and bd == get_correct(baseline[cid])
    )
    net     = plus_correct - minus_correct
    n_cases = len(common)
    return {
        "n_shifted":     len(shifted),
        "n_cases":       n_cases,
        "shift_rate":    len(shifted) / n_cases if n_cases else None,
        "plus_correct":  plus_correct,
        "minus_correct": minus_correct,
        "net":           net,
        "net_acc_delta": net / n_cases if n_cases else None,
    }


# ── Disagreements between Bayesian and Behavioral ─────────────────────────────

def compute_disagreements(
    bayesian: dict[str, dict] | None,
    behavioral: dict[str, dict] | None,
) -> dict | None:
    """Count cases where Bayesian and Behavioral principals give different decisions."""
    if bayesian is None or behavioral is None:
        return None
    common = set(bayesian) & set(behavioral)
    if not common:
        return None
    disagreed = [
        cid for cid in common
        if get_decision(bayesian[cid]) != get_decision(behavioral[cid])
    ]
    n = len(common)
    # Among disagreements, count cases where Bayesian was right / Behavioral was right
    bay_right = sum(
        1 for cid in disagreed
        if get_decision(bayesian[cid]) is not None
        and get_decision(bayesian[cid]) == get_correct(bayesian[cid])
    )
    beh_right = sum(
        1 for cid in disagreed
        if get_decision(behavioral[cid]) is not None
        and get_decision(behavioral[cid]) == get_correct(behavioral[cid])
    )
    return {
        "n_disagreed":       len(disagreed),
        "n_cases":           n,
        "disagreement_rate": len(disagreed) / n if n else None,
        "bay_right_when_disagree": bay_right,
        "beh_right_when_disagree": beh_right,
    }


# ── Build results ─────────────────────────────────────────────────────────────

def build_results(data: dict) -> dict:
    accuracy:      dict = {}
    shifts:        dict = {}
    disagreements: dict = {}

    for agent in AGENTS:
        accuracy[agent]      = {}
        shifts[agent]        = {}
        disagreements[agent] = {}
        for cond in CONDITIONS:
            accuracy[agent][cond] = {}
            for pt in PROMPT_TYPES:
                accuracy[agent][cond][pt] = compute_accuracy(data[agent][cond][pt])

            disagreements[agent][cond] = compute_disagreements(
                data[agent][cond]["bayesian"],
                data[agent][cond]["behavioral"],
            )

        for cond in ["framing", "information"]:
            shifts[agent][cond] = {}
            for pt in PROMPT_TYPES:
                shifts[agent][cond][pt] = compute_shifts(
                    data[agent]["baseline"][pt],
                    data[agent][cond][pt],
                )

    return {"accuracy": accuracy, "shifts": shifts, "disagreements": disagreements}


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt_acc(v: dict | None) -> str:
    if v is None:
        return "  N/A  "
    return f"{v['accuracy']:.3f}"

def _fmt_delta(base: dict | None, exp: dict | None) -> str:
    if base is None or exp is None or base["accuracy"] is None or exp["accuracy"] is None:
        return "  N/A  "
    return f"{exp['accuracy'] - base['accuracy']:+.3f}"


def print_accuracy_table(results: dict, detail: bool = False) -> None:
    acc = results["accuracy"]

    print("=== TABLE 1: Decision Accuracy ===\n")
    header = f"{'Agent':<12}  {'base_bay':>8}  {'base_beh':>8}  {'fram_bay':>8}  {'fram_beh':>8}  {'info_bay':>8}  {'info_beh':>8}"
    print(header)
    print("-" * len(header))
    for agent in AGENTS:
        row = f"{agent:<12}"
        for cond in CONDITIONS:
            for pt in PROMPT_TYPES:
                row += f"  {_fmt_acc(acc[agent][cond][pt]):>8}"
        print(row)

    print("\n--- Deltas vs Baseline ---\n")
    header2 = f"{'Agent':<12}  {'fram_bay':>8}  {'fram_beh':>8}  {'info_bay':>8}  {'info_beh':>8}"
    print(header2)
    print("-" * len(header2))
    for agent in AGENTS:
        row = f"{agent:<12}"
        for cond in ["framing", "information"]:
            for pt in PROMPT_TYPES:
                row += f"  {_fmt_delta(acc[agent]['baseline'][pt], acc[agent][cond][pt]):>8}"
        print(row)

    if detail:
        print("\n--- Per-Condition Averages (across agents with data) ---\n")
        for cond in CONDITIONS:
            for pt in PROMPT_TYPES:
                vals = [acc[a][cond][pt]["accuracy"] for a in AGENTS if acc[a][cond][pt] is not None]
                if vals:
                    print(f"  {cond:<12} {pt:<10}  mean={sum(vals)/len(vals):.3f}  n={len(vals)}")


def print_prompt_type_comparison(results: dict) -> None:
    """TABLE 3: Bayesian vs Behavioral accuracy per agent × condition, with delta."""
    acc = results["accuracy"]

    print("\n=== TABLE 3: Bayesian vs Behavioral Accuracy Comparison ===\n")
    for cond in CONDITIONS:
        print(f"--- {cond.upper()} ---")
        hdr = f"{'Agent':<20}  {'Bayesian':>8}  {'Behavioral':>10}  {'Bay-Beh':>8}"
        print(hdr)
        print("-" * len(hdr))
        bay_vals, beh_vals = [], []
        for agent in AGENTS:
            bay = acc[agent][cond]["bayesian"]
            beh = acc[agent][cond]["behavioral"]
            if bay is None and beh is None:
                continue
            bay_str = f"{bay['accuracy']:.3f}" if bay else "  N/A  "
            beh_str = f"{beh['accuracy']:.3f}" if beh else "  N/A  "
            if bay and beh and bay["accuracy"] is not None and beh["accuracy"] is not None:
                delta = bay["accuracy"] - beh["accuracy"]
                delta_str = f"{delta:+.3f}"
                bay_vals.append(bay["accuracy"])
                beh_vals.append(beh["accuracy"])
            else:
                delta_str = "  N/A  "
            print(f"{agent:<20}  {bay_str:>8}  {beh_str:>10}  {delta_str:>8}")
        if bay_vals and beh_vals:
            mean_bay = sum(bay_vals) / len(bay_vals)
            mean_beh = sum(beh_vals) / len(beh_vals)
            print("-" * len(hdr))
            print(f"{'MEAN':<20}  {mean_bay:>8.3f}  {mean_beh:>10.3f}  {mean_bay - mean_beh:>+8.3f}")
        print()


def print_disagreement_table(results: dict) -> None:
    """TABLE 4: Bayesian vs Behavioral disagreement rate per agent × condition."""
    dis = results["disagreements"]

    print("\n=== TABLE 4: Bayesian vs Behavioral Disagreement Rates ===\n")
    hdr = f"{'Agent':<20}  {'Cond':<12}  {'Disagree':>8}  {'Cases':>5}  {'Rate':>6}  {'Bay✓':>5}  {'Beh✓':>5}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for cond in CONDITIONS:
        rate_vals: list[float] = []
        for agent in AGENTS:
            d = dis[agent][cond]
            if d is None:
                print(f"{agent:<20}  {cond:<12}  N/A")
                continue
            rate_vals.append(d["disagreement_rate"])
            print(
                f"{agent:<20}  {cond:<12}"
                f"  {d['n_disagreed']:>8}  {d['n_cases']:>5}"
                f"  {d['disagreement_rate']:>6.1%}"
                f"  {d['bay_right_when_disagree']:>5}  {d['beh_right_when_disagree']:>5}"
            )
        if rate_vals:
            print(sep)
            print(f"{'MEAN':<20}  {cond:<12}  {'':>8}  {'':>5}  {sum(rate_vals)/len(rate_vals):>6.1%}")
        print()


def print_shift_table(results: dict) -> None:
    sh = results["shifts"]

    print("\n=== TABLE 2: Decision Shifts vs Baseline ===")
    hdr = f"{'Agent':<12}  {'Cond':<12}  {'Shifted':>7}  {'Cases':>5}  {'Shift%':>6}  {'+Corr':>6}  {'-Corr':>6}  {'Net':>4}  {'NetAcc':>7}"
    sep = "-" * len(hdr)

    for pt_label, pt in [("BAYESIAN", "bayesian"), ("BEHAVIORAL", "behavioral")]:
        print(f"\n--- {pt_label} ---")
        print(hdr)
        print(sep)
        for agent in AGENTS:
            for cond in ["framing", "information"]:
                s = sh[agent][cond][pt]
                if s is None:
                    print(f"{agent:<12}  {cond:<12}  N/A")
                    continue
                net_sign = "+" if s["net"] >= 0 else ""
                print(
                    f"{agent:<12}  {cond:<12}"
                    f"  {s['n_shifted']:>7}  {s['n_cases']:>5}"
                    f"  {s['shift_rate']:>6.1%}"
                    f"  {s['plus_correct']:>6}  {s['minus_correct']:>6}"
                    f"  {net_sign}{s['net']:>3}  {s['net_acc_delta']:>+7.3f}"
                )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--principals-dir", default=PRINCIPALS_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--detail", action="store_true", help="Print per-condition averages")
    args = parser.parse_args()

    data    = load_all(args.principals_dir)
    results = build_results(data)

    print_accuracy_table(results, detail=args.detail)
    print_prompt_type_comparison(results)
    print_disagreement_table(results)
    print_shift_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    main()
