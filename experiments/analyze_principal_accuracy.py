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
import sys
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR   = Path(__file__).parent
PRINCIPALS_DIR = str(_SCRIPT_DIR / "principals/usmle_sample")
DEFAULT_OUTPUT = str(_SCRIPT_DIR / "analysis/accuracy_shifts.json")

AGENTS      = ["claude", "deepseek", "gemini", "gpt", "llama", "llama-dpo", "llama-sft", "llama-small"]
CONDITIONS  = ["baseline", "framing", "information"]
PROMPT_TYPES = ["bayesian", "behavioral"]

_FILE_PATTERNS = {
    "baseline":    "principal_{a}_{pt}_choices.json",
    "framing":     "principal_framing_{a}_gt_factual_agg_{pt}_choices.json",
    "information": "principal_information_{a}_gt_factual_agg_{pt}_choices.json",
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


# ── Accuracy ──────────────────────────────────────────────────────────────────

def compute_accuracy(records: dict[str, dict] | None) -> dict | None:
    if records is None:
        return None
    total   = len(records)
    correct = sum(
        1 for e in records.values()
        if (e.get("decision") or "").strip().upper() ==
           (e.get("correct_answer_idx") or "").strip().upper()
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
        (cid, baseline[cid]["decision"], experiment[cid]["decision"])
        for cid in common
        if baseline[cid]["decision"] != experiment[cid]["decision"]
    ]
    plus_correct = sum(
        1 for cid, _, ed in shifted
        if ed == (experiment[cid].get("correct_answer_idx") or "").strip().upper()
    )
    minus_correct = sum(
        1 for cid, bd, _ in shifted
        if bd == (baseline[cid].get("correct_answer_idx") or "").strip().upper()
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


# ── Build results ─────────────────────────────────────────────────────────────

def build_results(data: dict) -> dict:
    accuracy: dict = {}
    shifts:   dict = {}

    for agent in AGENTS:
        accuracy[agent] = {}
        shifts[agent]   = {}
        for cond in CONDITIONS:
            accuracy[agent][cond] = {}
            for pt in PROMPT_TYPES:
                accuracy[agent][cond][pt] = compute_accuracy(data[agent][cond][pt])

        for cond in ["framing", "information"]:
            shifts[agent][cond] = {}
            for pt in PROMPT_TYPES:
                shifts[agent][cond][pt] = compute_shifts(
                    data[agent]["baseline"][pt],
                    data[agent][cond][pt],
                )

    return {"accuracy": accuracy, "shifts": shifts}


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
    print_shift_table(results)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results → {out_path}")


if __name__ == "__main__":
    main()
