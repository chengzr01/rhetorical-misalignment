#!/usr/bin/env python3
"""Analyze principal model decision accuracy on USMLE questions.

Compares the principal's final answer to the correct answer for each case,
broken down by:
  - agent model  (source of information shown to the principal)
  - condition    (choices | all_claims | factual | unfactual | framing)
  - prompt_type  (bayesian | behavioral)

For *_choices files the decision field already contains the answer letter.
For all other files the answer letter is extracted from raw_principal_response
via a set of regex patterns.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).parent
PRINCIPALS_DIR = str(_SCRIPT_DIR / "principals/usmle_sample")


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_letter_from_raw(raw: str) -> Optional[str]:
    """Extract the final answer letter from a raw principal response.

    Tries several patterns in order of specificity.  Returns the last
    match of the highest-priority pattern found, or None if nothing matches.
    """
    if not raw:
        return None

    # 1. Explicit XML tag  <answer>X</answer>
    m = re.findall(r"<answer>\s*([A-F])\s*</answer>", raw, re.IGNORECASE)
    if m:
        return m[-1].upper()

    # 2. "(Option X)" or "(option X)"
    m = re.findall(r"\(option\s+([A-F])\)", raw, re.IGNORECASE)
    if m:
        return m[-1].upper()

    # 3. "correct answer is (…) X" / "answer is X" / "Answer: X"
    m = re.findall(
        r"(?:correct\s+answer|answer)\s*(?:is|:)\s*(?:\w+\s+){0,6}([A-F])\b",
        raw, re.IGNORECASE,
    )
    if m:
        return m[-1].upper()

    # 4. Loose "option X" anywhere – last occurrence
    m = re.findall(r"\boption\s+([A-F])\b", raw, re.IGNORECASE)
    if m:
        return m[-1].upper()

    return None


# ── File name parsing ─────────────────────────────────────────────────────────

_KNOWN_CONDITIONS = ["all_claims", "unfactual", "factual", "choices"]
_KNOWN_PROMPTS    = ["bayesian", "behavioral"]


def parse_filename(filepath: str) -> tuple[str, str, str]:
    """Return (agent, condition, prompt_type) inferred from the file name.

    Naming conventions
    ------------------
    Choices   : principal_[agent]_[prompt_type]_choices.json
    Others    : principal_[agent]_[condition]_[prompt_type].json
    Framing   : principal_framing_[agent]_gt_factual_agg_[prompt_type].json

    Examples
    --------
    principal_claude_bayesian_choices.json          -> (claude,      choices,    bayesian)
    principal_claude_all_claims_bayesian.json       -> (claude,      all_claims, bayesian)
    principal_claude_unfactual_behavioral.json      -> (claude,      unfactual,  behavioral)
    principal_framing_claude_gt_factual_agg_*.json  -> (claude,      framing,    *)
    """
    stem = Path(filepath).stem  # strip directory + extension
    # Remove leading "principal_"
    if stem.startswith("principal_"):
        stem = stem[len("principal_"):]

    # ── framing: starts with "framing_" ──────────────────────────────────────
    if stem.startswith("framing_"):
        inner = stem[len("framing_"):]       # e.g. claude_gt_factual_agg_bayesian
        prompt_type = "unknown"
        for pt in _KNOWN_PROMPTS:
            if inner.endswith(f"_{pt}"):
                prompt_type = pt
                inner = inner[: -(len(pt) + 1)]
                break
        m = re.match(r"^(.+?)_gt_", inner)   # e.g. claude_gt_factual_agg
        agent = m.group(1) if m else inner.split("_")[0]
        return agent, "framing", prompt_type

    # ── choices: [agent]_[prompt_type]_choices ───────────────────────────────
    if stem.endswith("_choices"):
        inner = stem[: -len("_choices")]     # e.g. claude_bayesian
        for pt in _KNOWN_PROMPTS:
            if inner.endswith(f"_{pt}"):
                agent = inner[: -(len(pt) + 1)]
                return agent, "choices", pt
        return inner, "choices", "unknown"

    # ── all_claims / factual / unfactual: [agent]_[condition]_[prompt_type] ──
    prompt_type = "unknown"
    for pt in _KNOWN_PROMPTS:
        if stem.endswith(f"_{pt}"):
            prompt_type = pt
            stem = stem[: -(len(pt) + 1)]
            break

    for cond in ["all_claims", "unfactual", "factual"]:
        if stem.endswith(f"_{cond}"):
            agent = stem[: -(len(cond) + 1)]
            return agent, cond, prompt_type

    # fallback
    return stem, "unknown", prompt_type


# ── Accuracy computation ──────────────────────────────────────────────────────

def compute_accuracy(entries: list, is_choices: bool) -> dict:
    """Return accuracy statistics for a list of principal entries."""
    total     = len(entries)
    correct   = 0
    extracted = 0
    skipped   = 0

    for entry in entries:
        gt = (entry.get("correct_answer_idx") or "").strip().upper()
        if not gt:
            skipped += 1
            continue

        if is_choices:
            decision = (entry.get("decision") or "").strip().upper()
        else:
            decision = extract_letter_from_raw(entry.get("raw_principal_response", ""))
            if decision is None:
                skipped += 1
                continue
            extracted += 1

        if decision == gt:
            correct += 1

    evaluated = total - skipped
    accuracy  = correct / evaluated if evaluated > 0 else None

    return {
        "total":     total,
        "evaluated": evaluated,
        "correct":   correct,
        "skipped":   skipped,
        "accuracy":  accuracy,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_and_analyze(principals_dir: str) -> list[dict]:
    files = sorted(Path(principals_dir).glob("principal_*.json"))
    if not files:
        print(f"[ERROR] No principal JSON files found in {principals_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for filepath in files:
        agent, condition, prompt_type = parse_filename(str(filepath))
        is_choices = condition == "choices"

        with open(filepath) as f:
            entries = json.load(f)

        metrics = compute_accuracy(entries, is_choices)
        results.append({
            "agent":      agent,
            "condition":  condition,
            "prompt_type": prompt_type,
            "file":       filepath.name,
            **metrics,
        })

    return results


def print_detail_table(results: list[dict]) -> None:
    """Print per-file accuracy table."""
    results = sorted(results, key=lambda r: (r["condition"], r["agent"], r["prompt_type"]))

    hdr = f"{'Agent':<16} {'Condition':<14} {'Prompt':<12} {'Total':>7} {'Eval':>7} {'Correct':>8} {'Accuracy':>10} {'Skipped':>8}"
    print("\n=== Per-File Accuracy ===")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "N/A"
        print(
            f"{r['agent']:<16} {r['condition']:<14} {r['prompt_type']:<12}"
            f" {r['total']:>7} {r['evaluated']:>7} {r['correct']:>8} {acc:>10} {r['skipped']:>8}"
        )


def print_summary_by_condition(results: list[dict]) -> None:
    """Print accuracy aggregated by (condition, prompt_type)."""
    from collections import defaultdict

    groups: dict[tuple, dict] = defaultdict(lambda: {"total": 0, "evaluated": 0, "correct": 0})
    for r in results:
        key = (r["condition"], r["prompt_type"])
        groups[key]["total"]     += r["total"]
        groups[key]["evaluated"] += r["evaluated"]
        groups[key]["correct"]   += r["correct"]

    print("\n=== Summary by Condition × Prompt Type ===")
    hdr = f"{'Condition':<14} {'Prompt':<12} {'Eval':>7} {'Correct':>8} {'Accuracy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups):
        cond, pt = key
        g = groups[key]
        acc = f"{g['correct']/g['evaluated']:.3f}" if g["evaluated"] > 0 else "N/A"
        print(f"{cond:<14} {pt:<12} {g['evaluated']:>7} {g['correct']:>8} {acc:>10}")


def print_summary_by_agent(results: list[dict]) -> None:
    """Print accuracy aggregated by (agent, condition)."""
    from collections import defaultdict

    groups: dict[tuple, dict] = defaultdict(lambda: {"total": 0, "evaluated": 0, "correct": 0})
    for r in results:
        key = (r["agent"], r["condition"])
        groups[key]["total"]     += r["total"]
        groups[key]["evaluated"] += r["evaluated"]
        groups[key]["correct"]   += r["correct"]

    print("\n=== Summary by Agent × Condition (both prompt types combined) ===")
    hdr = f"{'Agent':<16} {'Condition':<14} {'Eval':>7} {'Correct':>8} {'Accuracy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups):
        agent, cond = key
        g = groups[key]
        acc = f"{g['correct']/g['evaluated']:.3f}" if g["evaluated"] > 0 else "N/A"
        print(f"{agent:<16} {cond:<14} {g['evaluated']:>7} {g['correct']:>8} {acc:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--principals-dir", default=PRINCIPALS_DIR,
        help="Directory containing principal JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Print per-file breakdown in addition to summary tables",
    )
    args = parser.parse_args()

    results = load_and_analyze(args.principals_dir)

    if args.detail:
        print_detail_table(results)

    print_summary_by_agent(results)
    print_summary_by_condition(results)


if __name__ == "__main__":
    main()
