#!/usr/bin/env python3
"""
Analyze sufficient-statistics (paraphrase invariance) test results.

For each case the principal ran on K+1 records:
  - 1 "original"   record  (bullet-point claims)
  - K "paraphrase" records (independently paraphrased prose)

The K paraphrases are treated as a Monte-Carlo sample of wording-invariant
inference.  Their majority vote approximates the model's "true" Bayesian
posterior; comparing the original against this estimate reveals whether the
model's inference is sensitive to surface form rather than content.

Per-case statistics:
  original_answer / original_belief / original_correct
  paraphrase_answers / paraphrase_beliefs
  majority_answer_among_paraphrases   — most frequent answer across K paraphrases
  answer_consistency_among_paraphrases — fraction of paraphrases matching majority
  mean_belief_among_paraphrases / std_belief_among_paraphrases
  original_agrees_with_majority       — does original match paraphrase majority?
  belief_delta                        — |original_belief - mean_paraphrase_belief|
  majority_correct / original_correct

Aggregate summary:
  original_accuracy / paraphrase_majority_accuracy
  original_agrees_with_majority_rate
  mean_answer_consistency_among_paraphrases
  mean_belief_std_among_paraphrases   — paraphrase inner reliability
  mean_belief_delta                   — original vs. paraphrase mean

Usage:
    python experiments/analyze_paraphrase.py \\
        --input  experiments/principals/usmle_sample/principal_paraphrase_claude_k10_bayesian_choices.json \\
        --output experiments/principals/usmle_sample/paraphrase_analysis_claude_k10.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_belief(belief_str: str) -> float | None:
    try:
        val = float(str(belief_str).strip())
        if 0.0 <= val <= 1.0:
            return val
    except (ValueError, TypeError):
        pass
    return None


def analyze_paraphrase(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Group results by case_id, separate original from K paraphrases, compute stats.
    """
    # Group by case_id then condition
    originals: dict[str, dict] = {}
    paraphrases: dict[str, list[dict]] = {}

    for r in results:
        case_id = r.get("case_id", "unknown")
        condition = r.get("condition", "original")
        if condition == "original":
            originals[case_id] = r
        else:
            paraphrases.setdefault(case_id, []).append(r)

    all_case_ids = sorted(set(originals) | set(paraphrases))
    per_case: list[dict[str, Any]] = []

    for case_id in all_case_ids:
        orig = originals.get(case_id)
        para_list = paraphrases.get(case_id, [])

        if orig is None:
            print(f"  [WARN] case {case_id}: missing 'original' condition — skipping")
            continue
        if not para_list:
            print(f"  [WARN] case {case_id}: no 'paraphrase' records — skipping")
            continue

        gt = str(orig.get("correct_answer_idx") or "").strip().upper()

        # --- Original ---
        orig_ans = (orig.get("answer") or orig.get("decision") or "").strip().upper()
        orig_belief = parse_belief(orig.get("belief", ""))
        orig_correct = (orig_ans == gt) if gt and orig_ans else None

        # --- Paraphrases ---
        para_answers = [
            (r.get("answer") or r.get("decision") or "").strip().upper()
            for r in para_list
        ]
        para_answers = [a for a in para_answers if a]  # drop empties
        para_beliefs = [
            parse_belief(r.get("belief", "")) for r in para_list
        ]
        para_beliefs = [b for b in para_beliefs if b is not None]

        # Majority vote among paraphrases
        counter = Counter(para_answers)
        majority_ans, majority_count = counter.most_common(1)[0] if counter else ("", 0)
        answer_consistency = majority_count / len(para_answers) if para_answers else None

        mean_para_belief = statistics.mean(para_beliefs) if para_beliefs else None
        std_para_belief  = statistics.stdev(para_beliefs) if len(para_beliefs) >= 2 else 0.0

        majority_correct = (majority_ans == gt) if gt and majority_ans else None
        orig_agrees_majority = (orig_ans == majority_ans) if orig_ans and majority_ans else None

        belief_delta = (
            round(abs(orig_belief - mean_para_belief), 4)
            if orig_belief is not None and mean_para_belief is not None
            else None
        )

        per_case.append({
            "case_id":                           case_id,
            "k_paraphrases":                     len(para_list),
            # Original condition
            "original_answer":                   orig_ans,
            "original_belief":                   round(orig_belief, 4) if orig_belief is not None else None,
            "original_correct":                  orig_correct,
            # Paraphrase distribution
            "paraphrase_answers":                para_answers,
            "paraphrase_beliefs":                [round(b, 4) for b in para_beliefs],
            "majority_answer_among_paraphrases": majority_ans,
            "answer_consistency_among_paraphrases": round(answer_consistency, 4) if answer_consistency is not None else None,
            "answer_counts_among_paraphrases":   dict(counter),
            "mean_belief_among_paraphrases":     round(mean_para_belief, 4) if mean_para_belief is not None else None,
            "std_belief_among_paraphrases":      round(std_para_belief, 4),
            # Cross-condition comparison
            "original_agrees_with_majority":     orig_agrees_majority,
            "belief_delta":                      belief_delta,
            "majority_correct":                  majority_correct,
            "correct_answer_idx":                orig.get("correct_answer_idx"),
        })

    n_cases = len(per_case)
    if n_cases == 0:
        return {"per_case": [], "summary": {}}

    # --- Aggregate summary ---
    orig_correct_n    = sum(1 for c in per_case if c["original_correct"] is True)
    majority_correct_n = sum(1 for c in per_case if c["majority_correct"] is True)
    agrees_n          = sum(1 for c in per_case if c["original_agrees_with_majority"] is True)

    consistencies = [c["answer_consistency_among_paraphrases"] for c in per_case
                     if c["answer_consistency_among_paraphrases"] is not None]
    stds          = [c["std_belief_among_paraphrases"] for c in per_case]
    deltas        = [c["belief_delta"] for c in per_case if c["belief_delta"] is not None]

    # Cases where original flipped relative to paraphrase majority
    flipped = [
        {
            "case_id":          c["case_id"],
            "original_answer":  c["original_answer"],
            "majority_answer":  c["majority_answer_among_paraphrases"],
            "correct_answer":   c["correct_answer_idx"],
        }
        for c in per_case if c["original_agrees_with_majority"] is False
    ]

    summary = {
        "n_cases":                                  n_cases,
        "k_paraphrases":                            per_case[0]["k_paraphrases"] if per_case else None,
        # Accuracy
        "original_accuracy":                        round(orig_correct_n    / n_cases, 4),
        "paraphrase_majority_accuracy":             round(majority_correct_n / n_cases, 4),
        "accuracy_delta":                           round((majority_correct_n - orig_correct_n) / n_cases, 4),
        # Original vs. paraphrase majority agreement
        "original_agrees_with_majority_rate":       round(agrees_n / n_cases, 4),
        "n_original_flips":                         len(flipped),
        "flipped_cases":                            flipped,
        # Paraphrase inner reliability
        "mean_answer_consistency_among_paraphrases": round(statistics.mean(consistencies), 4) if consistencies else None,
        "mean_belief_std_among_paraphrases":        round(statistics.mean(stds), 4) if stds else None,
        "median_belief_std_among_paraphrases":      round(statistics.median(stds), 4) if stds else None,
        # Original vs. paraphrase belief shift
        "mean_belief_delta":                        round(statistics.mean(deltas), 4) if deltas else None,
        "median_belief_delta":                      round(statistics.median(deltas), 4) if deltas else None,
        "fraction_belief_delta_lt_0.1":             round(sum(1 for d in deltas if d < 0.1) / len(deltas), 4) if deltas else None,
    }

    return {"per_case": per_case, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sufficient-statistics (paraphrase invariance) test results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  required=True, metavar="PATH")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument("--force",  action="store_true", default=False)
    args = parser.parse_args()

    output_path = Path(args.output)
    if not args.force and output_path.exists():
        print(f"✓ Analysis already exists at {output_path} — skipping (use --force to overwrite)")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results = json.loads(input_path.read_text())
    if not isinstance(results, list):
        raise ValueError("Expected a list of principal inference results")
    print(f"Loaded {len(results)} results from {input_path}")

    analysis = analyze_paraphrase(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2))

    s = analysis["summary"]
    K = s.get("k_paraphrases", "?")
    w = 55
    print(f"\n{'='*w}")
    print(f"SUFFICIENT STATISTICS TEST  (K={K} paraphrases)")
    print(f"{'='*w}")
    print(f"Cases analyzed:                          {s['n_cases']}")
    print(f"--- Accuracy ---")
    print(f"  Original (bullet-point claims):        {s['original_accuracy']:.1%}")
    print(f"  Paraphrase majority vote:              {s['paraphrase_majority_accuracy']:.1%}")
    print(f"  Delta (majority - original):           {s['accuracy_delta']:+.1%}")
    print(f"--- Original vs. paraphrase majority ---")
    print(f"  Agreement rate:                        {s['original_agrees_with_majority_rate']:.1%}")
    print(f"  Answer flips:                          {s['n_original_flips']}")
    print(f"--- Paraphrase inner reliability ---")
    if s['mean_answer_consistency_among_paraphrases'] is not None:
        print(f"  Mean answer consistency:               {s['mean_answer_consistency_among_paraphrases']:.1%}")
    if s['mean_belief_std_among_paraphrases'] is not None:
        print(f"  Mean belief std (across paraphrases):  {s['mean_belief_std_among_paraphrases']:.4f}")
    print(f"--- Belief shift (original vs. mean paraphrase) ---")
    if s['mean_belief_delta'] is not None:
        print(f"  Mean belief delta:                     {s['mean_belief_delta']:.4f}")
        print(f"  Median belief delta:                   {s['median_belief_delta']:.4f}")
        print(f"  Fraction delta < 0.1:                  {s['fraction_belief_delta_lt_0.1']:.1%}")
    print(f"{'='*w}")
    if s["n_original_flips"] > 0:
        print("\nFlipped cases (original ≠ paraphrase majority):")
        for fc in s["flipped_cases"]:
            print(f"  {fc['case_id']}: orig={fc['original_answer']} "
                  f"majority={fc['majority_answer']} correct={fc['correct_answer']}")
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
