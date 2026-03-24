#!/usr/bin/env python3
"""
Analyze martingale reliability of the Bayesian benchmark.

Reads the principal inference output from the martingale permutation experiment
and computes per-case statistics across K permutations:
  - majority_answer:       most frequent answer across permutations
  - answer_consistency:    fraction of permutations that agree with the majority
  - mean_belief:           mean confidence across permutations
  - std_belief:            standard deviation of confidence (order sensitivity proxy)
  - majority_correct:      whether the majority answer matches the ground truth

Also reports aggregate statistics:
  - majority-vote accuracy vs. single-run accuracy
  - mean / median belief std (reliability measure)
  - fraction of cases with perfect answer consistency

Usage:
    python experiments/analyze_martingale.py \
        --input experiments/principals/usmle_sample/principal_martingale_bayesian_martingale_choices.json \
        --output experiments/principals/usmle_sample/martingale_analysis.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def parse_belief(belief_str: str) -> float | None:
    """Parse a belief string to a float in [0, 1], returning None on failure."""
    try:
        val = float(str(belief_str).strip())
        if 0.0 <= val <= 1.0:
            return val
    except (ValueError, TypeError):
        pass
    return None


def analyze_martingale(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Group results by case_id and compute per-case martingale statistics.

    Args:
        results: Flat list of principal inference results, one per (case, permutation).

    Returns:
        Dict with per-case statistics and aggregate summary.
    """
    # Group by case_id
    by_case: dict[str, list[dict]] = {}
    for r in results:
        case_id = r.get("case_id", "unknown")
        by_case.setdefault(case_id, []).append(r)

    per_case: list[dict[str, Any]] = []

    for case_id, case_results in sorted(by_case.items()):
        # Collect answers and beliefs across permutations
        answers = []
        beliefs = []
        correct_answer_idx = None

        for r in case_results:
            # Answer: prefer 'answer' field (choices_mode), fall back to 'decision'
            ans = (r.get("answer") or r.get("decision") or "").strip().upper()
            if ans:
                answers.append(ans)

            belief = parse_belief(r.get("belief", ""))
            if belief is not None:
                beliefs.append(belief)

            if correct_answer_idx is None:
                correct_answer_idx = r.get("correct_answer_idx")

        if not answers:
            continue

        # Majority vote
        counter = Counter(answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        answer_consistency = majority_count / len(answers)

        # Belief statistics
        mean_belief = statistics.mean(beliefs) if beliefs else None
        std_belief = statistics.stdev(beliefs) if len(beliefs) >= 2 else 0.0

        majority_correct = (majority_answer == str(correct_answer_idx).strip().upper()
                            if correct_answer_idx else None)

        per_case.append({
            "case_id": case_id,
            "num_permutations": len(case_results),
            "majority_answer": majority_answer,
            "answer_consistency": round(answer_consistency, 4),
            "answer_counts": dict(counter),
            "mean_belief": round(mean_belief, 4) if mean_belief is not None else None,
            "std_belief": round(std_belief, 4),
            "correct_answer_idx": correct_answer_idx,
            "majority_correct": majority_correct,
        })

    # Aggregate summary
    n_cases = len(per_case)
    if n_cases == 0:
        return {"per_case": [], "summary": {}}

    majority_correct_cases = [c for c in per_case if c["majority_correct"] is True]
    majority_accuracy = len(majority_correct_cases) / n_cases

    # Single-run accuracy: accuracy of the first permutation (perm_idx == 0)
    first_run_results = [r for r in results if r.get("permutation_idx", 0) == 0]
    single_run_correct = sum(
        1 for r in first_run_results
        if (r.get("answer") or r.get("decision") or "").strip().upper()
           == str(r.get("correct_answer_idx") or "").strip().upper()
    )
    single_run_accuracy = single_run_correct / len(first_run_results) if first_run_results else None

    std_beliefs = [c["std_belief"] for c in per_case]
    consistencies = [c["answer_consistency"] for c in per_case]
    perfect_consistency = sum(1 for c in consistencies if c == 1.0)

    summary = {
        "n_cases": n_cases,
        "majority_vote_accuracy": round(majority_accuracy, 4),
        "single_run_accuracy": round(single_run_accuracy, 4) if single_run_accuracy is not None else None,
        "mean_belief_std": round(statistics.mean(std_beliefs), 4),
        "median_belief_std": round(statistics.median(std_beliefs), 4),
        "mean_answer_consistency": round(statistics.mean(consistencies), 4),
        "fraction_perfect_consistency": round(perfect_consistency / n_cases, 4),
    }

    return {"per_case": per_case, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze martingale reliability of the Bayesian benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to principal inference output JSON from martingale experiment",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path to write the analysis JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite output file even if it already exists",
    )
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

    analysis = analyze_martingale(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2))

    # Print summary to stdout
    s = analysis["summary"]
    print(f"\n{'='*50}")
    print("MARTINGALE RELIABILITY ANALYSIS")
    print(f"{'='*50}")
    print(f"Cases analyzed:              {s['n_cases']}")
    print(f"Majority-vote accuracy:      {s['majority_vote_accuracy']:.1%}")
    if s["single_run_accuracy"] is not None:
        print(f"Single-run accuracy:         {s['single_run_accuracy']:.1%}")
        improvement = s["majority_vote_accuracy"] - s["single_run_accuracy"]
        print(f"Accuracy improvement:        {improvement:+.1%}")
    print(f"Mean belief std (ordering):  {s['mean_belief_std']:.4f}")
    print(f"Median belief std (ordering):{s['median_belief_std']:.4f}")
    print(f"Mean answer consistency:     {s['mean_answer_consistency']:.1%}")
    print(f"Fraction perfect consistency:{s['fraction_perfect_consistency']:.1%}")
    print(f"{'='*50}")
    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
