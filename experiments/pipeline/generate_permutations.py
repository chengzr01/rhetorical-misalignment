#!/usr/bin/env python3
"""
Generate random permutations of aggregated claims for martingale reliability testing.

For each case in an aggregated-claims JSON, produces K random permutations of the
claims list. The output is a flat list of records compatible with
principal_inference.py's --agent-cache format, so the permuted data can be fed
directly into principal_inference.py without any further preprocessing.

Usage:
    python experiments/generate_permutations.py \
        --aggregated-info experiments/aggregation/aggregated_factual.json \
        --questions experiments/questions/clinical_questions_usmle_sample.json \
        --num-permutations 10 \
        --output experiments/agents/usmle_sample/martingale_permutations.json

The output records include a 'permutation_idx' and 'claim_order' field so that
downstream analysis can group results by case and permutation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _format_claims(claims: list[dict[str, Any]], fmt: str) -> str:
    """Join a list of claim dicts into a single information string."""
    lines: list[str] = []
    for i, c in enumerate(claims, start=1):
        text = c.get("claim", "").strip()
        if not text:
            continue
        if fmt == "numbered":
            lines.append(f"{i}. {text}")
        elif fmt == "bullets":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def generate_permutations(
    aggregated_path: Path,
    questions_path: Path,
    num_permutations: int,
    claim_format: str = "bullets",
    seed: int = 42,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate K random permutations of claims per case.

    Args:
        aggregated_path:   Path to aggregated claims JSON
                           (experiments/aggregation/aggregated_factual.json).
        questions_path:    Path to clinical questions JSON for USMLE metadata.
        num_permutations:  Number of permutations (K) per case.
        claim_format:      How to join claims: "bullets", "numbered", or "plain".
        seed:              Base random seed for reproducibility.
        max_cases:         If set, limit to the first N cases.

    Returns:
        Flat list of agent-result-compatible records, one per (case, permutation).
    """
    aggregated = json.loads(aggregated_path.read_text())
    cases: dict[str, dict] = aggregated.get("cases", {})

    questions = json.loads(questions_path.read_text())
    question_lookup: dict[str, dict] = {q["id"]: q for q in questions if "id" in q}

    case_ids = sorted(cases.keys())
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    print(f"Loaded {len(cases)} cases from {aggregated_path}")
    print(f"Processing {len(case_ids)} cases × {num_permutations} permutations "
          f"= {len(case_ids) * num_permutations} records")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for case_id in case_ids:
        case = cases[case_id]
        claims = case.get("claims", [])
        n_claims = len(claims)

        q = question_lookup.get(case_id, {})
        question_text = q.get("question", case_id)
        options = q.get("options", {})
        correct_answer = q.get("answer")
        correct_answer_idx = q.get("answer_idx")
        meta_info = q.get("meta_info")

        for perm_idx in range(num_permutations):
            indices = list(range(n_claims))
            rng.shuffle(indices)
            shuffled_claims = [claims[i] for i in indices]
            information = _format_claims(shuffled_claims, claim_format)

            records.append({
                "case_id": case_id,
                "permutation_idx": perm_idx,
                "claim_order": indices,
                "agent_name": "permuted_claims",
                "agent_model": "permutation",
                "agent_context": question_text,
                "principal_context": question_text,
                "information": information,
                "dataset_type": "usmle",
                "options": options,
                "correct_answer": correct_answer,
                "correct_answer_idx": correct_answer_idx,
                "meta_info": meta_info,
                "agent_task": None,
                "agent_objective": None,
            })

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random claim permutations for martingale reliability testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--aggregated-info",
        required=True,
        metavar="PATH",
        help="Path to aggregated claims JSON (e.g. experiments/aggregation/aggregated_factual.json)",
    )
    parser.add_argument(
        "--questions",
        required=True,
        metavar="PATH",
        help="Path to clinical questions JSON for USMLE metadata",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=10,
        metavar="K",
        help="Number of random permutations per case",
    )
    parser.add_argument(
        "--claim-format",
        default="bullets",
        choices=["bullets", "numbered", "plain"],
        help="How to join claims into the information string",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N cases (default: no limit)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for the permuted records JSON",
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
        print(f"✓ Permutations already exist at {output_path} — skipping (use --force to overwrite)")
        return

    records = generate_permutations(
        aggregated_path=Path(args.aggregated_info),
        questions_path=Path(args.questions),
        num_permutations=args.num_permutations,
        claim_format=args.claim_format,
        seed=args.seed,
        max_cases=args.max_cases,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))

    n_cases = len(records) // args.num_permutations if args.num_permutations else len(records)
    print(f"Saved {len(records)} records ({n_cases} cases × {args.num_permutations} permutations) "
          f"to {output_path}")


if __name__ == "__main__":
    main()
