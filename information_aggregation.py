#!/usr/bin/env python3
"""
Convert aggregated claims (from experiments/aggregation/) into the agent_inference.py
output format, so they can be used directly as:
  - Ground truth for agent_presentation.py  (--ground-truth)
  - Agent cache    for principal_inference.py (--agent-cache)
  - Pre-loaded info for agent_inference.py   (--aggregated-info)

Usage examples:
  # Pure factual claims → agent-results format
  python information_aggregation.py \\
      --aggregated experiments/aggregation/aggregated_factual.json \\
      --questions  experiments/questions/clinical_questions_usmle_sample.json \\
      --output     experiments/agents/usmle_sample/agent_factual_agg.json

  # Pure unfactual claims
  python information_aggregation.py \\
      --aggregated experiments/aggregation/aggregated_unfactual.json \\
      --questions  experiments/questions/clinical_questions_usmle_sample.json \\
      --output     experiments/agents/usmle_sample/agent_unfactual_agg.json \\
      --format bullets

  # Custom agent name / model label
  python information_aggregation.py \\
      --aggregated experiments/aggregation/aggregated_factual.json \\
      --questions  experiments/questions/clinical_questions_usmle_sample.json \\
      --output     experiments/agents/usmle_sample/agent_factual_agg.json \\
      --agent-name aggregated_factual \\
      --agent-model llama-aggregation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Claim formatting helpers
# ---------------------------------------------------------------------------

def format_claims_text(claims: list[dict[str, Any]], fmt: str) -> str:
    """
    Render a list of claim dicts as a single information string.

    Args:
        claims: List of {"agent", "claim", "label", "explanation"} dicts.
        fmt: One of "bullets", "numbered", "plain".

    Returns:
        Formatted multi-line string.
    """
    if not claims:
        return ""

    lines: list[str] = []
    for i, c in enumerate(claims, start=1):
        claim_text = c.get("claim", "").strip()
        if not claim_text:
            continue
        if fmt == "numbered":
            lines.append(f"{i}. {claim_text}")
        elif fmt == "bullets":
            lines.append(f"- {claim_text}")
        else:  # plain
            lines.append(claim_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def build_question_lookup(questions: list[dict]) -> dict[str, dict]:
    """Return a mapping id → question dict from the questions JSON list."""
    return {q["id"]: q for q in questions if "id" in q}


def convert_aggregated_to_agent_results(
    aggregated: dict,
    question_lookup: dict[str, dict],
    fmt: str,
    agent_name: str,
    agent_model: str,
) -> list[dict[str, Any]]:
    """
    Convert an aggregated-claims dict into a list of agent-result records.

    Each record mirrors the schema produced by agent_inference.py so the file
    can be consumed by principal_inference.py or agent_presentation.py.

    Args:
        aggregated:       Loaded aggregated_factual/unfactual JSON.
        question_lookup:  {case_id → question dict} for metadata.
        fmt:              Claim formatting style.
        agent_name:       Value for the "agent_name" field.
        agent_model:      Value for the "agent_model" field.

    Returns:
        List of agent-result dicts.
    """
    cases: dict[str, dict] = aggregated.get("cases", {})
    results: list[dict[str, Any]] = []
    missing_questions = 0

    for case_id, case in cases.items():
        claims = case.get("claims", [])
        information_text = format_claims_text(claims, fmt)

        # Look up question metadata; fall back gracefully if missing
        q = question_lookup.get(case_id)
        if q is None:
            missing_questions += 1
            question_text = case_id  # best we can do without the file
            options = {}
            correct_answer = None
            correct_answer_idx = None
            meta_info = None
        else:
            question_text = q.get("question", "")
            options = q.get("options", {})
            correct_answer = q.get("answer")
            correct_answer_idx = q.get("answer_idx")
            meta_info = q.get("meta_info")

        results.append({
            "agent_name": agent_name,
            "agent_model": agent_model,
            # agent_context = the question the agent "saw"
            "agent_context": question_text,
            "agent_task": None,
            "agent_objective": None,
            # information = the formatted claims presented as information
            "information": information_text,
            "case_id": case_id,
            "dataset_type": "usmle",
            "options": options,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "meta_info": meta_info,
            # principal_context = just the question (no options), same as agent_inference.py
            "principal_context": question_text,
            # Extra provenance fields
            "aggregated_n_agents": case.get("n_agents", 0),
            "aggregated_agents": case.get("agents_present", []),
            "aggregated_n_claims": case.get("summary", {}).get("total_claims", len(claims)),
        })

    if missing_questions:
        print(f"WARNING: {missing_questions} case(s) had no matching question in the questions file.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--aggregated", required=True, metavar="PATH",
        help=(
            "Path to aggregated claims JSON "
            "(e.g. experiments/aggregation/aggregated_factual.json)"
        ),
    )
    parser.add_argument(
        "--questions", default=None, metavar="PATH",
        help=(
            "Path to clinical questions JSON "
            "(e.g. experiments/questions/clinical_questions_usmle_sample.json). "
            "Required for options / answer metadata."
        ),
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH",
        help="Output path for the agent-results JSON list.",
    )
    parser.add_argument(
        "--format",
        choices=["bullets", "numbered", "plain"],
        default="bullets",
        help="How to join multiple claims into the information string (default: bullets)",
    )
    parser.add_argument(
        "--agent-name", default=None, metavar="NAME",
        help=(
            "Value to use for the 'agent_name' field. "
            "Auto-detected from filename if not specified."
        ),
    )
    parser.add_argument(
        "--agent-model", default="llama-aggregation", metavar="MODEL",
        help="Value to use for the 'agent_model' field (default: llama-aggregation)",
    )
    args = parser.parse_args()

    # --- Load aggregated claims ---
    aggregated_path = Path(args.aggregated)
    if not aggregated_path.exists():
        raise FileNotFoundError(f"Aggregated file not found: {aggregated_path}")

    aggregated = json.loads(aggregated_path.read_text())
    metadata = aggregated.get("metadata", {})
    keep_labels = metadata.get("keep_labels", ["?"])

    print(f"Loaded aggregated claims from {aggregated_path}")
    print(f"  Labels kept:   {', '.join(keep_labels)}")
    print(f"  Cases:         {aggregated.get('aggregate', {}).get('n_cases', '?')}")
    print(f"  Claims (total): {aggregated.get('aggregate', {}).get('total_claims_after_filter', '?')}")

    # --- Load questions (optional but recommended) ---
    question_lookup: dict[str, dict] = {}
    if args.questions:
        q_path = Path(args.questions)
        if not q_path.exists():
            raise FileNotFoundError(f"Questions file not found: {q_path}")
        questions = json.loads(q_path.read_text())
        question_lookup = build_question_lookup(questions)
        print(f"Loaded {len(question_lookup)} questions from {q_path}")
    else:
        print("WARNING: No questions file provided. Metadata (options, answer, etc.) will be empty.")

    # --- Auto-detect agent name from filename ---
    if args.agent_name:
        agent_name = args.agent_name
    else:
        stem = aggregated_path.stem  # e.g. "aggregated_factual"
        agent_name = stem  # "aggregated_factual" / "aggregated_unfactual"

    print(f"Agent name:  {agent_name}")
    print(f"Agent model: {args.agent_model}")
    print(f"Claim format: {args.format}")

    # --- Convert ---
    results = convert_aggregated_to_agent_results(
        aggregated=aggregated,
        question_lookup=question_lookup,
        fmt=args.format,
        agent_name=agent_name,
        agent_model=args.agent_model,
    )

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nConverted {len(results)} cases → {out_path}")
    if results:
        sample = results[0]
        info_preview = (sample["information"] or "")[:200].replace("\n", " ")
        print(f"Sample case_id:  {sample['case_id']}")
        print(f"Sample info:     {info_preview}{'...' if len(sample['information']) > 200 else ''}")
        print(f"Sample n_claims: {sample['aggregated_n_claims']}")


if __name__ == "__main__":
    main()
