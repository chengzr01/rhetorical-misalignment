#!/usr/bin/env python3
"""
Generate K paraphrased versions of aggregated claims for sufficient-statistics testing.

For each case in an aggregated-claims JSON, this script produces K+1 records:
  - condition="original"              : bullet-point claims (no API call)
  - condition="paraphrase", idx=0..K-1: fluent-prose rewrites by an LLM, each
                                        independently generated with instructions to
                                        preserve all sufficient statistics

The output is a flat list compatible with principal_inference.py's --agent-cache
format.  Running inference on all K+1 records and aggregating over the K paraphrases
gives a Monte-Carlo approximation to the model's Bayesian posterior that is robust
to surface-level wording choices.

Usage:
    python experiments/generate_paraphrases.py \\
        --aggregated-info experiments/aggregation/aggregated_factual.json \\
        --questions experiments/questions/clinical_questions_usmle_sample.json \\
        --num-paraphrases 10 \\
        --output experiments/agents/usmle_sample/paraphrase_records_k10.json

Environment:
    OPENROUTER_API_KEY  – required for the paraphrase API calls
    PARAPHRASE_MODEL    – override the model (default: anthropic/claude-haiku-4-5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "paraphrase" / "paraphrase_claims.yaml"
DEFAULT_PARAPHRASE_MODEL = "anthropic/claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt(path: Path) -> dict[str, str]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if "system_prompt" not in data or "user_template" not in data:
        raise ValueError(f"Prompt YAML at {path} must contain 'system_prompt' and 'user_template'")
    return data


def format_claims(claims: list[dict[str, Any]], fmt: str = "bullets") -> str:
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


def paraphrase_claims(
    claims_text: str,
    question_context: str,
    client: OpenRouterChatClient,
    model: str,
    prompt: dict[str, str],
    temperature: float = 0.7,
) -> str:
    """Call the paraphrase model to rewrite claims as fluent prose."""
    user_message = prompt["user_template"].format(
        question=question_context.strip(),
        claims=claims_text.strip(),
    )
    return client.create_completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system_prompt"].strip()},
            {"role": "user",   "content": user_message.strip()},
        ],
        temperature=temperature,
    )


def _base_record(
    case_id: str,
    case: dict[str, Any],
    question: dict[str, Any],
) -> dict[str, Any]:
    """Shared metadata fields for all records of a case."""
    return {
        "case_id":            case_id,
        "agent_context":      question.get("question", case_id),
        "principal_context":  question.get("question", case_id),
        "dataset_type":       "usmle",
        "options":            question.get("options", {}),
        "correct_answer":     question.get("answer"),
        "correct_answer_idx": question.get("answer_idx"),
        "meta_info":          question.get("meta_info"),
        "agent_task":         None,
        "agent_objective":    None,
        "n_original_claims":  case.get("summary", {}).get("total_claims",
                                                           len(case.get("claims", []))),
    }


# ---------------------------------------------------------------------------
# Per-case generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main generation driver
# ---------------------------------------------------------------------------

def generate_paraphrases(
    aggregated_path: Path,
    questions_path: Path,
    num_paraphrases: int,
    paraphrase_model: str,
    claim_format: str = "bullets",
    temperature: float = 0.7,
    max_cases: int | None = None,
    max_workers: int = 8,
    existing_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate original + K paraphrase record sets for all cases.

    Returns:
        Flat list of agent-cache-compatible records ((K+1) per case).
    """
    aggregated = json.loads(aggregated_path.read_text())
    cases: dict[str, dict] = aggregated.get("cases", {})

    questions = json.loads(questions_path.read_text())
    question_lookup: dict[str, dict] = {q["id"]: q for q in questions if "id" in q}

    prompt = load_prompt(PROMPT_PATH)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    client = OpenRouterChatClient(api_key=api_key)

    case_ids = sorted(cases.keys())
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    # Build resume index: case_id -> set of already-done paraphrase_idx values
    done_by_case: dict[str, set[int]] = {cid: set() for cid in case_ids}
    completed_records: list[dict[str, Any]] = []

    if existing_records:
        for r in existing_records:
            cid = r.get("case_id")
            if cid not in done_by_case:
                continue
            if r.get("condition") == "paraphrase" and r.get("paraphrase_idx") is not None:
                done_by_case[cid].add(r["paraphrase_idx"])
                completed_records.append(r)

    total_pending = sum(
        num_paraphrases - len(done_by_case[cid]) for cid in case_ids
    )
    total_done = sum(len(done_by_case[cid]) for cid in case_ids)

    print(f"Loaded {len(cases)} cases from {aggregated_path}")
    print(f"Num paraphrases (K): {num_paraphrases}")
    print(f"Paraphrase model:    {paraphrase_model}")
    print(f"Max workers:         {max_workers}")
    print(f"Total API calls:     {total_pending}  ({total_done} already done)")

    # Build per-case metadata needed by each task
    case_meta: dict[str, tuple] = {}
    for case_id in case_ids:
        case = cases[case_id]
        question = question_lookup.get(case_id, {})
        original_text = format_claims(case.get("claims", []), claim_format)
        case_meta[case_id] = (case, question, original_text)

    # Flatten to individual (case_id, paraphrase_idx) tasks for true parallelism
    tasks: list[tuple[str, int]] = [
        (case_id, idx)
        for case_id in case_ids
        for idx in range(num_paraphrases)
        if idx not in done_by_case[case_id]
    ]

    def process_one(case_id: str, para_idx: int) -> dict[str, Any]:
        case, question, original_text = case_meta[case_id]
        base = _base_record(case_id, case, question)
        if not original_text.strip():
            text = "(no claims available)"
        else:
            text = paraphrase_claims(
                claims_text=original_text,
                question_context=question.get("question", case_id),
                client=client,
                model=paraphrase_model,
                prompt=prompt,
                temperature=temperature,
            )
        return {
            **base,
            "agent_name":       "paraphrase_paraphrased",
            "agent_model":      paraphrase_model,
            "condition":        "paraphrase",
            "paraphrase_idx":   para_idx,
            "information":      text,
            "paraphrase_model": paraphrase_model,
        }

    # Original records require no API call — build them immediately
    original_records: list[dict[str, Any]] = []
    for case_id in case_ids:
        case, question, original_text = case_meta[case_id]
        base = _base_record(case_id, case, question)
        original_records.append({
            **base,
            "agent_name":       "paraphrase_original",
            "agent_model":      "aggregated_claims",
            "condition":        "original",
            "paraphrase_idx":   None,
            "information":      original_text,
            "paraphrase_model": None,
        })

    new_records: list[dict[str, Any]] = list(original_records)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(process_one, cid, idx): (cid, idx)
            for cid, idx in tasks
        }
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Paraphrasing"):
            cid, idx = future_to_task[future]
            try:
                new_records.append(future.result())
            except Exception as e:
                print(f"  [ERROR] case {cid} idx {idx}: {e}")

    # Merge: completed (paraphrase only) + newly generated (original + new paraphrases)
    all_records = completed_records + new_records

    # Deduplicate: keep the last occurrence of each (case_id, condition, paraphrase_idx)
    seen: dict[tuple, dict] = {}
    for r in all_records:
        key = (r["case_id"], r.get("condition"), r.get("paraphrase_idx"))
        seen[key] = r

    deduped = list(seen.values())
    deduped.sort(key=lambda r: (
        r["case_id"],
        0 if r.get("condition") == "original" else 1,
        r.get("paraphrase_idx") if r.get("paraphrase_idx") is not None else -1,
    ))
    return deduped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate original + K paraphrase record pairs for "
            "sufficient-statistics (paraphrase invariance) testing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aggregated-info", required=True, metavar="PATH")
    parser.add_argument("--questions",        required=True, metavar="PATH")
    parser.add_argument(
        "--num-paraphrases",
        type=int,
        default=10,
        metavar="K",
        help="Number of independent paraphrases to generate per case",
    )
    parser.add_argument(
        "--paraphrase-model",
        default=os.getenv("PARAPHRASE_MODEL", DEFAULT_PARAPHRASE_MODEL),
        metavar="MODEL",
        help="OpenRouter model ID to use for paraphrasing",
    )
    parser.add_argument("--claim-format",  default="bullets", choices=["bullets", "numbered", "plain"])
    parser.add_argument("--temperature",   type=float, default=0.7)
    parser.add_argument("--max-cases",     type=int,   default=None, metavar="N")
    parser.add_argument("--max-workers",   type=int,   default=8,    metavar="N")
    parser.add_argument("--output",        required=True, metavar="PATH")
    parser.add_argument("--force",         action="store_true", default=False,
                        help="Overwrite output file (disables resume)")
    args = parser.parse_args()

    output_path = Path(args.output)

    existing_records: list[dict[str, Any]] | None = None
    if not args.force and output_path.exists():
        existing_records = json.loads(output_path.read_text())
        n_done = sum(1 for r in existing_records if r.get("condition") == "paraphrase")
        print(f"Resuming: {n_done} paraphrase records already in {output_path}")

    records = generate_paraphrases(
        aggregated_path=Path(args.aggregated_info),
        questions_path=Path(args.questions),
        num_paraphrases=args.num_paraphrases,
        paraphrase_model=args.paraphrase_model,
        claim_format=args.claim_format,
        temperature=args.temperature,
        max_cases=args.max_cases,
        max_workers=args.max_workers,
        existing_records=existing_records,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))

    n_orig = sum(1 for r in records if r.get("condition") == "original")
    n_para = sum(1 for r in records if r.get("condition") == "paraphrase")
    print(f"\nSaved {len(records)} records "
          f"({n_orig} original + {n_para} paraphrase, K={args.num_paraphrases}) → {output_path}")


if __name__ == "__main__":
    main()
