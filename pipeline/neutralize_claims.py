#!/usr/bin/env python3
"""Rewrite claims in an aggregated-claims JSON in neutral, model-agnostic language.

Takes an aggregated-claims file (e.g. aggregated_gemini_factual.json from
aggregate_information.py or extract_agent_claims.py) and rewrites each claim
in neutral, standardized clinical language — removing any source-model style,
hedging, or attribution phrasing while preserving all factual content.

This is the step before running framing / information-design experiments with
claims that originate from a specific model (e.g. Gemini-2.5-Pro): neutralizing
removes stylistic confounds so that any effect observed is due to *what* is said,
not *how* the source model said it.

Output is in the same aggregated-claims format as the input and can be fed
directly into:
  - core/agent_presentation.py  (--aggregated-info)
  - core/agent_selection.py     (--aggregated-info)

Each claim dict gains an "original_claim" field that preserves the pre-neutral text.

Usage:
    python pipeline/neutralize_claims.py \\
        --input     experiments/aggregation/aggregated_gemini_factual.json \\
        --questions experiments/questions/clinical_questions_usmle_sample.json \\
        --output    experiments/aggregation/aggregated_gemini_factual_neutral.json

    # Resume an interrupted run automatically (no --force needed):
    python pipeline/neutralize_claims.py \\
        --input  experiments/aggregation/aggregated_gemini_factual.json \\
        --output experiments/aggregation/aggregated_gemini_factual_neutral.json

Environment:
    OPENROUTER_API_KEY  – required
    NEUTRALIZE_MODEL    – override the model (default: deepseek/deepseek-chat-v3.1)
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


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "neutralize" / "neutralize_claims.yaml"
DEFAULT_NEUTRALIZE_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_QUESTIONS_PATH = "experiments/questions/clinical_questions_usmle_sample.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt(path: Path) -> dict[str, str]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if "system_prompt" not in data or "user_template" not in data:
        raise ValueError(f"Prompt YAML at {path} must contain 'system_prompt' and 'user_template'")
    return data


def format_claims_numbered(claims: list[dict[str, Any]]) -> str:
    """Format claims as a 1-based numbered list for the prompt."""
    lines: list[str] = []
    for i, c in enumerate(claims, start=1):
        text = c.get("claim", "").strip()
        if text:
            lines.append(f"{i}. {text}")
    return "\n".join(lines)


def parse_bullet_claims(text: str) -> list[str]:
    """Parse bullet-list LLM output into a list of claim strings."""
    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            claim = line[2:].strip()
            if claim:
                claims.append(claim)
        elif line.startswith("* "):
            claim = line[2:].strip()
            if claim:
                claims.append(claim)
    return claims


# ---------------------------------------------------------------------------
# Per-case neutralization
# ---------------------------------------------------------------------------

def neutralize_case(
    case_id: str,
    case: dict[str, Any],
    question: dict[str, Any],
    client: OpenRouterChatClient,
    model: str,
    prompt: dict[str, str],
    temperature: float = 0.0,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Neutralize all claims for a single case.

    Returns:
        (neutralized_claim_dicts, success)
        On count mismatch or error, success=False and the original claims are returned.
    """
    claims = case.get("claims", [])
    if not claims:
        return [], True

    question_text = question.get("question", case_id)
    numbered = format_claims_numbered(claims)

    user_message = prompt["user_template"].format(
        question=question_text.strip(),
        claims=numbered.strip(),
        n_claims=len(claims),
    )

    raw = client.create_completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system_prompt"].strip()},
            {"role": "user",   "content": user_message.strip()},
        ],
        temperature=temperature,
    )

    neutral_texts = parse_bullet_claims(raw)

    if len(neutral_texts) != len(claims):
        print(
            f"  WARNING [{case_id}]: expected {len(claims)} neutral claims, "
            f"got {len(neutral_texts)} — falling back to original"
        )
        return list(claims), False

    # Rebuild claim dicts: replace text, preserve all other fields
    neutralized: list[dict[str, Any]] = []
    for original, neutral_text in zip(claims, neutral_texts):
        neutralized.append({
            **original,
            "claim": neutral_text,
            "original_claim": original.get("claim", ""),
        })

    return neutralized, True


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def neutralize_aggregated_claims(
    aggregated_path: Path,
    questions_path: Path,
    neutralize_model: str,
    temperature: float = 0.0,
    max_cases: int | None = None,
    max_workers: int = 8,
    existing_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Neutralize all claims in an aggregated-claims JSON file.

    Resumes from existing_output if provided (skips already-done cases).

    Returns:
        Updated aggregated dict in the same schema as the input.
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

    # Resume: cases already in the output file are skipped
    done: dict[str, dict] = {}
    if existing_output and "cases" in existing_output:
        done = dict(existing_output["cases"])

    pending = [cid for cid in case_ids if cid not in done]
    print(f"Loaded {len(cases)} cases from {aggregated_path}")
    print(f"  Already done : {len(done)}")
    print(f"  To process   : {len(pending)}")
    print(f"  Model        : {neutralize_model}")
    print(f"  Temperature  : {temperature}")

    n_fallbacks = 0

    def process_one(cid: str) -> tuple[str, dict[str, Any], bool]:
        case = cases[cid]
        question = question_lookup.get(cid, {})
        neutralized_claims, success = neutralize_case(
            case_id=cid,
            case=case,
            question=question,
            client=client,
            model=neutralize_model,
            prompt=prompt,
            temperature=temperature,
        )
        # Rebuild the case dict with neutralized claims, preserving all other fields
        updated_case = {
            **case,
            "claims": neutralized_claims,
        }
        return cid, updated_case, success

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, cid): cid for cid in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Neutralizing"):
            cid = futures[future]
            try:
                _, updated_case, success = future.result()
                done[cid] = updated_case
                if not success:
                    n_fallbacks += 1
            except Exception as exc:
                print(f"  WARNING: case {cid} raised {exc} — keeping original claims")
                done[cid] = cases[cid]
                n_fallbacks += 1

    if n_fallbacks:
        print(f"\n  ⚠  {n_fallbacks}/{len(pending)} case(s) fell back to original claims")

    # Restore original ordering
    output_cases = {cid: done[cid] for cid in case_ids if cid in done}

    total_claims = sum(len(c.get("claims", [])) for c in output_cases.values())

    return {
        "metadata": {
            **aggregated.get("metadata", {}),
            "neutralize_model": neutralize_model,
            "neutralize_temperature": temperature,
            "neutralize_source": str(aggregated_path),
            "n_cases_neutralized": len(output_cases),
            "n_fallbacks": n_fallbacks,
        },
        "aggregate": {
            **aggregated.get("aggregate", {}),
            "n_cases": len(output_cases),
            "total_claims_after_filter": total_claims,
        },
        "cases": output_cases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, metavar="PATH",
        help="Aggregated-claims JSON (output of aggregate_information.py or extract_agent_claims.py)",
    )
    parser.add_argument(
        "--questions", default=DEFAULT_QUESTIONS_PATH, metavar="PATH",
        help="Clinical questions JSON (for question context in the prompt)",
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH",
        help="Output path for neutralized aggregated-claims JSON",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("NEUTRALIZE_MODEL", DEFAULT_NEUTRALIZE_MODEL),
        metavar="MODEL",
        help=f"OpenRouter model ID for neutralization (default: {DEFAULT_NEUTRALIZE_MODEL})",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic output)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, metavar="N",
        help="Parallel API calls (default: 8)",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None, metavar="N",
        help="Limit to first N cases (for testing)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output and re-neutralize all cases (disables resume)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    questions_path = Path(args.questions)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)
    if not questions_path.exists():
        print(f"Error: questions file not found: {questions_path}")
        sys.exit(1)

    # Resume if output exists and --force not set
    existing_output: dict[str, Any] | None = None
    if not args.force and output_path.exists():
        existing_output = json.loads(output_path.read_text())
        n_done = len(existing_output.get("cases", {}))
        print(f"Resuming: {n_done} cases already in {output_path}")

    result = neutralize_aggregated_claims(
        aggregated_path=input_path,
        questions_path=questions_path,
        neutralize_model=args.model,
        temperature=args.temperature,
        max_cases=args.max_cases,
        max_workers=args.max_workers,
        existing_output=existing_output,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    n_cases = result["aggregate"]["n_cases"]
    n_claims = result["aggregate"].get("total_claims_after_filter", "?")
    print(f"\nSaved {n_cases} cases ({n_claims} claims) → {output_path}")


if __name__ == "__main__":
    main()
