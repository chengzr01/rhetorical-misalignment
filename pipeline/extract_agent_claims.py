#!/usr/bin/env python3
"""
Extract individual factual claims from agent framing responses for martingale testing.

Reads a framing results file (framing_<agent>_gt_factual_agg.json) where each case
has a free-form `information` field (the agent's markdown response). Uses an LLM to
extract discrete claims verbatim from that text, then outputs in aggregated_factual.json
format so the result can be fed directly into generate_permutations.py.

The wording of each claim is preserved exactly as written by the agent — the LLM
only splits the response into individual claim units.

Usage:
    python experiments/extract_agent_claims.py \\
        --input  experiments/agents/usmle_sample/framing_claude_gt_factual_agg.json \\
        --output experiments/aggregation/aggregated_claude_framing.json \\
        --agent-key claude

    python experiments/extract_agent_claims.py \\
        --input  experiments/agents/usmle_sample/framing_deepseek_gt_factual_agg.json \\
        --output experiments/aggregation/aggregated_deepseek_framing.json \\
        --agent-key deepseek

Environment:
    OPENROUTER_API_KEY  – required
    EXTRACT_MODEL       – override the extraction model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import re

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract" / "extract_claims.yaml"
DEFAULT_EXTRACT_MODEL = "deepseek/deepseek-chat-v3.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt(path: Path) -> dict[str, str]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if "system_prompt" not in data or "user_template" not in data:
        raise ValueError(f"Prompt YAML at {path} must contain 'system_prompt' and 'user_template'")
    return data


_MATCH_LEVELS = ("exact", "case_folded", "markdown_stripped", "whitespace_normalized", "none")


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting symbols, leaving only the content."""
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)   # bold / italic
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)       # underline / italic
    text = re.sub(r"`([^`]+)`", r"\1", text)                # inline code
    text = re.sub(r"^[#>\-\*\d\.]+\s*", "", text, flags=re.MULTILINE)  # list / header markers
    text = re.sub(r"\|", " ", text)                         # table pipes
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_offset(needle: str, haystack: str) -> int | None:
    """Return the character offset of needle in haystack, or None if not found."""
    idx = haystack.find(needle)
    return idx if idx != -1 else None


def match_claim_in_source(claim: str, source: str) -> dict:
    """
    Try to locate a claim in the source text using a sequence of deterministic,
    mechanistic normalization levels. Each level is a pure string operation —
    no fuzzy or semantic matching.

    Levels tried in order:
      1. exact            — claim is a literal substring of source (no changes)
      2. case_folded      — lowercased claim is in lowercased source
      3. markdown_stripped — markdown symbols removed from both sides, then substring check
      4. whitespace_normalized — additionally collapse all whitespace runs to single space
      5. none             — no match found at any level

    Returns a dict with:
      match_level : one of the strings above
      match_offset: character offset in the (level-appropriate) normalized source,
                    or None if no match
    """
    # Level 1: exact
    offset = _find_offset(claim, source)
    if offset is not None:
        return {"match_level": "exact", "match_offset": offset}

    # Level 2: case-folded
    offset = _find_offset(claim.lower(), source.lower())
    if offset is not None:
        return {"match_level": "case_folded", "match_offset": offset}

    # Level 3: markdown stripped (both sides)
    claim_md  = _strip_markdown(claim)
    source_md = _strip_markdown(source)
    offset = _find_offset(claim_md.lower(), source_md.lower())
    if offset is not None:
        return {"match_level": "markdown_stripped", "match_offset": offset}

    # Level 4: additionally collapse whitespace
    claim_ws  = _collapse_whitespace(claim_md)
    source_ws = _collapse_whitespace(source_md)
    offset = _find_offset(claim_ws.lower(), source_ws.lower())
    if offset is not None:
        return {"match_level": "whitespace_normalized", "match_offset": offset}

    # Level 5: no match
    return {"match_level": "none", "match_offset": None}


def parse_bullet_claims(text: str) -> list[str]:
    """Parse LLM bullet-list output into a list of claim strings."""
    claims = []
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


def extract_claims_for_case(
    case_record: dict[str, Any],
    client: OpenRouterChatClient,
    model: str,
    prompt: dict[str, str],
    agent_key: str,
    temperature: float = 0.0,
) -> tuple[str, list[dict[str, Any]], int]:
    """
    Extract claims from a single case record's `information` field.

    Returns:
        (case_id, list of claim dicts, number of claims that failed verbatim check)
    """
    case_id = case_record["case_id"]
    information = case_record.get("information", "").strip()
    question = case_record.get("agent_context", "").strip()

    if not information:
        return case_id, [], 0

    user_message = prompt["user_template"].format(
        question=question,
        information=information,
    )

    raw = client.create_completion(
        model=model,
        messages=[
            {"role": "system", "content": prompt["system_prompt"].strip()},
            {"role": "user",   "content": user_message.strip()},
        ],
        temperature=temperature,
    )

    claim_texts = parse_bullet_claims(raw)

    n_unmatched = 0
    claim_dicts = []
    last_offset = -1
    order_violations = 0

    for c in claim_texts:
        m = match_claim_in_source(c, information)
        if m["match_level"] == "none":
            n_unmatched += 1
        # Check that claims appear in the same order as in the source text
        if m["match_offset"] is not None:
            if m["match_offset"] < last_offset:
                order_violations += 1
            else:
                last_offset = m["match_offset"]
        claim_dicts.append({
            "agent": agent_key,
            "claim": c,
            "label": "factual",
            "match_level": m["match_level"],
            "match_offset": m["match_offset"],
        })

    return case_id, claim_dicts, n_unmatched, order_violations


# ---------------------------------------------------------------------------
# Main extraction driver
# ---------------------------------------------------------------------------

def extract_agent_claims(
    input_path: Path,
    agent_key: str,
    extract_model: str,
    max_workers: int = 8,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """
    Extract claims from all cases in a framing results file.

    Returns:
        aggregated_factual.json-compatible dict.
    """
    records: list[dict[str, Any]] = json.loads(input_path.read_text())
    if max_cases is not None:
        records = records[:max_cases]

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    client = OpenRouterChatClient(api_key=api_key)
    prompt = load_prompt(PROMPT_PATH)

    cases_out: dict[str, dict[str, Any]] = {}
    total_claims = 0
    match_level_counts: dict[str, int] = {lvl: 0 for lvl in _MATCH_LEVELS}
    total_order_violations = 0

    print(f"Extracting claims from {len(records)} cases using {extract_model} ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_claims_for_case,
                rec, client, extract_model, prompt, agent_key,
            ): rec["case_id"]
            for rec in records
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            case_id = futures[future]
            try:
                _, claim_dicts, n_unmatched, order_violations = future.result()
            except Exception as exc:
                print(f"  WARNING: case {case_id} failed: {exc}")
                claim_dicts, n_unmatched, order_violations = [], 0, 0

            if n_unmatched > 0:
                print(f"  WARNING [{case_id}]: {n_unmatched} claim(s) not found in source "
                      f"at any normalization level — wording was changed by the extractor")
            if order_violations > 0:
                print(f"  WARNING [{case_id}]: {order_violations} claim(s) appear out of "
                      f"source order — extractor may have reordered content")

            for cd in claim_dicts:
                match_level_counts[cd["match_level"]] += 1

            cases_out[case_id] = {
                "case_id": case_id,
                "agents_present": [agent_key],
                "n_agents": 1,
                "claims": claim_dicts,
            }
            total_claims += len(claim_dicts)
            total_order_violations += order_violations

    # Summary report
    print(f"\n{'='*50}")
    print(f"Extraction complete — {len(cases_out)} cases, {total_claims} claims")
    print(f"Match level breakdown:")
    for lvl in _MATCH_LEVELS:
        count = match_level_counts[lvl]
        pct = 100 * count / total_claims if total_claims else 0
        marker = "  ✓" if lvl in ("exact", "case_folded") else ("  ~" if lvl != "none" else "  ✗")
        print(f"  {marker} {lvl:<25} {count:>5}  ({pct:.1f}%)")
    if total_order_violations > 0:
        print(f"\n  ⚠ {total_order_violations} claim(s) appeared out of source order")
    print(f"{'='*50}\n")

    output = {
        "metadata": {
            "agents": [agent_key],
            "source_file": str(input_path),
            "extract_model": extract_model,
            "keep_labels": ["factual"],
            "n_agents_loaded": 1,
            "n_cases": len(cases_out),
            "match_level_counts": match_level_counts,
            "order_violations": total_order_violations,
        },
        "aggregate": {
            "n_cases": len(cases_out),
            "total_claims": total_claims,
        },
        "cases": cases_out,
    }
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Path to framing_<agent>_gt_factual_agg.json",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path (aggregated_factual.json-compatible)",
    )
    parser.add_argument(
        "--agent-key", required=True,
        help="Short name for this agent (e.g. 'claude', 'deepseek')",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EXTRACT_MODEL", DEFAULT_EXTRACT_MODEL),
        help=f"Extraction model via OpenRouter (default: {DEFAULT_EXTRACT_MODEL})",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="Number of parallel API calls (default: 8)",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Limit to first N cases (for testing)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite output if it already exists",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    if output_path.exists() and not args.force:
        print(f"Output already exists: {output_path}  (use --force to overwrite)")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = extract_agent_claims(
        input_path=input_path,
        agent_key=args.agent_key,
        extract_model=args.model,
        max_workers=args.max_workers,
        max_cases=args.max_cases,
    )

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {result['aggregate']['n_cases']} cases "
          f"({result['aggregate']['total_claims']} claims) → {output_path}")


if __name__ == "__main__":
    main()
