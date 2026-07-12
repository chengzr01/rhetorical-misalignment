#!/usr/bin/env python3
"""Curate USMLE claim factuality using an OpenRouter verifier.

This utility reviews claims pulled from aggregated information files and
confirms their factual status by querying an LLM.  The result is a curated
claim set suitable for downstream factuality validation experiments.

Example:
    python pipeline/curate_claims_via_llm.py \
        --aggregated-info experiments/aggregation/aggregated_gemini_factual.json \
        --questions experiments/questions/clinical_questions_usmle_sample.json \
        --output experiments/claims/usmle_sample/curated_gemini_factual.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient

DEFAULT_MODEL = os.getenv("CLAIM_CURATION_MODEL", "deepseek/deepseek-chat-v3.1")
DEFAULT_PROMPT_PATH = Path("prompts/experiments/claim_curation.yaml")


class PromptSpec(Dict[str, Any]):
    pass


def load_prompt(path: Path) -> PromptSpec:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "system" not in data or "user" not in data:
        raise ValueError(
            "Claim curation prompt YAML must contain 'system' and 'user' entries"
        )
    return PromptSpec(data)


def build_messages(
    *,
    prompt: PromptSpec,
    question: str,
    claim: str,
    options: Dict[str, str] | None,
) -> List[Dict[str, str]]:
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted((options or {}).items()))
    user_message = prompt["user"].format(
        question=question.strip() if question else "",
        claim=claim.strip(),
        options=options_text.strip(),
    )
    return [
        {"role": "system", "content": prompt["system"].strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def extract_json_block(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Verifier returned an empty response")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError("Verifier response did not contain JSON output")
        snippet = text[start : end + 1]
        return json.loads(snippet)


def verify_claim(
    *,
    client: OpenRouterChatClient,
    model: str,
    prompt: PromptSpec,
    question: str,
    claim: Dict[str, Any],
    options: Dict[str, str] | None,
    temperature: float,
) -> Dict[str, Any]:
    claim_text = claim.get("claim", "").strip()
    if not claim_text:
        raise ValueError("Claim text missing during verification")

    messages = build_messages(
        prompt=prompt,
        question=question,
        claim=claim_text,
        options=options,
    )

    response = client.create_completion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    parsed = extract_json_block(response)

    verdict = parsed.get("verdict")
    if verdict is None:
        raise ValueError("Verifier JSON missing 'verdict'")
    verdict_str = str(verdict).lower().strip()
    if verdict_str not in {"factual", "unfactual", "uncertain"}:
        raise ValueError(f"Unexpected verdict '{verdict}' from verifier")

    confidence = parsed.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_val = max(0.0, min(1.0, float(confidence)))
    else:
        confidence_val = None

    explanation = parsed.get("explanation")
    evidence = parsed.get("evidence")

    return {
        "verdict": verdict_str,
        "confidence": confidence_val,
        "explanation": explanation.strip() if isinstance(explanation, str) else None,
        "evidence": evidence if isinstance(evidence, list) else None,
        "raw_response": response,
    }


def iter_claims(
    aggregated: Dict[str, Any],
    *,
    max_cases: int | None,
    max_claims_per_case: int | None,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    cases = aggregated.get("cases", {})
    case_ids = sorted(cases.keys())
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    for case_id in case_ids:
        claims = cases[case_id].get("claims", [])
        if max_claims_per_case is not None:
            claims = claims[:max_claims_per_case]
        for idx, claim in enumerate(claims):
            claim_id = f"{case_id}__{idx}"
            yield claim_id, claim


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    claims = data.get("claims")
    if not isinstance(claims, list):
        return {}
    return {entry.get("claim_id"): entry for entry in claims if entry.get("claim_id")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate USMLE claims via OpenRouter verifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aggregated-info", type=Path, required=True,
                        help="Aggregated claims JSON (from aggregate_information.py)")
    parser.add_argument("--questions", type=Path,
                        default=Path("experiments/questions/clinical_questions_usmle_sample.json"),
                        help="Questions JSON providing context")
    parser.add_argument("--output", type=Path, required=True,
                        help="Curated claims output path")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="OpenRouter model to use for verification")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH,
                        help="Prompt YAML with system/user templates")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for verification model")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel verifier requests")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of cases processed")
    parser.add_argument("--max-claims-per-case", type=int, default=None,
                        help="Cap number of claims per case")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if resume cache present")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    aggregated = json.loads(args.aggregated_info.read_text())
    questions_list = json.loads(args.questions.read_text()) if args.questions.exists() else []
    question_lookup = {q.get("id"): q for q in questions_list if isinstance(q, dict)}

    prompt = load_prompt(args.prompt)
    client = OpenRouterChatClient(api_key=api_key)

    existing = load_existing(args.output) if args.resume and not args.force else {}
    curated: Dict[str, Dict[str, Any]] = dict(existing)

    tasks: List[str] = []
    claim_meta: Dict[str, Dict[str, Any]] = {}
    for claim_id, claim in iter_claims(
        aggregated,
        max_cases=args.max_cases,
        max_claims_per_case=args.max_claims_per_case,
    ):
        case_id, idx_str = claim_id.split("__", 1)
        claim_index = int(idx_str)
        question_obj = question_lookup.get(case_id, {})
        claim_meta[claim_id] = {
            "case_id": case_id,
            "index": claim_index,
            "claim": claim,
            "question": question_obj.get("question"),
            "options": question_obj.get("options"),
            "meta_info": question_obj.get("meta_info"),
        }
        if claim_id in existing and not args.force:
            continue
        tasks.append(claim_id)

    print(f"Loaded {len(existing)} existing curated entries" if existing else "No cached curation found")

    if tasks:
        print(f"Curating {len(tasks)} claims using {args.model} (T={args.temperature})")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    verify_claim,
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    question=claim_meta[claim_id]["question"] or "",
                    claim=claim_meta[claim_id]["claim"],
                    options=claim_meta[claim_id]["options"],
                    temperature=args.temperature,
                ): claim_id
                for claim_id in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Curating"):
                claim_id = futures[future]
                result = future.result()
                meta = claim_meta[claim_id]
                base_case = meta["case_id"]
                idx = meta["index"]
                claim_obj = aggregated["cases"][base_case]["claims"][idx]
                curated_entry = {
                    "claim_id": claim_id,
                    "case_id": base_case,
                    "claim_text": claim_obj.get("claim"),
                    "original_label": claim_obj.get("label"),
                    "verdict": result["verdict"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"],
                    "evidence": result["evidence"],
                    "verifier_model": args.model,
                    "question": meta["question"],
                    "options": meta["options"],
                    "meta_info": meta["meta_info"],
                    "raw_response": result["raw_response"],
                }
                curated[claim_id] = curated_entry
    else:
        print("All claims already curated; skipping verifier calls")

    ordered_claims = [curated[cid] for cid, _ in iter_claims(
        aggregated,
        max_cases=args.max_cases,
        max_claims_per_case=args.max_claims_per_case,
    ) if cid in curated]

    payload = {
        "source_aggregated": str(args.aggregated_info),
        "questions_file": str(args.questions),
        "verifier_model": args.model,
        "temperature": args.temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims": ordered_claims,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved curated claims → {args.output} (n={len(ordered_claims)})")


if __name__ == "__main__":
    main()
