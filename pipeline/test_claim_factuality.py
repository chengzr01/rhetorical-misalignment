#!/usr/bin/env python3
"""Evaluate model ability to classify claim factuality on curated datasets."""

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
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient

DEFAULT_PROMPT = Path("prompts/experiments/claim_factuality.yaml")


def load_prompt_template(path: Path) -> Dict[str, str]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "prompt" not in data:
        raise ValueError("Claim factuality prompt must contain 'prompt' field")
    return data


def format_prompt(template: str, claim_entry: Dict[str, Any]) -> str:
    question = claim_entry.get("question", "") or "(question unavailable)"
    options_dict = claim_entry.get("options") or {}
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options_dict.items()))
    claim_text = claim_entry.get("claim_text") or claim_entry.get("claim") or ""
    return (
        template
        .replace("<QUESTION>", question)
        .replace("<OPTIONS>", options_text)
        .replace("<CLAIM>", claim_text)
    )


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty response from model")
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
            raise ValueError("Model response missing JSON payload")
        snippet = text[start : end + 1]
        return json.loads(snippet)


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower()
    if text in {"factual", "true", "correct", "accurate"}:
        return "factual"
    if text in {"unfactual", "false", "incorrect", "inaccurate"}:
        return "unfactual"
    if text in {"uncertain", "unsure"}:
        return "uncertain"
    return None


def setup_client(
    backend: str,
    *,
    sglang_port: int = 30000,
    sglang_base_url: str = "http://127.0.0.1",
) -> NvidiaChatClient | OpenRouterChatClient | SGLangChatClient:
    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        return OpenRouterChatClient(api_key=api_key)
    if backend == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not set")
        return NvidiaChatClient(api_key=api_key)
    if backend == "sglang":
        return SGLangChatClient(port=sglang_port, base_url=sglang_base_url)
    raise ValueError(f"Unsupported backend: {backend}")


def evaluate_claim(
    entry: Dict[str, Any],
    *,
    client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    model: str,
    prompt_template: str,
    temperature: float,
) -> Dict[str, Any]:
    prompt = format_prompt(prompt_template, entry)
    messages = [{"role": "user", "content": prompt}]

    response = client.create_completion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    parsed = extract_json(response)

    verdict = normalize_label(parsed.get("verdict"))
    confidence = parsed.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_val = max(0.0, min(1.0, float(confidence)))
    else:
        confidence_val = None

    explanation = parsed.get("explanation")

    return {
        "verdict": verdict,
        "confidence": confidence_val,
        "explanation": explanation.strip() if isinstance(explanation, str) else None,
        "raw_response": response,
    }


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    skipped = 0
    by_gold: Dict[str, Dict[str, int]] = {}

    for res in results:
        total += 1
        gold = res.get("curated_label")
        pred = res.get("predicted_label")
        if pred is None:
            skipped += 1
        if gold == pred:
            correct += 1
        by_gold.setdefault(gold, {"total": 0, "correct": 0})
        by_gold[gold]["total"] += 1
        if gold == pred:
            by_gold[gold]["correct"] += 1

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "skipped": skipped,
        "per_label": {
            label: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
            for label, stats in by_gold.items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate factuality judgement on curated USMLE claims",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--claims", type=Path, required=True,
                        help="Curated claims JSON produced by curate_claims_via_llm.py")
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier to evaluate")
    parser.add_argument("--backend", type=str, default="openrouter",
                        choices=["openrouter", "nvidia", "sglang"],
                        help="API backend for model access")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT,
                        help="Prompt template YAML for evaluation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel evaluation workers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of claims (debugging)")
    parser.add_argument("--output", type=Path,
                        help="Optional explicit output file")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/tests/claim_factuality"),
                        help="Directory for test results when --output not provided")
    parser.add_argument("--sglang-port", type=int, default=30000,
                        help="SGLang server port (when backend=sglang)")
    parser.add_argument("--sglang-base-url", type=str, default="http://127.0.0.1",
                        help="SGLang base URL")
    args = parser.parse_args()

    claims_payload = json.loads(args.claims.read_text())
    claims = claims_payload.get("claims") if isinstance(claims_payload, dict) else claims_payload
    if not isinstance(claims, list):
        raise ValueError("Curated claims file must contain a 'claims' list")

    filtered_claims = [c for c in claims if normalize_label(c.get("verdict")) in {"factual", "unfactual"}]
    for claim in filtered_claims:
        claim["curated_label"] = normalize_label(claim.get("verdict"))

    if args.limit is not None:
        filtered_claims = filtered_claims[: args.limit]

    if not filtered_claims:
        raise ValueError("No curated claims with factual/unfactual labels found")

    prompt_data = load_prompt_template(args.prompt)
    prompt_template = prompt_data["prompt"]

    client = setup_client(
        args.backend,
        sglang_port=args.sglang_port,
        sglang_base_url=args.sglang_base_url,
    )

    results: List[Dict[str, Any]] = []

    def run_task(claim_entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        if isinstance(client, SGLangChatClient):
            thread_client = SGLangChatClient(port=args.sglang_port, base_url=args.sglang_base_url)
            outcome = evaluate_claim(
                claim_entry,
                client=thread_client,
                model=args.model,
                prompt_template=prompt_template,
                temperature=args.temperature,
            )
        else:
            outcome = evaluate_claim(
                claim_entry,
                client=client,
                model=args.model,
                prompt_template=prompt_template,
                temperature=args.temperature,
            )
        return claim_entry.get("claim_id"), outcome

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_task, claim_entry): claim_entry
            for claim_entry in filtered_claims
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            claim_entry = futures[future]
            claim_id = claim_entry.get("claim_id")
            try:
                outcome = future.result()
                _, annotation = outcome
                predicted_label = annotation.get("verdict")
            except Exception as exc:
                annotation = {
                    "verdict": None,
                    "confidence": None,
                    "explanation": None,
                    "raw_response": str(exc),
                }
                predicted_label = None
            result_row = {
                "claim_id": claim_id,
                "case_id": claim_entry.get("case_id"),
                "claim_text": claim_entry.get("claim_text"),
                "curated_label": claim_entry.get("curated_label"),
                "predicted_label": predicted_label,
                "correct": predicted_label == claim_entry.get("curated_label"),
                "model_output": annotation,
            }
            results.append(result_row)

    results.sort(key=lambda r: r["claim_id"] or "")

    summary = summarize(results)

    output_dir = args.output_dir
    output_path = args.output
    if output_path is None:
        safe_model = args.model.replace("/", "-").replace(":", "-")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"test_claim_factuality_{safe_model}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": args.model,
        "backend": args.backend,
        "prompt_file": str(args.prompt),
        "temperature": args.temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims_source": str(args.claims),
        "metrics": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved claim factuality evaluation → {output_path}")


if __name__ == "__main__":
    main()
