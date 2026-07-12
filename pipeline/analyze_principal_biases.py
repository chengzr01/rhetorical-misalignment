#!/usr/bin/env python3
"""Annotate cognitive biases in simulated principal responses using OpenRouter.

This script scans baseline principal inference outputs and labels each
response with the behavioral biases defined in
`prompts/principal/behavioral_belief.yaml`.  The annotations support
analysis of how often simulated decision-makers exhibit each bias.

Example usage:
    python pipeline/analyze_principal_biases.py \
        --input-dir experiments/principals/usmle_sample/baseline \
        --output-dir experiments/analysis/bias_annotations/usmle_sample/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient

DEFAULT_INPUT_DIR = Path("experiments/principals/usmle_sample/baseline")
DEFAULT_OUTPUT_DIR = Path("experiments/analysis/bias_annotations/usmle_sample/baseline")
DEFAULT_PROMPT_PATH = Path("prompts/principal/behavioral_belief.yaml")
DEFAULT_MODEL = os.getenv("BIAS_ANNOTATOR_MODEL", "deepseek/deepseek-chat-v3.1")
SYSTEM_PROMPT = (
    "You are an expert cognitive psychologist. Given a clinical decision-making "
    "response, identify which (if any) of the specified cognitive biases are "
    "clearly exhibited. Only use the provided bias definitions. Return strict JSON"
)

JSON_INSTRUCTIONS = (
    "Respond with JSON of the form:\n"
    "{\n"
    "  \"bias_labels\": [\"Bias Name\", ...],\n"
    "  \"explanations\": {\"Bias Name\": \"Concise evidence from the response...\"},\n"
    "  \"confidence\": <number between 0 and 1 indicating classification confidence>\n"
    "}.\n"
    "Use exact bias names. Include explanations only for biases you list."
)

BIAS_REGEX = re.compile(r"-\s+\*\*(?P<name>[^*]+)\*\*:\s*(?P<desc>.+)")


def load_bias_definitions(path: Path) -> List[Dict[str, str]]:
    data = yaml.safe_load(path.read_text())
    prompt_text = data.get("prompt", "")
    biases: List[Dict[str, str]] = []
    for line in prompt_text.splitlines():
        match = BIAS_REGEX.match(line.strip())
        if match:
            biases.append({
                "name": match.group("name").strip(),
                "description": match.group("desc").strip(),
            })
    if not biases:
        raise ValueError(f"Could not extract bias definitions from {path}")
    return biases


def format_options(options: Dict[str, str]) -> str:
    parts = [f"{label}. {text}" for label, text in sorted(options.items())]
    return "\n".join(parts)


def build_user_message(entry: Dict[str, Any], bias_text: str) -> str:
    question = entry.get("principal_context") or entry.get("agent_context") or "(missing)"
    options = format_options(entry.get("options", {})) or "(missing)"
    decision = entry.get("decision") or "None"
    belief = entry.get("belief")
    reasoning = entry.get("reasoning") or entry.get("raw_principal_response") or "(no reasoning provided)"

    belief_line = "Belief (probability): " + (f"{belief}" if belief is not None else "None")

    return (
        "Evaluate the decision-maker's reasoning for evidence of the listed cognitive biases.\n\n"
        f"Cognitive biases:\n{bias_text}\n\n"
        "Clinical question:\n"
        f"{question}\n\n"
        "Answer options:\n"
        f"{options}\n\n"
        "Decision-maker output:\n"
        f"Chosen answer: {decision}\n"
        f"{belief_line}\n"
        f"Reasoning text:\n{reasoning}\n\n"
        "Determine which biases are strongly or clearly expressed. Choose zero or more.\n"
        "Only label a bias when the reasoning provides explicit or strongly implied evidence.\n"
        "If none apply, return an empty list.\n\n"
        f"{JSON_INSTRUCTIONS}"
    )


def extract_json_block(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty response from annotator")

    if text.startswith("```"):
        # Strip optional code fences
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Attempt direct parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Search for JSON object substring
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidate = text[start : end + 1]
        json.loads(candidate)  # raises if invalid
        return candidate

    raise ValueError("Annotator response did not contain valid JSON")


def annotate_entry(
    entry: Dict[str, Any],
    *,
    client: OpenRouterChatClient,
    model: str,
    bias_text: str,
    temperature: float,
) -> Dict[str, Any]:
    user_message = build_user_message(entry, bias_text)
    response = client.create_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    payload = extract_json_block(response)
    parsed = json.loads(payload)

    labels = parsed.get("bias_labels")
    if labels is None or not isinstance(labels, list):
        raise ValueError("Annotator JSON missing 'bias_labels' list")

    explanations = parsed.get("explanations") or {}
    if not isinstance(explanations, dict):
        explanations = {}

    confidence = parsed.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_val = max(0.0, min(1.0, float(confidence)))
    else:
        confidence_val = None

    return {
        "case_id": entry.get("case_id"),
        "principal_name": entry.get("principal_name"),
        "principal_model": entry.get("principal_model"),
        "decision": entry.get("decision"),
        "belief": entry.get("belief"),
        "bias_labels": [str(label).strip() for label in labels if str(label).strip()],
        "explanations": {str(k).strip(): str(v).strip() for k, v in explanations.items() if str(k).strip()},
        "confidence": confidence_val,
        "raw_response": entry.get("raw_principal_response"),
    }


def load_existing_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    annotations = data.get("annotations")
    if not isinstance(annotations, list):
        return {}
    return {a.get("case_id"): a for a in annotations if a.get("case_id") is not None}


def summarize_annotations(annotations: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counter = Counter()
    total = 0
    for ann in annotations:
        total += 1
        for label in ann.get("bias_labels", []):
            counter[label] += 1
    return {
        "total_cases": total,
        "bias_counts": dict(counter),
        "bias_prevalence": {label: count / total if total else 0.0 for label, count in counter.items()},
    }


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    client: OpenRouterChatClient,
    model: str,
    bias_text: str,
    max_workers: int,
    resume: bool,
    temperature: float,
    limit: int | None,
) -> None:
    entries = json.loads(input_path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"Expected list in {input_path}")

    if limit is not None:
        entries = entries[:limit]

    existing = load_existing_annotations(output_path) if resume else {}

    todo: List[Tuple[str, Dict[str, Any]]] = []
    annotations: Dict[str, Dict[str, Any]] = dict(existing)

    for entry in entries:
        raw_case_id = entry.get("case_id")
        if raw_case_id is None:
            raise ValueError(f"Entry missing 'case_id' in {input_path}")
        case_id = str(raw_case_id)
        if case_id in existing:
            continue
        todo.append((case_id, entry))

    if todo:
        print(f"Annotating {len(todo)} responses from {input_path.name} using {model}...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    annotate_entry,
                    entry,
                    client=client,
                    model=model,
                    bias_text=bias_text,
                    temperature=temperature,
                ): case_id
                for case_id, entry in todo
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=input_path.stem):
                case_id = futures[future]
                try:
                    annotations[str(case_id)] = future.result()
                except Exception as e:
                    print(f"  [WARNING] Failed to annotate case {case_id}: {e}")
    else:
        print(f"All responses already annotated for {input_path.name}; skipping API calls.")

    ordered_annotations = [annotations[str(entry.get("case_id"))] for entry in entries if str(entry.get("case_id")) in annotations]
    summary = summarize_annotations(ordered_annotations)

    payload = {
        "source_file": str(input_path),
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bias_definitions": bias_text,
        "summary": summary,
        "annotations": ordered_annotations,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved annotations → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate cognitive biases in principal baseline responses via OpenRouter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, nargs="*",
                        help="Specific principal JSON files to annotate")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing principal baseline outputs")
    parser.add_argument("--glob", type=str, default="principal_*.json",
                        help="Glob pattern (relative to input dir) to match files when --input not provided")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write bias annotation files")
    parser.add_argument("--bias-prompt", type=Path, default=DEFAULT_PROMPT_PATH,
                        help="Prompt YAML containing behavioral bias definitions")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="OpenRouter model to use for annotation")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for annotation model")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers for annotation requests")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing annotation files when present")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of responses per file (debugging)")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    bias_defs = load_bias_definitions(args.bias_prompt)
    bias_text = "\n".join(f"- {item['name']}: {item['description']}" for item in bias_defs)

    client = OpenRouterChatClient(api_key=api_key)

    if args.input:
        files = [path if path.is_absolute() else Path(path) for path in args.input]
    else:
        files = sorted(args.input_dir.glob(args.glob))

    if not files:
        print("No principal files found to annotate.")
        return

    for input_path in files:
        if not input_path.exists():
            print(f"[WARNING] Skipping missing file: {input_path}")
            continue
        relative_name = input_path.stem + "_bias_annotations.json"
        output_path = args.output_dir / relative_name
        process_file(
            input_path=input_path,
            output_path=output_path,
            client=client,
            model=args.model,
            bias_text=bias_text,
            max_workers=args.max_workers,
            resume=args.resume,
            temperature=args.temperature,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
