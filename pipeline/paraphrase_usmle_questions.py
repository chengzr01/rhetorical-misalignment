#!/usr/bin/env python3
"""Paraphrase USMLE question stems for data contamination analysis.

This utility rewrites the question text of `clinical_questions` style datasets
using an OpenRouter-accessible model.  The paraphrases preserve the original
medical facts while changing surface wording, enabling robustness checks
against memorisation or question leakage.

Example usage:
    python pipeline/paraphrase_usmle_questions.py \
        --input experiments/questions/clinical_questions_usmle_sample.json \
        --output experiments/questions/clinical_questions_usmle_sample_paraphrased.json \
        --model anthropic/claude-haiku-4.5

Environment:
    OPENROUTER_API_KEY (required)
    PARAPHRASE_QUESTION_MODEL (optional default override)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

# Allow importing shared OpenRouter client
sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient

DEFAULT_MODEL = os.getenv("PARAPHRASE_QUESTION_MODEL", "anthropic/claude-haiku-4.5")
DEFAULT_PROMPT_PATH = Path("prompts/paraphrase/paraphrase_question.yaml")


@dataclass
class PromptSpec:
    system_prompt: str
    user_template: str


def load_prompt(path: Path) -> PromptSpec:
    data = yaml.safe_load(path.read_text())
    system_prompt = data.get("system_prompt")
    user_template = data.get("user_template")
    if not system_prompt or not user_template:
        raise ValueError(
            f"Prompt YAML at {path} must contain 'system_prompt' and 'user_template'"
        )
    return PromptSpec(system_prompt=system_prompt, user_template=user_template)


def prepare_messages(question_text: str, prompt: PromptSpec) -> List[Dict[str, str]]:
    user_message = prompt.user_template.format(question=question_text.strip())
    return [
        {"role": "system", "content": prompt.system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def clean_response(text: str) -> str:
    """Normalise model responses to a plain question string."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    # Remove trivial wrappers some models sometimes prepend
    prefixes = [
        "Paraphrased question:",
        "Paraphrased Question:",
        "Rewritten question:",
        "Rewritten Question:",
        "Question:",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    if cleaned.startswith("\"") and cleaned.endswith("\"") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    # Remove accidental code fences
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.strip()

    return cleaned


def paraphrase_question(
    question_record: Dict[str, Any],
    client: OpenRouterChatClient,
    model: str,
    prompt: PromptSpec,
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    """Paraphrase one question record."""
    question_id = question_record.get("id")
    question_text = question_record.get("question", "").strip()
    if not question_text:
        raise ValueError(f"Question {question_id} is missing 'question' text")

    response = client.create_completion(
        model=model,
        messages=prepare_messages(question_text, prompt),
        temperature=temperature,
    )
    rewritten = clean_response(response)
    if not rewritten:
        raise ValueError(f"Empty paraphrase produced for question {question_id}")

    new_record = dict(question_record)
    new_record["original_question"] = question_text
    new_record["question"] = rewritten
    new_record["variant"] = "question_paraphrase"
    new_record["paraphrase_model"] = model
    new_record["paraphrase_temperature"] = temperature

    return question_id, new_record


def paraphrase_questions(
    *,
    questions: Iterable[Dict[str, Any]],
    existing_records: Dict[str, Dict[str, Any]] | None,
    prompt: PromptSpec,
    model: str,
    temperature: float,
    max_workers: int,
) -> List[Dict[str, Any]]:
    # Collect work items and inject already-finished records
    completed: Dict[str, Dict[str, Any]] = {}
    to_process: List[Dict[str, Any]] = []

    for record in questions:
        qid = record.get("id")
        if qid is None:
            raise ValueError("Each question must have an 'id'")
        if existing_records and qid in existing_records:
            existing = existing_records[qid]
            original = record.get("question")
            if existing.get("original_question") == original:
                completed[qid] = existing
                continue
        to_process.append(record)

    if to_process:
        print(f"Paraphrasing {len(to_process)} questions using {model} (T={temperature})…")
    else:
        print("All questions already paraphrased; skipping API calls.")

    if to_process:
        client = OpenRouterChatClient()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    paraphrase_question,
                    record,
                    client,
                    model,
                    prompt,
                    temperature,
                ): record.get("id")
                for record in to_process
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Paraphrasing"):
                qid, rewritten = future.result()
                completed[qid] = rewritten

    ordered_ids = [record.get("id") for record in questions]
    missing = [qid for qid in ordered_ids if qid not in completed]
    if missing:
        raise RuntimeError(f"Missing paraphrased records for ids: {missing[:10]}")

    return [completed[qid] for qid in ordered_ids]


def load_existing_records(path: Path) -> Dict[str, Dict[str, Any]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Existing paraphrased file {path} must contain a list")
    return {item.get("id"): item for item in data if item.get("id")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paraphrase USMLE question stems via OpenRouter to build contamination probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path,
                        default=Path("experiments/questions/clinical_questions_usmle_sample.json"),
                        help="Path to original clinical questions JSON")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/questions/clinical_questions_usmle_sample_paraphrased.json"),
                        help="Destination for paraphrased dataset")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH,
                        help="Prompt YAML with system/user instructions")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="OpenRouter model to use for paraphrasing")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for paraphrasing")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers for API calls")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing paraphrases when output already exists")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing output even when --resume is supplied")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N questions (debugging)")

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    questions = json.loads(args.input.read_text())
    if not isinstance(questions, list):
        raise ValueError("Input questions file must be a JSON list")

    if args.limit is not None:
        questions = questions[: args.limit]

    prompt = load_prompt(args.prompt)

    existing: Dict[str, Dict[str, Any]] | None = None
    if args.resume and not args.force:
        existing = load_existing_records(args.output)
        if existing:
            print(f"Loaded {len(existing)} existing paraphrased records from {args.output}")

    paraphrased = paraphrase_questions(
        questions=questions,
        existing_records=existing,
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(paraphrased, indent=2, ensure_ascii=False))
    print(f"Saved paraphrased dataset with {len(paraphrased)} questions → {args.output}")


if __name__ == "__main__":
    main()
