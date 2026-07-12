#!/usr/bin/env python3
"""Paraphrase USMLE answer choices for data-contamination probes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

# Ensure shared OpenRouter client is importable when invoked directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient  # noqa: E402

DEFAULT_MODEL = os.getenv(
    "PARAPHRASE_OPTION_MODEL",
    os.getenv("PARAPHRASE_MODEL", "deepseek/deepseek-chat-v3.1"),
)
DEFAULT_PROMPT_PATH = Path("prompts/paraphrase/paraphrase_option.yaml")


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
            f"Prompt YAML at {path} must contain 'system_prompt' and 'user_template' entries"
        )
    return PromptSpec(system_prompt=system_prompt, user_template=user_template)


def prepare_messages(
    *,
    question_text: str,
    option_label: str,
    option_text: str,
    prompt: PromptSpec,
) -> List[Dict[str, str]]:
    user_message = prompt.user_template.format(
        question=question_text.strip(),
        option_label=option_label,
        option_text=option_text.strip(),
    )
    return [
        {"role": "system", "content": prompt.system_prompt.strip()},
        {"role": "user", "content": user_message.strip()},
    ]


def clean_response(text: str) -> str:
    """Normalise model responses to plain option text."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    prefixes = [
        "Paraphrased option:",
        "Paraphrased Option:",
        "Rewritten option:",
        "Rewritten Option:",
        "Option:",
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    if cleaned.startswith("\"") and cleaned.endswith("\"") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()

    cleaned = cleaned.lstrip("-•")
    if (
        len(cleaned) > 2
        and cleaned[0] in string.ascii_uppercase
        and cleaned[1:3] in {". ", ": ", ") "}
    ):
        cleaned = cleaned[2:].strip()

    return cleaned


def paraphrase_options_for_question(
    question_record: Dict[str, Any],
    *,
    client: OpenRouterChatClient,
    prompt: PromptSpec,
    model: str,
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    question_id = question_record.get("id")
    if question_id is None:
        raise ValueError("Each question must include an 'id' field")

    question_text = question_record.get("question", "").strip()
    if not question_text:
        raise ValueError(f"Question {question_id} is missing 'question' text")

    options = question_record.get("options")
    if not isinstance(options, dict) or not options:
        raise ValueError(f"Question {question_id} must contain a non-empty 'options' dict")
    if len(options) > len(string.ascii_uppercase):
        raise ValueError(
            f"Question {question_id} has {len(options)} options; maximum supported is {len(string.ascii_uppercase)}"
        )

    new_options: Dict[str, str] = {}
    for label, option_text in sorted(options.items()):
        response = client.create_completion(
            model=model,
            messages=prepare_messages(
                question_text=question_text,
                option_label=label,
                option_text=str(option_text),
                prompt=prompt,
            ),
            temperature=temperature,
        )
        rewritten = clean_response(response)
        if not rewritten:
            raise ValueError(
                f"Empty paraphrase produced for question {question_id} option {label}"
            )
        new_options[label] = rewritten

    original_answer_idx = question_record.get("answer_idx")
    if original_answer_idx is None:
        raise ValueError(f"Question {question_id} is missing 'answer_idx'")
    if original_answer_idx not in new_options:
        raise ValueError(
            f"Question {question_id} correct label {original_answer_idx} not found in options"
        )

    new_record = copy.deepcopy(question_record)
    new_record["original_question"] = question_text
    new_record["original_options"] = copy.deepcopy(options)
    if "answer" in question_record:
        new_record.setdefault("original_answer", question_record.get("answer"))
    new_record.setdefault("original_answer_idx", original_answer_idx)

    new_record["options"] = new_options
    new_record["answer_idx"] = original_answer_idx
    new_record["answer"] = new_options[original_answer_idx]
    new_record["variant"] = "option_paraphrase"
    new_record["paraphrase_model"] = model
    new_record["paraphrase_temperature"] = temperature

    return str(question_id), new_record


def paraphrase_option_dataset(
    *,
    questions: Iterable[Dict[str, Any]],
    existing_records: Dict[str, Dict[str, Any]] | None,
    prompt: PromptSpec,
    model: str,
    temperature: float,
    max_workers: int,
) -> List[Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    to_process: List[Dict[str, Any]] = []

    for record in questions:
        qid = record.get("id")
        if qid is None:
            raise ValueError("Every question must have an 'id'")
        if existing_records and qid in existing_records:
            cached = existing_records[qid]
            if (
                cached.get("original_question") == record.get("question")
                and cached.get("original_options") == record.get("options")
            ):
                completed[qid] = cached
                continue
        to_process.append(record)

    if to_process:
        print(
            f"Paraphrasing answer choices for {len(to_process)} questions using {model} (T={temperature})…"
        )
    else:
        print("All option paraphrases already cached; skipping API calls.")

    if to_process:
        client = OpenRouterChatClient()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    paraphrase_options_for_question,
                    record,
                    client=client,
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                ): record.get("id")
                for record in to_process
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Paraphrasing options",
            ):
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
        raise ValueError(f"Cached paraphrase file {path} must contain a list of records")
    records: Dict[str, Dict[str, Any]] = {}
    for item in data:
        qid = item.get("id")
        if qid is not None:
            records[qid] = item
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paraphrase USMLE answer choices to create contamination probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/questions/clinical_questions_usmle_sample.json"),
        help="Source clinical questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/questions/data_contamination/clinical_questions_usmle_sample_options_paraphrased.json"
        ),
        help="Destination for paraphrased option dataset",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Prompt YAML providing paraphrasing instructions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter model to use for option paraphrasing",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for paraphrasing",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent paraphrasing workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached paraphrases when available",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached paraphrases even if --resume is set",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N questions (debugging)",
    )

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
            print(
                f"Loaded {len(existing)} cached option paraphrases from {args.output}"
            )

    paraphrased_records = paraphrase_option_dataset(
        questions=questions,
        existing_records=existing,
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_workers=args.max_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(paraphrased_records, indent=2, ensure_ascii=False))
    print(
        f"Saved paraphrased options for {len(paraphrased_records)} questions → {args.output}"
    )


if __name__ == "__main__":
    main()
