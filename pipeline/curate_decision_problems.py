#!/usr/bin/env python3
"""Curate simplified decision problems with rhetorical variants from USMLE data.

The script reads structured USMLE questions, calls an OpenRouter model to
compress each case into a short decision scenario, and produces multiple
rhetorical presentations intended to probe rational vs behavioural responses.

Example:
    python pipeline/curate_decision_problems.py \
        --questions-path experiments/questions/clinical_questions_usmle_sample.json \
        --output experiments/decision_problems/usmle_rhetorical_decisions.json \
        --model deepseek/deepseek-chat-v3.1 \
        --max-problems 5

Environment:
    OPENROUTER_API_KEY – required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient


DEFAULT_QUESTIONS_PATH = "experiments/questions/clinical_questions_usmle_sample.json"
DEFAULT_OUTPUT_PATH = "experiments/decision_problems/usmle_rhetorical_decisions.json"
DEFAULT_PROMPT_PATH = (
    Path(__file__).parent.parent
    / "prompts"
    / "experiments"
    / "curate_decision_problems.yaml"
)
DEFAULT_MODEL = "anthropic/claude-haiku-4-5"


NEAR_MISS_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "without",
    "patient",
    "patients",
    "option",
    "treatment",
    "therapy",
    "management",
    "strategy",
    "approach",
    "dose",
    "dosing",
    "daily",
    "per",
    "or",
    "via",
    "route",
    "risk",
    "plan",
    "first",
    "line",
    "choice",
}


def load_prompt(path: Path) -> Mapping[str, str]:
    data = yaml.safe_load(path.read_text())
    if "system_prompt" not in data or "user_template" not in data:
        raise ValueError(
            f"Prompt YAML at {path} must contain system_prompt and user_template"
        )
    return data


def load_questions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def format_options(options: Mapping[str, str]) -> str:
    pairs = []
    for key in sorted(options.keys()):
        text = options[key].strip()
        pairs.append(f"{key}. {text}")
    return "\n".join(pairs)


def extract_json_block(text: str) -> dict[str, Any]:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object detected in model output")
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse JSON snippet: {err}") from err


def count_words(text: str) -> int:
    return len(text.strip().split())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def option_tokens(text: str) -> set[str]:
    tokens = {
        tok
        for tok in re.split(r"[^a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in NEAR_MISS_STOPWORDS
    }
    return tokens


def validate_curated_payload(payload: Mapping[str, Any]) -> None:
    required_str_fields = [
        "topic",
        "patient_profile",
        "concise_context",
        "decision_axis",
        "decision_question",
        "correct_option_id",
        "validation_notes",
    ]
    for field in required_str_fields:
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"Missing or empty field: {field}")

    axis = payload["decision_axis"].strip()
    if count_words(axis) > 12:
        raise ValueError("decision_axis must be at most 12 words")

    axis_tokens = [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", axis.lower())
        if len(token) >= 3
    ]
    if not axis_tokens:
        axis_tokens = [axis.lower()]

    haystacks = [payload["decision_question"].lower(), payload["concise_context"].lower()]
    options_block = " ".join(str(opt.get("text", "")) for opt in payload.get("options", []))
    haystacks.append(options_block.lower())
    if not any(keyword in hay for keyword in axis_tokens for hay in haystacks):
        raise ValueError("decision_axis keywords must appear in question, context, or options")

    options = payload.get("options")
    if not isinstance(options, list) or len(options) != 2:
        raise ValueError("Options must be a list with exactly two items")

    seen_ids: set[str] = set()
    correct_count = 0
    false_count = 0
    option_texts: dict[str, str] = {}
    option_token_sets: dict[str, set[str]] = {}
    for opt in options:
        if not isinstance(opt, Mapping):
            raise ValueError("Each option must be an object")
        opt_id = opt.get("id")
        if opt_id not in {"A", "B"}:
            raise ValueError(f"Option id must be 'A' or 'B' (got {opt_id!r})")
        if opt_id in seen_ids:
            raise ValueError("Duplicate option id detected")
        seen_ids.add(opt_id)
        if not isinstance(opt.get("text"), str) or not opt["text"].strip():
            raise ValueError("Option text missing")
        if count_words(opt["text"]) > 18:
            raise ValueError("Option text must be 18 words or fewer")
        opt_text_lower = opt["text"].lower()
        if not any(keyword in opt_text_lower for keyword in axis_tokens):
            raise ValueError(
                f"Option {opt_id} must restate the decision_axis wording"
            )
        option_texts[opt_id] = opt["text"].strip()
        option_token_sets[opt_id] = option_tokens(opt["text"])
        if not isinstance(opt.get("is_correct"), bool):
            raise ValueError("Option is_correct must be boolean")
        if opt["is_correct"]:
            correct_count += 1
        else:
            false_count += 1
    if correct_count != 1 or false_count != 1:
        raise ValueError("Binary options must include one true and one false choice")
    if payload["correct_option_id"] not in seen_ids:
        raise ValueError("correct_option_id must match one of the option ids")

    correct_option_id = payload["correct_option_id"]
    incorrect_option_id = next(opt_id for opt_id in seen_ids if opt_id != correct_option_id)
    correct_text = normalize_text(option_texts[correct_option_id])
    incorrect_text = normalize_text(option_texts[incorrect_option_id])
    if correct_text == incorrect_text:
        raise ValueError("Options must differ beyond the truth value")

    correct_tokens = option_token_sets[correct_option_id]
    incorrect_tokens = option_token_sets[incorrect_option_id]
    if not correct_tokens or not incorrect_tokens:
        raise ValueError("Options must contain clinically meaningful tokens")

    overlap = correct_tokens & incorrect_tokens
    if not overlap:
        raise ValueError("Near-miss distractor must share core clinical cues with correct option")
    min_token_count = min(len(correct_tokens), len(incorrect_tokens))
    if min_token_count and (len(overlap) / min_token_count) < 0.4:
        raise ValueError("Distractor must substantially overlap with correct option to qualify as near-miss")
    symmetric_diff = correct_tokens ^ incorrect_tokens
    if not symmetric_diff:
        raise ValueError("Near-miss distractor must differ on at least one decisive cue")


def curate_single_case(
    question: Mapping[str, Any],
    prompt: Mapping[str, str],
    client: OpenRouterChatClient,
    model: str,
    temperature: float,
    max_attempts: int,
) -> dict[str, Any]:
    system_prompt = prompt["system_prompt"].strip()
    user_template = prompt["user_template"]

    question_text = textwrap.dedent(str(question.get("question", ""))).strip()
    original_options = format_options(question.get("options", {}))
    answer_idx = question.get("answer_idx", "")
    answer_text = str(question.get("answer", "")).strip()

    user_message = user_template.format(
        question=question_text,
        options=original_options,
        answer_idx=answer_idx,
        answer_text=answer_text,
    )

    for attempt in range(1, max_attempts + 1):
        raw = client.create_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        ).strip()

        try:
            payload = extract_json_block(raw)
            validate_curated_payload(payload)
            return payload
        except Exception as exc:  # noqa: BLE001
            if attempt == max_attempts:
                raise ValueError(
                    f"Failed to curate case {question.get('id')}: {exc}"
                ) from exc

    raise RuntimeError("Unreachable: attempts exhausted without return")


def build_output_record(
    curated: Mapping[str, Any],
    question: Mapping[str, Any],
    record_id: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "id": record_id,
        "source_question_id": question.get("id"),
        "source_exam": question.get("meta_info"),
        "original_answer_idx": question.get("answer_idx"),
        "original_answer": question.get("answer"),
        "original_options": question.get("options"),
        "curation": {
            "model": model,
            "temperature": temperature,
            "timestamp": timestamp,
        },
        **curated,
    }


def prepare_existing(
    output_path: Path, overwrite: bool
) -> tuple[list[dict[str, Any]], set[str]]:
    if not output_path.exists() or overwrite:
        return [], set()
    existing = json.loads(output_path.read_text())
    records = existing.get("records", [])
    seen = {r.get("source_question_id") for r in records if r.get("source_question_id")}
    return records, seen


def write_output(
    *,
    output_path: Path,
    questions_path: Path,
    prompt_path: Path,
    model: str,
    temperature: float,
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> None:
    all_records = existing_records + new_records
    payload = {
        "source_questions_path": str(questions_path),
        "prompt_path": str(prompt_path),
        "model": model,
        "temperature": temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(all_records),
        "records": all_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-path", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-problems", type=int)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based index into the question list",
    )
    parser.add_argument("--id-prefix", default="usmle_decision")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8, help="Parallel workers for curation")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    questions_path = Path(args.questions_path)
    prompt_path = Path(args.prompt_path)
    output_path = Path(args.output)

    questions = load_questions(questions_path)
    prompt = load_prompt(prompt_path)
    existing_records, seen_sources = prepare_existing(output_path, args.overwrite)

    start = max(args.start_index, 0)
    if start >= len(questions):
        raise ValueError("start-index exceeds number of available questions")

    remaining_questions = questions[start:]
    if args.max_problems is not None:
        remaining_questions = [
            q for q in remaining_questions if q.get("id") not in seen_sources
        ]
        remaining_questions = remaining_questions[: args.max_problems]
    else:
        remaining_questions = [
            q for q in remaining_questions if q.get("id") not in seen_sources
        ]

    total_existing = len(existing_records)
    save_interval = max(args.save_interval, 1)

    thread_local_client: threading.local = threading.local()

    def get_thread_client() -> OpenRouterChatClient:
        client = getattr(thread_local_client, "client", None)
        if client is None:
            client = OpenRouterChatClient(api_key=api_key)
            thread_local_client.client = client
        return client

    def process_case(idx: int, question_data: Mapping[str, Any], record_id: str) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            client = get_thread_client()
            curated_payload = curate_single_case(
                question=question_data,
                prompt=prompt,
                client=client,
                model=args.model,
                temperature=args.temperature,
                max_attempts=args.max_attempts,
            )
            record = build_output_record(
                curated_payload,
                question_data,
                record_id,
                args.model,
                args.temperature,
            )
            return record_id, record, None
        except Exception as exc:  # noqa: BLE001
            question_id = str(question_data.get("id", f"idx_{idx}"))
            return record_id, None, f"{question_id}: {exc}"

    jobs: list[tuple[int, Mapping[str, Any], str]] = []
    counter = total_existing
    for idx, question in enumerate(remaining_questions, start=1):
        counter += 1
        record_id = f"{args.id_prefix}_{counter:05d}"
        jobs.append((idx, question, record_id))

    successful_records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = {
            executor.submit(process_case, idx, question, record_id): (idx, question, record_id)
            for idx, question, record_id in jobs
        }
        progress = tqdm(as_completed(futures), total=len(futures), desc="Curating decision problems")
        for future in progress:
            record_id, record, error = future.result()
            if record is not None:
                successful_records[record_id] = record
                if len(successful_records) % save_interval == 0:
                    ordered_records = [successful_records[key] for key in sorted(successful_records.keys())]
                    write_output(
                        output_path=output_path,
                        questions_path=questions_path,
                        prompt_path=prompt_path,
                        model=args.model,
                        temperature=args.temperature,
                        existing_records=existing_records,
                        new_records=ordered_records,
                    )
            else:
                failures.append(error or f"{record_id}: unknown error")

    ordered_records = [successful_records[key] for key in sorted(successful_records.keys())]
    if ordered_records:
        write_output(
            output_path=output_path,
            questions_path=questions_path,
            prompt_path=prompt_path,
            model=args.model,
            temperature=args.temperature,
            existing_records=existing_records,
            new_records=ordered_records,
        )

    total_saved = len(existing_records) + len(ordered_records)
    print(f"Saved {len(ordered_records)} new decision problems (total: {total_saved}) to {output_path}")
    if failures:
        print(f"Skipped {len(failures)} cases due to errors:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
