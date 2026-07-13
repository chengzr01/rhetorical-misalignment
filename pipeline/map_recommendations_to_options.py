#!/usr/bin/env python3
"""Map open-ended principal recommendations back to original answer options.

This utility loads the open-ended validation output produced by
`pipeline/validate_decision_makers.py` along with the source decision problems.
For each principal response, it queries an LLM to classify the free-text
recommendation (and accompanying reasoning) as matching option A or option B in
the original decision problem. The results provide per-principal accuracy and
confidence summaries, enabling comparison of rational vs behavioural personas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient


DEFAULT_DECISION_PATH = Path("experiments/decision_problems/usmle_rhetorical_decisions.json")
DEFAULT_VALIDATION_PATH = Path("experiments/analysis/decision_maker_validation_openended.json")
DEFAULT_OUTPUT_PATH = Path("experiments/analysis/principal_option_mapping.json")
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_TEMPERATURE = 0.1

JSON_INSTRUCTIONS = (
    "Respond strictly as JSON: {\n"
    "  \"chosen_option_id\": \"A|B|unknown\",\n"
    "  \"match_confidence\": <float 0-1>,\n"
    "  \"rationale\": \"Short explanation referencing the principal response and option text\"\n"
    "}."
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def build_fingerprint(item: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    for key in (
        "decision_id",
        "principal",
        "style_label",
        "representation_type",
        "recommendation",
        "reasoning",
    ):
        hasher.update(str(item.get(key, "")).encode("utf-8"))
    return hasher.hexdigest()


def build_prompt(item: Mapping[str, Any]) -> str:
    case = item["case"]
    option_lines = []
    for opt in case.get("options", []):
        option_lines.append(f"{opt.get('id')}. {opt.get('text')}")
    option_block = "\n".join(option_lines)

    sections = [
        "You are verifying whether a clinician's open-ended recommendation aligns with the original answer options.",
        "Base your decision on semantic meaning, not just keywords. Choose the option (A or B) that best matches the recommendation.",
        f"Decision ID: {item['decision_id']}",
        f"Principal persona: {item['principal']} (style: {item['style_label']})",
        f"Representation type: {item['representation_type']}",
        "Clinical vignette:",
        f"  Patient summary: {case.get('patient_profile', '')}",
        f"  Scenario: {case.get('concise_context', '')}",
        f"  Decision question: {case.get('decision_question', '')}",
        "Answer options:",
        option_block,
        "Principal response:",
        f"  Recommendation: {item.get('recommendation', '').strip()}",
        f"  Belief: {item.get('belief', '')}",
        f"  Reasoning: {item.get('reasoning', '').strip()}",
        "If the recommendation clearly matches one option, choose that option. If it is ambiguous or conflicts with both, output 'unknown'.",
        JSON_INSTRUCTIONS,
    ]
    return "\n\n".join(sections)


def classify_response(
    *,
    client: OpenRouterChatClient,
    model: str,
    temperature: float,
    item: Mapping[str, Any],
) -> Mapping[str, Any]:
    prompt = build_prompt(item)
    messages = [{"role": "user", "content": prompt}]
    response = client.create_completion(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    payload = response.strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Classification response did not contain JSON")
        parsed = json.loads(payload[start : end + 1])
    return {
        "classification": parsed,
        "raw_response": response,
    }


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-problems", type=Path, default=DEFAULT_DECISION_PATH)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-path", type=Path)

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    decision_data = load_json(args.decision_problems)
    decision_map = {str(rec.get("id")): rec for rec in decision_data.get("records", [])}
    if not decision_map:
        raise ValueError("Decision problems file did not contain records")

    validation_data = load_json(args.validation)
    validation_records = validation_data.get("records", [])
    if not validation_records:
        raise ValueError("Validation file did not contain records")

    tasks: list[dict[str, Any]] = []
    for record in validation_records:
        decision_id = str(record.get("decision_id"))
        case = decision_map.get(decision_id)
        if not case:
            continue
        principals = record.get("principals") or {}
        for principal_name, style_map in principals.items():
            if not isinstance(style_map, Mapping):
                continue
            for style_label, info in style_map.items():
                if not isinstance(info, Mapping):
                    continue
                tasks.append(
                    {
                        "decision_id": decision_id,
                        "topic": record.get("topic"),
                        "principal": principal_name,
                        "style_label": style_label,
                        "representation_type": info.get("representation_type"),
                        "bias": info.get("bias"),
                        "recommendation": info.get("recommendation"),
                        "belief": info.get("belief"),
                        "reasoning": info.get("reasoning"),
                        "case": case,
                        "correct_option_id": case.get("correct_option_id"),
                    }
                )

    start = max(args.start_index, 0)
    tasks = tasks[start:]
    if args.max_records is not None:
        tasks = tasks[: args.max_records]
    if not tasks:
        print("No tasks to classify")
        return

    cache_path = args.cache_path or args.output.with_name(
        f"{args.output.stem}_cache.json"
    )
    if args.overwrite or not cache_path.exists():
        cache_data = {"entries": {}, "updated_at": None}
    else:
        try:
            cache_data = load_json(cache_path)
        except json.JSONDecodeError:
            cache_data = {"entries": {}, "updated_at": None}
    cache_entries = cache_data.get("entries", {}) or {}

    client_pool: threading.local = threading.local()

    def get_client() -> OpenRouterChatClient:
        client = getattr(client_pool, "client", None)
        if client is None:
            client = OpenRouterChatClient(api_key=api_key)
            client_pool.client = client
        return client

    classified: list[Mapping[str, Any]] = []
    work_items: list[Mapping[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    for item in tasks:
        fingerprint = build_fingerprint(item)
        cached = cache_entries.get(fingerprint)
        if cached and not args.overwrite:
            classified.append({**item, **cached})
            cache_hits += 1
        else:
            work_items.append({**item, "fingerprint": fingerprint})
            cache_misses += 1

    def process(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, str | None]:
        try:
            result = classify_response(
                client=get_client(),
                model=args.model,
                temperature=args.temperature,
                item=item,
            )
            return item, result, None
        except Exception as exc:  # noqa: BLE001
            return item, None, str(exc)

    if work_items:
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
            future_map = {executor.submit(process, item): item for item in work_items}
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Mapping recommendations"):
                item = future_map[future]
                fingerprint = item["fingerprint"]
                base_item = {k: v for k, v in item.items() if k != "fingerprint"}
                _, result, error = future.result()
                if result is not None:
                    record = {**base_item, **result}
                    classified.append(record)
                    cache_entries[fingerprint] = result
                else:
                    record = {**base_item, "error": error}
                    classified.append(record)

        cache_payload = {
            "entries": cache_entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_payload, indent=2))

    # compute accuracy summaries
    summary: dict[str, Any] = {
        "global": {
            "total": len(classified),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "accuracy": None,
        },
        "by_principal": defaultdict(
            lambda: {
                "total": 0,
                "correct": 0,
                "accuracy": None,
                "avg_match_confidence": None,
                "by_style": defaultdict(
                    lambda: {
                        "total": 0,
                        "correct": 0,
                        "accuracy": None,
                        "avg_match_confidence": None,
                    }
                ),
            }
        ),
    }

    global_confidences: list[float] = []
    per_principal_conf: dict[str, list[float]] = defaultdict(list)
    per_style_conf: dict[tuple[str, str], list[float]] = defaultdict(list)

    for record in classified:
        classification = record.get("classification") or {}
        chosen = (classification.get("chosen_option_id") or "").strip().upper()
        confidence = classification.get("match_confidence")
        correct_option = (record.get("correct_option_id") or "").strip().upper()
        principal = record.get("principal")
        style_label = record.get("style_label")
        principal_summary = summary["by_principal"][principal]
        style_summary = principal_summary["by_style"][style_label]

        principal_summary["total"] += 1
        style_summary["total"] += 1
        summary["global"]["total"] += 0  # keep key visible

        if confidence is not None:
            confidence_value = float(confidence)
            global_confidences.append(confidence_value)
            per_principal_conf[principal].append(confidence_value)
            per_style_conf[(principal, style_label)].append(confidence_value)

        if chosen and correct_option and chosen == correct_option:
            principal_summary["correct"] += 1
            style_summary["correct"] += 1

    for principal, data in summary["by_principal"].items():
        if data["total"]:
            data["accuracy"] = data["correct"] / data["total"]
        data["avg_match_confidence"] = safe_mean(per_principal_conf.get(principal, []))
        for style, style_data in data["by_style"].items():
            if style_data["total"]:
                style_data["accuracy"] = style_data["correct"] / style_data["total"]
            style_data["avg_match_confidence"] = safe_mean(
                per_style_conf.get((principal, style), [])
            )

    total = sum(data["total"] for data in summary["by_principal"].values())
    correct = sum(data["correct"] for data in summary["by_principal"].values())
    summary["global"]["total"] = total
    summary["global"]["accuracy"] = (correct / total) if total else None
    summary["global"]["avg_match_confidence"] = safe_mean(global_confidences)

    payload = {
        "decision_problems_path": str(args.decision_problems),
        "validation_path": str(args.validation),
        "model": args.model,
        "temperature": args.temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "records": classified,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(
        f"Mapped {len(classified)} responses | global accuracy: {summary['global']['accuracy']}"
    )
    for principal, data in summary["by_principal"].items():
        print(
            f"  {principal}: accuracy={data['accuracy']} (correct {data['correct']}/{data['total']}), "
            f"avg_match_confidence={data['avg_match_confidence']}"
        )


if __name__ == "__main__":
    main()
