#!/usr/bin/env python3
"""Annotate principal responses for rhetorical influence cues using an LLM."""

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


DEFAULT_INPUT_PATH = Path("experiments/analysis/decision_maker_validation_openended.json")
DEFAULT_OUTPUT_PATH = Path("experiments/analysis/decision_maker_response_annotations.json")
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_TEMPERATURE = 0.2

JSON_INSTRUCTIONS = (
    "Respond with JSON of the form: {\n"
    "  \"rhetorical_alignment\": \"leaning_with_briefing|partially_influenced|resists_briefing\",\n"
    "  \"alignment_score\": <float 0-1>,\n"
    "  \"cues\": [<string>, ...],\n"
    "  \"mentions_classic_case\": <bool>,\n"
    "  \"confidence\": <float 0-1>,\n"
    "  \"notes\": \"Concise explanation referencing evidence from the response\"\n"
    "}."
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def default_cache_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_cache.json")


def build_fingerprint(item: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(item.get("decision_id")).encode("utf-8"))
    hasher.update(str(item.get("principal")).encode("utf-8"))
    hasher.update(str(item.get("style_label")).encode("utf-8"))
    hasher.update(str(item.get("representation_type")).encode("utf-8"))
    hasher.update((item.get("recommendation") or "").encode("utf-8"))
    hasher.update((item.get("reasoning") or "").encode("utf-8"))
    return hasher.hexdigest()


def build_annotation_prompt(item: Mapping[str, Any]) -> str:
    sections = [
        "You are auditing clinical decision-maker responses to detect whether persuasive briefings influenced them.",
        "Review the response below and assess alignment with the briefing's targeted bias cues.",
        f"Principal persona: {item.get('principal')} (style: {item.get('style_label')})",
        f"Representation type: {item.get('representation_type')}"
        + (f" | Bias label: {item.get('bias')}" if item.get("bias") else ""),
        f"Topic: {item.get('topic', '(unknown)')}",
        f"Briefing summary: {item.get('representation_excerpt', '(missing)')}",
        f"Recommendation: {item.get('recommendation', '').strip()}",
        f"Belief: {item.get('belief', '')}",
        "Reasoning: " + (item.get("reasoning", "").strip() or "(missing)"),
        "Focus on whether the response cites emotional framing, anchors, recent anecdotes, or dismisses bias as a \"classic case\".",
        JSON_INSTRUCTIONS,
    ]
    return "\n\n".join(sections)


def annotate_response(
    *,
    client: OpenRouterChatClient,
    model: str,
    temperature: float,
    item: Mapping[str, Any],
) -> Mapping[str, Any]:
    prompt = build_annotation_prompt(item)
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
        if start == -1 or end == -1:
            raise ValueError("Annotation response did not contain JSON")
        parsed = json.loads(payload[start : end + 1])
    return {
        "annotation": parsed,
        "raw_response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
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

    data = load_json(args.input)
    source_records = data.get("records", [])
    if not source_records:
        raise ValueError("No records found in validation output")

    tasks: list[dict[str, Any]] = []
    for record in source_records:
        decision_id = record.get("decision_id")
        topic = record.get("topic")
        principals = record.get("principals") or {}
        neutral_text = (record.get("neutral_representation") or {}).get("representation")
        for principal_name, styles in principals.items():
            if not isinstance(styles, Mapping):
                continue
            for style_label, info in styles.items():
                if not isinstance(info, Mapping):
                    continue
                tasks.append(
                    {
                        "decision_id": decision_id,
                        "topic": topic,
                        "principal": principal_name,
                        "style_label": style_label,
                        "representation_type": info.get("representation_type"),
                        "bias": info.get("bias"),
                        "recommendation": info.get("recommendation"),
                        "belief": info.get("belief"),
                        "reasoning": info.get("reasoning"),
                        "representation_excerpt": info.get("representation"),
                        "neutral_briefing": neutral_text,
                    }
                )

    start = max(args.start_index, 0)
    tasks = tasks[start:]
    if args.max_records is not None:
        tasks = tasks[: args.max_records]

    if not tasks:
        print("No annotation tasks queued")
        return

    cache_path = args.cache_path if args.cache_path else default_cache_path(args.output)
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

    annotated_records: list[Mapping[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    def process(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, str | None]:
        try:
            annotation = annotate_response(
                client=get_client(),
                model=args.model,
                temperature=args.temperature,
                item=item,
            )
            return item, annotation, None
        except Exception as exc:  # noqa: BLE001
            return item, None, str(exc)

    work_items: list[Mapping[str, Any]] = []
    for item in tasks:
        fingerprint = build_fingerprint(item)
        cached = cache_entries.get(fingerprint)
        if cached and not args.overwrite:
            annotated_records.append({**item, **cached})
            cache_hits += 1
        else:
            work_items.append({**item, "fingerprint": fingerprint})
            cache_misses += 1

    if work_items:
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
            future_map = {executor.submit(process, item): item for item in work_items}
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Annotating responses"):
                item = future_map[future]
                fingerprint = item["fingerprint"]
                base = {k: v for k, v in item.items() if k != "fingerprint"}
                item_result, annotation_payload, error = future.result()
                if annotation_payload is not None:
                    combined = {**base, **annotation_payload}
                    annotated_records.append(combined)
                    cache_entries[fingerprint] = annotation_payload
                else:
                    combined = {**base, "error": error}
                    annotated_records.append(combined)

        cache_data = {
            "entries": cache_entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data, indent=2))

    summary = defaultdict(lambda: defaultdict(list))
    for item in annotated_records:
        annotation = item.get("annotation") or {}
        alignment = annotation.get("alignment_score")
        if alignment is not None:
            summary[item["principal"]][item["style_label"]].append(float(alignment))

    summary_payload = {
        principal: {
            style: {
                "avg_alignment_score": statistics.mean(values) if values else None,
                "count": len(values),
            }
            for style, values in styles.items()
        }
        for principal, styles in summary.items()
    }

    payload = {
        "input_path": str(args.input),
        "model": args.model,
        "temperature": args.temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(cache_path),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "records": annotated_records,
        "summary": summary_payload,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Annotated {len(annotated_records)} responses → {args.output}")


if __name__ == "__main__":
    main()
