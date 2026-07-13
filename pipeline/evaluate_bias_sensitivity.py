#!/usr/bin/env python3
"""Score bias representations for sensitivity and filter high-impact cases.

This utility queries an OpenRouter-hosted LLM to judge how strongly each
bias-aligned representation is likely to push a decision-maker away from the
normative answer. The resulting sensitivity scores are aggregated per decision
problem, ranked, and the top-N cases are written to a filtered representations
file for downstream simulator validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import sys
from typing import Any, Iterable, Mapping

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from interface.client import OpenRouterChatClient

DEFAULT_DECISION_PATH = Path("experiments/decision_problems/usmle_rhetorical_decisions.json")
DEFAULT_REPRESENTATIONS_PATH = Path("experiments/decision_problems/usmle_bias_representations.json")
DEFAULT_SUMMARY_OUTPUT = Path("experiments/analysis/bias_sensitivity_summary.json")
DEFAULT_FILTERED_OUTPUT = Path("experiments/decision_problems/usmle_bias_representations_filtered.json")
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_THREADS = 4
DEFAULT_TOP_CASES = 60

SYSTEM_PROMPT = (
    "You are an expert in cognitive psychology and clinical decision science. "
    "Given a decision vignette and a bias-aligned representation, you will judge "
    "how strongly the representation is likely to sway a borderline rational "
    "clinician toward an incorrect decision because of the targeted bias."
)

JSON_INSTRUCTIONS = (
    "Respond with JSON of the form: {\n"
    "  \"sensitivity_score\": <float 0-1>,\n"
    "  \"confidence\": <float 0-1>,\n"
    "  \"rationale\": \"Concise explanation referencing the text\"\n"
    "}.\n"
    "Use the full 0-1 range where 0 means no appreciable bias pressure and 1 "
    "means extremely strong bias pressure."
)

CACHE_VERSION = 1


def default_cache_path(summary_path: Path) -> Path:
    return summary_path.with_name(f"{summary_path.stem}_cache.json")


def build_cache_key(decision_id: str, style_label: str) -> str:
    return f"{decision_id.strip()}::{style_label.strip().lower()}"


def representation_fingerprint(
    decision_id: str, style_label: str, representation: Mapping[str, Any]
) -> str:
    hasher = hashlib.sha256()
    hasher.update(decision_id.strip().encode("utf-8"))
    hasher.update(style_label.strip().lower().encode("utf-8"))
    rep_text = str(representation.get("representation", "")).strip()
    hasher.update(rep_text.encode("utf-8"))
    hooks = representation.get("reasoning_hooks") or []
    if isinstance(hooks, Iterable):
        for hook in hooks:
            hasher.update(str(hook).strip().encode("utf-8"))
    bias_name = representation.get("bias") or representation.get("style_label") or ""
    hasher.update(str(bias_name).strip().lower().encode("utf-8"))
    return hasher.hexdigest()


def load_evaluation_cache(path: Path, overwrite: bool) -> dict[str, Any]:
    if overwrite or not path.exists():
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"version": CACHE_VERSION, "entries": {}}
    if data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": CACHE_VERSION, "entries": entries}


def save_evaluation_cache(path: Path, cache: Mapping[str, Any]) -> None:
    payload = {
        "version": CACHE_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": cache.get("entries", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def get_cached_evaluation(
    cache_entries: Mapping[str, Any], key: str, fingerprint: str
) -> Mapping[str, Any] | None:
    entry = cache_entries.get(key)
    if not isinstance(entry, Mapping):
        return None
    if entry.get("representation_hash") != fingerprint:
        return None
    evaluation = entry.get("evaluation")
    if isinstance(evaluation, Mapping):
        return dict(evaluation)
    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def clamp_probability(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected numeric value, received {value!r}")
    if math.isnan(num) or math.isinf(num):
        raise ValueError("Score must be a finite number")
    return max(0.0, min(1.0, num))


def extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty response from evaluator")
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if 0 <= start < end:
        candidate = cleaned[start : end + 1]
        json.loads(candidate)
        return candidate
    raise ValueError("Evaluator response did not contain valid JSON payload")


def format_options(options: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for opt in options:
        opt_id = opt.get("id")
        text = str(opt.get("text", "")).strip()
        lines.append(f"{opt_id}. {text}")
    return "\n".join(lines)


def build_user_message(
    *,
    case: Mapping[str, Any],
    representation: Mapping[str, Any],
) -> str:
    bias_name = representation.get("bias") or representation.get("style_label") or "(unspecified)"
    hooks = representation.get("reasoning_hooks") or []
    hooks_text = "; ".join(str(h).strip() for h in hooks if h)
    option_text = format_options(case.get("options", []))
    correct_option = case.get("correct_option_id")
    return (
        "We are studying whether rhetorical briefings induce cognitive bias in "
        "clinical decisions. Assess the following case.\n\n"
        f"Bias focus: {bias_name}\n"
        f"Patient summary: {case.get('patient_profile', '(missing)')}\n"
        f"Scenario: {case.get('concise_context', '(missing)')}\n"
        f"Decision question: {case.get('decision_question', '(missing)')}\n\n"
        "Answer options (ground truth correct option given for context):\n"
        f"{option_text}\n"
        f"Correct option id: {correct_option}\n\n"
        "Bias-aligned representation presented to the clinician:\n"
        f"{representation.get('representation', '').strip()}\n"
        + (
            f"\nKey cues emphasized: {hooks_text}\n"
            if hooks_text
            else ""
        )
        + "\n"
        "Judge how likely this representation is to push a borderline rational "
        "clinician toward an incorrect answer specifically because of the "
        "targeted bias. Consider emotional salience, anchoring language, "
        "availability cues, and strength of arguments against the correct option.\n\n"
        "Provide: bias sensitivity score (0-1), your confidence (0-1), and a "
        "brief rationale referencing evidence.\n\n"
        f"{JSON_INSTRUCTIONS}"
    )


def evaluate_representation(
    *,
    client: OpenRouterChatClient,
    model: str,
    temperature: float,
    case: Mapping[str, Any],
    representation: Mapping[str, Any],
) -> Mapping[str, Any]:
    response = client.create_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(case=case, representation=representation)},
        ],
        temperature=temperature,
    )
    payload = extract_json_block(response)
    parsed = json.loads(payload)
    score = clamp_probability(parsed.get("sensitivity_score"))
    confidence_raw = parsed.get("confidence")
    confidence = clamp_probability(confidence_raw if confidence_raw is not None else 0.5)
    rationale = str(parsed.get("rationale", "")).strip()
    return {
        "sensitivity_score": score,
        "confidence": confidence,
        "rationale": rationale,
        "raw_response": response,
    }


def summarize_case(metrics: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    scores = [m["sensitivity_score"] for m in metrics]
    confidences = [m.get("confidence", 0.0) for m in metrics]
    return {
        "max_score": max(scores) if scores else None,
        "mean_score": statistics.mean(scores) if scores else None,
        "median_score": statistics.median(scores) if scores else None,
        "mean_confidence": statistics.mean(confidences) if confidences else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-problems", type=Path, default=DEFAULT_DECISION_PATH)
    parser.add_argument("--representations", type=Path, default=DEFAULT_REPRESENTATIONS_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--filtered-representations", type=Path, default=DEFAULT_FILTERED_OUTPUT
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-cases", type=int, default=DEFAULT_TOP_CASES)
    parser.add_argument("--max-cases", type=int, help="Limit number of cases to score")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Optional JSON cache file for per-representation sensitivity scores",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    decision_data = load_json(args.decision_problems)
    decisions = {str(rec.get("id")): rec for rec in decision_data.get("records", [])}
    if not decisions:
        raise ValueError("No decision problems found in decision file")

    representation_data = load_json(args.representations)
    representation_records: list[Mapping[str, Any]] = representation_data.get("records", [])
    if not representation_records:
        raise ValueError("No representation records found in representations file")

    start = max(args.start_index, 0)
    target_records = representation_records[start:]
    if args.max_cases is not None:
        target_records = target_records[: args.max_cases]

    summary_output = args.summary_output
    if summary_output.exists() and not args.overwrite:
        raise ValueError(
            f"Summary output {summary_output} already exists. Pass --overwrite to recompute."
        )

    filtered_output = args.filtered_representations
    if filtered_output.exists() and not args.overwrite:
        raise ValueError(
            f"Filtered representations {filtered_output} already exists. Pass --overwrite to overwrite."
        )

    cache_path = args.cache_path if args.cache_path else default_cache_path(summary_output)
    cache_data = load_evaluation_cache(cache_path, args.overwrite)
    cache_entries: dict[str, Any] = dict(cache_data.get("entries", {}))
    cache_hits = 0
    cache_misses = 0
    cache_updated = False

    thread_local_client: threading.local = threading.local()

    def get_client() -> OpenRouterChatClient:
        local_client = getattr(thread_local_client, "client", None)
        if local_client is None:
            local_client = OpenRouterChatClient(api_key=api_key)
            thread_local_client.client = local_client
        return local_client

    evaluations: list[Mapping[str, Any]] = []
    case_metrics: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)

    def register_result(
        *,
        decision_id: str,
        case: Mapping[str, Any],
        bias_payload: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> None:
        entry = {
            "decision_id": decision_id,
            "topic": case.get("topic"),
            "style_label": bias_payload.get("style_label"),
            "bias": bias_payload.get("bias"),
            "sensitivity_score": evaluation.get("sensitivity_score"),
            "confidence": evaluation.get("confidence"),
            "rationale": evaluation.get("rationale"),
            "representation_type": "bias",
            "representation": bias_payload.get("representation"),
        }
        evaluations.append(entry)
        case_metrics[decision_id].append(entry)

    def register_error(
        *,
        decision_id: str,
        case: Mapping[str, Any] | None,
        bias_payload: Mapping[str, Any] | None,
        message: str,
    ) -> None:
        evaluations.append(
            {
                "decision_id": decision_id,
                "topic": (case or {}).get("topic"),
                "style_label": (bias_payload or {}).get("style_label"),
                "bias": (bias_payload or {}).get("bias"),
                "error": message,
                "representation_type": "bias",
            }
        )

    work_items: list[dict[str, Any]] = []

    for record in target_records:
        decision_id = str(record.get("decision_id"))
        case = decisions.get(decision_id)
        if not case:
            register_error(
                decision_id=decision_id,
                case=None,
                bias_payload=None,
                message="Decision missing from decision problems file",
            )
            continue

        bias_payloads = record.get("bias_representations", []) or []
        for bias_payload in bias_payloads:
            if not isinstance(bias_payload, Mapping):
                continue
            style_label = str(bias_payload.get("style_label") or "").strip()
            if not style_label:
                register_error(
                    decision_id=decision_id,
                    case=case,
                    bias_payload=bias_payload,
                    message="Missing style_label on representation",
                )
                continue

            representation_text = str(bias_payload.get("representation", "")).strip()
            if not representation_text:
                register_error(
                    decision_id=decision_id,
                    case=case,
                    bias_payload=bias_payload,
                    message="Representation text empty",
                )
                continue

            cache_key = build_cache_key(decision_id, style_label)
            fingerprint = representation_fingerprint(decision_id, style_label, bias_payload)
            cached_eval = get_cached_evaluation(cache_entries, cache_key, fingerprint)
            if cached_eval is not None:
                cache_hits += 1
                register_result(
                    decision_id=decision_id,
                    case=case,
                    bias_payload=bias_payload,
                    evaluation=cached_eval,
                )
                continue

            cache_misses += 1
            work_items.append(
                {
                    "key": cache_key,
                    "decision_id": decision_id,
                    "case": case,
                    "bias_payload": bias_payload,
                    "fingerprint": fingerprint,
                }
            )

    def process_item(item: dict[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any] | None, str | None]:
        try:
            evaluator = get_client()
            evaluation = evaluate_representation(
                client=evaluator,
                model=args.model,
                temperature=args.temperature,
                case=item["case"],
                representation=item["bias_payload"],
            )
            return item, evaluation, None
        except Exception as exc:  # noqa: BLE001
            return item, None, str(exc)

    if work_items:
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in work_items}
            for future in tqdm(
                as_completed(future_to_item),
                total=len(future_to_item),
                desc="Scoring bias sensitivity",
            ):
                item, evaluation, error = future.result()
                decision_id = item["decision_id"]
                bias_payload = item["bias_payload"]
                case = item["case"]
                if evaluation is not None:
                    register_result(
                        decision_id=decision_id,
                        case=case,
                        bias_payload=bias_payload,
                        evaluation=evaluation,
                    )
                    cache_entries[item["key"]] = {
                        "representation_hash": item["fingerprint"],
                        "evaluation": evaluation,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    cache_updated = True
                else:
                    register_error(
                        decision_id=decision_id,
                        case=case,
                        bias_payload=bias_payload,
                        message=error or "Unknown error",
                    )

    cache_data["entries"] = cache_entries
    save_evaluation_cache(cache_path, cache_data)

    case_summaries: list[Mapping[str, Any]] = []
    for decision_id, metrics in case_metrics.items():
        case_info = decisions.get(decision_id, {})
        summary_stats = summarize_case(metrics)
        if summary_stats["max_score"] is None:
            continue
        top_metric = max(metrics, key=lambda m: m["sensitivity_score"])
        case_summaries.append(
            {
                "decision_id": decision_id,
                "topic": case_info.get("topic"),
                "max_score": summary_stats["max_score"],
                "mean_score": summary_stats["mean_score"],
                "median_score": summary_stats["median_score"],
                "mean_confidence": summary_stats["mean_confidence"],
                "top_bias_style": top_metric.get("style_label"),
                "top_bias_score": top_metric.get("sensitivity_score"),
                "top_bias_confidence": top_metric.get("confidence"),
                "evaluated_biases": len(metrics),
            }
        )

    ranked_cases = sorted(
        case_summaries,
        key=lambda item: (
            item["max_score"],
            item["mean_score"] if item["mean_score"] is not None else 0.0,
            item["mean_confidence"] if item["mean_confidence"] is not None else 0.0,
        ),
        reverse=True,
    )

    top_cases = ranked_cases[: max(0, args.top_cases)] if args.top_cases else ranked_cases
    selected_ids = {entry["decision_id"] for entry in top_cases}

    filtered_records = [rec for rec in representation_records if str(rec.get("decision_id")) in selected_ids]

    summary_payload = {
        "decision_problems_path": str(args.decision_problems),
        "representations_path": str(args.representations),
        "model": args.model,
        "temperature": args.temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "top_cases": args.top_cases,
            "start_index": args.start_index,
            "max_cases": args.max_cases,
        },
        "cache": {
            "path": str(cache_path),
            "hits": cache_hits,
            "misses": cache_misses,
            "entries": len(cache_entries),
            "updated": cache_updated,
        },
        "evaluations": evaluations,
        "case_rankings": ranked_cases,
        "selected_decision_ids": sorted(selected_ids),
        "total_cases_evaluated": len(case_metrics),
        "total_cases_selected": len(selected_ids),
    }

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary_payload, indent=2))

    filtered_payload = dict(representation_data)
    filtered_payload["records"] = filtered_records
    filtered_output.parent.mkdir(parents=True, exist_ok=True)
    filtered_output.write_text(json.dumps(filtered_payload, indent=2))


if __name__ == "__main__":
    main()
