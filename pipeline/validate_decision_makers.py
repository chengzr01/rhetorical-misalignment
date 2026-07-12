#!/usr/bin/env python3
"""Evaluate rational vs behavioral simulated decision-makers on curated cases.

The script loads decision problems, their neutral/bias representations, and
calls LLM-based principal simulators to measure how rational and
behaviorally-biased decision-makers perform. Metrics such as accuracy and
confidence are summarised for quick validation of the simulated agents.

Example:
    python pipeline/validate_decision_makers.py \
        --decision-problems experiments/decision_problems/usmle_rhetorical_decisions.json \
        --representations experiments/decision_problems/usmle_bias_representations.json \
        --output experiments/analysis/decision_maker_validation.json \
        --model deepseek/deepseek-chat-v3.1 \
        --max-cases 20

Environment:
    OPENROUTER_API_KEY – required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.principal import Principal
from interface.client import OpenRouterChatClient


DEFAULT_DECISION_PATH = "experiments/decision_problems/usmle_rhetorical_decisions.json"
DEFAULT_REPRESENTATIONS_PATH = "experiments/decision_problems/usmle_bias_representations.json"
DEFAULT_OUTPUT_PATH = "experiments/analysis/decision_maker_validation.json"
DEFAULT_RATIONAL_PROMPT = Path(__file__).parent.parent / "prompts" / "principal" / "bayesian_belief.yaml"
DEFAULT_BEHAVIORAL_PROMPT = Path(__file__).parent.parent / "prompts" / "principal" / "behavioral_belief.yaml"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def format_options(options: list[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for opt in options:
        opt_id = opt.get("id")
        text = str(opt.get("text", "")).strip()
        lines.append(f"{opt_id}. {text}")
    return "\n".join(lines)


def correct_option_text(case: Mapping[str, Any]) -> str:
    correct_id = case.get("correct_option_id")
    for opt in case.get("options", []):
        if opt.get("id") == correct_id:
            return str(opt.get("text", "")).strip()
    return ""


def build_context(case: Mapping[str, Any], representation: Mapping[str, Any]) -> str:
    hooks = representation.get("reasoning_hooks", [])
    hooks_text = "; ".join(hooks) if hooks else ""
    context_parts = [
        f"Patient Summary: {case.get('patient_profile', '')}",
        f"Scenario: {case.get('concise_context', '')}",
        f"Decision Question: {case.get('decision_question', '')}",
        f"Briefing: {representation.get('representation', '').strip()}",
    ]
    if hooks_text:
        context_parts.append(f"Key Cues: {hooks_text}")
    return "\n".join(context_parts).strip()


def parse_belief(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        belief = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(belief) or math.isinf(belief):
        return None
    return max(0.0, min(1.0, belief))


def run_principal(
    principal: Principal,
    *,
    case: Mapping[str, Any],
    representation: Mapping[str, Any],
    options_str: str,
) -> Mapping[str, Any]:
    context = build_context(case, representation)
    result = principal.act(context=context, options=options_str)
    belief_value = parse_belief(result.get("belief"))
    answer = (result.get("answer") or result.get("decision") or "").strip().upper()
    return {
        "answer": answer,
        "belief": result.get("belief"),
        "belief_value": belief_value,
        "reasoning": result.get("reasoning", ""),
        "raw_response": result.get("raw_response", ""),
    }


def annotate_correctness(run: Mapping[str, Any], correct_option: str) -> None:
    run["correct"] = run.get("answer") == correct_option


def summarize_metrics(runs: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    total = len(runs)
    if total == 0:
        return {"total": 0, "accuracy": None}
    accuracy = sum(1 for r in runs if r.get("correct")) / total
    beliefs = [r["belief_value"] for r in runs if r.get("belief_value") is not None]
    avg_belief = statistics.mean(beliefs) if beliefs else None
    beliefs_correct = [r["belief_value"] for r in runs if r.get("correct") and r.get("belief_value") is not None]
    beliefs_incorrect = [
        r["belief_value"]
        for r in runs
        if not r.get("correct") and r.get("belief_value") is not None
    ]
    return {
        "total": total,
        "accuracy": accuracy,
        "avg_belief": avg_belief,
        "avg_belief_correct": statistics.mean(beliefs_correct) if beliefs_correct else None,
        "avg_belief_incorrect": statistics.mean(beliefs_incorrect) if beliefs_incorrect else None,
        "overconfidence_gap": (avg_belief - accuracy) if (avg_belief is not None) else None,
    }


def prepare_existing(
    output_path: Path,
    overwrite: bool,
    requested_styles: set[str] | None,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    list[Mapping[str, Any]],
    defaultdict[str, list[Mapping[str, Any]]],
]:
    if not output_path.exists() or overwrite:
        return [], set(), [], defaultdict(list)

    existing_data = load_json(output_path)
    records = existing_data.get("records", [])
    processed_ids = {rec.get("decision_id") for rec in records if rec.get("decision_id")}

    rational_runs: list[Mapping[str, Any]] = []
    behavioral_runs: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for rec in records:
        correct_option = str(rec.get("correct_option_id", "")).strip().upper()

        rational = rec.get("rational", {})
        rational_answer = str(rational.get("answer", "")).strip().upper()
        rational_runs.append(
            {
                "answer": rational_answer,
                "belief_value": parse_belief(rational.get("belief")),
                "correct": rational_answer == correct_option if correct_option else False,
            }
        )

        for style_key, info in (rec.get("behavioral") or {}).items():
            style_label = str(info.get("style_label", style_key))
            style_norm = style_label.lower()
            if requested_styles and style_norm not in requested_styles:
                continue
            answer_text = str(info.get("answer", "")).strip().upper()
            behavioral_runs[style_norm].append(
                {
                    "answer": answer_text,
                    "belief_value": parse_belief(info.get("belief")),
                    "correct": answer_text == correct_option if correct_option else False,
                }
            )

    return records, processed_ids, rational_runs, behavioral_runs


def write_output(
    *,
    output_path: Path,
    decision_path: Path,
    representations_path: Path,
    rational_prompt: str,
    behavioral_prompt: str,
    model: str,
    temperature: float,
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    rational_runs: list[Mapping[str, Any]],
    behavioral_runs: Mapping[str, list[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], Mapping[str, Mapping[str, float]]]:
    all_records = existing_records + new_records
    rational_summary = summarize_metrics(rational_runs)
    behavioral_summary = {
        style: summarize_metrics(runs) for style, runs in behavioral_runs.items()
    }

    comparison: dict[str, Mapping[str, float]] = {}
    if rational_summary.get("accuracy") is not None:
        for style, metrics in behavioral_summary.items():
            if metrics.get("accuracy") is None:
                continue
            comparison[style] = {
                "accuracy_delta": metrics["accuracy"] - rational_summary["accuracy"],
                "overconfidence_delta": (
                    (metrics.get("overconfidence_gap") or 0.0)
                    - (rational_summary.get("overconfidence_gap") or 0.0)
                ),
            }

    payload = {
        "decision_problems_path": str(decision_path),
        "representations_path": str(representations_path),
        "model": model,
        "temperature": temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rational_prompt": rational_prompt,
        "behavioral_prompt": behavioral_prompt,
        "rational_summary": rational_summary,
        "behavioral_summary": behavioral_summary,
        "comparison_vs_rational": comparison,
        "records": all_records,
        "total_records": len(all_records),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    return rational_summary, behavioral_summary, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-problems", default=DEFAULT_DECISION_PATH)
    parser.add_argument("--representations", default=DEFAULT_REPRESENTATIONS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rational-prompt", default=str(DEFAULT_RATIONAL_PROMPT))
    parser.add_argument("--behavioral-prompt", default=str(DEFAULT_BEHAVIORAL_PROMPT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--biases", nargs="*", help="Optional subset of bias style labels to evaluate")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8, help="Parallel workers for validation")

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    decision_path = Path(args.decision_problems)
    representations_path = Path(args.representations)
    output_path = Path(args.output)

    decision_data = load_json(decision_path)
    decisions = {str(rec.get("id")): rec for rec in decision_data.get("records", [])}
    if not decisions:
        raise ValueError("No decision problems found in decision file")

    representation_data = load_json(representations_path)
    representation_records = representation_data.get("records", [])
    if not representation_records:
        raise ValueError("No representation records found in representations file")

    if args.biases:
        requested_styles = {b.lower() for b in args.biases}
    else:
        requested_styles = None

    existing_records, processed_ids, existing_rational_runs, existing_behavioral_runs = prepare_existing(
        output_path, args.overwrite, requested_styles
    )
    processed_ids = {str(pid) for pid in processed_ids}

    start = max(args.start_index, 0)
    target_records = representation_records[start:]
    if args.max_cases is not None:
        target_records = target_records[: args.max_cases]

    target_records = [
        rec for rec in target_records if str(rec.get("decision_id")) not in processed_ids
    ]

    rational_runs_all: list[Mapping[str, Any]] = list(existing_rational_runs)
    behavioral_runs_by_style: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for style, runs in existing_behavioral_runs.items():
        behavioral_runs_by_style[style].extend(runs)

    save_interval = max(args.save_interval, 1)
    last_rational_summary: Mapping[str, Any] = summarize_metrics(rational_runs_all)
    last_behavioral_summary: Mapping[str, Mapping[str, Any]] = {
        style: summarize_metrics(runs) for style, runs in behavioral_runs_by_style.items()
    }
    last_comparison: Mapping[str, Mapping[str, float]] = {}

    order_lookup = {
        str(rec.get("decision_id")): idx for idx, rec in enumerate(target_records)
    }

    thread_local_objects: threading.local = threading.local()

    def get_principals() -> tuple[Principal, Principal]:
        bundle = getattr(thread_local_objects, "bundle", None)
        if bundle is None:
            client = OpenRouterChatClient(api_key=api_key)
            bundle = {
                "client": client,
                "rational": Principal(
                    name="rational_principal",
                    client=client,
                    model=args.model,
                    prompt_path=args.rational_prompt,
                    temperature=args.temperature,
                ),
                "behavioral": Principal(
                    name="behavioral_principal",
                    client=client,
                    model=args.model,
                    prompt_path=args.behavioral_prompt,
                    temperature=args.temperature,
                ),
            }
            thread_local_objects.bundle = bundle
        return bundle["rational"], bundle["behavioral"]

    def process_record(record: Mapping[str, Any]) -> tuple[
        str,
        dict[str, Any] | None,
        Mapping[str, Any] | None,
        dict[str, Mapping[str, Any]],
        str | None,
    ]:
        decision_id = str(record.get("decision_id"))
        case = decisions.get(decision_id)
        if not case:
            return decision_id, None, None, {}, "missing decision case"

        try:
            rational_principal, behavioral_principal = get_principals()
            options_str = format_options(case.get("options", []))
            correct_option = case.get("correct_option_id")

            neutral_info = record.get("neutral_representation", {})
            rational_run = run_principal(
                rational_principal,
                case=case,
                representation=neutral_info,
                options_str=options_str,
            )
            annotate_correctness(rational_run, correct_option)

            behavioral_runs: dict[str, Mapping[str, Any]] = {}
            for bias_payload in record.get("bias_representations", []):
                style_label = str(bias_payload.get("style_label", "")).lower()
                if requested_styles and style_label not in requested_styles:
                    continue
                run = run_principal(
                    behavioral_principal,
                    case=case,
                    representation=bias_payload,
                    options_str=options_str,
                )
                annotate_correctness(run, correct_option)
                run_info = {
                    **run,
                    "bias": bias_payload.get("bias"),
                    "style_label": bias_payload.get("style_label"),
                    "target_behavior": bias_payload.get("target_behavior"),
                    "representation": bias_payload.get("representation"),
                    "reasoning_hooks": bias_payload.get("reasoning_hooks"),
                    "bias_alignment_notes": bias_payload.get("bias_alignment_notes"),
                    "raw_response": run.get("raw_response", ""),
                }
                behavioral_runs[style_label] = run_info

            record_payload = {
                "decision_id": decision_id,
                "topic": case.get("topic"),
                "correct_option_id": correct_option,
                "correct_option_text": correct_option_text(case),
                "rational": {
                    **rational_run,
                    "representation": neutral_info.get("representation"),
                    "reasoning_hooks": neutral_info.get("reasoning_hooks"),
                },
                "behavioral": behavioral_runs,
            }

            return decision_id, record_payload, rational_run, behavioral_runs, None
        except Exception as exc:  # noqa: BLE001
            return decision_id, None, None, {}, str(exc)

    successful_records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = {executor.submit(process_record, rec): rec for rec in target_records}
        progress = tqdm(as_completed(futures), total=len(futures), desc="Evaluating decision-makers")
        for future in progress:
            decision_id, record_payload, rational_run, behavioral_runs, error = future.result()
            if record_payload is not None and rational_run is not None:
                successful_records[decision_id] = record_payload
                rational_runs_all.append(rational_run)
                for style_label, run_info in behavioral_runs.items():
                    behavioral_runs_by_style[style_label].append(run_info)
                if len(successful_records) % save_interval == 0:
                    ordered_records = [
                        successful_records[key]
                        for key in sorted(successful_records.keys(), key=lambda k: order_lookup.get(k, float("inf")))
                    ]
                    last_rational_summary, last_behavioral_summary, last_comparison = write_output(
                        output_path=output_path,
                        decision_path=decision_path,
                        representations_path=representations_path,
                        rational_prompt=str(args.rational_prompt),
                        behavioral_prompt=str(args.behavioral_prompt),
                        model=args.model,
                        temperature=args.temperature,
                        existing_records=existing_records,
                        new_records=ordered_records,
                        rational_runs=rational_runs_all,
                        behavioral_runs=behavioral_runs_by_style,
                    )
            else:
                failures.append(f"{decision_id}: {error}")

    ordered_records = [
        successful_records[key]
        for key in sorted(successful_records.keys(), key=lambda k: order_lookup.get(k, float("inf")))
    ]

    last_rational_summary, last_behavioral_summary, last_comparison = write_output(
        output_path=output_path,
        decision_path=decision_path,
        representations_path=representations_path,
        rational_prompt=str(args.rational_prompt),
        behavioral_prompt=str(args.behavioral_prompt),
        model=args.model,
        temperature=args.temperature,
        existing_records=existing_records,
        new_records=ordered_records,
        rational_runs=rational_runs_all,
        behavioral_runs=behavioral_runs_by_style,
    )

    print("Rational accuracy:", last_rational_summary.get("accuracy"))
    for style, metrics in last_behavioral_summary.items():
        print(f"Behavioral ({style}) accuracy: {metrics.get('accuracy')}")
    if failures:
        print("Skipped cases:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
