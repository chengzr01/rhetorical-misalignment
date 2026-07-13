#!/usr/bin/env python3
"""Evaluate rational vs behavioral simulated decision-makers with open-ended outputs.

This script loads decision problems alongside neutral/bias-aligned representations
and queries LLM-based principals to generate free-text recommendations. Rather
than scoring multiple-choice accuracy, it aggregates confidence statistics and
stores the full narrative responses for downstream analysis.

Example:
    python pipeline/validate_decision_makers.py \
        --decision-problems experiments/decision_problems/usmle_rhetorical_decisions.json \
        --representations experiments/decision_problems/usmle_bias_representations_filtered.json \
        --output experiments/analysis/decision_maker_validation_openended.json \
        --rational-prompt prompts/principal/bayesian_belief_openended.yaml \
        --behavioral-prompt prompts/principal/behavioral_belief_openended.yaml

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
DEFAULT_REPRESENTATIONS_PATH = "experiments/decision_problems/usmle_bias_representations_filtered.json"
DEFAULT_OUTPUT_PATH = "experiments/analysis/decision_maker_validation_openended.json"
DEFAULT_RATIONAL_PROMPT = Path(__file__).parent.parent / "prompts" / "principal" / "bayesian_belief_openended.yaml"
DEFAULT_BEHAVIORAL_PROMPT = Path(__file__).parent.parent / "prompts" / "principal" / "behavioral_belief_openended.yaml"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def build_context(case: Mapping[str, Any], representation: Mapping[str, Any]) -> str:
    hooks = representation.get("reasoning_hooks", [])
    hooks_text = "; ".join(str(h).strip()) if hooks else ""
    context_parts = [
        f"Patient Summary: {case.get('patient_profile', '')}",
        f"Scenario: {case.get('concise_context', '')}",
        f"Decision Question: {case.get('decision_question', '')}",
        f"Briefing: {representation.get('representation', '').strip()}",
    ]
    if hooks_text:
        context_parts.append(f"Key Cues Highlighted: {hooks_text}")
    context_parts.append(
        "Response Instruction: Provide a free-text recommendation. Do not reference lettered answer options."
    )
    return "\n".join(context_parts).strip()


def run_principal(
    principal: Principal,
    *,
    case: Mapping[str, Any],
    representation: Mapping[str, Any],
) -> Mapping[str, Any]:
    context = build_context(case, representation)
    result = principal.act(context=context)
    belief_value = parse_belief(result.get("belief"))
    recommendation = (
        result.get("recommendation")
        or result.get("decision")
        or result.get("answer")
        or ""
    ).strip()
    return {
        "recommendation": recommendation,
        "belief": result.get("belief"),
        "belief_value": belief_value,
        "reasoning": result.get("reasoning", ""),
        "raw_response": result.get("raw_response", ""),
    }


def summarize_beliefs(runs: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    total = len(runs)
    values = [r["belief_value"] for r in runs if r.get("belief_value") is not None]
    if not values:
        return {"total": total, "avg_belief": None, "median_belief": None, "stdev_belief": None}
    avg = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    high_conf_rate = sum(1 for v in values if v >= 0.8) / len(values)
    return {
        "total": total,
        "avg_belief": avg,
        "median_belief": median,
        "stdev_belief": stdev,
        "high_confidence_rate": high_conf_rate,
    }


def prepare_existing(
    cache_path: Path,
    overwrite: bool,
    requested_styles: set[str] | None,
) -> tuple[
    list[dict[str, Any]],
    set[str],
    defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]],
]:
    if not cache_path.exists() or overwrite:
        return [], set(), defaultdict(lambda: defaultdict(list))

    existing_data = load_json(cache_path)
    records = existing_data.get("records", [])
    processed_ids = {rec.get("decision_id") for rec in records if rec.get("decision_id")}

    principal_runs: defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in records:
        principals_blob = rec.get("principals")
        if not isinstance(principals_blob, Mapping):
            continue
        for principal_name, styles in principals_blob.items():
            if not isinstance(styles, Mapping):
                continue
            for style_label, info in styles.items():
                if not isinstance(info, Mapping):
                    continue
                style_norm = style_label.lower()
                if requested_styles and style_norm not in requested_styles and style_norm != "neutral_briefing":
                    continue
                principal_runs[principal_name][style_norm].append(
                    {
                        "belief_value": parse_belief(info.get("belief")),
                    }
                )

    return records, processed_ids, principal_runs


def write_output(
    *,
    output_path: Path,
    cache_path: Path | None,
    decision_path: Path,
    representations_path: Path,
    rational_prompt: str,
    behavioral_prompt: str,
    model: str,
    temperature: float,
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    principal_runs: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
) -> tuple[Mapping[str, Mapping[str, Mapping[str, Any]]], Mapping[str, Mapping[str, float]]]:
    all_records = existing_records + new_records

    principal_summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for principal_name, style_runs in principal_runs.items():
        style_summary = {style: summarize_beliefs(runs) for style, runs in style_runs.items()}
        all_runs = [run for runs in style_runs.values() for run in runs]
        principal_summaries[principal_name] = {
            "overall": summarize_beliefs(all_runs),
            "by_style": style_summary,
        }

    comparison: dict[str, Mapping[str, float]] = {}
    rational_styles = principal_summaries.get("rational", {}).get("by_style", {})
    behavioral_styles = principal_summaries.get("behavioral", {}).get("by_style", {})
    for style in sorted(set(rational_styles) | set(behavioral_styles)):
        behavioral_metrics = behavioral_styles.get(style)
        if not behavioral_metrics:
            continue
        rational_metrics = rational_styles.get(style)
        if not rational_metrics:
            continue
        b_avg = behavioral_metrics.get("avg_belief")
        r_avg = rational_metrics.get("avg_belief")
        if b_avg is None or r_avg is None:
            continue
        comparison[style] = {
            "avg_belief_delta": b_avg - r_avg,
        }

    payload = {
        "decision_problems_path": str(decision_path),
        "representations_path": str(representations_path),
        "model": model,
        "temperature": temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rational_prompt": rational_prompt,
        "behavioral_prompt": behavioral_prompt,
        "principal_summaries": principal_summaries,
        "comparison_vs_rational": comparison,
        "records": all_records,
        "total_records": len(all_records),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    if cache_path is not None and cache_path != output_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2))

    return principal_summaries, comparison


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
    parser.add_argument("--threads", type=int, default=16, help="Parallel workers for validation")
    parser.add_argument(
        "--cache-path",
        help="Optional cache location for previously evaluated principal runs",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    decision_path = Path(args.decision_problems)
    representations_path = Path(args.representations)
    output_path = Path(args.output)
    cache_path = Path(args.cache_path) if args.cache_path else output_path

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

    existing_records, processed_ids, existing_principal_runs = prepare_existing(
        cache_path, args.overwrite, requested_styles
    )
    processed_ids = {str(pid) for pid in processed_ids}

    start = max(args.start_index, 0)
    target_records = representation_records[start:]
    if args.max_cases is not None:
        target_records = target_records[: args.max_cases]

    if processed_ids:
        target_records = [
            rec for rec in target_records if str(rec.get("decision_id")) not in processed_ids
        ]

    principal_runs: defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for principal_name, style_runs in existing_principal_runs.items():
        for style_label, runs in style_runs.items():
            principal_runs[principal_name][style_label].extend(runs)

    save_interval = max(args.save_interval, 1)
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
        dict[str, dict[str, list[Mapping[str, Any]]]] | None,
        str | None,
    ]:
        decision_id = str(record.get("decision_id"))
        case = decisions.get(decision_id)
        if not case:
            return decision_id, None, None, "missing decision case"

        try:
            rational_principal, behavioral_principal = get_principals()
            neutral_info = record.get("neutral_representation", {}) or {}
            representation_payloads: list[tuple[str, Mapping[str, Any], bool]] = []
            neutral_style = str(neutral_info.get("style_label") or "neutral_briefing")
            representation_payloads.append((neutral_style, neutral_info, True))

            for bias_payload in record.get("bias_representations", []) or []:
                if not isinstance(bias_payload, Mapping):
                    continue
                style_label = str(bias_payload.get("style_label") or "").strip()
                if not style_label:
                    continue
                style_norm = style_label.lower()
                if requested_styles and style_norm not in requested_styles:
                    continue
                representation_payloads.append((style_label, bias_payload, False))

            principal_runs_local: dict[str, dict[str, Mapping[str, Any]]] = {
                "rational": {},
                "behavioral": {},
            }
            principal_metrics_local: dict[str, dict[str, list[Mapping[str, Any]]]] = {
                "rational": defaultdict(list),
                "behavioral": defaultdict(list),
            }

            principals_bundle = {
                "rational": rational_principal,
                "behavioral": behavioral_principal,
            }

            for style_label_original, payload, is_neutral in representation_payloads:
                style_norm = style_label_original.lower()
                for principal_name, principal_obj in principals_bundle.items():
                    run = run_principal(
                        principal_obj,
                        case=case,
                        representation=payload,
                    )

                    run_info = {
                        **run,
                        "style_label": style_label_original,
                        "representation": payload.get("representation"),
                        "reasoning_hooks": payload.get("reasoning_hooks"),
                        "bias": payload.get("bias"),
                        "target_behavior": payload.get("target_behavior"),
                        "bias_alignment_notes": payload.get("bias_alignment_notes"),
                        "representation_type": "neutral" if is_neutral else "bias",
                    }
                    principal_runs_local[principal_name][style_norm] = run_info
                    principal_metrics_local[principal_name][style_norm].append(
                        {"belief_value": run.get("belief_value")}
                    )

            record_payload = {
                "decision_id": decision_id,
                "topic": case.get("topic"),
                "source_question_id": case.get("source_question_id"),
                "neutral_representation": neutral_info,
                "bias_representations": record.get("bias_representations", []),
                "principals": principal_runs_local,
            }

            return decision_id, record_payload, principal_metrics_local, None
        except Exception as exc:  # noqa: BLE001
            return decision_id, None, None, str(exc)

    successful_records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    if target_records:
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
            futures = {executor.submit(process_record, rec): rec for rec in target_records}
            progress = tqdm(as_completed(futures), total=len(futures), desc="Evaluating decision-makers")
            for future in progress:
                decision_id, record_payload, principal_metrics_local, error = future.result()
                if record_payload is not None:
                    successful_records[decision_id] = record_payload
                    if principal_metrics_local:
                        for principal_name, styles in principal_metrics_local.items():
                            for style_label, metrics in styles.items():
                                principal_runs[principal_name][style_label].extend(metrics)
                    if len(successful_records) % save_interval == 0:
                        ordered_records = [
                            successful_records[key]
                            for key in sorted(
                                successful_records.keys(),
                                key=lambda k: order_lookup.get(k, float("inf")),
                            )
                        ]
                        write_output(
                            output_path=output_path,
                            cache_path=cache_path,
                            decision_path=decision_path,
                            representations_path=representations_path,
                            rational_prompt=str(args.rational_prompt),
                            behavioral_prompt=str(args.behavioral_prompt),
                            model=args.model,
                            temperature=args.temperature,
                            existing_records=existing_records,
                            new_records=ordered_records,
                            principal_runs=principal_runs,
                        )
                else:
                    failures.append(f"{decision_id}: {error}")

    ordered_records = [
        successful_records[key]
        for key in sorted(successful_records.keys(), key=lambda k: order_lookup.get(k, float("inf")))
    ]

    summaries, comparison = write_output(
        output_path=output_path,
        cache_path=cache_path,
        decision_path=decision_path,
        representations_path=representations_path,
        rational_prompt=str(args.rational_prompt),
        behavioral_prompt=str(args.behavioral_prompt),
        model=args.model,
        temperature=args.temperature,
        existing_records=existing_records,
        new_records=ordered_records,
        principal_runs=principal_runs,
    )

    for principal_name, summary in summaries.items():
        overall = summary.get("overall", {})
        print(f"{principal_name.capitalize()} (overall) avg belief: {overall.get('avg_belief')}")
        for style, metrics in sorted(summary.get("by_style", {}).items()):
            print(f"{principal_name.capitalize()} [{style}] avg belief: {metrics.get('avg_belief')}")

    if comparison:
        print("Average belief deltas (behavioral - rational):")
        for style, metrics in sorted(comparison.items()):
            print(f"  {style}: {metrics.get('avg_belief_delta')}")

    if failures:
        print("Skipped cases:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
