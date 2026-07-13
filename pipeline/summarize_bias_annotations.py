#!/usr/bin/env python3
"""Aggregate metrics over bias annotation files for simulated principals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

DEFAULT_INPUT_DIR = Path("experiments/analysis/bias_annotations/usmle_sample/baseline")
DEFAULT_OUTPUT_PATH = Path("experiments/analysis/bias_annotations_summary.json")


@dataclass
class AnnotationRecord:
    model: str
    scenario: str
    path: Path
    total_cases: int
    annotated_cases: int
    bias_counts: Counter
    bias_prevalence: Dict[str, float]
    total_bias_labels: int
    cooccurrence: Counter


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if not stem.startswith("principal_") or not stem.endswith("_bias_annotations"):
        raise ValueError(f"Unexpected annotation filename: {path}")
    core = stem[len("principal_") : -len("_bias_annotations")]
    if "_" not in core:
        raise ValueError(f"Could not extract model/scenario from {path}")
    model, scenario = core.split("_", 1)
    return model, scenario


def load_annotation(path: Path) -> AnnotationRecord:
    data = json.loads(path.read_text())
    model, scenario = parse_filename(path)

    summary: Mapping[str, Any] = data.get("summary", {})
    total_cases = int(summary.get("total_cases") or 0)
    annotations: List[Mapping[str, Any]] = data.get("annotations", [])
    annotated_cases = len(annotations)
    if not total_cases:
        total_cases = annotated_cases

    bias_counts = Counter(summary.get("bias_counts", {}))
    bias_prevalence = {str(bias): float(val) for bias, val in summary.get("bias_prevalence", {}).items()}

    cooccurrence: Counter = Counter()
    total_bias_labels = 0
    for entry in annotations:
        labels = [str(label).strip() for label in entry.get("bias_labels", []) if label]
        if not labels:
            continue
        unique_labels = sorted(set(labels))
        total_bias_labels += len(unique_labels)
        for first, second in combinations(unique_labels, 2):
            key = (first, second)
            cooccurrence[key] += 1

    return AnnotationRecord(
        model=model,
        scenario=scenario,
        path=path,
        total_cases=total_cases,
        annotated_cases=annotated_cases,
        bias_counts=bias_counts,
        bias_prevalence=bias_prevalence,
        total_bias_labels=total_bias_labels,
        cooccurrence=cooccurrence,
    )


def summarise(records: Iterable[AnnotationRecord]) -> Dict[str, Any]:
    scenario_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total_cases": 0,
        "annotated_cases": 0,
        "bias_counts": Counter(),
        "total_bias_labels": 0,
        "cooccurrence": Counter(),
        "models": {},
    })

    global_counts = Counter()
    global_total_cases = 0
    global_total_bias_labels = 0
    global_cooccurrence = Counter()

    for record in records:
        scenario_bucket = scenario_data[record.scenario]
        scenario_bucket["total_cases"] += record.total_cases
        scenario_bucket["annotated_cases"] += record.annotated_cases
        scenario_bucket["bias_counts"].update(record.bias_counts)
        scenario_bucket["total_bias_labels"] += record.total_bias_labels
        scenario_bucket["cooccurrence"].update(record.cooccurrence)

        scenario_bucket["models"][record.model] = {
            "path": str(record.path),
            "total_cases": record.total_cases,
            "annotated_cases": record.annotated_cases,
            "bias_counts": dict(record.bias_counts),
            "bias_prevalence": record.bias_prevalence,
            "avg_biases_per_case": round(record.total_bias_labels / record.annotated_cases, 4) if record.annotated_cases else 0.0,
            "top_biases": sorted(
                (
                    {"bias": bias, "count": count, "prevalence": record.bias_prevalence.get(bias, count / record.total_cases if record.total_cases else 0.0)}
                    for bias, count in record.bias_counts.most_common()
                ),
                key=lambda item: (-item["count"], item["bias"]),
            )[:5],
        }

        global_counts.update(record.bias_counts)
        global_total_cases += record.total_cases
        global_total_bias_labels += record.total_bias_labels
        global_cooccurrence.update(record.cooccurrence)

    scenarios_summary: Dict[str, Any] = {}
    for scenario, payload in scenario_data.items():
        total_cases = payload["total_cases"] or 1
        annotated_cases = payload["annotated_cases"] or 1
        bias_counts: Counter = payload["bias_counts"]
        cooccurrence: Counter = payload["cooccurrence"]
        total_bias_labels = payload["total_bias_labels"]

        prevalence = {bias: count / total_cases for bias, count in bias_counts.items()}
        avg_biases_per_case = total_bias_labels / annotated_cases

        pair_rates = {
            "__".join(pair): count / annotated_cases for pair, count in cooccurrence.items()
        }

        scenarios_summary[scenario] = {
            "total_cases": total_cases,
            "annotated_cases": annotated_cases,
            "bias_counts": dict(sorted(bias_counts.items(), key=lambda item: (-item[1], item[0]))),
            "bias_prevalence": {bias: round(val, 6) for bias, val in sorted(prevalence.items(), key=lambda item: (-item[1], item[0]))},
            "avg_biases_per_case": round(avg_biases_per_case, 6),
            "top_cooccurrences": [
                {
                    "bias_pair": list(pair),
                    "count": count,
                    "rate": round(count / annotated_cases, 6),
                }
                for pair, count in cooccurrence.most_common(5)
            ],
            "cooccurrence_rates": {pair_key: round(rate, 6) for pair_key, rate in sorted(pair_rates.items(), key=lambda item: (-item[1], item[0]))},
            "models": payload["models"],
        }

    overall_prevalence = {bias: count / global_total_cases for bias, count in global_counts.items()} if global_total_cases else {}

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(DEFAULT_INPUT_DIR.resolve()),
        "scenario_count": len(scenarios_summary),
        "total_cases": global_total_cases,
        "bias_counts": dict(sorted(global_counts.items(), key=lambda item: (-item[1], item[0]))),
        "bias_prevalence": {bias: round(val, 6) for bias, val in sorted(overall_prevalence.items(), key=lambda item: (-item[1], item[0]))},
        "avg_biases_per_case": round(global_total_bias_labels / global_total_cases, 6) if global_total_cases else 0.0,
        "top_cooccurrences": [
            {
                "bias_pair": list(pair),
                "count": count,
                "rate": round(count / global_total_cases, 6) if global_total_cases else 0.0,
            }
            for pair, count in global_cooccurrence.most_common(10)
        ],
        "scenarios": scenarios_summary,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise bias annotation JSON files into aggregate metrics.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing *_bias_annotations.json files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the aggregated summary JSON")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("principal_*_bias_annotations.json"))
    if not files:
        raise SystemExit(f"No annotation files found under {input_dir}")

    records = [load_annotation(path) for path in files]
    summary = summarise(records)
    summary["input_dir"] = str(input_dir.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
