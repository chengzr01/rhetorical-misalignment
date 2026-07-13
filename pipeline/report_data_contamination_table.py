#!/usr/bin/env python3
"""Generate a tabular summary of data-contamination evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "experiments/tests"
VARIANT_DIRS = {
    "paraphrased_questions": ROOT / "experiments/tests/data_contamination/paraphrased_questions",
    "paraphrased_options": ROOT / "experiments/tests/data_contamination/paraphrased_options",
}
OUTPUT_MARKDOWN = ROOT / "experiments/analysis/data_contamination_table.md"


@dataclass
class Metrics:
    total: int
    correct: int
    incorrect: int
    no_answer: int
    accuracy: float

    @classmethod
    def from_payload(
        cls,
        metrics_payload: Mapping[str, object] | None,
        results: Iterable[Mapping[str, object]],
    ) -> "Metrics":
        if metrics_payload and metrics_payload.get("total"):
            return cls(
                total=int(metrics_payload.get("total", 0)),
                correct=int(metrics_payload.get("correct", 0)),
                incorrect=int(metrics_payload.get("incorrect", 0)),
                no_answer=int(metrics_payload.get("no_answer", 0)),
                accuracy=float(metrics_payload.get("accuracy", 0.0)),
            )
        results_list = list(results)
        total = len(results_list)
        correct = sum(1 for item in results_list if item.get("correct"))
        incorrect = total - correct
        no_answer = sum(
            1
            for item in results_list
            if not item.get("predicted_answer_idx") and item.get("predicted_answer_idx") != "0"
        )
        accuracy = correct / total if total else 0.0
        return cls(total=total, correct=correct, incorrect=incorrect, no_answer=no_answer, accuracy=accuracy)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "no_answer": self.no_answer,
            "accuracy": self.accuracy,
        }


def load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text())


def infer_model_name(file_name: str, payload: Mapping[str, object]) -> str:
    model = payload.get("model")
    if isinstance(model, dict):
        return model.get("name") or model.get("id") or file_name
    if isinstance(model, str) and model:
        return model
    prefix = "test_usmle_sample_"
    suffix = "_belief.json"
    if file_name.startswith(prefix) and file_name.endswith(suffix):
        return file_name[len(prefix) : -len(suffix)]
    return file_name


def compute_subset_metrics(results: List[Mapping[str, object]], subset_ids: set[str]) -> Metrics:
    filtered = [item for item in results if item.get("id") in subset_ids]
    return Metrics.from_payload(None, filtered)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{100.0 * value:.1f}"  # basis points percentage difference


def fmt_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def build_table() -> list[str]:
    baseline_data: dict[str, dict[str, object]] = {}

    for baseline_path in BASELINE_DIR.glob("test_usmle_sample_*_belief.json"):
        payload = load_json(baseline_path)
        results = payload.get("results") or []
        metrics = Metrics.from_payload(payload.get("metrics"), results)
        baseline_data[baseline_path.name] = {
            "model": infer_model_name(baseline_path.name, payload),
            "results": results,
            "metrics": metrics,
            "subset_metrics": {},
        }

    # gather subset case IDs per variant per file
    subset_ids: dict[tuple[str, str], set[str]] = {}
    variant_metrics: dict[tuple[str, str], Metrics] = {}

    for variant_name, variant_dir in VARIANT_DIRS.items():
        if not variant_dir.exists():
            continue
        for model_dir in variant_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for variant_path in model_dir.glob("test_usmle_sample_*_belief.json"):
                payload = load_json(variant_path)
                results = payload.get("results") or []
                metrics = Metrics.from_payload(payload.get("metrics"), results)
                key = (variant_path.name, variant_name)
                subset_ids[key] = {item.get("id") for item in results if item.get("id")}
                variant_metrics[key] = metrics

    rows: list[str] = []
    header = (
        "| Model | Cases (full) | Accuracy (full) | Cases (subset) | Baseline subset acc | "
        "Paraphrased questions acc | Paraphrased options acc |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    rows.append(header)
    rows.append(separator)

    skip_models = {
        "meta-llama/llama-3.1-405b-instruct",
        "mistralai/mistral-7b-instruct",
        "qwen/qwen-2.5-7b-instruct",
    }

    for file_name in sorted(baseline_data.keys()):
        baseline_entry = baseline_data[file_name]
        model_name = baseline_entry["model"]
        if model_name in skip_models:
            continue
        baseline_metrics: Metrics = baseline_entry["metrics"]
        results = baseline_entry["results"]

        subset_union_ids = set()
        for variant_name in VARIANT_DIRS.keys():
            key = (file_name, variant_name)
            subset_union_ids.update(subset_ids.get(key, set()))

        if subset_union_ids:
            baseline_subset_metrics = compute_subset_metrics(results, subset_union_ids)
        else:
            baseline_subset_metrics = baseline_metrics

        pq_metrics = variant_metrics.get((file_name, "paraphrased_questions"))
        po_metrics = variant_metrics.get((file_name, "paraphrased_options"))

        row = "| {model} | {full_cases} | {full_acc} | {subset_cases} | {subset_acc} | {pq_acc} | {po_acc} |".format(
            model=model_name,
            full_cases=fmt_int(baseline_metrics.total),
            full_acc=fmt_pct(baseline_metrics.accuracy),
            subset_cases=fmt_int(baseline_subset_metrics.total),
            subset_acc=fmt_pct(baseline_subset_metrics.accuracy),
            pq_acc=fmt_pct(pq_metrics.accuracy) if pq_metrics else "n/a",
            po_acc=fmt_pct(po_metrics.accuracy) if po_metrics else "n/a",
        )
        rows.append(row)

    return rows


def main() -> None:
    lines = build_table()
    OUTPUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MARKDOWN.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
