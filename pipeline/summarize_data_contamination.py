#!/usr/bin/env python3
"""Summarize USMLE data contamination evaluations across paraphrasing variants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = ROOT / "experiments/tests"
VARIANT_DIRS = {
    "paraphrased_questions": ROOT / "experiments/tests/data_contamination/paraphrased_questions",
    "paraphrased_options": ROOT / "experiments/tests/data_contamination/paraphrased_options",
}
OUTPUT_PATH = ROOT / "experiments/analysis/data_contamination_summary.json"


@dataclass
class Metrics:
    total: int
    correct: int
    incorrect: int
    no_answer: int
    accuracy: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], fallback_results: List[Mapping[str, object]] | None = None) -> "Metrics":
        if payload:
            return cls(
                total=int(payload.get("total", 0)),
                correct=int(payload.get("correct", 0)),
                incorrect=int(payload.get("incorrect", 0)),
                no_answer=int(payload.get("no_answer", 0)),
                accuracy=float(payload.get("accuracy", 0.0)) if payload.get("total", 0) else 0.0,
            )
        results = fallback_results or []
        total = len(results)
        correct = sum(1 for record in results if record.get("correct"))
        no_answer = sum(1 for record in results if not record.get("predicted_answer_idx"))
        incorrect = total - correct
        accuracy = correct / total if total else 0.0
        return cls(total=total, correct=correct, incorrect=incorrect, no_answer=no_answer, accuracy=accuracy)

    def to_dict(self) -> Dict[str, object]:
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


def collect() -> Dict[str, object]:
    summary: Dict[str, Dict[str, object]] = {}

    for original_path in ORIGINAL_DIR.glob("test_usmle_sample_*_belief.json"):
        payload = load_json(original_path)
        metrics = Metrics.from_payload(payload.get("metrics"), payload.get("results"))
        model_name = infer_model_name(original_path.name, payload)
        summary[original_path.name] = {
            "model": model_name,
            "original": metrics.to_dict(),
            "variants": {},
        }

    for variant, variant_dir in VARIANT_DIRS.items():
        if not variant_dir.exists():
            continue
        for model_dir in variant_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for variant_path in model_dir.glob("test_usmle_sample_*_belief.json"):
                payload = load_json(variant_path)
                metrics = Metrics.from_payload(payload.get("metrics"), payload.get("results"))
                key = variant_path.name
                entry = summary.setdefault(
                    key,
                    {
                        "model": infer_model_name(variant_path.name, payload),
                        "original": None,
                        "variants": {},
                    },
                )
                if not entry.get("model"):
                    entry["model"] = infer_model_name(variant_path.name, payload)
                entry["variants"][variant] = metrics.to_dict()

    records = []
    for file_name, payload in summary.items():
        base = {
            "file_name": file_name,
            "model": payload.get("model"),
            "original": payload.get("original"),
        }
        variants = payload.get("variants", {})
        for variant_name, metrics in variants.items():
            original_metrics = payload.get("original") or {}
            delta = None
            if original_metrics.get("accuracy") is not None:
                delta = metrics.get("accuracy") - original_metrics.get("accuracy", 0.0)
            records.append(
                {
                    **base,
                    "variant": variant_name,
                    "variant_metrics": metrics,
                    "accuracy_delta": delta,
                }
            )
    return {
        "generated_by": "pipeline/summarize_data_contamination.py",
        "records": records,
    }


def main() -> None:
    data = collect()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
