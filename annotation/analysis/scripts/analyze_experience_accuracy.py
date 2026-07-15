#!/usr/bin/env python3
"""Compute decision accuracy by annotator experience buckets."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ExperienceBucket:
    label: str
    lower: Optional[float]
    upper: Optional[float]

    def contains(self, value: float) -> bool:
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value >= self.upper:
            return False
        return True


EXPERIENCE_BUCKETS: List[ExperienceBucket] = [
    ExperienceBucket(label="<2 years", lower=None, upper=2.0),
    ExperienceBucket(label="2–5 years", lower=2.0, upper=5.0),
    ExperienceBucket(label="5–10 years", lower=5.0, upper=10.0),
    ExperienceBucket(label="10–20 years", lower=10.0, upper=20.0),
    ExperienceBucket(label="20+ years", lower=20.0, upper=None),
]

STEP_KEYS = ["step1", "step2", "step3"]


@dataclass
class BucketMetrics:
    total: int = 0
    correct_counts: Dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in STEP_KEYS}
    )
    step1_correct_total: int = 0
    harmful_changes: int = 0
    helpful_changes: int = 0
    harmful_reasoning_present: int = 0
    helpful_reasoning_present: int = 0
    harmful_reasoning_counter: Counter = field(default_factory=Counter)
    helpful_reasoning_counter: Counter = field(default_factory=Counter)


KEYWORD_CATEGORIES = {
    "AI authority / trust": [
        r"\\bai\\b",
        r"model",
        r"thought.*knew",
        r"knew more",
        r"trust(ed|ing)?",
        r"confident.*ai",
        r"ai.*confident",
        r"ai.*(right|correct)",
        r"ai.*detailed",
        r"more detailed",
        r"better explanation",
    ],
    "Misleading detail": [
        r"mislead",
        r"confus",
        r"distract",
        r"threw me off",
        r"led me",
        r"disruption",
        r"highlighted",
    ],
    "Self-doubt": [
        r"doubt",
        r"unsure",
        r"second.?guess",
        r"not sure",
        r"wasn'?t sure",
        r"changed my mind",
        r"persuad",
        r"swayed",
        r"gut feeling",
        r"should have stuck",
        r"should.*trust",
    ],
    "Evidence / reasoning cited": [
        r"evidence",
        r"guideline",
        r"study",
        r"literature",
        r"data",
        r"research",
        r"based on",
        r"according to",
        r"cited",
        r"recommend",
    ],
    "Acknowledged AI error": [
        r"wrong",
        r"incorrect",
        r"error",
        r"mistake",
        r"ai.*wrong",
        r"shouldn'?t have",
        r"regret",
    ],
    "No / vague reasoning": [
        r"^$",
        r"n/?a",
        r"not sure why",
        r"unsure why",
        r"unclear",
    ],
}

_NO_REASON_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in KEYWORD_CATEGORIES["No / vague reasoning"]
]
_CATEGORY_PATTERNS = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in KEYWORD_CATEGORIES.items()
    if cat != "No / vague reasoning"
}
_OTHER_CATEGORY = "Other / unclassified"


def iter_annotation_paths(results_dir: str) -> Iterable[str]:
    """Yield all JSON files inside the results directory."""
    for root, _, files in os.walk(results_dir):
        for filename in files:
            if filename.endswith(".json"):
                yield os.path.join(root, filename)


def load_annotations(results_dir: str) -> List[Dict]:
    annotations: List[Dict] = []
    for path in iter_annotation_paths(results_dir):
        try:
            with open(path, "r", encoding="utf-8") as f:
                annotations.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: skipping {path}: {exc}")
    return annotations


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_years_of_practice(raw_value: Optional[str]) -> Optional[float]:
    """Extract a numeric years-of-practice value from free-form text."""
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None
    if text.lower() in {"na", "n/a", "none"}:
        return None
    match = _NUMERIC_RE.search(text.replace(",", ""))
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    if "+" in text and value.is_integer():
        # Treat "20+" style answers as the stated integer.
        return float(int(value))
    return value


def bucket_years_of_practice(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    for bucket in EXPERIENCE_BUCKETS:
        if bucket.contains(value):
            return bucket.label
    return None


def categorise_reasoning(text: str) -> List[str]:
    if text is None:
        return ["No / vague reasoning"]
    stripped = text.strip()
    if not stripped:
        return ["No / vague reasoning"]
    lowered = stripped.lower()
    if any(pattern.search(lowered) for pattern in _NO_REASON_PATTERNS):
        return ["No / vague reasoning"]

    matches: List[str] = []
    for category, patterns in _CATEGORY_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            matches.append(category)

    if not matches:
        return [_OTHER_CATEGORY]
    return matches


def compute_metrics_by_experience(annotations: Iterable[Dict]):
    metrics = {bucket.label: BucketMetrics() for bucket in EXPERIENCE_BUCKETS}
    missing_experience = 0

    for annotation in annotations:
        years_str = annotation.get("demographics", {}).get("years_of_practice")
        years_value = parse_years_of_practice(years_str)
        bucket_label = bucket_years_of_practice(years_value)
        if bucket_label is None:
            missing_experience += 1
            continue

        bucket_metrics = metrics[bucket_label]
        bucket_metrics.total += 1

        step1 = annotation.get("step1", {}) or {}
        step2 = annotation.get("step2", {}) or {}
        step3 = annotation.get("step3", {}) or {}

        for step_key, step in zip(STEP_KEYS, [step1, step2, step3]):
            if step.get("is_correct"):
                bucket_metrics.correct_counts[step_key] += 1

        step1_correct = bool(step1.get("is_correct"))
        step2_correct = bool(step2.get("is_correct"))

        if step1_correct:
            bucket_metrics.step1_correct_total += 1

        reasoning = annotation.get("reasoning", "")
        reasoning_categories = None

        def _ensure_categories():
            nonlocal reasoning_categories
            if reasoning_categories is None:
                reasoning_categories = categorise_reasoning(reasoning)
            return reasoning_categories

        if step1_correct and not step2_correct:
            bucket_metrics.harmful_changes += 1
            if reasoning.strip():
                bucket_metrics.harmful_reasoning_present += 1
            for category in _ensure_categories():
                bucket_metrics.harmful_reasoning_counter[category] += 1
        elif (not step1_correct) and step2_correct:
            bucket_metrics.helpful_changes += 1
            if reasoning.strip():
                bucket_metrics.helpful_reasoning_present += 1
            for category in _ensure_categories():
                bucket_metrics.helpful_reasoning_counter[category] += 1

    return metrics, missing_experience


def format_percentage(correct: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{(correct / total) * 100:6.1f}%"


def print_accuracy_table(metrics) -> None:
    header = (
        f"{'Reported experience':<20}"
        f"{'Decisions':>12}"
        f"{'Step 1':>10}"
        f"{'Step 2':>10}"
        f"{'Step 3':>10}"
    )
    print(header)
    print("-" * len(header))

    for bucket in EXPERIENCE_BUCKETS:
        label = bucket.label
        bucket_metrics = metrics.get(label)
        if bucket_metrics is None:
            total = 0
            step1 = step2 = step3 = "N/A"
        else:
            total = bucket_metrics.total
            step1 = format_percentage(bucket_metrics.correct_counts["step1"], total)
            step2 = format_percentage(bucket_metrics.correct_counts["step2"], total)
            step3 = format_percentage(bucket_metrics.correct_counts["step3"], total)
        print(f"{label:<20}{total:12d}{step1:>10}{step2:>10}{step3:>10}")


def print_harmful_change_table(metrics) -> None:
    print("\nHarmful decision changes (Step 1 correct → Step 2 incorrect):")
    header = (
        f"{'Reported experience':<20}"
        f"{'Eligible':>10}"
        f"{'Harmful':>10}"
        f"{'Rate':>10}"
        f"{'Reasoned':>12}"
    )
    print(header)
    print("-" * len(header))

    for bucket in EXPERIENCE_BUCKETS:
        label = bucket.label
        bucket_metrics = metrics.get(label)
        if bucket_metrics is None:
            eligible = harmful = reasoned = 0
            rate = "N/A"
        else:
            eligible = bucket_metrics.step1_correct_total
            harmful = bucket_metrics.harmful_changes
            reasoned = bucket_metrics.harmful_reasoning_present
            rate = format_percentage(harmful, eligible)
        print(f"{label:<20}{eligible:10d}{harmful:10d}{rate:>10}{reasoned:12d}")


def print_reasoning_breakdown(metrics) -> None:
    print("\nReasoning themes among harmful changes (multi-label counts):")
    for bucket in EXPERIENCE_BUCKETS:
        label = bucket.label
        bucket_metrics = metrics.get(label)
        if bucket_metrics is None or bucket_metrics.harmful_changes == 0:
            print(f"  {label}: no harmful cases")
            continue

        total_harmful = bucket_metrics.harmful_changes
        counter = bucket_metrics.harmful_reasoning_counter
        print(f"  {label} (n={total_harmful} harmful cases):")
        most_common = counter.most_common()
        if not most_common:
            print("    No reasoning provided")
            continue

        for category, count in most_common:
            pct = (count / total_harmful) * 100
            print(f"    - {category}: {count} ({pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to the directory containing annotation JSON files.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, "..", "..", "results", "usmle_sample")
    results_dir = args.results_dir or default_dir

    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory not found: {results_dir}")

    annotations = load_annotations(results_dir)
    if not annotations:
        raise SystemExit("No annotations found. Check the results directory path.")

    metrics, missing = compute_metrics_by_experience(annotations)

    print(f"Loaded {len(annotations)} annotations from {results_dir}")
    if missing:
        print(f"Skipping {missing} annotations without usable experience data")
    print()
    print_accuracy_table(metrics)
    print_harmful_change_table(metrics)
    print_reasoning_breakdown(metrics)


if __name__ == "__main__":
    main()
