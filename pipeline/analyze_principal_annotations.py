#!/usr/bin/env python3
"""Summarise principal-response annotations to assess rhetorical influence.

Given the open-ended validation output and the annotation file produced by
`pipeline/annotate_principal_responses.py`, this script aggregates alignment
scores, counts rhetorical cue detections, and compares rational vs behavioural
personae on a per-style and per-case basis. The resulting JSON payload is
intended to support downstream analysis or lightweight reporting.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_VALIDATION_PATH = Path(
    "experiments/analysis/decision_maker_validation_openended.json"
)
DEFAULT_ANNOTATION_PATH = Path("experiments/analysis/principal_response_annotations.json")
DEFAULT_OUTPUT_PATH = Path("experiments/analysis/principal_annotation_summary.json")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarise_annotations(records: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    summary: dict[str, Any] = {
        "global": {
            "total_records": len(records),
            "alignment_counts": defaultdict(int),
            "mentions_classic_case": 0,
        },
        "by_principal": defaultdict(
            lambda: {
                "total": 0,
                "alignment_counts": defaultdict(int),
                "avg_alignment_score": None,
                "avg_confidence": None,
                "mentions_classic_case": 0,
                "by_style": defaultdict(
                    lambda: {
                        "total": 0,
                        "alignment_counts": defaultdict(int),
                        "avg_alignment_score": None,
                        "avg_confidence": None,
                        "mentions_classic_case": 0,
                    }
                ),
            }
        ),
        "style_differences": defaultdict(
            lambda: {
                "pair_counts": 0,
                "alignment_delta_mean": None,
                "alignment_delta_values": [],
                "records": [],
            }
        ),
    }

    # temporary storage for mean calculations
    overall_scores: list[float] = []
    overall_confidences: list[float] = []
    per_principal_scores: dict[str, list[float]] = defaultdict(list)
    per_principal_conf: dict[str, list[float]] = defaultdict(list)
    per_style_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_style_conf: dict[tuple[str, str], list[float]] = defaultdict(list)

    for item in records:
        annotation = item.get("annotation") or {}
        alignment = annotation.get("alignment_score")
        confidence = annotation.get("confidence")
        category = annotation.get("rhetorical_alignment", "unknown")
        principal = str(item.get("principal") or "unknown")
        style = str(item.get("style_label") or "unknown")
        principal_summary = summary["by_principal"][principal]
        style_summary = principal_summary["by_style"][style]

        # update counts
        summary["global"]["alignment_counts"][category] += 1
        principal_summary["alignment_counts"][category] += 1
        style_summary["alignment_counts"][category] += 1
        summary["global"]["total_records"] += 0  # placeholder to emphasise key
        principal_summary["total"] += 1
        style_summary["total"] += 1

        if annotation.get("mentions_classic_case"):
            summary["global"]["mentions_classic_case"] += 1
            principal_summary["mentions_classic_case"] += 1
            style_summary["mentions_classic_case"] += 1

        if alignment is not None:
            alignment_value = float(alignment)
            overall_scores.append(alignment_value)
            per_principal_scores[principal].append(alignment_value)
            per_style_scores[(principal, style)].append(alignment_value)
        if confidence is not None:
            confidence_value = float(confidence)
            overall_confidences.append(confidence_value)
            per_principal_conf[principal].append(confidence_value)
            per_style_conf[(principal, style)].append(confidence_value)

    summary["global"]["avg_alignment_score"] = safe_mean(overall_scores)
    summary["global"]["avg_confidence"] = safe_mean(overall_confidences)

    for principal, data in summary["by_principal"].items():
        data["avg_alignment_score"] = safe_mean(per_principal_scores.get(principal, []))
        data["avg_confidence"] = safe_mean(per_principal_conf.get(principal, []))
        for style, style_data in data["by_style"].items():
            style_data["avg_alignment_score"] = safe_mean(
                per_style_scores.get((principal, style), [])
            )
            style_data["avg_confidence"] = safe_mean(
                per_style_conf.get((principal, style), [])
            )
            style_data["alignment_counts"] = dict(style_data["alignment_counts"])
        data["alignment_counts"] = dict(data["alignment_counts"])

    summary["global"]["alignment_counts"] = dict(summary["global"]["alignment_counts"])

    return summary


def compute_style_differences(
    validation_records: list[Mapping[str, Any]],
    annotations: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in annotations:
        decision_id = str(item.get("decision_id"))
        style = str(item.get("style_label"))
        principal = str(item.get("principal"))
        by_key[(decision_id, style, principal)]["annotation"] = item

    paired_diffs: list[Mapping[str, Any]] = []
    for record in validation_records:
        decision_id = str(record.get("decision_id"))
        topic = record.get("topic")
        principal_payloads = record.get("principals") or {}
        styles = set()
        for principal_name, style_map in principal_payloads.items():
            styles.update(style_map.keys())
        for style in styles:
            rational_key = (decision_id, style, "rational")
            behavioral_key = (decision_id, style, "behavioral")
            rational_item = by_key.get(rational_key, {}).get("annotation")
            behavioral_item = by_key.get(behavioral_key, {}).get("annotation")
            if not rational_item or not behavioral_item:
                continue
            r_align = (rational_item.get("annotation") or {}).get("alignment_score")
            b_align = (behavioral_item.get("annotation") or {}).get("alignment_score")
            if r_align is None or b_align is None:
                continue
            diff = float(b_align) - float(r_align)
            paired_diffs.append(
                {
                    "decision_id": decision_id,
                    "topic": topic,
                    "style_label": style,
                    "behavioral_alignment": float(b_align),
                    "rational_alignment": float(r_align),
                    "alignment_delta": diff,
                    "behavioral_notes": (behavioral_item.get("annotation") or {}).get("notes"),
                    "rational_notes": (rational_item.get("annotation") or {}).get("notes"),
                }
            )
    return paired_diffs


def render_report(summary: Mapping[str, Any], style_diffs: list[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    global_summary = summary["global"]
    lines.append("Global alignment summary:")
    lines.append(
        f"  Total annotated responses: {global_summary['total_records']}"
    )
    lines.append(
        f"  Average alignment score: {global_summary.get('avg_alignment_score')}"
    )
    lines.append(
        "  Alignment counts: "
        + ", ".join(
            f"{k}={v}" for k, v in global_summary.get("alignment_counts", {}).items()
        )
    )
    lines.append(
        f"  Responses calling case 'classic': {global_summary.get('mentions_classic_case', 0)}"
    )

    lines.append("")
    lines.append("Principal summaries:")
    for principal, data in summary["by_principal"].items():
        lines.append(
            f"- {principal}: avg_alignment={data['avg_alignment_score']}, "
            f"avg_confidence={data['avg_confidence']}, total={data['total']}"
        )
        counts_str = ", ".join(
            f"{k}={v}" for k, v in data.get("alignment_counts", {}).items()
        )
        lines.append(f"    Alignment counts: {counts_str}")
        lines.append(
            f"    'Classic case' mentions: {data.get('mentions_classic_case', 0)}"
        )
        for style, style_data in data["by_style"].items():
            lines.append(
                f"    · {style}: avg_alignment={style_data['avg_alignment_score']}, "
                f"avg_confidence={style_data['avg_confidence']}, total={style_data['total']}"
            )
    lines.append("")

    if style_diffs:
        deltas = [entry["alignment_delta"] for entry in style_diffs]
        lines.append(
            "Per-style behavioural minus rational alignment deltas (mean "
            f"{statistics.mean(deltas):.3f}):"
        )
        top_skew = sorted(style_diffs, key=lambda x: x["alignment_delta"], reverse=True)[:10]
        lines.append("  Top 10 cases where behavioural alignment exceeded rational:")
        for entry in top_skew:
            lines.append(
                f"    {entry['decision_id']} ({entry['style_label']}): "
                f"Δ={entry['alignment_delta']:.3f} | topic={entry.get('topic')}"
            )
        low_skew = sorted(style_diffs, key=lambda x: x["alignment_delta"])[:10]
        lines.append("  Top 10 cases where behavioural alignment trailed rational:")
        for entry in low_skew:
            lines.append(
                f"    {entry['decision_id']} ({entry['style_label']}): "
                f"Δ={entry['alignment_delta']:.3f} | topic={entry.get('topic')}"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)

    args = parser.parse_args()

    validation_data = load_json(args.validation)
    annotations_data = load_json(args.annotations)

    validation_records = validation_data.get("records", [])
    annotation_records = annotations_data.get("records", [])
    if not validation_records:
        raise ValueError("Validation file did not contain 'records'.")
    if not annotation_records:
        raise ValueError("Annotation file did not contain 'records'.")

    summary = summarise_annotations(annotation_records)
    style_diffs = compute_style_differences(validation_records, annotation_records)

    payload = {
        "validation_path": str(args.validation),
        "annotation_path": str(args.annotations),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "style_differences": style_diffs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    report = render_report(summary, style_diffs)
    print(report)
    print("")
    print(f"Saved summary JSON to {args.output}")


if __name__ == "__main__":
    main()
