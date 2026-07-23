#!/usr/bin/env python3
"""Compare human harmful revisions with simulated principal decisions.

This script scans human annotation results, identifies harmful revisions
(correct step1 answer switched to incorrect step2 answer), and checks how
simulated rational (Bayesian) and behavioral decision-makers respond to the
same question under the same agent model.

It reports alignment metrics and can optionally export detailed case-level
results for further analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

# Mapping from full agent model identifiers to shorthand keys used in
# experiments/principals filenames.
MODEL_NAME_TO_KEY: Dict[str, str] = {
    "deepseek/deepseek-chat-v3.1": "deepseek",
    "google/gemini-2.5-pro": "gemini",
    "openai/gpt-5.1": "gpt",
    "anthropic/claude-haiku-4.5": "claude",
    "deepseek/deepseek-r1-distill-llama-70b": "deepseek-llama",
    "meta-llama/llama-3.3-70b-instruct": "llama",
    "meta-llama/llama-3.1-8b-instruct": "llama-small",
    "meta-llama/llama-3.1-405b-instruct": "llama-large",
    "allenai/Llama-3.1-Tulu-3-8B-DPO": "llama-dpo",
    "allenai/Llama-3.1-Tulu-3-8B-SFT": "llama-sft",
    "meta-llama/Llama-3.1-8B": "llama-base",
    "allenai/Llama-3.1-Tulu-3-70B-SFT": "llama-medium-sft",
    "allenai/Llama-3.1-Tulu-3-70B-DPO": "llama-medium-dpo",
    "allenai/Olmo-3-7B-Instruct": "olmo",
    "allenai/Olmo-3-7B-Instruct-SFT": "olmo-sft",
    "allenai/Olmo-3-7B-Instruct-DPO": "olmo-dpo",
    "allenai/Olmo-3-1025-7B": "olmo-base",
    "allenai/Olmo-3.1-32B-Instruct": "olmo-large",
    "allenai/Olmo-3.1-32B-Instruct-SFT": "olmo-large-sft",
    "allenai/Olmo-3.1-32B-Instruct-DPO": "olmo-large-dpo",
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_HUMAN_DIR = str((REPO_ROOT / "annotation/results/usmle_sample").resolve())
DEFAULT_PRINCIPAL_DIR = str(
    (REPO_ROOT / "experiments/principals/usmle_sample/baseline").resolve()
)


REVISION_TYPES = {"harmful", "helpful"}


DEFAULT_MODEL_KEYS = {
    "gpt",
    "gemini",
    "claude",
    "deepseek",
    "llama",
    "llama-small",
    "llama-sft",
    "llama-dpo",
}


PRINCIPAL_TYPES = {
    "rational": "bayesian_choices",
    "behavioral": "behavioral_choices",
}


@dataclass
class AnnotationCase:
    """Container for harmful human annotation details."""

    case_id: str
    agent_model: str
    agent_model_key: Optional[str]
    correct_answer_idx: Optional[str]
    human_step1_answer: Optional[str]
    human_step2_answer: Optional[str]
    human_step2_correct: Optional[bool]
    annotation_path: str


def list_json_files(directory: str) -> List[str]:
    return [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if name.endswith(".json")
    ]


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Failed to read {path}: {exc}")
        return None


def normalize_answer(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 1 and text.isalpha():
        return text.upper()
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return match.group(1)
    return None


def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clip_text(text: Optional[str], limit: int = 200) -> Optional[str]:
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def parse_decision(decision: Optional[str], options: Dict[str, str]) -> Optional[str]:
    if decision is None:
        return None
    text = str(decision).strip()
    if not text:
        return None
    if len(text) == 1 and text.isalpha():
        return text.upper()

    upper = text.upper()
    match = re.search(r"\b([A-Z])\b", upper)
    if match:
        letter = match.group(1)
        if letter in options:
            return letter

    normalized_options = {normalize_option_text(v): k for k, v in options.items()}
    option_key = normalized_options.get(normalize_option_text(text))
    if option_key:
        return option_key

    if "OPTION" in upper:
        match = re.search(r"OPTION\s+([A-Z])", upper)
        if match and match.group(1) in options:
            return match.group(1)

    return None


def normalize_option_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def is_revision(annotation: dict, revision_type: str) -> bool:
    step1 = annotation.get("step1", {})
    step2 = annotation.get("step2", {})
    change = annotation.get("step1_to_step2_changes", {}).get("answer_changed", False)
    if not change:
        return False
    if revision_type == "harmful":
        return bool(step1.get("is_correct") is True and step2.get("is_correct") is False)
    if revision_type == "helpful":
        return bool(step1.get("is_correct") is False and step2.get("is_correct") is True)
    raise ValueError(f"Unknown revision type: {revision_type}")


def collect_revision_cases(
    results_dir: str,
    dataset: Optional[str],
    allowed_models: Optional[Set[str]],
    revision_type: str,
) -> List[AnnotationCase]:
    cases: List[AnnotationCase] = []
    for path in list_json_files(results_dir):
        data = load_json(path)
        if not data:
            continue
        if dataset and data.get("dataset") != dataset:
            continue
        if not is_revision(data, revision_type):
            continue
        case_id = data.get("case_id")
        agent_model = data.get("agent_model")
        model_key = MODEL_NAME_TO_KEY.get(agent_model or "")
        if allowed_models and model_key not in allowed_models:
            continue
        step1_answer = normalize_answer(data.get("step1", {}).get("answer"))
        step2_answer = normalize_answer(data.get("step2", {}).get("answer"))
        step2_correct = data.get("step2", {}).get("is_correct")
        cases.append(
            AnnotationCase(
                case_id=case_id,
                agent_model=agent_model,
                agent_model_key=model_key,
                correct_answer_idx=normalize_answer(data.get("correct_answer_idx")),
                human_step1_answer=step1_answer,
                human_step2_answer=step2_answer,
                human_step2_correct=step2_correct,
                annotation_path=path,
            )
        )
    return cases


class PrincipalCache:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._cache: Dict[Tuple[str, str], Dict[str, dict]] = {}

    def _load(self, model_key: str, principal_suffix: str) -> Dict[str, dict]:
        cache_key = (model_key, principal_suffix)
        if cache_key in self._cache:
            return self._cache[cache_key]

        filename = f"principal_{model_key}_{principal_suffix}.json"
        path = os.path.join(self.base_dir, filename)
        records: Dict[str, dict] = {}
        if os.path.exists(path):
            data = load_json(path) or []
            for entry in data:
                cid = entry.get("case_id")
                if cid:
                    records[cid] = entry
        self._cache[cache_key] = records
        return records

    def get(self, model_key: str, principal_suffix: str, case_id: str) -> Optional[dict]:
        records = self._load(model_key, principal_suffix)
        return records.get(case_id)


def build_case_records(
    harmful_cases: Iterable[AnnotationCase],
    principal_dir: str,
) -> List[dict]:
    cache = PrincipalCache(principal_dir)
    records: List[dict] = []

    for case in harmful_cases:
        entry = {
            "case_id": case.case_id,
            "agent_model": case.agent_model,
            "agent_model_key": case.agent_model_key,
            "correct_answer_idx": case.correct_answer_idx,
            "human_step1_answer": case.human_step1_answer,
            "human_step2_answer": case.human_step2_answer,
            "human_step2_correct": case.human_step2_correct,
            "annotation_path": case.annotation_path,
        }

        for label, suffix in PRINCIPAL_TYPES.items():
            decision = None
            decision_text = None
            decision_correct = None
            belief = None
            question_snippet = None
            principal_model = None
            missing = False

            if case.agent_model_key and case.case_id:
                principal_entry = cache.get(case.agent_model_key, suffix, case.case_id)
            else:
                principal_entry = None

            if principal_entry:
                options = principal_entry.get("options") or {}
                decision_text = principal_entry.get("decision")
                decision = parse_decision(decision_text, options)
                belief = safe_float(principal_entry.get("belief"))
                principal_model = principal_entry.get("principal_model")
                if not question_snippet:
                    question_snippet = clip_text(principal_entry.get("principal_context"))
                if decision and case.correct_answer_idx:
                    decision_correct = decision == case.correct_answer_idx
                elif decision is None and case.correct_answer_idx is not None:
                    decision_correct = False
            else:
                missing = True

            entry[f"{label}_decision"] = decision
            entry[f"{label}_decision_text"] = decision_text
            entry[f"{label}_aligned"] = bool(decision and decision == case.human_step2_answer)
            entry[f"{label}_correct"] = decision_correct
            entry[f"{label}_belief"] = belief
            entry[f"{label}_principal_model"] = principal_model
            entry[f"{label}_missing"] = missing
            if question_snippet:
                entry.setdefault("question_snippet", question_snippet)

        records.append(entry)

    return records


def compute_metrics(records: Iterable[dict]) -> dict:
    records = list(records)
    total = len(records)
    stats = {
        "total_harmful_cases": total,
        "missing_model_key": sum(1 for r in records if r.get("agent_model_key") is None),
        "principal": {},
        "per_agent_model": {},
        "joint": {},
    }

    for label in PRINCIPAL_TYPES.keys():
        decision_key = f"{label}_decision"
        aligned_key = f"{label}_aligned"
        correct_key = f"{label}_correct"
        missing_key = f"{label}_missing"

        available = [r for r in records if not r.get(missing_key)]
        available_count = len(available)
        aligned_count = sum(1 for r in available if r.get(aligned_key))
        correct_disagree = sum(
            1
            for r in available
            if not r.get(aligned_key) and r.get(correct_key) is True
        )
        incorrect_disagree = sum(
            1
            for r in available
            if not r.get(aligned_key) and r.get(correct_key) is False
        )
        missing_count = total - available_count

        stats["principal"][label] = {
            "available": available_count,
            "aligned": aligned_count,
            "alignment_rate": aligned_count / available_count if available_count else 0.0,
            "contrarian_correct": correct_disagree,
            "contrarian_correct_rate": correct_disagree / available_count if available_count else 0.0,
            "contrarian_incorrect": incorrect_disagree,
            "missing": missing_count,
        }

    by_model: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        key = record.get("agent_model_key") or "unknown"
        by_model[key].append(record)

    per_model_stats = {}
    for model_key, items in sorted(by_model.items()):
        model_stats = {"count": len(items)}
        for label in PRINCIPAL_TYPES.keys():
            decision_key = f"{label}_decision"
            aligned_key = f"{label}_aligned"
            correct_key = f"{label}_correct"
            missing_key = f"{label}_missing"
            available = [r for r in items if not r.get(missing_key)]
            aligned = sum(1 for r in available if r.get(aligned_key))
            contrarian_correct = sum(
                1
                for r in available
                if not r.get(aligned_key) and r.get(correct_key) is True
            )
            model_stats[label] = {
                "available": len(available),
                "aligned": aligned,
                "alignment_rate": aligned / len(available) if available else 0.0,
                "contrarian_correct": contrarian_correct,
            }
        per_model_stats[model_key] = model_stats

    stats["per_agent_model"] = per_model_stats

    joint_available = [
        r
        for r in records
        if (not r.get("rational_missing") and not r.get("behavioral_missing"))
        and (r.get("rational_correct") is not None)
        and (r.get("behavioral_correct") is not None)
    ]
    joint_count = len(joint_available)
    categories = {
        "both_correct": [],
        "rational_only_correct": [],
        "behavioral_only_correct": [],
        "both_incorrect": [],
    }
    for record in joint_available:
        rc = bool(record.get("rational_correct"))
        bc = bool(record.get("behavioral_correct"))
        if rc and bc:
            categories["both_correct"].append(record)
        elif rc and not bc:
            categories["rational_only_correct"].append(record)
        elif not rc and bc:
            categories["behavioral_only_correct"].append(record)
        else:
            categories["both_incorrect"].append(record)

    stats["joint"] = {
        "both_available": joint_count,
        "categories": {key: len(val) for key, val in categories.items()},
        "records": categories,
    }
    return stats


def select_examples(
    records: Iterable[dict],
    label: str,
    *,
    aligned: bool,
    limit: int,
) -> List[dict]:
    filtered = [
        r for r in records
        if r.get(f"{label}_decision") is not None
        and not r.get(f"{label}_missing")
        and r.get(f"{label}_aligned") is aligned
    ]
    return filtered[:limit]


def export_json(records: Iterable[dict], path: str) -> None:
    with open(path, "w") as handle:
        json.dump(list(records), handle, indent=2)


def print_summary(stats: dict, records: Iterable[dict], top_k: int, revision_type: str) -> None:
    records = list(records)
    print(f"\n=== {revision_type.capitalize()} Revision Coverage ===")
    print(f"Total {revision_type} human decisions: {stats['total_harmful_cases']}")
    if stats["missing_model_key"]:
        print(f"Cases with unmapped agent model: {stats['missing_model_key']}")

    print("\n=== Principal Alignment Metrics ===")
    if revision_type == "harmful":
        print("(Alignment = principal repeats the human's harmful Step-2 answer)")
    else:
        print("(Alignment = principal repeats the human's helpful Step-2 answer)")
    for label, metrics in stats["principal"].items():
        print(f"{label.capitalize()} principal:")
        print(f"  Available cases:       {metrics['available']}")
        print(f"  Alignment rate:        {metrics['alignment_rate']*100:5.1f}% ({metrics['aligned']}/{metrics['available']})")
        print(f"  Contrarian correct:    {metrics['contrarian_correct']} ({metrics['contrarian_correct_rate']*100:5.1f}%)")
        print(f"  Contrarian incorrect:  {metrics['contrarian_incorrect']}")
        print(f"  Missing coverage:      {metrics['missing']}")

    print("\n=== Alignment by Agent Model ===")
    for model_key, model_stats in stats["per_agent_model"].items():
        print(f"Model {model_key}: {model_stats['count']} {revision_type} cases")
        for label in PRINCIPAL_TYPES.keys():
            info = model_stats[label]
            if info["available"]:
                rate = info["alignment_rate"] * 100
                print(
                    f"  {label.capitalize():10} aligned {info['aligned']}/{info['available']} ({rate:5.1f}%)"
                    f", contrarian correct {info['contrarian_correct']}"
                )
            else:
                print(f"  {label.capitalize():10} no coverage")

    for label in PRINCIPAL_TYPES.keys():
        aligned_examples = select_examples(records, label, aligned=True, limit=top_k)
        misaligned_examples = select_examples(records, label, aligned=False, limit=top_k)
        print(f"\n=== Example cases for {label} principal (aligned) ===")
        if aligned_examples:
            for ex in aligned_examples:
                print(
                    f"- {ex['case_id']} | agent={ex['agent_model_key']} | human {ex['human_step2_answer']}"
                    f" | principal {ex[f'{label}_decision']} | belief={ex[f'{label}_belief']}"
                )
                if ex.get("question_snippet"):
                    print(f"  Q: {ex['question_snippet']}")
        else:
            print("  (none)")

        print(f"\n=== Example cases for {label} principal (misaligned) ===")
        if misaligned_examples:
            for ex in misaligned_examples:
                print(
                    f"- {ex['case_id']} | agent={ex['agent_model_key']} | human {ex['human_step2_answer']}"
                    f" | principal {ex[f'{label}_decision']} | correct={ex[f'{label}_correct']}"
                )
                if ex.get("question_snippet"):
                    print(f"  Q: {ex['question_snippet']}")
        else:
            print("  (none)")

    joint_stats = stats.get("joint", {})
    print("\n=== Principal correctness comparison (vs. ground-truth answer) ===")
    print(f"Cases with outputs from both principals: {joint_stats.get('both_available', 0)}")
    labels = [
        ("both_correct", "Both principals correct"),
        ("rational_only_correct", "Only rational correct"),
        ("behavioral_only_correct", "Only behavioral correct"),
        ("both_incorrect", "Both incorrect"),
    ]
    for key, description in labels:
        count = joint_stats.get("categories", {}).get(key, 0)
        rate = (
            count / joint_stats.get("both_available", 1)
            if joint_stats.get("both_available")
            else 0.0
        )
        print(f"  {description:<28}: {count:3d} ({rate*100:5.1f}%)")
        examples = joint_stats.get("records", {}).get(key, [])[:top_k]
        if examples:
            for ex in examples:
                rc = "correct" if ex.get("rational_correct") else "wrong"
                bc = "correct" if ex.get("behavioral_correct") else "wrong"
                print(
                    f"    - {ex['case_id']} | agent={ex['agent_model_key']} | human {revision_type}={ex['human_step2_answer']}"
                    f" | rational={ex['rational_decision']} ({rc}) | behavioral={ex['behavioral_decision']} ({bc})"
                )
                if ex.get("question_snippet"):
                    print(f"      Q: {ex['question_snippet']}")
        else:
            print("    (none)")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-dir",
        default=DEFAULT_HUMAN_DIR,
        help="Directory with human annotation JSON files.",
    )
    parser.add_argument(
        "--principal-dir",
        default=DEFAULT_PRINCIPAL_DIR,
        help="Directory with principal decision JSON files.",
    )
    parser.add_argument(
        "--dataset",
        default="usmle_sample",
        help="Filter annotations to this dataset key (default: usmle_sample).",
    )
    parser.add_argument(
        "--revision-type",
        choices=sorted(REVISION_TYPES),
        default="harmful",
        help="Which human revision type to analyze (default: harmful).",
    )
    parser.add_argument(
        "--model-keys",
        default=",".join(sorted(DEFAULT_MODEL_KEYS)),
        help=(
            "Comma-separated list of agent model keys to include."
            " Use 'all' to disable filtering."
        ),
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to export detailed matched records as JSON.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of example cases to show for each alignment category.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.human_dir):
        raise FileNotFoundError(f"Human results directory not found: {args.human_dir}")
    if not os.path.isdir(args.principal_dir):
        raise FileNotFoundError(f"Principal results directory not found: {args.principal_dir}")

    model_keys_arg = (args.model_keys or "").strip()
    if model_keys_arg.lower() == "all":
        allowed_models = None
    else:
        allowed_models = {
            key.strip()
            for key in model_keys_arg.split(",")
            if key.strip()
        }
        if not allowed_models:
            allowed_models = DEFAULT_MODEL_KEYS

    revision_type = args.revision_type

    revision_cases = collect_revision_cases(
        args.human_dir,
        dataset=args.dataset,
        allowed_models=allowed_models,
        revision_type=revision_type,
    )
    if not revision_cases:
        print(f"No {revision_type} revisions found with the provided filters.")
        return

    records = build_case_records(revision_cases, args.principal_dir)
    stats = compute_metrics(records)
    print_summary(stats, records, args.topk, revision_type)

    if args.json_out:
        export_json(records, args.json_out)
        print(f"\nDetailed records saved to {args.json_out}")


if __name__ == "__main__":
    main()
