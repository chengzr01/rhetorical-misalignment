#!/usr/bin/env python3
"""Generate USMLE question variants by permuting answer option order.

The resulting datasets keep question stems and answer texts identical but
shuffle the label assignments (A/B/C/…) to stress-test model robustness
against deterministic option ordering.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import string
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_INPUT = Path("experiments/questions/clinical_questions_usmle_sample.json")
DEFAULT_OUTPUT = Path("experiments/questions/clinical_questions_usmle_sample_permuted.json")
DEFAULT_PREFIX = "clinical_questions_usmle_sample_permuted"


def validate_options(options: Dict[str, Any], question_id: str) -> None:
    if not options:
        raise ValueError(f"Question {question_id} has no options to permute")
    if len(options) > len(string.ascii_uppercase):
        raise ValueError(
            f"Question {question_id} has {len(options)} options; maximum supported is {len(string.ascii_uppercase)}"
        )


def permute_question(
    question: Dict[str, Any],
    *,
    rng: random.Random,
    seed: int,
) -> Dict[str, Any]:
    question_id = question.get("id", "unknown")
    original_options = question.get("options")
    if not isinstance(original_options, dict):
        raise ValueError(f"Question {question_id} must have 'options' as a dict")

    validate_options(original_options, question_id)

    option_items = list(original_options.items())
    rng.shuffle(option_items)

    labels = list(string.ascii_uppercase[: len(option_items)])

    new_options: Dict[str, Any] = {}
    permutation: Dict[str, str] = {}
    new_correct_idx: str | None = None

    original_answer_idx = question.get("answer_idx")
    original_answer_text = question.get("answer")

    for label, (original_label, text) in zip(labels, option_items):
        new_options[label] = text
        permutation[label] = original_label
        if original_label == original_answer_idx:
            new_correct_idx = label

    if new_correct_idx is None:
        raise ValueError(
            f"Failed to map correct answer for question {question_id} during permutation"
        )

    new_record = copy.deepcopy(question)
    new_record["options"] = new_options
    new_record["answer_idx"] = new_correct_idx
    new_record["answer"] = new_options[new_correct_idx]
    new_record["variant"] = "option_permutation"
    new_record["option_permutation"] = permutation
    new_record["permutation_seed"] = seed
    new_record["original_answer_idx"] = original_answer_idx
    new_record.setdefault("original_answer", original_answer_text)

    return new_record


def generate_variants(
    questions: List[Dict[str, Any]],
    *,
    base_seed: int,
    num_variants: int,
) -> List[List[Dict[str, Any]]]:
    variants: List[List[Dict[str, Any]]] = []
    for offset in range(num_variants):
        seed = base_seed + offset
        rng = random.Random(seed)
        variant_records = [
            permute_question(question, rng=rng, seed=seed) for question in questions
        ]
        variants.append(variant_records)
    return variants


def write_variants(
    variants: List[List[Dict[str, Any]]],
    *,
    single_output: Path | None,
    output_dir: Path,
    prefix: str,
    base_seed: int,
    overwrite: bool,
) -> List[Path]:
    output_paths: List[Path] = []

    if len(variants) == 1 and single_output is not None:
        path = single_output
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {path} already exists. Pass --overwrite to replace it."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(variants[0], indent=2, ensure_ascii=False))
        output_paths.append(path)
        return output_paths

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, records in enumerate(variants):
        seed = base_seed + idx
        filename = f"{prefix}_seed{seed}.json"
        path = output_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file {path} already exists. Pass --overwrite to replace it."
            )
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        output_paths.append(path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create option-permuted USMLE question datasets for contamination checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Source clinical questions JSON file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output file when generating a single variant")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT.parent,
                        help="Directory to write multiple variants")
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX,
                        help="Filename prefix for multi-variant outputs")
    parser.add_argument("--seed", type=int, default=13,
                        help="Base random seed for option permutations")
    parser.add_argument("--num-variants", type=int, default=1,
                        help="Number of independent permutations to generate")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only permute the first N questions (debugging)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing output files")

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    questions = json.loads(args.input.read_text())
    if not isinstance(questions, list):
        raise ValueError("Input questions file must be a list of question records")

    if args.limit is not None:
        questions = questions[: args.limit]

    variants = generate_variants(
        questions,
        base_seed=args.seed,
        num_variants=args.num_variants,
    )

    written_paths = write_variants(
        variants,
        single_output=args.output if args.num_variants == 1 else None,
        output_dir=args.output_dir,
        prefix=args.prefix,
        base_seed=args.seed,
        overwrite=args.overwrite,
    )

    if len(written_paths) == 1:
        print(f"Saved permuted dataset → {written_paths[0]}")
    else:
        print("Saved permuted datasets:")
        for path in written_paths:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
