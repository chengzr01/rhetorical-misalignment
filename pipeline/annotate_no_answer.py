#!/usr/bin/env python3
"""
Use DeepSeek-V3.1 via OpenRouter to annotate questions where models failed
to produce a parseable answer.

For each no-answer entry in a results file, sends the original question,
options, and model response to DeepSeek and asks it to extract the intended
answer (A-E). Updates the results file in-place.

Example usage:
    python annotate_no_answer.py --input tests/test_usmle_sample_openai-gpt-5.1_belief.json
    python annotate_no_answer.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient

ANNOTATOR_MODEL = "deepseek/deepseek-chat-v3-0324"

ANNOTATOR_PROMPT = """\
A language model was asked to answer a multiple-choice medical question but \
failed to produce a parseable answer. Your task is to read the question, the \
answer options, and the model's response, then determine which option (A–E) the \
model intended to select.

If the model's response clearly indicates or implies a specific option, output \
that letter. If the response contains no useful signal (e.g. a refusal, a blank, \
or a placeholder like "letter"), output "None".

Respond with ONLY a single line in this exact format:
<answer>X</answer>
where X is a letter A–E, or:
<answer>None</answer>

---
QUESTION:
{question}

OPTIONS:
{options}

MODEL RESPONSE:
{response}
"""


def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*([A-E]|None)\s*</answer>", text, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        return None if val == "NONE" else val
    return None


def annotate_entry(
    entry: dict,
    client: OpenRouterChatClient,
) -> dict:
    """Ask DeepSeek to extract the answer from the model's response."""
    prompt = ANNOTATOR_PROMPT.format(
        question=entry["question"],
        options=format_options(entry["options"]),
        response=entry.get("response") or "(no response)",
    )
    try:
        response = client.create_completion(
            model=ANNOTATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        extracted = extract_answer(response)
    except Exception as e:
        print(f"  Error annotating {entry.get('id')}: {e}")
        extracted = None

    return {**entry, "predicted_answer_idx": extracted,
            "predicted_answer": entry["options"].get(extracted) if extracted else None,
            "correct": (extracted == entry["correct_answer_idx"]) if extracted else False,
            "annotated_by": ANNOTATOR_MODEL}


def annotate_file(path: Path, client: OpenRouterChatClient, max_workers: int = 4) -> None:
    with open(path) as f:
        data = json.load(f)

    # Support both old format (list) and new format (dict with "results" key)
    if isinstance(data, list):
        results = data
        wrapper = None
    else:
        results = data.get("results", [])
        wrapper = data

    no_answer = [r for r in results if r.get("predicted_answer_idx") is None]
    if not no_answer:
        print(f"  No missing answers in {path.name}, skipping.")
        return

    print(f"  Annotating {len(no_answer)} entries in {path.name}...")

    annotated_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(annotate_entry, e, client): e for e in no_answer}
        for future in tqdm(as_completed(futures), total=len(no_answer)):
            result = future.result()
            annotated_map[result["id"]] = result

    # Merge back
    updated = []
    newly_correct = 0
    still_none = 0
    for r in results:
        if r["id"] in annotated_map:
            new_r = annotated_map[r["id"]]
            if new_r["predicted_answer_idx"] is None:
                still_none += 1
            elif new_r["correct"]:
                newly_correct += 1
            updated.append(new_r)
        else:
            updated.append(r)

    print(f"  Recovered answers: {len(no_answer) - still_none}/{len(no_answer)}")
    print(f"  Of those, correct: {newly_correct}")
    print(f"  Still no answer:   {still_none}")

    # Recalculate metrics
    total = len(updated)
    correct = sum(1 for r in updated if r["correct"])
    no_ans = sum(1 for r in updated if r["predicted_answer_idx"] is None)
    metrics = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct - no_ans,
        "no_answer": no_ans,
        "accuracy": correct / total if total else 0.0,
    }
    for step in ["step1", "step2", "step3"]:
        sr = [r for r in updated if r.get("meta_info") == step]
        sc = sum(1 for r in sr if r["correct"])
        metrics[step] = {
            "total": len(sr),
            "correct": sc,
            "accuracy": sc / len(sr) if sr else 0.0,
        }

    print(f"  Updated accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    # Save (preserve original format)
    if wrapper is not None:
        wrapper["results"] = updated
        wrapper["metrics"] = metrics
        out = wrapper
    else:
        out = {"results": updated, "metrics": metrics}

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate no-answer entries using DeepSeek-V3.1 via OpenRouter."
    )
    parser.add_argument("--input", type=str, help="Single results JSON file to annotate")
    parser.add_argument("--all", action="store_true", help="Annotate all files in tests/")
    parser.add_argument(
        "--tests-dir", type=str, default="experiments/tests",
        help="Directory containing result files (used with --all)"
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if not args.input and not args.all:
        parser.error("Specify --input <file> or --all")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    client = OpenRouterChatClient(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    if args.input:
        annotate_file(Path(args.input), client, max_workers=args.max_workers)
    else:
        files = sorted(Path(args.tests_dir).glob("*.json"))
        for f in files:
            annotate_file(f, client, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
