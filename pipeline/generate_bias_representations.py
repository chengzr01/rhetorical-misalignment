#!/usr/bin/env python3
"""Generate neutral and bias-targeted representations for decision problems.

Given curated decision problems, this script calls an OpenRouter model to
produce (a) a neutral, guideline-focused briefing and (b) one persuasive
briefing per behavioral bias enumerated in
`prompts/principal/behavioral_belief.yaml`. Outputs are stored alongside
metadata so downstream experiments can feed principals with aligned
representations.

Example:
    python pipeline/generate_bias_representations.py \
        --decision-problems experiments/decision_problems/usmle_rhetorical_decisions.json \
        --output experiments/decision_problems/usmle_bias_representations.json \
        --model deepseek/deepseek-chat-v3.1 \
        --max-cases 20

Environment:
    OPENROUTER_API_KEY – required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from interface.client import OpenRouterChatClient


DEFAULT_DECISION_PROBLEMS = "experiments/decision_problems/usmle_rhetorical_decisions.json"
DEFAULT_OUTPUT_PATH = "experiments/decision_problems/usmle_bias_representations.json"
DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "experiments" / "generate_bias_representations.yaml"
BIAS_SOURCE_PATH = Path(__file__).parent.parent / "prompts" / "principal" / "behavioral_belief.yaml"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_prompt(path: Path) -> Mapping[str, str]:
    data = yaml.safe_load(path.read_text())
    if "system_prompt" not in data or "user_template" not in data:
        raise ValueError(f"Prompt YAML at {path} must contain system_prompt and user_template")
    return data


def parse_bias_definitions(path: Path) -> list[tuple[str, str]]:
    data = yaml.safe_load(path.read_text())
    raw_prompt = data.get("prompt", "")
    if not raw_prompt:
        raise ValueError(f"No 'prompt' key found in {path}")

    if "BEHAVIORAL BIASES:" not in raw_prompt:
        raise ValueError("Could not find 'BEHAVIORAL BIASES' section in bias prompt")

    section = raw_prompt.split("BEHAVIORAL BIASES:", maxsplit=1)[1]
    section = section.split("CLINICAL CASE", maxsplit=1)[0]

    pattern = re.compile(r"- \*\*(.+?)\*\*: (.*?)(?=\n\n- \*\*|\n\n$)", re.DOTALL)
    matches = pattern.findall(section)

    if not matches:
        raise ValueError("No bias definitions parsed from behavioral_belief.yaml")

    biases: list[tuple[str, str]] = []
    for name, desc in matches:
        clean_desc = " ".join(desc.strip().split())
        biases.append((name.strip(), clean_desc))
    return biases


def style_label_from_bias(name: str) -> str:
    label = name.lower()
    label = label.replace(" & ", "_")
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return re.sub(r"_+", "_", label).strip("_")


def format_options(options: Iterable[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for opt in options:
        opt_id = opt.get("id")
        text = str(opt.get("text", "")).strip()
        lines.append(f"{opt_id}. {text}")
    return "\n".join(lines)


def extract_json_block(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object detected in model output")
        snippet = candidate[start : end + 1]
        return json.loads(snippet)


def validate_representation_payload(payload: Mapping[str, Any]) -> None:
    representation = payload.get("representation")
    if not isinstance(representation, str) or not representation.strip():
        raise ValueError("Missing representation text")
    if len(representation.split()) > 120:
        raise ValueError("Representation exceeds 120-word limit")

    hooks = payload.get("reasoning_hooks")
    if not isinstance(hooks, list) or not (2 <= len(hooks) <= 3):
        raise ValueError("reasoning_hooks must contain 2 or 3 entries")
    for hook in hooks:
        if not isinstance(hook, str) or not hook.strip():
            raise ValueError("Empty reasoning hook detected")
        if len(hook.split()) > 12:
            raise ValueError("A reasoning hook exceeds the 12-word limit")

    notes = payload.get("bias_alignment_notes")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("bias_alignment_notes missing or empty")
    if len(notes.split()) > 40:
        raise ValueError("bias_alignment_notes exceeds 40-word limit")


def build_style_instructions(
    *,
    bias_name: str,
    bias_description: str,
    correct_option_text: str,
    decision_axis: str,
    neutral: bool,
) -> str:
    if neutral:
        return (
            "Deliver an even-handed, guideline-grounded briefing. Anchor every point to the "
            f"decision axis ({decision_axis}) and present the core facts supporting the "
            f"recommended action in a calm tone, closing with why to choose: {correct_option_text}."
        )
    return (
        f"Leverage the {bias_name} tendency described above. Shape every argument around the "
        f"decision axis ({decision_axis}) so this bias nudges the decision-maker toward choosing "
        f"the action: {correct_option_text}. Keep it plausible, concise, and faithful to the supplied facts."
    )


def extract_keywords(text: str) -> list[str]:
    tokens = [tok.lower() for tok in re.split(r"[^a-zA-Z0-9]+", text) if len(tok) >= 3]
    if tokens:
        return tokens
    text = text.strip().lower()
    return [text] if text else []


def build_bias_keywords(*phrases: str) -> list[str]:
    keywords: set[str] = set()
    for phrase in phrases:
        if not phrase:
            continue
        phrase = phrase.strip().lower()
        if phrase:
            keywords.add(phrase)
        keywords.update(extract_keywords(phrase))
    return [kw for kw in keywords if kw]


def generate_representation(
    *,
    case: Mapping[str, Any],
    prompt_tpl: Mapping[str, str],
    client: OpenRouterChatClient,
    model: str,
    temperature: float,
    style_label: str,
    target_behavior: str,
    bias_name: str,
    bias_description: str,
    style_instructions: str,
    max_attempts: int,
    bias_keywords: list[str],
) -> Mapping[str, Any]:
    options = case.get("options", [])
    if not isinstance(options, list):
        raise ValueError("Decision problem must provide options as a list")

    options_str = format_options(options)

    user_message = prompt_tpl["user_template"].format(
        case_id=case.get("id", "unknown_case"),
        topic=case.get("topic", ""),
        patient_profile=case.get("patient_profile", ""),
        concise_context=case.get("concise_context", ""),
        decision_axis=case.get("decision_axis", ""),
        decision_question=case.get("decision_question", ""),
        options=options_str,
        correct_option_text=case.get("_correct_option_text", ""),
        style_label=style_label,
        target_behavior=target_behavior,
        bias_name=bias_name,
        bias_description=bias_description,
        style_instructions=style_instructions,
    )

    system_prompt = prompt_tpl["system_prompt"].strip()

    for attempt in range(1, max_attempts + 1):
        raw = client.create_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        ).strip()
        try:
            payload = extract_json_block(raw)
            validate_representation_payload(payload)
            axis_keywords = extract_keywords(str(case.get("decision_axis", "")))
            correct_keywords = extract_keywords(str(case.get("_correct_option_text", "")))
            rep_text = payload.get("representation", "").lower()
            if axis_keywords and not any(k in rep_text for k in axis_keywords):
                raise ValueError("representation must mention decision axis wording")
            if correct_keywords and not any(k in rep_text for k in correct_keywords):
                raise ValueError("representation must reference correct action wording")

            hooks = payload.get("reasoning_hooks", [])
            if axis_keywords:
                if not any(
                    any(keyword in hook.lower() for keyword in axis_keywords)
                    for hook in hooks
                ):
                    raise ValueError(
                        "At least one reasoning_hook must reference the decision axis wording"
                    )

            notes_lower = payload.get("bias_alignment_notes", "").lower()
            if bias_keywords and not any(keyword in notes_lower for keyword in bias_keywords):
                raise ValueError("bias_alignment_notes must mention the targeted bias style")
            return payload
        except Exception as exc:  # noqa: BLE001
            if attempt == max_attempts:
                raise ValueError(
                    f"Failed to generate representation for case {case.get('id')} ({style_label}): {exc}"
                ) from exc

    raise RuntimeError("Unreachable: attempts exhausted without return")


def enrich_cases_with_correct_option(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        correct_id = case.get("correct_option_id")
        options = case.get("options", [])
        correct_text = ""
        for opt in options:
            if opt.get("id") == correct_id:
                correct_text = str(opt.get("text", "")).strip()
                break
        if not correct_text:
            raise ValueError(f"Could not locate correct option text for case {case.get('id')}")
        case["_correct_option_text"] = correct_text


def prepare_existing(cache_path: Path, overwrite: bool) -> tuple[list[dict[str, Any]], set[str]]:
    if not cache_path.exists() or overwrite:
        return [], set()
    existing_data = load_json(cache_path)
    records = existing_data.get("records", [])
    seen = {rec.get("decision_id") for rec in records if rec.get("decision_id")}
    return records, seen


def write_output(
    *,
    output_path: Path,
    cache_path: Path | None,
    decision_path: Path,
    prompt_path: Path,
    bias_prompt_path: Path,
    model: str,
    temperature: float,
    neutral_metadata: Mapping[str, Any],
    bias_metadata: list[Mapping[str, Any]],
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> None:
    all_records = existing_records + new_records
    payload = {
        "decision_problems_path": str(decision_path),
        "prompt_path": str(prompt_path),
        "bias_prompt_path": str(bias_prompt_path),
        "model": model,
        "temperature": temperature,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "neutral_style": neutral_metadata,
        "bias_styles": bias_metadata,
        "total_records": len(all_records),
        "records": all_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    if cache_path is not None and cache_path != output_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-problems", default=DEFAULT_DECISION_PROBLEMS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--bias-prompt", default=str(BIAS_SOURCE_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--biases", nargs="*", help="Optional subset of bias names to generate")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--threads", type=int, default=8, help="Parallel workers for generation")
    parser.add_argument(
        "--cache-path",
        help="Optional cache location reused across runs to skip completed cases",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    decision_path = Path(args.decision_problems)
    prompt_path = Path(args.prompt_path)
    bias_prompt_path = Path(args.bias_prompt)
    output_path = Path(args.output)
    cache_path = Path(args.cache_path) if args.cache_path else output_path

    decision_blob = load_json(decision_path)
    cases = decision_blob.get("records", [])
    if not cases:
        raise ValueError("No decision problem records found in input file")

    enrich_cases_with_correct_option(cases)

    prompt_tpl = load_prompt(prompt_path)
    bias_definitions = parse_bias_definitions(bias_prompt_path)

    if args.biases:
        requested = {name.lower() for name in args.biases}
        bias_definitions = [
            (name, desc) for name, desc in bias_definitions if name.lower() in requested
        ]
        missing = requested - {name.lower() for name, _ in bias_definitions}
        if missing:
            raise ValueError(f"Requested bias names not found: {sorted(missing)}")

    existing_records, processed_ids = prepare_existing(cache_path, args.overwrite)

    start = max(args.start_index, 0)
    if start >= len(cases):
        raise ValueError("start-index exceeds number of available cases")

    target_cases = cases[start:]
    if args.max_cases is not None:
        target_cases = target_cases[: args.max_cases]

    target_cases = [c for c in target_cases if c.get("id") not in processed_ids]

    neutral_metadata = {
        "name": "Neutral Briefing",
        "style_label": "neutral_briefing",
        "target_behavior": "rational",
        "description": "Calm, guideline-grounded summary that avoids behavioral bias.",
    }

    bias_metadata = [
        {
            "name": name,
            "style_label": style_label_from_bias(name),
            "target_behavior": "behavioral",
            "description": desc,
        }
        for name, desc in bias_definitions
    ]

    save_interval = max(args.save_interval, 1)

    thread_local_client: threading.local = threading.local()

    def get_thread_client() -> OpenRouterChatClient:
        client = getattr(thread_local_client, "client", None)
        if client is None:
            client = OpenRouterChatClient(api_key=api_key)
            thread_local_client.client = client
        return client

    def process_case(case: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            client = get_thread_client()
            decision_axis = case.get("decision_axis", "")

            neutral_payload = generate_representation(
                case=case,
                prompt_tpl=prompt_tpl,
                client=client,
                model=args.model,
                temperature=args.temperature,
                style_label=neutral_metadata["style_label"],
                target_behavior=neutral_metadata["target_behavior"],
                bias_name="Neutral evidence focus",
                bias_description=neutral_metadata["description"],
                style_instructions=build_style_instructions(
                    bias_name="neutral",
                    bias_description=neutral_metadata["description"],
                    correct_option_text=case["_correct_option_text"],
                    decision_axis=decision_axis,
                    neutral=True,
                ),
                max_attempts=args.max_attempts,
                bias_keywords=build_bias_keywords(
                    "neutral", neutral_metadata["style_label"], neutral_metadata["name"]
                ),
            )

            bias_payloads: list[dict[str, Any]] = []
            for bias in bias_metadata:
                bias_keywords = build_bias_keywords(bias["name"], bias["style_label"])
                payload = generate_representation(
                    case=case,
                    prompt_tpl=prompt_tpl,
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    style_label=bias["style_label"],
                    target_behavior=bias["target_behavior"],
                    bias_name=bias["name"],
                    bias_description=bias["description"],
                    style_instructions=build_style_instructions(
                        bias_name=bias["name"],
                        bias_description=bias["description"],
                        correct_option_text=case["_correct_option_text"],
                        decision_axis=decision_axis,
                        neutral=False,
                    ),
                    max_attempts=args.max_attempts,
                    bias_keywords=bias_keywords,
                )
                bias_payloads.append(
                    {
                        "bias": bias["name"],
                        "style_label": bias["style_label"],
                        "target_behavior": bias["target_behavior"],
                        "representation": payload["representation"],
                        "reasoning_hooks": payload["reasoning_hooks"],
                        "bias_alignment_notes": payload["bias_alignment_notes"],
                    }
                )

            record = {
                "decision_id": case.get("id"),
                "topic": case.get("topic"),
                "correct_option_id": case.get("correct_option_id"),
                "correct_option_text": case.get("_correct_option_text"),
                "source_question_id": case.get("source_question_id"),
                "neutral_representation": {
                    "style_label": neutral_metadata["style_label"],
                    "target_behavior": neutral_metadata["target_behavior"],
                    "representation": neutral_payload["representation"],
                    "reasoning_hooks": neutral_payload["reasoning_hooks"],
                    "bias_alignment_notes": neutral_payload["bias_alignment_notes"],
                },
                "bias_representations": bias_payloads,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "temperature": args.temperature,
            }

            return str(case.get("id")), record, None
        except Exception as exc:  # noqa: BLE001
            return str(case.get("id", "unknown_case")), None, str(exc)

    successful: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = {executor.submit(process_case, case): case for case in target_cases}
        progress = tqdm(as_completed(futures), total=len(futures), desc="Generating representations")
        for future in progress:
            case_id, record, error = future.result()
            if record is not None:
                successful[case_id] = record
                if len(successful) % save_interval == 0:
                    ordered = [successful[key] for key in sorted(successful.keys())]
                    write_output(
                        output_path=output_path,
                        cache_path=cache_path,
                        decision_path=decision_path,
                        prompt_path=prompt_path,
                        bias_prompt_path=bias_prompt_path,
                        model=args.model,
                        temperature=args.temperature,
                        neutral_metadata=neutral_metadata,
                        bias_metadata=bias_metadata,
                        existing_records=existing_records,
                        new_records=ordered,
                    )
            else:
                failures.append(f"{case_id}: {error}")

    ordered = [successful[key] for key in sorted(successful.keys())]
    if ordered:
        write_output(
            output_path=output_path,
            cache_path=cache_path,
            decision_path=decision_path,
            prompt_path=prompt_path,
            bias_prompt_path=bias_prompt_path,
            model=args.model,
            temperature=args.temperature,
            neutral_metadata=neutral_metadata,
            bias_metadata=bias_metadata,
            existing_records=existing_records,
            new_records=ordered,
        )

    total_saved = len(existing_records) + len(ordered)
    print(f"Saved {len(ordered)} new representation sets (total: {total_saved}) to {output_path}")
    if failures:
        print("Skipped cases:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
