#!/usr/bin/env python3
"""
Run agent information-design experiments where agents select a subset of claims.

The agent is shown a numbered list of all available claims for a case and returns
a JSON array of the claim numbers it wants to include. The selected claims are then
rendered in the same standardized bullet format used across all other experiments,
so the principal sees no agent-specific language — only the agent's selection choice.

This is the complement to agent_presentation.py (run_framing.sh):
  - Framing:   information fixed (all claims), language free  → measures framing effect
  - Selection: information free (agent selects), language fixed → measures information design effect

Usage:
    python agent_selection.py \\
        --aggregated-info experiments/aggregation/aggregated_factual.json \\
        --questions       experiments/questions/clinical_questions_usmle_sample.json \\
        --agent-server    openrouter \\
        --agent-model     meta-llama/llama-3.3-70b-instruct \\
        --output          experiments/agents/usmle_sample/information_llama_gt_factual_agg.json

    python agent_selection.py \\
        --ground-truth    experiments/agents/usmle_sample/agent_deepseek.json \\
        --agent-server    openrouter \\
        --agent-model     deepseek/deepseek-chat-v3-0324 \\
        --output          experiments/agents/usmle_sample/information_deepseek_gt_deepseek.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from agents.agent import Agent
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


DEFAULT_PROMPT_PATH = "prompts/agent/information_usmle.yaml"


# ---------------------------------------------------------------------------
# Claim formatting helpers
# ---------------------------------------------------------------------------

def format_claims_numbered(claims: list[dict[str, Any]]) -> str:
    """Format a list of claim dicts as a 1-based numbered list for the agent prompt."""
    lines = []
    for i, c in enumerate(claims, start=1):
        text = c.get("claim", "").strip()
        if text:
            lines.append(f"{i}. {text}")
    return "\n".join(lines)


def format_claims_bullets(claims: list[dict[str, Any]]) -> str:
    """Render claims as standardized bullet points (the format shown to the principal)."""
    lines = []
    for c in claims:
        text = c.get("claim", "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Index parsing
# ---------------------------------------------------------------------------

def parse_selected_indices(response: str, n_available: int) -> list[int]:
    """
    Parse the agent's response into a list of 1-based claim indices.

    Tries in order:
      1. JSON array anywhere in the response
      2. Comma/space-separated integers
    Returns only valid indices (1 … n_available). Duplicates are removed.
    Logs a warning and returns all indices if parsing fails entirely.
    """
    # Try to extract a JSON array from <answer> tag first, then anywhere in response
    answer_tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    search_text = answer_tag_match.group(1) if answer_tag_match else response
    json_match = re.search(r"\[[\d,\s]+\]", search_text)
    if json_match:
        try:
            indices = json.loads(json_match.group())
            valid = sorted({i for i in indices if isinstance(i, int) and 1 <= i <= n_available})
            if valid:
                return valid
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: extract any integers from the response
    numbers = [int(m) for m in re.findall(r"\b\d+\b", response)]
    valid = sorted({i for i in numbers if 1 <= i <= n_available})
    if valid:
        return valid

    # Could not parse — warn and include all claims
    print(f"  WARNING: could not parse selection from response; including all {n_available} claims.\n"
          f"  Response was: {response[:200]!r}")
    return list(range(1, n_available + 1))


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_aggregated_claims(
    aggregated_path: Path,
    questions_path: Path | None,
) -> list[dict[str, Any]]:
    """
    Load aggregated_factual.json and return records with raw claim lists.
    Each record includes 'claims' as a list of claim dicts (not pre-formatted),
    so agent_selection can format them as a numbered list for the agent.
    """
    aggregated = json.loads(aggregated_path.read_text())
    cases: dict[str, dict] = aggregated.get("cases", {})
    metadata = aggregated.get("metadata", {})
    keep_labels = metadata.get("keep_labels", ["?"])
    print(f"Loaded aggregated claims from {aggregated_path}")
    print(f"  Labels: {', '.join(keep_labels)}  |  Cases: {len(cases)}")

    question_lookup: dict[str, dict] = {}
    if questions_path is not None:
        questions = json.loads(questions_path.read_text())
        question_lookup = {q["id"]: q for q in questions if "id" in q}
        print(f"  Loaded {len(question_lookup)} questions from {questions_path}")
    else:
        print("  WARNING: No --questions file provided; metadata will be empty.")

    agent_name = aggregated_path.stem  # e.g. "aggregated_factual"
    records: list[dict[str, Any]] = []

    for case_id, case in cases.items():
        claims = case.get("claims", [])
        q = question_lookup.get(case_id)
        if q is not None:
            question_text = q.get("question", "")
            options = q.get("options", {})
            correct_answer = q.get("answer")
            correct_answer_idx = q.get("answer_idx")
            meta_info = q.get("meta_info")
        else:
            question_text = case_id
            options = {}
            correct_answer = None
            correct_answer_idx = None
            meta_info = None

        records.append({
            "case_id": case_id,
            "agent_name": agent_name,
            "agent_model": "aggregated",
            "agent_context": question_text,
            "principal_context": question_text,
            "claims": claims,           # raw claim dicts — used for selection
            "dataset_type": "usmle",
            "options": options,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "meta_info": meta_info,
        })

    return records


def load_ground_truth(ground_truth_path: Path) -> list[dict[str, Any]]:
    """
    Load standard agent-results JSON (output of agent_inference.py).
    Converts the free-text 'information' field into a single-item claim list
    so the same selection pipeline can be applied.
    """
    records = json.loads(ground_truth_path.read_text())
    print(f"Loaded {len(records)} ground-truth records from {ground_truth_path}")
    out = []
    for rec in records:
        info = rec.get("information", "")
        if isinstance(info, dict):
            info = (info.get("answer") or info.get("recommendation")
                    or info.get("raw_response") or str(info))
        # Treat the whole text as one claim so the agent can choose to include/exclude it
        claims = [{"claim": info.strip(), "label": "factual"}] if info.strip() else []
        out.append({**rec, "claims": claims})
    return out


# ---------------------------------------------------------------------------
# Per-case selection
# ---------------------------------------------------------------------------

def run_selection_for_case(
    agent: Agent,
    record: dict[str, Any],
    include_options: bool,
) -> dict[str, Any]:
    """
    Run the selection agent on a single case.

    Returns a result dict compatible with principal_inference.py.
    """
    claims = record.get("claims", [])
    question = record.get("principal_context") or record.get("agent_context", "")
    options = record.get("options")

    # Format context shown to the agent
    context = question
    if include_options and options:
        context += "\n\nAnswer Options:\n"
        for key in sorted(options.keys()):
            context += f"{key}. {options[key]}\n"

    # Format the numbered claim list
    numbered_claims = format_claims_numbered(claims)

    # Fill prompt placeholders
    prompt = agent.prompt_template
    if "<CONTEXT>" in prompt:
        prompt = prompt.replace("<CONTEXT>", context)
    if "<CLAIMS>" in prompt:
        prompt = prompt.replace("<CLAIMS>", numbered_claims)

    # Call the agent
    messages = [{"role": "user", "content": prompt}]
    raw_response = agent._call_llm(messages)

    # Parse selected indices and render as standardized bullets
    selected_indices = parse_selected_indices(raw_response, n_available=len(claims))
    selected_claims = [claims[i - 1] for i in selected_indices if i - 1 < len(claims)]
    information = format_claims_bullets(selected_claims)

    return {
        "case_id": record["case_id"],
        "agent_name": agent.name,
        "agent_model": agent.model,
        "agent_context": question,
        "principal_context": question,
        # 'information' is standardized bullets of the selected claims — no agent language
        "information": information,
        "selected_indices": selected_indices,
        "n_available": len(claims),
        "n_selected": len(selected_claims),
        "raw_selection_response": raw_response,
        "ground_truth_agent": record.get("agent_name"),
        "ground_truth_model": record.get("agent_model"),
        "dataset_type": record.get("dataset_type", "usmle"),
        "options": record.get("options"),
        "correct_answer": record.get("correct_answer"),
        "correct_answer_idx": record.get("correct_answer_idx"),
        "meta_info": record.get("meta_info"),
        "agent_task": None,
        "agent_objective": None,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_selection_experiments(
    records: list[dict[str, Any]],
    agent: Agent,
    output_path: Path,
    include_options: bool = True,
    save_interval: int = 10,
    skip_existing: bool = True,
    max_workers: int = 8,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:

    if max_cases is not None and len(records) > max_cases:
        print(f"Limiting to {max_cases} cases (from {len(records)} total)")
        records = records[:max_cases]

    # Resume from partial results
    completed_ids: set[str] = set()
    results: list[dict[str, Any]] = []
    if skip_existing and output_path.exists():
        cached = json.loads(output_path.read_text())
        if isinstance(cached, list):
            results = cached
            completed_ids = {r["case_id"] for r in results}
            if len(completed_ids) >= len(records):
                print(f"Loading existing results from {output_path} ({len(results)} records)")
                return results
            print(f"Resuming from {len(completed_ids)}/{len(records)} completed cases")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pending = [r for r in records if r["case_id"] not in completed_ids]
    print(f"Running selection for {len(pending)} cases ...")

    new_results: list[dict[str, Any]] = []
    iteration_count = len(results)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(run_selection_for_case, agent, rec, include_options): rec["case_id"]
            for rec in pending
        }
        for future in tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Selecting"):
            case_id = future_to_id[future]
            try:
                res = future.result()
            except Exception as exc:
                print(f"  WARNING: case {case_id} failed: {exc}")
                continue

            new_results.append(res)
            iteration_count += 1

            if iteration_count % save_interval == 0:
                with open(output_path, "w") as f:
                    json.dump(results + new_results, f, indent=2)

    all_results = results + new_results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} results to {output_path}")
    return all_results


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def setup_client(
    backend: str,
    sglang_port: int = 30001,
    sglang_base_url: str = "http://127.0.0.1",
) -> NvidiaChatClient | OpenRouterChatClient | SGLangChatClient:
    if backend == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("Set NVIDIA_API_KEY environment variable")
        return NvidiaChatClient(api_key=api_key)
    elif backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif backend == "sglang":
        return SGLangChatClient(port=sglang_port, base_url=sglang_base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent claim-selection experiments (information design).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--aggregated-info", type=str, metavar="PATH",
        help="Path to aggregated_factual.json (claim pool source).",
    )
    source_group.add_argument(
        "--ground-truth", type=str, metavar="PATH",
        help="Path to agent_inference.py output JSON (standard ground-truth source).",
    )
    parser.add_argument(
        "--questions", type=str, default=None, metavar="PATH",
        help="Path to clinical questions JSON (required with --aggregated-info).",
    )
    parser.add_argument(
        "--agent-server", type=str, default="openrouter",
        choices=["nvidia", "openrouter", "sglang"],
    )
    parser.add_argument("--agent-sglang-port", type=int, default=30001)
    parser.add_argument("--agent-sglang-base-url", type=str, default="http://127.0.0.1")
    parser.add_argument(
        "--agent-model", type=str, default="meta-llama/llama-3.3-70b-instruct",
    )
    parser.add_argument(
        "--agent-prompt", type=str, default=DEFAULT_PROMPT_PATH,
        help="Path to the selection prompt YAML.",
    )
    parser.add_argument("--no-options", action="store_true",
                        help="Exclude answer options from the agent's context.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists.")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--max-cases", type=int, default=0, metavar="N",
        help="Maximum cases to process (0 = no limit).",
    )

    args = parser.parse_args()

    client = setup_client(
        backend=args.agent_server,
        sglang_port=args.agent_sglang_port,
        sglang_base_url=args.agent_sglang_base_url,
    )

    agent = Agent(
        name="selection_agent",
        client=client,
        model=args.agent_model,
        prompt_path=args.agent_prompt,
        temperature=1.0,
    )

    if args.aggregated_info:
        questions_path = Path(args.questions) if args.questions else None
        records = load_aggregated_claims(Path(args.aggregated_info), questions_path)
    else:
        records = load_ground_truth(Path(args.ground_truth))

    print(f"\nAgent prompt: {args.agent_prompt}")
    print(f"Agent model:  {args.agent_model}")
    print(f"Include options: {not args.no_options}")
    print("=" * 60)

    run_selection_experiments(
        records=records,
        agent=agent,
        output_path=Path(args.output),
        include_options=not args.no_options,
        save_interval=args.save_interval,
        skip_existing=not args.force,
        max_workers=args.max_workers,
        max_cases=args.max_cases if args.max_cases > 0 else None,
    )


if __name__ == "__main__":
    main()
