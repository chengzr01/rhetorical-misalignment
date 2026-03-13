#!/usr/bin/env python3
"""
Run agent presentation experiments where agents receive pre-generated information.
This script loads cached agent results as ground truth information and presents them
to agents using the framing prompt template, allowing fine-grained control over
what information is shown.

Two ground-truth sources are supported:
  --ground-truth PATH   Standard agent-results JSON (output of agent_inference.py or
                        information_aggregation.py).  Each record must have "case_id",
                        "information", and USMLE metadata fields.
  --aggregated-info PATH  Raw aggregated-claims JSON (output of experiments/aggregate_information.py,
                          i.e. aggregated_factual.json / aggregated_unfactual.json).  Requires
                          --questions to supply USMLE metadata.  Claims are joined as bullet
                          points before being passed to the framing agent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.agent import Agent
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


# ---------------------------------------------------------------------------
# Aggregated-claims loading helpers
# ---------------------------------------------------------------------------

def load_aggregated_claims(
    aggregated_path: Path,
    questions_path: Path | None,
    claim_format: str = "bullets",
) -> list[dict[str, Any]]:
    """
    Load an aggregated-claims JSON file (aggregated_factual.json /
    aggregated_unfactual.json) and return a list of records in the same
    schema as agent_inference.py output so they can be fed directly into
    run_presentation_experiments as ground truth.

    Args:
        aggregated_path: Path to the aggregated claims JSON.
        questions_path:  Path to the clinical questions JSON (for metadata).
                         If None, metadata fields will be empty.
        claim_format:    How to join claims: "bullets", "numbered", or "plain".

    Returns:
        List of dicts compatible with the ground-truth format expected by
        run_presentation_experiments.
    """
    aggregated = json.loads(aggregated_path.read_text())
    cases: dict[str, dict] = aggregated.get("cases", {})
    metadata = aggregated.get("metadata", {})
    keep_labels = metadata.get("keep_labels", ["?"])
    print(f"Loaded aggregated claims from {aggregated_path}")
    print(f"  Labels: {', '.join(keep_labels)}  |  Cases: {len(cases)}")

    # Build question lookup for metadata
    question_lookup: dict[str, dict] = {}
    if questions_path is not None:
        questions = json.loads(questions_path.read_text())
        question_lookup = {q["id"]: q for q in questions if "id" in q}
        print(f"  Loaded {len(question_lookup)} questions from {questions_path}")
    else:
        print("  WARNING: No --questions file provided; metadata (options/answer) will be empty.")

    # Derive a stable agent_name from the filename
    agent_name = aggregated_path.stem  # e.g. "aggregated_factual"

    records: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        claims = case.get("claims", [])
        information = _format_claims(claims, claim_format)

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
            "agent_model": "llama-aggregation",
            "agent_context": question_text,
            "principal_context": question_text,
            "information": information,
            "dataset_type": "usmle",
            "options": options,
            "correct_answer": correct_answer,
            "correct_answer_idx": correct_answer_idx,
            "meta_info": meta_info,
            "agent_task": None,
            "agent_objective": None,
        })

    return records


def _format_claims(claims: list[dict[str, Any]], fmt: str) -> str:
    """Join a list of claim dicts into a single information string."""
    lines: list[str] = []
    for i, c in enumerate(claims, start=1):
        text = c.get("claim", "").strip()
        if not text:
            continue
        if fmt == "numbered":
            lines.append(f"{i}. {text}")
        elif fmt == "bullets":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def setup_client(
    backend: str = "nvidia",
    sglang_port: int = 30000,
    sglang_base_url: str = "http://127.0.0.1",
) -> NvidiaChatClient | OpenRouterChatClient | SGLangChatClient:
    """Setup API client based on backend type."""
    if backend == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("Set NVIDIA_API_KEY environment variable")
        return NvidiaChatClient(api_key=api_key)
    elif backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=sglang_port, base_url=sglang_base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def extract_information_content(information: Any, content_filter: str = "full") -> str:
    """
    Extract and filter information content from cached agent results.

    Args:
        information: The information field from agent result
        content_filter: What content to include:
            - "full": Complete information (default)
            - "key_points": Extract only key clinical points
            - "diagnosis_only": Only diagnostic reasoning
            - "treatment_only": Only treatment recommendations
            - "facts_only": Only objective facts, no reasoning

    Returns:
        Filtered information string
    """
    if isinstance(information, dict):
        info_str = (information.get("answer") or
                   information.get("recommendation") or
                   information.get("raw_response") or
                   str(information))
    elif isinstance(information, str):
        info_str = information
    else:
        info_str = str(information)

    # For now, return full information
    # TODO: Implement more sophisticated filtering based on content_filter
    if content_filter == "full":
        return info_str
    else:
        # Simple filtering - can be enhanced with NLP techniques
        return info_str


def format_usmle_context_with_options(
    question: str,
    options: dict[str, str] | None = None,
) -> str:
    """
    Format USMLE question with options.

    Args:
        question: The clinical question
        options: Dictionary of answer options

    Returns:
        Formatted context string
    """
    formatted = question

    if options:
        formatted += "\n\nAnswer Options:\n"
        for key in sorted(options.keys()):
            formatted += f"{key}. {options[key]}\n"

    return formatted


def run_agent_with_information(
    agent: Agent,
    context: str,
    information: str,
    question: str,
    include_options: bool = True,
) -> Mapping[str, Any]:
    """
    Run agent with pre-provided information using framing template.

    Args:
        agent: Agent instance with framing prompt template
        context: Clinical question context (may include options for agent)
        information: Pre-generated information to provide
        question: Raw question text only (for principal_context)
        include_options: Whether to include answer options in context

    Returns:
        Agent response with recommendation
    """
    # The framing template should have both <CONTEXT> and <INFORMATION> placeholders
    prompt = agent.prompt_template

    # Replace placeholders
    if "<CONTEXT>" in prompt:
        prompt = prompt.replace("<CONTEXT>", context)
    if "<INFORMATION>" in prompt:
        prompt = prompt.replace("<INFORMATION>", information)

    # Call the LLM
    messages = [{"role": "user", "content": prompt}]
    response = agent._call_llm(messages)

    return {
        "agent_name": agent.name,
        "agent_model": agent.model,
        "agent_context": question,  # Just the question (matches agent_inference.py)
        "principal_context": question,  # Just the question (matches agent_inference.py)
        "provided_information": information,
        "information": response,  # Agent's response (matches agent_inference.py)
    }


def run_presentation_experiments(
    ground_truth_path: Path | None,
    agent_configs: list[Mapping[str, Any]],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    content_filter: str = "full",
    include_options: bool = True,
    save_interval: int = 10,
    skip_existing: bool = True,
    max_workers: int = 8,
    ground_truth_data: list[dict] | None = None,
    max_cases: int | None = 100,
) -> list[Mapping[str, Any]]:
    """
    Run agent presentation experiments using ground truth information.

    Args:
        ground_truth_path: Path to cached agent results (ground truth).
                           Ignored when ground_truth_data is supplied directly.
        agent_configs: List of agent configurations with framing templates
        agent_client: Client for agent API
        output_path: Path to save experiment results
        content_filter: What content to include from ground truth
        include_options: Whether to include answer options
        save_interval: Save results every N iterations
        skip_existing: If True and output exists, load existing results
        max_workers: Number of parallel workers
        ground_truth_data: Pre-loaded list of ground-truth records. When provided,
                           ground_truth_path is not read from disk (used for
                           --aggregated-info mode).
        max_cases: Maximum number of ground-truth cases to process. None means
                   no limit. Default is 100.

    Returns:
        List of experiment results
    """
    # Load ground truth information
    if ground_truth_data is None:
        with open(ground_truth_path, "r") as f:
            ground_truth_data = json.load(f)
        print(f"Loaded {len(ground_truth_data)} ground truth samples from {ground_truth_path}")
    else:
        print(f"Using {len(ground_truth_data)} pre-loaded ground truth records")

    if max_cases is not None and len(ground_truth_data) > max_cases:
        print(f"Limiting to {max_cases} cases (from {len(ground_truth_data)} total)")
        ground_truth_data = ground_truth_data[:max_cases]

    expected_total = len(ground_truth_data) * len(agent_configs)

    # Check for existing results
    if skip_existing and output_path.exists():
        with open(output_path, "r") as f:
            cached_results = json.load(f)
        if isinstance(cached_results, list) and len(cached_results) >= expected_total:
            print(f"Loading existing results from {output_path} ({len(cached_results)}/{expected_total} results)")
            return cached_results
        else:
            print(f"Found partial results at {output_path} ({len(cached_results)}/{expected_total}). Continuing...")
            results = cached_results
            completed_pairs = set()
            for r in results:
                key = (
                    r.get("case_id"),
                    r.get("agent_name"),
                    r.get("agent_model"),
                )
                completed_pairs.add(key)
    else:
        results = []
        completed_pairs = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize agents
    agents = [
        Agent(
            name=config["name"],
            client=agent_client,
            model=config["model"],
            prompt_path=config["prompt_path"],
            temperature=config.get("temperature", 1.0),
        )
        for config in agent_configs
    ]

    # Build tasks
    tasks = []
    for gt_sample in ground_truth_data:
        case_id = gt_sample.get("case_id")
        question = gt_sample.get("principal_context") or gt_sample.get("agent_context", "")
        options = gt_sample.get("options")
        information = gt_sample.get("information", "")

        # Extract and filter information
        filtered_info = extract_information_content(information, content_filter)

        # Format context
        if include_options and options:
            context = format_usmle_context_with_options(question, options)
        else:
            context = question

        for agent in agents:
            pair_key = (case_id, agent.name, agent.model)
            if pair_key in completed_pairs:
                continue

            tasks.append((
                agent,
                context,
                filtered_info,
                question,
                include_options,
                case_id,
                gt_sample,
                pair_key,
            ))

    print(f"Running {len(tasks)} presentation experiments...")

    # Run experiments in parallel
    def one_presentation(agent, context, information, question, include_options, case_id, gt_sample, pair_key):
        result = run_agent_with_information(
            agent=agent,
            context=context,
            information=information,
            question=question,
            include_options=include_options,
        )

        # Add metadata for principal_inference compatibility
        result["case_id"] = case_id
        result["dataset_type"] = gt_sample.get("dataset_type", "usmle")
        result["options"] = gt_sample.get("options")
        result["correct_answer"] = gt_sample.get("correct_answer")
        result["correct_answer_idx"] = gt_sample.get("correct_answer_idx")
        result["meta_info"] = gt_sample.get("meta_info")
        result["ground_truth_agent"] = gt_sample.get("agent_name")
        result["ground_truth_model"] = gt_sample.get("agent_model")
        # Add fields expected by principal_inference.py
        result["agent_task"] = None
        result["agent_objective"] = None
        result["_pair_key"] = pair_key

        return result

    new_results = []
    iteration_count = len(results)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for task in tasks:
            future = executor.submit(one_presentation, *task)
            future_to_task[future] = task

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing presentations"):
            res = future.result()
            new_results.append(res)
            completed_pairs.add(res["_pair_key"])
            iteration_count += 1

            # Save at intervals
            if iteration_count % save_interval == 0:
                tmp_list = results + [dict(r) for r in new_results]
                for r in tmp_list:
                    if "_pair_key" in r:
                        del r["_pair_key"]
                with open(output_path, "w") as f:
                    json.dump(tmp_list, f, indent=2)

    # Final save
    all_final = results + [dict(r) for r in new_results]
    for r in all_final:
        if "_pair_key" in r:
            del r["_pair_key"]

    with open(output_path, "w") as f:
        json.dump(all_final, f, indent=2)

    print(f"Saved {len(all_final)} presentation results to {output_path}")
    return all_final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent presentation experiments with controlled information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Ground-truth source: exactly one of --ground-truth or --aggregated-info is required.
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--ground-truth",
        type=str,
        metavar="PATH",
        help=(
            "Path to cached agent results JSON (output of agent_inference.py or "
            "information_aggregation.py).  Each record must have 'case_id', "
            "'information', and USMLE metadata fields."
        ),
    )
    source_group.add_argument(
        "--aggregated-info",
        type=str,
        metavar="PATH",
        help=(
            "Path to a raw aggregated-claims JSON file "
            "(e.g. experiments/aggregation/aggregated_factual.json). "
            "Claims are joined as bullet points and used as the information "
            "fed to the framing agent.  Use --questions to supply USMLE metadata."
        ),
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the clinical questions JSON file "
            "(e.g. experiments/questions/clinical_questions_usmle_sample.json). "
            "Required when using --aggregated-info to supply options / answer metadata."
        ),
    )
    parser.add_argument(
        "--claim-format",
        type=str,
        default="bullets",
        choices=["bullets", "numbered", "plain"],
        help="How to join multiple claims into the information string when using --aggregated-info (default: bullets)",
    )
    parser.add_argument(
        "--agent-server",
        type=str,
        default="sglang",
        choices=["nvidia", "openrouter", "sglang"],
        help="Server backend for agents",
    )
    parser.add_argument(
        "--agent-sglang-port",
        type=int,
        default=30001,
        help="Port for agent SGLang server",
    )
    parser.add_argument(
        "--agent-sglang-base-url",
        type=str,
        default="http://127.0.0.1",
        help="Base URL for agent SGLang server",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default="meta/llama-3.3-70b-instruct",
        help="Model to use for agents",
    )
    parser.add_argument(
        "--agent-prompt",
        type=str,
        default="prompts/agent/framing_usmle.yaml",
        help="Agent framing prompt template path"
    )
    parser.add_argument(
        "--content-filter",
        type=str,
        default="full",
        choices=["full", "key_points", "diagnosis_only", "treatment_only", "facts_only"],
        help="What content to include from ground truth information"
    )
    parser.add_argument(
        "--no-options",
        action="store_true",
        help="Exclude answer options from context"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for presentation experiment results"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip experiments if cache exists (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-run experiments even if cache exists"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save results every N iterations",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of cases to process (default: 100; use 0 for no limit)",
    )

    args = parser.parse_args()

    # Setup client
    agent_client = setup_client(
        backend=args.agent_server,
        sglang_port=args.agent_sglang_port,
        sglang_base_url=args.agent_sglang_base_url,
    )

    # Configure agents
    agent_configs = [{
        "name": "framing_agent",
        "model": args.agent_model,
        "prompt_path": args.agent_prompt,
        "temperature": 1.0,
    }]

    # Resolve ground-truth source
    if args.aggregated_info:
        # Load directly from aggregated claims JSON, converting on the fly
        questions_path = Path(args.questions) if args.questions else None
        ground_truth_data = load_aggregated_claims(
            aggregated_path=Path(args.aggregated_info),
            questions_path=questions_path,
            claim_format=args.claim_format,
        )
        # Write to a temp-style path so run_presentation_experiments can use the
        # existing file-based interface; we pass it as a pre-loaded list instead.
        ground_truth_path = None
        print(f"Aggregated-info source: {args.aggregated_info} ({len(ground_truth_data)} cases)")
    else:
        ground_truth_path = Path(args.ground_truth)
        ground_truth_data = None
        print(f"Ground truth: {args.ground_truth}")

    print(f"Agent prompt: {args.agent_prompt}")
    print(f"Content filter: {args.content_filter}")
    print(f"Include options: {not args.no_options}")

    print("\n" + "="*60)
    print("AGENT PRESENTATION EXPERIMENTS")
    print("="*60)

    results = run_presentation_experiments(
        ground_truth_path=ground_truth_path,
        ground_truth_data=ground_truth_data,
        agent_configs=agent_configs,
        agent_client=agent_client,
        output_path=Path(args.output),
        content_filter=args.content_filter,
        include_options=not args.no_options,
        save_interval=args.save_interval,
        skip_existing=not args.force,
        max_workers=args.max_workers,
        max_cases=args.max_cases if args.max_cases > 0 else None,
    )

    print(f"\nPresentation experiments complete: {len(results)} results")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
