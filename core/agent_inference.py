#!/usr/bin/env python3
"""
Run agent inferences on clinical questions and cache results.
This script handles STAGE 1: Agent Inferences only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.agent import Agent
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


def detect_dataset_type(item: dict) -> str:
    """
    Detect dataset type based on the structure of the item.

    Returns:
        "mimic" for MIMIC-IV dataset (has context, ground_truth)
        "usmle" for USMLE dataset (has options, answer, answer_idx)
    """
    if "options" in item and "answer_idx" in item:
        return "usmle"
    elif "context" in item and "ground_truth" in item:
        return "mimic"
    else:
        # Default to mimic if unclear
        return "mimic"


def build_clinical_question_contexts(questions: list[dict]) -> list[dict]:
    """
    Build contexts from clinical questions data.
    Handles both MIMIC-IV and USMLE dataset formats.
    """
    contexts = []
    for item in questions:
        dataset_type = detect_dataset_type(item)

        context_dict = {
            "id": item.get("id"),
            "dataset_type": dataset_type,
        }

        if dataset_type == "mimic":
            # MIMIC-IV format
            context_dict.update({
                "hadm_id": item.get("hadm_id"),
                "subject_id": item.get("subject_id"),
                "context": item["context"],
                "question": item["question"],
                "ground_truth": item.get("ground_truth", {}),
            })
        elif dataset_type == "usmle":
            # USMLE format
            context_dict.update({
                "question": item["question"],
                "options": item.get("options", {}),
                "answer": item.get("answer"),
                "answer_idx": item.get("answer_idx"),
                "meta_info": item.get("meta_info"),
                "metamap_phrases": item.get("metamap_phrases", []),
            })

        contexts.append(context_dict)
    return contexts


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


def run_agent_inference(
    agent: Agent,
    agent_context: str,
    agent_task: str | None = None,
    agent_objective: str | None = None,
) -> Mapping[str, Any]:
    """
    Run agent inference only.
    """
    information = agent.act(context=agent_context, task=agent_task, objective=agent_objective)
    return {
        "agent_name": agent.name,
        "agent_model": agent.model,
        "agent_context": agent_context,
        "agent_task": agent_task,
        "agent_objective": agent_objective,
        "information": information,
    }


def run_agent_inferences(
    contexts: list[dict],
    agent_configs: list[Mapping[str, Any]],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
    skip_existing: bool = True,
    max_workers: int = 8,
    max_cases: int | None = 100,
) -> list[Mapping[str, Any]]:
    """
    Run agent inferences on all contexts and save results, in parallel (concurrent).

    Args:
        contexts: List of context dictionaries
        agent_configs: List of agent configurations
        agent_client: Client for agent API
        output_path: Path to save agent results
        save_interval: Save results every N iterations
        skip_existing: If True and output_path exists, load and return existing results
        max_workers: Number of parallel workers
        max_cases: Maximum number of contexts to process. None means no limit.
                   Default is 100.

    Returns:
        List of agent results
    """
    if max_cases is not None and len(contexts) > max_cases:
        print(f"Limiting to {max_cases} cases (from {len(contexts)} total)")
        contexts = contexts[:max_cases]

    expected_total = len(contexts) * len(agent_configs)

    if skip_existing and output_path.exists():
        with open(output_path, "r") as f:
            cached_results = json.load(f)
        if isinstance(cached_results, list) and len(cached_results) >= expected_total:
            print(f"Loading existing agent results from {output_path} ({len(cached_results)}/{expected_total} results)")
            return cached_results
        else:
            print(f"Agent cache found at {output_path} ({len(cached_results)}/{expected_total} results). Continuing incomplete inferences...")

        results = cached_results
        completed_pairs = set()
        for r in results:
            key = (
                r.get("case_id"),
                r.get("agent_name"),
                r.get("agent_model"),
                r.get("agent_context"),
            )
            completed_pairs.add(key)
    else:
        results = []
        completed_pairs = set()

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    total_iterations = len(contexts) * len(agents)
    already_done = len(results)
    progress_bar = tqdm(
        total=total_iterations,
        initial=already_done,
        desc="Running agent inferences",
    )
    iteration_count = already_done

    # Build tasks: (context, agent)
    tasks = []
    for context in contexts:
        for agent in agents:
            # Compose context unique key for skipping
            if isinstance(context, dict) and "id" in context:
                context_id = context["id"]
            else:
                context_id = hash(str(context))

            if isinstance(context, str):
                agent_context = context
                agent_task = agent_objective = None
            else:
                dataset_type = context.get("dataset_type", "mimic")
                if dataset_type == "usmle":
                    # USMLE: only provide the question, not the options
                    agent_context = context["question"]
                else:
                    # MIMIC: context + detailed question
                    agent_context = f"{context['context']}\n\n{context['question']}"
                agent_task = context.get("agent_task")
                agent_objective = context.get("agent_objective")
            pair_key = (
                context_id,
                agent.name,
                agent.model,
                agent_context,
            )
            if pair_key in completed_pairs:
                continue
            tasks.append((context, agent, agent_context, agent_task, agent_objective, pair_key))

    # Run in parallel
    def one_agent_inference(context, agent, agent_context, agent_task, agent_objective, pair_key):
        result = run_agent_inference(
            agent=agent,
            agent_context=agent_context,
            agent_task=agent_task,
            agent_objective=agent_objective,
        )
        # Add metadata
        if isinstance(context, dict) and "id" in context:
            result["case_id"] = context["id"]
            result["dataset_type"] = context.get("dataset_type", "mimic")

            dataset_type = context.get("dataset_type", "mimic")
            if dataset_type == "mimic":
                result["hadm_id"] = context.get("hadm_id")
                result["subject_id"] = context.get("subject_id")
                result["principal_context"] = context["context"]
                result["ground_truth"] = context.get("ground_truth")
            elif dataset_type == "usmle":
                result["options"] = context.get("options")
                result["correct_answer"] = context.get("answer")
                result["correct_answer_idx"] = context.get("answer_idx")
                result["meta_info"] = context.get("meta_info")
                # For USMLE, principal_context can be the question without options
                result["principal_context"] = context.get("question")
        result["_pair_key"] = pair_key  # for tracking in results
        return result

    new_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for task in tasks:
            future = executor.submit(one_agent_inference, *task)
            future_to_task[future] = task
        for idx, future in enumerate(tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing agent results")):
            res = future.result()
            new_results.append(res)
            completed_pairs.add(res["_pair_key"])
            iteration_count += 1
            progress_bar.update(1)
            # Save at intervals (append-only semantics—existing results are kept in 'results')
            if iteration_count % save_interval == 0:
                tmp_list = results + [dict(r) for r in new_results]
                for r in tmp_list:
                    if "_pair_key" in r:
                        del r["_pair_key"]
                with open(output_path, "w") as f:
                    json.dump(tmp_list, f, indent=2)
        progress_bar.close()

    # Final save
    # Remove helper fields
    all_final = results + [dict(r) for r in new_results]
    for r in all_final:
        if "_pair_key" in r:
            del r["_pair_key"]

    with open(output_path, "w") as f:
        json.dump(all_final, f, indent=2)

    print(f"Saved {len(all_final)} agent results to {output_path}")
    return all_final


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent inferences on clinical questions and cache results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--input",
        type=str,
        default="experiments/questions/clinical_questions.json",
        help="Input clinical questions JSON file"
    )
    parser.add_argument(
        "--agent-prompt",
        type=str,
        default=None,
        help="Agent prompt template path (auto-detected based on dataset type if not specified)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default=None,
        choices=["mimic", "usmle"],
        help="Dataset type (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/agents/agent_results.json",
        help="Output path for agent results cache"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip agent inferences if cache exists (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-run agent inferences even if cache exists"
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
        help="Number of parallel workers for agent inferences"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of cases to process (default: 100; use 0 for no limit)",
    )
    args = parser.parse_args()

    # Setup client for agents
    agent_client = setup_client(
        backend=args.agent_server,
        sglang_port=args.agent_sglang_port,
        sglang_base_url=args.agent_sglang_base_url,
    )

    # Load input data
    with open(args.input, "r") as f:
        input_data = json.load(f)

    if not isinstance(input_data, list):
        raise ValueError(
            f"Invalid input format. Expected a list of clinical question dictionaries, "
            f"but got {type(input_data).__name__}"
        )

    print(f"Loading {len(input_data)} clinical questions")
    contexts = build_clinical_question_contexts(input_data)
    print(f"Built {len(contexts)} contexts")

    # Auto-detect dataset type if not specified
    if args.dataset_type:
        detected_type = args.dataset_type
        print(f"Using specified dataset type: {detected_type}")
    else:
        # Detect from first item
        if contexts:
            detected_type = contexts[0].get("dataset_type", "mimic")
            print(f"Auto-detected dataset type: {detected_type}")
        else:
            detected_type = "mimic"
            print("No contexts found, defaulting to MIMIC dataset type")

    # Auto-select prompt based on dataset type if not specified
    if args.agent_prompt:
        prompt_path = args.agent_prompt
    else:
        if detected_type == "usmle":
            prompt_path = "prompts/agent/default_usmle.yaml"
        else:
            prompt_path = "prompts/agent/default_mimiciv_demo.yaml"
        print(f"Auto-selected prompt: {prompt_path}")

    agent_configs = [{
        "name": "default_agent",
        "model": args.agent_model,
        "prompt_path": prompt_path,
        "temperature": 1.0,
    }]

    print(f"Dataset type: {detected_type}")

    print(f"Agent configs: {len(agent_configs)}")
    print(f"Total contexts: {len(contexts)}")
    print(f"Total agent inferences: {len(contexts) * len(agent_configs)}")

    output_path = Path(args.output)

    print("\n" + "="*60)
    print("AGENT INFERENCE")
    print("="*60)

    agent_results = run_agent_inferences(
        contexts=contexts,
        agent_configs=agent_configs,
        agent_client=agent_client,
        output_path=output_path,
        save_interval=args.save_interval,
        skip_existing=not args.force,
        max_workers=args.max_workers,
        max_cases=args.max_cases if args.max_cases > 0 else None,
    )

    print(f"\nAgent inferences complete: {len(agent_results)} results")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
