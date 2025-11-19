#!/usr/bin/env python3
"""
Run principal inferences using cached agent results.
This script handles STAGE 2: Principal Inferences only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.principal import Principal
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


def format_usmle_context_for_principal(agent_result: Mapping[str, Any]) -> str:
    """
    Format USMLE question context for principal evaluation.
    Includes the question and agent's answer recommendation.
    """
    principal_context = agent_result.get("principal_context", "")
    options = agent_result.get("options", {})

    formatted = f"{principal_context}\n\n"
    formatted += "Answer Options:\n"
    for key in sorted(options.keys()):
        formatted += f"{key}. {options[key]}\n"

    return formatted


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


def run_principal_decision(
    principal: Principal,
    principal_context: str,
    agent_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Run principal decision using cached agent results.
    Handles both MIMIC and USMLE dataset types.
    """
    dataset_type = agent_result.get("dataset_type", "mimic")
    info = agent_result.get("information")

    if isinstance(info, dict):
        recommendation = (info.get("answer") or
                          info.get("recommendation") or
                          info.get("raw_response") or
                          str(info))
    elif isinstance(info, str):
        recommendation = info
    else:
        recommendation = str(info)

    # For USMLE, format the context to include options
    if dataset_type == "usmle":
        pc = format_usmle_context_for_principal(agent_result)
        # Add agent's recommendation
        pc += f"\n\nAgent's Recommendation:\n{recommendation.strip() if recommendation else ''}"
    else:
        # MIMIC format: use principal_context with <RECOMMENDATIONS> placeholder
        pc = principal_context
        if pc and "<RECOMMENDATIONS>" in pc:
            pc = pc.replace("<RECOMMENDATIONS>", recommendation.strip() if recommendation else "")

    decision_result = principal.act(context=pc, information=agent_result["information"])

    result = {
        "agent_name": agent_result["agent_name"],
        "agent_model": agent_result["agent_model"],
        "principal_name": principal.name,
        "principal_model": principal.model,
        "principal_context": pc,
        "agent_context": agent_result["agent_context"],
        "agent_task": agent_result.get("agent_task"),
        "agent_objective": agent_result.get("agent_objective"),
        "information": agent_result["information"],
        "decision": decision_result["decision"],
        "belief": decision_result["belief"],
        "reasoning": decision_result["reasoning"],
        "raw_principal_response": decision_result["raw_response"],
        "dataset_type": dataset_type,
    }

    # Add dataset-specific metadata
    if dataset_type == "usmle":
        result["options"] = agent_result.get("options")
        result["correct_answer"] = agent_result.get("correct_answer")
        result["correct_answer_idx"] = agent_result.get("correct_answer_idx")
        result["meta_info"] = agent_result.get("meta_info")

    return result


def run_principal_inferences(
    agent_results: list[Mapping[str, Any]],
    principal_configs: list[Mapping[str, Any]],
    principal_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
    skip_existing: bool = True,
    max_workers: int = 8,
) -> dict[str, list[Mapping[str, Any]]]:
    """
    Run principal inferences using cached agent results, in parallel.
    Saves each principal type to a separate output file.

    Args:
        agent_results: List of cached agent results
        principal_configs: List of principal configurations
        principal_client: Client for principal API
        output_path: Base path for output files (will be modified per principal type)
        save_interval: Save results every N iterations
        skip_existing: If True and output files exist, load and return existing results
        max_workers: Number of parallel workers

    Returns:
        Dictionary mapping principal names to their results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check which principals need to be run
    principals_to_run = []
    all_existing_results = {}

    for config in principal_configs:
        principal_name = config["name"]
        # Extract principal type from name (e.g., "bayesian_principal" -> "bayesian")
        principal_type = principal_name.replace("_principal", "")

        # Generate output path for this principal type
        output_base = output_path.stem
        output_dir = output_path.parent
        output_ext = output_path.suffix

        # Remove any existing principal type suffix from the base name
        for ptype in ["bayesian", "behavioral", "anchoring", "availability",
                      "confirmation", "conservatism", "overconfidence", "prospect", "all"]:
            if output_base.endswith(f"_{ptype}"):
                output_base = output_base[:-len(f"_{ptype}")]
                break

        principal_output_path = output_dir / f"{output_base}_{principal_type}{output_ext}"
        config["output_path"] = principal_output_path

        expected_count = len(agent_results)

        # Check if this principal's results already exist
        if skip_existing and principal_output_path.exists():
            with open(principal_output_path, "r") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list) and len(existing_results) >= expected_count:
                print(f"✓ Loading existing {principal_type} results from {principal_output_path} ({len(existing_results)}/{expected_count} results)")
                all_existing_results[principal_name] = existing_results
                continue
            else:
                print(f"⚠ {principal_type.capitalize()} results incomplete ({len(existing_results)}/{expected_count}), will re-run")

        principals_to_run.append(config)

    # If all principals already have complete results, return them
    if not principals_to_run:
        print(f"\nAll principal inferences already complete - skipping")
        return all_existing_results

    # Create Principal objects only for principals that need to run
    principals = [
        Principal(
            name=config["name"],
            client=principal_client,
            model=config["model"],
            prompt_path=config["prompt_path"],
            temperature=config.get("temperature", 1.0),
        )
        for config in principals_to_run
    ]

    # Map principal objects to their configs for later use
    principal_to_config = {p.name: config for p, config in zip(principals, principals_to_run)}

    # Group agent results by case_id to process all principals for each case together
    case_to_agent_results = {}
    for agent_result in agent_results:
        case_id = agent_result.get("case_id")
        if case_id not in case_to_agent_results:
            case_to_agent_results[case_id] = []
        case_to_agent_results[case_id].append(agent_result)

    total_iterations = len(agent_results) * len(principals)
    progress_bar = tqdm(total=total_iterations, desc="Running principal inferences")

    # Build tasks ordered by case: for each case, run all principals
    tasks = []
    for case_id in sorted(case_to_agent_results.keys()):
        for agent_result in case_to_agent_results[case_id]:
            for principal in principals:
                tasks.append((agent_result, principal))

    def run_one_principal_decision(agent_result, principal):
        principal_context = agent_result.get("principal_context", agent_result["agent_context"])
        result = run_principal_decision(
            principal=principal,
            principal_context=principal_context,
            agent_result=agent_result,
        )
        # Add case ID
        if "case_id" in agent_result:
            result["case_id"] = agent_result["case_id"]

        # Add dataset-specific metadata (MIMIC only, USMLE handled in run_principal_decision)
        dataset_type = agent_result.get("dataset_type", "mimic")
        if dataset_type == "mimic":
            result["hadm_id"] = agent_result.get("hadm_id")
            result["subject_id"] = agent_result.get("subject_id")
            result["ground_truth"] = agent_result.get("ground_truth")

        return result

    # Group results by principal name
    results_by_principal = {p.name: [] for p in principals}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for agent_result, principal in tasks:
            futures.append(executor.submit(run_one_principal_decision, agent_result, principal))

        iteration_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing principal results"):
            result = future.result()
            principal_name = result["principal_name"]
            results_by_principal[principal_name].append(result)
            iteration_count += 1
            progress_bar.update(1)

            # Save at intervals - save each principal type separately
            if save_interval > 0 and iteration_count % save_interval == 0:
                for principal_name, results in results_by_principal.items():
                    if results:  # Only save if there are results
                        output_file = principal_to_config[principal_name]["output_path"]
                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=2)
        progress_bar.close()

    # Final save - save each principal type to its own file
    for principal_name, results in results_by_principal.items():
        output_file = principal_to_config[principal_name]["output_path"]
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} {principal_name.replace('_principal', '')} results to {output_file}")

    # Combine with existing results
    all_results = {**all_existing_results, **results_by_principal}
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run principal inferences using cached agent results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--principal-server",
        type=str,
        default="nvidia",
        choices=["nvidia", "openrouter", "sglang"],
        help="Server backend for principals",
    )
    parser.add_argument(
        "--principal-sglang-port",
        type=int,
        default=30000,
        help="Port for principal SGLang server",
    )
    parser.add_argument(
        "--principal-sglang-base-url",
        type=str,
        default="http://127.0.0.1",
        help="Base URL for principal SGLang server",
    )
    parser.add_argument(
        "--principal-model",
        type=str,
        default="deepseek-ai/deepseek-v3.1",
        help="Model to use for principals",
    )
    parser.add_argument(
        "--agent-cache",
        type=str,
        default="experiments/cache/agent_results.json",
        help="Path to cached agent inference results"
    )
    parser.add_argument(
        "--principal-types",
        type=str,
        nargs="+",
        default=["bayesian"],
        choices=["all", "bayesian", "behavioral", "anchoring", "availability", "confirmation", "conservatism", "overconfidence", "prospect"],
        help="Principal types to use. Use 'all' to run all types except bayesian, or specify one or more types"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/output/results.json",
        help="Output path for final results"
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
        help="Number of parallel workers for principal inferences"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip principal inferences if output exists (default: True)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-run principal inferences even if output exists"
    )
    args = parser.parse_args()

    # Setup client for principals
    principal_client = setup_client(
        backend=args.principal_server,
        sglang_port=args.principal_sglang_port,
        sglang_base_url=args.principal_sglang_base_url,
    )

    # Load cached agent results
    agent_cache_path = Path(args.agent_cache)
    if not agent_cache_path.exists():
        raise FileNotFoundError(
            f"Agent cache not found at {agent_cache_path}. "
            f"Please run agent_inference.py first to generate agent results."
        )

    print(f"Loading agent results from {agent_cache_path}...")
    with open(agent_cache_path, "r") as f:
        agent_results = json.load(f)

    if not isinstance(agent_results, list):
        raise ValueError(
            f"Invalid agent cache format. Expected a list, but got {type(agent_results).__name__}"
        )

    print(f"Loaded {len(agent_results)} agent results")

    # Configure principal types
    principal_types = args.principal_types
    all_available_principal_types = ["behavioral", "anchoring", "availability", "confirmation",
                                    "conservatism", "overconfidence", "prospect"]
    if "all" in principal_types:
        principal_types = all_available_principal_types

    principal_configs = []
    for ptype in principal_types:
        principal_configs.append({
            "name": f"{ptype}_principal",
            "model": args.principal_model,
            "prompt_path": f"prompts/principal/{ptype}.yaml",
            "temperature": 1.0,
        })

    print(f"Principal types: {', '.join(principal_types)}")
    print(f"Principal configs: {len(principal_configs)}")
    print(f"Total principal inferences: {len(agent_results) * len(principal_configs)}")

    output_path = Path(args.output)

    print("\n" + "="*60)
    print("PRINCIPAL INFERENCE")
    print("="*60)

    final_results = run_principal_inferences(
        agent_results=agent_results,
        principal_configs=principal_configs,
        principal_client=principal_client,
        output_path=output_path,
        save_interval=args.save_interval,
        skip_existing=not args.force,
        max_workers=args.max_workers,
    )

    print(f"\nPrincipal inferences complete:")
    total_results = 0
    for principal_name, results in final_results.items():
        principal_type = principal_name.replace("_principal", "")
        print(f"  - {principal_type}: {len(results)} results")
        total_results += len(results)
    print(f"Total: {total_results} results across {len(final_results)} principal types")


if __name__ == "__main__":
    main()
