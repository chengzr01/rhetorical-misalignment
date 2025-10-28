from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor

from agents.agent import Agent
from agents.principal import Principal
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


def build_mimic_context(hypothesis_data: dict) -> dict:
    """
    Build context from MIMIC hypothesis data.

    MIMIC hypotheses contain rich medical history data including medications and labs.
    """
    # Extract demographics with enhanced info
    demographics = (
        f"Patient ID: {hypothesis_data['patient_id']}\n"
        f"Sex: {hypothesis_data['sex']}, Age: {hypothesis_data['age']} years\n"
        f"Race: {hypothesis_data['race']}\n"
        f"Marital Status: {hypothesis_data['marital_status']}\n"
        f"Language: {hypothesis_data['language']}\n"
        f"Insurance: {hypothesis_data['insurance']}\n\n"
        f"Admission Type: {hypothesis_data['admission_type']}\n"
        f"Admitted From: {hypothesis_data['admission_location']}"
    )

    # Build data report from medication and lab history
    data_report = (
        f"Medication History:\n{hypothesis_data['medication_history']}\n\n"
        f"Lab Results:\n{hypothesis_data['lab_history']}"
    )

    return {
        "patient_id": hypothesis_data['patient_id'],
        "hadm_id": hypothesis_data['hadm_id'],
        "sex": hypothesis_data['sex'],
        "age": hypothesis_data['age'],
        "demographics": demographics,
        "hypothesis": hypothesis_data['hypothesis'],
        "data_report": data_report,
        "source": "mimic",
    }


def build_pmc_context(hypothesis_data: dict) -> dict:
    """
    Build context from PMC hypothesis data.

    PMC hypotheses are based on published case reports with patient summaries
    and supporting research abstracts.
    """
    # Extract demographics
    demographics = (
        f"Patient ID: {hypothesis_data['patient_id']}\n"
        f"Gender: {hypothesis_data['gender']}, Age: {hypothesis_data['age']}"
    )

    # Build data report from patient summary and abstracts
    pmids_str = ", ".join(hypothesis_data['pmids'])
    data_report = (
        f"Patient Summary:\n{hypothesis_data['patient_summary']}\n\n"
        f"Supporting Research Evidence (PMIDs: {pmids_str}):\n"
    )

    # Add paper abstracts
    for i, (pmid, abstract) in enumerate(zip(hypothesis_data['pmids'], hypothesis_data['paper_abstracts']), 1):
        data_report += f"\nPaper {i} (PMID: {pmid}):\n{abstract}\n"

    return {
        "patient_id": hypothesis_data['patient_id'],
        "patient_uid": hypothesis_data['patient_uid'],
        "gender": hypothesis_data['gender'],
        "age": hypothesis_data['age'],
        "demographics": demographics,
        "hypothesis": hypothesis_data['hypothesis'],
        "data_report": data_report,
        "source": "pmc",
    }


def build_contexts(hypotheses: list[dict], hypothesis_type: str) -> list[dict]:
    """
    Build contexts from raw hypothesis data based on hypothesis type.

    Args:
        hypotheses: List of hypothesis dictionaries
        hypothesis_type: Either 'mimic' or 'pmc'

    Returns:
        List of context dictionaries formatted for the simulation
    """
    if hypothesis_type == "mimic":
        return [build_mimic_context(h) for h in hypotheses]
    elif hypothesis_type == "pmc":
        return [build_pmc_context(h) for h in hypotheses]
    else:
        raise ValueError(f"Unknown hypothesis type: {hypothesis_type}")


def setup_clients(
    backend: str = "nvidia",
    sglang_port: int = 30000,
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
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=sglang_port, base_url=sglang_base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_single_interaction(
    agent: Agent,
    principal: Principal,
    principal_context: str,
    agent_context: str,
    agent_task: str | None = None,
    agent_objective: str | None = None,
) -> Mapping[str, Any]:
    """
    Run a single principal-agent interaction with information asymmetry.

    Args:
        agent: Agent instance with full data access
        principal: Principal instance with partial data
        principal_context: Partial information for principal (hypothesis + demographics)
        agent_context: Full information for agent (hypothesis + demographics + data distribution)
        agent_task: Optional task instruction for agent
        agent_objective: Optional objective for agent

    Returns:
        Dictionary with interaction results
    """
    # Agent has full data access, generates recommendation
    information = agent.act(context=agent_context, task=agent_task, objective=agent_objective)

    # Principal has partial data, makes decision based on agent's information
    decision_result = principal.act(context=principal_context, information=information)

    return {
        "agent_name": agent.name,
        "agent_model": agent.model,
        "principal_name": principal.name,
        "principal_model": principal.model,
        "principal_context": principal_context,
        "agent_context": agent_context,
        "agent_task": agent_task,
        "agent_objective": agent_objective,
        "information": information,
        "decision": decision_result["decision"],
        "belief": decision_result["belief"],
        "reasoning": decision_result["reasoning"],
        "raw_principal_response": decision_result["raw_response"],
    }


def run_experiment_thread(
    contexts: list[str],
    agent_config: Mapping[str, Any],
    principal_config: Mapping[str, Any],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    principal_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
) -> list[Mapping[str, Any]]:
    results = []

    agent = Agent(
        name=agent_config["name"],
        client=agent_client,
        model=agent_config["model"],
        prompt_path=agent_config["prompt_path"],
        temperature=agent_config.get("temperature", 1.0),
    )

    principal = Principal(
        name=principal_config["name"],
        client=principal_client,
        model=principal_config["model"],
        prompt_path=principal_config["prompt_path"],
        temperature=principal_config.get("temperature", 1.0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_iterations = len(contexts)
    progress_bar = tqdm(total=total_iterations, desc=f"Running {agent.name}-{principal.name}")
    
    iteration_count = 0

    for context in contexts:
        # Handle both string and dict contexts
        if isinstance(context, str):
            # Simple string context - no information asymmetry
            principal_context = agent_context = context
            agent_task = agent_objective = None
        else:
            # Dict context - implement information asymmetry
            # Principal gets: patient demographics + hypothesis (partial info)
            principal_context = (
                f"{context['demographics']}\n\n"
                f"Hypothesis: {context['hypothesis']}"
            )

            # Agent gets: principal context + full data distribution
            agent_context = (
                f"{principal_context}\n\n"
                f"Available Data:\n{context['data_report']}"
            )

            agent_task = context.get("agent_task")
            agent_objective = context.get("agent_objective")

        result = run_single_interaction(
            agent=agent,
            principal=principal,
            principal_context=principal_context,
            agent_context=agent_context,
            agent_task=agent_task,
            agent_objective=agent_objective,
        )
        results.append(result)
        iteration_count += 1
        progress_bar.update(1)

        # Save results at specified intervals
        if iteration_count % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    progress_bar.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_experiment_sequential(
    contexts: list[str],
    agent_configs: list[Mapping[str, Any]],
    principal_configs: list[Mapping[str, Any]],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    principal_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
) -> list[Mapping[str, Any]]:
    results = []
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

    principals = [
        Principal(
            name=config["name"],
            client=principal_client,
            model=config["model"],
            prompt_path=config["prompt_path"],
            temperature=config.get("temperature", 1.0),
        )
        for config in principal_configs
    ]

    total_iterations = len(contexts) * len(agents) * len(principals)
    progress_bar = tqdm(total=total_iterations, desc="Running experiments")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iteration_count = 0

    for context in contexts:
        for agent in agents:
            for principal in principals:
                # Handle both string and dict contexts
                if isinstance(context, str):
                    # Simple string context - no information asymmetry
                    principal_context = agent_context = context
                    agent_task = agent_objective = None
                else:
                    # Dict context - implement information asymmetry
                    # Principal gets: patient demographics + hypothesis (partial info)
                    principal_context = (
                        f"{context['demographics']}\n\n"
                        f"Hypothesis: {context['hypothesis']}"
                    )

                    # Agent gets: principal context + full data distribution
                    agent_context = (
                        f"{principal_context}\n\n"
                        f"Available Data:\n{context['data_report']}"
                    )

                    agent_task = context.get("agent_task")
                    agent_objective = context.get("agent_objective")

                result = run_single_interaction(
                    agent=agent,
                    principal=principal,
                    principal_context=principal_context,
                    agent_context=agent_context,
                    agent_task=agent_task,
                    agent_objective=agent_objective,
                )
                results.append(result)
                iteration_count += 1
                progress_bar.update(1)

                # Save results at specified intervals
                if iteration_count % save_interval == 0:
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=2)

    progress_bar.close()

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_experiment_concurrent(
    contexts: list[str],
    agent_configs: list[Mapping[str, Any]],
    principal_configs: list[Mapping[str, Any]],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    principal_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
    max_workers: int = 4,
) -> list[Mapping[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {}
        for agent_config in agent_configs:
            for principal_config in principal_configs:
                future = executor.submit(
                    run_experiment_thread,
                    contexts,
                    agent_config,
                    principal_config,
                    agent_client,
                    principal_client,
                    output_path,
                    save_interval,
                )
                future_to_config[future] = (agent_config, principal_config)

        for future in future_to_config:
            results = future.result()
            all_results.extend(results)

    # Save final results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--principal-server",
        type=str,
        default="nvidia",
        choices=["nvidia", "openrouter", "sglang"],
        help="Server backend for principals",
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
        "--principal-model",
        type=str,
        default="deepseek-ai/deepseek-v3.1",
        help="Model to use for principals",
    )
    parser.add_argument(
        "--input", type=str, default="experiments/input/hypothesis_mimic.json",
        help="Input hypothesis JSON file (MIMIC or PMC format) or old-style config JSON"
    )
    parser.add_argument(
        "--hypothesis-type",
        type=str,
        choices=["mimic", "pmc", "auto"],
        default="auto",
        help="Type of hypothesis data: 'mimic' for MIMIC-IV, 'pmc' for PMC, 'auto' to detect from filename"
    )
    parser.add_argument(
        "--agent-prompt",
        type=str,
        default="prompts/agent/default.yaml",
        help="Agent prompt template path"
    )
    parser.add_argument(
        "--principal-prompt",
        type=str,
        default="prompts/principal/prospect.yaml",
        help="Principal prompt template path"
    )
    parser.add_argument("--output", type=str, default="experiments/output/results.json")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save results every N iterations",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        default=True,
        help="Run experiments concurrently",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent threads when running concurrently",
    )
    args = parser.parse_args()

    # Setup separate clients for principals and agents
    principal_client = setup_clients(backend=args.principal_server)
    agent_client = setup_clients(
        backend=args.agent_server,
        sglang_port=args.agent_sglang_port,
        sglang_base_url=args.agent_sglang_base_url,
    )

    with open(args.input, "r") as f:
        input_data = json.load(f)

    # Detect input format: old-style config vs new hypothesis format
    if isinstance(input_data, dict) and "contexts" in input_data:
        # Old-style config format with pre-built contexts
        print("Detected old-style config format")
        contexts = input_data["contexts"]
        # Only use default agent config
        agent_configs = [input_data["agent_configs"][0]]
        # Only use prospect-theoretic principal config
        principal_configs = [config for config in input_data["principal_configs"] 
                           if "prospect" in config["prompt_path"].lower()]
        if not principal_configs:
            principal_configs = [input_data["principal_configs"][0]]  # Fallback to first config
    elif isinstance(input_data, list):
        # New hypothesis format - need to build contexts
        print("Detected hypothesis list format")

        # Auto-detect hypothesis type from filename if requested
        hypothesis_type = args.hypothesis_type
        if hypothesis_type == "auto":
            if "mimic" in args.input.lower():
                hypothesis_type = "mimic"
            elif "pmc" in args.input.lower():
                hypothesis_type = "pmc"
            else:
                raise ValueError(
                    "Cannot auto-detect hypothesis type from filename. "
                    "Please specify --hypothesis-type explicitly."
                )

        print(f"Using hypothesis type: {hypothesis_type}")

        # Build contexts from hypotheses
        contexts = build_contexts(input_data, hypothesis_type)
        print(f"Built {len(contexts)} contexts")

        # Create only default agent config
        agent_configs = [{
            "name": "default_agent",
            "model": args.agent_model,
            "prompt_path": args.agent_prompt,
            "temperature": 1.0,
        }]

        # Create only prospect-theoretic principal config
        principal_configs = [{
            "name": "prospect_principal",
            "model": args.principal_model,
            "prompt_path": args.principal_prompt,
            "temperature": 1.0,
        }]
    else:
        raise ValueError(
            "Invalid input format. Expected either:\n"
            "  1. Dict with 'contexts', 'agent_configs', 'principal_configs' (old format)\n"
            "  2. List of hypothesis dictionaries (new format)"
        )

    # Set models for agents and principals from command line args (override if needed)
    for agent_config in agent_configs:
        if "model" not in agent_config or agent_config["model"] is None:
            agent_config["model"] = args.agent_model
    for principal_config in principal_configs:
        if "model" not in principal_config or principal_config["model"] is None:
            principal_config["model"] = args.principal_model

    print(f"Agent configs: {len(agent_configs)}")
    print(f"Principal configs: {len(principal_configs)}")
    print(f"Total contexts: {len(contexts)}")
    print(f"Total experiments: {len(contexts) * len(agent_configs) * len(principal_configs)}")

    output_path = Path(args.output)

    if args.concurrent:
        results = run_experiment_concurrent(
            contexts=contexts,
            agent_configs=agent_configs,
            principal_configs=principal_configs,
            agent_client=agent_client,
            principal_client=principal_client,
            output_path=output_path,
            save_interval=args.save_interval,
            max_workers=args.max_workers,
        )
    else:
        results = run_experiment_sequential(
            contexts=contexts,
            agent_configs=agent_configs,
            principal_configs=principal_configs,
            agent_client=agent_client,
            principal_client=principal_client,
            output_path=output_path,
            save_interval=args.save_interval,
        )


if __name__ == "__main__":
    main()
