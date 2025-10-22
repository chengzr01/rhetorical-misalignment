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
    output_dir: Path,
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

    # Create unique output file for this agent-principal pair
    output_file = output_dir / f"{agent.name}_{principal.name}_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

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
                f"Patient ID: {context['patient_id']}\n"
                f"Sex: {context['sex']}, Age: {context['age']} years\n\n"
                f"Question: Should this patient receive {context['drug']}?\n\n"
                f"Hypothesis: {context['hypothesis']}"
            )

            # Agent gets: principal context + full data distribution
            agent_context = (
                f"{principal_context}\n\n"
                f"Historical Data:\n{context['data_report']}"
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
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

    progress_bar.close()

    # Final save
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_experiment_sequential(
    contexts: list[str],
    agent_configs: list[Mapping[str, Any]],
    principal_configs: list[Mapping[str, Any]],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    principal_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: str | Path,
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
    
    output_path = Path(output_path)
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
                        f"Patient ID: {context['patient_id']}\n"
                        f"Sex: {context['sex']}, Age: {context['age']} years\n\n"
                        f"Question: Should this patient receive {context['drug']}?\n\n"
                        f"Hypothesis: {context['hypothesis']}"
                    )

                    # Agent gets: principal context + full data distribution
                    agent_context = (
                        f"{principal_context}\n\n"
                        f"Historical Data:\n{context['data_report']}"
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
    output_dir: str | Path,
    save_interval: int = 10,
    max_workers: int = 4,
) -> list[Mapping[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                    output_dir,
                    save_interval,
                )
                future_to_config[future] = (agent_config, principal_config)

        for future in future_to_config:
            results = future.result()
            all_results.extend(results)

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
        default=30002,
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
        "--input", type=str, default="experiments/input/hypothesis.json"
    )
    parser.add_argument("--output", type=str, default="experiments/output_dpo/results.json")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
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
        config = json.load(f)

    # Set models for agents and principals from command line args
    for agent_config in config["agent_configs"]:
        if "model" not in agent_config:
            agent_config["model"] = args.agent_model
    for principal_config in config["principal_configs"]:
        if "model" not in principal_config:
            principal_config["model"] = args.principal_model

    if args.concurrent:
        output_dir = Path(args.output).parent
        run_experiment_concurrent(
            contexts=config["contexts"],
            agent_configs=config["agent_configs"],
            principal_configs=config["principal_configs"],
            agent_client=agent_client,
            principal_client=principal_client,
            output_dir=output_dir,
            save_interval=args.save_interval,
            max_workers=args.max_workers,
        )
    else:
        run_experiment_sequential(
            contexts=config["contexts"],
            agent_configs=config["agent_configs"],
            principal_configs=config["principal_configs"],
            agent_client=agent_client,
            principal_client=principal_client,
            output_path=args.output,
            save_interval=args.save_interval,
        )


if __name__ == "__main__":
    main()
