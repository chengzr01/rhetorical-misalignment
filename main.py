from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.agent import Agent
from agents.principal import Principal
from interface.client import NvidiaChatClient, OpenRouterChatClient, SGLangChatClient


def build_clinical_question_contexts(questions: list[dict]) -> list[dict]:
    """
    Build contexts from clinical questions data.
    """
    contexts = []
    for item in questions:
        context_dict = {
            "id": item.get("id"),
            "hadm_id": item.get("hadm_id"),
            "subject_id": item.get("subject_id"),
            "context": item["context"],
            "question": item["question"],
            "ground_truth": item.get("ground_truth", {}),
        }
        contexts.append(context_dict)
    return contexts


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


def run_principal_decision(
    principal: Principal,
    principal_context: str,
    agent_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Run principal decision using cached agent results.
    The principal_context should contain a <RECOMMENDATIONS> placeholder that will be replaced
    by the agent's information.
    """
    
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
    
    pc = principal_context
    if pc and "<RECOMMENDATIONS>" in pc:
        pc = pc.replace("<RECOMMENDATIONS>", recommendation.strip() if recommendation else "")
    
    decision_result = principal.act(context=pc, information=agent_result["information"])
    
    return {
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
    }


def run_agent_inferences(
    contexts: list[dict],
    agent_configs: list[Mapping[str, Any]],
    agent_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
    skip_existing: bool = True,
    max_workers: int = 8,
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

    Returns:
        List of agent results
    """
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
            result["hadm_id"] = context.get("hadm_id")
            result["subject_id"] = context.get("subject_id")
            result["principal_context"] = context["context"]
            result["ground_truth"] = context.get("ground_truth")
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


def run_principal_inferences(
    agent_results: list[Mapping[str, Any]],
    principal_configs: list[Mapping[str, Any]],
    principal_client: NvidiaChatClient | OpenRouterChatClient | SGLangChatClient,
    output_path: Path,
    save_interval: int = 10,
    max_workers: int = 8,
) -> list[Mapping[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
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
    
    # Group agent results by case_id to process all principals for each case together
    case_to_agent_results = {}
    for agent_result in agent_results:
        case_id = agent_result.get("case_id")
        if case_id not in case_to_agent_results:
            case_to_agent_results[case_id] = []
        case_to_agent_results[case_id].append(agent_result)
    
    total_iterations = len(agent_results) * len(principals)
    progress_bar = tqdm(total=total_iterations, desc="Running principal inferences")
    iteration_count = 0

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
        if "case_id" in agent_result:
            result["case_id"] = agent_result["case_id"]
            result["hadm_id"] = agent_result.get("hadm_id")
            result["subject_id"] = agent_result.get("subject_id")
            result["ground_truth"] = agent_result.get("ground_truth")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for agent_result, principal in tasks:
            futures.append(executor.submit(run_one_principal_decision, agent_result, principal))
        results = []
        for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing principal results")):
            result = future.result()
            results.append(result)
            iteration_count += 1
            progress_bar.update(1)
            if save_interval > 0 and iteration_count % save_interval == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
        progress_bar.close()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} complete results to {output_path}")
    return results


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
        if isinstance(context, str):
            principal_context = agent_context = context
            agent_task = agent_objective = None
        else:
            principal_context = context['context']
            agent_context = (
                f"{context['context']}\n\n"
                f"{context['question']}"
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

        if isinstance(context, dict) and "id" in context:
            result["case_id"] = context["id"]
            result["hadm_id"] = context.get("hadm_id")
            result["subject_id"] = context.get("subject_id")
            result["ground_truth"] = context.get("ground_truth")

        results.append(result)
        iteration_count += 1
        progress_bar.update(1)

        if iteration_count % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    progress_bar.close()

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
                if isinstance(context, str):
                    principal_context = agent_context = context
                    agent_task = agent_objective = None
                else:
                    principal_context = context['context']
                    agent_context = (
                        f"{context['context']}\n\n"
                        f"{context['question']}"
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

                if isinstance(context, dict) and "id" in context:
                    result["case_id"] = context["id"]
                    result["hadm_id"] = context.get("hadm_id")
                    result["subject_id"] = context.get("subject_id")
                    result["ground_truth"] = context.get("ground_truth")

                results.append(result)
                iteration_count += 1
                progress_bar.update(1)

                if iteration_count % save_interval == 0:
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=2)

    progress_bar.close()

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
        "--input", type=str, default="experiments/input/clinical_questions.json",
        help="Input clinical questions JSON file"
    )
    parser.add_argument(
        "--agent-prompt",
        type=str,
        default="prompts/agent/default.yaml",
        help="Agent prompt template path"
    )
    parser.add_argument(
        "--principal-types",
        type=str,
        nargs="+",
        default=["bayesian"],
        choices=["all", "bayesian", "anchoring", "availability", "confirmation", "conservatism", "overconfidence", "prospect"],
        help="Principal types to use. Use 'all' to run all types except bayesian, or specify one or more types"
    )
    parser.add_argument("--output", type=str, default="experiments/output/results.json",
                        help="Output path for final results")
    parser.add_argument(
        "--agent-cache",
        type=str,
        default="experiments/cache/agent_results.json",
        help="Path to cache agent inference results"
    )
    parser.add_argument(
        "--skip-agent-cache",
        action="store_true",
        default=False,
        help="Force re-run agent inferences even if cache exists"
    )
    parser.add_argument(
        "--only-agents",
        action="store_true",
        default=False,
        help="Only run agent inferences and exit (useful for caching)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save results every N iterations",
    )
    parser.add_argument(
        "--principal-workers",
        type=int,
        default=8,
        help="Number of workers for parallel principal inferences"
    )
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=8,
        help="Number of workers for parallel agent inferences"
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

    if not isinstance(input_data, list):
        raise ValueError(
            f"Invalid input format. Expected a list of clinical question dictionaries, "
            f"but got {type(input_data).__name__}"
        )

    print(f"Loading {len(input_data)} clinical questions")
    contexts = build_clinical_question_contexts(input_data)
    print(f"Built {len(contexts)} contexts")

    agent_configs = [{
        "name": "default_agent",
        "model": args.agent_model,
        "prompt_path": args.agent_prompt,
        "temperature": 1.0,
    }]

    principal_types = args.principal_types
    all_available_principal_types = ["anchoring", "availability", "confirmation",
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

    for agent_config in agent_configs:
        if "model" not in agent_config or agent_config["model"] is None:
            agent_config["model"] = args.agent_model
    for principal_config in principal_configs:
        if "model" not in principal_config or principal_config["model"] is None:
            principal_config["model"] = args.principal_model

    print(f"Agent configs: {len(agent_configs)}")
    print(f"Principal configs: {len(principal_configs)}")
    print(f"Total contexts: {len(contexts)}")
    print(f"Total agent inferences: {len(contexts) * len(agent_configs)}")
    print(f"Total principal inferences: {len(contexts) * len(agent_configs) * len(principal_configs)}")
    print(f"Total experiments: {len(contexts) * len(agent_configs) * len(principal_configs)}")

    agent_cache_path = Path(args.agent_cache)
    output_path = Path(args.output)

    print("\n" + "="*60)
    print("STAGE 1: Agent Inferences")
    print("="*60)

    agent_results = run_agent_inferences(
        contexts=contexts,
        agent_configs=agent_configs,
        agent_client=agent_client,
        output_path=agent_cache_path,
        save_interval=args.save_interval,
        skip_existing=not args.skip_agent_cache,
        max_workers=getattr(args, "agent_workers", 8),
    )

    print(f"\nAgent inferences complete: {len(agent_results)} results")

    if args.only_agents:
        print(f"\n--only-agents flag set. Exiting after agent inferences.")
        print(f"Agent results saved to: {agent_cache_path}")
        return

    print("\n" + "="*60)
    print("STAGE 2: Principal Inferences")
    print("="*60)

    final_results = run_principal_inferences(
        agent_results=agent_results,
        principal_configs=principal_configs,
        principal_client=principal_client,
        output_path=output_path,
        save_interval=args.save_interval,
        max_workers=args.principal_workers,
    )

    print(f"\nAll experiments complete!")
    print(f"Agent results: {agent_cache_path}")
    print(f"Final results: {output_path}")
    print(f"Total interactions: {len(final_results)}")


if __name__ == "__main__":
    main()
