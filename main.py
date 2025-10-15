from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from agents.agent import Agent
from agents.principal import Principal
from interface.client import OpenRouterChatClient, SGLangChatClient


def setup_clients(
    backend: str = "openrouter",
) -> OpenRouterChatClient | SGLangChatClient:
    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        return OpenRouterChatClient(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    elif backend == "sglang":
        return SGLangChatClient(port=30000, base_url="http://127.0.0.1")
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_single_interaction(
    agent: Agent,
    principal: Principal,
    context: str,
    agent_task: str | None = None,
    agent_objective: str | None = None,
) -> Mapping[str, Any]:
    information = agent.act(context=context, task=agent_task, objective=agent_objective)
    decision_result = principal.act(context=context, information=information)
    return {
        "agent_name": agent.name,
        "principal_name": principal.name,
        "context": context,
        "agent_task": agent_task,
        "agent_objective": agent_objective,
        "information": information,
        "decision": decision_result["decision"],
        "belief": decision_result["belief"],
        "reasoning": decision_result["reasoning"],
        "raw_principal_response": decision_result["raw_response"],
    }


def run_experiment(
    contexts: list[str],
    agent_configs: list[Mapping[str, Any]],
    principal_configs: list[Mapping[str, Any]],
    client: OpenRouterChatClient | SGLangChatClient,
    output_path: str | Path,
) -> list[Mapping[str, Any]]:
    results = []
    agents = [
        Agent(
            name=config["name"],
            client=client,
            model=config["model"],
            prompt_path=config["prompt_path"],
            temperature=config.get("temperature", 1.0),
        )
        for config in agent_configs
    ]

    principals = [
        Principal(
            name=config["name"],
            client=client,
            model=config["model"],
            prompt_path=config["prompt_path"],
            temperature=config.get("temperature", 1.0),
        )
        for config in principal_configs
    ]

    for context in contexts:
        for agent in agents:
            for principal in principals:
                ctx_str = (
                    context
                    if isinstance(context, str)
                    else context.get("text", str(context))
                )
                result = run_single_interaction(
                    agent=agent,
                    principal=principal,
                    context=ctx_str,
                    agent_task=context.get("agent_task")
                    if isinstance(context, dict)
                    else None,
                    agent_objective=context.get("agent_objective")
                    if isinstance(context, dict)
                    else None,
                )
                results.append(result)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", type=str, default="openrouter", choices=["openrouter", "sglang"]
    )
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-sonnet")
    parser.add_argument(
        "--input", type=str, default="experiments/input/hypothesis.json"
    )
    parser.add_argument("--output", type=str, default="experiments/output/results.json")
    args = parser.parse_args()

    client = setup_clients(backend=args.server)

    with open(args.input, "r") as f:
        config = json.load(f)

    for agent_config in config["agent_configs"]:
        if "model" not in agent_config:
            agent_config["model"] = args.model
    for principal_config in config["principal_configs"]:
        if "model" not in principal_config:
            principal_config["model"] = args.model

    run_experiment(
        contexts=config["contexts"],
        agent_configs=config["agent_configs"],
        principal_configs=config["principal_configs"],
        client=client,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
