from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from agents.base import BaseAgent


class Agent(BaseAgent):
    """
    Expert agent that designs and provides information to support principal's decision-making.

    The agent:
    - Receives a decision context and task
    - Designs information presentation to influence the principal
    - May have objectives that differ from the principal's best interests
    """

    def __init__(
        self,
        name: str,
        client: Any,
        model: str,
        prompt_path: str | Path,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize an expert agent.

        Args:
            name: Name identifier for this agent
            client: LLM client for generating responses
            model: Model name/identifier to use
            prompt_path: Path to YAML file containing the agent's prompt template
            temperature: Sampling temperature for LLM responses
        """
        super().__init__(name, client, model, temperature)
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: str | Path) -> str:
        """Load prompt template from YAML file."""
        with open(prompt_path, "r") as f:
            data = yaml.safe_load(f)
        return data["prompt"]

    def act(
        self,
        context: str,
        task: str | None = None,
        objective: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Design information to provide to the principal.

        Args:
            context: The decision context/scenario
            task: Specific task or instruction for the agent
            objective: The agent's objective (e.g., "persuade principal to convict")
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The information/message designed by the agent
        """
        # Build the prompt
        prompt = self.prompt_template

        # Add context if needed
        if "<CONTEXT>" in prompt:
            prompt = prompt.replace("<CONTEXT>", context)

        # Build full prompt with task and objective
        full_prompt_parts = [prompt]

        if context and "<CONTEXT>" not in self.prompt_template:
            full_prompt_parts.append(f"\n\nContext: {context}")

        if task:
            full_prompt_parts.append(f"\n\nTask: {task}")

        if objective:
            full_prompt_parts.append(f"\n\nObjective: {objective}")

        full_prompt = "".join(full_prompt_parts)

        # Call the LLM
        messages = [{"role": "user", "content": full_prompt}]
        response = self._call_llm(messages, **kwargs)

        return response

    def design_information(
        self,
        context: str,
        goal: str,
        constraints: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Design information to achieve a specific goal.

        Args:
            context: The decision context/scenario
            goal: What the agent wants to achieve
            constraints: Any constraints on the information design
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The designed information message
        """
        task = f"Design information that will help achieve this goal: {goal}"
        if constraints:
            task += f"\n\nConstraints: {constraints}"

        return self.act(context=context, task=task, **kwargs)

    def provide_information(
        self,
        context: str,
        data: str | None = None,
        framing: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Provide information about a context, optionally with specific data and framing.

        Args:
            context: The decision context/scenario
            data: Specific data or evidence to include
            framing: How to frame the information
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            The information message
        """
        task_parts = ["Provide information to help the principal make a decision."]

        if data:
            task_parts.append(f"\n\nData to include: {data}")

        if framing:
            task_parts.append(f"\n\nFraming approach: {framing}")

        task = "".join(task_parts)

        return self.act(context=context, task=task, **kwargs)
