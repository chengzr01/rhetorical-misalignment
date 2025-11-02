from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from agents.base import BaseAgent


class Principal(BaseAgent):
    """
    Principal agent that makes decisions based on information provided by agents.

    The principal:
    - Receives context and information from an expert agent
    - Updates beliefs and makes decisions according to their decision-making style
    - Returns structured output with decision, belief, and reasoning
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
        Initialize a principal agent.

        Args:
            name: Name identifier for this principal
            client: LLM client for generating responses
            model: Model name/identifier to use
            prompt_path: Path to YAML file containing the principal's prompt template
            temperature: Sampling temperature for LLM responses
        """
        super().__init__(name, client, model, temperature)
        self.prompt_template = self._load_prompt(prompt_path)

    def _load_prompt(self, prompt_path: str | Path) -> str:
        """Load prompt template from YAML file."""
        with open(prompt_path, "r") as f:
            data = yaml.safe_load(f)
        return data["prompt"]

    def act(self, context: str, information: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Make a decision based on context and information from an agent.

        Args:
            context: The decision context/scenario
            information: Information provided by the expert agent
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Dictionary containing:
                - decision: The principal's decision
                - belief: The principal's updated belief
                - reasoning: The principal's reasoning
                - raw_response: The raw LLM response
        """
        # Fill in the prompt template
        prompt = self.prompt_template.replace("<CONTEXT>", context).replace(
            "<RECOMMENDATIONS>", information
        )

        # Call the LLM
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages, **kwargs)

        # Parse the structured response
        parsed = self._parse_response(response)
        parsed["raw_response"] = response

        return parsed

    def _parse_response(self, response: str) -> Mapping[str, str]:
        """
        Parse the principal's response to extract decision, belief, and reasoning.

        Expected format: <decision>...</decision><belief>...</belief><reasoning>...</reasoning>
        """
        result = {"decision": "", "belief": "", "reasoning": ""}

        # Extract decision
        decision_match = re.search(r"<decision>(.*?)</decision>", response, re.DOTALL)
        if decision_match:
            result["decision"] = decision_match.group(1).strip()

        # Extract belief
        belief_match = re.search(r"<belief>(.*?)</belief>", response, re.DOTALL)
        if belief_match:
            result["belief"] = belief_match.group(1).strip()

        # Extract reasoning
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        return result

    def make_decision(
        self,
        context: str,
        information: str,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Mapping[str, Any] | str:
        """
        Convenience method for making a decision.

        Args:
            context: The decision context/scenario
            information: Information provided by the expert agent
            return_dict: If True, return full dictionary; if False, return only decision
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Full result dictionary or just the decision string
        """
        result = self.act(context, information, **kwargs)
        return result if return_dict else result["decision"]
