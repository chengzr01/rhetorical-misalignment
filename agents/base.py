from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseAgent(ABC):
    """Base class for all agents in the principal-agent framework."""

    def __init__(self, name: str, client: Any, model: str, temperature: float = 1.0) -> None:
        """
        Initialize a base agent.

        Args:
            name: Name identifier for this agent
            client: LLM client (OpenRouterChatClient or SGLangChatClient)
            model: Model name/identifier to use
            temperature: Sampling temperature for LLM responses
        """
        self.name = name
        self.client = client
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> Any:
        """
        Perform the agent's action. Must be implemented by subclasses.

        Returns:
            The result of the agent's action
        """
        pass

    def _call_llm(self, messages: list[Mapping[str, str]], **kwargs: Any) -> str:
        """
        Call the LLM client with given messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments to pass to the client

        Returns:
            The LLM's response content
        """
        return self.client.create_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.model}')"
