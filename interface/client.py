from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Iterable, Mapping

from openai import OpenAI


class BaseChatClient(ABC):
    """Abstract base class for chat clients with retry logic."""
    
    def __init__(self) -> None:
        self._client: OpenAI | None = None
    
    @abstractmethod
    def _initialize_client(self) -> OpenAI:
        """Initialize and return the OpenAI client."""
        pass
    
    def create_completion(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, str]],
        temperature: float = 1.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Create a chat completion with retry logic.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The completion text
            
        Raises:
            Exception: If all retries fail
        """
        if self._client is None:
            self._client = self._initialize_client()
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    temperature=temperature,
                    **kwargs,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise Exception(
                        f"Failed after {max_retries} attempts. Last error: {str(e)}"
                    ) from e
        
        # This should never be reached, but for type safety
        raise last_exception


class OpenRouterChatClient(BaseChatClient):
    def __init__(
        self, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1"
    ) -> None:
        super().__init__()
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self._api_key = api_key
        self._base_url = base_url
    
    def _initialize_client(self) -> OpenAI:
        return OpenAI(base_url=self._base_url, api_key=self._api_key)


class NvidiaChatClient(BaseChatClient):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        if not api_key:
            api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA API key is required")
        self._api_key = api_key
    
    def _initialize_client(self) -> OpenAI:
        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", api_key=self._api_key
        )


class SGLangChatClient(BaseChatClient):
    def __init__(self, port: int = 30000, base_url: str = "http://127.0.0.1") -> None:
        super().__init__()
        self._port = port
        self._base_url = base_url
    
    def _initialize_client(self) -> OpenAI:
        return OpenAI(base_url=f"{self._base_url}:{self._port}/v1", api_key="None")
