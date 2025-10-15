from __future__ import annotations

import os
from typing import Iterable, Mapping

from openai import OpenAI


class OpenRouterChatClient:
    def __init__(
        self, api_key: str | None = None, base_url: str = "https://openrouter.ai/api/v1"
    ) -> None:
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def create_completion(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, str]],
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content


class NvidiaChatClient:
    def __init__(self, api_key: str | None = None) -> None:
        if not api_key:
            api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA API key is required")
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", api_key=api_key
        )

    def create_completion(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, str]],
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content


class SGLangChatClient:
    def __init__(self, port: int = 30000, base_url: str = "http://127.0.0.1") -> None:
        self._client = OpenAI(base_url=f"{base_url}:{port}/v1", api_key="None")

    def create_completion(
        self,
        *,
        model: str,
        messages: Iterable[Mapping[str, str]],
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content
