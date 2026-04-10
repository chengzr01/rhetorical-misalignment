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

    def act(self, context: str, information: str = "", options: str = "", **kwargs: Any) -> Mapping[str, Any]:
        """
        Make a decision based on context and information from an agent.

        Args:
            context: The decision context/scenario
            information: Information provided by the expert agent (optional)
            options: Answer options for belief-elicitation mode (optional)
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Dictionary containing:
                - decision/answer: The principal's decision or answer
                - belief: The principal's updated belief
                - reasoning: The principal's reasoning
                - raw_response: The raw LLM response
        """
        # Fill in the prompt template
        prompt = self.prompt_template.replace("<CONTEXT>", context)

        # Handle different prompt formats
        if "<RECOMMENDATIONS>" in prompt:
            # Original format: agent recommendations
            prompt = prompt.replace("<RECOMMENDATIONS>", information)

        if "<OPTIONS>" in prompt:
            # Belief-elicitation format: answer options
            prompt = prompt.replace("<OPTIONS>", options)

        # Call the LLM
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages, **kwargs)

        # Parse the structured response
        parsed = self._parse_response(response)
        parsed["raw_response"] = response

        return parsed

    @staticmethod
    def _normalize_response(response: str) -> str:
        """
        Normalize raw LLM response to fix known OpenRouter streaming artifacts.

        Two artifacts are observed in practice:
        - Missing opening '<': response starts with 'answer>...' instead of '<answer>...'
          Caused by the '<' being consumed as a streaming chunk boundary.
        - Double '<<' prefix: response starts with '<<answer>' or '<<reasoning>'.
          Caused by a stray '<' prepended to the response.
        - Leading quote before tag: response starts with '"answer>...' (JSON-escaped artifact).
        """
        if not response:
            return response
        # Strip a stray leading double-quote before an unbracketed tag
        s = response.lstrip('"')
        # Fix missing '<': 'answer>...' or 'decision>...' at the very start
        if re.match(r"^(answer|decision|belief|reasoning)>", s):
            s = "<" + s
        # Fix double '<<': collapse to single '<'
        elif s.startswith("<<"):
            s = s[1:]
        return s

    def _parse_response(self, response: str) -> Mapping[str, str]:
        """
        Parse the principal's response to extract decision/answer, belief, and reasoning.

        Expected formats:
        - Original:     <decision>X</decision><belief>p</belief><reasoning>...</reasoning>
        - Choices mode: <answer>X</answer><belief>p</belief><reasoning>...</reasoning>

        Handles streaming/API artifacts via _normalize_response, and uses layered
        fallback patterns so that partial or malformed responses still yield as much
        signal as possible.
        """
        result = {"decision": "", "answer": "", "belief": "", "reasoning": ""}

        text = self._normalize_response(response)

        # ── Answer / decision ────────────────────────────────────────────────
        letter = ""

        # 1. Well-formed <answer>A</answer> or <decision>A</decision>
        for tag in ("answer", "decision"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                # Accept single letter, or letter followed by punctuation/space
                lm = re.match(r"^([A-F])[.\s)\-]?", candidate, re.IGNORECASE)
                if lm:
                    letter = lm.group(1).upper()
                    break
                # Accept the full text value for non-USMLE (MIMIC) decisions
                if candidate:
                    result["decision"] = candidate
                    result["answer"] = candidate
                    break

        # 2. Unclosed tag: <answer>A (no closing tag, e.g. response was truncated)
        if not letter and not result["decision"]:
            m = re.search(r"<(?:answer|decision)>\s*([A-F])\s*(?:[.\s)\-]|$)", text, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()

        # 3. Belief tag with embedded letter before the malformed close:
        #    e.g. '<belief>0.85</belief, the answer is A ...'  — check reasoning text
        if not letter and not result["decision"]:
            m = re.search(r"\bthe answer is\s+([A-F])\b", text, re.IGNORECASE)
            if m:
                letter = m.group(1).upper()

        if letter:
            result["decision"] = letter
            result["answer"] = letter

        # ── Belief ───────────────────────────────────────────────────────────
        # 1. Well-formed <belief>0.95</belief>
        m = re.search(r"<belief>\s*([\d.]+)\s*</belief>", text, re.IGNORECASE)
        if m:
            result["belief"] = m.group(1)
        else:
            # 2. Malformed closing tag: <belief>0.95</belief, ... or <belief>0.95<
            m = re.search(r"<belief>\s*([\d.]+)\s*(?:</belief|$)", text, re.IGNORECASE)
            if m:
                result["belief"] = m.group(1)

        # ── Reasoning ────────────────────────────────────────────────────────
        # 1. Well-formed <reasoning>...</reasoning>
        m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
        if m:
            result["reasoning"] = m.group(1).strip()
        else:
            # 2. Unclosed tag or start-truncated response ending with </reasoning>:
            #    capture everything up to the closing tag
            m = re.search(r"<reasoning>(.*?)$", text, re.DOTALL | re.IGNORECASE)
            if m:
                result["reasoning"] = m.group(1).strip()
            else:
                m = re.search(r"^(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
                if m:
                    result["reasoning"] = m.group(1).strip()

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
