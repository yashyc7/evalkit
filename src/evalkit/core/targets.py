import time
from abc import ABC, abstractmethod
from dataclasses import replace
from decimal import Decimal
from typing import Any

import httpx

from evalkit.core.types import TargetOutput, to_text


class BaseTarget(ABC):
    """Contract for anything we can test. Subclasses only write _call()."""

    async def call(self, input: Any) -> TargetOutput:
        """Public method used by the runner. Measures time and never raises."""
        start = time.perf_counter()
        try:
            result = await self._call(input)
        except Exception as exc:
            # One bad question must not crash the whole run.
            # We save the error and move on, like the Timeout row in our seed data.
            result = TargetOutput(error=f"{type(exc).__name__}: {exc}")
        latency_ms = int((time.perf_counter() - start) * 1000)
        return replace(result, latency_ms=latency_ms)

    @abstractmethod
    async def _call(self, input: Any) -> TargetOutput:
        """Subclasses write the real work here and return a TargetOutput."""


class EchoTarget(BaseTarget):
    """Fake target for practice: repeats the question. No API key needed."""

    def __init__(self, prefix: str = "Echo: ") -> None:
        self.prefix = prefix

    async def _call(self, input: Any) -> TargetOutput:
        return TargetOutput(output=f"{self.prefix}{to_text(input)}")


class HttpTarget(BaseTarget):
    """Calls your own bot over HTTP.

    Sends   {"input": <question>}
    Expects {"output": <answer>, "input_tokens": 10, "output_tokens": 5} (tokens optional)
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.url = url
        self.headers = headers or {}
        self.timeout_s = timeout_s

    async def _call(self, input: Any) -> TargetOutput:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                self.url, json={"input": input}, headers=self.headers
            )
            response.raise_for_status()  # turns 4xx/5xx replies into errors
            data = response.json()
        return TargetOutput(
            output=data.get("output"),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
        )


class LLMTarget(BaseTarget):
    """Calls an LLM directly (any provider, through LiteLLM) with a system prompt."""

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    async def _call(self, input: Any) -> TargetOutput:
        import litellm  # imported here because it is slow to load (a few seconds)

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": to_text(input)})

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            drop_params=True,  # ignore settings a model does not support
        )

        try:
            cost: Decimal | None = Decimal(
                str(litellm.completion_cost(completion_response=response))
            )
        except Exception:
            cost = None  # price unknown for this model; better empty than wrong

        return TargetOutput(
            output=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            cost_usd=cost,
        )


def build_target(kind: str, config: dict[str, Any]) -> BaseTarget:
    """Turns a TargetVersion row (kind + config JSON) into a real target object."""
    if kind == "echo":
        return EchoTarget(prefix=config.get("prefix", "Echo: "))
    if kind == "http":
        return HttpTarget(
            url=config["url"],
            headers=config.get("headers"),
            timeout_s=config.get("timeout_s", 30.0),
        )
    if kind == "prompt":
        return LLMTarget(
            model=config["model"],
            system_prompt=config.get("prompt"),
            temperature=config.get("temperature", 0.0),
        )
    raise ValueError(f"Unknown target kind: {kind!r}")
