import os
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from decimal import Decimal
from typing import Any

import httpx

from evalkit.core.types import TargetOutput, to_text


def compute_cost(
    input_tokens: int,
    output_tokens: int,
    price_per_1m_input: float,
    price_per_1m_output: float,
) -> Decimal:
    """Cost in USD, from token counts and prices per 1 million tokens."""
    total = (
        Decimal(str(price_per_1m_input)) * input_tokens
        + Decimal(str(price_per_1m_output)) * output_tokens
    )
    return total / Decimal(1_000_000)


class BaseTarget(ABC):
    """Contract for anything we can test. Subclasses only write _call()."""

    async def call(self, input: Any) -> TargetOutput:
        """Public method used by the runner. Measures time and never raises."""
        start = time.perf_counter()
        try:
            result = await self._call(input)
        except Exception as exc:
            # One bad question must not crash the whole run.
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
    """Calls an LLM (through LiteLLM) with a system prompt.

    Works with normal providers, and with any OpenAI-compatible server
    (DeepInfra, vLLM, ...) when api_base is given.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        api_base: str | None = None,
        api_key_env: str | None = None,
        price_per_1m_input: float | None = None,
        price_per_1m_output: float | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.api_base = api_base
        self.price_per_1m_input = price_per_1m_input
        self.price_per_1m_output = price_per_1m_output

        # The config only holds the NAME of the environment variable.
        # The secret itself is read here and lives only in memory.
        self.api_key: str | None = None
        if api_key_env:
            self.api_key = os.environ.get(api_key_env)
            if not self.api_key:
                raise ValueError(
                    f"Environment variable {api_key_env!r} is not set. "
                    "Add it to your .env file."
                )

    async def _call(self, input: Any) -> TargetOutput:
        import litellm  # imported here because it is slow to load (a few seconds)

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": to_text(input)})

        extra: dict[str, Any] = {}
        if self.api_base:
            extra["api_base"] = self.api_base
        if self.api_key:
            extra["api_key"] = self.api_key

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            drop_params=True,  # ignore settings a model does not support
            **extra,
        )

        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0

        cost: Decimal | None
        if self.price_per_1m_input is not None and self.price_per_1m_output is not None:
            # Prices you wrote yourself beat guessing.
            cost = compute_cost(
                input_tokens,
                output_tokens,
                self.price_per_1m_input,
                self.price_per_1m_output,
            )
        else:
            try:
                cost = Decimal(
                    str(litellm.completion_cost(completion_response=response))
                )
            except Exception:
                cost = None  # price unknown for this model; better empty than wrong

        return TargetOutput(
            output=response.choices[0].message.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
            api_base=config.get("api_base"),
            api_key_env=config.get("api_key_env"),
            price_per_1m_input=config.get("price_per_1m_input"),
            price_per_1m_output=config.get("price_per_1m_output"),
        )
    raise ValueError(f"Unknown target kind: {kind!r}")