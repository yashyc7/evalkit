import pytest

from evalkit.core.targets import (
    BaseTarget,
    EchoTarget,
    LLMTarget,
    build_target,
    compute_cost,
)
from decimal import Decimal

class BrokenTarget(BaseTarget):
    async def _call(self, input):
        raise RuntimeError("boom")


async def test_echo_target_returns_output_and_latency():
    out = await EchoTarget().call("hi")
    assert out.output == "Echo: hi"
    assert out.error is None
    assert out.latency_ms is not None


async def test_broken_target_never_raises():
    out = await BrokenTarget().call("hi")
    assert out.output is None
    assert "RuntimeError: boom" in out.error
    assert out.latency_ms is not None


def test_build_target_unknown_kind():
    with pytest.raises(ValueError, match="Unknown target kind"):
        build_target("does_not_exist", {})


def test_build_target_echo():
    assert isinstance(build_target("echo", {}), EchoTarget)


def test_prompt_target_needs_its_api_key(monkeypatch):
    monkeypatch.delenv("MY_FAKE_KEY", raising=False)
    with pytest.raises(ValueError, match="MY_FAKE_KEY"):
        build_target("prompt", {"model": "openai/x", "api_key_env": "MY_FAKE_KEY"})


def test_prompt_target_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("MY_FAKE_KEY", "secret")
    target = build_target(
        "prompt",
        {
            "model": "openai/x",
            "api_base": "http://localhost:8000/v1",
            "api_key_env": "MY_FAKE_KEY",
        },
    )
    assert isinstance(target, LLMTarget)


def test_compute_cost_from_token_prices():
    # 2000 input tokens at $0.1 per 1M, 500 output tokens at $0.4 per 1M
    assert compute_cost(2000, 500, 0.1, 0.4) == Decimal("0.0004")