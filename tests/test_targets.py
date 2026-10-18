import pytest

from evalkit.core.targets import BaseTarget, EchoTarget, build_target


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
