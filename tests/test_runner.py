import asyncio
from decimal import Decimal

import pytest

from evalkit.core.runner import CaseOutcome, run_cases, summarise
from evalkit.core.targets import BaseTarget, EchoTarget
from evalkit.core.types import EvalCase, ScoreResult, TargetOutput
from evalkit.scorers.base import BaseScorer
from evalkit.scorers.deterministic import ContainsScorer


def make_cases(n):
    return [EvalCase(id=f"c{i}", input=f"question {i}") for i in range(n)]


class CountingTarget(BaseTarget):
    """Remembers the highest number of questions it handled at the same moment."""

    def __init__(self):
        self.active = 0
        self.peak = 0

    async def _call(self, input):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return TargetOutput(output=str(input))


class CrashingScorer(BaseScorer):
    name = "crasher"

    async def _evaluate(self, case, result):
        raise RuntimeError("bad scorer")


async def test_results_keep_the_order_of_cases():
    outcomes = await run_cases(EchoTarget(), make_cases(5), [])
    assert [o.case_id for o in outcomes] == ["c0", "c1", "c2", "c3", "c4"]
    assert outcomes[2].output.output == "Echo: question 2"


async def test_concurrency_limit_is_respected():
    target = CountingTarget()
    await run_cases(target, make_cases(10), [], max_concurrency=3)
    assert 1 < target.peak <= 3


async def test_every_case_gets_a_score_from_every_scorer():
    scorers = [ContainsScorer(["question"]), ContainsScorer(["zzz"], name="has_zzz")]
    outcomes = await run_cases(EchoTarget(), make_cases(3), scorers)
    assert all(len(o.scores) == 2 for o in outcomes)
    assert [s.scorer_name for s in outcomes[0].scores] == ["contains", "has_zzz"]


async def test_crashing_scorer_does_not_kill_the_run():
    outcomes = await run_cases(EchoTarget(), make_cases(2), [CrashingScorer()])
    score = outcomes[0].scores[0]
    assert score.passed is False
    assert "Scorer crashed" in score.reason


async def test_progress_callback_is_called_once_per_case():
    seen = []
    await run_cases(EchoTarget(), make_cases(4), [], progress=seen.append)
    assert len(seen) == 4


def outcome(case_id, error=None, latency=100, cost=None, passed=True):
    output = TargetOutput(
        output=None if error else "ok",
        error=error,
        latency_ms=latency,
        input_tokens=10,
        output_tokens=5,
        cost_usd=cost,
    )
    score = ScoreResult(scorer_name="a", value=1.0 if passed else 0.0, passed=passed)
    return CaseOutcome(case_id=case_id, output=output, scores=[score])


def test_summarise_numbers():
    outcomes = [
        outcome("c1", latency=100, cost=Decimal("0.001")),
        outcome("c2", error="Timeout", latency=300, passed=False),
        outcome("c3", latency=200),
    ]
    summary = summarise(outcomes)
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["errors"] == 1
    assert summary["pass_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert summary["avg_latency_ms"] == 200
    assert summary["p95_latency_ms"] == 300
    assert summary["total_cost_usd"] == pytest.approx(0.001)
    assert summary["total_input_tokens"] == 30
    assert summary["scorers"]["a"]["pass_rate"] == pytest.approx(0.6667, abs=1e-4)


def test_summarise_empty_run():
    assert summarise([])["total"] == 0
