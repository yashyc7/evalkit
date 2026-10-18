import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from evalkit.core.targets import BaseTarget
from evalkit.core.types import EvalCase, ScoreResult, TargetOutput
from evalkit.scorers.base import BaseScorer


@dataclass(frozen=True)
class CaseOutcome:
    """Everything we learned about ONE question: the answer and all its marks."""

    case_id: str
    output: TargetOutput
    scores: list[ScoreResult]


async def _safe_score(
    scorer: BaseScorer, case: EvalCase, output: TargetOutput
) -> ScoreResult:
    """Runs one scorer. If the scorer itself crashes, we record a failed mark."""
    try:
        return await scorer.score(case, output)
    except Exception as exc:
        return ScoreResult(
            scorer_name=scorer.name,
            value=0.0,
            passed=False,
            reason=f"Scorer crashed: {type(exc).__name__}: {exc}",
        )


async def run_cases(
    target: BaseTarget,
    cases: list[EvalCase],
    scorers: list[BaseScorer],
    max_concurrency: int = 8,
    progress: Callable[[CaseOutcome], None] | None = None,
) -> list[CaseOutcome]:
    """Ask every question, mark every answer. At most max_concurrency at a time."""
    semaphore = asyncio.Semaphore(max_concurrency)  # the room with limited chairs

    async def run_one(case: EvalCase) -> CaseOutcome:
        async with semaphore:  # wait here until a chair is free
            output = await target.call(case.input)
            scores = await asyncio.gather(
                *(_safe_score(scorer, case, output) for scorer in scorers)
            )
        outcome = CaseOutcome(case_id=case.id, output=output, scores=list(scores))
        if progress is not None:
            progress(outcome)  # lets the CLI draw a progress bar
        return outcome

    # gather starts all of them together and returns results in the SAME order
    # as the cases, even if they finish in a different order.
    return list(await asyncio.gather(*(run_one(case) for case in cases)))


def _percentile(sorted_values: list[int], pct: int) -> int | None:
    """Nearest-rank percentile. Input must already be sorted."""
    if not sorted_values:
        return None
    rank = math.ceil(pct / 100 * len(sorted_values))
    return sorted_values[max(rank - 1, 0)]


def summarise(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    """Turns all outcomes into the totals we store in runs.summary."""
    total = len(outcomes)
    if total == 0:
        return {"total": 0, "passed": 0, "errors": 0, "pass_rate": None, "scorers": {}}

    errors = sum(1 for o in outcomes if o.output.error is not None)
    # A question passes only if the target did not crash AND every scorer passed.
    passed = sum(
        1
        for o in outcomes
        if o.output.error is None and all(s.passed for s in o.scores)
    )

    latencies = sorted(
        o.output.latency_ms for o in outcomes if o.output.latency_ms is not None
    )
    costs = [o.output.cost_usd for o in outcomes if o.output.cost_usd is not None]
    input_tokens = sum(o.output.input_tokens or 0 for o in outcomes)
    output_tokens = sum(o.output.output_tokens or 0 for o in outcomes)

    # Group all marks by scorer name, so we can say "faithfulness averaged 0.9".
    by_scorer: dict[str, list[ScoreResult]] = {}
    for outcome in outcomes:
        for score in outcome.scores:
            by_scorer.setdefault(score.scorer_name, []).append(score)

    return {
        "total": total,
        "passed": passed,
        "errors": errors,
        "pass_rate": round(passed / total, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 95),
        "total_cost_usd": float(sum(costs, Decimal(0))) if costs else None,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "scorers": {
            name: {
                "count": len(scores),
                "avg_value": round(sum(s.value for s in scores) / len(scores), 4),
                "pass_rate": round(sum(1 for s in scores if s.passed) / len(scores), 4),
            }
            for name, scores in by_scorer.items()
        },
    }
