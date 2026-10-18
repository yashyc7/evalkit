import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """One question to ask the target (plain-Python version of a TestCase row)."""

    id: str
    input: Any
    expected: Any = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TargetOutput:
    """What came back from the target (plain-Python version of a Result row)."""

    output: Any = None
    error: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: Decimal | None = None


@dataclass(frozen=True)
class ScoreResult:
    """One scorer's marks for one answer (plain-Python version of a Score row)."""

    scorer_name: str
    value: float
    passed: bool
    reason: str | None = None
    details: dict[str, Any] | None = None
    judge_model: str | None = None


def to_text(value: Any) -> str:
    """Turn any output (text, dict, list, None) into a plain string for comparing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)