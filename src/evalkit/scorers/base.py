from abc import ABC, abstractmethod
from typing import Any

from evalkit.core.types import EvalCase, ScoreResult, TargetOutput


class BaseScorer(ABC):
    """Contract for anything that marks an answer. Subclasses only write _evaluate()."""

    name: str = "scorer"

    def __init__(self, name: str | None = None) -> None:
        # Optional custom name, so two scorers of the same type
        # (for example two "contains" checks) can be told apart in the database.
        if name:
            self.name = name

    async def score(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        """Public method used by the runner."""
        if result.error is not None:
            # No answer means nothing to mark. Fail it clearly.
            return self._make(False, f"Target failed: {result.error}")
        return await self._evaluate(case, result)

    @abstractmethod
    async def _evaluate(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        """Subclasses write the real checking here."""

    def _make(
        self,
        passed: bool,
        reason: str,
        value: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Small helper so subclasses do not repeat themselves."""
        if value is None:
            value = 1.0 if passed else 0.0
        return ScoreResult(
            scorer_name=self.name,
            value=value,
            passed=passed,
            reason=reason,
            details=details,
        )