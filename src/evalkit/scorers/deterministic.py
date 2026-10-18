import json

from evalkit.core.types import EvalCase, ScoreResult, TargetOutput, to_text
from evalkit.scorers.base import BaseScorer


class ExactMatchScorer(BaseScorer):
    """Passes only if the answer is the same as case.expected."""

    name = "exact_match"

    def __init__(self, ignore_case: bool = True, name: str | None = None) -> None:
        super().__init__(name)
        self.ignore_case = ignore_case

    async def _evaluate(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        if case.expected is None:
            return self._make(False, "Test case has no expected value to compare.")
        expected = to_text(case.expected).strip()
        actual = to_text(result.output).strip()
        if self.ignore_case:
            expected, actual = expected.lower(), actual.lower()
        passed = expected == actual
        return self._make(
            passed,
            "Output matches expected." if passed else "Output differs from expected.",
            details={"expected": expected, "actual": actual},
        )


class ContainsScorer(BaseScorer):
    """Checks that all given keywords appear in the answer. Gives partial marks."""

    name = "contains"

    def __init__(
        self,
        keywords: list[str],
        ignore_case: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        if not keywords:
            raise ValueError("ContainsScorer needs at least one keyword.")
        self.keywords = keywords
        self.ignore_case = ignore_case

    async def _evaluate(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        text = to_text(result.output)
        if self.ignore_case:
            text = text.lower()
        found, missing = [], []
        for keyword in self.keywords:
            needle = keyword.lower() if self.ignore_case else keyword
            (found if needle in text else missing).append(keyword)
        passed = not missing
        return self._make(
            passed,
            f"Found all {len(found)} keywords."
            if passed
            else f"Missing: {', '.join(missing)}",
            value=len(found) / len(self.keywords),
            details={"found": found, "missing": missing},
        )


class JsonValidScorer(BaseScorer):
    """Passes if the answer is valid JSON (already parsed, or a JSON string)."""

    name = "json_valid"

    async def _evaluate(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        if isinstance(result.output, (dict, list)):
            return self._make(True, "Output is already structured JSON.")
        try:
            json.loads(to_text(result.output))
        except json.JSONDecodeError as exc:
            return self._make(False, f"Output is not valid JSON: {exc.msg}")
        return self._make(True, "Output is valid JSON.")


class MaxLatencyScorer(BaseScorer):
    """Passes if the target answered fast enough."""

    name = "max_latency"

    def __init__(self, max_ms: int, name: str | None = None) -> None:
        super().__init__(name)
        self.max_ms = max_ms

    async def _evaluate(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        if result.latency_ms is None:
            return self._make(False, "No latency was recorded.")
        passed = result.latency_ms <= self.max_ms
        return self._make(
            passed,
            f"Took {result.latency_ms}ms, limit is {self.max_ms}ms.",
            details={"latency_ms": result.latency_ms, "max_ms": self.max_ms},
        )

def _keyword_report(
    text: str, keywords: list[str], ignore_case: bool
) -> tuple[list[str], list[str]]:
    """Returns (found, missing) keyword lists."""
    haystack = text.lower() if ignore_case else text
    found: list[str] = []
    missing: list[str] = []
    for keyword in keywords:
        needle = keyword.lower() if ignore_case else keyword
        (found if needle in haystack else missing).append(keyword)
    return found, missing


class ExpectedContainsScorer(BaseScorer):
    """Each test case carries its OWN keywords in 'expected' (a list, or one string)."""

    name = "expected_contains"

    def __init__(self, ignore_case: bool = True, name: str | None = None) -> None:
        super().__init__(name)
        self.ignore_case = ignore_case

    async def _evaluate(self, case: EvalCase, result: TargetOutput) -> ScoreResult:
        if not case.expected:
            return self._make(False, "Test case has no expected keywords.")
        keywords = case.expected if isinstance(case.expected, list) else [case.expected]
        keywords = [str(keyword) for keyword in keywords]
        found, missing = _keyword_report(
            to_text(result.output), keywords, self.ignore_case
        )
        passed = not missing
        return self._make(
            passed,
            f"Found all {len(found)} expected keywords."
            if passed
            else f"Missing: {', '.join(missing)}",
            value=len(found) / len(keywords),
            details={"found": found, "missing": missing},
        )