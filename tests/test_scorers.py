import pytest

from evalkit.core.types import EvalCase, TargetOutput
from evalkit.scorers import build_scorer
from evalkit.scorers.deterministic import (
    ContainsScorer,
    ExactMatchScorer,
    JsonValidScorer,
    MaxLatencyScorer,
)


def case(expected=None):
    return EvalCase(id="c1", input="question", expected=expected)


async def test_exact_match_ignores_case_and_spaces():
    result = await ExactMatchScorer().score(
        case("Hello"), TargetOutput(output="  hello ")
    )
    assert result.passed is True
    assert result.value == 1.0


async def test_exact_match_fails_when_different():
    result = await ExactMatchScorer().score(case("Hello"), TargetOutput(output="Bye"))
    assert result.passed is False
    assert result.value == 0.0


async def test_exact_match_fails_without_expected():
    result = await ExactMatchScorer().score(case(None), TargetOutput(output="anything"))
    assert result.passed is False
    assert "no expected" in result.reason


async def test_contains_gives_partial_credit():
    scorer = ContainsScorer(["30 days", "refund"])
    result = await scorer.score(case(), TargetOutput(output="Return within 30 days."))
    assert result.passed is False
    assert result.value == 0.5
    assert result.details["missing"] == ["refund"]


async def test_contains_passes_when_all_found():
    scorer = ContainsScorer(["30 days"])
    result = await scorer.score(case(), TargetOutput(output="Returns within 30 DAYS"))
    assert result.passed is True
    assert result.value == 1.0


async def test_json_valid_accepts_json_text():
    result = await JsonValidScorer().score(case(), TargetOutput(output='{"a": 1}'))
    assert result.passed is True


async def test_json_valid_rejects_plain_text():
    result = await JsonValidScorer().score(case(), TargetOutput(output="not json"))
    assert result.passed is False


async def test_max_latency_passes_when_fast():
    result = await MaxLatencyScorer(1000).score(
        case(), TargetOutput(output="x", latency_ms=500)
    )
    assert result.passed is True


async def test_max_latency_fails_when_slow():
    result = await MaxLatencyScorer(1000).score(
        case(), TargetOutput(output="x", latency_ms=1500)
    )
    assert result.passed is False
    assert "1500" in result.reason


@pytest.mark.parametrize(
    "scorer",
    [
        ExactMatchScorer(),
        ContainsScorer(["x"]),
        JsonValidScorer(),
        MaxLatencyScorer(1000),
    ],
)
async def test_failed_target_fails_every_scorer(scorer):
    result = await scorer.score(case("x"), TargetOutput(error="Timeout after 30s"))
    assert result.passed is False
    assert "Timeout" in result.reason


def test_build_scorer_from_dict():
    scorer = build_scorer({"type": "contains", "keywords": ["a"]})
    assert isinstance(scorer, ContainsScorer)
    assert scorer.name == "contains"


def test_build_scorer_custom_name():
    scorer = build_scorer({"type": "contains", "keywords": ["a"], "name": "mentions_a"})
    assert scorer.name == "mentions_a"


def test_build_scorer_unknown_type():
    with pytest.raises(ValueError, match="Unknown scorer type"):
        build_scorer({"type": "does_not_exist"})
