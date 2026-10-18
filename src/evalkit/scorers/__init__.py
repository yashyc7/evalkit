from typing import Any

from evalkit.scorers.base import BaseScorer
from evalkit.scorers.deterministic import (
    ContainsScorer,
    ExactMatchScorer,
    JsonValidScorer,
    MaxLatencyScorer,
)

SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    scorer_class.name: scorer_class
    for scorer_class in (
        ExactMatchScorer,
        ContainsScorer,
        JsonValidScorer,
        MaxLatencyScorer,
    )
}


def build_scorer(spec: dict[str, Any]) -> BaseScorer:
    """Create a scorer from a dict like {"type": "contains", "keywords": ["a"]}."""
    params = dict(spec)  # copy, so we do not change the caller's dict
    scorer_type = params.pop("type")
    try:
        scorer_class = SCORER_REGISTRY[scorer_type]
    except KeyError:
        raise ValueError(
            f"Unknown scorer type {scorer_type!r}. Available: {sorted(SCORER_REGISTRY)}"
        ) from None
    return scorer_class(**params)
