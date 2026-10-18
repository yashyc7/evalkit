import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import String, cast, select
from sqlalchemy.orm import selectinload

from evalkit.db.models import Result, Run, TargetVersion
from evalkit.db.session import SessionLocal


@dataclass(frozen=True)
class RunRow:
    """One line of the 'evalkit runs' table."""

    id: uuid.UUID
    created_at: datetime
    dataset: str
    target: str
    version: str
    status: str
    summary: dict[str, Any] | None


def resolve_run_id(prefix: str) -> uuid.UUID:
    """Lets you type the first few characters of a run id instead of all 36."""
    with SessionLocal() as session:
        ids = session.scalars(
            select(Run.id).where(cast(Run.id, String).startswith(prefix.lower()))
        ).all()
    if not ids:
        raise LookupError(f"No run found starting with {prefix!r}.")
    if len(ids) > 1:
        raise LookupError(f"{prefix!r} matches {len(ids)} runs. Type more characters.")
    return ids[0]


def list_runs(limit: int = 10) -> list[RunRow]:
    """Newest runs first."""
    with SessionLocal() as session:
        runs = session.scalars(
            select(Run)
            .options(
                selectinload(Run.dataset),
                selectinload(Run.target_version).selectinload(TargetVersion.target),
            )
            .order_by(Run.created_at.desc())
            .limit(limit)
        ).all()
        return [
            RunRow(
                id=run.id,
                created_at=run.created_at,
                dataset=run.dataset.name,
                target=run.target_version.target.name,
                version=run.target_version.label,
                status=run.status,
                summary=run.summary,
            )
            for run in runs
        ]


def get_run_report(run_id: uuid.UUID) -> dict[str, Any]:
    """Everything about one run, ready to print: totals plus every question."""
    with SessionLocal() as session:
        run = session.scalar(
            select(Run)
            .where(Run.id == run_id)
            .options(
                selectinload(Run.dataset),
                selectinload(Run.target_version).selectinload(TargetVersion.target),
                selectinload(Run.results).selectinload(Result.scores),
                selectinload(Run.results).selectinload(Result.test_case),
            )
        )
        if run is None:
            raise LookupError(f"Run {run_id} not found.")

        cases = []
        for result in sorted(run.results, key=lambda r: r.test_case.position):
            cases.append(
                {
                    "position": result.test_case.position,
                    "input": result.test_case.input,
                    "expected": result.test_case.expected,
                    "output": result.output,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                    "scores": [
                        {
                            "name": score.scorer_name,
                            "value": score.value,
                            "passed": score.passed,
                            "reason": score.reason,
                        }
                        for score in result.scores
                    ],
                }
            )
        return {
            "id": run.id,
            "status": run.status,
            "dataset": run.dataset.name,
            "target": run.target_version.target.name,
            "version": run.target_version.label,
            "git_sha": run.git_sha,
            "created_at": run.created_at,
            "scorer_config": run.scorer_config,
            "summary": run.summary,
            "cases": cases,
        }