import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalkit.config import settings
from evalkit.core.runner import CaseOutcome, run_cases, summarise
from evalkit.core.targets import build_target
from evalkit.core.types import EvalCase
from evalkit.db.models import (
    Dataset,
    Result,
    Run,
    Score,
    Target,
    TargetVersion,
    TestCase,
)
from evalkit.db.session import SessionLocal
from evalkit.scorers import build_scorer


def _save_outcomes(session: Session, run: Run, outcomes: list[CaseOutcome]) -> None:
    """Turns plain outcome objects into Result and Score rows."""
    for outcome in outcomes:
        answer = outcome.output
        result = Result(
            run_id=run.id,
            test_case_id=uuid.UUID(outcome.case_id),
            output=answer.output,
            error=answer.error,
            latency_ms=answer.latency_ms,
            input_tokens=answer.input_tokens,
            output_tokens=answer.output_tokens,
            cost_usd=answer.cost_usd,
        )
        result.scores = [
            Score(
                scorer_name=score.scorer_name,
                value=score.value,
                passed=score.passed,
                reason=score.reason,
                details=score.details,
                judge_model=score.judge_model,
            )
            for score in outcome.scores
        ]
        session.add(result)


def execute_run(
    dataset_name: str,
    target_name: str,
    version_label: str,
    scorer_specs: list[dict[str, Any]],
    git_sha: str | None = None,
    max_concurrency: int | None = None,
    progress: Callable[[CaseOutcome], None] | None = None,
) -> Run:
    """The full job: load, run, mark, save. Returns the finished Run row."""
    with SessionLocal() as session:
        # 1. Load everything we need from the database.
        dataset = session.scalar(select(Dataset).where(Dataset.name == dataset_name))
        if dataset is None:
            raise LookupError(f"Dataset {dataset_name!r} not found.")

        version = session.scalar(
            select(TargetVersion)
            .join(TargetVersion.target)
            .where(Target.name == target_name, TargetVersion.label == version_label)
        )
        if version is None:
            raise LookupError(
                f"Target version {target_name!r} / {version_label!r} not found."
            )

        test_cases = session.scalars(
            select(TestCase)
            .where(TestCase.dataset_id == dataset.id)
            .order_by(TestCase.position)
        ).all()
        if not test_cases:
            raise ValueError(f"Dataset {dataset_name!r} has no test cases.")

        # 2. Build the plain objects. If the config is wrong, we fail HERE,
        #    before any Run row exists, so no half-finished run is left behind.
        cases = [
            EvalCase(
                id=str(tc.id),
                input=tc.input,
                expected=tc.expected,
                tags=list(tc.tags or []),
            )
            for tc in test_cases
        ]
        target = build_target(version.target.kind, version.config)
        scorers = [build_scorer(spec) for spec in scorer_specs]

        # 3. Create the Run row with status "running" and save it straight away,
        #    so a dashboard could already show that it has started.
        run = Run(
            dataset=dataset,
            target_version=version,
            status="running",
            started_at=datetime.now(timezone.utc),
            git_sha=git_sha,
            scorer_config=scorer_specs,
        )
        session.add(run)
        session.commit()

        # 4. Do the real work. Any failure marks the run as "failed".
        try:
            outcomes = asyncio.run(
                run_cases(
                    target,
                    cases,
                    scorers,
                    max_concurrency=max_concurrency or settings.max_concurrency,
                    progress=progress,
                )
            )
            _save_outcomes(session, run, outcomes)
            run.summary = summarise(outcomes)
            run.status = "completed"
            run.finished_at = datetime.now(timezone.utc)
            session.commit()
        except Exception as exc:
            session.rollback()
            run.status = "failed"
            run.finished_at = datetime.now(timezone.utc)
            run.summary = {"error": f"{type(exc).__name__}: {exc}"}
            session.commit()
            raise

        return run
