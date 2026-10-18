import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# Base and mixins (shared building blocks)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Every model inherits from this. SQLAlchemy keeps a list of all tables here."""


class UUIDMixin:
    """Gives every table an 'id' column filled with a random unique id."""

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """Gives every table a 'created_at' column filled by the database itself."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# 1. Dataset: a named collection of test cases
# ---------------------------------------------------------------------------
class Dataset(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(unique=True)
    description: Mapped[str | None]

    test_cases: Mapped[list["TestCase"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    runs: Mapped[list["Run"]] = relationship(back_populates="dataset")


# ---------------------------------------------------------------------------
# 2. TestCase: one question (input) with its expected answer
# ---------------------------------------------------------------------------
class TestCase(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "test_cases"
    __test__ = False  # tells pytest: "this is not a test class, do not collect it"

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int]
    input: Mapped[Any] = mapped_column(JSONB)
    expected: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSONB, default=list)

    dataset: Mapped["Dataset"] = relationship(back_populates="test_cases")
    results: Mapped[list["Result"]] = relationship(back_populates="test_case")


# ---------------------------------------------------------------------------
# 3. Target: the AI system we are testing (e.g. "support-bot")
# ---------------------------------------------------------------------------
class Target(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "targets"

    name: Mapped[str] = mapped_column(unique=True)
    kind: Mapped[str]  # "http", "python" or "prompt"

    versions: Mapped[list["TargetVersion"]] = relationship(
        back_populates="target", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# 4. TargetVersion: one exact, frozen configuration of a Target
# ---------------------------------------------------------------------------
class TargetVersion(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "target_versions"
    __table_args__ = (
        UniqueConstraint("target_id", "label", name="uq_target_version_label"),
    )

    target_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("targets.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str]  # "v1", "v2", ...
    config: Mapped[Any] = mapped_column(JSONB)

    target: Mapped["Target"] = relationship(back_populates="versions")
    runs: Mapped[list["Run"]] = relationship(back_populates="target_version")


# ---------------------------------------------------------------------------
# 5. Run: one execution of (one TargetVersion) x (one Dataset)
# ---------------------------------------------------------------------------
class Run(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "runs"

    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"), index=True)
    target_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("target_versions.id"), index=True
    )
    status: Mapped[str] = mapped_column(default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    git_sha: Mapped[str | None]
    summary: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    scorer_config: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    dataset: Mapped["Dataset"] = relationship(back_populates="runs")
    target_version: Mapped["TargetVersion"] = relationship(back_populates="runs")
    results: Mapped[list["Result"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# 6. Result: what the target answered for ONE test case in ONE run
# ---------------------------------------------------------------------------
class Result(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "results"
    __table_args__ = (
        UniqueConstraint("run_id", "test_case_id", name="uq_result_run_test_case"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    test_case_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("test_cases.id"), index=True
    )
    output: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    run: Mapped["Run"] = relationship(back_populates="results")
    test_case: Mapped["TestCase"] = relationship(back_populates="results")
    scores: Mapped[list["Score"]] = relationship(
        back_populates="result", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# 7. Score: one scorer's marks for one result
# ---------------------------------------------------------------------------
class Score(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "scores"

    result_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("results.id", ondelete="CASCADE"), index=True
    )
    scorer_name: Mapped[str] = mapped_column(index=True)
    value: Mapped[float]  # 0.0 (worst) to 1.0 (best)
    passed: Mapped[bool]
    reason: Mapped[str | None] = mapped_column(Text)
    details: Mapped[Any] = mapped_column(JSONB(none_as_null=True), nullable=True)
    judge_model: Mapped[str | None]

    result: Mapped["Result"] = relationship(back_populates="scores")
