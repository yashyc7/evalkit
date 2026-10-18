from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select

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


def seed() -> None:
    # "with" opens a session and closes it automatically at the end.
    with SessionLocal() as session:
        # Safety check: run this script twice and it must not create duplicates.
        already = session.scalar(
            select(Dataset).where(Dataset.name == "support-golden-v1")
        )
        if already:
            print("Seed data already exists. Nothing to do.")
            return

        # 1. Dataset with three test cases (the "exam paper")
        dataset = Dataset(
            name="support-golden-v1",
            description="Common customer questions with correct answers",
        )
        q1 = TestCase(
            position=1,
            input="What is your return policy?",
            expected="Returns are accepted within 30 days.",
            tags=["policy"],
        )
        q2 = TestCase(
            position=2,
            input="Where is my order 1234?",
            expected={"must_call_tool": "track_order"},
            tags=["orders", "tools"],
        )
        q3 = TestCase(
            position=3,
            input="Tell me a joke",
            expected=None,
            tags=["off-topic"],
        )
        dataset.test_cases = [q1, q2, q3]

        # 2. Target with two versions (the "student", before and after a prompt change)
        target = Target(name="support-bot", kind="http")
        v1 = TargetVersion(
            label="v1",
            config={
                "url": "http://localhost:9000/chat",
                "model": "gpt-4o-mini",
                "prompt": "You are a helpful shop assistant.",
            },
        )
        v2 = TargetVersion(
            label="v2",
            config={
                "url": "http://localhost:9000/chat",
                "model": "gpt-4o-mini",
                "prompt": "You are a polite, concise shop assistant. Never guess.",
            },
        )
        target.versions = [v1, v2]

        # 3. One run: v2 taking the exam (one "exam sitting")
        now = datetime.now(timezone.utc)
        run = Run(
            dataset=dataset,
            target_version=v2,
            status="completed",
            started_at=now - timedelta(minutes=2),
            finished_at=now,
            git_sha="a1b2c3d",
            summary={
                "total": 3,
                "passed": 1,
                "errors": 1,
                "pass_rate": 0.33,
                "avg_latency_ms": 10423,
                "total_cost_usd": 0.000031,
            },
        )

        # 4. Results: one answer per question (the "answer sheet")
        r1 = Result(
            run=run,
            test_case=q1,
            output="You can return items within 30 days.",
            latency_ms=820,
            input_tokens=45,
            output_tokens=12,
            cost_usd=Decimal("0.000031"),
        )
        r2 = Result(
            run=run,
            test_case=q2,
            error="Timeout after 30s",
            latency_ms=30000,
        )
        r3 = Result(
            run=run,
            test_case=q3,
            output="Why did the chicken cross the road?",
            latency_ms=450,
        )

        # 5. Scores: the teacher's marks
        r1.scores = [
            Score(
                scorer_name="contains_keyword",
                value=1.0,
                passed=True,
                reason="Found '30 days'",
            ),
            Score(
                scorer_name="faithfulness",
                value=0.9,
                passed=True,
                reason="Answer matches policy, minor rewording.",
                judge_model="gpt-4o-mini",
            ),
        ]
        r3.scores = [
            Score(
                scorer_name="stays_on_topic",
                value=0.0,
                passed=False,
                reason="Bot answered an off-topic question.",
            ),
        ]
        # r2 crashed with a timeout, so it has no scores.

        # Because of our relationships and cascades, adding these three
        # objects automatically saves everything connected to them.
        session.add_all([dataset, target, run])
        session.commit()
        print("Seed data inserted.")


if __name__ == "__main__":
    seed()
