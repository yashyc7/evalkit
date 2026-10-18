from sqlalchemy import select

from evalkit.db.models import Target, TargetVersion
from evalkit.db.session import SessionLocal
from evalkit.service import execute_run


def ensure_echo_target() -> None:
    """Creates a practice target the first time. Does nothing afterwards."""
    with SessionLocal() as session:
        if session.scalar(select(Target).where(Target.name == "echo-bot")):
            return
        target = Target(name="echo-bot", kind="echo")
        target.versions = [TargetVersion(label="v1", config={"prefix": "Echo: "})]
        session.add(target)
        session.commit()


def main() -> None:
    ensure_echo_target()
    run = execute_run(
        dataset_name="support-golden-v1",
        target_name="echo-bot",
        version_label="v1",
        scorer_specs=[
            {"type": "contains", "keywords": ["policy"], "name": "mentions_policy"},
            {"type": "max_latency", "max_ms": 1000},
        ],
    )
    print("Run id :", run.id)
    print("Status :", run.status)
    print("Summary:", run.summary)


if __name__ == "__main__":
    main()
