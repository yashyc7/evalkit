import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from evalkit.core.targets import build_target
from evalkit.db.models import Dataset, Target, TargetVersion, TestCase
from evalkit.db.session import SessionLocal


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Reads text where each line is one JSON test case. Pure function, no database."""
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue  # blank lines are fine
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON ({exc.msg}).") from exc
        if not isinstance(row, dict) or "input" not in row:
            raise ValueError(
                f"Line {line_number}: every line must be a JSON object "
                "with an 'input' field."
            )
        cases.append(
            {
                "input": row["input"],
                "expected": row.get("expected"),
                "tags": list(row.get("tags", [])),
            }
        )
    if not cases:
        raise ValueError("The file has no test cases.")
    return cases


def import_dataset(name: str, file_path: Path, description: str | None = None) -> int:
    """Creates a Dataset from a JSONL file. Returns how many test cases were saved."""
    # utf-8-sig also accepts files that Windows Notepad saved with a hidden BOM mark.
    cases = parse_jsonl(file_path.read_text(encoding="utf-8-sig"))
    with SessionLocal() as session:
        if session.scalar(select(Dataset).where(Dataset.name == name)):
            raise ValueError(
                f"Dataset {name!r} already exists. Datasets are never edited, "
                f"so use a new name, for example {name}-v2."
            )
        dataset = Dataset(name=name, description=description)
        dataset.test_cases = [
            TestCase(
                position=position,
                input=case["input"],
                expected=case["expected"],
                tags=case["tags"],
            )
            for position, case in enumerate(cases, start=1)
        ]
        session.add(dataset)
        session.commit()
    return len(cases)


def count_test_cases(dataset_name: str) -> int:
    """How many questions a dataset has. The CLI needs it for the progress bar."""
    with SessionLocal() as session:
        dataset = session.scalar(select(Dataset).where(Dataset.name == dataset_name))
        if dataset is None:
            raise LookupError(f"Dataset {dataset_name!r} not found.")
        return session.scalar(
            select(func.count())
            .select_from(TestCase)
            .where(TestCase.dataset_id == dataset.id)
        )


def add_target_version(
    name: str, kind: str, label: str, config: dict[str, Any]
) -> None:
    """Registers a new immutable version. Creates the Target too if it is new."""
    # Check the config now, so a typo fails here and not in the middle of a run.
    try:
        build_target(kind, config)
    except KeyError as exc:
        raise ValueError(f"The config is missing the key {exc}.") from exc

    with SessionLocal() as session:
        target = session.scalar(select(Target).where(Target.name == name))
        if target is None:
            target = Target(name=name, kind=kind)
            session.add(target)
        elif target.kind != kind:
            raise ValueError(
                f"Target {name!r} already exists with kind {target.kind!r}, "
                f"not {kind!r}."
            )
        if any(version.label == label for version in target.versions):
            raise ValueError(
                f"Version {label!r} of {name!r} already exists. Versions are "
                "never edited, so use a new label."
            )
        target.versions.append(TargetVersion(label=label, config=config))
        session.commit()