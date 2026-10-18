import json
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from evalkit import admin, reports, service
from evalkit.core.types import to_text

load_dotenv()  # makes API keys from the .env file visible to the LLM library

console = Console()
app = typer.Typer(
    help="evalkit: evaluate and regression-test LLM apps.", no_args_is_help=True
)
dataset_app = typer.Typer(help="Manage datasets.", no_args_is_help=True)
target_app = typer.Typer(help="Manage targets (the systems under test).", no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(target_app, name="target")


# ---------------------------------------------------------------- helpers
def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        _fail(f"{path} is not valid JSON: {exc.msg} (line {exc.lineno}).")


def _fmt(value: Any, template: str, default: str = "n/a") -> str:
    return default if value is None else template.format(value)


def _short(value: Any, width: int) -> str:
    text = to_text(value).replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"] or {}
    console.print(
        f"\n[bold]Run {str(report['id'])[:8]}[/bold]  "
        f"{report['target']}/{report['version']} on {report['dataset']}  "
        f"(status: {report['status']})"
    )
    if "error" in summary:
        console.print(f"[red]Run failed:[/red] {summary['error']}")
        return

    pass_rate = _fmt(summary.get("pass_rate"), "{:.1%}")
    totals = Table(title="Summary", show_header=False)
    totals.add_column(style="bold")
    totals.add_column()
    totals.add_row(
        "Pass rate",
        f"{pass_rate}  ({summary.get('passed')} of {summary.get('total')} questions)",
    )
    totals.add_row("Errors", str(summary.get("errors")))
    totals.add_row("Average latency", _fmt(summary.get("avg_latency_ms"), "{} ms"))
    totals.add_row("p95 latency", _fmt(summary.get("p95_latency_ms"), "{} ms"))
    totals.add_row("Total cost", _fmt(summary.get("total_cost_usd"), "${:.6f}"))
    totals.add_row(
        "Tokens in / out",
        f"{summary.get('total_input_tokens')} / {summary.get('total_output_tokens')}",
    )
    console.print(totals)

    scorers = Table(title="Scorers")
    scorers.add_column("Scorer")
    scorers.add_column("Average value", justify="right")
    scorers.add_column("Pass rate", justify="right")
    for name, stats in summary.get("scorers", {}).items():
        scorers.add_row(
            name,
            f"{stats['avg_value']:.2f}",
            _fmt(stats["pass_rate"], "{:.1%}"),
        )
    console.print(scorers)

    cases = Table(title="Questions")
    cases.add_column("#", justify="right")
    cases.add_column("Question")
    cases.add_column("Answer")
    cases.add_column("Latency", justify="right")
    cases.add_column("Result")
    cases.add_column("Why it failed")
    for case in report["cases"]:
        failed = [s for s in case["scores"] if not s["passed"]]
        if case["error"]:
            status = "[red]ERROR[/red]"
        elif failed:
            status = "[red]FAIL[/red]"
        else:
            status = "[green]PASS[/green]"
        shown = case["error"] if case["error"] else case["output"]
        cases.add_row(
            str(case["position"]),
            _short(case["input"], 34),
            _short(shown, 44),
            _fmt(case["latency_ms"], "{} ms"),
            status,
            "; ".join(f"{s['name']}: {s['reason']}" for s in failed),
        )
    console.print(cases)


# --------------------------------------------------------------- commands
@dataset_app.command("import")
def dataset_import(
    file: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="JSONL file, one test case per line."),
    ],
    name: Annotated[str, typer.Option("--name", "-n", help="Name for the new dataset.")],
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
) -> None:
    """Create a dataset from a JSONL file."""
    try:
        count = admin.import_dataset(name, file, description)
    except (ValueError, LookupError) as exc:
        _fail(str(exc))
    console.print(f"[green]Imported[/green] {count} test cases into [bold]{name}[/bold].")


@target_app.command("add")
def target_add(
    name: Annotated[str, typer.Argument(help="Target name, for example shopeasy-bot.")],
    kind: Annotated[str, typer.Option("--kind", "-k", help="echo, http or prompt.")],
    label: Annotated[str, typer.Option("--label", "-l", help="Version label, for example v1.")],
    config_file: Annotated[
        Path,
        typer.Option("--config-file", "-c", exists=True, dir_okay=False, help="JSON config file."),
    ],
) -> None:
    """Register a new version of a target."""
    config = _read_json(config_file)
    try:
        admin.add_target_version(name, kind, label, config)
    except (ValueError, LookupError) as exc:
        _fail(str(exc))
    console.print(f"[green]Added[/green] version [bold]{label}[/bold] of [bold]{name}[/bold].")


@app.command()
def run(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset name.")],
    target: Annotated[str, typer.Option("--target", "-t", help="Target name.")],
    version: Annotated[str, typer.Option("--version", "-v", help="Target version label.")],
    scorers: Annotated[
        Path,
        typer.Option("--scorers", "-s", exists=True, dir_okay=False, help="JSON file with the scorer list."),
    ],
    git_sha: Annotated[str | None, typer.Option("--git-sha", help="Commit id, useful in CI.")] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", "-c", min=1, help="Questions asked at the same time.")
    ] = None,
) -> None:
    """Ask every question of a dataset to a target version and mark the answers."""
    specs = _read_json(scorers)
    try:
        total = admin.count_test_cases(dataset)
        with Progress(
            TextColumn("[bold]Running[/bold]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as bar:
            task = bar.add_task("run", total=total)
            finished = service.execute_run(
                dataset_name=dataset,
                target_name=target,
                version_label=version,
                scorer_specs=specs,
                git_sha=git_sha,
                max_concurrency=concurrency,
                progress=lambda outcome: bar.advance(task),
            )
    except (ValueError, LookupError) as exc:
        _fail(str(exc))
    _print_report(reports.get_run_report(finished.id))


@app.command("runs")
def runs_list(
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="How many runs to show.")] = 10,
) -> None:
    """List the newest runs."""
    table = Table(title="Runs")
    for column in ("Id", "When", "Target", "Dataset", "Status", "Pass rate", "Avg latency", "Cost"):
        table.add_column(column)
    for row in reports.list_runs(limit):
        summary = row.summary or {}
        table.add_row(
            str(row.id)[:8],
            row.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            f"{row.target}/{row.version}",
            row.dataset,
            row.status,
            _fmt(summary.get("pass_rate"), "{:.1%}"),
            _fmt(summary.get("avg_latency_ms"), "{} ms"),
            _fmt(summary.get("total_cost_usd"), "${:.6f}"),
        )
    console.print(table)


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run id, or its first few characters.")],
) -> None:
    """Show the full report of one run."""
    try:
        full_id = reports.resolve_run_id(run_id)
        report = reports.get_run_report(full_id)
    except LookupError as exc:
        _fail(str(exc))
    _print_report(report)


if __name__ == "__main__":
    app()