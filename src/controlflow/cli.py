from __future__ import annotations

import typer

from controlflow.data.download import app as data_app

app = typer.Typer(help="ControlFlow-G reproducible local workflows")
app.add_typer(data_app, name="data")


@app.command()
def status() -> None:
    from controlflow.core.state import ProjectPaths

    typer.echo((ProjectPaths.discover().execution_state).read_text(encoding="utf-8"))


@app.command()
def freeze() -> None:
    """Freeze the reviewed candidate; refuses a dirty or already-consumed tree."""
    from controlflow.release import create_freeze

    typer.echo(create_freeze())


@app.command("final-once")
def final_once() -> None:
    """Consume and evaluate the locked holdout exactly once."""
    from controlflow.final_evaluation import run_final_once

    typer.echo(run_final_once())
