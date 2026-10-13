"""Command-line entry point (`volsurface ...`)."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from volsurface import __version__
from volsurface.app.config import Settings
from volsurface.app.headless import (
    collect_updates,
    fast_synthetic,
    format_report,
    record,
    run_benchmarks,
)
from volsurface.app.logs import configure_logging
from volsurface.domain.errors import VolsurfaceError
from volsurface.marketdata.factory import SourceSpec

app = typer.Typer(
    name="volsurface",
    help="Implied-volatility surface workstation.",
    no_args_is_help=True,
    add_completion=False,
)

SourceOption = Annotated[
    str,
    typer.Option(
        "--source",
        "-s",
        help="synthetic[:SYMBOL] | replay:/path/to/recording | ibkr[:SYMBOL]",
    ),
]
ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="TOML settings file (overrides environment)")
]


def _load(config: Path | None) -> Settings:
    settings = Settings.load(config)
    configure_logging(settings.log_level, json=settings.log_json)
    return settings


def _fail(error: Exception) -> typer.Exit:
    typer.secho(f"error: {error}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@app.callback()
def main() -> None:
    """Keep `volsurface <command>` a sub-command interface."""


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def gui(source: SourceOption = "synthetic", config: ConfigOption = None) -> None:
    """Open the desktop application and start streaming from SOURCE."""
    try:
        settings = _load(config)
        spec = SourceSpec.parse(source)
        from volsurface.ui.app import run_gui  # noqa: PLC0415 - Qt is an optional extra
    except ImportError as exc:
        typer.secho("the GUI needs the ui extra: uv sync --extra ui", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except VolsurfaceError as exc:
        raise _fail(exc) from exc
    raise typer.Exit(code=run_gui(settings, spec))


@app.command()
def demo(
    source: SourceOption = "synthetic",
    ticks: Annotated[int, typer.Option(help="number of updates to process")] = 3,
    config: ConfigOption = None,
) -> None:
    """Process a few updates without a window and print the fitted surface, GEX and timings."""
    try:
        settings = _load(config)
        spec = SourceSpec.parse(source)
        options = fast_synthetic() if spec.kind == "synthetic" else None
        updates = asyncio.run(collect_updates(settings, spec, ticks, options=options))
    except VolsurfaceError as exc:
        raise _fail(exc) from exc
    for update in updates:
        typer.echo(format_report(update))
        typer.echo()


@app.command("record")
def record_command(
    out: Annotated[Path, typer.Argument(help="directory to write the recording into")],
    source: SourceOption = "synthetic",
    seconds: Annotated[float, typer.Option(help="how long to record")] = 10.0,
    config: ConfigOption = None,
) -> None:
    """Record SOURCE to parquet segments under OUT (replay it with --source replay:OUT)."""
    try:
        settings = _load(config)
        spec = SourceSpec.parse(source)
        options = fast_synthetic() if spec.kind == "synthetic" else None
        written = asyncio.run(record(settings, spec, out, seconds=seconds, options=options))
    except VolsurfaceError as exc:
        raise _fail(exc) from exc
    typer.echo(f"recorded {written} snapshots to {out}")


@app.command()
def bench() -> None:
    """Time the main numerical kernels on this machine."""
    run_benchmarks(typer.echo)


if __name__ == "__main__":  # pragma: no cover
    app()
