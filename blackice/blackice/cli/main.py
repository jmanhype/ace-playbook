"""BLACKICE CLI Entry Point.

Usage:
    blackice build <vision> [--model=<model>] [--edition=<edition>]
    blackice status [--run-id=<id>]
    blackice doctor
    blackice version
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import typer
from rich.console import Console

from blackice import __version__
from blackice.cli.commands import build, doctor, receipt, status
from blackice.instrumentation import get_logger

logger = get_logger(__name__)
console = Console()

# Main application
app = typer.Typer(
    name="blackice",
    help="BLACKICE 3.0 - Agentic Software Factory",
    add_completion=False,
    rich_markup_mode="rich",
)

# Register sub-commands
app.command(name="build")(build.build)
app.command(name="status")(status.status)
app.command(name="doctor")(doctor.doctor)

# Register sub-apps
app.add_typer(receipt.app, name="receipt")


@app.command()
def version() -> None:
    """Show BLACKICE version and edition info."""
    from blackice.core.config import BlackiceConfig

    config = BlackiceConfig()

    console.print(f"[bold blue]BLACKICE[/bold blue] [cyan]{__version__}[/cyan]")
    console.print(f"Edition: [yellow]{config.edition.value}[/yellow]")
    console.print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
) -> None:
    """BLACKICE 3.0 - Agentic Software Factory.

    Convert a single human vision into working software with tests,
    documentation, and reproducible artifact trails.
    """
    # Store verbose flag in context for sub-commands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


def run() -> None:
    """CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.exception("CLI error", error=str(e))
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
