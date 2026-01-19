"""Build command - Execute vision to working software.

The core command that drives the BLACKICE flywheel.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from blackice.core.config import BlackiceConfig, Edition
from blackice.flywheel import FlywheelConfig, FlywheelPhase, UnifiedFlywheel
from blackice.instrumentation import get_logger

logger = get_logger(__name__)
console = Console()


def build(
    vision: Annotated[str, typer.Argument(help="Vision description for what to build")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model to use")] = "claude-sonnet-4-20250514",
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Workspace directory")
    ] = None,
    edition: Annotated[
        str,
        typer.Option("--edition", "-e", help="Edition tier (lite, core, enterprise)")
    ] = "lite",
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Custom run ID (auto-generated if not provided)")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without executing")
    ] = False,
    timeout: Annotated[
        float,
        typer.Option("--timeout", "-t", help="Total timeout in seconds")
    ] = 3600.0,
) -> None:
    """Build working software from a vision description.

    Takes a natural language description of what you want to build
    and produces working software with tests and documentation.

    Example:
        blackice build "Create a REST API for user management with JWT auth"
    """
    # Validate edition
    try:
        edition_enum = Edition(edition.lower())
    except ValueError:
        console.print(f"[red]Invalid edition:[/red] {edition}")
        console.print("Valid editions: lite, core, enterprise")
        raise typer.Exit(1)

    # Generate or use provided run ID
    actual_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"

    # Show plan
    console.print(Panel(
        f"[bold]Vision:[/bold] {vision[:200]}{'...' if len(vision) > 200 else ''}\n\n"
        f"[bold]Run ID:[/bold] {actual_run_id}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Edition:[/bold] {edition_enum.value}\n"
        f"[bold]Workspace:[/bold] {workspace or 'auto'}",
        title="[blue]BLACKICE Build[/blue]",
        border_style="blue",
    ))

    if dry_run:
        console.print("\n[yellow]Dry run - no changes will be made[/yellow]")
        return

    # Run the flywheel
    asyncio.run(_run_build(
        vision=vision,
        run_id=actual_run_id,
        model=model,
        workspace=workspace,
        edition=edition_enum,
        timeout=timeout,
    ))


async def _run_build(
    vision: str,
    run_id: str,
    model: str,
    workspace: Path | None,
    edition: Edition,
    timeout: float,
) -> None:
    """Execute the build asynchronously."""
    # Configure flywheel
    config = FlywheelConfig(
        workspace_root=workspace,
        plan_timeout=300.0,
        implement_timeout=timeout / 2,
        test_timeout=300.0,
        verify_timeout=300.0,
    )

    flywheel = UnifiedFlywheel(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting flywheel...", total=None)

        try:
            # Execute flywheel
            result = await flywheel.run(
                run_id=run_id,
                vision=vision,
                context={"model": model, "edition": edition.value},
            )

            # Update progress for each completed phase
            for phase_result in result.phases:
                status = "[green]✓[/green]" if phase_result.success else "[red]✗[/red]"
                progress.update(
                    task,
                    description=f"{status} {phase_result.phase.value}",
                )

        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}[/red]")
            raise

    # Show results
    _show_results(result)


def _show_results(result) -> None:
    """Display build results."""
    if result.success:
        console.print(Panel(
            f"[green]Build completed successfully![/green]\n\n"
            f"[bold]Run ID:[/bold] {result.run_id}\n"
            f"[bold]Duration:[/bold] {result.total_duration:.1f}s\n"
            f"[bold]Phases:[/bold] {result.phase_count}\n"
            f"[bold]Artifacts:[/bold] {len(result.artifacts)}\n"
            f"[bold]Workspace:[/bold] {result.workspace_path}",
            title="[green]Build Complete[/green]",
            border_style="green",
        ))

        if result.artifacts:
            console.print("\n[bold]Generated Artifacts:[/bold]")
            for artifact in result.artifacts[:10]:  # Show first 10
                console.print(f"  • {artifact}")
            if len(result.artifacts) > 10:
                console.print(f"  ... and {len(result.artifacts) - 10} more")
    else:
        console.print(Panel(
            f"[red]Build failed[/red]\n\n"
            f"[bold]Run ID:[/bold] {result.run_id}\n"
            f"[bold]Final Phase:[/bold] {result.final_phase.value}\n"
            f"[bold]Duration:[/bold] {result.total_duration:.1f}s",
            title="[red]Build Failed[/red]",
            border_style="red",
        ))

        # Show failed phases
        for phase_result in result.failed_phases:
            console.print(f"\n[red]Phase {phase_result.phase.value} failed:[/red]")
            if phase_result.error:
                console.print(f"  Error: {phase_result.error}")
