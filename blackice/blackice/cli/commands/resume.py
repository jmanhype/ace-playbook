"""Resume command for BLACKICE 3.0.

Provides crash recovery resume functionality:
- Resume crashed or interrupted runs
- Skip completed work
- Retry in-flight tasks with new attempts
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from blackice.persistence.event_store import EventStore, EventStoreConfig
from blackice.persistence.projections import (
    RunProjection,
    RunStatus,
    TaskProjection,
    reconstruct_run_state,
    reconstruct_task_states,
)
from blackice.recovery.checkpoint import CheckpointManager
from blackice.recovery.resume import ResumeManager

console = Console()
app = typer.Typer(help="Resume crashed or interrupted runs")


def get_status_style(status: str) -> str:
    """Get Rich style for status."""
    styles = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
        "pending": "dim",
        "executing": "yellow",
        "planning": "blue",
        "verifying": "cyan",
        "skipped": "dim",
    }
    return styles.get(status.lower(), "white")


@app.command("check")
def check_resumable(
    run_id: Annotated[str, typer.Argument(help="Run ID to check")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
) -> None:
    """Check if a run can be resumed.

    Analyzes the run state to determine if resume is possible
    and what work would be skipped or retried.
    """
    import asyncio

    async def _check() -> None:
        # Initialize stores
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        checkpoint_dir = config.storage_dir.parent / "checkpoints"
        resume_mgr = ResumeManager(store, checkpoint_dir)

        try:
            # Check if resumable
            can_resume, reason = await resume_mgr.can_resume(run_id)

            if not can_resume:
                console.print(
                    Panel(
                        f"[red]Cannot resume: {reason}[/red]",
                        title="[bold red]Not Resumable[/bold red]",
                        border_style="red",
                    )
                )
                raise typer.Exit(1)

            # Get resume state
            resume_state = await resume_mgr.get_resume_state(run_id)

            # Get run state
            events = await store.get_events(run_id)
            run_state = reconstruct_run_state(events, run_id)

            # Display resume information
            console.print(
                Panel(
                    f"[bold]Run ID:[/bold] {run_id}\n"
                    f"[bold]Current Status:[/bold] [{get_status_style(run_state.status.value)}]{run_state.status.value}[/]\n"
                    f"[bold]Events Processed:[/bold] {run_state.event_count}\n"
                    f"[bold]Last Checkpoint:[/bold] {resume_state.last_checkpoint_id or 'None'}\n"
                    f"[bold]Resume Sequence:[/bold] {resume_state.resume_from_sequence}\n\n"
                    f"[green]✓ Completed Tasks:[/green] {len(resume_state.completed_tasks)}\n"
                    f"[yellow]→ In-Flight Tasks:[/yellow] {len(resume_state.in_flight_tasks)}\n"
                    f"[dim]○ Idempotency Keys:[/dim] {len(resume_state.used_idempotency_keys)}",
                    title="[bold blue]Resume Analysis[/bold blue]",
                    border_style="blue",
                )
            )

            # Show task breakdown
            if resume_state.completed_tasks or resume_state.in_flight_tasks:
                table = Table(title="Task State")
                table.add_column("Task", style="cyan")
                table.add_column("Status")
                table.add_column("On Resume")

                for task in sorted(resume_state.completed_tasks):
                    table.add_row(task, "[green]completed[/green]", "[dim]skip[/dim]")

                for task in sorted(resume_state.in_flight_tasks):
                    next_attempt = resume_state.get_next_attempt(task)
                    table.add_row(
                        task,
                        "[yellow]in-flight[/yellow]",
                        f"[yellow]retry (attempt {next_attempt})[/yellow]",
                    )

                console.print(table)

            console.print("\n[green]✓ Run can be resumed[/green]")

        finally:
            await store.close()

    asyncio.run(_check())


@app.command("run")
def resume_run(
    run_id: Annotated[str, typer.Argument(help="Run ID to resume")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Show what would be done without executing"),
    ] = False,
    create_checkpoint: Annotated[
        bool,
        typer.Option("--checkpoint", "-c", help="Create checkpoint before resuming"),
    ] = True,
) -> None:
    """Resume a crashed or interrupted run.

    Skips completed work and retries in-flight tasks with
    new attempt IDs to ensure idempotency.
    """
    import asyncio

    async def _resume() -> None:
        # Initialize stores
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        checkpoint_dir = config.storage_dir.parent / "checkpoints"
        resume_mgr = ResumeManager(store, checkpoint_dir)

        try:
            # Check if resumable
            can_resume, reason = await resume_mgr.can_resume(run_id)

            if not can_resume:
                console.print(f"[red]Cannot resume: {reason}[/red]")
                raise typer.Exit(1)

            # Get resume state
            resume_state = await resume_mgr.get_resume_state(run_id)

            if dry_run:
                console.print("[yellow]DRY RUN - No changes will be made[/yellow]\n")

            # Show what will happen
            console.print(f"[bold]Resuming run {run_id}[/bold]\n")

            skip_count = len(resume_state.completed_tasks)
            retry_count = len(resume_state.in_flight_tasks)

            console.print(f"  [green]✓[/green] Will skip {skip_count} completed tasks")
            console.print(f"  [yellow]→[/yellow] Will retry {retry_count} in-flight tasks")

            if create_checkpoint and not dry_run:
                console.print(f"  [blue]📌[/blue] Creating resume checkpoint")
                await resume_mgr.create_resume_checkpoint(run_id)

            if dry_run:
                console.print("\n[dim]No actions taken (dry run)[/dim]")
                return

            # Here we would integrate with the flywheel to actually resume
            # For now, just show what would happen
            console.print(
                "\n[yellow]Note: Full resume execution requires flywheel integration.[/yellow]"
            )
            console.print(
                "[dim]Use 'blackice build --resume {run_id}' for full resume.[/dim]"
            )

        finally:
            await store.close()

    asyncio.run(_resume())


@app.command("list")
def list_resumable(
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum runs to show"),
    ] = 20,
) -> None:
    """List runs that can be resumed."""
    import asyncio

    async def _list() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        checkpoint_dir = config.storage_dir.parent / "checkpoints"
        resume_mgr = ResumeManager(store, checkpoint_dir)

        try:
            runs = await store.list_runs()

            resumable: list[tuple[str, str, int, int]] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Checking runs...", total=len(runs))

                for run_id in runs:
                    progress.update(task, advance=1)

                    can_resume, reason = await resume_mgr.can_resume(run_id)
                    if can_resume:
                        resume_state = await resume_mgr.get_resume_state(run_id)
                        resumable.append((
                            run_id,
                            reason,
                            len(resume_state.completed_tasks),
                            len(resume_state.in_flight_tasks),
                        ))

            if not resumable:
                console.print("[dim]No resumable runs found[/dim]")
                return

            table = Table(title=f"Resumable Runs ({len(resumable)} found)")
            table.add_column("Run ID", style="cyan")
            table.add_column("Completed", justify="right")
            table.add_column("In-Flight", justify="right")

            for run_id, _, completed, in_flight in resumable[:limit]:
                table.add_row(
                    run_id[:36],
                    str(completed),
                    f"[yellow]{in_flight}[/yellow]" if in_flight else str(in_flight),
                )

            console.print(table)

            if len(resumable) > limit:
                console.print(f"[dim]... and {len(resumable) - limit} more[/dim]")

        finally:
            await store.close()

    asyncio.run(_list())


@app.command("checkpoint")
def create_checkpoint(
    run_id: Annotated[str, typer.Argument(help="Run ID to checkpoint")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
) -> None:
    """Create a checkpoint for a run.

    Checkpoints capture the current state for faster resume.
    """
    import asyncio

    async def _checkpoint() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        checkpoint_dir = config.storage_dir.parent / "checkpoints"
        checkpoint_mgr = CheckpointManager(store, checkpoint_dir)

        try:
            events = await store.get_events(run_id)

            if not events:
                console.print(f"[red]No events found for run {run_id}[/red]")
                raise typer.Exit(1)

            checkpoint = await checkpoint_mgr.create_checkpoint(run_id)

            console.print(
                Panel(
                    f"[bold]Checkpoint ID:[/bold] {checkpoint.id}\n"
                    f"[bold]Run ID:[/bold] {run_id}\n"
                    f"[bold]Event Sequence:[/bold] {checkpoint.event_sequence}\n"
                    f"[bold]Completed Tasks:[/bold] {len(checkpoint.completed_tasks)}\n"
                    f"[bold]In-Flight Tasks:[/bold] {len(checkpoint.in_flight_tasks)}\n"
                    f"[bold]Created At:[/bold] {checkpoint.created_at.isoformat()}",
                    title="[bold green]Checkpoint Created[/bold green]",
                    border_style="green",
                )
            )

        finally:
            await store.close()

    asyncio.run(_checkpoint())


if __name__ == "__main__":
    app()
