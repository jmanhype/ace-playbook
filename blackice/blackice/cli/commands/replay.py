"""Replay command for BLACKICE 3.0.

Provides deterministic state reconstruction from events:
- Replay run events to reconstruct state
- Verify state matches expected outcomes
- Debug and audit run history
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from blackice.persistence.event_store import EventStore, EventStoreConfig
from blackice.persistence.projections import (
    RunProjection,
    RunStatus,
    TaskProjection,
    TaskStatus,
    reconstruct_run_state,
    reconstruct_task_states,
)
from blackice.primitives.types import RunId

console = Console()
app = typer.Typer(help="Replay and reconstruct run state from events")


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


@app.command("state")
def replay_state(
    run_id: Annotated[str, typer.Argument(help="Run ID to replay")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
    show_tasks: Annotated[
        bool,
        typer.Option("--tasks", "-t", help="Show task details"),
    ] = True,
    show_events: Annotated[
        bool,
        typer.Option("--events", "-e", help="Show event list"),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", "-v", help="Verify hash chain integrity"),
    ] = False,
) -> None:
    """Replay events to reconstruct run state.

    Deterministically reconstructs the run state from the event log,
    proving that state can be recovered from events alone.
    """
    import asyncio

    async def _replay() -> None:
        # Initialize event store
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            # Load events
            events = await store.get_events(run_id)

            if not events:
                console.print(f"[red]No events found for run {run_id}[/red]")
                raise typer.Exit(1)

            # Verify integrity if requested
            if verify:
                is_valid, error = await store.verify_integrity(run_id)
                if not is_valid:
                    console.print(f"[red]Hash chain verification failed: {error}[/red]")
                    raise typer.Exit(1)
                console.print("[green]✓[/green] Hash chain integrity verified")

            # Reconstruct state
            run_state = reconstruct_run_state(events, run_id)

            # Display run state
            status_style = get_status_style(run_state.status.value)
            console.print(
                Panel(
                    f"[bold]Run ID:[/bold] {run_id}\n"
                    f"[bold]Status:[/bold] [{status_style}]{run_state.status.value}[/{status_style}]\n"
                    f"[bold]Edition:[/bold] {run_state.edition}\n"
                    f"[bold]Vision:[/bold] {run_state.vision[:80]}{'...' if len(run_state.vision) > 80 else ''}\n"
                    f"[bold]Events:[/bold] {run_state.event_count}\n"
                    f"[bold]Last Sequence:[/bold] {run_state.last_sequence}\n"
                    f"[bold]Completed Tasks:[/bold] {len(run_state.completed_tasks)}\n"
                    f"[bold]Failed Tasks:[/bold] {len(run_state.failed_tasks)}\n"
                    f"[bold]In-Flight Tasks:[/bold] {len(run_state.in_flight_tasks)}\n"
                    f"[bold]Checkpoints:[/bold] {len(run_state.checkpoints)}",
                    title="[bold blue]Reconstructed Run State[/bold blue]",
                    border_style="blue",
                )
            )

            # Show task details if requested
            if show_tasks and (
                run_state.completed_tasks
                or run_state.failed_tasks
                or run_state.in_flight_tasks
            ):
                task_states = reconstruct_task_states(events, run_id)

                table = Table(title="Task States")
                table.add_column("Task", style="cyan")
                table.add_column("Status")
                table.add_column("Attempt", justify="right")
                table.add_column("Duration", justify="right")

                for task_name, task in sorted(task_states.items()):
                    status_style = get_status_style(task.status.value)
                    duration = (
                        f"{task.duration_seconds:.2f}s"
                        if task.duration_seconds
                        else "-"
                    )
                    table.add_row(
                        task_name,
                        f"[{status_style}]{task.status.value}[/{status_style}]",
                        str(task.attempt),
                        duration,
                    )

                console.print(table)

            # Show events if requested
            if show_events:
                console.print("\n[bold]Event History:[/bold]")
                tree = Tree(f"[bold]Run {run_id}[/bold]")

                for event in events:
                    # Format timestamp
                    if hasattr(event.timestamp, "value"):
                        ts = event.timestamp.value.strftime("%H:%M:%S")
                    else:
                        ts = str(event.timestamp)[:8]

                    event_label = f"[dim]{ts}[/dim] [{event.sequence}] {event.type.value}"

                    # Add payload details for key events
                    if event.type.value in ("task_started", "task_completed", "task_failed"):
                        task_name = event.payload.get("task_name", "unknown")
                        event_label += f" - [cyan]{task_name}[/cyan]"

                    tree.add(event_label)

                console.print(tree)

        finally:
            await store.close()

    asyncio.run(_replay())


@app.command("verify")
def verify_integrity(
    run_id: Annotated[str, typer.Argument(help="Run ID to verify")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
) -> None:
    """Verify hash chain integrity for a run.

    Checks that the event log has not been tampered with by
    verifying the hash chain links between events.
    """
    import asyncio

    async def _verify() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            events = await store.get_events(run_id)

            if not events:
                console.print(f"[red]No events found for run {run_id}[/red]")
                raise typer.Exit(1)

            is_valid, error = await store.verify_integrity(run_id)

            if is_valid:
                console.print(
                    Panel(
                        f"[green]✓ Hash chain verified[/green]\n\n"
                        f"Events checked: {len(events)}\n"
                        f"First event: seq {events[0].sequence}\n"
                        f"Last event: seq {events[-1].sequence}",
                        title="[bold green]Integrity Verified[/bold green]",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[red]✗ Hash chain verification failed[/red]\n\n"
                        f"Error: {error}",
                        title="[bold red]Integrity Failed[/bold red]",
                        border_style="red",
                    )
                )
                raise typer.Exit(1)

        finally:
            await store.close()

    asyncio.run(_verify())


@app.command("list")
def list_runs(
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum runs to show"),
    ] = 20,
) -> None:
    """List all runs with stored events."""
    import asyncio

    async def _list() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            runs = await store.list_runs()

            if not runs:
                console.print("[dim]No runs found[/dim]")
                return

            table = Table(title=f"Stored Runs ({len(runs)} total)")
            table.add_column("Run ID", style="cyan")
            table.add_column("Events", justify="right")
            table.add_column("Status")

            for run_id in runs[:limit]:
                events = await store.get_events(run_id)
                run_state = reconstruct_run_state(events, run_id)

                status_style = get_status_style(run_state.status.value)
                table.add_row(
                    run_id[:36],
                    str(run_state.event_count),
                    f"[{status_style}]{run_state.status.value}[/{status_style}]",
                )

            console.print(table)

            if len(runs) > limit:
                console.print(f"[dim]... and {len(runs) - limit} more[/dim]")

        finally:
            await store.close()

    asyncio.run(_list())


@app.command("diff")
def diff_states(
    run_id: Annotated[str, typer.Argument(help="Run ID to compare")],
    from_seq: Annotated[
        int,
        typer.Option("--from", "-f", help="Starting sequence number"),
    ] = 0,
    to_seq: Annotated[
        int | None,
        typer.Option("--to", "-t", help="Ending sequence number"),
    ] = None,
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
) -> None:
    """Show state changes between two event sequences.

    Useful for debugging and understanding how state evolved.
    """
    import asyncio

    async def _diff() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            all_events = await store.get_events(run_id)

            if not all_events:
                console.print(f"[red]No events found for run {run_id}[/red]")
                raise typer.Exit(1)

            # Get events up to from_seq
            from_events = [e for e in all_events if e.sequence <= from_seq]

            # Get events up to to_seq
            end_seq = to_seq if to_seq is not None else all_events[-1].sequence
            to_events = [e for e in all_events if e.sequence <= end_seq]

            # Reconstruct states
            from_state = reconstruct_run_state(from_events, run_id) if from_events else None
            to_state = reconstruct_run_state(to_events, run_id)

            # Show diff
            console.print(
                f"\n[bold]State Diff: seq {from_seq} → seq {end_seq}[/bold]\n"
            )

            if from_state:
                if from_state.status != to_state.status:
                    console.print(
                        f"Status: [{get_status_style(from_state.status.value)}]{from_state.status.value}[/] → "
                        f"[{get_status_style(to_state.status.value)}]{to_state.status.value}[/]"
                    )

                new_completed = set(to_state.completed_tasks) - set(from_state.completed_tasks)
                if new_completed:
                    console.print(f"[green]+ Completed:[/green] {', '.join(new_completed)}")

                new_failed = set(to_state.failed_tasks) - set(from_state.failed_tasks)
                if new_failed:
                    console.print(f"[red]+ Failed:[/red] {', '.join(new_failed)}")
            else:
                console.print(f"Status: [dim]none[/dim] → [{get_status_style(to_state.status.value)}]{to_state.status.value}[/]")
                if to_state.completed_tasks:
                    console.print(f"[green]+ Completed:[/green] {', '.join(to_state.completed_tasks)}")

            console.print(f"\nEvents in range: {len(to_events) - len(from_events)}")

        finally:
            await store.close()

    asyncio.run(_diff())


if __name__ == "__main__":
    app()
