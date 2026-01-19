"""Watch command for BLACKICE 3.0.

Provides real-time run progress monitoring:
- Stream run events as they occur
- Display task progress and status
- Show live metrics and timing
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from blackice.persistence.event_store import EventStore, EventStoreConfig
from blackice.persistence.projections import (
    RunProjection,
    RunStatus,
    TaskProjection,
    reconstruct_run_state,
)
from blackice.primitives.types import EventType

console = Console()
app = typer.Typer(help="Watch run progress in real-time")


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
    }
    return styles.get(status.lower(), "white")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


@app.command("events")
def watch_events(
    run_id: Annotated[str, typer.Argument(help="Run ID to watch")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--interval", "-i", help="Poll interval in seconds"),
    ] = 1.0,
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Continue watching for new events"),
    ] = True,
    show_payload: Annotated[
        bool,
        typer.Option("--payload", "-p", help="Show event payloads"),
    ] = False,
) -> None:
    """Watch events for a run in real-time.

    Streams events as they are written, similar to 'tail -f'.
    """

    async def _watch() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            last_sequence = -1
            run_completed = False

            console.print(f"[bold]Watching events for run {run_id}[/bold]\n")

            while not run_completed:
                events = await store.get_events(run_id, since_sequence=last_sequence + 1)

                for event in events:
                    last_sequence = event.sequence

                    # Format event
                    if hasattr(event.timestamp, "value"):
                        ts = event.timestamp.value.strftime("%H:%M:%S.%f")[:12]
                    else:
                        ts = str(event.timestamp)[:12]

                    event_type = event.type.value
                    style = "white"

                    if event.type in (EventType.RUN_COMPLETED, EventType.TASK_COMPLETED):
                        style = "green"
                    elif event.type in (EventType.RUN_FAILED, EventType.TASK_FAILED):
                        style = "red"
                    elif event.type in (EventType.TASK_STARTED, EventType.PHASE_STARTED):
                        style = "yellow"

                    # Build event line
                    line = Text()
                    line.append(f"[{ts}] ", style="dim")
                    line.append(f"#{event.sequence:04d} ", style="cyan")
                    line.append(event_type, style=style)

                    # Add context for task events
                    if "task_name" in event.payload:
                        line.append(f" - {event.payload['task_name']}", style="cyan")
                    if "phase" in event.payload:
                        line.append(f" - {event.payload['phase']}", style="blue")
                    if "error" in event.payload:
                        line.append(f": {event.payload['error'][:50]}", style="red")

                    console.print(line)

                    # Show payload if requested
                    if show_payload and event.payload:
                        import json
                        payload_str = json.dumps(event.payload, indent=2)
                        console.print(f"  [dim]{payload_str}[/dim]")

                    # Check for completion
                    if event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
                        run_completed = True

                if not follow:
                    break

                if not run_completed:
                    await asyncio.sleep(poll_interval)

            if run_completed:
                console.print("\n[bold]Run finished[/bold]")

        finally:
            await store.close()

    asyncio.run(_watch())


@app.command("progress")
def watch_progress(
    run_id: Annotated[str, typer.Argument(help="Run ID to watch")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--interval", "-i", help="Poll interval in seconds"),
    ] = 1.0,
) -> None:
    """Watch run progress with live updates.

    Shows a progress display that updates as tasks complete.
    """

    async def _watch_progress() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            start_time = datetime.utcnow()

            def generate_table() -> Table:
                """Generate progress table."""
                table = Table(title=f"Run {run_id[:16]}...", show_header=False)
                table.add_column("Metric", style="bold")
                table.add_column("Value")
                return table

            with Live(generate_table(), console=console, refresh_per_second=4) as live:
                run_completed = False

                while not run_completed:
                    events = await store.get_events(run_id)

                    if not events:
                        live.update(Panel("[yellow]Waiting for events...[/yellow]"))
                        await asyncio.sleep(poll_interval)
                        continue

                    # Reconstruct state
                    run_state = reconstruct_run_state(events, run_id)

                    # Calculate metrics
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    total_tasks = len(run_state.completed_tasks) + len(run_state.failed_tasks) + len(run_state.in_flight_tasks)

                    # Build table
                    table = Table(show_header=False, box=None)
                    table.add_column("Metric", style="bold", width=20)
                    table.add_column("Value", width=40)

                    status_style = get_status_style(run_state.status.value)
                    table.add_row("Status", f"[{status_style}]{run_state.status.value}[/]")
                    table.add_row("Elapsed", format_duration(elapsed))
                    table.add_row("Events", str(run_state.event_count))
                    table.add_row("", "")
                    table.add_row("[green]Completed[/]", f"[green]{len(run_state.completed_tasks)}[/]")
                    table.add_row("[yellow]In-Flight[/]", f"[yellow]{len(run_state.in_flight_tasks)}[/]")
                    table.add_row("[red]Failed[/]", f"[red]{len(run_state.failed_tasks)}[/]")

                    # Show current task
                    if run_state.in_flight_tasks:
                        current = list(run_state.in_flight_tasks)[0]
                        table.add_row("", "")
                        table.add_row("Current", f"[cyan]{current}[/]")

                    # Build panel
                    panel = Panel(
                        table,
                        title=f"[bold]Run Progress[/bold]",
                        border_style="blue",
                    )
                    live.update(panel)

                    # Check for completion
                    if run_state.status in (RunStatus.COMPLETED, RunStatus.FAILED):
                        run_completed = True
                    else:
                        await asyncio.sleep(poll_interval)

                # Show final status
                final_style = "green" if run_state.status == RunStatus.COMPLETED else "red"
                console.print(
                    f"\n[{final_style}]Run {run_state.status.value}![/{final_style}]"
                )

        finally:
            await store.close()

    asyncio.run(_watch_progress())


@app.command("tail")
def tail_events(
    run_id: Annotated[str, typer.Argument(help="Run ID to tail")],
    lines: Annotated[
        int,
        typer.Option("--lines", "-n", help="Number of lines to show"),
    ] = 20,
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
) -> None:
    """Show the last N events for a run.

    Similar to 'tail -n' for event logs.
    """

    async def _tail() -> None:
        config = EventStoreConfig()
        if storage_dir:
            config.storage_dir = storage_dir

        store = EventStore(config)
        await store.initialize()

        try:
            events = await store.get_events(run_id)

            if not events:
                console.print(f"[red]No events found for run {run_id}[/red]")
                return

            # Get last N events
            tail_events = events[-lines:]

            console.print(f"[bold]Last {len(tail_events)} events for run {run_id}[/bold]\n")

            for event in tail_events:
                if hasattr(event.timestamp, "value"):
                    ts = event.timestamp.value.strftime("%H:%M:%S")
                else:
                    ts = str(event.timestamp)[:8]

                style = "white"
                if event.type.value.endswith("_completed"):
                    style = "green"
                elif event.type.value.endswith("_failed"):
                    style = "red"
                elif event.type.value.endswith("_started"):
                    style = "yellow"

                line = f"[dim]{ts}[/] [cyan]#{event.sequence:04d}[/] [{style}]{event.type.value}[/]"

                if "task_name" in event.payload:
                    line += f" - {event.payload['task_name']}"

                console.print(line)

        finally:
            await store.close()

    asyncio.run(_tail())


@app.command("status")
def quick_status(
    run_id: Annotated[str, typer.Argument(help="Run ID to check")],
    storage_dir: Annotated[
        Path | None,
        typer.Option("--storage", "-s", help="Event storage directory"),
    ] = None,
) -> None:
    """Quick status check for a run."""

    async def _status() -> None:
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

            run_state = reconstruct_run_state(events, run_id)

            status_style = get_status_style(run_state.status.value)
            status_icon = {
                RunStatus.COMPLETED: "✓",
                RunStatus.FAILED: "✗",
                RunStatus.EXECUTING: "→",
                RunStatus.PLANNING: "○",
                RunStatus.PENDING: "○",
            }.get(run_state.status, "?")

            console.print(
                f"[{status_style}]{status_icon}[/] [{status_style}]{run_state.status.value}[/] - "
                f"{len(run_state.completed_tasks)} done, "
                f"{len(run_state.in_flight_tasks)} running, "
                f"{len(run_state.failed_tasks)} failed"
            )

        finally:
            await store.close()

    asyncio.run(_status())


if __name__ == "__main__":
    app()
