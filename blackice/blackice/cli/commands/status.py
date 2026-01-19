"""Status command - View run status and history.

Shows current and historical run information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from blackice.instrumentation import get_logger

logger = get_logger(__name__)
console = Console()


def status(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Specific run ID to check")
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Workspace directory")
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON")
    ] = False,
    all_runs: Annotated[
        bool,
        typer.Option("--all", "-a", help="Show all runs")
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum runs to show")
    ] = 10,
) -> None:
    """Show status of BLACKICE runs.

    Without arguments, shows recent runs.
    With a run ID, shows detailed status for that specific run.

    Examples:
        blackice status                    # Show recent runs
        blackice status run-abc123         # Show specific run
        blackice status --all              # Show all runs
    """
    workspace_root = workspace or Path.cwd() / ".blackice"

    if run_id:
        _show_run_details(run_id, workspace_root, json_output)
    else:
        _show_run_list(workspace_root, json_output, all_runs, limit)


def _show_run_details(run_id: str, workspace_root: Path, json_output: bool) -> None:
    """Show detailed information about a specific run."""
    run_dir = workspace_root / run_id

    if not run_dir.exists():
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(1)

    # Load run data
    summary_file = run_dir / "summary.json"
    vision_file = run_dir / "vision.md"
    plan_file = run_dir / "plan.json"

    run_data = {
        "run_id": run_id,
        "workspace": str(run_dir),
        "exists": True,
    }

    if summary_file.exists():
        try:
            run_data["summary"] = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            run_data["summary"] = None

    if vision_file.exists():
        vision_content = vision_file.read_text()
        # Extract vision from markdown
        lines = vision_content.strip().split("\n")
        vision = next((line for line in lines if line and not line.startswith("#")), "")
        run_data["vision"] = vision[:200]

    if plan_file.exists():
        try:
            plan = json.loads(plan_file.read_text())
            run_data["tasks"] = len(plan.get("tasks", []))
        except json.JSONDecodeError:
            pass

    # Count artifacts
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        run_data["artifact_count"] = len(list(artifacts_dir.iterdir()))

    if json_output:
        console.print(json.dumps(run_data, indent=2))
        return

    # Rich output
    summary = run_data.get("summary", {})
    success = summary.get("success", False) if summary else None

    status_color = "green" if success else "red" if success is False else "yellow"
    status_text = "Completed" if success else "Failed" if success is False else "Unknown"

    console.print(Panel(
        f"[bold]Run ID:[/bold] {run_id}\n"
        f"[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]\n"
        f"[bold]Vision:[/bold] {run_data.get('vision', 'N/A')}\n"
        f"[bold]Tasks:[/bold] {run_data.get('tasks', 'N/A')}\n"
        f"[bold]Artifacts:[/bold] {run_data.get('artifact_count', 0)}\n"
        f"[bold]Workspace:[/bold] {run_dir}",
        title=f"[blue]Run: {run_id}[/blue]",
        border_style="blue",
    ))

    # Show phases if available
    if summary and "phases" in summary:
        console.print("\n[bold]Phases:[/bold]")
        for phase in summary.get("phases", []):
            console.print(f"  • {phase}")


def _show_run_list(
    workspace_root: Path,
    json_output: bool,
    all_runs: bool,
    limit: int,
) -> None:
    """Show list of runs."""
    if not workspace_root.exists():
        if json_output:
            console.print(json.dumps({"runs": []}))
        else:
            console.print("[yellow]No runs found[/yellow]")
            console.print(f"Workspace: {workspace_root}")
        return

    # Find run directories
    runs = []
    for item in workspace_root.iterdir():
        if item.is_dir() and item.name.startswith("run-"):
            run_info = {"run_id": item.name, "path": str(item)}

            # Get summary if available
            summary_file = item / "summary.json"
            if summary_file.exists():
                try:
                    summary = json.loads(summary_file.read_text())
                    run_info["success"] = summary.get("success")
                    run_info["artifact_count"] = len(summary.get("artifacts", []))
                except json.JSONDecodeError:
                    pass

            # Get modification time
            run_info["modified"] = item.stat().st_mtime

            runs.append(run_info)

    # Sort by modification time (newest first)
    runs.sort(key=lambda x: x.get("modified", 0), reverse=True)

    # Apply limit
    if not all_runs:
        runs = runs[:limit]

    if json_output:
        console.print(json.dumps({"runs": runs}, indent=2))
        return

    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return

    # Create table
    table = Table(title="BLACKICE Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Artifacts", justify="right")

    for run in runs:
        success = run.get("success")
        if success is True:
            status = "[green]✓ Complete[/green]"
        elif success is False:
            status = "[red]✗ Failed[/red]"
        else:
            status = "[yellow]? Unknown[/yellow]"

        table.add_row(
            run["run_id"],
            status,
            str(run.get("artifact_count", "-")),
        )

    console.print(table)

    if not all_runs and len(runs) == limit:
        console.print(f"\n[dim]Showing {limit} most recent runs. Use --all for all runs.[/dim]")
