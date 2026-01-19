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
from rich.table import Table

from blackice.core.config import BlackiceConfig, Edition
from blackice.core.providers import (
    ProviderType,
    create_model_provider,
    create_memory_provider,
    create_execution_provider,
    verify_ai_factory_connection,
)
from blackice.flywheel import FlywheelConfig, FlywheelPhase, UnifiedFlywheel
from blackice.infrastructure import get_ai_factory_config
from blackice.instrumentation import get_logger

logger = get_logger(__name__)
console = Console()


def build(
    vision: Annotated[str, typer.Argument(help="Vision description for what to build")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Model to use (default: from AI Factory config)")] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="LLM provider: ollama, claude, openai, z.ai (default: ollama)")
    ] = "ollama",
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="API key for cloud providers (or use env var)")
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="Custom base URL for provider")
    ] = None,
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
    use_local: Annotated[
        bool,
        typer.Option("--local", help="Use local AI Factory (Ollama + Letta)")
    ] = True,
    skip_memory: Annotated[
        bool,
        typer.Option("--skip-memory", help="Skip Letta memory provider (use in-memory)")
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Verify AI Factory connection before building")
    ] = True,
) -> None:
    """Build working software from a vision description.

    Takes a natural language description of what you want to build
    and produces working software with tests and documentation.

    Example:
        blackice build "Create a REST API for user management with JWT auth"
        blackice build "Create a CLI tool" --provider claude
        blackice build "Create a web scraper" --provider z.ai --api-key YOUR_KEY

    Uses the local AI Factory by default (Ollama at 192.168.1.143:11434).
    """
    # Validate provider
    valid_providers = {"ollama", "claude", "claude-max", "openai", "zhipu", "z.ai"}
    provider_lower = provider.lower()
    if provider_lower not in valid_providers:
        console.print(f"[red]Invalid provider:[/red] {provider}")
        console.print(f"Valid providers: {', '.join(sorted(valid_providers))}")
        raise typer.Exit(1)

    # Validate edition
    try:
        edition_enum = Edition(edition.lower())
    except ValueError:
        console.print(f"[red]Invalid edition:[/red] {edition}")
        console.print("Valid editions: lite, core, enterprise")
        raise typer.Exit(1)

    # Generate or use provided run ID
    actual_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"

    # Get AI Factory config for local providers (ollama, claude-max)
    ai_factory_config = get_ai_factory_config() if (use_local and provider_lower in ("ollama", "claude-max")) else None

    # Determine actual model name based on provider
    if model:
        actual_model = model
    elif provider_lower == "ollama" and ai_factory_config:
        actual_model = ai_factory_config.ollama.default_model
    elif provider_lower in ("claude", "claude-max"):
        actual_model = "claude-sonnet-4-20250514"
    elif provider_lower == "zhipu":
        actual_model = "codegeex-4"
    elif provider_lower in ("openai", "z.ai"):
        actual_model = "gpt-4o"
    else:
        actual_model = "qwen2.5-coder:32b"

    # Determine backend name for display
    if provider_lower == "ollama":
        backend_name = "Local AI Factory (Ollama)"
    elif provider_lower == "claude":
        backend_name = "Anthropic Claude API"
    elif provider_lower == "claude-max":
        backend_name = "Claude Max Router (FREE via AI Factory)"
    elif provider_lower == "zhipu":
        backend_name = "Zhipu/GLM BigModel (CodeGeeX)"
    elif provider_lower == "z.ai":
        backend_name = "z.ai (OpenAI-compatible)"
    else:
        backend_name = "OpenAI API"

    # Show plan
    console.print(Panel(
        f"[bold]Vision:[/bold] {vision[:200]}{'...' if len(vision) > 200 else ''}\n\n"
        f"[bold]Run ID:[/bold] {actual_run_id}\n"
        f"[bold]Provider:[/bold] {provider_lower}\n"
        f"[bold]Model:[/bold] {actual_model}\n"
        f"[bold]Edition:[/bold] {edition_enum.value}\n"
        f"[bold]Workspace:[/bold] {workspace or 'auto'}\n"
        f"[bold]Backend:[/bold] {backend_name}",
        title="[blue]BLACKICE Build[/blue]",
        border_style="blue",
    ))

    # Verify AI Factory connection if requested (only for Ollama)
    if verify and provider_lower == "ollama" and ai_factory_config:
        console.print("\n[dim]Verifying AI Factory connection...[/dim]")
        connection_status = asyncio.run(_verify_connection(ai_factory_config, skip_memory))

        if not connection_status["all_healthy"]:
            console.print("[red]AI Factory connection failed![/red]")
            _show_connection_status(connection_status)
            raise typer.Exit(1)

        _show_connection_status(connection_status)

    if dry_run:
        console.print("\n[yellow]Dry run - no changes will be made[/yellow]")
        return

    # Run the flywheel
    asyncio.run(_run_build(
        vision=vision,
        run_id=actual_run_id,
        model=actual_model,
        workspace=workspace,
        edition=edition_enum,
        timeout=timeout,
        provider_type=provider_lower,
        api_key=api_key,
        base_url=base_url,
        skip_memory=skip_memory,
        ai_factory_config=ai_factory_config,
    ))


async def _verify_connection(ai_factory_config, skip_memory: bool) -> dict:
    """Verify connection to AI Factory components."""
    return await verify_ai_factory_connection(
        config=ai_factory_config,
        check_memory=not skip_memory,
    )


def _show_connection_status(status: dict) -> None:
    """Display connection status in a table."""
    table = Table(title="AI Factory Status", show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    for component, info in status.items():
        if component == "all_healthy":
            continue

        if info.get("healthy"):
            status_icon = "[green]✓ Healthy[/green]"
        else:
            status_icon = f"[red]✗ Error: {info.get('error', 'Unknown')}[/red]"

        details = ""
        if component == "ollama" and info.get("healthy"):
            models = info.get("models", [])
            if models:
                details = f"Models: {', '.join(models[:3])}"
                if len(models) > 3:
                    details += f" (+{len(models) - 3} more)"
        elif component == "letta" and info.get("healthy"):
            agents = info.get("agent_count", 0)
            details = f"Agents: {agents}"

        table.add_row(component.title(), status_icon, details)

    console.print(table)


async def _run_build(
    vision: str,
    run_id: str,
    model: str,
    workspace: Path | None,
    edition: Edition,
    timeout: float,
    provider_type: str = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
    skip_memory: bool = False,
    ai_factory_config=None,
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

    # Create providers
    model_provider = None
    memory_provider = None
    execution_provider = None

    # Create model provider based on type
    model_provider = create_model_provider(
        config=ai_factory_config,
        model=model,
        provider_type=provider_type,
        api_key=api_key,
        base_url=base_url,
    )

    # Create execution provider
    execution_provider = create_execution_provider(
        working_dir=str(workspace) if workspace else None
    )

    # Create memory provider if not skipped and using Ollama
    if not skip_memory and provider_type == "ollama" and ai_factory_config:
        try:
            memory_provider = create_memory_provider(config=ai_factory_config)
        except Exception as e:
            logger.warning(f"Failed to create memory provider: {e}, continuing without memory")

    flywheel = UnifiedFlywheel(
        config=config,
        model_provider=model_provider,
        execution_provider=execution_provider,
        memory_provider=memory_provider,
    )

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
