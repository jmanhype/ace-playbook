"""BLACKICE API server command.

Usage:
    blackice serve [--host=<host>] [--port=<port>] [--reload]

Phase 9 Fixes:
- Changed -h to -H (avoid conflict with --help convention)
- Default to 127.0.0.1 (secure default - localhost only)
- Refuse external bind without auth unless --insecure is passed
"""

from __future__ import annotations

import ipaddress

import typer
from rich.console import Console

console = Console()


def _is_loopback(host: str) -> bool:
    """Check if host is a loopback address."""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in {"localhost"}


def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host", "-H",
        help="Host to bind the server to (default: localhost only)",
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port to bind the server to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload", "-r",
        help="Enable auto-reload for development",
    ),
    workers: int = typer.Option(
        1,
        "--workers", "-w",
        help="Number of worker processes",
    ),
    insecure: bool = typer.Option(
        False,
        "--insecure",
        help="Allow external binding without auth (DANGEROUS)",
    ),
) -> None:
    """Start the BLACKICE API server.

    The API provides HTTP endpoints for programmatic access to BLACKICE:
    - POST /api/v1/runs - Create and start a new run
    - GET /api/v1/runs - List all runs
    - GET /api/v1/runs/{id} - Get run details
    - GET /api/v1/runs/{id}/result - Get run result
    - POST /api/v1/runs/{id}/cancel - Cancel a run
    - DELETE /api/v1/runs/{id} - Delete a run
    - WebSocket /api/v1/runs/{id}/stream - Stream run events
    - GET /api/v1/health/live - Liveness probe
    - GET /api/v1/health/ready - Readiness probe
    - GET /api/v1/providers - List LLM providers

    Set BLACKICE_API_KEY to enable authentication.
    """
    import uvicorn

    from blackice.api.auth import is_auth_configured

    auth_enabled = is_auth_configured()

    # Phase 9 Security: Refuse external bind without auth
    if not _is_loopback(host) and not auth_enabled and not insecure:
        console.print("[bold red]ERROR:[/bold red] Cannot bind to external address without authentication.")
        console.print()
        console.print("Options:")
        console.print("  1. Set BLACKICE_API_KEY environment variable to enable auth")
        console.print("  2. Use --host 127.0.0.1 to bind to localhost only (default)")
        console.print("  3. Use --insecure to bypass this check (DANGEROUS)")
        raise typer.Exit(code=2)

    console.print(f"[bold blue]BLACKICE[/bold blue] API Server")
    console.print(f"Starting on [cyan]http://{host}:{port}[/cyan]")
    console.print(f"API docs: [cyan]http://{host}:{port}/docs[/cyan]")
    if auth_enabled:
        console.print("[green]Authentication: ENABLED[/green] (X-API-Key header or ?api_key= query param)")
    else:
        console.print("[yellow]Authentication: DISABLED[/yellow] (set BLACKICE_API_KEY to enable)")
        if not _is_loopback(host):
            console.print("[bold red]WARNING:[/bold red] Running without auth on external address!")
    console.print()

    # P0 Fix: Use factory pattern to ensure auth is detected at runtime
    uvicorn.run(
        "blackice.api:create_app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
        factory=True,  # Use create_app() factory, not import-time app
    )
