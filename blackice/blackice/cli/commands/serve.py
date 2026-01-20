"""BLACKICE API server command.

Usage:
    blackice serve [--host=<host>] [--port=<port>] [--reload]
"""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()


def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host", "-h",
        help="Host to bind the server to",
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
    - GET /api/v1/health - Health check
    - GET /api/v1/providers - List LLM providers

    Set BLACKICE_API_KEY to enable authentication.
    """
    import os
    import uvicorn

    from blackice.api.auth import is_auth_configured

    auth_enabled = is_auth_configured()

    console.print(f"[bold blue]BLACKICE[/bold blue] API Server")
    console.print(f"Starting on [cyan]http://{host}:{port}[/cyan]")
    console.print(f"API docs: [cyan]http://{host}:{port}/docs[/cyan]")
    if auth_enabled:
        console.print("[green]Authentication: ENABLED[/green] (X-API-Key header required)")
    else:
        console.print("[yellow]Authentication: DISABLED[/yellow] (set BLACKICE_API_KEY to enable)")
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
