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
    - WebSocket /api/v1/runs/{id}/stream - Stream run events
    - GET /api/v1/health - Health check
    - GET /api/v1/providers - List LLM providers
    """
    import uvicorn

    console.print(f"[bold blue]BLACKICE[/bold blue] API Server")
    console.print(f"Starting on [cyan]http://{host}:{port}[/cyan]")
    console.print(f"API docs: [cyan]http://{host}:{port}/docs[/cyan]")
    console.print()

    uvicorn.run(
        "blackice.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )
