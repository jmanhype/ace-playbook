"""Doctor command - Diagnose system health.

Checks providers, dependencies, and configuration.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from blackice import __version__
from blackice.core.config import BlackiceConfig
from blackice.instrumentation import get_logger

logger = get_logger(__name__)
console = Console()


def doctor(
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Attempt to fix issues")
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output")
    ] = False,
) -> None:
    """Diagnose BLACKICE installation and configuration.

    Checks:
    - Python version and dependencies
    - Configuration validity
    - Provider availability
    - Required tools (git, docker, etc.)
    - Workspace permissions

    Example:
        blackice doctor          # Run diagnostics
        blackice doctor --fix    # Attempt to fix issues
    """
    console.print(Panel(
        f"[bold]BLACKICE Doctor[/bold]\n"
        f"Version: {__version__}",
        border_style="blue",
    ))

    results = asyncio.run(_run_checks(verbose))
    _show_results(results, fix)


async def _run_checks(verbose: bool) -> dict:
    """Run all diagnostic checks."""
    results = {
        "checks": [],
        "passed": 0,
        "failed": 0,
        "warnings": 0,
    }

    # Check Python version
    results["checks"].append(_check_python())

    # Check dependencies
    results["checks"].append(_check_dependencies())

    # Check configuration
    results["checks"].append(_check_config())

    # Check tools
    results["checks"].append(_check_tools())

    # Check providers
    results["checks"].extend(await _check_providers(verbose))

    # Check workspace
    results["checks"].append(_check_workspace())

    # Tally results
    for check in results["checks"]:
        if check["status"] == "pass":
            results["passed"] += 1
        elif check["status"] == "fail":
            results["failed"] += 1
        else:
            results["warnings"] += 1

    return results


def _check_python() -> dict:
    """Check Python version."""
    version = sys.version_info
    required = (3, 11)

    if version >= required:
        return {
            "name": "Python Version",
            "status": "pass",
            "message": f"{version.major}.{version.minor}.{version.micro}",
        }
    else:
        return {
            "name": "Python Version",
            "status": "fail",
            "message": f"{version.major}.{version.minor} (requires 3.11+)",
            "fix": "Install Python 3.11 or higher",
        }


def _check_dependencies() -> dict:
    """Check required Python packages."""
    required = [
        "typer",
        "rich",
        "pydantic",
        "httpx",
        "structlog",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return {
            "name": "Python Dependencies",
            "status": "pass",
            "message": f"All {len(required)} required packages installed",
        }
    else:
        return {
            "name": "Python Dependencies",
            "status": "fail",
            "message": f"Missing: {', '.join(missing)}",
            "fix": f"pip install {' '.join(missing)}",
        }


def _check_config() -> dict:
    """Check configuration validity."""
    try:
        config = BlackiceConfig()
        return {
            "name": "Configuration",
            "status": "pass",
            "message": f"Edition: {config.edition.value}",
        }
    except Exception as e:
        return {
            "name": "Configuration",
            "status": "fail",
            "message": str(e),
            "fix": "Check environment variables and config files",
        }


def _check_tools() -> dict:
    """Check required external tools."""
    tools = {
        "git": "Version control",
        "docker": "Container execution (optional)",
    }

    available = []
    missing = []

    for tool, description in tools.items():
        if shutil.which(tool):
            available.append(tool)
        else:
            missing.append(tool)

    if not missing:
        return {
            "name": "External Tools",
            "status": "pass",
            "message": f"All tools available: {', '.join(available)}",
        }
    elif "git" in missing:
        return {
            "name": "External Tools",
            "status": "fail",
            "message": f"Missing: {', '.join(missing)}",
            "fix": "Install git (required)",
        }
    else:
        return {
            "name": "External Tools",
            "status": "warn",
            "message": f"Optional missing: {', '.join(missing)}",
        }


async def _check_providers(verbose: bool) -> list[dict]:
    """Check provider availability."""
    results = []

    # Check model providers
    model_checks = await _check_model_providers(verbose)
    results.extend(model_checks)

    # Check secrets provider
    results.append(_check_secrets_provider())

    return results


async def _check_model_providers(verbose: bool) -> list[dict]:
    """Check model provider connectivity."""
    results = []

    # Check for API keys
    providers = {
        "ANTHROPIC_API_KEY": "Claude",
        "OPENAI_API_KEY": "OpenAI",
    }

    found_providers = []
    for env_var, name in providers.items():
        if os.environ.get(env_var):
            found_providers.append(name)

    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                found_providers.append("Ollama")
    except Exception:
        pass

    if found_providers:
        results.append({
            "name": "Model Providers",
            "status": "pass",
            "message": f"Available: {', '.join(found_providers)}",
        })
    else:
        results.append({
            "name": "Model Providers",
            "status": "fail",
            "message": "No model providers configured",
            "fix": "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or start Ollama",
        })

    return results


def _check_secrets_provider() -> dict:
    """Check secrets provider configuration."""
    # Check for common secret patterns
    secret_vars = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "BLACKICE_",
    ]

    found = 0
    for pattern in secret_vars:
        for key in os.environ:
            if key.startswith(pattern) or key == pattern:
                found += 1
                break

    if found > 0:
        return {
            "name": "Secrets Provider",
            "status": "pass",
            "message": f"Environment secrets configured",
        }
    else:
        return {
            "name": "Secrets Provider",
            "status": "warn",
            "message": "No secrets found in environment",
        }


def _check_workspace() -> dict:
    """Check workspace directory permissions."""
    import tempfile
    from pathlib import Path

    # Try to create a temp file in the current directory
    try:
        test_dir = Path.cwd() / ".blackice"
        test_dir.mkdir(exist_ok=True)

        test_file = test_dir / ".doctor_test"
        test_file.write_text("test")
        test_file.unlink()

        return {
            "name": "Workspace",
            "status": "pass",
            "message": f"Writable: {Path.cwd()}",
        }
    except PermissionError:
        return {
            "name": "Workspace",
            "status": "fail",
            "message": "Cannot write to current directory",
            "fix": "Check directory permissions or run from a different location",
        }
    except Exception as e:
        return {
            "name": "Workspace",
            "status": "warn",
            "message": str(e),
        }


def _show_results(results: dict, fix: bool) -> None:
    """Display diagnostic results."""
    # Create results table
    table = Table(title="Diagnostic Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    for check in results["checks"]:
        status = check["status"]
        if status == "pass":
            status_text = "[green]✓ Pass[/green]"
        elif status == "fail":
            status_text = "[red]✗ Fail[/red]"
        else:
            status_text = "[yellow]! Warn[/yellow]"

        table.add_row(
            check["name"],
            status_text,
            check["message"],
        )

    console.print(table)

    # Summary
    total = len(results["checks"])
    console.print(f"\n[bold]Summary:[/bold] {results['passed']}/{total} passed", end="")

    if results["warnings"] > 0:
        console.print(f", [yellow]{results['warnings']} warnings[/yellow]", end="")

    if results["failed"] > 0:
        console.print(f", [red]{results['failed']} failed[/red]")
    else:
        console.print()

    # Show fixes if requested and there are failures
    if fix and results["failed"] > 0:
        console.print("\n[bold]Suggested Fixes:[/bold]")
        for check in results["checks"]:
            if check["status"] == "fail" and "fix" in check:
                console.print(f"  • {check['name']}: {check['fix']}")

    # Exit with error if any checks failed
    if results["failed"] > 0:
        raise typer.Exit(1)
