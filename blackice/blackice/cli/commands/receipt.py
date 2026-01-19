"""Receipt command - Manage receipts and signatures (Enterprise).

Provides operations for:
- Generating signing key pairs
- Signing receipts
- Verifying receipt signatures
- Viewing receipt information
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from blackice.instrumentation import get_logger

logger = get_logger(__name__)
console = Console()

# Create receipt subcommand app
app = typer.Typer(
    name="receipt",
    help="Receipt management commands (Enterprise)",
    no_args_is_help=True,
)


@app.command(name="keygen")
def keygen(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for the key file"),
    ] = None,
    key_id: Annotated[
        str | None,
        typer.Option("--key-id", "-k", help="Custom key identifier"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing key file"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Generate a new Ed25519 signing key pair.

    Creates a cryptographic key pair for signing BLACKICE receipts.
    The private key should be kept secure; the public key can be shared
    for verification.

    Examples:
        blackice receipt keygen
        blackice receipt keygen --output keys/signing.key
        blackice receipt keygen --key-id "production-signer-2025"
    """
    from blackice.security import generate_key_pair

    # Determine output path
    if output is None:
        keys_dir = Path.cwd() / ".blackice" / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        output = keys_dir / "signing.key"

    # Check for existing file
    if output.exists() and not force:
        console.print(f"[red]Key file already exists:[/red] {output}")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Generate key pair
    key_pair = generate_key_pair(key_id=key_id)

    # Ensure parent directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    key_pair.to_file(output)

    # Restrict file permissions (Unix only)
    try:
        os.chmod(output, 0o600)
    except (OSError, AttributeError):
        pass  # Windows doesn't support this

    if json_output:
        result = {
            "key_id": key_pair.key_id,
            "public_key": key_pair.public_key,
            "key_file": str(output),
            "created_at": str(key_pair.created_at),
        }
        print(json.dumps(result, indent=2))
        return

    console.print(Panel(
        f"[bold]Key ID:[/bold] {key_pair.key_id}\n"
        f"[bold]Public Key:[/bold] {key_pair.public_key[:32]}...\n"
        f"[bold]Created:[/bold] {key_pair.created_at}\n"
        f"[bold]Saved to:[/bold] {output}",
        title="[green]✓ Key Pair Generated[/green]",
        border_style="green",
    ))

    console.print("\n[yellow]⚠ Keep your private key secure![/yellow]")
    console.print(f"[dim]Share only the public key for verification.[/dim]")


@app.command(name="sign")
def sign(
    receipt_file: Annotated[
        Path,
        typer.Argument(help="Path to the receipt JSON file"),
    ],
    key_file: Annotated[
        Path | None,
        typer.Option("--key", "-k", help="Path to the signing key file"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for signed receipt"),
    ] = None,
    in_place: Annotated[
        bool,
        typer.Option("--in-place", "-i", help="Sign receipt in place"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Sign a receipt with an Ed25519 private key.

    Creates a cryptographic signature for a BLACKICE receipt, enabling
    verification of its authenticity and integrity.

    Examples:
        blackice receipt sign receipt.json
        blackice receipt sign receipt.json --key keys/signing.key
        blackice receipt sign receipt.json --output signed-receipt.json
        blackice receipt sign receipt.json --in-place
    """
    from blackice.schemas.receipt import Receipt
    from blackice.security import KeyPair, sign_receipt

    # Validate receipt file
    if not receipt_file.exists():
        console.print(f"[red]Receipt file not found:[/red] {receipt_file}")
        raise typer.Exit(1)

    # Determine key file
    if key_file is None:
        key_file = Path.cwd() / ".blackice" / "keys" / "signing.key"

    if not key_file.exists():
        console.print(f"[red]Key file not found:[/red] {key_file}")
        console.print("Generate a key with: blackice receipt keygen")
        raise typer.Exit(1)

    # Load key pair
    try:
        key_pair = KeyPair.from_file(key_file)
    except Exception as e:
        console.print(f"[red]Failed to load key:[/red] {e}")
        raise typer.Exit(1)

    # Load receipt
    try:
        receipt_data = json.loads(receipt_file.read_text())
        receipt = Receipt.model_validate(receipt_data)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in receipt file:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Invalid receipt format:[/red] {e}")
        raise typer.Exit(1)

    # Check if already signed
    if receipt.signature is not None:
        console.print("[yellow]Warning: Receipt is already signed[/yellow]")
        console.print("The existing signature will be replaced")

    # Sign the receipt
    signature = sign_receipt(receipt, key_pair)

    # Create signed receipt (receipts are frozen, so rebuild)
    signed_data = receipt.model_dump()
    signed_data["signature"] = signature.model_dump()

    # Determine output path
    if in_place:
        output = receipt_file
    elif output is None:
        output = receipt_file.with_name(receipt_file.stem + "-signed.json")

    # Write signed receipt
    output.write_text(json.dumps(signed_data, indent=2, default=str))

    if json_output:
        result = {
            "receipt_id": receipt.id,
            "key_id": signature.public_key_id,
            "signature": signature.signature[:32] + "...",
            "output_file": str(output),
            "signed_at": str(signature.timestamp),
        }
        print(json.dumps(result, indent=2))
        return

    console.print(Panel(
        f"[bold]Receipt:[/bold] {receipt.id}\n"
        f"[bold]Key ID:[/bold] {signature.public_key_id}\n"
        f"[bold]Algorithm:[/bold] {signature.algorithm}\n"
        f"[bold]Signed at:[/bold] {signature.timestamp}\n"
        f"[bold]Output:[/bold] {output}",
        title="[green]✓ Receipt Signed[/green]",
        border_style="green",
    ))


@app.command(name="verify")
def verify(
    receipt_file: Annotated[
        Path,
        typer.Argument(help="Path to the signed receipt JSON file"),
    ],
    public_key: Annotated[
        str | None,
        typer.Option("--public-key", "-p", help="Base64-encoded public key"),
    ] = None,
    key_file: Annotated[
        Path | None,
        typer.Option("--key-file", "-k", help="Path to key file containing public key"),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="Workspace path for artifact verification"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
) -> None:
    """Verify a signed receipt.

    Checks the cryptographic signature and optionally verifies artifact
    hashes against actual files.

    Examples:
        blackice receipt verify signed-receipt.json
        blackice receipt verify receipt.json --public-key "base64..."
        blackice receipt verify receipt.json --key-file keys/signing.key
        blackice receipt verify receipt.json --workspace .blackice/run-001
    """
    from blackice.schemas.receipt import Receipt
    from blackice.security import KeyPair, verify_receipt_signature

    # Validate receipt file
    if not receipt_file.exists():
        console.print(f"[red]Receipt file not found:[/red] {receipt_file}")
        raise typer.Exit(1)

    # Load receipt
    try:
        receipt_data = json.loads(receipt_file.read_text())
        receipt = Receipt.model_validate(receipt_data)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in receipt file:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Invalid receipt format:[/red] {e}")
        raise typer.Exit(1)

    # Check if signed
    if receipt.signature is None:
        console.print("[red]Receipt is not signed[/red]")
        raise typer.Exit(1)

    # Get public key
    if public_key is None:
        if key_file is not None:
            try:
                key_pair = KeyPair.from_file(key_file)
                public_key = key_pair.public_key
            except Exception as e:
                console.print(f"[red]Failed to load key file:[/red] {e}")
                raise typer.Exit(1)
        else:
            # Try default key location
            default_key = Path.cwd() / ".blackice" / "keys" / "signing.key"
            if default_key.exists():
                try:
                    key_pair = KeyPair.from_file(default_key)
                    public_key = key_pair.public_key
                except Exception:
                    pass

    if public_key is None:
        console.print("[red]No public key provided[/red]")
        console.print("Use --public-key or --key-file to specify a key")
        raise typer.Exit(1)

    # Verify signature
    is_valid, message = verify_receipt_signature(receipt, receipt.signature, public_key)

    # Verify artifacts if workspace provided
    artifact_valid = True
    artifact_mismatches: list[str] = []
    if workspace is not None:
        if workspace.exists():
            artifact_valid, artifact_mismatches = receipt.verify_artifacts(str(workspace))
        else:
            artifact_mismatches.append(f"Workspace not found: {workspace}")
            artifact_valid = False

    # Build result
    result = {
        "receipt_id": receipt.id,
        "signature_valid": is_valid,
        "signature_message": message,
        "key_id": receipt.signature.public_key_id,
        "signed_at": str(receipt.signature.timestamp),
    }

    if workspace is not None:
        result["artifacts_valid"] = artifact_valid
        result["artifact_mismatches"] = artifact_mismatches

    if json_output:
        print(json.dumps(result, indent=2))
        if not is_valid or (workspace and not artifact_valid):
            raise typer.Exit(1)
        return

    # Rich output
    if is_valid:
        sig_status = "[green]✓ Valid[/green]"
    else:
        sig_status = "[red]✗ Invalid[/red]"

    content = (
        f"[bold]Receipt:[/bold] {receipt.id}\n"
        f"[bold]Signature:[/bold] {sig_status}\n"
        f"[bold]Key ID:[/bold] {receipt.signature.public_key_id}\n"
        f"[bold]Signed at:[/bold] {receipt.signature.timestamp}"
    )

    if workspace is not None:
        if artifact_valid:
            content += f"\n[bold]Artifacts:[/bold] [green]✓ Verified ({receipt.artifact_count} files)[/green]"
        else:
            content += f"\n[bold]Artifacts:[/bold] [red]✗ Verification failed[/red]"
            for mismatch in artifact_mismatches[:5]:
                content += f"\n  • {mismatch}"
            if len(artifact_mismatches) > 5:
                content += f"\n  • ... and {len(artifact_mismatches) - 5} more"

    overall_valid = is_valid and (workspace is None or artifact_valid)
    title = "[green]✓ Verification Passed[/green]" if overall_valid else "[red]✗ Verification Failed[/red]"
    border = "green" if overall_valid else "red"

    console.print(Panel(content, title=title, border_style=border))

    if not overall_valid:
        raise typer.Exit(1)


@app.command(name="show")
def show(
    receipt_file: Annotated[
        Path,
        typer.Argument(help="Path to the receipt JSON file"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed information"),
    ] = False,
) -> None:
    """Display receipt information.

    Shows details about a BLACKICE receipt including artifacts,
    provenance, and signature status.

    Examples:
        blackice receipt show receipt.json
        blackice receipt show receipt.json --verbose
    """
    from blackice.schemas.receipt import Receipt

    # Validate receipt file
    if not receipt_file.exists():
        console.print(f"[red]Receipt file not found:[/red] {receipt_file}")
        raise typer.Exit(1)

    # Load receipt
    try:
        receipt_data = json.loads(receipt_file.read_text())
        receipt = Receipt.model_validate(receipt_data)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in receipt file:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Invalid receipt format:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(receipt.to_verification_summary(), indent=2))
        return

    # Basic information
    signed_status = "[green]✓ Signed[/green]" if receipt.signature else "[yellow]Not signed[/yellow]"
    evidence_status = "[green]✓ All passed[/green]" if receipt.evidence_all_passed else "[red]✗ Some failed[/red]"

    content = (
        f"[bold]ID:[/bold] {receipt.id}\n"
        f"[bold]Run ID:[/bold] {receipt.run_id}\n"
        f"[bold]Version:[/bold] {receipt.version}\n"
        f"[bold]Created:[/bold] {receipt.created_at}\n"
        f"[bold]Signed:[/bold] {signed_status}\n"
        f"\n[bold cyan]Artifacts[/bold cyan]\n"
        f"  Count: {receipt.artifact_count}\n"
        f"  Total Size: {_format_bytes(receipt.total_size_bytes)}\n"
        f"\n[bold cyan]Evidence[/bold cyan]\n"
        f"  Count: {len(receipt.evidence_refs)}\n"
        f"  Status: {evidence_status}\n"
        f"\n[bold cyan]Verification[/bold cyan]\n"
        f"  Events: {receipt.verification.event_count}\n"
        f"  Deviations: {receipt.verification.deviation_count}"
    )

    if receipt.verification.taskspec_id:
        content += f"\n  TaskSpec: {receipt.verification.taskspec_id}"

    console.print(Panel(
        content,
        title=f"[blue]Receipt: {receipt.id}[/blue]",
        border_style="blue",
    ))

    # Verbose: show provenance
    if verbose:
        console.print("\n[bold cyan]Provenance[/bold cyan]")
        console.print(f"  Model: {receipt.provenance.model_provider}/{receipt.provenance.model_name}")
        console.print(f"  BLACKICE: v{receipt.provenance.blackice_version}")
        console.print(f"  Python: {receipt.provenance.python_version}")
        console.print(f"  Platform: {receipt.provenance.platform}")
        console.print(f"  Duration: {receipt.provenance.duration_seconds:.1f}s")
        console.print(f"  Tokens: {receipt.provenance.total_prompt_tokens} prompt, {receipt.provenance.total_completion_tokens} completion")

    # Verbose: show artifacts
    if verbose and receipt.artifact_hashes:
        console.print("\n[bold cyan]Artifact Hashes[/bold cyan]")
        table = Table()
        table.add_column("Path", style="cyan")
        table.add_column("Hash", style="dim")
        table.add_column("Size", justify="right")

        for artifact in receipt.artifact_hashes[:20]:
            hash_short = artifact.hash.value[:16] + "..."
            table.add_row(artifact.path, hash_short, _format_bytes(artifact.size_bytes))

        if len(receipt.artifact_hashes) > 20:
            table.add_row("...", "...", f"({len(receipt.artifact_hashes) - 20} more)")

        console.print(table)

    # Verbose: show evidence
    if verbose and receipt.evidence_refs:
        console.print("\n[bold cyan]Evidence References[/bold cyan]")
        table = Table()
        table.add_column("ID", style="cyan")
        table.add_column("Type")
        table.add_column("Status")

        for ref in receipt.evidence_refs:
            status_color = "green" if ref.status == "passed" else "red" if ref.status == "failed" else "yellow"
            table.add_row(ref.evidence_id, ref.evidence_type, f"[{status_color}]{ref.status}[/{status_color}]")

        console.print(table)

    # Signature info
    if verbose and receipt.signature:
        console.print("\n[bold cyan]Signature[/bold cyan]")
        console.print(f"  Algorithm: {receipt.signature.algorithm}")
        console.print(f"  Key ID: {receipt.signature.public_key_id}")
        console.print(f"  Signed at: {receipt.signature.timestamp}")
        console.print(f"  Signature: {receipt.signature.signature[:32]}...")


@app.command(name="export-public-key")
def export_public_key(
    key_file: Annotated[
        Path | None,
        typer.Option("--key-file", "-k", help="Path to the key file"),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file for public key"),
    ] = None,
) -> None:
    """Export the public key from a key pair file.

    Extracts and displays the public key for sharing with verifiers.

    Examples:
        blackice receipt export-public-key
        blackice receipt export-public-key --key-file keys/signing.key
        blackice receipt export-public-key --output public.key
    """
    from blackice.security import KeyPair

    # Determine key file
    if key_file is None:
        key_file = Path.cwd() / ".blackice" / "keys" / "signing.key"

    if not key_file.exists():
        console.print(f"[red]Key file not found:[/red] {key_file}")
        raise typer.Exit(1)

    # Load key pair
    try:
        key_pair = KeyPair.from_file(key_file)
    except Exception as e:
        console.print(f"[red]Failed to load key:[/red] {e}")
        raise typer.Exit(1)

    if output:
        output.write_text(f"key_id={key_pair.key_id}\npublic_key={key_pair.public_key}\n")
        console.print(f"[green]Public key exported to:[/green] {output}")
    else:
        console.print(f"[bold]Key ID:[/bold] {key_pair.key_id}")
        console.print(f"[bold]Public Key:[/bold]")
        console.print(key_pair.public_key)


def _format_bytes(size: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
