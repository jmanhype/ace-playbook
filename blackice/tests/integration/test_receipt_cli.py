"""Integration tests for Receipt CLI commands (IT-009).

Tests the receipt CLI subcommands:
- keygen: Generate signing key pairs
- sign: Sign receipts
- verify: Verify receipt signatures
- show: Display receipt information
- export-public-key: Export public key
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from typer.testing import CliRunner

from blackice.cli.main import app
from blackice.primitives.types import Hash, PIIPolicy, RunId, Timestamp
from blackice.schemas.receipt import (
    ProvenanceInfo,
    Receipt,
    ReceiptBuilder,
    VerificationInfo,
)
from blackice.security import generate_key_pair, sign_receipt

runner = CliRunner()


def make_run_id() -> RunId:
    """Create a valid RunId (UUID)."""
    return RunId(uuid4())


def make_hash(content: str = "test") -> str:
    """Create a valid SHA-256 hash."""
    return hashlib.sha256(content.encode()).hexdigest()


def make_provenance() -> ProvenanceInfo:
    """Create a valid provenance info."""
    now = Timestamp.now()
    return ProvenanceInfo(
        model_provider="anthropic",
        model_name="claude-3",
        blackice_version="3.0.0",
        python_version="3.10.0",
        platform="darwin",
        system_prompt_hash=make_hash("system"),
        total_prompt_tokens=100,
        total_completion_tokens=500,
        started_at=now,
        completed_at=now,
        duration_seconds=10.0,
    )


def make_verification() -> VerificationInfo:
    """Create a valid verification info."""
    return VerificationInfo(
        event_log_hash=make_hash("events"),
        event_count=25,
    )


def make_receipt() -> Receipt:
    """Create a valid receipt for testing."""
    run_id = make_run_id()
    builder = ReceiptBuilder(run_id, "Test vision")
    builder.add_artifact("src/main.py", b"print('hello')")
    builder.set_provenance(make_provenance())
    builder.set_verification(make_verification())
    return builder.build("rcpt-test")


class TestKeygenCommand:
    """Test the keygen command."""

    def test_keygen_creates_key_file(self, tmp_path: Path) -> None:
        """Should create a key file."""
        key_file = tmp_path / "test.key"

        result = runner.invoke(app, ["receipt", "keygen", "--output", str(key_file)])

        assert result.exit_code == 0
        assert key_file.exists()
        assert "Key Pair Generated" in result.output

    def test_keygen_with_custom_id(self, tmp_path: Path) -> None:
        """Should use custom key ID."""
        key_file = tmp_path / "test.key"

        result = runner.invoke(
            app,
            ["receipt", "keygen", "--output", str(key_file), "--key-id", "my-custom-key"],
        )

        assert result.exit_code == 0
        assert "my-custom-key" in result.output

    def test_keygen_refuses_overwrite(self, tmp_path: Path) -> None:
        """Should refuse to overwrite existing key."""
        key_file = tmp_path / "test.key"
        key_file.write_text("existing")

        result = runner.invoke(app, ["receipt", "keygen", "--output", str(key_file)])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_keygen_force_overwrites(self, tmp_path: Path) -> None:
        """Should overwrite with --force."""
        key_file = tmp_path / "test.key"
        key_file.write_text("existing")

        result = runner.invoke(
            app,
            ["receipt", "keygen", "--output", str(key_file), "--force"],
        )

        assert result.exit_code == 0
        assert "Key Pair Generated" in result.output

    def test_keygen_json_output(self, tmp_path: Path) -> None:
        """Should output JSON when requested."""
        key_file = tmp_path / "test.key"

        result = runner.invoke(
            app,
            ["receipt", "keygen", "--output", str(key_file), "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "key_id" in data
        assert "public_key" in data
        assert "key_file" in data

    def test_keygen_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Should create parent directories."""
        key_file = tmp_path / "nested" / "dirs" / "test.key"

        result = runner.invoke(app, ["receipt", "keygen", "--output", str(key_file)])

        assert result.exit_code == 0
        assert key_file.exists()


class TestSignCommand:
    """Test the sign command."""

    def test_sign_receipt(self, tmp_path: Path) -> None:
        """Should sign a receipt."""
        # Create key
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair()
        key_pair.to_file(key_file)

        # Create receipt
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        # Sign
        result = runner.invoke(
            app,
            ["receipt", "sign", str(receipt_file), "--key", str(key_file)],
        )

        assert result.exit_code == 0
        assert "Receipt Signed" in result.output

        # Verify output file exists
        signed_file = tmp_path / "receipt-signed.json"
        assert signed_file.exists()

    def test_sign_in_place(self, tmp_path: Path) -> None:
        """Should sign receipt in place."""
        # Create key
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair()
        key_pair.to_file(key_file)

        # Create receipt
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        original_content = receipt_file.read_text()

        # Sign in place
        result = runner.invoke(
            app,
            ["receipt", "sign", str(receipt_file), "--key", str(key_file), "--in-place"],
        )

        assert result.exit_code == 0

        # File should be modified
        new_content = receipt_file.read_text()
        assert new_content != original_content
        assert "signature" in new_content

    def test_sign_with_custom_output(self, tmp_path: Path) -> None:
        """Should write to custom output path."""
        # Create key
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair()
        key_pair.to_file(key_file)

        # Create receipt
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        output_file = tmp_path / "custom-signed.json"

        result = runner.invoke(
            app,
            ["receipt", "sign", str(receipt_file), "--key", str(key_file), "--output", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_sign_missing_receipt(self, tmp_path: Path) -> None:
        """Should fail for missing receipt."""
        result = runner.invoke(app, ["receipt", "sign", "/nonexistent/receipt.json"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_sign_missing_key(self, tmp_path: Path) -> None:
        """Should fail for missing key."""
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        result = runner.invoke(
            app,
            ["receipt", "sign", str(receipt_file), "--key", "/nonexistent/key.key"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_sign_json_output(self, tmp_path: Path) -> None:
        """Should output JSON when requested."""
        # Create key
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair()
        key_pair.to_file(key_file)

        # Create receipt
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        result = runner.invoke(
            app,
            ["receipt", "sign", str(receipt_file), "--key", str(key_file), "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "receipt_id" in data
        assert "key_id" in data
        assert "signature" in data


class TestVerifyCommand:
    """Test the verify command."""

    def test_verify_valid_signature(self, tmp_path: Path) -> None:
        """Should verify a valid signature."""
        # Create key
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair()
        key_pair.to_file(key_file)

        # Create and sign receipt
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        signed_data = receipt.model_dump()
        signed_data["signature"] = signature.model_dump()

        signed_file = tmp_path / "signed.json"
        signed_file.write_text(json.dumps(signed_data, default=str))

        # Verify
        result = runner.invoke(
            app,
            ["receipt", "verify", str(signed_file), "--key-file", str(key_file)],
        )

        assert result.exit_code == 0
        assert "Verification Passed" in result.output

    def test_verify_with_public_key(self, tmp_path: Path) -> None:
        """Should verify using base64 public key."""
        # Create key
        key_pair = generate_key_pair()

        # Create and sign receipt
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        signed_data = receipt.model_dump()
        signed_data["signature"] = signature.model_dump()

        signed_file = tmp_path / "signed.json"
        signed_file.write_text(json.dumps(signed_data, default=str))

        # Verify with public key
        result = runner.invoke(
            app,
            ["receipt", "verify", str(signed_file), "--public-key", key_pair.public_key],
        )

        assert result.exit_code == 0
        assert "Verification Passed" in result.output

    def test_verify_invalid_signature(self, tmp_path: Path) -> None:
        """Should fail for invalid signature."""
        # Create two key pairs
        key_pair1 = generate_key_pair()
        key_pair2 = generate_key_pair()

        # Sign with key1
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair1)

        signed_data = receipt.model_dump()
        signed_data["signature"] = signature.model_dump()

        signed_file = tmp_path / "signed.json"
        signed_file.write_text(json.dumps(signed_data, default=str))

        # Verify with key2 (should fail)
        result = runner.invoke(
            app,
            ["receipt", "verify", str(signed_file), "--public-key", key_pair2.public_key],
        )

        assert result.exit_code == 1
        assert "Verification Failed" in result.output or "Invalid" in result.output

    def test_verify_unsigned_receipt(self, tmp_path: Path) -> None:
        """Should fail for unsigned receipt."""
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        result = runner.invoke(app, ["receipt", "verify", str(receipt_file)])

        assert result.exit_code == 1
        assert "not signed" in result.output

    def test_verify_json_output(self, tmp_path: Path) -> None:
        """Should output JSON when requested."""
        # Create key
        key_pair = generate_key_pair()

        # Create and sign receipt
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        signed_data = receipt.model_dump()
        signed_data["signature"] = signature.model_dump()

        signed_file = tmp_path / "signed.json"
        signed_file.write_text(json.dumps(signed_data, default=str))

        result = runner.invoke(
            app,
            ["receipt", "verify", str(signed_file), "--public-key", key_pair.public_key, "--json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["signature_valid"] is True


class TestShowCommand:
    """Test the show command."""

    def test_show_receipt(self, tmp_path: Path) -> None:
        """Should display receipt information."""
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        result = runner.invoke(app, ["receipt", "show", str(receipt_file)])

        assert result.exit_code == 0
        assert "rcpt-test" in result.output
        assert "Artifacts" in result.output

    def test_show_receipt_verbose(self, tmp_path: Path) -> None:
        """Should display detailed info with --verbose."""
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        result = runner.invoke(app, ["receipt", "show", str(receipt_file), "--verbose"])

        assert result.exit_code == 0
        assert "Provenance" in result.output
        assert "anthropic" in result.output
        assert "claude-3" in result.output

    def test_show_signed_receipt(self, tmp_path: Path) -> None:
        """Should show signed status."""
        key_pair = generate_key_pair()
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        signed_data = receipt.model_dump()
        signed_data["signature"] = signature.model_dump()

        signed_file = tmp_path / "signed.json"
        signed_file.write_text(json.dumps(signed_data, default=str))

        result = runner.invoke(app, ["receipt", "show", str(signed_file)])

        assert result.exit_code == 0
        assert "Signed" in result.output

    def test_show_json_output(self, tmp_path: Path) -> None:
        """Should output JSON when requested."""
        receipt = make_receipt()
        receipt_file = tmp_path / "receipt.json"
        receipt_file.write_text(receipt.model_dump_json())

        result = runner.invoke(app, ["receipt", "show", str(receipt_file), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "receipt_id" in data
        assert data["artifact_count"] == 1

    def test_show_missing_receipt(self, tmp_path: Path) -> None:
        """Should fail for missing receipt."""
        result = runner.invoke(app, ["receipt", "show", "/nonexistent/receipt.json"])

        assert result.exit_code == 1
        assert "not found" in result.output


class TestExportPublicKeyCommand:
    """Test the export-public-key command."""

    def test_export_public_key(self, tmp_path: Path) -> None:
        """Should export public key to stdout."""
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair(key_id="test-key-123")
        key_pair.to_file(key_file)

        result = runner.invoke(
            app,
            ["receipt", "export-public-key", "--key-file", str(key_file)],
        )

        assert result.exit_code == 0
        assert "test-key-123" in result.output
        assert key_pair.public_key in result.output

    def test_export_public_key_to_file(self, tmp_path: Path) -> None:
        """Should export public key to file."""
        key_file = tmp_path / "signing.key"
        key_pair = generate_key_pair()
        key_pair.to_file(key_file)

        output_file = tmp_path / "public.key"

        result = runner.invoke(
            app,
            ["receipt", "export-public-key", "--key-file", str(key_file), "--output", str(output_file)],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "public_key=" in content

    def test_export_missing_key(self, tmp_path: Path) -> None:
        """Should fail for missing key file."""
        result = runner.invoke(
            app,
            ["receipt", "export-public-key", "--key-file", "/nonexistent/key.key"],
        )

        assert result.exit_code == 1
        assert "not found" in result.output


class TestReceiptHelpCommand:
    """Test the receipt help command."""

    def test_receipt_help(self) -> None:
        """Should show help for receipt subcommand."""
        result = runner.invoke(app, ["receipt", "--help"])

        assert result.exit_code == 0
        assert "keygen" in result.output
        assert "sign" in result.output
        assert "verify" in result.output
        assert "show" in result.output
