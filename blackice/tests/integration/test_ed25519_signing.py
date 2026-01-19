"""Integration tests for Ed25519 signing (IT-008).

Tests the Ed25519 signing implementation for receipts:
- Key pair generation
- Receipt signing
- Signature verification
- Key serialization
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from blackice.primitives.types import Hash, PIIPolicy, RunId, Timestamp
from blackice.schemas.receipt import (
    ProvenanceInfo,
    Receipt,
    ReceiptBuilder,
    VerificationInfo,
)
from blackice.security import (
    KeyPair,
    SigningError,
    VerificationError,
    generate_key_pair,
    load_private_key,
    load_public_key,
    sign_receipt,
    verify_receipt_signature,
)
from blackice.security.signing import sign_data, verify_data


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


class TestKeyPairGeneration:
    """Test key pair generation."""

    def test_generate_key_pair(self) -> None:
        """Should generate valid Ed25519 key pair."""
        key_pair = generate_key_pair()

        assert key_pair.key_id.startswith("blackice-")
        assert len(key_pair.private_key) > 0
        assert len(key_pair.public_key) > 0
        assert key_pair.created_at is not None

    def test_generate_key_pair_with_custom_id(self) -> None:
        """Should use custom key ID when provided."""
        key_pair = generate_key_pair(key_id="custom-key-123")

        assert key_pair.key_id == "custom-key-123"

    def test_generated_keys_are_unique(self) -> None:
        """Each generated key pair should be unique."""
        key1 = generate_key_pair()
        key2 = generate_key_pair()

        assert key1.private_key != key2.private_key
        assert key1.public_key != key2.public_key

    def test_key_pair_can_get_signing_key(self) -> None:
        """KeyPair should provide NaCl signing key."""
        key_pair = generate_key_pair()
        signing_key = key_pair.get_signing_key()

        assert signing_key is not None

    def test_key_pair_can_get_verify_key(self) -> None:
        """KeyPair should provide NaCl verify key."""
        key_pair = generate_key_pair()
        verify_key = key_pair.get_verify_key()

        assert verify_key is not None


class TestKeyLoading:
    """Test key loading functions."""

    def test_load_private_key(self) -> None:
        """Should load private key from base64."""
        key_pair = generate_key_pair()
        loaded = load_private_key(key_pair.private_key)

        assert loaded is not None
        # Signing with both should produce same result
        test_data = b"test message"
        orig_sig = key_pair.get_signing_key().sign(test_data).signature
        loaded_sig = loaded.sign(test_data).signature
        assert orig_sig == loaded_sig

    def test_load_public_key(self) -> None:
        """Should load public key from base64."""
        key_pair = generate_key_pair()
        loaded = load_public_key(key_pair.public_key)

        assert loaded is not None

    def test_load_invalid_private_key_raises(self) -> None:
        """Should raise SigningError for invalid private key."""
        with pytest.raises(SigningError):
            load_private_key("not-a-valid-key")

    def test_load_invalid_public_key_raises(self) -> None:
        """Should raise VerificationError for invalid public key."""
        with pytest.raises(VerificationError):
            load_public_key("not-a-valid-key")


class TestKeyFileSerialization:
    """Test key file serialization."""

    def test_save_and_load_key_pair(self, tmp_path: Path) -> None:
        """Should save and load key pair from file."""
        key_pair = generate_key_pair(key_id="test-key")
        key_file = tmp_path / "test.key"

        key_pair.to_file(key_file)
        loaded = KeyPair.from_file(key_file)

        assert loaded.key_id == key_pair.key_id
        assert loaded.private_key == key_pair.private_key
        assert loaded.public_key == key_pair.public_key


class TestReceiptSigning:
    """Test receipt signing operations."""

    def test_sign_receipt(self) -> None:
        """Should sign receipt and return Signature."""
        key_pair = generate_key_pair()
        receipt = make_receipt()

        signature = sign_receipt(receipt, key_pair)

        assert signature.algorithm == "ed25519"
        assert signature.public_key_id == key_pair.key_id
        assert len(signature.signature) > 0

    def test_signature_is_deterministic(self) -> None:
        """Same receipt+key should produce same signature content."""
        key_pair = generate_key_pair()
        receipt = make_receipt()

        sig1 = sign_receipt(receipt, key_pair)
        sig2 = sign_receipt(receipt, key_pair)

        # The signatures themselves should be the same (same key, same digest)
        assert sig1.signature == sig2.signature

    def test_different_keys_produce_different_signatures(self) -> None:
        """Different keys should produce different signatures."""
        key1 = generate_key_pair()
        key2 = generate_key_pair()
        receipt = make_receipt()

        sig1 = sign_receipt(receipt, key1)
        sig2 = sign_receipt(receipt, key2)

        assert sig1.signature != sig2.signature


class TestSignatureVerification:
    """Test signature verification operations."""

    def test_verify_valid_signature(self) -> None:
        """Should verify valid receipt signature."""
        key_pair = generate_key_pair()
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        is_valid, message = verify_receipt_signature(
            receipt, signature, key_pair.public_key
        )

        assert is_valid is True
        assert "verified" in message.lower()

    def test_verify_with_verify_key_object(self) -> None:
        """Should verify using VerifyKey object."""
        key_pair = generate_key_pair()
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        verify_key = key_pair.get_verify_key()
        is_valid, message = verify_receipt_signature(receipt, signature, verify_key)

        assert is_valid is True

    def test_reject_wrong_key(self) -> None:
        """Should reject signature verified with wrong key."""
        key1 = generate_key_pair()
        key2 = generate_key_pair()
        receipt = make_receipt()
        signature = sign_receipt(receipt, key1)

        is_valid, message = verify_receipt_signature(
            receipt, signature, key2.public_key
        )

        assert is_valid is False
        assert "failed" in message.lower()

    def test_reject_tampered_receipt(self) -> None:
        """Should reject if receipt content changed after signing."""
        key_pair = generate_key_pair()

        # Sign original receipt
        receipt1 = make_receipt()
        signature = sign_receipt(receipt1, key_pair)

        # Create different receipt (different vision hash)
        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "Different vision")
        builder.add_artifact("src/main.py", b"print('hello')")
        builder.set_provenance(make_provenance())
        builder.set_verification(make_verification())
        receipt2 = builder.build("rcpt-test")

        is_valid, message = verify_receipt_signature(
            receipt2, signature, key_pair.public_key
        )

        assert is_valid is False

    def test_reject_invalid_public_key(self) -> None:
        """Should reject invalid public key gracefully."""
        key_pair = generate_key_pair()
        receipt = make_receipt()
        signature = sign_receipt(receipt, key_pair)

        is_valid, message = verify_receipt_signature(
            receipt, signature, "invalid-key"
        )

        assert is_valid is False
        assert "invalid" in message.lower()


class TestDataSigning:
    """Test arbitrary data signing."""

    def test_sign_and_verify_data(self) -> None:
        """Should sign and verify arbitrary data."""
        key_pair = generate_key_pair()
        data = b"test data to sign"

        signature = sign_data(data, key_pair)
        is_valid, message = verify_data(data, signature, key_pair.public_key)

        assert is_valid is True

    def test_reject_modified_data(self) -> None:
        """Should reject signature on modified data."""
        key_pair = generate_key_pair()
        data = b"original data"
        modified_data = b"modified data"

        signature = sign_data(data, key_pair)
        is_valid, message = verify_data(modified_data, signature, key_pair.public_key)

        assert is_valid is False

    def test_reject_invalid_signature_encoding(self) -> None:
        """Should reject invalid signature encoding."""
        key_pair = generate_key_pair()
        data = b"test data"

        is_valid, message = verify_data(data, "not-base64!!!", key_pair.public_key)

        assert is_valid is False
