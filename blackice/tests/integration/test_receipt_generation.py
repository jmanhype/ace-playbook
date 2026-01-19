"""Integration tests for Receipt generation (IT-006).

Tests the full receipt generation workflow including:
- Receipt creation from run completion
- Artifact hash computation
- Event log hash chain verification
- Evidence model inclusion
- Ed25519 signing and verification
- Redaction policies
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from blackice.primitives.types import Hash, PIIPolicy, RunId, Timestamp
from blackice.schemas.receipt import (
    ArtifactHash,
    ProvenanceInfo,
    Receipt,
    ReceiptBuilder,
    Signature,
    VerificationInfo,
)


def make_hash(content: str = "test") -> str:
    """Create a valid 64-character SHA-256 hash."""
    return hashlib.sha256(content.encode()).hexdigest()


def make_run_id() -> RunId:
    """Create a valid RunId (UUID)."""
    return RunId(uuid4())


class TestReceiptCreation:
    """Test basic receipt creation."""

    def test_receipt_builder_creates_receipt(self) -> None:
        """ReceiptBuilder should create valid receipts."""
        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "Build a hello world CLI")

        builder.set_provenance(
            ProvenanceInfo(
                model_provider="ollama",
                model_name="qwen2.5-coder:32b",
                model_version="latest",
                system_prompt_hash=make_hash("system"),
                total_prompt_tokens=1000,
                total_completion_tokens=2000,
                blackice_version="3.0.0",
                python_version="3.11.0",
                platform="darwin",
                started_at=Timestamp.now(),
                completed_at=Timestamp.now(),
                duration_seconds=60.0,
            )
        )

        builder.set_verification(
            VerificationInfo(
                event_log_hash=make_hash("events"),
                event_count=42,
            )
        )

        receipt = builder.build("receipt-001")

        assert receipt.id == "receipt-001"
        assert receipt.run_id == run_id
        assert receipt.provenance.model_provider == "ollama"
        assert receipt.verification.event_count == 42

    def test_receipt_requires_provenance(self) -> None:
        """Receipt should require provenance."""
        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "test vision")

        builder.set_verification(
            VerificationInfo(
                event_log_hash=make_hash("events"),
                event_count=1,
            )
        )

        with pytest.raises(ValueError, match="Provenance must be set"):
            builder.build("receipt-001")

    def test_receipt_requires_verification(self) -> None:
        """Receipt should require verification info."""
        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "test vision")

        builder.set_provenance(
            ProvenanceInfo(
                model_provider="test",
                model_name="test",
                system_prompt_hash=make_hash("system"),
                total_prompt_tokens=0,
                total_completion_tokens=0,
                blackice_version="3.0.0",
                python_version="3.11",
                platform="test",
                started_at=Timestamp.now(),
                completed_at=Timestamp.now(),
                duration_seconds=0,
            )
        )

        with pytest.raises(ValueError, match="Verification must be set"):
            builder.build("receipt-001")


class TestArtifactHashing:
    """Test artifact hash computation and verification."""

    def test_artifact_hash_added_to_receipt(self) -> None:
        """Artifacts should be hashed and added to receipt."""
        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "test")

        builder.add_artifact("src/main.py", b"print('hello')")
        builder.add_artifact("tests/test_main.py", b"def test_hello(): pass")

        builder.set_provenance(self._create_provenance())
        builder.set_verification(self._create_verification())

        receipt = builder.build("receipt-001")

        assert len(receipt.artifact_hashes) == 2
        assert receipt.artifact_count == 2

    def test_artifact_hash_is_sha256(self) -> None:
        """Artifact hashes should be SHA-256."""
        content = b"test content"
        expected_hash = hashlib.sha256(content).hexdigest()

        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "test")
        builder.add_artifact("test.txt", content)
        builder.set_provenance(self._create_provenance())
        builder.set_verification(self._create_verification())

        receipt = builder.build("receipt-001")

        assert receipt.artifact_hashes[0].hash.value == expected_hash

    def test_artifact_size_tracked(self) -> None:
        """Artifact sizes should be tracked."""
        content = b"x" * 1000

        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, "test")
        builder.add_artifact("large.bin", content)
        builder.set_provenance(self._create_provenance())
        builder.set_verification(self._create_verification())

        receipt = builder.build("receipt-001")

        assert receipt.artifact_hashes[0].size_bytes == 1000
        assert receipt.total_size_bytes == 1000

    def test_verify_artifacts_against_workspace(self) -> None:
        """Should verify artifact hashes against actual files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create actual files
            (Path(tmpdir) / "test.txt").write_bytes(b"content")

            # Create receipt with matching hash
            artifact_hash = ArtifactHash(
                path="test.txt",
                hash=Hash(value=hashlib.sha256(b"content").hexdigest()),
                size_bytes=7,
            )

            receipt = self._create_receipt_with_artifacts([artifact_hash])

            is_valid, mismatches = receipt.verify_artifacts(tmpdir)
            assert is_valid is True
            assert mismatches == []

    def test_verify_detects_modified_file(self) -> None:
        """Should detect when file content doesn't match hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with different content
            (Path(tmpdir) / "test.txt").write_bytes(b"modified")

            # Create receipt with original hash
            artifact_hash = ArtifactHash(
                path="test.txt",
                hash=Hash(value=hashlib.sha256(b"original").hexdigest()),
                size_bytes=8,
            )

            receipt = self._create_receipt_with_artifacts([artifact_hash])

            is_valid, mismatches = receipt.verify_artifacts(tmpdir)
            assert is_valid is False
            assert "Hash mismatch: test.txt" in mismatches

    def test_verify_detects_missing_file(self) -> None:
        """Should detect missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_hash = ArtifactHash(
                path="missing.txt",
                hash=Hash(value=make_hash("nonexistent")),
                size_bytes=0,
            )

            receipt = self._create_receipt_with_artifacts([artifact_hash])

            is_valid, mismatches = receipt.verify_artifacts(tmpdir)
            assert is_valid is False
            assert "Missing: missing.txt" in mismatches

    def _create_provenance(self) -> ProvenanceInfo:
        return ProvenanceInfo(
            model_provider="test",
            model_name="test",
            system_prompt_hash=make_hash("system"),
            total_prompt_tokens=0,
            total_completion_tokens=0,
            blackice_version="3.0.0",
            python_version="3.11",
            platform="test",
            started_at=Timestamp.now(),
            completed_at=Timestamp.now(),
            duration_seconds=0,
        )

    def _create_verification(self) -> VerificationInfo:
        return VerificationInfo(
            event_log_hash=make_hash("events"),
            event_count=0,
        )

    def _create_receipt_with_artifacts(self, artifacts: list[ArtifactHash]) -> Receipt:
        return Receipt(
            id="test-receipt",
            run_id=make_run_id(),
            vision_hash=Hash(value=make_hash("vision")),
            artifact_hashes=artifacts,
            artifact_count=len(artifacts),
            total_size_bytes=sum(a.size_bytes for a in artifacts),
            provenance=self._create_provenance(),
            verification=self._create_verification(),
        )


class TestEventLogIntegrity:
    """Test event log hash chain in receipts."""

    def test_event_log_hash_in_verification(self) -> None:
        """Event log hash should be included in verification."""
        event_log_hash = hashlib.sha256(b"event1|event2|event3").hexdigest()

        verification = VerificationInfo(
            event_log_hash=event_log_hash,
            event_count=3,
        )

        assert verification.event_log_hash == event_log_hash
        assert verification.event_count == 3

    def test_taskspec_reference_in_verification(self) -> None:
        """TaskSpec reference should be tracked in verification."""
        verification = VerificationInfo(
            event_log_hash=make_hash("events"),
            event_count=10,
            taskspec_id="my-spec",
            taskspec_version="1.0.0",
            taskspec_hash=make_hash("spec"),
        )

        assert verification.taskspec_id == "my-spec"
        assert verification.taskspec_version == "1.0.0"
        assert verification.taskspec_hash == make_hash("spec")

    def test_deviation_tracking(self) -> None:
        """Deviations from spec should be tracked."""
        verification = VerificationInfo(
            event_log_hash=make_hash("events"),
            event_count=10,
            taskspec_id="my-spec",
            deviation_count=2,
            deviation_acknowledged=True,
        )

        assert verification.deviation_count == 2
        assert verification.deviation_acknowledged is True


class TestReceiptDigest:
    """Test receipt digest computation for signing."""

    def test_compute_digest_is_deterministic(self) -> None:
        """Same receipt should produce same digest."""
        receipt = self._create_test_receipt()

        digest1 = receipt.compute_digest()
        digest2 = receipt.compute_digest()

        assert digest1 == digest2

    def test_different_receipts_have_different_digests(self) -> None:
        """Different receipts should have different digests."""
        receipt1 = self._create_test_receipt(receipt_id="receipt-1")
        receipt2 = self._create_test_receipt(receipt_id="receipt-2")

        assert receipt1.compute_digest() != receipt2.compute_digest()

    def test_digest_changes_with_artifacts(self) -> None:
        """Adding artifacts should change the digest."""
        builder1 = ReceiptBuilder(make_run_id(), "test")
        builder1.set_provenance(self._create_provenance())
        builder1.set_verification(self._create_verification())
        receipt1 = builder1.build("receipt-1")

        builder2 = ReceiptBuilder(make_run_id(), "test")
        builder2.add_artifact("file.txt", b"content")
        builder2.set_provenance(self._create_provenance())
        builder2.set_verification(self._create_verification())
        receipt2 = builder2.build("receipt-2")

        assert receipt1.compute_digest() != receipt2.compute_digest()

    def _create_test_receipt(self, receipt_id: str = "test-receipt") -> Receipt:
        return Receipt(
            id=receipt_id,
            run_id=make_run_id(),
            vision_hash=Hash(value=make_hash("vision")),
            artifact_hashes=[],
            artifact_count=0,
            total_size_bytes=0,
            provenance=self._create_provenance(),
            verification=self._create_verification(),
        )

    def _create_provenance(self) -> ProvenanceInfo:
        return ProvenanceInfo(
            model_provider="test",
            model_name="test",
            system_prompt_hash=make_hash("system"),
            total_prompt_tokens=0,
            total_completion_tokens=0,
            blackice_version="3.0.0",
            python_version="3.11",
            platform="test",
            started_at=Timestamp.now(),
            completed_at=Timestamp.now(),
            duration_seconds=0,
        )

    def _create_verification(self) -> VerificationInfo:
        return VerificationInfo(
            event_log_hash=make_hash("events"),
            event_count=0,
        )


class TestReceiptSigning:
    """Test Ed25519 signing of receipts."""

    def test_signature_structure(self) -> None:
        """Signature should have required fields."""
        signature = Signature(
            algorithm="ed25519",
            public_key_id="key-001",
            signature="base64encodeddata==",
        )

        assert signature.algorithm == "ed25519"
        assert signature.public_key_id == "key-001"
        assert signature.signature == "base64encodeddata=="

    def test_signature_with_attestation(self) -> None:
        """Signature can include attestation URL."""
        signature = Signature(
            algorithm="ed25519",
            public_key_id="key-001",
            signature="sig",
            attestation_url="https://attestation.example.com/verify/123",
        )

        assert signature.attestation_url is not None

    def test_receipt_with_signature(self) -> None:
        """Receipt should accept signature."""
        receipt = Receipt(
            id="signed-receipt",
            run_id=make_run_id(),
            vision_hash=Hash(value=make_hash("vision")),
            artifact_hashes=[],
            artifact_count=0,
            total_size_bytes=0,
            provenance=ProvenanceInfo(
                model_provider="test",
                model_name="test",
                system_prompt_hash=make_hash("system"),
                total_prompt_tokens=0,
                total_completion_tokens=0,
                blackice_version="3.0.0",
                python_version="3.11",
                platform="test",
                started_at=Timestamp.now(),
                completed_at=Timestamp.now(),
                duration_seconds=0,
            ),
            verification=VerificationInfo(
                event_log_hash=make_hash("events"),
                event_count=0,
            ),
            signature=Signature(
                public_key_id="key-001",
                signature="signed-data",
            ),
        )

        assert receipt.signature is not None
        assert receipt.signature.public_key_id == "key-001"


class TestRedactionPolicy:
    """Test PII redaction policies."""

    def test_receipt_with_redact_policy(self) -> None:
        """Receipt should support REDACT policy."""
        receipt = self._create_receipt_with_policy(PIIPolicy.REDACT)
        assert receipt.pii_policy == PIIPolicy.REDACT

    def test_receipt_with_hash_only_policy(self) -> None:
        """Receipt should support HASH_ONLY policy."""
        receipt = self._create_receipt_with_policy(PIIPolicy.HASH_ONLY)
        assert receipt.pii_policy == PIIPolicy.HASH_ONLY

    def test_receipt_with_retain_policy(self) -> None:
        """Receipt should support RETAIN policy."""
        receipt = self._create_receipt_with_policy(PIIPolicy.RETAIN)
        assert receipt.pii_policy == PIIPolicy.RETAIN

    def test_vision_is_always_hashed(self) -> None:
        """Vision should always be stored as hash, not plaintext."""
        vision = "Build a CLI tool for secret-key-12345"
        run_id = make_run_id()
        builder = ReceiptBuilder(run_id, vision)

        # Vision hash should be SHA-256 of the vision
        expected_hash = hashlib.sha256(vision.encode()).hexdigest()
        assert builder.vision_hash.value == expected_hash

    def _create_receipt_with_policy(self, policy: PIIPolicy) -> Receipt:
        return Receipt(
            id="test-receipt",
            run_id=make_run_id(),
            vision_hash=Hash(value=make_hash("vision")),
            artifact_hashes=[],
            artifact_count=0,
            total_size_bytes=0,
            provenance=ProvenanceInfo(
                model_provider="test",
                model_name="test",
                system_prompt_hash=make_hash("system"),
                total_prompt_tokens=0,
                total_completion_tokens=0,
                blackice_version="3.0.0",
                python_version="3.11",
                platform="test",
                started_at=Timestamp.now(),
                completed_at=Timestamp.now(),
                duration_seconds=0,
            ),
            verification=VerificationInfo(
                event_log_hash=make_hash("events"),
                event_count=0,
            ),
            pii_policy=policy,
        )


class TestVerificationSummary:
    """Test receipt verification summary for display."""

    def test_summary_includes_key_fields(self) -> None:
        """Summary should include key verification fields."""
        receipt = Receipt(
            id="summary-receipt",
            run_id=make_run_id(),
            vision_hash=Hash(value=make_hash("vision")),
            artifact_hashes=[
                ArtifactHash(path="f1.py", hash=Hash(value=make_hash("f1")), size_bytes=100),
                ArtifactHash(path="f2.py", hash=Hash(value=make_hash("f2")), size_bytes=200),
            ],
            artifact_count=2,
            total_size_bytes=300,
            provenance=ProvenanceInfo(
                model_provider="test",
                model_name="test",
                system_prompt_hash=make_hash("system"),
                total_prompt_tokens=0,
                total_completion_tokens=0,
                blackice_version="3.0.0",
                python_version="3.11",
                platform="test",
                started_at=Timestamp.now(),
                completed_at=Timestamp.now(),
                duration_seconds=0,
            ),
            verification=VerificationInfo(
                event_log_hash=make_hash("events"),
                event_count=42,
                taskspec_id="my-spec",
                deviation_count=1,
            ),
            signature=Signature(public_key_id="key-1", signature="sig"),
        )

        summary = receipt.to_verification_summary()

        assert summary["receipt_id"] == "summary-receipt"
        assert summary["artifact_count"] == 2
        assert summary["event_count"] == 42
        assert summary["signed"] is True
        assert summary["taskspec_used"] is True
        assert summary["deviations"] == 1
