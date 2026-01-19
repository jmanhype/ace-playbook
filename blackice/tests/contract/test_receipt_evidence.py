"""Contract tests validating evidence presence in receipts (IT-006/T074c).

These tests verify the contract between the Evidence model and Receipt schema:
- Receipts MUST include evidence references when evidence is added
- Evidence hashes MUST be computed correctly
- evidence_all_passed MUST correctly reflect the evidence status
- Evidence references MUST preserve the evidence type and status
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from blackice.primitives.types import Hash, RunId, Timestamp
from blackice.schemas.evidence import (
    Evidence,
    EvidenceCollection,
    EvidenceStatus,
    EvidenceType,
    SecurityScan,
    TestReport,
)
from blackice.schemas.receipt import (
    EvidenceReference,
    ProvenanceInfo,
    Receipt,
    ReceiptBuilder,
    VerificationInfo,
)


def make_run_id() -> RunId:
    """Create a valid RunId."""
    return RunId(uuid4())


def make_hash(content: str = "test") -> str:
    """Create a valid SHA-256 hash."""
    return hashlib.sha256(content.encode()).hexdigest()


def make_provenance() -> ProvenanceInfo:
    """Create valid provenance info."""
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
    """Create valid verification info."""
    return VerificationInfo(
        event_log_hash=make_hash("events"),
        event_count=25,
    )


def make_test_evidence(passed: bool = True, run_id: RunId | None = None) -> Evidence:
    """Create a test report evidence item."""
    report = TestReport(
        framework="pytest",
        total_tests=10,
        passed=10 if passed else 5,
        failed=0 if passed else 5,
        skipped=0,
        errors=0,
        duration_seconds=5.0,
        coverage_percent=90.0,
    )
    return Evidence.from_test_report(
        evidence_id=f"ev-test-{uuid4().hex[:8]}",
        run_id=run_id or make_run_id(),
        report=report,
    )


def make_security_evidence(passed: bool = True, run_id: RunId | None = None) -> Evidence:
    """Create a security scan evidence item."""
    scan = SecurityScan(
        scanner="bandit",
        scan_type="sast",
        total_findings=0 if passed else 3,
        critical=0 if passed else 1,
        high=0 if passed else 2,
        medium=0,
        low=0,
        info=0,
        findings=[],
        scan_duration_seconds=2.0,
    )
    return Evidence.from_security_scan(
        evidence_id=f"ev-sec-{uuid4().hex[:8]}",
        run_id=run_id or make_run_id(),
        scan=scan,
    )


def build_receipt(builder: ReceiptBuilder, receipt_id: str) -> Receipt:
    """Helper to build a receipt with required provenance and verification."""
    builder.set_provenance(make_provenance())
    builder.set_verification(make_verification())
    return builder.build(receipt_id)


class TestEvidenceReceiptContract:
    """Contract tests for evidence-receipt integration."""

    def test_receipt_must_include_evidence_when_added(self) -> None:
        """CONTRACT: Receipts MUST contain evidence references when evidence is added."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")
        evidence = make_test_evidence()

        builder.add_evidence(
            evidence_id=evidence.id,
            evidence_type=evidence.evidence_type.value,
            status=evidence.status.value,
            content_hash=evidence.compute_hash().value,
        )

        receipt = build_receipt(builder, "rcpt-001")

        # Contract: evidence_refs must contain the added evidence
        assert len(receipt.evidence_refs) == 1
        assert receipt.evidence_refs[0].evidence_id == evidence.id

    def test_evidence_hash_must_be_preserved(self) -> None:
        """CONTRACT: Evidence content hash MUST be preserved in receipt."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")
        evidence = make_test_evidence()
        evidence_hash = evidence.compute_hash().value

        builder.add_evidence(
            evidence_id=evidence.id,
            evidence_type=evidence.evidence_type.value,
            status=evidence.status.value,
            content_hash=evidence_hash,
        )

        receipt = build_receipt(builder, "rcpt-002")

        # Contract: hash must match exactly
        assert receipt.evidence_refs[0].content_hash == evidence_hash

    def test_evidence_type_must_be_preserved(self) -> None:
        """CONTRACT: Evidence type MUST be preserved in receipt reference."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")
        evidence = make_test_evidence()

        builder.add_evidence(
            evidence_id=evidence.id,
            evidence_type=evidence.evidence_type.value,
            status=evidence.status.value,
            content_hash=evidence.compute_hash().value,
        )

        receipt = build_receipt(builder, "rcpt-003")

        # Contract: type must match
        assert receipt.evidence_refs[0].evidence_type == evidence.evidence_type.value

    def test_evidence_status_must_be_preserved(self) -> None:
        """CONTRACT: Evidence status MUST be preserved in receipt reference."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")
        evidence = make_test_evidence()

        builder.add_evidence(
            evidence_id=evidence.id,
            evidence_type=evidence.evidence_type.value,
            status=evidence.status.value,
            content_hash=evidence.compute_hash().value,
        )

        receipt = build_receipt(builder, "rcpt-004")

        # Contract: status must match
        assert receipt.evidence_refs[0].status == evidence.status.value

    def test_evidence_all_passed_true_when_all_pass(self) -> None:
        """CONTRACT: evidence_all_passed MUST be True when all evidence passes."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")

        # Add two passing evidence items
        test_evidence = make_test_evidence(passed=True)
        security_evidence = make_security_evidence(passed=True)

        builder.add_evidence(
            evidence_id=test_evidence.id,
            evidence_type=test_evidence.evidence_type.value,
            status="passed",
            content_hash=test_evidence.compute_hash().value,
        )
        builder.add_evidence(
            evidence_id=security_evidence.id,
            evidence_type=security_evidence.evidence_type.value,
            status="passed",
            content_hash=security_evidence.compute_hash().value,
        )

        receipt = build_receipt(builder, "rcpt-005")

        # Contract: all passed flag must be True
        assert receipt.evidence_all_passed is True

    def test_evidence_all_passed_false_when_any_fails(self) -> None:
        """CONTRACT: evidence_all_passed MUST be False when any evidence fails."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")

        # Add one passing, one failing
        test_evidence = make_test_evidence(passed=True)
        security_evidence = make_security_evidence(passed=False)

        builder.add_evidence(
            evidence_id=test_evidence.id,
            evidence_type=test_evidence.evidence_type.value,
            status="passed",
            content_hash=test_evidence.compute_hash().value,
        )
        builder.add_evidence(
            evidence_id=security_evidence.id,
            evidence_type=security_evidence.evidence_type.value,
            status="failed",
            content_hash=security_evidence.compute_hash().value,
        )

        receipt = build_receipt(builder, "rcpt-006")

        # Contract: all passed flag must be False
        assert receipt.evidence_all_passed is False

    def test_empty_evidence_defaults_to_passed(self) -> None:
        """CONTRACT: evidence_all_passed MUST be True when no evidence is added."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")
        receipt = build_receipt(builder, "rcpt-007")

        # Contract: no evidence means "vacuously passed"
        assert receipt.evidence_all_passed is True
        assert len(receipt.evidence_refs) == 0

    def test_multiple_evidence_items_all_preserved(self) -> None:
        """CONTRACT: All evidence items MUST be preserved in receipt."""
        builder = ReceiptBuilder(make_run_id(), "Test vision")

        evidence_items = [
            make_test_evidence(),
            make_security_evidence(),
            make_test_evidence(),  # Can have multiple test reports
        ]

        for evidence in evidence_items:
            builder.add_evidence(
                evidence_id=evidence.id,
                evidence_type=evidence.evidence_type.value,
                status=evidence.status.value,
                content_hash=evidence.compute_hash().value,
            )

        receipt = build_receipt(builder, "rcpt-008")

        # Contract: all evidence must be preserved
        assert len(receipt.evidence_refs) == 3
        evidence_ids = {ref.evidence_id for ref in receipt.evidence_refs}
        for evidence in evidence_items:
            assert evidence.id in evidence_ids


class TestEvidenceReferenceModel:
    """Contract tests for the EvidenceReference model."""

    def test_evidence_reference_requires_id(self) -> None:
        """CONTRACT: EvidenceReference MUST require evidence_id."""
        with pytest.raises(Exception):  # ValidationError
            EvidenceReference(
                evidence_type="test_report",
                status="passed",
                content_hash=make_hash(),
            )

    def test_evidence_reference_requires_type(self) -> None:
        """CONTRACT: EvidenceReference MUST require evidence_type."""
        with pytest.raises(Exception):  # ValidationError
            EvidenceReference(
                evidence_id="ev-001",
                status="passed",
                content_hash=make_hash(),
            )

    def test_evidence_reference_requires_hash(self) -> None:
        """CONTRACT: EvidenceReference MUST require content_hash."""
        with pytest.raises(Exception):  # ValidationError
            EvidenceReference(
                evidence_id="ev-001",
                evidence_type="test_report",
                status="passed",
            )

    def test_evidence_reference_requires_status(self) -> None:
        """CONTRACT: EvidenceReference MUST require status."""
        with pytest.raises(Exception):  # ValidationError
            EvidenceReference(
                evidence_id="ev-001",
                evidence_type="test_report",
                content_hash=make_hash(),
            )


class TestEvidenceCollectionContract:
    """Contract tests for EvidenceCollection integration."""

    def test_collection_can_build_receipt_evidence_refs(self) -> None:
        """CONTRACT: EvidenceCollection items MUST be convertible to receipt refs."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        evidence1 = make_test_evidence(run_id=run_id)
        evidence2 = make_security_evidence(run_id=run_id)

        collection.add(evidence1)
        collection.add(evidence2)

        # Build receipt from collection
        builder = ReceiptBuilder(run_id, "Test vision")

        for evidence in collection.evidence_items:
            builder.add_evidence(
                evidence_id=evidence.id,
                evidence_type=evidence.evidence_type.value,
                status=evidence.status.value,
                content_hash=evidence.compute_hash().value,
            )

        receipt = build_receipt(builder, "rcpt-009")

        # Contract: all collection items must be in receipt
        assert len(receipt.evidence_refs) == 2

    def test_collection_summary_matches_receipt_status(self) -> None:
        """CONTRACT: Collection all_passed MUST match receipt evidence_all_passed."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        evidence1 = make_test_evidence(passed=True, run_id=run_id)
        evidence2 = make_security_evidence(passed=False, run_id=run_id)

        collection.add(evidence1)
        collection.add(evidence2)

        # Build receipt from collection
        builder = ReceiptBuilder(run_id, "Test vision")

        for evidence in collection.evidence_items:
            builder.add_evidence(
                evidence_id=evidence.id,
                evidence_type=evidence.evidence_type.value,
                status=evidence.status.value,
                content_hash=evidence.compute_hash().value,
            )

        receipt = build_receipt(builder, "rcpt-010")
        summary = collection.summary  # It's a property, not a method

        # Contract: statuses must align
        assert summary["all_passed"] == receipt.evidence_all_passed
