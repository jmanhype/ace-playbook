"""Receipt schema for BLACKICE 3.0 Enterprise.

Receipts provide cryptographically signed proof of what was generated,
enabling verification, audit, and compliance for Enterprise deployments.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

from blackice.primitives.types import (
    Hash,
    PIIPolicy,
    RunId,
    Timestamp,
)


class ArtifactHash(BaseModel):
    """Hash of a generated artifact for verification."""

    path: str = Field(..., description="Relative path within workspace")
    hash: Hash = Field(..., description="SHA-256 hash of file contents")
    size_bytes: int = Field(ge=0)
    artifact_type: str = Field(default="file", description="Type: file, directory, symlink")


class ProvenanceInfo(BaseModel):
    """Provenance information about the generation process."""

    # Model information
    model_provider: str
    model_name: str
    model_version: str | None = None

    # Prompt information (hashed, not stored in full)
    system_prompt_hash: str
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)

    # Environment
    blackice_version: str
    python_version: str
    platform: str

    # Timing
    started_at: Timestamp
    completed_at: Timestamp
    duration_seconds: float = Field(ge=0.0)


class VerificationInfo(BaseModel):
    """Information for verifying the receipt."""

    # Hash chain
    event_log_hash: str = Field(..., description="Hash of the complete event log")
    event_count: int = Field(ge=0)

    # TaskSpec reference (if used)
    taskspec_id: str | None = None
    taskspec_version: str | None = None
    taskspec_hash: str | None = None

    # Deviation tracking
    deviation_count: int = Field(default=0, ge=0)
    deviation_acknowledged: bool = Field(default=False)


class Signature(BaseModel):
    """Cryptographic signature for receipt verification."""

    algorithm: str = Field(default="ed25519", description="Signature algorithm")
    public_key_id: str = Field(..., description="ID of the signing key")
    signature: str = Field(..., description="Base64-encoded signature")
    timestamp: Timestamp = Field(default_factory=Timestamp.now)

    # Optional attestation
    attestation_url: str | None = Field(default=None, description="URL for third-party attestation")


class Receipt(BaseModel):
    """A verifiable receipt for a BLACKICE run.

    Receipts (Enterprise feature) provide cryptographic proof of:
    - What software was generated
    - How it was generated (provenance)
    - Compliance with TaskSpecs
    - Integrity of the generation process

    Attributes:
        id: Unique receipt identifier
        run_id: Associated run
        version: Receipt format version
        vision_hash: Hash of the input vision (privacy: no raw vision stored)
        artifact_hashes: Hashes of all generated artifacts
        provenance: Information about the generation process
        verification: Data for verifying the receipt
        signature: Cryptographic signature
        pii_policy: How PII was handled
    """

    id: str = Field(..., min_length=1, max_length=100)
    run_id: RunId
    version: str = Field(default="1.0.0", description="Receipt format version")

    # Input reference (hash only for privacy)
    vision_hash: Hash = Field(..., description="SHA-256 hash of the vision description")

    # Output verification
    artifact_hashes: list[ArtifactHash] = Field(default_factory=list)
    artifact_count: int = Field(ge=0)
    total_size_bytes: int = Field(ge=0)

    # Process information
    provenance: ProvenanceInfo
    verification: VerificationInfo

    # Signature (optional, but recommended)
    signature: Signature | None = Field(default=None)

    # Privacy
    pii_policy: PIIPolicy = Field(default=PIIPolicy.REDACT)

    # Timestamps
    created_at: Timestamp = Field(default_factory=Timestamp.now)
    expires_at: Timestamp | None = Field(default=None, description="Optional expiration")

    # Metadata (redacted as needed)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = True  # Receipts are immutable

    def compute_digest(self) -> str:
        """Compute the digest to be signed.

        Creates a deterministic representation of the receipt content.
        """
        content = {
            "id": self.id,
            "run_id": str(self.run_id),
            "version": self.version,
            "vision_hash": self.vision_hash.value,
            "artifact_hashes": [
                {"path": a.path, "hash": a.hash.value, "size": a.size_bytes}
                for a in self.artifact_hashes
            ],
            "verification": {
                "event_log_hash": self.verification.event_log_hash,
                "event_count": self.verification.event_count,
                "taskspec_hash": self.verification.taskspec_hash,
            },
            "provenance": {
                "model_provider": self.provenance.model_provider,
                "model_name": self.provenance.model_name,
                "system_prompt_hash": self.provenance.system_prompt_hash,
            },
        }
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_artifacts(self, workspace_path: str) -> tuple[bool, list[str]]:
        """Verify artifact hashes against actual files.

        Returns (all_valid, list_of_mismatches).
        """
        import os
        from pathlib import Path

        mismatches: list[str] = []

        for artifact in self.artifact_hashes:
            file_path = Path(workspace_path) / artifact.path

            if not file_path.exists():
                mismatches.append(f"Missing: {artifact.path}")
                continue

            if file_path.is_file():
                with open(file_path, "rb") as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()

                if actual_hash != artifact.hash.value:
                    mismatches.append(f"Hash mismatch: {artifact.path}")

        return len(mismatches) == 0, mismatches

    def to_verification_summary(self) -> dict[str, Any]:
        """Create a summary for verification display."""
        return {
            "receipt_id": self.id,
            "run_id": str(self.run_id),
            "created_at": str(self.created_at),
            "artifact_count": self.artifact_count,
            "total_size_bytes": self.total_size_bytes,
            "event_count": self.verification.event_count,
            "signed": self.signature is not None,
            "taskspec_used": self.verification.taskspec_id is not None,
            "deviations": self.verification.deviation_count,
        }


class ReceiptBuilder:
    """Builder for constructing receipts during run completion."""

    def __init__(self, run_id: RunId, vision: str) -> None:
        self.run_id = run_id
        self.vision_hash = Hash(value=hashlib.sha256(vision.encode()).hexdigest())
        self.artifact_hashes: list[ArtifactHash] = []
        self.provenance: ProvenanceInfo | None = None
        self.verification: VerificationInfo | None = None
        self.pii_policy = PIIPolicy.REDACT

    def add_artifact(self, path: str, content: bytes) -> ReceiptBuilder:
        """Add an artifact hash."""
        hash_value = hashlib.sha256(content).hexdigest()
        self.artifact_hashes.append(
            ArtifactHash(
                path=path,
                hash=Hash(value=hash_value),
                size_bytes=len(content),
            )
        )
        return self

    def set_provenance(self, provenance: ProvenanceInfo) -> ReceiptBuilder:
        """Set provenance information."""
        self.provenance = provenance
        return self

    def set_verification(self, verification: VerificationInfo) -> ReceiptBuilder:
        """Set verification information."""
        self.verification = verification
        return self

    def set_pii_policy(self, policy: PIIPolicy) -> ReceiptBuilder:
        """Set PII handling policy."""
        self.pii_policy = policy
        return self

    def build(self, receipt_id: str) -> Receipt:
        """Build the final receipt."""
        if self.provenance is None:
            raise ValueError("Provenance must be set")
        if self.verification is None:
            raise ValueError("Verification must be set")

        return Receipt(
            id=receipt_id,
            run_id=self.run_id,
            vision_hash=self.vision_hash,
            artifact_hashes=self.artifact_hashes,
            artifact_count=len(self.artifact_hashes),
            total_size_bytes=sum(a.size_bytes for a in self.artifact_hashes),
            provenance=self.provenance,
            verification=self.verification,
            pii_policy=self.pii_policy,
        )
