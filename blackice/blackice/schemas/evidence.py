"""Evidence schema for BLACKICE 3.0 Enterprise.

Evidence models capture proof of software quality during generation,
including test reports, security scans, and command outputs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from blackice.primitives.types import Hash, RunId, Timestamp


class EvidenceType(str, Enum):
    """Types of evidence that can be collected."""

    TEST_REPORT = "test_report"
    SECURITY_SCAN = "security_scan"
    LINT_REPORT = "lint_report"
    TYPE_CHECK = "type_check"
    COVERAGE_REPORT = "coverage_report"
    COMMAND_OUTPUT = "command_output"
    BUILD_LOG = "build_log"
    DEPENDENCY_AUDIT = "dependency_audit"
    CUSTOM = "custom"


class EvidenceStatus(str, Enum):
    """Status of evidence collection."""

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    ERROR = "error"


class TestResult(BaseModel):
    """Individual test result within a test report."""

    name: str
    status: str  # passed, failed, skipped, error
    duration_ms: float = Field(ge=0.0)
    error_message: str | None = None
    file_path: str | None = None
    line_number: int | None = None


class TestReport(BaseModel):
    """Test execution report evidence."""

    framework: str = Field(..., description="Test framework used (pytest, jest, etc.)")
    total_tests: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    errors: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    coverage_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    test_results: list[TestResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100.0


class SecurityFinding(BaseModel):
    """Individual security finding from a scan."""

    severity: str  # critical, high, medium, low, info
    category: str
    message: str
    file_path: str | None = None
    line_number: int | None = None
    cwe_id: str | None = None
    cve_id: str | None = None
    remediation: str | None = None


class SecurityScan(BaseModel):
    """Security scan report evidence."""

    scanner: str = Field(..., description="Security scanner used (bandit, semgrep, etc.)")
    scan_type: str = Field(default="sast", description="Type: sast, dast, sca, secrets")
    total_findings: int = Field(ge=0)
    critical: int = Field(default=0, ge=0)
    high: int = Field(default=0, ge=0)
    medium: int = Field(default=0, ge=0)
    low: int = Field(default=0, ge=0)
    info: int = Field(default=0, ge=0)
    findings: list[SecurityFinding] = Field(default_factory=list)
    scan_duration_seconds: float = Field(ge=0.0)

    @property
    def has_critical_issues(self) -> bool:
        """Check if scan found critical or high issues."""
        return self.critical > 0 or self.high > 0


class LintIssue(BaseModel):
    """Individual linting issue."""

    rule: str
    severity: str  # error, warning, info
    message: str
    file_path: str
    line_number: int
    column: int | None = None


class LintReport(BaseModel):
    """Linting/static analysis report evidence."""

    linter: str = Field(..., description="Linter used (ruff, eslint, etc.)")
    total_issues: int = Field(ge=0)
    errors: int = Field(default=0, ge=0)
    warnings: int = Field(default=0, ge=0)
    info: int = Field(default=0, ge=0)
    issues: list[LintIssue] = Field(default_factory=list)
    files_checked: int = Field(default=0, ge=0)

    @property
    def is_clean(self) -> bool:
        """Check if linting passed with no errors."""
        return self.errors == 0


class CommandOutput(BaseModel):
    """Command execution output evidence."""

    command: str
    exit_code: int
    stdout: str = Field(default="", max_length=100000)
    stderr: str = Field(default="", max_length=100000)
    duration_seconds: float = Field(ge=0.0)
    working_directory: str | None = None

    @property
    def succeeded(self) -> bool:
        """Check if command succeeded (exit code 0)."""
        return self.exit_code == 0


class Evidence(BaseModel):
    """Evidence of software quality during generation.

    Evidence (Enterprise feature) provides verifiable proof of:
    - Test execution and results
    - Security scan findings
    - Code quality metrics
    - Build success/failure

    Evidence is attached to receipts for audit and compliance.
    """

    id: str = Field(..., min_length=1, max_length=100)
    run_id: RunId
    evidence_type: EvidenceType
    status: EvidenceStatus
    collected_at: Timestamp = Field(default_factory=Timestamp.now)

    # Evidence data (one of these will be populated based on type)
    test_report: TestReport | None = None
    security_scan: SecurityScan | None = None
    lint_report: LintReport | None = None
    command_output: CommandOutput | None = None

    # For custom evidence types
    custom_data: dict[str, Any] = Field(default_factory=dict)

    # Content hash for integrity
    content_hash: Hash | None = None

    # Metadata
    tool_version: str | None = None
    tags: list[str] = Field(default_factory=list)

    def compute_hash(self) -> Hash:
        """Compute hash of evidence content for integrity."""
        content = self.model_dump_json(exclude={"content_hash"})
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        return Hash(value=hash_value)

    @classmethod
    def from_test_report(
        cls, evidence_id: str, run_id: RunId, report: TestReport
    ) -> Evidence:
        """Create evidence from a test report."""
        status = (
            EvidenceStatus.PASSED
            if report.failed == 0 and report.errors == 0
            else EvidenceStatus.FAILED
        )
        evidence = cls(
            id=evidence_id,
            run_id=run_id,
            evidence_type=EvidenceType.TEST_REPORT,
            status=status,
            test_report=report,
        )
        evidence.content_hash = evidence.compute_hash()
        return evidence

    @classmethod
    def from_security_scan(
        cls, evidence_id: str, run_id: RunId, scan: SecurityScan
    ) -> Evidence:
        """Create evidence from a security scan."""
        if scan.has_critical_issues:
            status = EvidenceStatus.FAILED
        elif scan.medium > 0 or scan.low > 0:
            status = EvidenceStatus.PARTIAL
        else:
            status = EvidenceStatus.PASSED
        evidence = cls(
            id=evidence_id,
            run_id=run_id,
            evidence_type=EvidenceType.SECURITY_SCAN,
            status=status,
            security_scan=scan,
        )
        evidence.content_hash = evidence.compute_hash()
        return evidence

    @classmethod
    def from_lint_report(
        cls, evidence_id: str, run_id: RunId, report: LintReport
    ) -> Evidence:
        """Create evidence from a lint report."""
        status = EvidenceStatus.PASSED if report.is_clean else EvidenceStatus.FAILED
        evidence = cls(
            id=evidence_id,
            run_id=run_id,
            evidence_type=EvidenceType.LINT_REPORT,
            status=status,
            lint_report=report,
        )
        evidence.content_hash = evidence.compute_hash()
        return evidence

    @classmethod
    def from_command(
        cls,
        evidence_id: str,
        run_id: RunId,
        output: CommandOutput,
        evidence_type: EvidenceType = EvidenceType.COMMAND_OUTPUT,
    ) -> Evidence:
        """Create evidence from command output."""
        status = EvidenceStatus.PASSED if output.succeeded else EvidenceStatus.FAILED
        evidence = cls(
            id=evidence_id,
            run_id=run_id,
            evidence_type=evidence_type,
            status=status,
            command_output=output,
        )
        evidence.content_hash = evidence.compute_hash()
        return evidence


class EvidenceCollection(BaseModel):
    """Collection of evidence for a run.

    Provides aggregate views and summary statistics.
    """

    run_id: RunId
    evidence_items: list[Evidence] = Field(default_factory=list)
    collected_at: Timestamp = Field(default_factory=Timestamp.now)

    def add(self, evidence: Evidence) -> None:
        """Add evidence to the collection."""
        self.evidence_items.append(evidence)

    def get_by_type(self, evidence_type: EvidenceType) -> list[Evidence]:
        """Get all evidence of a specific type."""
        return [e for e in self.evidence_items if e.evidence_type == evidence_type]

    @property
    def all_passed(self) -> bool:
        """Check if all evidence passed."""
        return all(e.status == EvidenceStatus.PASSED for e in self.evidence_items)

    @property
    def has_failures(self) -> bool:
        """Check if any evidence failed."""
        return any(e.status == EvidenceStatus.FAILED for e in self.evidence_items)

    @property
    def summary(self) -> dict[str, Any]:
        """Get summary of evidence collection."""
        by_type: dict[str, dict[str, int]] = {}
        for evidence in self.evidence_items:
            type_name = evidence.evidence_type.value
            if type_name not in by_type:
                by_type[type_name] = {"passed": 0, "failed": 0, "partial": 0}
            by_type[type_name][evidence.status.value] = (
                by_type[type_name].get(evidence.status.value, 0) + 1
            )

        return {
            "total": len(self.evidence_items),
            "passed": sum(1 for e in self.evidence_items if e.status == EvidenceStatus.PASSED),
            "failed": sum(1 for e in self.evidence_items if e.status == EvidenceStatus.FAILED),
            "by_type": by_type,
            "all_passed": self.all_passed,
        }
