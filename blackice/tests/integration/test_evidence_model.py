"""Integration tests for Evidence model (IT-007).

Tests the evidence collection and validation workflow including:
- Test report evidence
- Security scan evidence
- Lint report evidence
- Command output evidence
- Evidence collections and summaries
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from blackice.primitives.types import RunId
from blackice.schemas.evidence import (
    CommandOutput,
    Evidence,
    EvidenceCollection,
    EvidenceStatus,
    EvidenceType,
    LintIssue,
    LintReport,
    SecurityFinding,
    SecurityScan,
    TestReport,
    TestResult,
)


def make_run_id() -> RunId:
    """Create a valid RunId (UUID)."""
    return RunId(uuid4())


class TestTestReport:
    """Test TestReport evidence creation."""

    def test_create_test_report(self) -> None:
        """Should create test report with basic stats."""
        report = TestReport(
            framework="pytest",
            total_tests=100,
            passed=95,
            failed=3,
            skipped=2,
            errors=0,
            duration_seconds=12.5,
            coverage_percent=85.0,
        )

        assert report.framework == "pytest"
        assert report.total_tests == 100
        assert report.passed == 95
        assert report.failed == 3

    def test_pass_rate_calculation(self) -> None:
        """Pass rate should be calculated correctly."""
        report = TestReport(
            framework="pytest",
            total_tests=100,
            passed=80,
            failed=15,
            skipped=5,
            errors=0,
            duration_seconds=10.0,
        )

        assert report.pass_rate == 80.0

    def test_pass_rate_with_zero_tests(self) -> None:
        """Pass rate should be 0 when no tests."""
        report = TestReport(
            framework="pytest",
            total_tests=0,
            passed=0,
            failed=0,
            skipped=0,
            errors=0,
            duration_seconds=0.0,
        )

        assert report.pass_rate == 0.0

    def test_test_results_detail(self) -> None:
        """Should track individual test results."""
        report = TestReport(
            framework="pytest",
            total_tests=2,
            passed=1,
            failed=1,
            skipped=0,
            errors=0,
            duration_seconds=1.0,
            test_results=[
                TestResult(
                    name="test_success",
                    status="passed",
                    duration_ms=100.0,
                ),
                TestResult(
                    name="test_failure",
                    status="failed",
                    duration_ms=50.0,
                    error_message="AssertionError: expected 1, got 2",
                    file_path="tests/test_example.py",
                    line_number=42,
                ),
            ],
        )

        assert len(report.test_results) == 2
        assert report.test_results[1].error_message is not None


class TestSecurityScan:
    """Test SecurityScan evidence creation."""

    def test_create_security_scan(self) -> None:
        """Should create security scan with findings."""
        scan = SecurityScan(
            scanner="bandit",
            scan_type="sast",
            total_findings=5,
            critical=0,
            high=1,
            medium=2,
            low=2,
            info=0,
            scan_duration_seconds=5.0,
        )

        assert scan.scanner == "bandit"
        assert scan.total_findings == 5
        assert scan.high == 1

    def test_has_critical_issues(self) -> None:
        """Should detect critical/high issues."""
        scan_with_critical = SecurityScan(
            scanner="semgrep",
            total_findings=1,
            critical=1,
            scan_duration_seconds=3.0,
        )
        scan_clean = SecurityScan(
            scanner="semgrep",
            total_findings=2,
            low=2,
            scan_duration_seconds=3.0,
        )

        assert scan_with_critical.has_critical_issues is True
        assert scan_clean.has_critical_issues is False

    def test_security_findings_detail(self) -> None:
        """Should track individual security findings."""
        scan = SecurityScan(
            scanner="bandit",
            total_findings=1,
            high=1,
            scan_duration_seconds=2.0,
            findings=[
                SecurityFinding(
                    severity="high",
                    category="injection",
                    message="SQL injection vulnerability",
                    file_path="src/db.py",
                    line_number=25,
                    cwe_id="CWE-89",
                    remediation="Use parameterized queries",
                ),
            ],
        )

        assert len(scan.findings) == 1
        assert scan.findings[0].cwe_id == "CWE-89"


class TestLintReport:
    """Test LintReport evidence creation."""

    def test_create_lint_report(self) -> None:
        """Should create lint report with issues."""
        report = LintReport(
            linter="ruff",
            total_issues=10,
            errors=2,
            warnings=5,
            info=3,
            files_checked=50,
        )

        assert report.linter == "ruff"
        assert report.total_issues == 10
        assert report.errors == 2

    def test_is_clean(self) -> None:
        """Should detect clean vs error reports."""
        clean_report = LintReport(
            linter="ruff",
            total_issues=3,
            errors=0,
            warnings=3,
            files_checked=10,
        )
        error_report = LintReport(
            linter="ruff",
            total_issues=5,
            errors=2,
            warnings=3,
            files_checked=10,
        )

        assert clean_report.is_clean is True
        assert error_report.is_clean is False

    def test_lint_issues_detail(self) -> None:
        """Should track individual lint issues."""
        report = LintReport(
            linter="eslint",
            total_issues=1,
            errors=1,
            files_checked=5,
            issues=[
                LintIssue(
                    rule="no-unused-vars",
                    severity="error",
                    message="'foo' is defined but never used",
                    file_path="src/app.js",
                    line_number=10,
                    column=5,
                ),
            ],
        )

        assert len(report.issues) == 1
        assert report.issues[0].rule == "no-unused-vars"


class TestCommandOutput:
    """Test CommandOutput evidence creation."""

    def test_create_command_output(self) -> None:
        """Should create command output evidence."""
        output = CommandOutput(
            command="pytest tests/",
            exit_code=0,
            stdout="All tests passed",
            stderr="",
            duration_seconds=10.5,
            working_directory="/project",
        )

        assert output.command == "pytest tests/"
        assert output.exit_code == 0
        assert output.succeeded is True

    def test_failed_command(self) -> None:
        """Should detect failed commands."""
        output = CommandOutput(
            command="npm test",
            exit_code=1,
            stdout="",
            stderr="Test failed: assertion error",
            duration_seconds=5.0,
        )

        assert output.succeeded is False


class TestEvidence:
    """Test Evidence model creation and factory methods."""

    def test_create_evidence_from_test_report(self) -> None:
        """Should create evidence from test report."""
        run_id = make_run_id()
        report = TestReport(
            framework="pytest",
            total_tests=10,
            passed=10,
            failed=0,
            skipped=0,
            errors=0,
            duration_seconds=5.0,
        )

        evidence = Evidence.from_test_report("ev-001", run_id, report)

        assert evidence.evidence_type == EvidenceType.TEST_REPORT
        assert evidence.status == EvidenceStatus.PASSED
        assert evidence.test_report == report
        assert evidence.content_hash is not None

    def test_evidence_status_from_failed_tests(self) -> None:
        """Evidence status should be FAILED when tests fail."""
        run_id = make_run_id()
        report = TestReport(
            framework="pytest",
            total_tests=10,
            passed=8,
            failed=2,
            skipped=0,
            errors=0,
            duration_seconds=5.0,
        )

        evidence = Evidence.from_test_report("ev-002", run_id, report)

        assert evidence.status == EvidenceStatus.FAILED

    def test_create_evidence_from_security_scan(self) -> None:
        """Should create evidence from security scan."""
        run_id = make_run_id()
        scan = SecurityScan(
            scanner="bandit",
            total_findings=3,
            medium=3,
            scan_duration_seconds=2.0,
        )

        evidence = Evidence.from_security_scan("ev-003", run_id, scan)

        assert evidence.evidence_type == EvidenceType.SECURITY_SCAN
        assert evidence.status == EvidenceStatus.PARTIAL
        assert evidence.security_scan == scan

    def test_create_evidence_from_lint_report(self) -> None:
        """Should create evidence from lint report."""
        run_id = make_run_id()
        report = LintReport(
            linter="ruff",
            total_issues=0,
            errors=0,
            files_checked=20,
        )

        evidence = Evidence.from_lint_report("ev-004", run_id, report)

        assert evidence.evidence_type == EvidenceType.LINT_REPORT
        assert evidence.status == EvidenceStatus.PASSED
        assert evidence.lint_report == report

    def test_create_evidence_from_command(self) -> None:
        """Should create evidence from command output."""
        run_id = make_run_id()
        output = CommandOutput(
            command="make build",
            exit_code=0,
            stdout="Build successful",
            stderr="",
            duration_seconds=30.0,
        )

        evidence = Evidence.from_command(
            "ev-005", run_id, output, EvidenceType.BUILD_LOG
        )

        assert evidence.evidence_type == EvidenceType.BUILD_LOG
        assert evidence.status == EvidenceStatus.PASSED
        assert evidence.command_output == output

    def test_evidence_hash_is_deterministic(self) -> None:
        """Same evidence should produce same hash."""
        run_id = make_run_id()
        report = TestReport(
            framework="pytest",
            total_tests=1,
            passed=1,
            failed=0,
            skipped=0,
            errors=0,
            duration_seconds=1.0,
        )

        evidence = Evidence(
            id="ev-hash-test",
            run_id=run_id,
            evidence_type=EvidenceType.TEST_REPORT,
            status=EvidenceStatus.PASSED,
            test_report=report,
        )

        hash1 = evidence.compute_hash()
        hash2 = evidence.compute_hash()

        assert hash1.value == hash2.value


class TestEvidenceCollection:
    """Test EvidenceCollection aggregation."""

    def test_create_evidence_collection(self) -> None:
        """Should create evidence collection."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        assert collection.run_id == run_id
        assert len(collection.evidence_items) == 0

    def test_add_evidence_to_collection(self) -> None:
        """Should add evidence to collection."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        evidence = Evidence.from_test_report(
            "ev-001",
            run_id,
            TestReport(
                framework="pytest",
                total_tests=1,
                passed=1,
                failed=0,
                skipped=0,
                errors=0,
                duration_seconds=1.0,
            ),
        )

        collection.add(evidence)

        assert len(collection.evidence_items) == 1

    def test_get_evidence_by_type(self) -> None:
        """Should filter evidence by type."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        # Add test report
        collection.add(
            Evidence.from_test_report(
                "ev-001",
                run_id,
                TestReport(
                    framework="pytest",
                    total_tests=1,
                    passed=1,
                    failed=0,
                    skipped=0,
                    errors=0,
                    duration_seconds=1.0,
                ),
            )
        )

        # Add security scan
        collection.add(
            Evidence.from_security_scan(
                "ev-002",
                run_id,
                SecurityScan(
                    scanner="bandit",
                    total_findings=0,
                    scan_duration_seconds=1.0,
                ),
            )
        )

        test_evidence = collection.get_by_type(EvidenceType.TEST_REPORT)
        security_evidence = collection.get_by_type(EvidenceType.SECURITY_SCAN)

        assert len(test_evidence) == 1
        assert len(security_evidence) == 1

    def test_all_passed(self) -> None:
        """Should detect when all evidence passed."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        collection.add(
            Evidence.from_test_report(
                "ev-001",
                run_id,
                TestReport(
                    framework="pytest",
                    total_tests=10,
                    passed=10,
                    failed=0,
                    skipped=0,
                    errors=0,
                    duration_seconds=5.0,
                ),
            )
        )
        collection.add(
            Evidence.from_lint_report(
                "ev-002",
                run_id,
                LintReport(
                    linter="ruff",
                    total_issues=0,
                    errors=0,
                    files_checked=10,
                ),
            )
        )

        assert collection.all_passed is True
        assert collection.has_failures is False

    def test_has_failures(self) -> None:
        """Should detect when any evidence failed."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        collection.add(
            Evidence.from_test_report(
                "ev-001",
                run_id,
                TestReport(
                    framework="pytest",
                    total_tests=10,
                    passed=10,
                    failed=0,
                    skipped=0,
                    errors=0,
                    duration_seconds=5.0,
                ),
            )
        )
        collection.add(
            Evidence.from_security_scan(
                "ev-002",
                run_id,
                SecurityScan(
                    scanner="bandit",
                    total_findings=1,
                    critical=1,
                    scan_duration_seconds=2.0,
                ),
            )
        )

        assert collection.all_passed is False
        assert collection.has_failures is True

    def test_collection_summary(self) -> None:
        """Should provide summary of evidence."""
        run_id = make_run_id()
        collection = EvidenceCollection(run_id=run_id)

        collection.add(
            Evidence.from_test_report(
                "ev-001",
                run_id,
                TestReport(
                    framework="pytest",
                    total_tests=10,
                    passed=10,
                    failed=0,
                    skipped=0,
                    errors=0,
                    duration_seconds=5.0,
                ),
            )
        )
        collection.add(
            Evidence.from_lint_report(
                "ev-002",
                run_id,
                LintReport(
                    linter="ruff",
                    total_issues=5,
                    errors=2,
                    files_checked=10,
                ),
            )
        )

        summary = collection.summary

        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["all_passed"] is False
        assert "test_report" in summary["by_type"]
        assert "lint_report" in summary["by_type"]
