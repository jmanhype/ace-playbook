"""Evaluator for BLACKICE 3.0.

Provides test verification and automated repair capabilities.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from blackice.instrumentation import get_logger
from blackice.primitives.errors import ValidationError

logger = get_logger(__name__)


class TestStatus(str, Enum):
    """Status of a test execution."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class VerificationLevel(str, Enum):
    """Level of verification strictness."""

    MINIMAL = "minimal"  # Just check exit code
    STANDARD = "standard"  # Check output patterns
    STRICT = "strict"  # Full assertion validation
    PARANOID = "paranoid"  # Include static analysis


@dataclass
class TestResult:
    """Result of a single test execution."""

    name: str
    status: TestStatus
    duration_seconds: float
    output: str = ""
    error_message: str | None = None
    error_traceback: str | None = None
    assertions_passed: int = 0
    assertions_failed: int = 0
    line_coverage: float | None = None
    branch_coverage: float | None = None


@dataclass
class TestSuiteResult:
    """Result of running a test suite."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    duration_seconds: float
    results: list[TestResult] = field(default_factory=list)
    coverage: float | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_tests == 0:
            return 100.0
        return (self.passed / self.total_tests) * 100

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0 and self.errors == 0


@dataclass
class RepairSuggestion:
    """A suggested repair for a failing test."""

    test_name: str
    error_type: str
    file_path: str | None
    line_number: int | None
    suggestion: str
    confidence: float  # 0.0 to 1.0
    code_diff: str | None = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    passed: bool
    test_results: TestSuiteResult
    repair_suggestions: list[RepairSuggestion] = field(default_factory=list)
    verification_level: VerificationLevel = VerificationLevel.STANDARD
    notes: list[str] = field(default_factory=list)


class TestRunner:
    """Runs tests and captures results.

    Abstract base for different test frameworks.
    """

    async def run(
        self,
        target: str | Path,
        *,
        timeout: float = 300.0,
        coverage: bool = True,
        verbose: bool = False,
    ) -> TestSuiteResult:
        """Run tests against a target.

        Args:
            target: Path to test file/directory or test pattern
            timeout: Maximum execution time
            coverage: Whether to collect coverage
            verbose: Enable verbose output

        Returns:
            TestSuiteResult with all test results
        """
        raise NotImplementedError


class PytestRunner(TestRunner):
    """Test runner for pytest."""

    def __init__(
        self,
        python_path: str = "python",
        pytest_args: list[str] | None = None,
    ) -> None:
        self.python_path = python_path
        self.pytest_args = pytest_args or []

    async def run(
        self,
        target: str | Path,
        *,
        timeout: float = 300.0,
        coverage: bool = True,
        verbose: bool = False,
    ) -> TestSuiteResult:
        """Run pytest tests."""
        args = [self.python_path, "-m", "pytest"]

        if verbose:
            args.append("-v")

        if coverage:
            args.extend(["--cov", "--cov-report=term"])

        args.extend(self.pytest_args)
        args.append(str(target))

        start = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            stdout_bytes, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            output = stdout_bytes.decode("utf-8", errors="replace")
            duration = asyncio.get_event_loop().time() - start

            return self._parse_pytest_output(output, duration)

        except asyncio.TimeoutError:
            return TestSuiteResult(
                suite_name=str(target),
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration_seconds=timeout,
                results=[
                    TestResult(
                        name="test_suite",
                        status=TestStatus.TIMEOUT,
                        duration_seconds=timeout,
                        error_message="Test suite timed out",
                    )
                ],
            )

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start
            return TestSuiteResult(
                suite_name=str(target),
                total_tests=0,
                passed=0,
                failed=0,
                errors=1,
                skipped=0,
                duration_seconds=duration,
                results=[
                    TestResult(
                        name="test_suite",
                        status=TestStatus.ERROR,
                        duration_seconds=duration,
                        error_message=str(e),
                    )
                ],
            )

    def _parse_pytest_output(self, output: str, duration: float) -> TestSuiteResult:
        """Parse pytest output to extract results."""
        results: list[TestResult] = []

        # Parse test results from output
        # Pattern: test_file.py::test_name PASSED/FAILED/ERROR
        test_pattern = re.compile(
            r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)",
            re.MULTILINE,
        )

        for match in test_pattern.finditer(output):
            test_name = match.group(1)
            status_str = match.group(2)
            status = TestStatus(status_str.lower())

            results.append(
                TestResult(
                    name=test_name,
                    status=status,
                    duration_seconds=0.0,  # Individual timing not always available
                )
            )

        # Parse summary line
        # Pattern: X passed, Y failed, Z errors in N.NNs
        summary_pattern = re.compile(
            r"(\d+)\s+passed.*?(?:(\d+)\s+failed)?.*?(?:(\d+)\s+error)?.*?(?:(\d+)\s+skipped)?.*?in\s+([\d.]+)s",
            re.IGNORECASE,
        )

        passed = failed = errors = skipped = 0
        summary_match = summary_pattern.search(output)

        if summary_match:
            passed = int(summary_match.group(1)) if summary_match.group(1) else 0
            failed = int(summary_match.group(2)) if summary_match.group(2) else 0
            errors = int(summary_match.group(3)) if summary_match.group(3) else 0
            skipped = int(summary_match.group(4)) if summary_match.group(4) else 0
        else:
            # Count from individual results if no summary
            passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            failed = sum(1 for r in results if r.status == TestStatus.FAILED)
            errors = sum(1 for r in results if r.status == TestStatus.ERROR)
            skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

        # Parse coverage if available
        coverage = None
        coverage_pattern = re.compile(r"TOTAL\s+\d+\s+\d+\s+(\d+)%")
        coverage_match = coverage_pattern.search(output)
        if coverage_match:
            coverage = float(coverage_match.group(1))

        return TestSuiteResult(
            suite_name="pytest",
            total_tests=passed + failed + errors + skipped,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_seconds=duration,
            results=results,
            coverage=coverage,
        )


class RepairAnalyzer:
    """Analyzes test failures and suggests repairs."""

    def __init__(
        self,
        model_provider: Any | None = None,  # ModelProvider for LLM-based analysis
    ) -> None:
        self.model_provider = model_provider

    async def analyze_failure(
        self,
        result: TestResult,
        source_code: str | None = None,
    ) -> RepairSuggestion | None:
        """Analyze a test failure and suggest a repair.

        Args:
            result: The failed test result
            source_code: Optional source code for context

        Returns:
            RepairSuggestion if a repair can be suggested
        """
        if result.status not in (TestStatus.FAILED, TestStatus.ERROR):
            return None

        # Extract file and line from error
        file_path, line_number = self._extract_location(result)

        # Generate suggestion based on error type
        suggestion = self._generate_suggestion(result)

        if not suggestion:
            return None

        return RepairSuggestion(
            test_name=result.name,
            error_type=self._classify_error(result),
            file_path=file_path,
            line_number=line_number,
            suggestion=suggestion,
            confidence=0.5,  # Default confidence
        )

    def _extract_location(
        self,
        result: TestResult,
    ) -> tuple[str | None, int | None]:
        """Extract file path and line number from error."""
        if not result.error_traceback:
            return None, None

        # Pattern: File "path/to/file.py", line N
        pattern = re.compile(r'File "([^"]+)", line (\d+)')
        matches = pattern.findall(result.error_traceback)

        if matches:
            # Get the last (most specific) location
            file_path, line_str = matches[-1]
            return file_path, int(line_str)

        return None, None

    def _classify_error(self, result: TestResult) -> str:
        """Classify the type of error."""
        error = result.error_message or ""

        if "AssertionError" in error:
            return "assertion"
        elif "TypeError" in error:
            return "type_error"
        elif "AttributeError" in error:
            return "attribute_error"
        elif "ImportError" in error or "ModuleNotFoundError" in error:
            return "import_error"
        elif "SyntaxError" in error:
            return "syntax_error"
        elif "ValueError" in error:
            return "value_error"
        elif "KeyError" in error:
            return "key_error"
        elif "IndexError" in error:
            return "index_error"
        elif "TimeoutError" in error:
            return "timeout"
        else:
            return "unknown"

    def _generate_suggestion(self, result: TestResult) -> str | None:
        """Generate a repair suggestion based on error type."""
        error_type = self._classify_error(result)
        error = result.error_message or ""

        suggestions = {
            "assertion": f"Review the assertion: {error[:100]}. Check expected vs actual values.",
            "type_error": "Verify the types of arguments being passed match the expected types.",
            "attribute_error": "Check that the object has the expected attribute or method.",
            "import_error": "Verify the module is installed and the import path is correct.",
            "syntax_error": "Fix the syntax error in the indicated file and line.",
            "value_error": "Verify the value being passed is valid for the operation.",
            "key_error": "Check that the key exists in the dictionary before accessing.",
            "index_error": "Verify the index is within the valid range for the sequence.",
            "timeout": "Consider increasing the timeout or optimizing the operation.",
        }

        return suggestions.get(error_type)


class Evaluator:
    """Main evaluator for test verification and repair.

    Orchestrates test running, result analysis, and repair suggestions.
    """

    def __init__(
        self,
        test_runner: TestRunner | None = None,
        repair_analyzer: RepairAnalyzer | None = None,
        verification_level: VerificationLevel = VerificationLevel.STANDARD,
    ) -> None:
        """Initialize the evaluator.

        Args:
            test_runner: The test runner to use
            repair_analyzer: The repair analyzer for suggestions
            verification_level: Level of verification strictness
        """
        self.test_runner = test_runner or PytestRunner()
        self.repair_analyzer = repair_analyzer or RepairAnalyzer()
        self.verification_level = verification_level

    async def evaluate(
        self,
        target: str | Path,
        *,
        timeout: float = 300.0,
        coverage: bool = True,
        min_coverage: float | None = None,
        suggest_repairs: bool = True,
    ) -> EvaluationResult:
        """Evaluate a target by running tests and analyzing results.

        Args:
            target: Path to test file/directory
            timeout: Maximum execution time
            coverage: Whether to collect coverage
            min_coverage: Minimum required coverage percentage
            suggest_repairs: Whether to generate repair suggestions

        Returns:
            EvaluationResult with test results and suggestions
        """
        logger.info(
            "Starting evaluation",
            target=str(target),
            verification_level=self.verification_level.value,
        )

        # Run tests
        test_results = await self.test_runner.run(
            target,
            timeout=timeout,
            coverage=coverage,
            verbose=self.verification_level in (VerificationLevel.STRICT, VerificationLevel.PARANOID),
        )

        notes: list[str] = []
        repair_suggestions: list[RepairSuggestion] = []

        # Check coverage if required
        passed = test_results.all_passed
        if min_coverage is not None and test_results.coverage is not None:
            if test_results.coverage < min_coverage:
                passed = False
                notes.append(
                    f"Coverage {test_results.coverage:.1f}% below minimum {min_coverage}%"
                )

        # Generate repair suggestions for failures
        if suggest_repairs and not test_results.all_passed:
            for result in test_results.results:
                if result.status in (TestStatus.FAILED, TestStatus.ERROR):
                    suggestion = await self.repair_analyzer.analyze_failure(result)
                    if suggestion:
                        repair_suggestions.append(suggestion)

        logger.info(
            "Evaluation complete",
            passed=passed,
            total_tests=test_results.total_tests,
            failures=test_results.failed,
            errors=test_results.errors,
            suggestions=len(repair_suggestions),
        )

        return EvaluationResult(
            passed=passed,
            test_results=test_results,
            repair_suggestions=repair_suggestions,
            verification_level=self.verification_level,
            notes=notes,
        )

    async def verify_and_repair(
        self,
        target: str | Path,
        max_repair_attempts: int = 3,
        repair_callback: Any | None = None,
    ) -> EvaluationResult:
        """Verify tests and attempt repairs if they fail.

        This integrates with the Ralph loop for iterative repair.

        Args:
            target: Path to test file/directory
            max_repair_attempts: Maximum repair attempts
            repair_callback: Async function to apply repairs

        Returns:
            Final EvaluationResult after all attempts
        """
        from blackice.reflexion.ralph_loop import RalphLoop, LoopConfig

        async def verification_operation(attempt: int = 1) -> EvaluationResult:
            result = await self.evaluate(target, suggest_repairs=True)
            if not result.passed:
                # If we have a repair callback and suggestions, apply repairs
                if repair_callback and result.repair_suggestions:
                    for suggestion in result.repair_suggestions:
                        await repair_callback(suggestion)

                raise ValidationError(
                    f"Tests failed: {result.test_results.failed} failures, "
                    f"{result.test_results.errors} errors",
                    context={
                        "passed": result.test_results.passed,
                        "failed": result.test_results.failed,
                    },
                )
            return result

        loop = RalphLoop(
            operation=verification_operation,
            config=LoopConfig(
                max_attempts=max_repair_attempts,
                initial_delay=1.0,
            ),
        )

        loop_result = await loop.run(initial_params={"attempt": 1})

        if loop_result.success and loop_result.result:
            return loop_result.result

        # Return the last evaluation result even if failed
        return await self.evaluate(target, suggest_repairs=True)
