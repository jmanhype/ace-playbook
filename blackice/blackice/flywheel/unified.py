"""Unified Flywheel for BLACKICE 3.0.

The flywheel drives end-to-end pipeline execution from vision to working software.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from blackice.instrumentation import get_logger
from blackice.primitives.errors import BlackiceError, ExecutionError

logger = get_logger(__name__)


class FlywheelPhase(str, Enum):
    """Phases of the flywheel execution."""

    INIT = "init"
    PLAN = "plan"
    IMPLEMENT = "implement"
    TEST = "test"
    VERIFY = "verify"
    FINALIZE = "finalize"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result of a flywheel phase."""

    phase: FlywheelPhase
    success: bool
    duration_seconds: float
    artifacts: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FlywheelResult:
    """Complete result of a flywheel run."""

    run_id: str
    vision: str
    success: bool
    total_duration: float
    phases: list[PhaseResult] = field(default_factory=list)
    workspace_path: Path | None = None
    artifacts: list[str] = field(default_factory=list)
    final_phase: FlywheelPhase = FlywheelPhase.INIT
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phase_count(self) -> int:
        return len(self.phases)

    @property
    def failed_phases(self) -> list[PhaseResult]:
        return [p for p in self.phases if not p.success]


@dataclass
class FlywheelConfig:
    """Configuration for the unified flywheel."""

    # Phase timeouts
    plan_timeout: float = 300.0
    implement_timeout: float = 1800.0  # 30 minutes
    test_timeout: float = 600.0  # 10 minutes
    verify_timeout: float = 300.0

    # Retry settings
    max_phase_retries: int = 3
    retry_delay: float = 5.0

    # Quality gates
    require_tests: bool = True
    require_verification: bool = True
    min_test_coverage: float = 80.0

    # Workspace settings
    workspace_root: Path | None = None
    cleanup_on_failure: bool = False

    # Feature flags
    enable_multi_agent: bool = False
    enable_memory: bool = False


class PhaseHandler:
    """Handler for a specific phase of the flywheel.

    Subclass this for each phase to implement specific logic.
    """

    def __init__(self, config: FlywheelConfig) -> None:
        self.config = config

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Execute the phase.

        Args:
            context: Shared context for the run
            previous_results: Results from previous phases

        Returns:
            PhaseResult with success status and outputs
        """
        raise NotImplementedError

    @property
    def phase(self) -> FlywheelPhase:
        """The phase this handler is for."""
        raise NotImplementedError


class InitPhaseHandler(PhaseHandler):
    """Handler for the initialization phase."""

    @property
    def phase(self) -> FlywheelPhase:
        return FlywheelPhase.INIT

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Initialize the run workspace and context."""
        start = time.monotonic()

        try:
            run_id = context.get("run_id", "run-001")
            vision = context.get("vision", "")

            # Create workspace directory
            workspace_root = self.config.workspace_root or Path.cwd() / ".blackice"
            workspace = workspace_root / run_id
            workspace.mkdir(parents=True, exist_ok=True)

            # Create standard directories
            (workspace / "artifacts").mkdir(exist_ok=True)
            (workspace / "logs").mkdir(exist_ok=True)
            (workspace / "events").mkdir(exist_ok=True)

            # Store vision
            (workspace / "vision.md").write_text(f"# Vision\n\n{vision}\n")

            logger.info(
                "Flywheel initialized",
                run_id=run_id,
                workspace=str(workspace),
            )

            return PhaseResult(
                phase=self.phase,
                success=True,
                duration_seconds=time.monotonic() - start,
                artifacts=[str(workspace / "vision.md")],
                outputs={"workspace": str(workspace)},
            )

        except Exception as e:
            return PhaseResult(
                phase=self.phase,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class PlanPhaseHandler(PhaseHandler):
    """Handler for the planning phase."""

    @property
    def phase(self) -> FlywheelPhase:
        return FlywheelPhase.PLAN

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Create a plan from the vision."""
        start = time.monotonic()

        try:
            vision = context.get("vision", "")
            workspace = Path(context.get("workspace", "."))

            # Generate plan (placeholder - would integrate with model provider)
            plan = {
                "vision": vision,
                "tasks": [],
                "dependencies": [],
                "estimates": {},
            }

            # Store plan
            plan_file = workspace / "plan.json"
            import json

            plan_file.write_text(json.dumps(plan, indent=2))

            logger.info(
                "Plan created",
                task_count=len(plan["tasks"]),
            )

            return PhaseResult(
                phase=self.phase,
                success=True,
                duration_seconds=time.monotonic() - start,
                artifacts=[str(plan_file)],
                outputs={"plan": plan},
            )

        except Exception as e:
            return PhaseResult(
                phase=self.phase,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class ImplementPhaseHandler(PhaseHandler):
    """Handler for the implementation phase."""

    @property
    def phase(self) -> FlywheelPhase:
        return FlywheelPhase.IMPLEMENT

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Implement the plan."""
        start = time.monotonic()

        try:
            workspace = Path(context.get("workspace", "."))
            plan = context.get("plan", {})

            # Implementation would happen here
            # For now, just create a placeholder

            src_dir = workspace / "src"
            src_dir.mkdir(exist_ok=True)

            # Create placeholder file
            (src_dir / "__init__.py").write_text("# Generated by BLACKICE\n")

            logger.info("Implementation complete")

            return PhaseResult(
                phase=self.phase,
                success=True,
                duration_seconds=time.monotonic() - start,
                artifacts=[str(src_dir / "__init__.py")],
                outputs={"src_dir": str(src_dir)},
            )

        except Exception as e:
            return PhaseResult(
                phase=self.phase,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class TestPhaseHandler(PhaseHandler):
    """Handler for the testing phase."""

    @property
    def phase(self) -> FlywheelPhase:
        return FlywheelPhase.TEST

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Run tests on the implementation."""
        start = time.monotonic()

        try:
            workspace = Path(context.get("workspace", "."))

            # Create tests directory
            tests_dir = workspace / "tests"
            tests_dir.mkdir(exist_ok=True)

            # Create placeholder test
            test_file = tests_dir / "test_placeholder.py"
            test_file.write_text(
                'def test_placeholder():\n    """Placeholder test."""\n    assert True\n'
            )

            # Would integrate with evaluator here
            test_results = {
                "passed": 1,
                "failed": 0,
                "coverage": 100.0,
            }

            logger.info(
                "Tests complete",
                passed=test_results["passed"],
                failed=test_results["failed"],
            )

            return PhaseResult(
                phase=self.phase,
                success=test_results["failed"] == 0,
                duration_seconds=time.monotonic() - start,
                artifacts=[str(test_file)],
                outputs={"test_results": test_results},
            )

        except Exception as e:
            return PhaseResult(
                phase=self.phase,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class VerifyPhaseHandler(PhaseHandler):
    """Handler for the verification phase."""

    @property
    def phase(self) -> FlywheelPhase:
        return FlywheelPhase.VERIFY

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Verify the implementation meets the vision."""
        start = time.monotonic()

        try:
            # Check that previous phases succeeded
            test_result = next(
                (r for r in previous_results if r.phase == FlywheelPhase.TEST),
                None,
            )

            if test_result and not test_result.success:
                return PhaseResult(
                    phase=self.phase,
                    success=False,
                    duration_seconds=time.monotonic() - start,
                    error="Test phase failed, cannot verify",
                )

            # Verification checks would happen here
            verification = {
                "tests_passed": True,
                "coverage_met": True,
                "artifacts_present": True,
            }

            logger.info("Verification complete", **verification)

            return PhaseResult(
                phase=self.phase,
                success=all(verification.values()),
                duration_seconds=time.monotonic() - start,
                outputs={"verification": verification},
            )

        except Exception as e:
            return PhaseResult(
                phase=self.phase,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class FinalizePhaseHandler(PhaseHandler):
    """Handler for the finalization phase."""

    @property
    def phase(self) -> FlywheelPhase:
        return FlywheelPhase.FINALIZE

    async def execute(
        self,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Finalize the run and generate outputs."""
        start = time.monotonic()

        try:
            workspace = Path(context.get("workspace", "."))
            run_id = context.get("run_id", "run-001")

            # Generate summary
            summary = {
                "run_id": run_id,
                "phases": len(previous_results),
                "success": all(r.success for r in previous_results),
                "artifacts": [],
            }

            for result in previous_results:
                summary["artifacts"].extend(result.artifacts)

            # Write summary
            import json

            summary_file = workspace / "summary.json"
            summary_file.write_text(json.dumps(summary, indent=2))

            logger.info("Finalization complete", artifact_count=len(summary["artifacts"]))

            return PhaseResult(
                phase=self.phase,
                success=True,
                duration_seconds=time.monotonic() - start,
                artifacts=[str(summary_file)],
                outputs={"summary": summary},
            )

        except Exception as e:
            return PhaseResult(
                phase=self.phase,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class UnifiedFlywheel:
    """The unified flywheel for end-to-end pipeline execution.

    Orchestrates the complete flow from vision to working software:
    INIT -> PLAN -> IMPLEMENT -> TEST -> VERIFY -> FINALIZE -> COMPLETE

    Example:
        ```python
        flywheel = UnifiedFlywheel()
        result = await flywheel.run(
            run_id="run-001",
            vision="Create a REST API for user management",
        )
        if result.success:
            print(f"Success! Workspace: {result.workspace_path}")
        ```
    """

    def __init__(
        self,
        config: FlywheelConfig | None = None,
        phase_handlers: dict[FlywheelPhase, PhaseHandler] | None = None,
    ) -> None:
        """Initialize the flywheel.

        Args:
            config: Flywheel configuration
            phase_handlers: Custom phase handlers
        """
        self.config = config or FlywheelConfig()

        # Default phase handlers
        self.handlers: dict[FlywheelPhase, PhaseHandler] = {
            FlywheelPhase.INIT: InitPhaseHandler(self.config),
            FlywheelPhase.PLAN: PlanPhaseHandler(self.config),
            FlywheelPhase.IMPLEMENT: ImplementPhaseHandler(self.config),
            FlywheelPhase.TEST: TestPhaseHandler(self.config),
            FlywheelPhase.VERIFY: VerifyPhaseHandler(self.config),
            FlywheelPhase.FINALIZE: FinalizePhaseHandler(self.config),
        }

        # Override with custom handlers
        if phase_handlers:
            self.handlers.update(phase_handlers)

        # Execution order
        self.phase_order = [
            FlywheelPhase.INIT,
            FlywheelPhase.PLAN,
            FlywheelPhase.IMPLEMENT,
            FlywheelPhase.TEST,
            FlywheelPhase.VERIFY,
            FlywheelPhase.FINALIZE,
        ]

    async def run(
        self,
        run_id: str,
        vision: str,
        context: dict[str, Any] | None = None,
    ) -> FlywheelResult:
        """Run the complete flywheel.

        Args:
            run_id: Unique identifier for this run
            vision: The vision/description to implement
            context: Additional context for the run

        Returns:
            FlywheelResult with success status and all outputs
        """
        start = time.monotonic()
        ctx = context or {}
        ctx.update({"run_id": run_id, "vision": vision})

        phases: list[PhaseResult] = []
        workspace_path: Path | None = None
        final_phase = FlywheelPhase.INIT

        logger.info(
            "Flywheel starting",
            run_id=run_id,
            vision_length=len(vision),
        )

        for phase in self.phase_order:
            handler = self.handlers.get(phase)
            if not handler:
                logger.warning("No handler for phase", phase=phase.value)
                continue

            final_phase = phase
            logger.info("Starting phase", phase=phase.value)

            # Execute with retry
            result = await self._execute_with_retry(handler, ctx, phases)
            phases.append(result)

            # Update context with outputs
            ctx.update(result.outputs)

            # Track workspace
            if "workspace" in result.outputs:
                workspace_path = Path(result.outputs["workspace"])

            # Stop on failure if required
            if not result.success:
                logger.error(
                    "Phase failed",
                    phase=phase.value,
                    error=result.error,
                )

                # Check if this phase is required
                if self._is_phase_required(phase):
                    final_phase = FlywheelPhase.FAILED
                    break

        # Determine overall success
        success = all(p.success for p in phases)
        if success:
            final_phase = FlywheelPhase.COMPLETE

        # Collect all artifacts
        artifacts = []
        for result in phases:
            artifacts.extend(result.artifacts)

        total_duration = time.monotonic() - start

        logger.info(
            "Flywheel complete",
            run_id=run_id,
            success=success,
            phases=len(phases),
            duration=total_duration,
        )

        return FlywheelResult(
            run_id=run_id,
            vision=vision,
            success=success,
            total_duration=total_duration,
            phases=phases,
            workspace_path=workspace_path,
            artifacts=artifacts,
            final_phase=final_phase,
            metadata=ctx,
        )

    async def _execute_with_retry(
        self,
        handler: PhaseHandler,
        context: dict[str, Any],
        previous_results: list[PhaseResult],
    ) -> PhaseResult:
        """Execute a phase with retry logic."""
        last_result: PhaseResult | None = None

        for attempt in range(self.config.max_phase_retries):
            result = await handler.execute(context, previous_results)
            last_result = result

            if result.success:
                return result

            if attempt < self.config.max_phase_retries - 1:
                logger.info(
                    "Phase failed, retrying",
                    phase=handler.phase.value,
                    attempt=attempt + 1,
                    delay=self.config.retry_delay,
                )
                await asyncio.sleep(self.config.retry_delay)

        return last_result or PhaseResult(
            phase=handler.phase,
            success=False,
            duration_seconds=0.0,
            error="No result after retries",
        )

    def _is_phase_required(self, phase: FlywheelPhase) -> bool:
        """Check if a phase is required to succeed."""
        # All phases before FINALIZE are required by default
        required = {
            FlywheelPhase.INIT,
            FlywheelPhase.PLAN,
            FlywheelPhase.IMPLEMENT,
        }

        if self.config.require_tests:
            required.add(FlywheelPhase.TEST)

        if self.config.require_verification:
            required.add(FlywheelPhase.VERIFY)

        return phase in required
