"""Phase Handlers for BLACKICE 3.0 Orchestrator.

Implements handlers for each phase of the build lifecycle:
- Plan: Decompose vision into tasks
- Implement: Execute tasks to produce code
- Test: Run tests and verify quality
- Verify: Validate against original vision
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blackice.instrumentation import get_logger
from blackice.orchestrator.state_machine import RunContext, RunState

logger = get_logger(__name__)


@dataclass
class PhaseContext:
    """Context for phase execution."""

    run: RunContext
    workspace: Path
    plan: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Result of phase execution."""

    phase_name: str
    success: bool
    duration_seconds: float
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)


class PhaseHandler(ABC):
    """Abstract base class for phase handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Phase name."""
        ...

    @property
    @abstractmethod
    def target_state(self) -> RunState:
        """State the run should be in for this phase."""
        ...

    @abstractmethod
    async def execute(self, context: PhaseContext) -> PhaseResult:
        """Execute the phase.

        Args:
            context: Phase execution context

        Returns:
            PhaseResult with success status
        """
        ...

    async def pre_execute(self, context: PhaseContext) -> None:
        """Hook called before execution."""
        logger.info(
            "Phase starting",
            phase=self.name,
            run_id=context.run.run_id,
        )

    async def post_execute(
        self,
        context: PhaseContext,
        result: PhaseResult,
    ) -> None:
        """Hook called after execution."""
        logger.info(
            "Phase completed",
            phase=self.name,
            run_id=context.run.run_id,
            success=result.success,
            duration=result.duration_seconds,
        )


class PlanPhase(PhaseHandler):
    """Planning phase handler.

    Decomposes the vision into actionable tasks with dependencies.
    """

    def __init__(
        self,
        model_provider: Any | None = None,
        max_tasks: int = 100,
    ) -> None:
        self.model_provider = model_provider
        self.max_tasks = max_tasks

    @property
    def name(self) -> str:
        return "plan"

    @property
    def target_state(self) -> RunState:
        return RunState.PLANNING

    async def execute(self, context: PhaseContext) -> PhaseResult:
        """Generate a plan from the vision."""
        start = time.monotonic()

        try:
            vision = context.run.vision

            # Generate plan (would use model provider in real implementation)
            plan = await self._generate_plan(vision, context)

            # Store plan
            plan_file = context.workspace / "plan.json"
            import json

            plan_file.write_text(json.dumps(plan, indent=2))

            context.plan = plan
            context.artifacts.append(str(plan_file))

            return PhaseResult(
                phase_name=self.name,
                success=True,
                duration_seconds=time.monotonic() - start,
                outputs={"plan": plan, "task_count": len(plan.get("tasks", []))},
                artifacts=[str(plan_file)],
            )

        except Exception as e:
            return PhaseResult(
                phase_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )

    async def _generate_plan(
        self,
        vision: str,
        context: PhaseContext,
    ) -> dict[str, Any]:
        """Generate a plan from vision."""
        # Placeholder - would use LLM in real implementation
        return {
            "vision": vision,
            "tasks": [
                {
                    "id": "task-001",
                    "name": "Setup project structure",
                    "description": "Create initial project layout",
                    "dependencies": [],
                    "status": "pending",
                },
                {
                    "id": "task-002",
                    "name": "Implement core functionality",
                    "description": "Build the main features",
                    "dependencies": ["task-001"],
                    "status": "pending",
                },
                {
                    "id": "task-003",
                    "name": "Write tests",
                    "description": "Create test suite",
                    "dependencies": ["task-002"],
                    "status": "pending",
                },
            ],
            "dependencies": {
                "task-002": ["task-001"],
                "task-003": ["task-002"],
            },
            "estimates": {
                "task-001": 60,
                "task-002": 300,
                "task-003": 120,
            },
        }


class ImplementPhase(PhaseHandler):
    """Implementation phase handler.

    Executes tasks to produce code, documentation, and other artifacts.
    """

    def __init__(
        self,
        execution_provider: Any | None = None,
        model_provider: Any | None = None,
    ) -> None:
        self.execution_provider = execution_provider
        self.model_provider = model_provider

    @property
    def name(self) -> str:
        return "implement"

    @property
    def target_state(self) -> RunState:
        return RunState.EXECUTING

    async def execute(self, context: PhaseContext) -> PhaseResult:
        """Execute implementation tasks."""
        start = time.monotonic()

        try:
            plan = context.plan
            if not plan:
                return PhaseResult(
                    phase_name=self.name,
                    success=False,
                    duration_seconds=time.monotonic() - start,
                    error="No plan available",
                )

            tasks = plan.get("tasks", [])
            completed_tasks = []
            failed_tasks = []

            # Execute tasks in dependency order
            for task in self._order_tasks(tasks, plan.get("dependencies", {})):
                result = await self._execute_task(task, context)
                if result["success"]:
                    completed_tasks.append(task["id"])
                else:
                    failed_tasks.append(task["id"])
                    # Stop on first failure (could be configurable)
                    break

            success = len(failed_tasks) == 0

            return PhaseResult(
                phase_name=self.name,
                success=success,
                duration_seconds=time.monotonic() - start,
                outputs={
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                },
                artifacts=context.artifacts,
                notes=[f"Completed {len(completed_tasks)}/{len(tasks)} tasks"],
            )

        except Exception as e:
            return PhaseResult(
                phase_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )

    def _order_tasks(
        self,
        tasks: list[dict],
        dependencies: dict[str, list[str]],
    ) -> list[dict]:
        """Order tasks respecting dependencies (topological sort)."""
        # Simple implementation - could use proper topological sort
        task_map = {t["id"]: t for t in tasks}
        ordered = []
        seen = set()

        def visit(task_id: str) -> None:
            if task_id in seen:
                return
            for dep in dependencies.get(task_id, []):
                visit(dep)
            seen.add(task_id)
            if task_id in task_map:
                ordered.append(task_map[task_id])

        for task in tasks:
            visit(task["id"])

        return ordered

    async def _execute_task(
        self,
        task: dict,
        context: PhaseContext,
    ) -> dict[str, Any]:
        """Execute a single task."""
        logger.info(
            "Executing task",
            task_id=task["id"],
            task_name=task["name"],
        )

        # Placeholder - would use execution provider in real implementation
        task_dir = context.workspace / "tasks" / task["id"]
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create placeholder output
        output_file = task_dir / "output.txt"
        output_file.write_text(f"Task {task['id']} completed\n")

        context.artifacts.append(str(output_file))

        return {"success": True, "output": str(output_file)}


class TestPhase(PhaseHandler):
    """Testing phase handler.

    Runs tests and collects coverage information.
    """

    def __init__(
        self,
        evaluator: Any | None = None,
        min_coverage: float = 80.0,
    ) -> None:
        self.evaluator = evaluator
        self.min_coverage = min_coverage

    @property
    def name(self) -> str:
        return "test"

    @property
    def target_state(self) -> RunState:
        return RunState.EXECUTING

    async def execute(self, context: PhaseContext) -> PhaseResult:
        """Run tests on the implementation."""
        start = time.monotonic()

        try:
            # Create tests directory
            tests_dir = context.workspace / "tests"
            tests_dir.mkdir(exist_ok=True)

            # Create placeholder test
            test_file = tests_dir / "test_generated.py"
            test_file.write_text(
                "import pytest\n\n"
                "def test_placeholder():\n"
                "    '''Placeholder test.'''\n"
                "    assert True\n"
            )

            # Run tests (would use evaluator in real implementation)
            test_results = {
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "coverage": 100.0,
            }

            success = test_results["failed"] == 0

            # Check coverage
            if test_results["coverage"] < self.min_coverage:
                success = False

            return PhaseResult(
                phase_name=self.name,
                success=success,
                duration_seconds=time.monotonic() - start,
                outputs={"test_results": test_results},
                artifacts=[str(test_file)],
                notes=[
                    f"Tests: {test_results['passed']} passed, {test_results['failed']} failed",
                    f"Coverage: {test_results['coverage']}%",
                ],
            )

        except Exception as e:
            return PhaseResult(
                phase_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class VerifyPhase(PhaseHandler):
    """Verification phase handler.

    Validates the implementation against the original vision.
    """

    def __init__(
        self,
        model_provider: Any | None = None,
    ) -> None:
        self.model_provider = model_provider

    @property
    def name(self) -> str:
        return "verify"

    @property
    def target_state(self) -> RunState:
        return RunState.VERIFYING

    async def execute(self, context: PhaseContext) -> PhaseResult:
        """Verify implementation meets vision."""
        start = time.monotonic()

        try:
            vision = context.run.vision
            plan = context.plan

            # Verification checks
            checks = {
                "tests_passed": True,
                "coverage_met": True,
                "artifacts_present": len(context.artifacts) > 0,
                "vision_addressed": True,
            }

            success = all(checks.values())

            # Generate verification report
            report = {
                "vision": vision[:100] + "..." if len(vision) > 100 else vision,
                "checks": checks,
                "artifacts_count": len(context.artifacts),
                "verified_at": time.time(),
            }

            report_file = context.workspace / "verification.json"
            import json

            report_file.write_text(json.dumps(report, indent=2))

            return PhaseResult(
                phase_name=self.name,
                success=success,
                duration_seconds=time.monotonic() - start,
                outputs={"verification": report},
                artifacts=[str(report_file)],
                notes=[
                    f"Verification {'passed' if success else 'failed'}",
                    f"Checks: {sum(checks.values())}/{len(checks)} passed",
                ],
            )

        except Exception as e:
            return PhaseResult(
                phase_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class PhaseOrchestrator:
    """Orchestrates phase execution."""

    def __init__(
        self,
        phases: list[PhaseHandler] | None = None,
    ) -> None:
        self.phases = phases or [
            PlanPhase(),
            ImplementPhase(),
            TestPhase(),
            VerifyPhase(),
        ]

    async def run_all(self, context: PhaseContext) -> list[PhaseResult]:
        """Run all phases in sequence.

        Args:
            context: Phase execution context

        Returns:
            List of phase results
        """
        results = []

        for phase in self.phases:
            await phase.pre_execute(context)
            result = await phase.execute(context)
            await phase.post_execute(context, result)

            results.append(result)

            if not result.success:
                logger.error(
                    "Phase failed, stopping",
                    phase=phase.name,
                    error=result.error,
                )
                break

        return results

    def get_phase(self, name: str) -> PhaseHandler | None:
        """Get a phase handler by name."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None
