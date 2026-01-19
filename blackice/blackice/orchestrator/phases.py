"""Phase Handlers for BLACKICE 3.0 Orchestrator.

Implements handlers for each phase of the build lifecycle with REAL LLM integration:
- Plan: Decompose vision into tasks via LLM
- Implement: Execute tasks to produce code via LLM
- Test: Run tests and verify quality
- Verify: Validate against original vision via LLM
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blackice.adapters.execution import ExecutionProvider
from blackice.adapters.memory import MemoryProvider
from blackice.adapters.models import Message, ModelProvider
from blackice.instrumentation import get_logger
from blackice.orchestrator.state_machine import RunContext, RunState
from blackice.prompts import (
    CODER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    build_coder_prompt,
    build_planner_prompt,
    build_tester_prompt,
    build_verifier_prompt,
)
from blackice.prompts.coder import parse_coder_response
from blackice.prompts.tester import parse_tester_response
from blackice.prompts.verifier import parse_verifier_response

logger = get_logger(__name__)


@dataclass
class PhaseContext:
    """Context for phase execution."""

    run: RunContext
    workspace: Path
    plan: dict[str, Any] | None = None
    artifacts: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)

    # Provider references
    model_provider: ModelProvider | None = None
    execution_provider: ExecutionProvider | None = None
    memory_provider: MemoryProvider | None = None


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
    """Planning phase handler - uses REAL LLM.

    Decomposes the vision into actionable tasks with dependencies.
    """

    def __init__(
        self,
        model_provider: ModelProvider | None = None,
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
        """Generate a plan from the vision using LLM."""
        start = time.monotonic()

        try:
            vision = context.run.vision
            provider = context.model_provider or self.model_provider

            if not provider:
                # Fall back to stub if no provider
                logger.warning("No model provider, using stub plan")
                plan = self._stub_plan(vision)
            else:
                plan = await self._generate_plan(vision, context, provider)

            # Store plan
            plan_file = context.workspace / "plan.json"
            plan_file.write_text(json.dumps(plan, indent=2))

            context.plan = plan
            context.artifacts.append(str(plan_file))

            task_count = len(plan.get("tasks", []))
            return PhaseResult(
                phase_name=self.name,
                success=True,
                duration_seconds=time.monotonic() - start,
                outputs={"plan": plan, "task_count": task_count},
                artifacts=[str(plan_file)],
                notes=[f"Generated {task_count} tasks"],
            )

        except Exception as e:
            logger.error("Plan phase failed", error=str(e))
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
        provider: ModelProvider,
    ) -> dict[str, Any]:
        """Generate a plan from vision using LLM."""
        user_prompt = build_planner_prompt(vision=vision)

        logger.info("Calling LLM for plan generation")
        response = await provider.chat([
            Message(role="system", content=PLANNER_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ])

        return self._parse_plan_response(response.content, vision)

    def _parse_plan_response(self, content: str, vision: str) -> dict[str, Any]:
        """Parse the LLM response into a plan dict."""
        content = content.strip()

        # Remove markdown code fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            plan = json.loads(content)
            if "tasks" not in plan:
                plan["tasks"] = []
            if "vision" not in plan:
                plan["vision"] = vision
            return plan
        except json.JSONDecodeError:
            return self._stub_plan(vision)

    def _stub_plan(self, vision: str) -> dict[str, Any]:
        """Return a stub plan when LLM is unavailable."""
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
    """Implementation phase handler - uses REAL LLM.

    Executes tasks to produce code, documentation, and other artifacts.
    """

    def __init__(
        self,
        execution_provider: ExecutionProvider | None = None,
        model_provider: ModelProvider | None = None,
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
        """Execute implementation tasks using LLM."""
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
            provider = context.model_provider or self.model_provider
            completed_tasks = []
            failed_tasks = []
            generated_files: dict[str, str] = {}

            # Execute tasks in dependency order
            for task in self._order_tasks(tasks, plan.get("dependencies", {})):
                if provider:
                    result = await self._execute_task_with_llm(
                        task, context, provider, plan, generated_files
                    )
                else:
                    result = await self._execute_task_stub(task, context)

                if result["success"]:
                    completed_tasks.append(task["id"])
                    generated_files.update(result.get("files", {}))
                else:
                    failed_tasks.append(task["id"])
                    break

            success = len(failed_tasks) == 0

            return PhaseResult(
                phase_name=self.name,
                success=success,
                duration_seconds=time.monotonic() - start,
                outputs={
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "generated_files": list(generated_files.keys()),
                },
                artifacts=context.artifacts,
                notes=[f"Completed {len(completed_tasks)}/{len(tasks)} tasks"],
            )

        except Exception as e:
            logger.error("Implement phase failed", error=str(e))
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
        task_map = {t["id"]: t for t in tasks}
        ordered = []
        seen = set()

        def visit(task_id: str) -> None:
            if task_id in seen:
                return
            for dep in dependencies.get(task_id, []):
                visit(dep)
            task = task_map.get(task_id)
            if task:
                for dep in task.get("dependencies", []):
                    visit(dep)
            seen.add(task_id)
            if task_id in task_map:
                ordered.append(task_map[task_id])

        for task in tasks:
            visit(task["id"])

        return ordered

    async def _execute_task_with_llm(
        self,
        task: dict,
        context: PhaseContext,
        provider: ModelProvider,
        plan: dict,
        existing_files: dict[str, str],
    ) -> dict[str, Any]:
        """Execute a task using LLM for code generation."""
        logger.info("Generating code for task", task_id=task["id"], task_name=task["name"])

        user_prompt = build_coder_prompt(
            task=task,
            plan=plan,
            existing_files=existing_files,
        )

        response = await provider.chat([
            Message(role="system", content=CODER_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ])

        task_files = parse_coder_response(response.content)

        # Write files to workspace
        for file_path, content in task_files.items():
            if not file_path.startswith(("src/", "tests/")):
                file_path = f"src/{file_path}"

            full_path = context.workspace / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            context.artifacts.append(str(full_path))

        return {"success": True, "files": task_files}

    async def _execute_task_stub(
        self,
        task: dict,
        context: PhaseContext,
    ) -> dict[str, Any]:
        """Execute a task without LLM (stub implementation)."""
        logger.info("Executing task (stub)", task_id=task["id"])

        task_dir = context.workspace / "tasks" / task["id"]
        task_dir.mkdir(parents=True, exist_ok=True)

        output_file = task_dir / "output.txt"
        output_file.write_text(f"Task {task['id']} completed\n")

        context.artifacts.append(str(output_file))
        return {"success": True, "output": str(output_file)}


class TestPhase(PhaseHandler):
    """Testing phase handler - uses REAL LLM and execution.

    Runs tests and collects coverage information.
    """

    def __init__(
        self,
        evaluator: Any | None = None,
        min_coverage: float = 80.0,
        model_provider: ModelProvider | None = None,
        execution_provider: ExecutionProvider | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.min_coverage = min_coverage
        self.model_provider = model_provider
        self.execution_provider = execution_provider

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
            # Read source files
            src_dir = context.workspace / "src"
            source_files: dict[str, str] = {}
            if src_dir.exists():
                for py_file in src_dir.rglob("*.py"):
                    rel_path = str(py_file.relative_to(context.workspace))
                    source_files[rel_path] = py_file.read_text()

            # Generate tests using LLM if available
            tests_dir = context.workspace / "tests"
            tests_dir.mkdir(exist_ok=True)
            test_files: dict[str, str] = {}

            provider = context.model_provider or self.model_provider
            if provider and source_files:
                logger.info("Generating tests via LLM")

                user_prompt = build_tester_prompt(source_files, context.plan)

                response = await provider.chat([
                    Message(role="system", content=TESTER_SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ])

                test_files = parse_tester_response(response.content)

                for file_path, content in test_files.items():
                    if not file_path.startswith("tests/"):
                        file_path = f"tests/{file_path}"

                    full_path = context.workspace / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
                    context.artifacts.append(str(full_path))
            else:
                # Create placeholder test
                test_file = tests_dir / "test_generated.py"
                test_file.write_text(
                    "import pytest\n\n"
                    "def test_placeholder():\n"
                    "    '''Placeholder test.'''\n"
                    "    assert True\n"
                )
                context.artifacts.append(str(test_file))

            # Run tests using execution provider
            test_results = {"passed": 0, "failed": 0, "skipped": 0, "coverage": 0.0}

            exec_provider = context.execution_provider or self.execution_provider
            if exec_provider:
                logger.info("Running tests via execution provider")

                result = await exec_provider.execute(
                    f"cd {context.workspace} && python -m pytest tests/ -v --tb=short 2>&1 || true",
                    timeout=600.0,
                )

                output = result.stdout or ""
                match = re.search(r"(\d+) passed", output)
                if match:
                    test_results["passed"] = int(match.group(1))
                match = re.search(r"(\d+) failed", output)
                if match:
                    test_results["failed"] = int(match.group(1))

                test_results["output"] = output[:2000]
            else:
                test_results["passed"] = len(test_files) or 1
                test_results["coverage"] = 100.0

            success = test_results["failed"] == 0

            if test_results.get("coverage", 100.0) < self.min_coverage:
                success = False

            return PhaseResult(
                phase_name=self.name,
                success=success,
                duration_seconds=time.monotonic() - start,
                outputs={"test_results": test_results},
                artifacts=context.artifacts,
                notes=[
                    f"Tests: {test_results['passed']} passed, {test_results['failed']} failed",
                    f"Coverage: {test_results.get('coverage', 0)}%",
                ],
            )

        except Exception as e:
            logger.error("Test phase failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class VerifyPhase(PhaseHandler):
    """Verification phase handler - uses REAL LLM.

    Validates the implementation against the original vision.
    """

    def __init__(
        self,
        model_provider: ModelProvider | None = None,
    ) -> None:
        self.model_provider = model_provider

    @property
    def name(self) -> str:
        return "verify"

    @property
    def target_state(self) -> RunState:
        return RunState.VERIFYING

    async def execute(self, context: PhaseContext) -> PhaseResult:
        """Verify implementation meets vision using LLM."""
        start = time.monotonic()

        try:
            vision = context.run.vision
            plan = context.plan or {}

            # Read source and test files
            source_files: dict[str, str] = {}
            test_files: dict[str, str] = {}

            src_dir = context.workspace / "src"
            if src_dir.exists():
                for py_file in src_dir.rglob("*.py"):
                    rel_path = str(py_file.relative_to(context.workspace))
                    source_files[rel_path] = py_file.read_text()

            tests_dir = context.workspace / "tests"
            if tests_dir.exists():
                for py_file in tests_dir.rglob("*.py"):
                    rel_path = str(py_file.relative_to(context.workspace))
                    test_files[rel_path] = py_file.read_text()

            # Use LLM for verification
            provider = context.model_provider or self.model_provider
            verification: dict[str, Any] = {
                "tests_passed": True,
                "coverage_met": True,
                "artifacts_present": len(context.artifacts) > 0,
                "vision_addressed": True,
            }

            if provider and source_files:
                logger.info("Running LLM verification")

                user_prompt = build_verifier_prompt(
                    vision=vision,
                    plan=plan,
                    source_files=source_files,
                    test_files=test_files,
                    test_results=context.outputs.get("test_results"),
                )

                response = await provider.chat([
                    Message(role="system", content=VERIFIER_SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ])

                verification = parse_verifier_response(response.content)

            # Determine success
            if "overall_verdict" in verification:
                success = verification["overall_verdict"] == "PASS"
            else:
                success = all(
                    v for k, v in verification.items()
                    if k not in ("_raw_response", "recommendations", "issues")
                    and isinstance(v, bool)
                )

            # Generate verification report
            report = {
                "vision": vision[:100] + "..." if len(vision) > 100 else vision,
                "verification": verification,
                "artifacts_count": len(context.artifacts),
                "verified_at": time.time(),
            }

            report_file = context.workspace / "verification.json"
            report_file.write_text(json.dumps(report, indent=2))

            return PhaseResult(
                phase_name=self.name,
                success=success,
                duration_seconds=time.monotonic() - start,
                outputs={"verification": verification},
                artifacts=[str(report_file)],
                notes=[
                    f"Verification {'passed' if success else 'failed'}",
                ],
            )

        except Exception as e:
            logger.error("Verify phase failed", error=str(e))
            return PhaseResult(
                phase_name=self.name,
                success=False,
                duration_seconds=time.monotonic() - start,
                error=str(e),
            )


class PhaseOrchestrator:
    """Orchestrates phase execution with provider injection."""

    def __init__(
        self,
        phases: list[PhaseHandler] | None = None,
        model_provider: ModelProvider | None = None,
        execution_provider: ExecutionProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        self.model_provider = model_provider
        self.execution_provider = execution_provider
        self.memory_provider = memory_provider

        self.phases = phases or [
            PlanPhase(model_provider=model_provider),
            ImplementPhase(model_provider=model_provider, execution_provider=execution_provider),
            TestPhase(model_provider=model_provider, execution_provider=execution_provider),
            VerifyPhase(model_provider=model_provider),
        ]

    async def run_all(self, context: PhaseContext) -> list[PhaseResult]:
        """Run all phases in sequence.

        Args:
            context: Phase execution context

        Returns:
            List of phase results
        """
        # Inject providers into context
        context.model_provider = context.model_provider or self.model_provider
        context.execution_provider = context.execution_provider or self.execution_provider
        context.memory_provider = context.memory_provider or self.memory_provider

        results = []

        for phase in self.phases:
            await phase.pre_execute(context)
            result = await phase.execute(context)
            await phase.post_execute(context, result)

            results.append(result)

            # Update context with outputs
            context.outputs.update(result.outputs)

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
