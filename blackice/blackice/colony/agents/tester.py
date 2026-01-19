"""Tester Agent for BLACKICE 3.0 colony.

The Tester Agent is responsible for:
- Writing comprehensive unit and integration tests
- Identifying edge cases and boundary conditions
- Ensuring test coverage meets requirements
- Validating that code behaves correctly

Uses the TESTER role from the agent schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from blackice.colony.agents.base import BaseColonyAgent
from blackice.primitives.types import AgentId, AgentRole, TaskId
from blackice.schemas.agent import Agent, AgentCapabilities, AGENT_PROMPTS

if TYPE_CHECKING:
    from blackice.adapters.models.base import ModelProvider
    from blackice.adapters.memory.base import MemoryProvider


logger = structlog.get_logger(__name__)


TESTER_SYSTEM_PROMPT = """You are a software testing expert and QA agent. Your role is to:
- Write comprehensive unit tests covering all code paths
- Create integration tests for component interactions
- Identify edge cases, boundary conditions, and error scenarios
- Ensure test coverage meets or exceeds requirements
- Validate behavior matches specifications

When writing tests:
1. Test the happy path first
2. Test edge cases and boundary conditions
3. Test error handling and failure modes
4. Use descriptive test names that explain what's being tested
5. Follow the Arrange-Act-Assert pattern
6. Use fixtures and parameterization appropriately
7. Aim for 100% coverage of critical paths

Write tests that are:
- Readable and maintainable
- Independent and isolated
- Fast and deterministic
- Meaningful (not just for coverage)"""


class TesterAgent(BaseColonyAgent):
    """Tester agent for test creation and validation.

    The Tester creates comprehensive tests to ensure code correctness
    and maintains high test coverage.
    """

    def __init__(
        self,
        agent: Agent,
        model_provider: ModelProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        """Initialize the Tester agent."""
        super().__init__(agent, model_provider, memory_provider)
        if agent.system_prompt == AGENT_PROMPTS.get(AgentRole.TESTER):
            agent.system_prompt = TESTER_SYSTEM_PROMPT

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a testing task.

        Args:
            task: Task definition with 'type' and task-specific fields

        Returns:
            Task result with tests or test results
        """
        task_type = task.get("type", "write_tests")
        task_id = TaskId(task.get("id", "unknown"))

        await self.start_execution(task_id)

        try:
            if task_type == "write_tests":
                result = await self._write_tests(task)
            elif task_type == "unit_tests":
                result = await self._write_unit_tests(task)
            elif task_type == "integration_tests":
                result = await self._write_integration_tests(task)
            elif task_type == "edge_cases":
                result = await self._identify_edge_cases(task)
            elif task_type == "coverage":
                result = await self._analyze_coverage(task)
            else:
                result = await self._generic_testing(task)

            await self.complete_execution(success=True, output=json.dumps(result))
            return result

        except Exception as e:
            await self.complete_execution(success=False, error=str(e))
            raise

    async def _write_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        """Write tests for given code.

        Args:
            task: Task with 'code', 'file_path', optional 'specification'

        Returns:
            Test code and coverage analysis
        """
        code = task.get("code", "")
        file_path = task.get("file_path", "")
        spec = task.get("specification", "")
        framework = task.get("framework", "pytest")

        spec_section = f"\n\nSpecification:\n{spec}" if spec else ""

        prompt = f"""Write comprehensive tests for the following code:

File: {file_path}

```
{code}
```{spec_section}

Using: {framework}

Create tests that:
1. Cover all public functions/methods
2. Test happy path scenarios
3. Test edge cases and boundary conditions
4. Test error handling
5. Use appropriate fixtures and mocking

Format as JSON with:
- test_file_path: where to save the test file
- test_code: complete test file contents
- test_cases: list of test case names and what they test
- coverage_targets: list of code paths being tested
- mocks_needed: list of dependencies that need mocking"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_tests": response}
        except json.JSONDecodeError:
            result = {"raw_tests": response}

        return result

    async def _write_unit_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        """Write focused unit tests.

        Args:
            task: Task with 'function' or 'class', 'code'

        Returns:
            Unit test code
        """
        code = task.get("code", "")
        target = task.get("function", task.get("class", ""))
        framework = task.get("framework", "pytest")

        prompt = f"""Write unit tests for the following:

Target: {target}

```
{code}
```

Framework: {framework}

Focus on:
1. Testing each method in isolation
2. Mocking all external dependencies
3. Parameterizing for different inputs
4. Testing return values and exceptions

Format as JSON with:
- test_code: the unit test code
- test_count: number of test cases
- mocked_dependencies: list of mocked items"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_unit_tests": response}
        except json.JSONDecodeError:
            return {"raw_unit_tests": response}

    async def _write_integration_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        """Write integration tests.

        Args:
            task: Task with 'components', 'interactions'

        Returns:
            Integration test code
        """
        components = task.get("components", [])
        interactions = task.get("interactions", "")
        code = task.get("code", "")
        framework = task.get("framework", "pytest")

        components_str = "\n".join(f"- {c}" for c in components) if components else "Not specified"

        prompt = f"""Write integration tests for the following system:

Components:
{components_str}

Interactions:
{interactions or 'Not specified'}

Code:
```
{code}
```

Framework: {framework}

Focus on:
1. Testing component interactions
2. Testing data flow between components
3. Testing with real (or realistic mock) dependencies
4. Testing end-to-end scenarios

Format as JSON with:
- test_code: the integration test code
- scenarios_tested: list of scenarios covered
- test_fixtures: fixtures needed for the tests"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_integration_tests": response}
        except json.JSONDecodeError:
            return {"raw_integration_tests": response}

    async def _identify_edge_cases(self, task: dict[str, Any]) -> dict[str, Any]:
        """Identify edge cases that should be tested.

        Args:
            task: Task with 'code' or 'specification'

        Returns:
            List of edge cases and test suggestions
        """
        code = task.get("code", "")
        spec = task.get("specification", "")

        context = code if code else spec

        prompt = f"""Analyze the following and identify all edge cases that should be tested:

{context}

Consider:
1. Boundary conditions (min/max values, empty collections)
2. Invalid inputs (null, empty, malformed)
3. Race conditions and concurrency
4. Resource exhaustion (memory, connections)
5. Error cascades
6. Unicode and special characters
7. Time-related edge cases (timezone, DST)
8. State transitions

Format as JSON with:
- edge_cases: list with 'category', 'description', 'test_approach'
- priority_order: which to test first
- hardest_to_test: cases that are difficult to reproduce"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_edge_cases": response}
        except json.JSONDecodeError:
            return {"raw_edge_cases": response}

    async def _analyze_coverage(self, task: dict[str, Any]) -> dict[str, Any]:
        """Analyze test coverage gaps.

        Args:
            task: Task with 'code' and 'existing_tests'

        Returns:
            Coverage analysis and gap recommendations
        """
        code = task.get("code", "")
        existing_tests = task.get("existing_tests", "")

        prompt = f"""Analyze test coverage for the following:

Code:
```
{code}
```

Existing Tests:
```
{existing_tests}
```

Identify:
1. Code paths not covered by tests
2. Functions/methods without tests
3. Edge cases not tested
4. Error handling not tested

Format as JSON with:
- coverage_gaps: list of uncovered code areas
- missing_tests: specific tests that should be added
- priority: which gaps are most critical
- estimated_coverage: rough percentage covered"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_coverage_analysis": response}
        except json.JSONDecodeError:
            return {"raw_coverage_analysis": response}

    async def _generic_testing(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle generic testing requests."""
        description = task.get("description", str(task))

        prompt = f"""Testing task:

{description}

Create appropriate tests for this task.
Format as JSON with 'test_code' and 'test_cases'."""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_tests": response}
        except json.JSONDecodeError:
            return {"raw_tests": response}


def create_tester(
    agent_id: AgentId | None = None,
    model_provider: ModelProvider | None = None,
    memory_provider: MemoryProvider | None = None,
) -> TesterAgent:
    """Factory function to create a Tester agent."""
    from blackice.schemas.agent import create_agent

    agent = create_agent(
        role=AgentRole.TESTER,
        agent_id=agent_id,
        system_prompt=TESTER_SYSTEM_PROMPT,
        capabilities=AgentCapabilities(
            can_write_code=True,  # Testers write test code
            can_execute_commands=True,  # To run tests
            can_read_files=True,
            can_write_files=True,
            domains=["testing", "qa", "quality"],
        ),
    )

    return TesterAgent(agent, model_provider, memory_provider)
