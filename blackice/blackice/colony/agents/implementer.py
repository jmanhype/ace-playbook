"""Implementer Agent for BLACKICE 3.0 colony.

The Implementer Agent is responsible for:
- Writing clean, well-structured code
- Implementing features according to specifications
- Handling edge cases and error conditions
- Following project conventions and best practices

Uses the IMPLEMENTER role from the agent schema.
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


IMPLEMENTER_SYSTEM_PROMPT = """You are a senior software developer and implementation agent. Your role is to:
- Write clean, well-structured, production-quality code
- Implement features according to the architectural design and specifications
- Handle edge cases, error conditions, and boundary conditions
- Follow project conventions, coding standards, and best practices
- Write self-documenting code with appropriate comments

When implementing:
1. Start by understanding the full requirements and context
2. Consider the existing codebase patterns and conventions
3. Write code that is testable, maintainable, and extensible
4. Handle errors gracefully with appropriate error messages
5. Add type hints and documentation where helpful
6. Consider security implications of your code

Always explain your implementation decisions and any trade-offs made.
When generating code, include the full file path and complete file contents."""


class ImplementerAgent(BaseColonyAgent):
    """Implementer agent for writing code.

    The Implementer turns designs and specifications into working code,
    following best practices and project conventions.
    """

    def __init__(
        self,
        agent: Agent,
        model_provider: ModelProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        """Initialize the Implementer agent."""
        super().__init__(agent, model_provider, memory_provider)
        if agent.system_prompt == AGENT_PROMPTS.get(AgentRole.IMPLEMENTER):
            agent.system_prompt = IMPLEMENTER_SYSTEM_PROMPT

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute an implementation task.

        Args:
            task: Task definition with 'type' and task-specific fields

        Returns:
            Task result with generated code and artifacts
        """
        task_type = task.get("type", "implement")
        task_id = TaskId(task.get("id", "unknown"))

        await self.start_execution(task_id)

        try:
            if task_type == "implement":
                result = await self._implement_feature(task)
            elif task_type == "refactor":
                result = await self._refactor_code(task)
            elif task_type == "fix":
                result = await self._fix_bug(task)
            elif task_type == "complete":
                result = await self._complete_code(task)
            else:
                result = await self._generic_implementation(task)

            await self.complete_execution(success=True, output=json.dumps(result))
            return result

        except Exception as e:
            await self.complete_execution(success=False, error=str(e))
            raise

    async def _implement_feature(self, task: dict[str, Any]) -> dict[str, Any]:
        """Implement a new feature.

        Args:
            task: Task with 'specification', 'file_path', optional 'context'

        Returns:
            Implementation with code and explanation
        """
        spec = task.get("specification", task.get("description", ""))
        file_path = task.get("file_path", "")
        context = task.get("context", "")
        existing_code = task.get("existing_code", "")

        context_section = f"\n\nExisting Context:\n{context}" if context else ""
        existing_section = f"\n\nExisting Code:\n```\n{existing_code}\n```" if existing_code else ""

        prompt = f"""Implement the following feature:

Specification:
{spec}

Target File: {file_path or 'Not specified'}{context_section}{existing_section}

Requirements:
1. Write complete, production-ready code
2. Include type hints for Python or TypeScript types
3. Add docstrings/comments for complex logic
4. Handle error cases appropriately
5. Follow the existing code style if provided

Provide:
1. The complete code for the file
2. Brief explanation of key implementation decisions
3. Any dependencies that need to be added

Format as JSON with:
- code: the complete file contents as a string
- file_path: where to save the file
- explanation: implementation notes
- dependencies: list of new dependencies needed"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_implementation": response}
        except json.JSONDecodeError:
            result = {"raw_implementation": response}

        return result

    async def _refactor_code(self, task: dict[str, Any]) -> dict[str, Any]:
        """Refactor existing code.

        Args:
            task: Task with 'code', 'goals', optional 'constraints'

        Returns:
            Refactored code with explanation
        """
        code = task.get("code", "")
        goals = task.get("goals", ["improve readability", "reduce complexity"])
        constraints = task.get("constraints", [])

        goals_str = "\n".join(f"- {g}" for g in goals)
        constraints_str = "\n".join(f"- {c}" for c in constraints) if constraints else "None"

        prompt = f"""Refactor the following code:

```
{code}
```

Refactoring Goals:
{goals_str}

Constraints:
{constraints_str}

Provide:
1. The refactored code
2. Summary of changes made
3. Why each change improves the code

Format as JSON with:
- code: the refactored code
- changes: list of change descriptions
- improvements: how the code is better now"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_refactor": response}
        except json.JSONDecodeError:
            return {"raw_refactor": response}

    async def _fix_bug(self, task: dict[str, Any]) -> dict[str, Any]:
        """Fix a bug in existing code.

        Args:
            task: Task with 'code', 'bug_description', optional 'error_message'

        Returns:
            Fixed code with explanation
        """
        code = task.get("code", "")
        bug_description = task.get("bug_description", task.get("description", ""))
        error_message = task.get("error_message", "")

        error_section = f"\n\nError Message:\n{error_message}" if error_message else ""

        prompt = f"""Fix the bug in the following code:

```
{code}
```

Bug Description:
{bug_description}{error_section}

Provide:
1. The fixed code
2. Root cause analysis
3. Explanation of the fix
4. Suggestions to prevent similar bugs

Format as JSON with:
- code: the fixed code
- root_cause: what caused the bug
- fix_explanation: how the fix addresses the issue
- prevention: how to prevent similar bugs"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_fix": response}
        except json.JSONDecodeError:
            return {"raw_fix": response}

    async def _complete_code(self, task: dict[str, Any]) -> dict[str, Any]:
        """Complete partial code implementation.

        Args:
            task: Task with 'partial_code', 'completion_instructions'

        Returns:
            Completed code
        """
        partial_code = task.get("partial_code", task.get("code", ""))
        instructions = task.get("completion_instructions", task.get("instructions", ""))

        prompt = f"""Complete the following partial code:

```
{partial_code}
```

Instructions:
{instructions}

Complete the code following the established patterns.
Provide the full, completed code.

Format as JSON with:
- code: the complete code
- added_sections: list of what was added"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_completion": response}
        except json.JSONDecodeError:
            return {"raw_completion": response}

    async def _generic_implementation(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle generic implementation requests."""
        description = task.get("description", str(task))

        prompt = f"""Implementation task:

{description}

Write the code needed to accomplish this task.
Follow best practices and include appropriate error handling.

Format as JSON with 'code' containing the implementation."""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_implementation": response}
        except json.JSONDecodeError:
            return {"raw_implementation": response}


def create_implementer(
    agent_id: AgentId | None = None,
    model_provider: ModelProvider | None = None,
    memory_provider: MemoryProvider | None = None,
) -> ImplementerAgent:
    """Factory function to create an Implementer agent."""
    from blackice.schemas.agent import create_agent

    agent = create_agent(
        role=AgentRole.IMPLEMENTER,
        agent_id=agent_id,
        system_prompt=IMPLEMENTER_SYSTEM_PROMPT,
        capabilities=AgentCapabilities(
            can_write_code=True,
            can_execute_commands=True,
            can_read_files=True,
            can_write_files=True,
            languages=["python", "javascript", "typescript", "rust", "go"],
            domains=["implementation", "coding", "development"],
        ),
    )

    return ImplementerAgent(agent, model_provider, memory_provider)
