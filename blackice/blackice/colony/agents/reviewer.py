"""Reviewer Agent for BLACKICE 3.0 colony.

The Reviewer Agent is responsible for:
- Reviewing code for correctness and best practices
- Identifying bugs, vulnerabilities, and improvements
- Ensuring code follows project conventions
- Providing constructive feedback with specific suggestions

Uses the REVIEWER role from the agent schema.
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


REVIEWER_SYSTEM_PROMPT = """You are an expert code reviewer and quality assurance agent. Your role is to:
- Review code for correctness, security, and adherence to best practices
- Identify potential bugs, edge cases, and vulnerabilities
- Suggest improvements for readability, performance, and maintainability
- Ensure code follows project conventions and coding standards
- Provide constructive, actionable feedback with specific suggestions

When reviewing:
1. Check for logical correctness and edge cases
2. Verify error handling is appropriate
3. Look for security vulnerabilities (injection, XSS, etc.)
4. Assess code readability and maintainability
5. Check for performance issues
6. Verify type safety and proper use of language features
7. Ensure documentation is adequate

Be thorough but constructive. Every criticism should come with a specific suggestion.
Categorize issues by severity: critical, high, medium, low, suggestion."""


class ReviewerAgent(BaseColonyAgent):
    """Reviewer agent for code quality and correctness.

    The Reviewer examines code for bugs, security issues, and improvements,
    providing detailed feedback to improve overall code quality.
    """

    def __init__(
        self,
        agent: Agent,
        model_provider: ModelProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        """Initialize the Reviewer agent."""
        super().__init__(agent, model_provider, memory_provider)
        if agent.system_prompt == AGENT_PROMPTS.get(AgentRole.REVIEWER):
            agent.system_prompt = REVIEWER_SYSTEM_PROMPT

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a review task.

        Args:
            task: Task definition with 'type' and task-specific fields

        Returns:
            Task result with review findings and suggestions
        """
        task_type = task.get("type", "review")
        task_id = TaskId(task.get("id", "unknown"))

        await self.start_execution(task_id)

        try:
            if task_type == "review":
                result = await self._review_code(task)
            elif task_type == "security":
                result = await self._security_review(task)
            elif task_type == "architecture":
                result = await self._architecture_review(task)
            elif task_type == "diff":
                result = await self._review_diff(task)
            else:
                result = await self._generic_review(task)

            await self.complete_execution(success=True, output=json.dumps(result))
            return result

        except Exception as e:
            await self.complete_execution(success=False, error=str(e))
            raise

    async def _review_code(self, task: dict[str, Any]) -> dict[str, Any]:
        """Review code for quality and correctness.

        Args:
            task: Task with 'code', optional 'file_path', 'context'

        Returns:
            Review with issues, suggestions, and approval status
        """
        code = task.get("code", "")
        file_path = task.get("file_path", "")
        context = task.get("context", "")
        focus_areas = task.get("focus_areas", [])

        context_section = f"\n\nContext:\n{context}" if context else ""
        focus_section = ("\n\nFocus Areas:\n" + "\n".join(f"- {f}" for f in focus_areas)) if focus_areas else ""

        prompt = f"""Review the following code:

File: {file_path or 'Unknown'}

```
{code}
```{context_section}{focus_section}

Perform a thorough code review covering:
1. Correctness and logic errors
2. Error handling
3. Security vulnerabilities
4. Performance issues
5. Code style and readability
6. Documentation quality

For each issue found, provide:
- Severity (critical/high/medium/low/suggestion)
- Line number(s) if applicable
- Description of the issue
- Specific suggestion for fixing it

Format as JSON with:
- approved: boolean (true if code is acceptable)
- issues: list of issue objects with 'severity', 'line', 'description', 'suggestion'
- summary: brief overall assessment
- strengths: what's done well
- improvements: general suggestions not tied to specific issues"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_review": response}
        except json.JSONDecodeError:
            result = {"raw_review": response}

        return result

    async def _security_review(self, task: dict[str, Any]) -> dict[str, Any]:
        """Focused security review of code.

        Args:
            task: Task with 'code', optional 'threat_model'

        Returns:
            Security review with vulnerabilities and recommendations
        """
        code = task.get("code", "")
        threat_model = task.get("threat_model", "")

        threat_section = f"\n\nThreat Model:\n{threat_model}" if threat_model else ""

        prompt = f"""Perform a security-focused code review:

```
{code}
```{threat_section}

Check for:
1. Injection vulnerabilities (SQL, command, XSS, etc.)
2. Authentication/authorization issues
3. Sensitive data exposure
4. Insecure defaults
5. Missing input validation
6. Cryptographic issues
7. Error handling that leaks information
8. Race conditions and timing issues

For each vulnerability:
- Severity (critical/high/medium/low)
- CWE or OWASP category if applicable
- Exploitation scenario
- Remediation steps

Format as JSON with:
- vulnerabilities: list with 'severity', 'category', 'description', 'exploitation', 'remediation'
- risk_score: 0-10 overall risk assessment
- recommendations: prioritized security improvements
- compliant: boolean (true if no critical/high issues)"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_security_review": response}
        except json.JSONDecodeError:
            return {"raw_security_review": response}

    async def _architecture_review(self, task: dict[str, Any]) -> dict[str, Any]:
        """Review architectural decisions and design.

        Args:
            task: Task with 'architecture', optional 'requirements'

        Returns:
            Architecture review with concerns and suggestions
        """
        architecture = task.get("architecture", "")
        requirements = task.get("requirements", "")

        req_section = f"\n\nRequirements:\n{requirements}" if requirements else ""

        prompt = f"""Review the following system architecture:

{architecture}{req_section}

Evaluate:
1. Does it meet the requirements?
2. Is it appropriately scalable?
3. Are there single points of failure?
4. Is it maintainable and extensible?
5. Are security considerations addressed?
6. Are trade-offs documented and reasonable?

Format as JSON with:
- meets_requirements: boolean
- concerns: list of architectural concerns with 'severity', 'area', 'description'
- suggestions: list of improvement suggestions
- trade_offs: identified trade-offs and whether they're appropriate
- overall_assessment: brief summary"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_architecture_review": response}
        except json.JSONDecodeError:
            return {"raw_architecture_review": response}

    async def _review_diff(self, task: dict[str, Any]) -> dict[str, Any]:
        """Review a code diff/change.

        Args:
            task: Task with 'diff' or 'before'/'after'

        Returns:
            Review of the changes
        """
        diff = task.get("diff", "")
        before = task.get("before", "")
        after = task.get("after", "")

        if not diff and before and after:
            code_section = f"Before:\n```\n{before}\n```\n\nAfter:\n```\n{after}\n```"
        else:
            code_section = f"Diff:\n```\n{diff}\n```"

        prompt = f"""Review the following code changes:

{code_section}

Evaluate:
1. Are the changes correct and complete?
2. Do they introduce any bugs or regressions?
3. Are there unintended side effects?
4. Is the change consistent with the codebase?
5. Are tests needed for these changes?

Format as JSON with:
- approved: boolean
- issues: list of concerns with the changes
- missing: anything the change should have included
- tests_needed: list of tests that should accompany this change
- summary: brief assessment of the change"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_diff_review": response}
        except json.JSONDecodeError:
            return {"raw_diff_review": response}

    async def _generic_review(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle generic review requests."""
        description = task.get("description", str(task))

        prompt = f"""Review task:

{description}

Provide a thorough review with issues and suggestions.
Format as JSON with 'issues', 'suggestions', and 'approved' fields."""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_review": response}
        except json.JSONDecodeError:
            return {"raw_review": response}


def create_reviewer(
    agent_id: AgentId | None = None,
    model_provider: ModelProvider | None = None,
    memory_provider: MemoryProvider | None = None,
) -> ReviewerAgent:
    """Factory function to create a Reviewer agent."""
    from blackice.schemas.agent import create_agent

    agent = create_agent(
        role=AgentRole.REVIEWER,
        agent_id=agent_id,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        capabilities=AgentCapabilities(
            can_write_code=False,  # Reviewer doesn't write code
            can_execute_commands=False,
            can_read_files=True,
            can_write_files=False,
            domains=["review", "quality", "security"],
        ),
    )

    return ReviewerAgent(agent, model_provider, memory_provider)
