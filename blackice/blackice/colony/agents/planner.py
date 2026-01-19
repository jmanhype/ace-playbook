"""Planner Agent for BLACKICE 3.0 colony.

The Planner Agent is responsible for:
- Analyzing project requirements and vision
- Breaking down work into actionable tasks
- Designing system architecture
- Making technology choices
- Creating implementation plans

Uses the ARCHITECT role from the agent schema.
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


# Enhanced system prompt for the Planner
PLANNER_SYSTEM_PROMPT = """You are a software architect and planning agent. Your role is to:
- Analyze project requirements and break them down into concrete tasks
- Design system architecture with clear component boundaries
- Make technology choices based on requirements and constraints
- Create detailed implementation plans with dependencies
- Ensure designs are scalable, secure, and maintainable

When planning:
1. Start with understanding the full scope of requirements
2. Identify key components and their responsibilities
3. Define clear interfaces between components
4. Consider error handling, security, and performance from the start
5. Break work into small, testable increments

Output should be structured and actionable. Each task should be clear enough
for an implementer to work on independently.

Format your plans as JSON when requested for machine parsing."""


class PlannerAgent(BaseColonyAgent):
    """Planner agent for architecture and task breakdown.

    The Planner is typically the first agent to work on a project,
    transforming a vision statement into a concrete implementation plan.
    """

    def __init__(
        self,
        agent: Agent,
        model_provider: ModelProvider | None = None,
        memory_provider: MemoryProvider | None = None,
    ) -> None:
        """Initialize the Planner agent."""
        super().__init__(agent, model_provider, memory_provider)
        # Override system prompt with enhanced version
        if agent.system_prompt == AGENT_PROMPTS.get(AgentRole.ARCHITECT):
            agent.system_prompt = PLANNER_SYSTEM_PROMPT

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a planning task.

        Args:
            task: Task definition with 'type' and task-specific fields

        Returns:
            Task result with plan, tasks, or architecture output
        """
        task_type = task.get("type", "plan")
        task_id = TaskId(task.get("id", "unknown"))

        await self.start_execution(task_id)

        try:
            if task_type == "analyze":
                result = await self._analyze_requirements(task)
            elif task_type == "breakdown":
                result = await self._breakdown_work(task)
            elif task_type == "architecture":
                result = await self._design_architecture(task)
            elif task_type == "plan":
                result = await self._create_plan(task)
            else:
                result = await self._generic_planning(task)

            await self.complete_execution(success=True, output=json.dumps(result))
            return result

        except Exception as e:
            await self.complete_execution(success=False, error=str(e))
            raise

    async def _analyze_requirements(self, task: dict[str, Any]) -> dict[str, Any]:
        """Analyze project requirements from a vision statement.

        Args:
            task: Task with 'vision' and optional 'constraints'

        Returns:
            Analysis with requirements, stakeholders, scope
        """
        vision = task.get("vision", "")
        constraints = task.get("constraints", [])

        constraints_str = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified"

        prompt = f"""Analyze the following project vision and extract structured requirements.

Vision:
{vision}

Constraints:
{constraints_str}

Provide a structured analysis with:
1. Functional requirements (what the system must do)
2. Non-functional requirements (performance, security, etc.)
3. Key stakeholders and their needs
4. Scope boundaries (what's in/out of scope)
5. Assumptions being made

Format your response as valid JSON with these keys:
- functional_requirements: list of requirements
- non_functional_requirements: list of requirements
- stakeholders: list of stakeholder objects with 'name' and 'needs'
- scope: object with 'in_scope' and 'out_of_scope' lists
- assumptions: list of assumptions"""

        response = await self.think(prompt)

        # Try to parse as JSON, fall back to structured text
        try:
            # Find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_analysis": response}
        except json.JSONDecodeError:
            result = {"raw_analysis": response}

        result["vision"] = vision
        return result

    async def _breakdown_work(self, task: dict[str, Any]) -> dict[str, Any]:
        """Break down work into tasks.

        Args:
            task: Task with 'scope' or 'requirements'

        Returns:
            Task breakdown with dependencies
        """
        scope = task.get("scope", task.get("requirements", ""))

        prompt = f"""Break down the following work into concrete, actionable tasks.

Scope/Requirements:
{scope}

For each task, provide:
1. A unique ID (T001, T002, etc.)
2. Clear description
3. Dependencies on other tasks
4. Estimated complexity (low/medium/high)
5. Skills/role needed (architect/implementer/tester/etc.)

Format your response as valid JSON with:
- tasks: list of task objects with 'id', 'description', 'dependencies', 'complexity', 'role'
- phases: logical groupings of tasks
- critical_path: list of task IDs on the critical path"""

        response = await self.think(prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_breakdown": response}
        except json.JSONDecodeError:
            result = {"raw_breakdown": response}

        return result

    async def _design_architecture(self, task: dict[str, Any]) -> dict[str, Any]:
        """Design system architecture.

        Args:
            task: Task with 'requirements' and optional 'constraints'

        Returns:
            Architecture design with components and interfaces
        """
        requirements = task.get("requirements", "")
        constraints = task.get("constraints", [])
        tech_stack = task.get("tech_stack", {})

        constraints_str = "\n".join(f"- {c}" for c in constraints) if constraints else "None"
        tech_str = json.dumps(tech_stack) if tech_stack else "Not specified"

        prompt = f"""Design a system architecture for the following requirements.

Requirements:
{requirements}

Constraints:
{constraints_str}

Tech Stack Preferences:
{tech_str}

Provide:
1. High-level component diagram (describe components and connections)
2. Component responsibilities
3. Interfaces between components
4. Data flow
5. Technology recommendations with rationale
6. Key design decisions and trade-offs

Format as JSON with:
- components: list of component objects with 'name', 'responsibility', 'interfaces'
- data_flow: description of how data moves through the system
- technologies: list of tech recommendations with 'category', 'choice', 'rationale'
- decisions: list of key decisions with 'decision', 'alternatives', 'rationale'"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"raw_architecture": response}
        except json.JSONDecodeError:
            result = {"raw_architecture": response}

        return result

    async def _create_plan(self, task: dict[str, Any]) -> dict[str, Any]:
        """Create a complete implementation plan.

        Args:
            task: Task with 'vision' and optional additional context

        Returns:
            Complete plan with architecture, tasks, and timeline
        """
        vision = task.get("vision", "")
        context = task.get("context", "")

        # Step 1: Analyze requirements
        analysis = await self._analyze_requirements({
            "vision": vision,
            "constraints": task.get("constraints", []),
        })

        # Step 2: Design architecture
        requirements_summary = json.dumps(analysis.get("functional_requirements", []))
        architecture = await self._design_architecture({
            "requirements": requirements_summary,
            "constraints": task.get("constraints", []),
            "tech_stack": task.get("tech_stack", {}),
        })

        # Step 3: Break down into tasks
        breakdown = await self._breakdown_work({
            "scope": requirements_summary,
        })

        # Combine into final plan
        return {
            "vision": vision,
            "analysis": analysis,
            "architecture": architecture,
            "tasks": breakdown.get("tasks", []),
            "phases": breakdown.get("phases", []),
            "critical_path": breakdown.get("critical_path", []),
        }

    async def _generic_planning(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle generic planning requests.

        Args:
            task: Task with 'description' and optional context

        Returns:
            Planning output based on the request
        """
        description = task.get("description", str(task))

        prompt = f"""You are helping with a planning task:

{description}

Provide a thoughtful response that helps move this project forward.
If this requires analysis, break it down. If it requires decisions, explain the trade-offs.
If it requires a plan, provide actionable steps.

Format your response as JSON when possible for machine parsing."""

        response = await self.think(prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"response": response}
        except json.JSONDecodeError:
            result = {"response": response}

        return result

    async def review_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Review and refine an existing plan.

        Args:
            plan: Existing plan to review

        Returns:
            Review with issues, suggestions, and refined plan
        """
        prompt = f"""Review the following implementation plan for completeness,
correctness, and potential issues.

Plan:
{json.dumps(plan, indent=2)}

Provide:
1. Issues found (missing pieces, unclear requirements, risks)
2. Suggestions for improvement
3. Confidence level in the plan (0.0 to 1.0)
4. A refined version of the plan if needed

Format as JSON with:
- issues: list of issue objects with 'severity', 'description'
- suggestions: list of improvement suggestions
- confidence: float between 0 and 1
- refined_plan: updated plan (or null if no changes needed)"""

        response = await self.think(prompt, max_tokens=8192)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {"raw_review": response}
        except json.JSONDecodeError:
            return {"raw_review": response}


def create_planner(
    agent_id: AgentId | None = None,
    model_provider: ModelProvider | None = None,
    memory_provider: MemoryProvider | None = None,
) -> PlannerAgent:
    """Factory function to create a Planner agent.

    Args:
        agent_id: Optional custom agent ID
        model_provider: LLM provider for inference
        memory_provider: Memory provider for learning

    Returns:
        Configured PlannerAgent instance
    """
    from blackice.schemas.agent import create_agent

    agent = create_agent(
        role=AgentRole.ARCHITECT,
        agent_id=agent_id,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        capabilities=AgentCapabilities(
            can_write_code=False,  # Planner focuses on design, not code
            can_execute_commands=False,
            can_read_files=True,
            can_write_files=True,
            domains=["architecture", "planning", "design"],
        ),
    )

    return PlannerAgent(agent, model_provider, memory_provider)
