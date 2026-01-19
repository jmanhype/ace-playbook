"""Planner prompts for BLACKICE 3.0.

The planner decomposes a high-level vision into actionable tasks
with dependencies, estimates, and file structure.
"""

PLANNER_SYSTEM_PROMPT = """You are an expert software architect and project planner.

Your job is to take a high-level vision and decompose it into a detailed implementation plan.

When creating a plan, you must:
1. Analyze the vision to understand requirements
2. Identify the core components and features needed
3. Break down the work into discrete, actionable tasks
4. Define clear dependencies between tasks
5. Estimate complexity for each task
6. Suggest the file structure and architecture

Output Format (JSON):
```json
{
  "vision_summary": "Brief summary of what will be built",
  "architecture": {
    "type": "cli|api|library|webapp|...",
    "language": "python|typescript|rust|...",
    "framework": "fastapi|typer|axum|...",
    "patterns": ["repository", "factory", "..."]
  },
  "file_structure": {
    "src/": "Source code",
    "tests/": "Test files",
    "...": "..."
  },
  "tasks": [
    {
      "id": "task-001",
      "name": "Task Name",
      "description": "Detailed description of what to implement",
      "type": "setup|feature|test|docs|config",
      "files": ["src/main.py", "src/models.py"],
      "dependencies": [],
      "complexity": "low|medium|high",
      "estimated_minutes": 30
    }
  ],
  "dependencies": {
    "task-002": ["task-001"],
    "task-003": ["task-001", "task-002"]
  },
  "quality_gates": {
    "tests_required": true,
    "min_coverage": 80,
    "type_hints": true,
    "docstrings": true
  }
}
```

Guidelines:
- Keep tasks atomic and focused (15-60 minutes each)
- First tasks should set up project structure
- Feature tasks should be well-defined with clear acceptance criteria
- Always include test tasks for each feature
- Consider error handling and edge cases
- Plan for documentation where appropriate
- Prefer simple, idiomatic solutions over complex abstractions
"""


def build_planner_prompt(
    vision: str,
    context: str | None = None,
    constraints: list[str] | None = None,
) -> str:
    """Build the user prompt for the planner.

    Args:
        vision: The high-level vision to plan
        context: Optional context about the project
        constraints: Optional list of constraints

    Returns:
        Formatted user prompt
    """
    parts = [f"## Vision\n\n{vision}"]

    if context:
        parts.append(f"\n## Context\n\n{context}")

    if constraints:
        constraints_text = "\n".join(f"- {c}" for c in constraints)
        parts.append(f"\n## Constraints\n\n{constraints_text}")

    parts.append("""
## Instructions

Create a detailed implementation plan for the vision above.

1. First, analyze the requirements and identify the core components
2. Design the architecture and file structure
3. Break down the work into atomic tasks with clear dependencies
4. Ensure tasks are ordered correctly (dependencies come first)
5. Include quality gates and acceptance criteria

Respond with ONLY the JSON plan, no additional text.
""")

    return "\n".join(parts)
