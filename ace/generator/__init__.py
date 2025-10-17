"""
Generator Module

Implements Chain-of-Thought (CoT) and ReAct generators for task execution
with playbook context injection. Produces structured reasoning traces.

Components:
- TaskInput/TaskOutput: DSPy signatures for generator I/O
- CoTGenerator: Sequential reasoning without tool use
- ReActGenerator: Iterative reasoning with tool interactions (NEW)
- ReasoningStep: Structured trace for ReAct iterations (NEW)

Usage:
    # CoT Generator (existing)
    from ace.generator import CoTGenerator, TaskInput

    generator = CoTGenerator(model="gpt-4-turbo")
    task = TaskInput(task_id="task-001", description="Calculate 37 * 42")
    output = generator(task)

    # ReAct Generator (new)
    from ace.generator import ReActGenerator

    def search_db(query: str) -> str:
        return "results"

    agent = ReActGenerator(tools=[search_db], model="gpt-4")
    output = agent.forward(task)  # Note: forward() for ReAct, __call__() for CoT
"""

from ace.generator.signatures import (
    TaskInput,
    ChainOfThoughtSignature,
    ReasoningStep,
)
from ace.generator.cot_generator import (
    CoTGenerator,
    TaskOutput,
    create_cot_generator
)
from ace.generator.react_generator import (
    ReActGenerator,
    ToolValidationError,
    ToolExecutionError,
    DuplicateToolError,
    ToolNotFoundError,
    MaxIterationsExceededError,
    validate_tool,
)

__all__ = [
    "TaskInput",
    "TaskOutput",
    "ChainOfThoughtSignature",
    "ReasoningStep",
    "CoTGenerator",
    "create_cot_generator",
    "ReActGenerator",
    "ToolValidationError",
    "ToolExecutionError",
    "DuplicateToolError",
    "ToolNotFoundError",
    "MaxIterationsExceededError",
    "validate_tool",
]

__version__ = "v1.0.0"
