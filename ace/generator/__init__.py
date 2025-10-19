"""
Generator Module

Implements Chain-of-Thought (CoT) and ReAct generators for task execution
with playbook context injection. Produces structured reasoning traces.

Components:
- TaskInput/TaskOutput: DSPy signatures for generator I/O
- CoTGenerator: Sequential reasoning without tool use
- ReActGenerator: Tool-aware reasoning with deterministic tool registry
- ContextBuilder utilities for weighted prompt assembly

Usage:
    from ace.generator import CoTGenerator, TaskInput

    generator = CoTGenerator(model="gpt-4-turbo")

    task = TaskInput(
        task_id="task-001",
        description="Calculate 37 * 42",
        playbook_bullets=["Break arithmetic into steps"],
        domain="arithmetic"
    )

    output = generator(task)
    print(output.reasoning_trace)
    print(output.answer)
"""

from ace.generator.signatures import (
    TaskInput,
    ChainOfThoughtSignature
)
from ace.generator.cot_generator import CoTGenerator, TaskOutput, create_cot_generator
from ace.generator.react_generator import ReActGenerator, ToolRegistry, create_react_generator
from ace.generator.context_builder import ContextEntry, ContextBundle, build_bundle, build_strings

__all__ = [
    "TaskInput",
    "TaskOutput",
    "ChainOfThoughtSignature",
    "CoTGenerator",
    "ReActGenerator",
    "create_cot_generator",
    "create_react_generator",
    "ToolRegistry",
    "ContextEntry",
    "ContextBundle",
    "build_bundle",
    "build_strings",
]

__version__ = "v1.0.0"
