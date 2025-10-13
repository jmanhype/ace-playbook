"""
Generator DSPy Signatures

Defines input/output interfaces for Generator modules that execute tasks
with playbook context. Based on contracts/generator.py.

Version: v1.0.0
"""

import dspy
from typing import List, Optional


class TaskInput(dspy.Signature):
    """
    Input signature for Generator task execution.

    Maps to Task entity from data-model.md with additional runtime context.
    Generator consumes this to produce TaskOutput.
    """

    # Required fields
    task_id: str = dspy.InputField(
        desc="Unique identifier for this task (UUID format)"
    )
    description: str = dspy.InputField(
        desc="Natural language description of the problem to solve"
    )

    # Optional context
    domain: str = dspy.InputField(
        default="general",
        desc="Domain classification (arithmetic, code_gen, qa, agent_workflow)"
    )
    playbook_bullets: List[str] = dspy.InputField(
        default_factory=list,
        desc="Top-K relevant strategy bullets from playbook (≤40 per invocation)"
    )
    context: str = dspy.InputField(
        default="",
        desc="Additional context or background information for the task"
    )

    # Metadata
    max_reasoning_steps: int = dspy.InputField(
        default=10,
        desc="Maximum number of reasoning steps before termination"
    )


class TaskOutput(dspy.Signature):
    """
    Output signature for Generator task execution.

    Maps to TaskOutput entity from data-model.md. Contains reasoning trace,
    final answer, and execution metadata for Reflector analysis.
    """

    # Required fields
    task_id: str = dspy.OutputField(
        desc="Task identifier (must match input task_id)"
    )
    reasoning_trace: List[str] = dspy.OutputField(
        desc="Step-by-step reasoning process with explicit strategy references"
    )
    answer: str = dspy.OutputField(
        desc="Final answer or output produced by reasoning process"
    )

    # Confidence and metadata
    confidence: float = dspy.OutputField(
        desc="Confidence score for answer correctness (0.0 to 1.0)"
    )
    bullets_referenced: List[str] = dspy.OutputField(
        default_factory=list,
        desc="IDs of playbook bullets explicitly consulted during reasoning"
    )


class ChainOfThoughtSignature(dspy.Signature):
    """
    Internal signature for DSPy ChainOfThought reasoning.

    Used by CoTGenerator to produce structured reasoning traces.
    """

    # Input
    task_description: str = dspy.InputField(
        desc="The problem to solve"
    )
    playbook_context: str = dspy.InputField(
        desc="Relevant strategies from playbook formatted as numbered list"
    )
    domain: str = dspy.InputField(
        desc="Task domain for context"
    )

    # Output
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning process, referencing playbook strategies by ID"
    )
    answer: str = dspy.OutputField(
        desc="Final answer after reasoning"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )


__version__ = "v1.0.0"
