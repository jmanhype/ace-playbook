"""
Reflector DSPy Signatures

Defines input/output interfaces for Reflector modules that analyze task outcomes
and extract labeled insights from execution feedback.

Based on contracts/reflector.py.
Version: v1.0.0
"""

import dspy
from typing import List, Optional, Dict, Any
from enum import Enum


class FeedbackType(str, Enum):
    """Types of execution feedback signals."""
    GROUND_TRUTH = "ground_truth"
    TEST_RESULT = "test_result"
    ERROR = "error"
    PERFORMANCE = "performance"
    ENVIRONMENT = "environment"


class InsightSection(str, Enum):
    """Playbook bullet section labels."""
    HELPFUL = "Helpful"
    HARMFUL = "Harmful"
    NEUTRAL = "Neutral"


class ReflectorInput(dspy.Signature):
    """
    Input signature for Reflector outcome analysis.

    Combines TaskOutput from Generator with ExecutionFeedback to enable
    learning from objective signals rather than subjective labels.
    """

    # Required: Generator output to analyze
    task_id: str = dspy.InputField(
        desc="Task identifier linking to original Task and TaskOutput"
    )
    reasoning_trace: List[str] = dspy.InputField(
        desc="Step-by-step reasoning process from Generator"
    )
    answer: str = dspy.InputField(
        desc="Final answer produced by Generator"
    )
    confidence: float = dspy.InputField(
        desc="Generator's confidence in answer correctness (0.0 to 1.0)"
    )
    bullets_referenced: List[str] = dspy.InputField(
        default_factory=list,
        desc="IDs of playbook bullets consulted during reasoning"
    )

    # Optional: Execution feedback signals
    ground_truth: str = dspy.InputField(
        default="",
        desc="Known correct answer for correctness evaluation"
    )
    test_results: str = dspy.InputField(
        default="",
        desc="Test pass/fail outcomes as JSON string"
    )
    error_messages: List[str] = dspy.InputField(
        default_factory=list,
        desc="Runtime errors, exceptions, or failure messages"
    )
    performance_metrics: str = dspy.InputField(
        default="",
        desc="Performance data as JSON string"
    )

    # Metadata for context
    domain: str = dspy.InputField(
        default="general",
        desc="Task domain (arithmetic, code_gen, qa, agent_workflow)"
    )

    # T022: ReAct-specific fields (optional, only present for tool-calling tasks)
    structured_trace: List[Any] = dspy.InputField(
        default_factory=list,
        desc="ReAct structured trace with tool calls (List[ReasoningStep])"
    )
    tools_used: List[str] = dspy.InputField(
        default_factory=list,
        desc="List of tool names used during execution"
    )
    total_iterations: int = dspy.InputField(
        default=0,
        desc="Total ReAct iterations performed"
    )
    iteration_limit_reached: bool = dspy.InputField(
        default=False,
        desc="Whether max iterations was hit without completion"
    )


class AnalysisSignature(dspy.Signature):
    """
    Internal signature for DSPy reflection analysis.

    Used by GroundedReflector to produce structured analysis.
    """

    # Input
    task_context: str = dspy.InputField(
        desc="Task description with reasoning trace and answer"
    )
    feedback_context: str = dspy.InputField(
        desc="Execution feedback: ground truth, test results, errors"
    )
    domain: str = dspy.InputField(
        desc="Task domain for context"
    )

    # Output
    analysis: str = dspy.OutputField(
        desc="Detailed analysis of task outcome with insights"
    )
    helpful_insights: str = dspy.OutputField(
        desc="Strategies that contributed to success (one per line)"
    )
    harmful_insights: str = dspy.OutputField(
        desc="Strategies that caused failures or errors (one per line)"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in analysis quality (0.0 to 1.0)"
    )


__version__ = "v1.0.0"
