"""Reflexion layer for BLACKICE 3.0.

Implements the Ralph loop (try-fail-reflect-learn-retry) and
evaluation/repair capabilities.
"""

from blackice.reflexion.evaluator import (
    Evaluator,
    EvaluationResult,
    PytestRunner,
    RepairAnalyzer,
    RepairSuggestion,
    TestResult,
    TestRunner,
    TestStatus,
    TestSuiteResult,
    VerificationLevel,
)
from blackice.reflexion.ralph_loop import (
    Attempt,
    LLMReflectionProvider,
    LoopConfig,
    LoopPhase,
    LoopResult,
    RalphLoop,
    ReflectionProvider,
    with_ralph_loop,
)

__all__ = [
    # Ralph Loop
    "RalphLoop",
    "LoopConfig",
    "LoopPhase",
    "LoopResult",
    "Attempt",
    "ReflectionProvider",
    "LLMReflectionProvider",
    "with_ralph_loop",
    # Evaluator
    "Evaluator",
    "EvaluationResult",
    "TestRunner",
    "PytestRunner",
    "TestResult",
    "TestSuiteResult",
    "TestStatus",
    "VerificationLevel",
    "RepairAnalyzer",
    "RepairSuggestion",
]
