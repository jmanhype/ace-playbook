"""
Reflector Module

Analyzes task outcomes using execution feedback to extract labeled insights
(Helpful/Harmful/Neutral) without manual annotation.

Components:
- ReflectorInput/ReflectorOutput: DSPy signatures for reflector I/O
- InsightCandidate: Pydantic model for extracted insights
- GroundedReflector: Main reflector implementation with ground-truth comparison
- FeedbackType/InsightSection: Enums for feedback signals and labels

Usage:
    from ace.reflector import GroundedReflector, ReflectorInput

    reflector = GroundedReflector(model="gpt-4o-mini")

    reflector_input = ReflectorInput(
        task_id="task-001",
        reasoning_trace=["Step 1: ...", "Step 2: ..."],
        answer="42",
        confidence=0.95,
        ground_truth="42",
        test_results='{"test_addition": true, "test_multiplication": true}'
    )

    output = reflector(reflector_input)
    print(output.insights)
"""

from ace.reflector.signatures import (
    ReflectorInput,
    FeedbackType,
    InsightSection,
    AnalysisSignature
)
from ace.reflector.grounded_reflector import (
    GroundedReflector,
    InsightCandidate,
    ReflectorOutput,
    create_grounded_reflector
)

__all__ = [
    "ReflectorInput",
    "FeedbackType",
    "InsightSection",
    "AnalysisSignature",
    "GroundedReflector",
    "InsightCandidate",
    "ReflectorOutput",
    "create_grounded_reflector",
]

__version__ = "v1.0.0"
