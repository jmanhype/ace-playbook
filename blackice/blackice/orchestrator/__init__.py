"""Orchestrator layer for BLACKICE 3.0.

Manages run lifecycle through the state machine and phase handlers.
"""

from blackice.orchestrator.phases import (
    ImplementPhase,
    PhaseContext,
    PhaseHandler,
    PhaseOrchestrator,
    PhaseResult,
    PlanPhase,
    TestPhase,
    VerifyPhase,
)
from blackice.orchestrator.state_machine import (
    RunContext,
    RunState,
    RunStateMachine,
    StateTransition,
    TRANSITIONS,
)

__all__ = [
    # State machine
    "RunStateMachine",
    "RunState",
    "RunContext",
    "StateTransition",
    "TRANSITIONS",
    # Phases
    "PhaseOrchestrator",
    "PhaseHandler",
    "PhaseContext",
    "PhaseResult",
    "PlanPhase",
    "ImplementPhase",
    "TestPhase",
    "VerifyPhase",
]
