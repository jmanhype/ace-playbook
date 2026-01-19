"""BLACKICE schemas - Pydantic models for all domain entities.

This module exports all schema models for use throughout the BLACKICE system.
"""

from blackice.schemas.agent import (
    AGENT_PROMPTS,
    Agent,
    AgentCapabilities,
    AgentExecution,
    ConsensusResult,
    ConsensusVote,
    create_agent,
)
from blackice.schemas.event import (
    Event,
    EventLog,
    EventPayloads,
)
from blackice.schemas.receipt import (
    ArtifactHash,
    EvidenceReference,
    ProvenanceInfo,
    Receipt,
    ReceiptBuilder,
    Signature,
    VerificationInfo,
)
from blackice.schemas.run import (
    Run,
    RunConfig,
    RunSummary,
)
from blackice.schemas.task import (
    Task,
    TaskFiles,
    TaskGraph,
)
from blackice.schemas.taskspec import (
    SchemaDefinition,
    TaskSpec,
    TaskSpecRegistry,
    TaskSpecVersion,
    ValidationRule,
)
from blackice.schemas.evidence import (
    CommandOutput,
    Evidence,
    EvidenceCollection,
    EvidenceStatus,
    EvidenceType,
    LintIssue,
    LintReport,
    SecurityFinding,
    SecurityScan,
    TestReport,
    TestResult,
)

__all__ = [
    # Run
    "Run",
    "RunConfig",
    "RunSummary",
    # Task
    "Task",
    "TaskFiles",
    "TaskGraph",
    # Event
    "Event",
    "EventLog",
    "EventPayloads",
    # Agent
    "Agent",
    "AgentCapabilities",
    "AgentExecution",
    "ConsensusVote",
    "ConsensusResult",
    "AGENT_PROMPTS",
    "create_agent",
    # TaskSpec (Enterprise)
    "TaskSpec",
    "TaskSpecVersion",
    "TaskSpecRegistry",
    "SchemaDefinition",
    "ValidationRule",
    # Receipt (Enterprise)
    "Receipt",
    "ReceiptBuilder",
    "ArtifactHash",
    "EvidenceReference",
    "ProvenanceInfo",
    "VerificationInfo",
    "Signature",
    # Evidence (Enterprise)
    "Evidence",
    "EvidenceCollection",
    "EvidenceType",
    "EvidenceStatus",
    "TestReport",
    "TestResult",
    "SecurityScan",
    "SecurityFinding",
    "LintReport",
    "LintIssue",
    "CommandOutput",
]
