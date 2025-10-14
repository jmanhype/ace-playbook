"""
ACE (Adaptive Code Evolution) Framework

Self-improving LLM system using Generator-Reflector-Curator pattern.
"""

__version__ = "1.14.0"

from ace.models import (
    Base,
    Task,
    TaskOutput,
    Reflection,
    InsightCandidate,
    PlaybookBullet,
    PlaybookStage,
    DiffJournalEntry,
    MergeOperation,
)

__all__ = [
    "__version__",
    "Base",
    "Task",
    "TaskOutput",
    "Reflection",
    "InsightCandidate",
    "PlaybookBullet",
    "PlaybookStage",
    "DiffJournalEntry",
    "MergeOperation",
]
