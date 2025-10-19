"""
ACE Curator Module

Semantic deduplication and playbook management with multi-domain isolation.
"""

from ace.curator.semantic_curator import SemanticCurator
from ace.curator.curator_models import (
    CuratorInput,
    CuratorOutput,
    DeltaUpdate,
    SIMILARITY_THRESHOLD_DEFAULT,
)
from ace.curator.curator_service import CuratorService
from ace.curator.merge_coordinator import MergeCoordinator, MergeEvent

__all__ = [
    "SemanticCurator",
    "CuratorInput",
    "CuratorOutput",
    "DeltaUpdate",
    "CuratorService",
    "SIMILARITY_THRESHOLD_DEFAULT",
    "MergeCoordinator",
    "MergeEvent",
]
