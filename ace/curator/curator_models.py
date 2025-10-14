"""
Data models for Curator operations.

Extracted from semantic_curator.py to reduce file size and improve modularity.
"""

from typing import List, Dict, Optional
from datetime import datetime

from ace.models.playbook import PlaybookBullet, PlaybookStage

# Constants
SIMILARITY_THRESHOLD_DEFAULT = 0.8
DOMAIN_ISOLATION_PATTERN = r"^[a-z0-9-]+$"
RESERVED_DOMAINS = {"system", "admin", "test"}


class CuratorInput:
    """Input for Curator delta merging operation."""

    def __init__(
        self,
        task_id: str,
        domain_id: str,
        insights: List[Dict],
        current_playbook: List[PlaybookBullet],
        target_stage: PlaybookStage = PlaybookStage.SHADOW,
        similarity_threshold: float = SIMILARITY_THRESHOLD_DEFAULT,
        promotion_helpful_min: int = 3,
        promotion_ratio_min: float = 3.0,
        quarantine_threshold: float = 1.0,
    ):
        self.task_id = task_id
        self.domain_id = domain_id
        self.insights = insights
        self.current_playbook = current_playbook
        self.target_stage = target_stage
        self.similarity_threshold = similarity_threshold
        self.promotion_helpful_min = promotion_helpful_min
        self.promotion_ratio_min = promotion_ratio_min
        self.quarantine_threshold = quarantine_threshold


class DeltaUpdate:
    """Single atomic playbook update operation."""

    def __init__(
        self,
        operation: str,
        bullet_id: str,
        before_hash: Optional[str] = None,
        after_hash: Optional[str] = None,
        new_bullet: Optional[PlaybookBullet] = None,
        similar_to: Optional[str] = None,
        similarity_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        self.operation = operation
        self.bullet_id = bullet_id
        self.before_hash = before_hash
        self.after_hash = after_hash
        self.new_bullet = new_bullet
        self.similar_to = similar_to
        self.similarity_score = similarity_score
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class CuratorOutput:
    """Output from Curator delta merging operation."""

    def __init__(
        self,
        task_id: str,
        domain_id: str,
        delta_updates: List[DeltaUpdate],
        updated_playbook: List[PlaybookBullet],
    ):
        self.task_id = task_id
        self.domain_id = domain_id
        self.delta_updates = delta_updates
        self.updated_playbook = updated_playbook
        self.new_bullets_added = 0
        self.existing_bullets_incremented = 0
        self.duplicates_detected = 0
        self.bullets_quarantined = 0
        self.bullets_promoted = 0

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Curator Update Summary:\n"
            f"  New bullets: {self.new_bullets_added}\n"
            f"  Incremented: {self.existing_bullets_incremented}\n"
            f"  Duplicates: {self.duplicates_detected}\n"
            f"  Quarantined: {self.bullets_quarantined}\n"
            f"  Promoted: {self.bullets_promoted}\n"
            f"  Total operations: {len(self.delta_updates)}"
        )
