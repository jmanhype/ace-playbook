"""Typed data models for Curator operations and deltas."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ace.models.playbook import PlaybookBullet, PlaybookStage

# Constants
SIMILARITY_THRESHOLD_DEFAULT = 0.8
DOMAIN_ISOLATION_PATTERN = r"^[a-z0-9-]+$"
RESERVED_DOMAINS = {"system", "admin", "test"}


class InsightSection(str, Enum):
    """Permitted sections for playbook bullets."""

    HELPFUL = "Helpful"
    HARMFUL = "Harmful"
    NEUTRAL = "Neutral"


class CuratorInsight(BaseModel):
    """Structured insight emitted by the Reflector."""

    content: str = Field(..., min_length=1, description="Strategy or observation text")
    section: InsightSection = Field(..., description="Playbook section destination")
    tags: List[str] = Field(default_factory=list, description="Categorical tags for retrieval")
    bullet_id: Optional[str] = Field(
        default=None, description="Optional existing bullet identifier for edits"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    model_config = ConfigDict(extra="forbid")


class CuratorOperationType(str, Enum):
    """Enumerates curator delta operations."""

    ADD = "add"
    INCREMENT_HELPFUL = "increment_helpful"
    INCREMENT_HARMFUL = "increment_harmful"
    INCREMENT_NEUTRAL = "increment_neutral"
    QUARANTINE = "quarantine"


class CuratorOperation(BaseModel):
    """JSON-native delta contract for playbook updates."""

    type: CuratorOperationType = Field(..., description="Operation identifier")
    section: InsightSection = Field(..., description="Section affected by the operation")
    content: str = Field(..., description="Insight text associated with the operation")
    bullet_id: Optional[str] = Field(
        default=None, description="Target bullet identifier when applicable"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operation metadata")

    model_config = ConfigDict(extra="forbid")


class CuratorDelta(BaseModel):
    """Container for curator operations."""

    operations: List[CuratorOperation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def append(self, operation: CuratorOperation) -> None:
        self.operations.append(operation)


class CuratorInput(BaseModel):
    """Input contract for Curator delta merging."""

    task_id: str = Field(..., description="Identifier for the originating task")
    domain_id: str = Field(..., description="Domain namespace for the playbook")
    insights: List[CuratorInsight] = Field(..., description="Insights ready for curation")
    current_playbook: List[PlaybookBullet] = Field(
        default_factory=list, description="Existing bullets for the domain"
    )
    target_stage: PlaybookStage = Field(
        default=PlaybookStage.SHADOW, description="Stage for new bullets"
    )
    similarity_threshold: float = Field(
        default=SIMILARITY_THRESHOLD_DEFAULT,
        description="Cosine similarity threshold for duplicates",
    )
    promotion_helpful_min: int = Field(
        default=3, description="Helpful count required for staging promotion"
    )
    promotion_ratio_min: float = Field(
        default=3.0, description="Helpful:harmful ratio required for staging"
    )
    quarantine_threshold: float = Field(
        default=1.0, description="Helpful:harmful ratio below which bullets are quarantined"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, value: float) -> float:
        if not 0 < value <= 1.0:
            raise ValueError("similarity_threshold must be within (0, 1]")
        return value


class DeltaUpdate(BaseModel):
    """Single atomic playbook update operation."""

    operation: CuratorOperationType
    bullet_id: str
    before_hash: Optional[str] = None
    after_hash: Optional[str] = None
    new_bullet: Optional[PlaybookBullet] = None
    similar_to: Optional[str] = None
    similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CuratorOutput(BaseModel):
    """Output from Curator delta merging operation."""

    task_id: str
    domain_id: str
    delta_updates: List[DeltaUpdate]
    updated_playbook: List[PlaybookBullet]
    new_bullets_added: int = 0
    existing_bullets_incremented: int = 0
    duplicates_detected: int = 0
    bullets_quarantined: int = 0
    bullets_promoted: int = 0
    delta: CuratorDelta = Field(default_factory=CuratorDelta)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_summary(self) -> str:
        """Generate human-readable summary."""

        return (
            "Curator Update Summary:\n"
            f"  New bullets: {self.new_bullets_added}\n"
            f"  Incremented: {self.existing_bullets_incremented}\n"
            f"  Duplicates: {self.duplicates_detected}\n"
            f"  Quarantined: {self.bullets_quarantined}\n"
            f"  Promoted: {self.bullets_promoted}\n"
            f"  Total operations: {len(self.delta_updates)}"
        )
