"""
Task and TaskOutput Models

Maps to Task and TaskOutput entities from data-model.md.
"""

from sqlalchemy import Column, String, Text, Float, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ace.models.base import Base


class Task(Base):
    """
    Task entity - Input task for Generator execution.

    Schema defined in data-model.md lines 28-40.
    """

    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    domain_id = Column(
        String(64), nullable=False, index=True, comment="Multi-tenant namespace (CHK077-CHK078)"
    )
    prompt = Column(Text, nullable=False, comment="Task description or question")
    domain = Column(String(64), nullable=False, index=True, comment="Task category")
    ground_truth = Column(Text, nullable=True, comment="Known correct answer (optional)")
    metadata_json = Column(JSON, nullable=True, comment="Additional task context")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    outputs = relationship("TaskOutput", back_populates="task", cascade="all, delete-orphan")


class TaskOutput(Base):
    """
    TaskOutput entity - Generator execution results.

    Schema defined in data-model.md lines 42-59.
    """

    __tablename__ = "task_outputs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False, index=True)
    reasoning_trace = Column(
        JSON, nullable=False, comment="List[str] - step-by-step reasoning process"
    )
    answer = Column(Text, nullable=False, comment="Final answer from Generator")
    confidence = Column(Float, nullable=False, comment="Generator confidence (0.0-1.0)")
    bullets_referenced = Column(
        JSON, nullable=False, comment="List[str] - playbook bullet IDs consulted"
    )
    latency_ms = Column(Integer, nullable=False, comment="End-to-end execution time")
    token_count = Column(Integer, nullable=False, comment="LLM tokens consumed")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Test/feedback signals (optional)
    test_results = Column(JSON, nullable=True, comment="Dict[str, bool] - test pass/fail")
    error_messages = Column(JSON, nullable=True, comment="List[str] - runtime errors")
    performance_metrics = Column(JSON, nullable=True, comment="Dict[str, float] - metrics")
    environment_feedback = Column(JSON, nullable=True, comment="Dict - external system responses")

    # Relationships
    task = relationship("Task", back_populates="outputs")
