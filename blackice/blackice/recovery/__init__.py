"""Recovery layer for BLACKICE 3.0.

Provides crash recovery and resume functionality:
- Checkpoints for snapshot creation
- Resume for crash recovery
- Dead letter queue for failed tasks
- Idempotency key management
"""

from blackice.recovery.checkpoint import Checkpoint, CheckpointManager
from blackice.recovery.dead_letter import DeadLetterEntry, DeadLetterQueue
from blackice.recovery.idempotency import (
    EffectType,
    IdempotencyKeyGenerator,
    IdempotencyRecord,
    IdempotencyStore,
    IdempotentExecutor,
)
from blackice.recovery.resume import ResumeManager, ResumeState

__all__ = [
    # Checkpoints
    "Checkpoint",
    "CheckpointManager",
    # Dead Letter Queue
    "DeadLetterEntry",
    "DeadLetterQueue",
    # Resume
    "ResumeManager",
    "ResumeState",
    # Idempotency
    "EffectType",
    "IdempotencyKeyGenerator",
    "IdempotencyRecord",
    "IdempotencyStore",
    "IdempotentExecutor",
]
