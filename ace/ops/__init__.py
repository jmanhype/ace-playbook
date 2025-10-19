"""
Operations Module

Provides operational tooling for offline training, online learning,
stage management, review queue, and batch processing workflows.

Components:
- DatasetLoader: Load and parse datasets (GSM8K, custom JSON)
- OfflineTrainer: Orchestrate offline training workflow
- OnlineLearningLoop: Continuous adaptation for production
- StageManager: Shadow/Staging/Production promotion gates
- ReviewService: Human review queue for low-confidence insights
- TrainingConfig/TrainingMetrics: Configuration and metrics tracking

Usage:
    from ace.ops import create_offline_trainer, create_online_loop

    # Offline training
    trainer = create_offline_trainer(
        dataset_name="gsm8k",
        num_examples=100
    )
    result = trainer.train()

    # Online learning
    loop = create_online_loop(
        domain_id="production",
        use_shadow_mode=True
    )
    loop.run()

    # Review queue
    review_service = create_review_service(session)
    pending = review_service.list_pending()
"""

from ace.ops.dataset_loader import (
    DatasetLoader,
    DatasetExample,
    create_dataset_loader
)
from ace.ops.offline_trainer import (
    OfflineTrainer,
    TrainingConfig,
    TrainingMetrics,
    create_offline_trainer
)
from ace.ops.online_loop import (
    OnlineLearningLoop,
    OnlineLoopConfig,
    OnlineLoopMetrics,
    create_online_loop
)
from ace.ops.refinement_scheduler import RefinementScheduler, RefinementResult
from ace.ops.stage_manager import (
    StageManager,
    create_stage_manager
)
from ace.ops.review_service import (
    ReviewService,
    create_review_service,
    REVIEW_CONFIDENCE_THRESHOLD
)
from ace.ops.metrics import (
    MetricsCollector,
    get_metrics_collector,
    LatencyTimer
)
from ace.ops.guardrails import (
    GuardrailMonitor,
    PerformanceSnapshot,
    RollbackTrigger,
    create_guardrail_monitor
)

__all__ = [
    "DatasetLoader",
    "DatasetExample",
    "create_dataset_loader",
    "OfflineTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "create_offline_trainer",
    "OnlineLearningLoop",
    "OnlineLoopConfig",
    "OnlineLoopMetrics",
    "create_online_loop",
    "RefinementScheduler",
    "RefinementResult",
    "StageManager",
    "create_stage_manager",
    "ReviewService",
    "create_review_service",
    "REVIEW_CONFIDENCE_THRESHOLD",
    "MetricsCollector",
    "get_metrics_collector",
    "LatencyTimer",
    "GuardrailMonitor",
    "PerformanceSnapshot",
    "RollbackTrigger",
    "create_guardrail_monitor",
]

__version__ = "v1.2.0"
