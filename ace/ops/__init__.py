"""
Operations Module

Provides operational tooling for offline training, dataset loading,
and batch processing workflows.

Components:
- DatasetLoader: Load and parse datasets (GSM8K, custom JSON)
- OfflineTrainer: Orchestrate offline training workflow
- TrainingConfig/TrainingMetrics: Configuration and metrics tracking

Usage:
    from ace.ops import create_offline_trainer

    trainer = create_offline_trainer(
        dataset_name="gsm8k",
        num_examples=100,
        generator_model="gpt-4-turbo",
        reflector_model="gpt-4o-mini"
    )

    result = trainer.train()
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

__all__ = [
    "DatasetLoader",
    "DatasetExample",
    "create_dataset_loader",
    "OfflineTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "create_offline_trainer",
]

__version__ = "v1.0.0"
