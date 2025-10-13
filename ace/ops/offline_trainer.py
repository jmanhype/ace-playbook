"""
Offline Training Runner Implementation

Orchestrates batch training workflow: Dataset → Generator → Reflector → Curator
for bootstrapping playbooks from offline datasets before production deployment.

Based on tasks.md T052.
"""

import json
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

from ace.ops.dataset_loader import DatasetLoader, DatasetExample
from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput, InsightCandidate
from ace.curator import SemanticCurator
from ace.models.playbook import PlaybookBullet, PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="offline_trainer")


@dataclass
class TrainingMetrics:
    """Metrics collected during offline training."""
    total_examples: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    total_insights_extracted: int = 0
    helpful_insights: int = 0
    harmful_insights: int = 0
    neutral_insights: int = 0
    playbook_bullets_added: int = 0
    playbook_bullets_incremented: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for offline training run."""
    dataset_name: str = "gsm8k"
    dataset_split: str = "train"
    num_examples: Optional[int] = 100
    generator_model: str = "gpt-4-turbo"
    reflector_model: str = "gpt-4o-mini"
    batch_merge_size: int = 10  # Batch size for curator.batch_merge()
    output_playbook_path: str = "playbooks/offline_trained.json"
    output_metrics_path: str = "metrics/offline_training_metrics.json"
    similarity_threshold: float = 0.8
    target_stage: PlaybookStage = PlaybookStage.SHADOW
    save_checkpoints: bool = True
    checkpoint_interval: int = 50  # Save checkpoint every N examples


class OfflineTrainer:
    """
    Offline Training Runner for bootstrapping playbooks.

    T052: Orchestrates Dataset → Generator → Reflector → Curator workflow
    for batch processing of offline datasets.

    Workflow:
    1. Load dataset examples (GSM8K or custom)
    2. For each example:
        - Generate reasoning with CoTGenerator
        - Reflect on outcome with GroundedReflector
        - Collect insights for batch merging
    3. Batch merge insights with SemanticCurator
    4. Track metrics and save playbook
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize OfflineTrainer.

        Args:
            config: TrainingConfig with training parameters
        """
        self.config = config
        self.metrics = TrainingMetrics()

        # Initialize components
        self.dataset_loader = DatasetLoader(dataset_name=config.dataset_name)
        self.generator = CoTGenerator(model=config.generator_model)
        self.reflector = GroundedReflector(model=config.reflector_model)
        self.curator = SemanticCurator()

        # Current playbook state
        self.current_playbook: List[PlaybookBullet] = []

        # Batch accumulator for insights
        self.batch_insights: List[Dict] = []

        logger.info(
            "offline_trainer_initialized",
            dataset=config.dataset_name,
            num_examples=config.num_examples,
            generator_model=config.generator_model,
            reflector_model=config.reflector_model,
            batch_size=config.batch_merge_size
        )

    def load_initial_playbook(self, playbook_path: Optional[str] = None) -> None:
        """
        Load existing playbook to continue training.

        Args:
            playbook_path: Path to existing playbook JSON
        """
        if not playbook_path:
            logger.info("starting_with_empty_playbook")
            self.current_playbook = []
            return

        path = Path(playbook_path)
        if not path.exists():
            logger.warning("playbook_not_found", path=playbook_path)
            self.current_playbook = []
            return

        with open(path, "r", encoding="utf-8") as f:
            playbook_data = json.load(f)

        # Convert JSON to PlaybookBullet objects
        self.current_playbook = [
            PlaybookBullet(**bullet) for bullet in playbook_data.get("bullets", [])
        ]

        logger.info(
            "initial_playbook_loaded",
            path=playbook_path,
            num_bullets=len(self.current_playbook)
        )

    def process_example(
        self,
        example: DatasetExample,
        playbook_bullets: Optional[List[str]] = None
    ) -> Tuple[bool, List[InsightCandidate]]:
        """
        Process single dataset example through Generator → Reflector.

        Args:
            example: DatasetExample to process
            playbook_bullets: Optional playbook bullets to inject

        Returns:
            Tuple of (success: bool, insights: List[InsightCandidate])
        """
        try:
            # Convert to TaskInput
            task_input, ground_truth = self.dataset_loader.to_task_input(
                example,
                playbook_bullets=playbook_bullets
            )

            # Execute Generator
            generator_output = self.generator(task_input)

            self.metrics.successful_generations += 1

            # Check correctness
            is_correct = generator_output.answer.strip() == ground_truth.strip()
            if is_correct:
                self.metrics.correct_answers += 1
            else:
                self.metrics.incorrect_answers += 1

            # Execute Reflector with ground truth
            reflector_input = ReflectorInput(
                task_id=generator_output.task_id,
                reasoning_trace=generator_output.reasoning_trace,
                answer=generator_output.answer,
                confidence=generator_output.confidence,
                bullets_referenced=generator_output.bullets_referenced,
                ground_truth=ground_truth,
                domain=example.domain
            )

            reflector_output = self.reflector(reflector_input)

            # Collect insights
            insights = reflector_output.insights
            self.metrics.total_insights_extracted += len(insights)

            for insight in insights:
                if insight.section.value == "helpful":
                    self.metrics.helpful_insights += 1
                elif insight.section.value == "harmful":
                    self.metrics.harmful_insights += 1
                else:
                    self.metrics.neutral_insights += 1

            logger.info(
                "example_processed",
                task_id=example.task_id,
                is_correct=is_correct,
                num_insights=len(insights),
                confidence=generator_output.confidence
            )

            return True, insights

        except Exception as e:
            logger.error(
                "example_processing_failed",
                task_id=example.task_id,
                error=str(e)
            )
            self.metrics.failed_generations += 1
            return False, []

    def accumulate_insights_for_batch(
        self,
        task_id: str,
        domain_id: str,
        insights: List[InsightCandidate]
    ) -> None:
        """
        Accumulate insights for batch merging.

        Args:
            task_id: Task identifier
            domain_id: Domain identifier
            insights: Extracted insights
        """
        if not insights:
            return

        # Convert InsightCandidate to dict format for curator
        insight_dicts = [
            {
                "content": ins.content,
                "section": ins.section,
                "confidence": ins.confidence,
                "source_task_id": task_id
            }
            for ins in insights
        ]

        self.batch_insights.append({
            "task_id": task_id,
            "domain_id": domain_id,
            "insights": insight_dicts
        })

    def merge_batch(self) -> None:
        """
        Merge accumulated insights using batch_merge().
        """
        if not self.batch_insights:
            logger.info("no_insights_to_merge")
            return

        logger.info(
            "merging_batch",
            batch_size=len(self.batch_insights),
            total_insights=sum(len(t["insights"]) for t in self.batch_insights)
        )

        # Execute batch merge
        merge_result = self.curator.batch_merge(
            task_insights=self.batch_insights,
            current_playbook=self.current_playbook,
            target_stage=self.config.target_stage,
            similarity_threshold=self.config.similarity_threshold
        )

        # Update playbook
        self.current_playbook = merge_result["updated_playbook"]

        # Update metrics
        self.metrics.playbook_bullets_added += merge_result["total_new_bullets"]
        self.metrics.playbook_bullets_incremented += merge_result["total_increments"]

        logger.info(
            "batch_merged",
            new_bullets=merge_result["total_new_bullets"],
            increments=merge_result["total_increments"],
            total_playbook_size=len(self.current_playbook)
        )

        # Clear batch
        self.batch_insights = []

    def save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save training checkpoint.

        Args:
            checkpoint_name: Name for checkpoint file
        """
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.json"

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "dataset_name": self.config.dataset_name,
                "num_examples": self.config.num_examples,
                "generator_model": self.config.generator_model,
                "reflector_model": self.config.reflector_model
            },
            "metrics": {
                "total_examples": self.metrics.total_examples,
                "correct_answers": self.metrics.correct_answers,
                "incorrect_answers": self.metrics.incorrect_answers,
                "playbook_size": len(self.current_playbook)
            },
            "playbook": [bullet.to_dict() for bullet in self.current_playbook]
        }

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info("checkpoint_saved", path=str(checkpoint_path))

    def save_playbook(self, output_path: str) -> None:
        """
        Save final trained playbook.

        Args:
            output_path: Path to save playbook JSON
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        playbook_data = {
            "version": "v1.0.0",
            "created_at": datetime.now().isoformat(),
            "training_config": {
                "dataset": self.config.dataset_name,
                "num_examples": self.metrics.total_examples,
                "generator_model": self.config.generator_model,
                "reflector_model": self.config.reflector_model
            },
            "bullets": [bullet.to_dict() for bullet in self.current_playbook]
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(playbook_data, f, indent=2)

        logger.info("playbook_saved", path=str(output_file), size=len(self.current_playbook))

    def save_metrics(self, output_path: str) -> None:
        """
        Save training metrics.

        Args:
            output_path: Path to save metrics JSON
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "training_summary": {
                "total_examples": self.metrics.total_examples,
                "successful_generations": self.metrics.successful_generations,
                "failed_generations": self.metrics.failed_generations,
                "accuracy": (
                    self.metrics.correct_answers / self.metrics.total_examples
                    if self.metrics.total_examples > 0 else 0.0
                )
            },
            "insights": {
                "total_extracted": self.metrics.total_insights_extracted,
                "helpful": self.metrics.helpful_insights,
                "harmful": self.metrics.harmful_insights,
                "neutral": self.metrics.neutral_insights
            },
            "playbook": {
                "bullets_added": self.metrics.playbook_bullets_added,
                "bullets_incremented": self.metrics.playbook_bullets_incremented,
                "final_size": len(self.current_playbook)
            },
            "timing": {
                "start_time": self.metrics.start_time,
                "end_time": self.metrics.end_time,
                "duration_seconds": self.metrics.duration_seconds
            }
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        logger.info("metrics_saved", path=str(output_file))

    def train(self, initial_playbook_path: Optional[str] = None) -> Dict:
        """
        Execute offline training workflow.

        Args:
            initial_playbook_path: Optional path to existing playbook

        Returns:
            Dict with training results
        """
        self.metrics.start_time = datetime.now().isoformat()
        start_timestamp = datetime.now()

        logger.info(
            "training_started",
            dataset=self.config.dataset_name,
            num_examples=self.config.num_examples
        )

        # Load initial playbook if provided
        self.load_initial_playbook(initial_playbook_path)

        # Load dataset
        examples = self.dataset_loader.load(
            split=self.config.dataset_split,
            num_examples=self.config.num_examples
        )

        self.metrics.total_examples = len(examples)

        # Extract current playbook bullets as strings for injection
        playbook_bullets = [bullet.content for bullet in self.current_playbook]

        # Process examples with progress bar
        for idx, example in enumerate(tqdm(examples, desc="Processing examples")):
            # Process example
            success, insights = self.process_example(example, playbook_bullets)

            if success and insights:
                # Accumulate for batch merge
                self.accumulate_insights_for_batch(
                    task_id=example.task_id,
                    domain_id=example.domain,
                    insights=insights
                )

            # Batch merge at intervals
            if len(self.batch_insights) >= self.config.batch_merge_size:
                self.merge_batch()
                # Update playbook bullets after merge
                playbook_bullets = [bullet.content for bullet in self.current_playbook]

            # Save checkpoint at intervals
            if self.config.save_checkpoints and (idx + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{idx+1}")

        # Final batch merge for remaining insights
        if self.batch_insights:
            self.merge_batch()

        # Calculate timing
        self.metrics.end_time = datetime.now().isoformat()
        self.metrics.duration_seconds = (datetime.now() - start_timestamp).total_seconds()

        # Save outputs
        self.save_playbook(self.config.output_playbook_path)
        self.save_metrics(self.config.output_metrics_path)

        logger.info(
            "training_completed",
            total_examples=self.metrics.total_examples,
            accuracy=self.metrics.correct_answers / self.metrics.total_examples,
            playbook_size=len(self.current_playbook),
            duration_seconds=self.metrics.duration_seconds
        )

        return {
            "success": True,
            "metrics": self.metrics,
            "playbook_size": len(self.current_playbook),
            "output_paths": {
                "playbook": self.config.output_playbook_path,
                "metrics": self.config.output_metrics_path
            }
        }


def create_offline_trainer(
    dataset_name: str = "gsm8k",
    num_examples: int = 100,
    generator_model: str = "gpt-4-turbo",
    reflector_model: str = "gpt-4o-mini",
    output_playbook_path: str = "playbooks/offline_trained.json"
) -> OfflineTrainer:
    """
    Factory function to create OfflineTrainer.

    Args:
        dataset_name: Name of dataset (gsm8k, custom)
        num_examples: Number of examples to process
        generator_model: Model for Generator
        reflector_model: Model for Reflector
        output_playbook_path: Path to save trained playbook

    Returns:
        OfflineTrainer instance
    """
    config = TrainingConfig(
        dataset_name=dataset_name,
        num_examples=num_examples,
        generator_model=generator_model,
        reflector_model=reflector_model,
        output_playbook_path=output_playbook_path
    )

    return OfflineTrainer(config)


__version__ = "v1.0.0"
