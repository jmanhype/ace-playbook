"""
Online Learning Loop Implementation

Continuous adaptation workflow for production ACE deployment.
Polls for tasks, executes Generator → Reflector → Curator workflow,
and periodically checks for promotions.

Based on tasks.md T061.
"""

import time
import signal
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput
from ace.curator import CuratorService
from ace.curator.merge_coordinator import MergeCoordinator, MergeEvent
from ace.ops.stage_manager import StageManager
from ace.ops.refinement_scheduler import RefinementScheduler
from ace.runtime import RuntimeAdapter
from ace.ops.review_service import ReviewService, REVIEW_CONFIDENCE_THRESHOLD
from ace.models.playbook import PlaybookStage
from ace.utils.database import get_session
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="online_loop")


@dataclass
class OnlineLoopConfig:
    """Configuration for online learning loop."""
    generator_model: str = "gpt-4-turbo"
    reflector_model: str = "gpt-4o-mini"
    domain_id: str = "default"
    poll_interval_seconds: int = 5
    promotion_check_interval: int = 10  # Check every N tasks
    max_iterations: Optional[int] = None  # None = run forever
    use_shadow_mode: bool = True  # If True, new insights go to shadow stage
    merge_batch_size: int = 8
    merge_flush_interval: float = 5.0


@dataclass
class OnlineLoopMetrics:
    """Metrics tracked during online loop execution."""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    insights_extracted: int = 0
    insights_queued_for_review: int = 0  # T062: Track queued insights
    promotions_performed: int = 0
    quarantines_performed: int = 0
    prunes_performed: int = 0
    start_time: Optional[datetime] = None
    last_promotion_check: Optional[datetime] = None


class OnlineLearningLoop:
    """
    Continuous learning loop for production deployment.

    T061: Poll → Generate → Reflect → Curate → Promote workflow.
    """

    def __init__(self, config: OnlineLoopConfig):
        """
        Initialize OnlineLearningLoop.

        Args:
            config: OnlineLoopConfig with loop parameters
        """
        self.config = config
        self.metrics = OnlineLoopMetrics()

        # Initialize ACE components
        self.generator = CoTGenerator(model=config.generator_model)
        self.reflector = GroundedReflector(model=config.reflector_model)
        self.curator_service = CuratorService()
        self.merge_coordinator = MergeCoordinator(
            self.curator_service,
            batch_size=config.merge_batch_size,
            flush_interval=config.merge_flush_interval,
        )
        self.refinement_scheduler: Optional[RefinementScheduler] = None
        self.runtime_adapter: Optional[RuntimeAdapter] = None

        # Stage manager for promotions
        self.stage_manager: Optional[StageManager] = None

        # Review service for low-confidence insights
        self.review_service: Optional[ReviewService] = None

        # Shutdown flag
        self._should_stop = False

        # Task queue callback
        self._task_queue_callback: Optional[Callable[[], Optional[Dict[str, Any]]]] = None

        logger.info(
            "online_loop_initialized",
            generator_model=config.generator_model,
            reflector_model=config.reflector_model,
            domain_id=config.domain_id,
            use_shadow_mode=config.use_shadow_mode
        )

    def set_task_queue_callback(
        self,
        callback: Callable[[], Optional[Dict[str, Any]]]
    ) -> None:
        """
        Set callback function to poll for tasks.

        Callback should return task dict with:
        {
            "task_id": str,
            "description": str,
            "domain": str,
            "ground_truth": Optional[str],
            "test_results": Optional[str],
            "metadata": Optional[Dict]
        }

        Or None if no tasks available.

        Args:
            callback: Function returning next task or None
        """
        self._task_queue_callback = callback

    def _poll_for_task(self) -> Optional[Dict[str, Any]]:
        """
        Poll for next task from queue.

        Returns:
            Task dict or None if no tasks available
        """
        if self._task_queue_callback:
            return self._task_queue_callback()
        return None

    def _get_playbook_context(self, max_bullets: int = 40) -> tuple[list[str], list[Dict]]:
        """Retrieve active playbook bullets plus runtime entries."""
        entries: List[Dict] = []

        if self.runtime_adapter:
            entries.extend(self.runtime_adapter.get_hot_entries())

        if self.stage_manager:
            bullets = self.stage_manager.playbook_repo.get_active_playbook(
                domain_id=self.config.domain_id,
                exclude_quarantined=True
            )

            prioritized = sorted(
                bullets,
                key=lambda b: (
                    0 if b.stage == PlaybookStage.PROD else 1,
                    -b.helpful_count,
                    b.created_at
                )
            )

            for bullet in prioritized[:max_bullets]:
                entries.append(
                    {
                        "id": bullet.id,
                        "content": bullet.content,
                        "stage": bullet.stage.value,
                        "helpful_count": bullet.helpful_count,
                        "harmful_count": bullet.harmful_count,
                    }
                )

        strings = [entry["content"] for entry in entries]
        return strings, entries

    def _process_task(
        self,
        task_data: Dict[str, Any],
        session: Session
    ) -> bool:
        """
        Process single task through Generator → Reflector → Curator.

        Args:
            task_data: Task dictionary from queue
            session: Database session

        Returns:
            True if successful, False on failure
        """
        try:
            # Create TaskInput with current playbook context
            playbook_strings, structured_entries = self._get_playbook_context()
            task_input = TaskInput(
                task_id=task_data["task_id"],
                description=task_data["description"],
                domain=task_data.get("domain", self.config.domain_id),
                playbook_bullets=playbook_strings,
                playbook_context_entries=structured_entries,
                max_reasoning_steps=10
            )

            # Execute Generator
            generator_output = self.generator(task_input)

            logger.info(
                "task_generated",
                task_id=task_input.task_id,
                answer=generator_output.answer,
                confidence=generator_output.confidence
            )

            # Execute Reflector with feedback
            reflector_input = ReflectorInput(
                task_id=generator_output.task_id,
                reasoning_trace=generator_output.reasoning_trace,
                answer=generator_output.answer,
                confidence=generator_output.confidence,
                bullets_referenced=generator_output.bullets_referenced,
                ground_truth=task_data.get("ground_truth"),
                test_results=task_data.get("test_results"),
                domain=task_input.domain
            )

            reflector_output = self.reflector(reflector_input)

            logger.info(
                "task_reflected",
                task_id=task_input.task_id,
                num_insights=len(reflector_output.insights)
            )

            self.metrics.insights_extracted += len(reflector_output.insights)

            # T062: Separate high-confidence and low-confidence insights
            high_confidence_insights = []
            low_confidence_insights = []

            for insight in reflector_output.insights:
                if insight.confidence < REVIEW_CONFIDENCE_THRESHOLD:
                    low_confidence_insights.append(insight)
                else:
                    high_confidence_insights.append(insight)

            # Queue low-confidence insights for review
            if low_confidence_insights and self.review_service:
                for insight in low_confidence_insights:
                    self.review_service.queue_insight(
                        insight=insight,
                        source_task_id=task_input.task_id,
                        domain_id=self.config.domain_id
                    )
                    self.metrics.insights_queued_for_review += 1

                logger.info(
                    "insights_queued_for_review",
                    task_id=task_input.task_id,
                    count=len(low_confidence_insights)
                )

            # Merge high-confidence insights with Curator (shadow mode)
            if high_confidence_insights:
                insight_dicts = [
                    {
                        "content": ins.content,
                        "section": ins.section.value,
                        "confidence": ins.confidence,
                        "tags": ins.tags,
                        "source_task_id": task_input.task_id,
                    }
                    for ins in high_confidence_insights
                ]

                event = MergeEvent(
                    task_id=task_input.task_id,
                    insights=insight_dicts,
                    target_stage=PlaybookStage.SHADOW if self.config.use_shadow_mode else PlaybookStage.PROD,
                )

                merge_result = self.merge_coordinator.submit(self.config.domain_id, event)
                if merge_result:
                    logger.info(
                        "task_batch_merged",
                        domain_id=self.config.domain_id,
                        new_bullets=merge_result.new_bullets_added,
                        increments=merge_result.existing_bullets_incremented,
                    )
                    session.expire_all()

            self.metrics.successful_tasks += 1
            return True

        except Exception as e:
            logger.error(
                "task_processing_failed",
                task_id=task_data.get("task_id", "unknown"),
                error=str(e),
                exc_info=True
            )
            self.metrics.failed_tasks += 1
            return False

    def _run_refinement_cycle(self, session: Session) -> None:
        if not self.refinement_scheduler:
            return

        result = self.refinement_scheduler.run(self.config.domain_id)
        self.metrics.promotions_performed += result.promotions
        self.metrics.quarantines_performed += result.quarantines
        self.metrics.prunes_performed += result.pruned
        self.metrics.last_promotion_check = result.last_run

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("shutdown_signal_received", signal=signum)
            self._should_stop = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run(self) -> OnlineLoopMetrics:
        """
        Run online learning loop.

        T061: Continuous adaptation workflow.

        Returns:
            OnlineLoopMetrics with execution statistics
        """
        self._setup_signal_handlers()
        self.metrics.start_time = datetime.utcnow()

        logger.info(
            "online_loop_started",
            domain_id=self.config.domain_id,
            max_iterations=self.config.max_iterations,
            use_shadow_mode=self.config.use_shadow_mode
        )

        try:
            with get_session() as session:
                # Initialize stage manager and review service with session
                self.stage_manager = StageManager(session)
                self.review_service = ReviewService(session)
                self.refinement_scheduler = RefinementScheduler(
                    self.merge_coordinator,
                    self.stage_manager,
                    self.curator_service,
                    min_interval_seconds=self.config.merge_flush_interval,
                )
                self.runtime_adapter = RuntimeAdapter(
                    self.config.domain_id,
                    self.merge_coordinator,
                )

                while not self._should_stop:
                    # Check iteration limit
                    if self.config.max_iterations is not None:
                        if self.metrics.total_tasks_processed >= self.config.max_iterations:
                            logger.info(
                                "max_iterations_reached",
                                max_iterations=self.config.max_iterations
                            )
                            break

                    # Poll for task
                    task_data = self._poll_for_task()

                    if task_data is None:
                        # No task available, wait and retry
                        time.sleep(self.config.poll_interval_seconds)
                        continue

                    # Process task
                    self.metrics.total_tasks_processed += 1
                    self._process_task(task_data, session)

                    # Periodic promotion check
                    if (
                        self.metrics.total_tasks_processed % self.config.promotion_check_interval == 0
                    ):
                        self._run_refinement_cycle(session)

                    # Commit session after each task
                    session.commit()

                # Ensure a final refinement pass before exiting context
                self._run_refinement_cycle(session)

        except Exception as e:
            logger.error(
                "online_loop_error",
                error=str(e),
                exc_info=True
            )
            raise
        finally:
            flush_results = self.merge_coordinator.flush_all()
            if flush_results:
                logger.info(
                    "merge_coordinator_drain",
                    batches=len(flush_results)
                )
            logger.info(
                "online_loop_stopped",
                total_tasks=self.metrics.total_tasks_processed,
                successful=self.metrics.successful_tasks,
                failed=self.metrics.failed_tasks,
                insights_extracted=self.metrics.insights_extracted,
                insights_queued_for_review=self.metrics.insights_queued_for_review,
                promotions=self.metrics.promotions_performed,
                quarantines=self.metrics.quarantines_performed,
                prunes=self.metrics.prunes_performed
            )

        return self.metrics


def create_online_loop(
    generator_model: str = "gpt-4-turbo",
    reflector_model: str = "gpt-4o-mini",
    domain_id: str = "default",
    use_shadow_mode: bool = True,
    max_iterations: Optional[int] = None
) -> OnlineLearningLoop:
    """
    Factory function to create OnlineLearningLoop.

    Args:
        generator_model: Model for Generator
        reflector_model: Model for Reflector
        domain_id: Domain namespace
        use_shadow_mode: If True, create bullets in shadow stage
        max_iterations: Max iterations (None = run forever)

    Returns:
        OnlineLearningLoop instance
    """
    config = OnlineLoopConfig(
        generator_model=generator_model,
        reflector_model=reflector_model,
        domain_id=domain_id,
        use_shadow_mode=use_shadow_mode,
        max_iterations=max_iterations
    )

    return OnlineLearningLoop(config)


__version__ = "v1.0.0"
