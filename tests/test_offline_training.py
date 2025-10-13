"""
Integration Tests for Offline Training Workflow

Tests end-to-end offline training: Dataset → Generator → Reflector → Curator
with metrics tracking, checkpoint saving, and playbook bootstrapping.

Coverage:
- T056: Integration test for offline training workflow
- T057: Test offline training components
- Dataset loading and conversion
- Batch merging workflow
- Metrics tracking
- Checkpoint and output saving
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ace.ops import (
    OfflineTrainer,
    TrainingConfig,
    DatasetLoader,
    DatasetExample
)
from ace.generator import TaskInput, TaskOutput
from ace.reflector import ReflectorInput, ReflectorOutput, InsightCandidate, InsightSection
from ace.models.playbook import PlaybookBullet, PlaybookStage


class TestDatasetLoader:
    """Test DatasetLoader functionality."""

    def test_load_from_json_valid_dataset(self, tmp_path):
        """Test loading valid custom dataset from JSON."""
        # Create test dataset
        dataset_path = tmp_path / "test_dataset.json"
        test_data = [
            {"question": "What is 2+2?", "answer": "4", "domain": "arithmetic"},
            {"question": "What is 3*5?", "answer": "15", "domain": "arithmetic"},
            {"question": "What is the capital of France?", "answer": "Paris", "domain": "geography"}
        ]

        with open(dataset_path, "w") as f:
            json.dump(test_data, f)

        # Load dataset
        loader = DatasetLoader(dataset_name="custom")
        examples = loader.load_from_json(str(dataset_path))

        assert len(examples) == 3
        assert examples[0].question == "What is 2+2?"
        assert examples[0].ground_truth == "4"
        assert examples[0].domain == "arithmetic"
        assert examples[0].task_id == "custom-0"

    def test_load_from_json_with_limit(self, tmp_path):
        """Test loading dataset with num_examples limit."""
        dataset_path = tmp_path / "test_dataset.json"
        test_data = [
            {"question": f"Question {i}", "answer": f"Answer {i}", "domain": "general"}
            for i in range(10)
        ]

        with open(dataset_path, "w") as f:
            json.dump(test_data, f)

        loader = DatasetLoader(dataset_name="custom")
        examples = loader.load_from_json(str(dataset_path), num_examples=5)

        assert len(examples) == 5

    def test_to_task_input_conversion(self):
        """Test converting DatasetExample to TaskInput."""
        example = DatasetExample(
            task_id="test-001",
            question="Calculate 6 * 7",
            ground_truth="42",
            domain="arithmetic"
        )

        loader = DatasetLoader()
        task_input, ground_truth = loader.to_task_input(
            example,
            playbook_bullets=["Break problems into steps"]
        )

        assert isinstance(task_input, TaskInput)
        assert task_input.task_id == "test-001"
        assert task_input.description == "Calculate 6 * 7"
        assert task_input.domain == "arithmetic"
        assert task_input.playbook_bullets == ["Break problems into steps"]
        assert ground_truth == "42"

    def test_stream_batches(self, tmp_path):
        """Test streaming dataset in batches."""
        dataset_path = tmp_path / "test_dataset.json"
        test_data = [
            {"question": f"Q{i}", "answer": f"A{i}", "domain": "test"}
            for i in range(25)
        ]

        with open(dataset_path, "w") as f:
            json.dump(test_data, f)

        loader = DatasetLoader(dataset_name="custom")
        # Override load method to use our test file
        loader.load = lambda split, num_examples: loader.load_from_json(str(dataset_path), num_examples)

        batches = list(loader.stream(split="train", batch_size=10))

        assert len(batches) == 3  # 10, 10, 5
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5


class TestOfflineTrainer:
    """Test OfflineTrainer functionality."""

    @patch('ace.ops.offline_trainer.CoTGenerator')
    @patch('ace.ops.offline_trainer.GroundedReflector')
    @patch('ace.ops.offline_trainer.SemanticCurator')
    def test_process_example_success(
        self,
        mock_curator,
        mock_reflector,
        mock_generator
    ):
        """Test successful processing of single example."""
        # Setup mocks
        generator_output = TaskOutput(
            task_id="test-001",
            reasoning_trace=["Step 1: Calculate", "Step 2: Verify"],
            answer="42",
            confidence=0.95,
            bullets_referenced=[]
        )
        mock_generator.return_value = Mock(return_value=generator_output)

        insights = [
            InsightCandidate(
                content="Breaking down arithmetic is effective",
                section=InsightSection.HELPFUL,
                confidence=0.9,
                rationale="Answer matched ground truth"
            )
        ]
        reflector_output = ReflectorOutput(
            task_id="test-001",
            insights=insights,
            analysis_summary="Test analysis",
            confidence_score=0.9,
            feedback_types_used=["ground_truth"]
        )
        mock_reflector.return_value = Mock(return_value=reflector_output)

        # Create trainer
        config = TrainingConfig(num_examples=1)
        trainer = OfflineTrainer(config)

        # Process example
        example = DatasetExample(
            task_id="test-001",
            question="Calculate 6 * 7",
            ground_truth="42",
            domain="arithmetic"
        )

        success, extracted_insights = trainer.process_example(example)

        assert success is True
        assert len(extracted_insights) == 1
        assert trainer.metrics.successful_generations == 1
        assert trainer.metrics.correct_answers == 1
        assert trainer.metrics.total_insights_extracted == 1
        assert trainer.metrics.helpful_insights == 1

    @patch('ace.ops.offline_trainer.CoTGenerator')
    @patch('ace.ops.offline_trainer.GroundedReflector')
    @patch('ace.ops.offline_trainer.SemanticCurator')
    def test_process_example_incorrect_answer(
        self,
        mock_curator,
        mock_reflector,
        mock_generator
    ):
        """Test processing example with incorrect answer."""
        # Setup mocks - generator produces wrong answer
        generator_output = TaskOutput(
            task_id="test-002",
            reasoning_trace=["Step 1: Assume addition", "Step 2: Calculate 6+7=13"],
            answer="13",  # Wrong (should be 42)
            confidence=0.7,
            bullets_referenced=[]
        )
        mock_generator.return_value = Mock(return_value=generator_output)

        insights = [
            InsightCandidate(
                content="Used addition instead of multiplication",
                section=InsightSection.HARMFUL,
                confidence=0.85,
                rationale="Answer did not match ground truth"
            )
        ]
        reflector_output = ReflectorOutput(
            task_id="test-002",
            insights=insights,
            analysis_summary="Test analysis",
            confidence_score=0.85,
            feedback_types_used=["ground_truth"]
        )
        mock_reflector.return_value = Mock(return_value=reflector_output)

        # Create trainer
        config = TrainingConfig(num_examples=1)
        trainer = OfflineTrainer(config)

        # Process example
        example = DatasetExample(
            task_id="test-002",
            question="Calculate 6 * 7",
            ground_truth="42",
            domain="arithmetic"
        )

        success, extracted_insights = trainer.process_example(example)

        assert success is True
        assert len(extracted_insights) == 1
        assert trainer.metrics.incorrect_answers == 1
        assert trainer.metrics.harmful_insights == 1

    @patch('ace.ops.offline_trainer.CoTGenerator')
    @patch('ace.ops.offline_trainer.GroundedReflector')
    @patch('ace.ops.offline_trainer.SemanticCurator')
    def test_batch_accumulation_and_merge(
        self,
        mock_curator_class,
        mock_reflector,
        mock_generator
    ):
        """Test accumulating insights and batch merging."""
        # Setup curator mock
        mock_curator = Mock()
        mock_curator.batch_merge.return_value = {
            "batch_results": [],
            "updated_playbook": [],
            "total_new_bullets": 2,
            "total_increments": 1,
            "total_processed": 3
        }
        mock_curator_class.return_value = mock_curator

        # Create trainer
        config = TrainingConfig(batch_merge_size=2)
        trainer = OfflineTrainer(config)

        # Accumulate insights
        insights1 = [
            InsightCandidate(
                content="Insight 1",
                section=InsightSection.HELPFUL,
                confidence=0.9,
                rationale="Test rationale 1"
            )
        ]
        insights2 = [
            InsightCandidate(
                content="Insight 2",
                section=InsightSection.HELPFUL,
                confidence=0.85,
                rationale="Test rationale 2"
            )
        ]

        trainer.accumulate_insights_for_batch("task-001", "arithmetic", insights1)
        trainer.accumulate_insights_for_batch("task-002", "arithmetic", insights2)

        assert len(trainer.batch_insights) == 2

        # Trigger merge
        trainer.merge_batch()

        assert mock_curator.batch_merge.called
        assert len(trainer.batch_insights) == 0  # Batch cleared
        assert trainer.metrics.playbook_bullets_added == 2
        assert trainer.metrics.playbook_bullets_incremented == 1

    def test_save_playbook(self, tmp_path):
        """Test saving trained playbook to file."""
        output_path = tmp_path / "test_playbook.json"

        config = TrainingConfig(output_playbook_path=str(output_path))
        trainer = OfflineTrainer(config)

        # Add test bullet
        trainer.current_playbook = [
            PlaybookBullet(
                bullet_id="B001",
                content="Test bullet",
                domain_id="arithmetic",
                section=InsightSection.HELPFUL,
                stage=PlaybookStage.SHADOW,
                embedding=[0.1] * 384,
                activation_count=1,
                confidence=0.9
            )
        ]

        trainer.save_playbook(str(output_path))

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert "bullets" in data
        assert len(data["bullets"]) == 1
        assert data["bullets"][0]["content"] == "Test bullet"
        assert "training_config" in data

    def test_save_metrics(self, tmp_path):
        """Test saving training metrics to file."""
        metrics_path = tmp_path / "test_metrics.json"

        config = TrainingConfig(output_metrics_path=str(metrics_path))
        trainer = OfflineTrainer(config)

        # Set test metrics
        trainer.metrics.total_examples = 100
        trainer.metrics.successful_generations = 95
        trainer.metrics.correct_answers = 80
        trainer.metrics.total_insights_extracted = 150
        trainer.metrics.playbook_bullets_added = 25

        trainer.save_metrics(str(metrics_path))

        assert metrics_path.exists()

        with open(metrics_path, "r") as f:
            data = json.load(f)

        assert data["training_summary"]["total_examples"] == 100
        assert data["training_summary"]["successful_generations"] == 95
        assert data["training_summary"]["accuracy"] == 0.8
        assert data["insights"]["total_extracted"] == 150
        assert data["playbook"]["bullets_added"] == 25

    def test_checkpoint_saving(self, tmp_path):
        """Test saving training checkpoints."""
        config = TrainingConfig(save_checkpoints=True)
        trainer = OfflineTrainer(config)

        trainer.metrics.total_examples = 50
        trainer.metrics.correct_answers = 40

        # Change to tmp_path for test
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            trainer.save_checkpoint("test_checkpoint")

            checkpoint_path = tmp_path / "checkpoints" / "test_checkpoint.json"
            assert checkpoint_path.exists()

            with open(checkpoint_path, "r") as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "metrics" in data
            assert data["metrics"]["total_examples"] == 50
            assert data["metrics"]["correct_answers"] == 40

        finally:
            os.chdir(original_cwd)

    @patch('ace.ops.offline_trainer.CoTGenerator')
    @patch('ace.ops.offline_trainer.GroundedReflector')
    @patch('ace.ops.offline_trainer.SemanticCurator')
    @patch('ace.ops.offline_trainer.DatasetLoader')
    def test_full_training_workflow(
        self,
        mock_loader_class,
        mock_curator_class,
        mock_reflector_class,
        mock_generator_class,
        tmp_path
    ):
        """Test complete training workflow end-to-end."""
        # Setup dataset loader mock
        mock_loader = Mock()
        test_examples = [
            DatasetExample(
                task_id=f"test-{i:03d}",
                question=f"Question {i}",
                ground_truth=f"Answer {i}",
                domain="test"
            )
            for i in range(5)
        ]
        mock_loader.load.return_value = test_examples
        mock_loader.to_task_input.side_effect = lambda ex, playbook_bullets: (
            TaskInput(
                task_id=ex.task_id,
                description=ex.question,
                domain=ex.domain,
                playbook_bullets=playbook_bullets or []
            ),
            ex.ground_truth
        )
        mock_loader_class.return_value = mock_loader

        # Setup generator mock
        mock_generator = Mock()
        def gen_side_effect(task_input):
            return TaskOutput(
                task_id=task_input.task_id,
                reasoning_trace=["Step 1", "Step 2"],
                answer=task_input.task_id.replace("test-", "Answer ").lstrip("0"),
                confidence=0.9,
                bullets_referenced=[]
            )
        mock_generator.side_effect = gen_side_effect
        mock_generator_class.return_value = mock_generator

        # Setup reflector mock
        mock_reflector = Mock()
        def refl_side_effect(refl_input):
            return ReflectorOutput(
                task_id=refl_input.task_id,
                insights=[
                    InsightCandidate(
                        content=f"Insight from {refl_input.task_id}",
                        section=InsightSection.HELPFUL,
                        confidence=0.85,
                        rationale="Test feedback"
                    )
                ],
                analysis_summary="Test analysis",
                confidence_score=0.85,
                feedback_types_used=["ground_truth"]
            )
        mock_reflector.side_effect = refl_side_effect
        mock_reflector_class.return_value = mock_reflector

        # Setup curator mock
        mock_curator = Mock()
        mock_curator.batch_merge.return_value = {
            "batch_results": [],
            "updated_playbook": [
                PlaybookBullet(
                    bullet_id="B001",
                    content="Test bullet",
                    domain_id="test",
                    section=InsightSection.HELPFUL,
                    stage=PlaybookStage.SHADOW,
                    embedding=[0.1] * 384,
                    activation_count=1,
                    confidence=0.9
                )
            ],
            "total_new_bullets": 1,
            "total_increments": 0,
            "total_processed": 5
        }
        mock_curator_class.return_value = mock_curator

        # Create trainer with test paths
        output_playbook = tmp_path / "trained_playbook.json"
        output_metrics = tmp_path / "training_metrics.json"

        config = TrainingConfig(
            num_examples=5,
            batch_merge_size=3,
            output_playbook_path=str(output_playbook),
            output_metrics_path=str(output_metrics),
            save_checkpoints=False
        )

        trainer = OfflineTrainer(config)

        # Execute training
        result = trainer.train()

        # Assertions
        assert result["success"] is True
        assert result["metrics"].total_examples == 5
        assert result["metrics"].successful_generations == 5
        assert result["metrics"].correct_answers == 5
        assert result["metrics"].total_insights_extracted == 5
        assert result["playbook_size"] == 1

        # Check outputs were saved
        assert output_playbook.exists()
        assert output_metrics.exists()


class TestTrainingMetrics:
    """Test metrics tracking functionality."""

    def test_metrics_initialization(self):
        """Test TrainingMetrics initial state."""
        config = TrainingConfig()
        trainer = OfflineTrainer(config)

        assert trainer.metrics.total_examples == 0
        assert trainer.metrics.successful_generations == 0
        assert trainer.metrics.failed_generations == 0
        assert trainer.metrics.correct_answers == 0
        assert trainer.metrics.incorrect_answers == 0
        assert trainer.metrics.total_insights_extracted == 0
        assert trainer.metrics.helpful_insights == 0
        assert trainer.metrics.harmful_insights == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
