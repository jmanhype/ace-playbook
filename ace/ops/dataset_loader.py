"""
Dataset Loader Implementation

Loads and parses datasets for offline training. Supports GSM8K and custom formats.

Based on tasks.md T053.
"""

import json
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path

from ace.generator import TaskInput
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="dataset_loader")


@dataclass
class DatasetExample:
    """Single example from a dataset."""
    task_id: str
    question: str
    ground_truth: str
    domain: str = "general"
    metadata: Optional[Dict] = None


class DatasetLoader:
    """
    Dataset loader for offline training.

    T053: Load and parse datasets with train/validation splits.
    """

    def __init__(self, dataset_name: str = "gsm8k", cache_dir: Optional[str] = None):
        """
        Initialize DatasetLoader.

        Args:
            dataset_name: Name of dataset (gsm8k, custom)
            cache_dir: Optional cache directory for HuggingFace datasets
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self._dataset = None

        logger.info(
            "dataset_loader_initialized",
            dataset_name=dataset_name,
            cache_dir=cache_dir
        )

    def load_gsm8k(self, split: str = "train", num_examples: Optional[int] = None) -> List[DatasetExample]:
        """
        Load GSM8K dataset from HuggingFace.

        Args:
            split: Dataset split (train, test)
            num_examples: Limit number of examples (None = all)

        Returns:
            List of DatasetExample objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "datasets library not installed. Install with: pip install datasets"
            )

        logger.info(
            "loading_gsm8k",
            split=split,
            num_examples=num_examples or "all"
        )

        # Load dataset
        dataset = load_dataset("gsm8k", "main", split=split, cache_dir=self.cache_dir)

        examples = []
        for idx, item in enumerate(dataset):
            if num_examples and idx >= num_examples:
                break

            # GSM8K format: {"question": str, "answer": str}
            # Answer format: "#### final_answer"
            question = item["question"]
            answer = item["answer"]

            # Extract numeric answer from "#### 42" format
            if "####" in answer:
                ground_truth = answer.split("####")[1].strip()
            else:
                ground_truth = answer.strip()

            example = DatasetExample(
                task_id=f"gsm8k-{split}-{idx}",
                question=question,
                ground_truth=ground_truth,
                domain="arithmetic",
                metadata={"dataset": "gsm8k", "split": split, "index": idx}
            )
            examples.append(example)

        logger.info(
            "gsm8k_loaded",
            split=split,
            num_examples=len(examples)
        )

        return examples

    def load_from_json(self, file_path: str, num_examples: Optional[int] = None) -> List[DatasetExample]:
        """
        Load custom dataset from JSON file.

        Expected format:
        [
            {
                "question": "What is 2+2?",
                "answer": "4",
                "domain": "arithmetic"  # optional
            },
            ...
        ]

        Args:
            file_path: Path to JSON file
            num_examples: Limit number of examples

        Returns:
            List of DatasetExample objects
        """
        logger.info(
            "loading_json_dataset",
            file_path=file_path,
            num_examples=num_examples or "all"
        )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of examples")

        examples = []
        for idx, item in enumerate(data):
            if num_examples and idx >= num_examples:
                break

            if "question" not in item or "answer" not in item:
                logger.warning(
                    "skipping_invalid_example",
                    index=idx,
                    reason="missing question or answer"
                )
                continue

            example = DatasetExample(
                task_id=f"custom-{idx}",
                question=item["question"],
                ground_truth=item["answer"],
                domain=item.get("domain", "general"),
                metadata={"source": file_path, "index": idx}
            )
            examples.append(example)

        logger.info(
            "json_dataset_loaded",
            file_path=file_path,
            num_examples=len(examples)
        )

        return examples

    def to_task_input(self, example: DatasetExample, playbook_bullets: Optional[List[str]] = None) -> Tuple[TaskInput, str]:
        """
        Convert DatasetExample to TaskInput for Generator.

        Args:
            example: DatasetExample to convert
            playbook_bullets: Optional playbook bullets to inject

        Returns:
            Tuple of (TaskInput, ground_truth)
        """
        task_input = TaskInput(
            task_id=example.task_id,
            description=example.question,
            domain=example.domain,
            playbook_bullets=playbook_bullets or [],
            max_reasoning_steps=10
        )

        return task_input, example.ground_truth

    def load(self, split: str = "train", num_examples: Optional[int] = None) -> List[DatasetExample]:
        """
        Load dataset based on configured dataset_name.

        Args:
            split: Dataset split (train, test, validation)
            num_examples: Limit number of examples

        Returns:
            List of DatasetExample objects
        """
        if self.dataset_name == "gsm8k":
            return self.load_gsm8k(split=split, num_examples=num_examples)
        elif self.dataset_name == "custom":
            raise ValueError("For custom datasets, use load_from_json() directly")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def load_train(self, num_examples: Optional[int] = None) -> List[DatasetExample]:
        """Load training split."""
        return self.load(split="train", num_examples=num_examples)

    def load_test(self, num_examples: Optional[int] = None) -> List[DatasetExample]:
        """Load test split."""
        return self.load(split="test", num_examples=num_examples)

    def stream(self, split: str = "train", batch_size: int = 1) -> Iterator[List[DatasetExample]]:
        """
        Stream dataset in batches (for large datasets).

        Args:
            split: Dataset split
            batch_size: Number of examples per batch

        Yields:
            Batches of DatasetExample objects
        """
        examples = self.load(split=split, num_examples=None)

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i+batch_size]
            yield batch


def create_dataset_loader(dataset_name: str = "gsm8k", cache_dir: Optional[str] = None) -> DatasetLoader:
    """
    Factory function to create DatasetLoader.

    Args:
        dataset_name: Name of dataset
        cache_dir: Optional cache directory

    Returns:
        DatasetLoader instance
    """
    return DatasetLoader(dataset_name=dataset_name, cache_dir=cache_dir)


__version__ = "v1.0.0"
