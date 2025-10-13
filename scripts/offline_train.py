#!/usr/bin/env python3
"""
Offline Training CLI Script

Command-line interface for bootstrapping playbooks from offline datasets.
Executes Dataset → Generator → Reflector → Curator workflow.

Usage:
    python scripts/offline_train.py --dataset gsm8k --num_examples 100 --output playbooks/gsm8k_trained.json
    python scripts/offline_train.py --dataset custom --input data/custom_dataset.json --num_examples 50

Based on tasks.md T052.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ace.ops.offline_trainer import OfflineTrainer, TrainingConfig
from ace.curator import PlaybookStage
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="offline_train_cli")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Offline Training Runner for ACE Playbook Bootstrapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on GSM8K with 100 examples
  python scripts/offline_train.py --dataset gsm8k --num_examples 100

  # Train on custom dataset
  python scripts/offline_train.py --dataset custom --input data/my_dataset.json --num_examples 50

  # Continue training from existing playbook
  python scripts/offline_train.py --dataset gsm8k --num_examples 200 --initial_playbook playbooks/existing.json

  # Use different models
  python scripts/offline_train.py --dataset gsm8k --generator_model gpt-4o --reflector_model gpt-4o-mini
        """
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "custom"],
        help="Dataset to use for training (default: gsm8k)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to custom dataset JSON file (required if --dataset=custom)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split to use (default: train)"
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to process (default: 100)"
    )

    # Model arguments
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-4-turbo",
        help="Model for Generator (default: gpt-4-turbo)"
    )

    parser.add_argument(
        "--reflector_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for Reflector (default: gpt-4o-mini)"
    )

    # Playbook arguments
    parser.add_argument(
        "--initial_playbook",
        type=str,
        default=None,
        help="Path to existing playbook to continue training (default: start empty)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="playbooks/offline_trained.json",
        help="Path to save trained playbook (default: playbooks/offline_trained.json)"
    )

    parser.add_argument(
        "--metrics_output",
        type=str,
        default="metrics/offline_training_metrics.json",
        help="Path to save training metrics (default: metrics/offline_training_metrics.json)"
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for curator merging (default: 10)"
    )

    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for deduplication (default: 0.8)"
    )

    parser.add_argument(
        "--target_stage",
        type=str,
        default="shadow",
        choices=["shadow", "production"],
        help="Target playbook stage (default: shadow)"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N examples (default: 50)"
    )

    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable checkpoint saving"
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments."""
    errors = []

    # Validate custom dataset input
    if args.dataset == "custom" and not args.input:
        errors.append("--input is required when --dataset=custom")

    if args.input and not Path(args.input).exists():
        errors.append(f"Input file not found: {args.input}")

    # Validate num_examples
    if args.num_examples <= 0:
        errors.append("--num_examples must be positive")

    # Validate similarity threshold
    if not 0.0 <= args.similarity_threshold <= 1.0:
        errors.append("--similarity_threshold must be between 0.0 and 1.0")

    # Validate batch size
    if args.batch_size <= 0:
        errors.append("--batch_size must be positive")

    # Validate checkpoint interval
    if args.checkpoint_interval <= 0:
        errors.append("--checkpoint_interval must be positive")

    if errors:
        print("Validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for offline training CLI."""
    args = parse_args()
    validate_args(args)

    logger.info(
        "offline_train_cli_started",
        dataset=args.dataset,
        num_examples=args.num_examples,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model
    )

    # Create training configuration
    config = TrainingConfig(
        dataset_name=args.dataset,
        dataset_split=args.split,
        num_examples=args.num_examples,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        batch_merge_size=args.batch_size,
        output_playbook_path=args.output,
        output_metrics_path=args.metrics_output,
        similarity_threshold=args.similarity_threshold,
        target_stage=PlaybookStage.SHADOW if args.target_stage == "shadow" else PlaybookStage.PRODUCTION,
        save_checkpoints=not args.no_checkpoints,
        checkpoint_interval=args.checkpoint_interval
    )

    # Create trainer
    trainer = OfflineTrainer(config)

    # Handle custom dataset loading
    if args.dataset == "custom":
        logger.info("loading_custom_dataset", path=args.input)
        # Load custom dataset and inject into trainer
        custom_examples = trainer.dataset_loader.load_from_json(
            args.input,
            num_examples=args.num_examples
        )
        # Override dataset loader to use custom examples
        trainer._custom_examples = custom_examples

    # Execute training
    try:
        print(f"\n{'='*60}")
        print(f"ACE Offline Training")
        print(f"{'='*60}")
        print(f"Dataset:           {args.dataset}")
        print(f"Examples:          {args.num_examples}")
        print(f"Generator:         {args.generator_model}")
        print(f"Reflector:         {args.reflector_model}")
        print(f"Batch size:        {args.batch_size}")
        print(f"Output:            {args.output}")
        print(f"{'='*60}\n")

        result = trainer.train(initial_playbook_path=args.initial_playbook)

        if result["success"]:
            print(f"\n{'='*60}")
            print(f"Training Completed Successfully!")
            print(f"{'='*60}")
            print(f"Examples processed: {result['metrics'].total_examples}")
            print(f"Accuracy:           {result['metrics'].correct_answers}/{result['metrics'].total_examples} "
                  f"({result['metrics'].correct_answers/result['metrics'].total_examples*100:.1f}%)")
            print(f"Insights extracted: {result['metrics'].total_insights_extracted}")
            print(f"  - Helpful:        {result['metrics'].helpful_insights}")
            print(f"  - Harmful:        {result['metrics'].harmful_insights}")
            print(f"  - Neutral:        {result['metrics'].neutral_insights}")
            print(f"Playbook size:      {result['playbook_size']} bullets")
            print(f"  - Added:          {result['metrics'].playbook_bullets_added}")
            print(f"  - Incremented:    {result['metrics'].playbook_bullets_incremented}")
            print(f"Duration:           {result['metrics'].duration_seconds:.1f}s")
            print(f"\nOutputs saved:")
            print(f"  - Playbook:       {result['output_paths']['playbook']}")
            print(f"  - Metrics:        {result['output_paths']['metrics']}")
            print(f"{'='*60}\n")

            logger.info("offline_train_cli_completed", result=result)
            return 0
        else:
            print("\nTraining failed. Check logs for details.", file=sys.stderr)
            logger.error("offline_train_cli_failed")
            return 1

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.", file=sys.stderr)
        logger.warning("offline_train_cli_interrupted")
        return 130

    except Exception as e:
        print(f"\nTraining failed with error: {e}", file=sys.stderr)
        logger.error("offline_train_cli_error", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
