# ACE Offline Training Guide

This guide explains how to use the offline training system to bootstrap ACE playbooks from datasets before production deployment.

## Overview

Offline training processes datasets in batch mode through the **Generator → Reflector → Curator** workflow to extract and consolidate insights into playbooks. This approach:

- Bootstraps playbooks with high-quality strategies before production
- Uses ground-truth feedback for objective insight labeling
- Processes datasets efficiently with batch merging
- Tracks comprehensive metrics throughout training

## Quick Start

### Basic Usage with GSM8K

Train on 100 examples from the GSM8K dataset:

```bash
python scripts/offline_train.py \
  --dataset gsm8k \
  --num_examples 100 \
  --output playbooks/gsm8k_trained.json
```

### Custom Dataset

Train on your own dataset (JSON format):

```bash
python scripts/offline_train.py \
  --dataset custom \
  --input data/my_dataset.json \
  --num_examples 50 \
  --output playbooks/custom_trained.json
```

### Continue Training

Resume training from an existing playbook:

```bash
python scripts/offline_train.py \
  --dataset gsm8k \
  --num_examples 200 \
  --initial_playbook playbooks/existing.json \
  --output playbooks/continued.json
```

## Command-Line Options

### Dataset Options

- `--dataset`: Dataset to use (`gsm8k` or `custom`)
- `--input`: Path to custom dataset JSON (required if `--dataset=custom`)
- `--split`: Dataset split to use (`train`, `test`, `validation`)
- `--num_examples`: Number of examples to process

### Model Configuration

- `--generator_model`: Model for Generator (default: `gpt-4-turbo`)
- `--reflector_model`: Model for Reflector (default: `gpt-4o-mini`)

### Playbook Options

- `--initial_playbook`: Path to existing playbook to continue training
- `--output`: Path to save trained playbook (default: `playbooks/offline_trained.json`)
- `--metrics_output`: Path to save training metrics (default: `metrics/offline_training_metrics.json`)

### Training Parameters

- `--batch_size`: Batch size for curator merging (default: 10)
- `--similarity_threshold`: Similarity threshold for deduplication (default: 0.8)
- `--target_stage`: Target playbook stage (`shadow` or `production`)
- `--checkpoint_interval`: Save checkpoint every N examples (default: 50)
- `--no_checkpoints`: Disable checkpoint saving

## Custom Dataset Format

Custom datasets should be JSON files with the following structure:

```json
[
  {
    "question": "What is 2+2?",
    "answer": "4",
    "domain": "arithmetic"
  },
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "domain": "geography"
  }
]
```

**Required Fields:**
- `question`: The task description or question
- `answer`: The ground truth answer

**Optional Fields:**
- `domain`: Domain category (default: "general")

## Programmatic Usage

### Basic Training

```python
from ace.ops import create_offline_trainer

# Create trainer
trainer = create_offline_trainer(
    dataset_name="gsm8k",
    num_examples=100,
    generator_model="gpt-4-turbo",
    reflector_model="gpt-4o-mini",
    output_playbook_path="playbooks/trained.json"
)

# Execute training
result = trainer.train()

print(f"Trained playbook with {result['playbook_size']} bullets")
print(f"Accuracy: {result['metrics'].correct_answers}/{result['metrics'].total_examples}")
```

### Advanced Configuration

```python
from ace.ops import OfflineTrainer, TrainingConfig
from ace.models.playbook import PlaybookStage

# Create custom configuration
config = TrainingConfig(
    dataset_name="gsm8k",
    dataset_split="train",
    num_examples=200,
    generator_model="gpt-4o",
    reflector_model="gpt-4o-mini",
    batch_merge_size=20,  # Larger batches for efficiency
    similarity_threshold=0.85,  # Stricter deduplication
    target_stage=PlaybookStage.PRODUCTION,
    output_playbook_path="playbooks/production.json",
    output_metrics_path="metrics/production_metrics.json",
    save_checkpoints=True,
    checkpoint_interval=25
)

# Create trainer
trainer = OfflineTrainer(config)

# Train with existing playbook
result = trainer.train(initial_playbook_path="playbooks/seed.json")
```

### Custom Dataset Loading

```python
from ace.ops import DatasetLoader, OfflineTrainer, TrainingConfig

# Load custom dataset
loader = DatasetLoader(dataset_name="custom")
examples = loader.load_from_json(
    "data/my_dataset.json",
    num_examples=50
)

# Create trainer configuration
config = TrainingConfig(
    dataset_name="custom",
    num_examples=len(examples),
    output_playbook_path="playbooks/custom_trained.json"
)

trainer = OfflineTrainer(config)
trainer._custom_examples = examples

# Train
result = trainer.train()
```

## Training Workflow

The offline training process follows these steps:

1. **Dataset Loading**
   - Load examples from GSM8K or custom JSON
   - Convert to TaskInput format with ground truth

2. **Generation & Reflection** (per example)
   - Execute Generator to produce reasoning and answer
   - Execute Reflector with ground truth feedback
   - Extract Helpful/Harmful insights based on correctness

3. **Batch Merging** (every N examples)
   - Accumulate insights across multiple tasks
   - Merge with single FAISS index build (efficient)
   - Deduplicate semantically similar insights
   - Update playbook with new bullets or increment counts

4. **Checkpointing** (periodic)
   - Save training state at regular intervals
   - Resume from checkpoints if training interrupted

5. **Output Generation**
   - Save final trained playbook
   - Save comprehensive training metrics

## Output Files

### Trained Playbook

JSON file containing the bootstrapped playbook:

```json
{
  "version": "v1.0.0",
  "created_at": "2025-10-13T17:00:00",
  "training_config": {
    "dataset": "gsm8k",
    "num_examples": 100,
    "generator_model": "gpt-4-turbo",
    "reflector_model": "gpt-4o-mini"
  },
  "bullets": [
    {
      "id": "B001",
      "content": "Break complex problems into smaller steps",
      "domain_id": "arithmetic",
      "section": "helpful",
      "stage": "shadow",
      "activation_count": 15,
      "confidence": 0.92
    }
  ]
}
```

### Training Metrics

JSON file with detailed training statistics:

```json
{
  "training_summary": {
    "total_examples": 100,
    "successful_generations": 98,
    "failed_generations": 2,
    "accuracy": 0.85
  },
  "insights": {
    "total_extracted": 250,
    "helpful": 180,
    "harmful": 60,
    "neutral": 10
  },
  "playbook": {
    "bullets_added": 45,
    "bullets_incremented": 205,
    "final_size": 45
  },
  "timing": {
    "start_time": "2025-10-13T17:00:00",
    "end_time": "2025-10-13T17:30:00",
    "duration_seconds": 1800.0
  }
}
```

### Checkpoints

Periodic snapshots saved to `checkpoints/` directory:

```json
{
  "timestamp": "2025-10-13T17:15:00",
  "config": {...},
  "metrics": {
    "total_examples": 50,
    "correct_answers": 42,
    "playbook_size": 28
  },
  "playbook": [...]
}
```

## Performance Optimization

### Batch Size Tuning

- **Small batches (5-10)**: More frequent merging, higher overhead
- **Medium batches (10-20)**: Balanced performance (recommended)
- **Large batches (20-50)**: Better consolidation, less frequent merging

### Similarity Threshold

- **Lower threshold (0.7-0.8)**: More aggressive deduplication, smaller playbooks
- **Higher threshold (0.85-0.9)**: Preserve more variations, larger playbooks

### Model Selection

- **Generator**: Use stronger models (gpt-4-turbo, gpt-4o) for better reasoning
- **Reflector**: Use efficient models (gpt-4o-mini) since reflection is simpler

## Troubleshooting

### Import Error: datasets library

```bash
pip install datasets
```

The HuggingFace `datasets` library is required for GSM8K loading.

### Out of Memory

Reduce batch size or number of examples:

```bash
python scripts/offline_train.py \
  --dataset gsm8k \
  --num_examples 50 \
  --batch_size 5
```

### Low Accuracy

Check that:
1. Ground truth format matches expected answer format
2. Generator model is capable for the task domain
3. Dataset examples are valid and well-formed

### Checkpoint Recovery

If training crashes, checkpoints are saved in `checkpoints/`. To resume:

1. Find the latest checkpoint: `ls -lt checkpoints/`
2. Extract the playbook from checkpoint JSON
3. Use as `--initial_playbook` for continued training

## Best Practices

1. **Start Small**: Test with 10-20 examples before full training
2. **Monitor Metrics**: Check accuracy and insight distribution
3. **Use Checkpoints**: Enable for long-running training sessions
4. **Tune Batch Size**: Balance between efficiency and memory
5. **Domain Isolation**: Use consistent domain tags across datasets
6. **Validate Output**: Review trained playbook before production use

## Examples

### Arithmetic Domain

Train specialized arithmetic playbook:

```bash
python scripts/offline_train.py \
  --dataset gsm8k \
  --num_examples 500 \
  --generator_model gpt-4o \
  --batch_size 20 \
  --output playbooks/arithmetic_expert.json
```

### Multi-Domain Training

Train on mixed domain custom dataset:

```bash
python scripts/offline_train.py \
  --dataset custom \
  --input data/mixed_domains.json \
  --num_examples 200 \
  --similarity_threshold 0.85 \
  --output playbooks/multi_domain.json
```

### Production Deployment

Train production-ready playbook:

```bash
python scripts/offline_train.py \
  --dataset gsm8k \
  --num_examples 1000 \
  --generator_model gpt-4o \
  --reflector_model gpt-4o-mini \
  --target_stage production \
  --checkpoint_interval 100 \
  --output playbooks/production_v1.json
```

## See Also

- [Architecture Documentation](../README.md)
- [API Reference](../docs/API.md)
- [Testing Guide](../docs/TESTING.md)
