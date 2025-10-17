# Multiplication Learning Analysis Suite

Comprehensive evaluation suite for the ACE Framework multiplication learning experiment.

## Overview

This directory contains 6 evaluation scripts that rigorously test the ACE Playbook's ability to learn and apply prompting strategies for large integer multiplication (1-10,000 range).

## Prerequisites

```bash
# Install dependencies
pip install -e .

# Set API keys in .env
OPENROUTER_API_KEY=sk-or-v1-...  # For Qwen (default)
OPENAI_API_KEY=sk-...            # Optional: for model transfer
ANTHROPIC_API_KEY=sk-ant-...     # Optional: for model transfer
```

## Evaluation Scripts

### 1. Holdout Test (`holdout_test.py`)
**Gold standard generalization test**

Tests learned strategies on NEW problems (different seed).

```bash
python examples/multiplication_analysis/holdout_test.py
```

**What it measures:**
- Can strategies generalize to unseen problems?
- Is the playbook overfitting to training data?

**Runtime:** ~5 minutes

---

### 2. Ablation Study (`ablation_study.py`)
**Direct playbook impact measurement**

Compares performance WITH vs WITHOUT playbook strategies.

```bash
python examples/multiplication_analysis/ablation_study.py
```

**What it measures:**
- Control: Generator with empty playbook
- Treatment: Generator with learned strategies
- Absolute lift from playbook

**Runtime:** ~10 minutes

---

### 3. Difficulty Analysis (`difficulty_analysis.py`)
**Analyzes existing training data**

Examines how accuracy varies by problem difficulty:
- Easy: Both operands < 100
- Medium: One operand < 100, or both < 1000
- Hard: Both operands >= 1000

```bash
python examples/multiplication_analysis/difficulty_analysis.py
```

**What it measures:**
- Accuracy breakdown by difficulty
- Problem distribution
- Difficulty impact on success rate

**Runtime:** < 1 minute (database analysis only)

---

### 4. Strategy Quality (`strategy_quality.py`)
**Analyzes existing training data**

Identifies which specific strategies correlate with correct answers.

```bash
python examples/multiplication_analysis/strategy_quality.py
```

**What it measures:**
- Bullet usage in correct vs incorrect answers
- Success rate when each strategy is used
- Stage-level effectiveness (PROD vs STAGING vs SHADOW)
- Recommendations for promotion/quarantine

**Runtime:** < 1 minute (database analysis only)

---

### 5. Longer Training (`longer_training.py`)
**Extended learning experiment**

Runs 10 epochs instead of 3 to test learning curves.

```bash
python examples/multiplication_analysis/longer_training.py
```

**What it measures:**
- Does accuracy continue improving beyond 3 epochs?
- When do strategies plateau?
- How many PROD strategies emerge over time?

**Runtime:** ~30 minutes

---

### 6. Model Transfer (`model_transfer.py`)
**Cross-model portability test**

Tests if strategies learned with Qwen transfer to Claude/GPT-4.

```bash
python examples/multiplication_analysis/model_transfer.py
```

**What it measures:**
- Are strategies model-agnostic?
- Do strategies work better/worse on different models?
- Portability of learned patterns

**Runtime:** ~15 minutes

**Requirements:** ANTHROPIC_API_KEY or OPENAI_API_KEY

---

## Running All Evaluations

### Quick Analysis (Evaluations 3 & 4)
Runs database-only analysis on existing training data:

```bash
# Run both in sequence
python examples/multiplication_analysis/difficulty_analysis.py
python examples/multiplication_analysis/strategy_quality.py
```

**Total runtime:** ~2 minutes

### Fast Experiments (Evaluations 1 & 2)
Runs new experiments with reasonable runtime:

```bash
python examples/multiplication_analysis/holdout_test.py
python examples/multiplication_analysis/ablation_study.py
```

**Total runtime:** ~15 minutes

### Long Experiments (Evaluations 5 & 6)
Extended training and model transfer:

```bash
python examples/multiplication_analysis/longer_training.py
python examples/multiplication_analysis/model_transfer.py
```

**Total runtime:** ~45 minutes

### Master Runner
Run all evaluations sequentially:

```bash
bash examples/multiplication_analysis/run_all.sh
```

**Total runtime:** ~60 minutes

## Results Interpretation

### Holdout Test
- **Similar accuracy** = Good generalization
- **Higher accuracy** = Robust strategies
- **Lower accuracy** = Possible overfitting

### Ablation Study
- **+10% lift** = Strong positive effect
- **+5% lift** = Moderate positive effect
- **0% lift** = Negligible effect
- **Negative lift** = Strategies interfering

### Difficulty Analysis
- **Large gap** (>15%) = Model struggles with hard problems
- **Moderate gap** (5-15%) = Some difficulty sensitivity
- **Small gap** (<5%) = Consistent performance

### Strategy Quality
- **Positive lift** = Strategy helps (should promote)
- **Negative lift** = Strategy harms (should quarantine)
- **PROD strategies** should show highest lift

### Longer Training
- **Continued improvement** = Keep training
- **Plateau** = Optimal epoch count found
- **Regression** = Overfitting detected

### Model Transfer
- **Positive transfer** = Strategies are model-agnostic
- **Similar accuracy** = Good portability
- **Negative transfer** = Model-specific strategies

## Project Structure

```
examples/multiplication_analysis/
├── __init__.py                 # Shared utilities
├── README.md                   # This file
├── run_all.sh                  # Master runner
├── holdout_test.py            # Evaluation 1
├── ablation_study.py          # Evaluation 2
├── difficulty_analysis.py     # Evaluation 3
├── strategy_quality.py        # Evaluation 4
├── longer_training.py         # Evaluation 5
└── model_transfer.py          # Evaluation 6
```

## Key Findings (Expected)

Based on the original 3-epoch experiment:

1. **Accuracy improved 75%** (20% → 35%) by Epoch 3
2. **1 strategy reached PROD**, 1 reached STAGING
3. **198 total insights** discovered
4. **Strategies are computational**, not just "use calculator"

This analysis suite validates and extends these findings.

## Troubleshooting

### "No playbook bullets found"
Run the main experiment first:
```bash
NUM_PROBLEMS=20 NUM_EPOCHS=3 python examples/arithmetic_learning_multiplication.py
```

### "API key required"
Set appropriate keys in `.env`:
```bash
# For Qwen (default)
OPENROUTER_API_KEY=sk-or-v1-...

# For model transfer
ANTHROPIC_API_KEY=sk-ant-...  # Claude
OPENAI_API_KEY=sk-...         # GPT-4
```

### "ModuleNotFoundError"
Install the package:
```bash
pip install -e .
```

## Citation

If you use this evaluation suite, please cite the ACE Framework paper:

```bibtex
@article{ace2024,
  title={ACE: Adaptive Context Engineering for LLMs},
  journal={arXiv preprint arXiv:2510.04618v1},
  year={2024}
}
```

## License

Same as parent project (see root LICENSE file).
