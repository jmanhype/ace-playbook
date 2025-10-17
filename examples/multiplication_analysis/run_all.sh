#!/bin/bash

# Master runner for all multiplication analysis evaluations
# Total estimated runtime: ~60 minutes

set -e  # Exit on error

echo "========================================================================"
echo "ACE Framework - Comprehensive Multiplication Analysis Suite"
echo "========================================================================"
echo ""
echo "This will run all 6 evaluations sequentially:"
echo "  1. Holdout Test (~5 min)"
echo "  2. Ablation Study (~10 min)"
echo "  3. Difficulty Analysis (~1 min)"
echo "  4. Strategy Quality (~1 min)"
echo "  5. Longer Training (~30 min)"
echo "  6. Model Transfer (~15 min)"
echo ""
echo "Total estimated time: ~60 minutes"
echo ""

# Check if main experiment has been run
if [ ! -f "ace_playbook.db" ]; then
    echo "⚠️  Warning: Main experiment database not found"
    echo "   Run this first: NUM_PROBLEMS=20 NUM_EPOCHS=3 python examples/arithmetic_learning_multiplication.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create logs directory
mkdir -p logs

# Timestamp for log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================================================"
echo "Evaluation 1: Holdout Test"
echo "========================================================================"
python examples/multiplication_analysis/holdout_test.py 2>&1 | tee logs/holdout_${TIMESTAMP}.log
echo ""
echo "✅ Holdout test complete"
echo ""

echo "========================================================================"
echo "Evaluation 2: Ablation Study"
echo "========================================================================"
python examples/multiplication_analysis/ablation_study.py 2>&1 | tee logs/ablation_${TIMESTAMP}.log
echo ""
echo "✅ Ablation study complete"
echo ""

echo "========================================================================"
echo "Evaluation 3: Difficulty Analysis"
echo "========================================================================"
python examples/multiplication_analysis/difficulty_analysis.py 2>&1 | tee logs/difficulty_${TIMESTAMP}.log
echo ""
echo "✅ Difficulty analysis complete"
echo ""

echo "========================================================================"
echo "Evaluation 4: Strategy Quality"
echo "========================================================================"
python examples/multiplication_analysis/strategy_quality.py 2>&1 | tee logs/strategy_${TIMESTAMP}.log
echo ""
echo "✅ Strategy quality analysis complete"
echo ""

echo "========================================================================"
echo "Evaluation 5: Longer Training (10 epochs)"
echo "========================================================================"
echo "⏱️  This will take approximately 30 minutes..."
python examples/multiplication_analysis/longer_training.py 2>&1 | tee logs/longer_training_${TIMESTAMP}.log
echo ""
echo "✅ Longer training complete"
echo ""

echo "========================================================================"
echo "Evaluation 6: Model Transfer"
echo "========================================================================"

# Check if alternative model API keys are available
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Skipping model transfer test - no alternative model API keys found"
    echo "   Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env to enable this test"
elif [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ] && [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "⚠️  Skipping model transfer test - placeholder API keys detected"
    echo "   Set real ANTHROPIC_API_KEY or OPENAI_API_KEY in .env to enable this test"
else
    python examples/multiplication_analysis/model_transfer.py 2>&1 | tee logs/transfer_${TIMESTAMP}.log
    echo ""
    echo "✅ Model transfer test complete"
fi
echo ""

echo "========================================================================"
echo "🎉 ALL EVALUATIONS COMPLETE"
echo "========================================================================"
echo ""
echo "Logs saved to logs/ directory:"
echo "  - logs/holdout_${TIMESTAMP}.log"
echo "  - logs/ablation_${TIMESTAMP}.log"
echo "  - logs/difficulty_${TIMESTAMP}.log"
echo "  - logs/strategy_${TIMESTAMP}.log"
echo "  - logs/longer_training_${TIMESTAMP}.log"
if [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ]; then
    echo "  - logs/transfer_${TIMESTAMP}.log"
fi
echo ""
echo "Next steps:"
echo "  1. Review logs for detailed results"
echo "  2. Compare metrics across evaluations"
echo "  3. Identify best-performing strategies"
echo "  4. Consider additional training if needed"
echo ""
