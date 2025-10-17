#!/usr/bin/env python3
"""
Model Transfer Test - Evaluation 6

Tests if strategies learned with one model transfer to a different model.

Original training: Qwen 2.5 7B Instruct
Transfer test: Claude Haiku or GPT-4o-mini

Key question: Are strategies model-agnostic or model-specific?

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY in .env
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from multiplication_analysis import (
    setup_database,
    configure_dspy_lm,
    generate_multiplication_problems,
    get_playbook_bullets,
    create_task_db,
    save_task_output,
    print_header,
    print_section
)
from ace.generator import CoTGenerator, TaskInput
from ace.models import PlaybookStage


def run_model_transfer_test(
    session,
    transfer_model_name: str,
    num_problems: int = 20,
    test_seed: int = 888
):
    """
    Test strategy transfer to different model.

    Args:
        session: Database session
        transfer_model_name: Name of model to transfer to
        num_problems: Number of problems to test
        test_seed: Seed for test problems
    """
    print_header("Evaluation 6: Model Transfer Test")

    domain_id = "multiplication-large"
    transfer_domain = "multiplication-transfer"

    print(f"📚 Loading strategies trained with Qwen 2.5 7B...")

    # Get playbook bullets (trained with Qwen)
    bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
    bullet_contents = [b.content for b in bullets]

    print(f"✓ Loaded {len(bullets)} PROD strategies")
    if bullets:
        print(f"\n   Strategies:")
        for i, bullet in enumerate(bullets, 1):
            print(f"   {i}. {bullet.content[:70]}...")
    else:
        print(f"\n   ⚠️  No production strategies found!")

    print(f"\n🔄 Testing transfer to: {transfer_model_name}")

    # Configure transfer model
    print(f"\nConfiguring {transfer_model_name}...")

    # Create new DSPy LM for transfer model
    if transfer_model_name.startswith("openrouter/"):
        # OpenRouter model
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key or api_key.startswith("your_"):
            print("❌ OPENROUTER_API_KEY required for OpenRouter models")
            print("   Set in .env file and try again")
            return
        lm = dspy.LM(transfer_model_name, api_key=api_key)
    elif "claude" in transfer_model_name or "anthropic" in transfer_model_name:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key.startswith("your_"):
            print("❌ ANTHROPIC_API_KEY required for Claude models")
            print("   Set in .env file and try again")
            return
        lm = dspy.LM(transfer_model_name, api_key=api_key)
    elif "gpt" in transfer_model_name or "openai" in transfer_model_name:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            print("❌ OPENAI_API_KEY required for OpenAI models")
            print("   Set in .env file and try again")
            return
        lm = dspy.LM(transfer_model_name, api_key=api_key)
    else:
        print(f"❌ Unknown model: {transfer_model_name}")
        return

    dspy.configure(lm=lm)
    print(f"✓ Configured DSPy with {transfer_model_name}")

    # Initialize generator with transfer model
    generator = CoTGenerator(model=transfer_model_name, temperature=0.7)
    print(f"✓ Initialized Generator\n")

    # Generate test problems
    print(f"🎲 Generating {num_problems} test problems (seed={test_seed})...")
    test_problems = generate_multiplication_problems(num_problems, seed=test_seed)
    print(f"✓ Generated {len(test_problems)} problems\n")

    print_section(f"Testing with {transfer_model_name}")

    correct_count = 0
    total_count = 0
    results = []

    for idx, prob in enumerate(test_problems, 1):
        # Create task
        task_db = create_task_db(session, prob["problem"], prob["answer"], transfer_domain)

        # Generate with transferred strategies
        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=bullet_contents,  # Strategies from Qwen training
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)

            is_correct = generator_output.answer.strip() == prob["answer"]
            if is_correct:
                correct_count += 1
            total_count += 1

            results.append({
                "problem": prob["problem"],
                "expected": prob["answer"],
                "answer": generator_output.answer,
                "correct": is_correct
            })

            # Save output
            save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": generator_output.bullets_referenced,
                "latency_ms": generator_output.latency_ms or 0,
                "prompt_tokens": generator_output.prompt_tokens or 0,
                "completion_tokens": generator_output.completion_tokens or 0
            })

            # Progress indicator
            if idx % 5 == 0 or idx == 1:
                status = "✅" if is_correct else "❌"
                print(f"  [{idx}/{num_problems}] {prob['problem']} = {generator_output.answer} {status}")

        except Exception as e:
            print(f"\n❌ Error on problem {idx}: {e}")
            continue

    # Calculate metrics
    transfer_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    # Print results
    print_section("Transfer Test Results")

    print(f"  Transfer Model: {transfer_model_name}")
    print(f"  Total problems: {total_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {total_count - correct_count}")
    print(f"  Accuracy: {transfer_accuracy:.1f}%")

    # Compare to original training
    print_section("Transfer Analysis")

    original_accuracy = 35.0  # From Epoch 3 of original experiment
    print(f"  Original Model (Qwen):     {original_accuracy:.1f}%")
    print(f"  Transfer Model ({transfer_model_name.split('/')[-1]}): {transfer_accuracy:.1f}%")

    delta = transfer_accuracy - original_accuracy

    if delta > 10:
        print(f"\n  ✨ EXCELLENT TRANSFER (+{delta:.1f}%)")
        print(f"     Strategies work even better on {transfer_model_name}!")
    elif delta > 0:
        print(f"\n  ✅ POSITIVE TRANSFER (+{delta:.1f}%)")
        print(f"     Strategies transfer well to {transfer_model_name}")
    elif delta > -10:
        print(f"\n  ⚡ MODERATE TRANSFER ({delta:.1f}%)")
        print(f"     Strategies mostly transfer, some adaptation needed")
    else:
        print(f"\n  ⚠️  POOR TRANSFER ({delta:.1f}%)")
        print(f"     Strategies may be model-specific")

    # Show samples
    print_section("Sample Results")

    correct_samples = [r for r in results if r["correct"]][:3]
    incorrect_samples = [r for r in results if not r["correct"]][:3]

    if correct_samples:
        print(f"  ✅ Correct ({len([r for r in results if r['correct']])} total):")
        for r in correct_samples:
            print(f"     {r['problem']} = {r['answer']}")

    if incorrect_samples:
        print(f"\n  ❌ Incorrect ({len([r for r in results if not r['correct']])} total):")
        for r in incorrect_samples:
            print(f"     {r['problem']}")
            print(f"        Expected: {r['expected']}, Got: {r['answer']}")

    print("\n" + "="*70)
    print("✅ Model Transfer Test Complete")
    print("="*70 + "\n")

    print("💡 Interpretation:")
    print("   - Positive transfer = strategies are model-agnostic")
    print("   - Negative transfer = strategies may be model-specific")
    print("   - Similar accuracy = good portability\n")


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Model Transfer Test\n")

    # Determine transfer model
    transfer_models = []

    # Check for OpenRouter (supports many models)
    if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENROUTER_API_KEY").startswith("your_"):
        # Use a different model than training (Qwen) - try Claude Haiku or GPT-4o-mini
        transfer_models.append(("openrouter/anthropic/claude-3-haiku", "Claude 3 Haiku via OpenRouter"))
        transfer_models.append(("openrouter/openai/gpt-4o-mini", "GPT-4o-mini via OpenRouter"))
        transfer_models.append(("openrouter/meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B via OpenRouter"))

    if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_API_KEY").startswith("your_"):
        transfer_models.append(("anthropic/claude-3-haiku-20240307", "Claude 3 Haiku (direct)"))

    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY").startswith("your_"):
        transfer_models.append(("openai/gpt-4o-mini", "GPT-4o-mini (direct)"))

    if not transfer_models:
        print("❌ No alternative model API keys found!")
        print("\nTo test model transfer, set one of:")
        print("  - OPENROUTER_API_KEY (for access to many models)")
        print("  - ANTHROPIC_API_KEY (for Claude models)")
        print("  - OPENAI_API_KEY (for GPT models)")
        print("\nNote: Original training used Qwen via OpenRouter")
        sys.exit(1)

    # Use first available transfer model
    transfer_model, transfer_model_display = transfer_models[0]

    print(f"Available transfer model: {transfer_model_display}")
    print(f"Model identifier: {transfer_model}\n")

    try:
        session = setup_database()

        # Run transfer test
        run_model_transfer_test(
            session=session,
            transfer_model_name=transfer_model,
            num_problems=20,
            test_seed=888
        )

        session.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
