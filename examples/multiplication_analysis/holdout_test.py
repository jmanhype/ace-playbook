#!/usr/bin/env python3
"""
Holdout Test - Evaluation 1

Tests learned playbook strategies on NEW multiplication problems (different seed).

This is the gold standard evaluation: can strategies generalize to unseen problems?

Tests 20 new problems using the playbook learned from the original training.
"""

import sys
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


def run_holdout_test(
    session,
    generator: CoTGenerator,
    num_problems: int = 20,
    holdout_seed: int = 999
):
    """
    Run holdout test with new problems.

    Args:
        session: Database session
        generator: CoT generator
        num_problems: Number of new problems to test
        holdout_seed: Different seed for new problems
    """
    print_header("Evaluation 1: Holdout Test")

    domain_id = "multiplication-large"
    holdout_domain = "multiplication-holdout"

    # Get current playbook bullets
    bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
    bullet_contents = [b.content for b in bullets]

    print(f"📚 Loaded {len(bullets)} PROD strategies from training")
    if bullets:
        print(f"\n   Production strategies:")
        for i, bullet in enumerate(bullets, 1):
            print(f"   {i}. {bullet.content[:70]}...")
    else:
        print(f"   ⚠️  No production strategies found!")
        print(f"      Results will show baseline performance without playbook.")

    # Generate NEW problems with different seed
    print(f"\n🎲 Generating {num_problems} NEW holdout problems (seed={holdout_seed})...")
    holdout_problems = generate_multiplication_problems(num_problems, seed=holdout_seed)
    print(f"✓ Generated {len(holdout_problems)} unseen problems\n")

    print_section(f"Testing Holdout Problems")

    correct_count = 0
    total_count = 0
    results = []

    for idx, prob in enumerate(holdout_problems, 1):
        # Create task
        task_db = create_task_db(session, prob["problem"], prob["answer"], holdout_domain)

        # Generate solution with playbook context
        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=bullet_contents,
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)

            # Check correctness
            is_correct = generator_output.answer.strip() == prob["answer"]
            if is_correct:
                correct_count += 1
            total_count += 1

            results.append({
                "problem": prob["problem"],
                "expected": prob["answer"],
                "answer": generator_output.answer,
                "correct": is_correct,
                "confidence": generator_output.confidence
            })

            # Save to database
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
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    # Print results
    print_section("Holdout Test Results")

    print(f"  Total problems: {total_count}")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {total_count - correct_count}")
    print(f"  Accuracy: {accuracy:.1f}%")

    # Show sample correct and incorrect
    print(f"\n  ✅ Sample Correct ({len([r for r in results if r['correct']])} total):")
    for r in [r for r in results if r["correct"]][:3]:
        print(f"     {r['problem']} = {r['answer']}")

    print(f"\n  ❌ Sample Incorrect ({len([r for r in results if not r['correct']])} total):")
    for r in [r for r in results if not r["correct"]][:3]:
        print(f"     {r['problem']}")
        print(f"        Expected: {r['expected']}, Got: {r['answer']}")

    # Compare to training performance
    print_section("Generalization Analysis")

    print(f"  Holdout Accuracy: {accuracy:.1f}%")
    print(f"  Training Accuracy (Epoch 3): 35.0%  (from original experiment)")

    delta = accuracy - 35.0

    if abs(delta) < 5:
        print(f"  Status: ✅ Good generalization (within 5% of training)")
    elif delta > 0:
        print(f"  Status: ✨ Excellent! Strategies generalize to new problems (+{delta:.1f}%)")
    else:
        print(f"  Status: ⚠️  Some overfitting detected ({delta:.1f}% drop)")

    print("\n" + "="*70)
    print("✅ Holdout Test Complete")
    print("="*70 + "\n")

    print("💡 Interpretation:")
    print("   - Holdout tests unseen problems (different seed)")
    print("   - Tests if strategies generalize beyond training data")
    print("   - Similar accuracy = good generalization")
    print("   - Higher accuracy = strategies are robust")
    print("   - Lower accuracy = potential overfitting\n")


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Holdout Test Evaluation\n")

    try:
        # Configure DSPy
        lm, model_name = configure_dspy_lm()
        dspy.configure(lm=lm)

        # Initialize database
        session = setup_database()

        # Initialize generator (same as training)
        generator = CoTGenerator(model=model_name, temperature=0.7)
        print(f"✓ Initialized Generator with {model_name}\n")

        # Run holdout test
        run_holdout_test(
            session=session,
            generator=generator,
            num_problems=20,
            holdout_seed=999  # Different from training seed (42)
        )

        session.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
