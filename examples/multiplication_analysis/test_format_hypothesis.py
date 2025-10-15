#!/usr/bin/env python3
"""
Format Hypothesis Test

Tests if explicit output formatting instructions improve transfer to Llama.

Hypothesis: Llama is calculating correctly but outputting wrong format (commas).
Test: Use modified strategy with explicit "no commas" instruction.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from multiplication_analysis import (
    setup_database,
    generate_multiplication_problems,
    create_task_db,
    save_task_output,
    print_header,
    print_section
)
from ace.generator import CoTGenerator, TaskInput


def test_formatting_hypothesis():
    """Test if explicit format instructions help transfer."""
    print_header("Formatting Hypothesis Test")

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("❌ OPENROUTER_API_KEY required")
        sys.exit(1)

    transfer_model = "openrouter/meta-llama/llama-3.1-8b-instruct"

    print("🧪 HYPOTHESIS:")
    print("   Llama is doing math correctly but formatting output with commas")
    print("   Adding explicit 'no commas' instruction should improve accuracy")
    print()

    print(f"📝 Testing two strategies:")
    print()

    # Strategy 1: Original (minimal)
    strategy_original = "- Correctly applied the standard multiplication algorithm."
    print(f"   CONTROL (original):")
    print(f"   '{strategy_original}'")
    print()

    # Strategy 2: With explicit formatting
    strategy_formatted = (
        "- Correctly applied the standard multiplication algorithm. "
        "CRITICAL: Output ONLY the final answer as a plain integer with "
        "NO commas, NO spaces, NO formatting. "
        "Examples: Write '9263793' NOT '9,263,793'. Write '51458388' NOT '51,458,388'."
    )
    print(f"   TREATMENT (format-aware):")
    print(f"   '{strategy_formatted}'")
    print()

    # Configure Llama
    lm = dspy.LM(transfer_model, api_key=api_key)
    dspy.configure(lm=lm)
    generator = CoTGenerator(model=transfer_model, temperature=0.7)

    print(f"✓ Configured {transfer_model}\n")

    # Generate test problems (same seed as model transfer test)
    print(f"🎲 Generating 10 test problems (seed=888)...")
    test_problems = generate_multiplication_problems(10, seed=888)
    print(f"✓ Generated {len(test_problems)} problems\n")

    session = setup_database()

    # Test 1: Control (original strategy)
    print_section("Test 1: CONTROL (Original Strategy)")

    control_correct = 0
    control_results = []

    for idx, prob in enumerate(test_problems, 1):
        task_db = create_task_db(session, prob["problem"], prob["answer"], "multiplication-format-control")

        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=[strategy_original],  # Original minimal strategy
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)
            is_correct = generator_output.answer.strip() == prob["answer"]

            if is_correct:
                control_correct += 1

            control_results.append({
                "problem": prob["problem"],
                "expected": prob["answer"],
                "answer": generator_output.answer,
                "correct": is_correct
            })

            save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": generator_output.bullets_referenced,
                "latency_ms": generator_output.latency_ms or 0,
                "prompt_tokens": generator_output.prompt_tokens or 0,
                "completion_tokens": generator_output.completion_tokens or 0
            })

            status = "✅" if is_correct else "❌"
            print(f"  [{idx}/10] {prob['problem']} = {generator_output.answer} {status}")

        except Exception as e:
            print(f"❌ Error on problem {idx}: {e}")
            continue

    control_accuracy = (control_correct / len(test_problems) * 100) if test_problems else 0

    print(f"\n  Control Accuracy: {control_correct}/{len(test_problems)} = {control_accuracy:.1f}%\n")

    # Test 2: Treatment (format-aware strategy)
    print_section("Test 2: TREATMENT (Format-Aware Strategy)")

    treatment_correct = 0
    treatment_results = []

    for idx, prob in enumerate(test_problems, 1):
        task_db = create_task_db(session, prob["problem"], prob["answer"], "multiplication-format-treatment")

        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=[strategy_formatted],  # Format-aware strategy
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)
            is_correct = generator_output.answer.strip() == prob["answer"]

            if is_correct:
                treatment_correct += 1

            treatment_results.append({
                "problem": prob["problem"],
                "expected": prob["answer"],
                "answer": generator_output.answer,
                "correct": is_correct
            })

            save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": generator_output.bullets_referenced,
                "latency_ms": generator_output.latency_ms or 0,
                "prompt_tokens": generator_output.prompt_tokens or 0,
                "completion_tokens": generator_output.completion_tokens or 0
            })

            status = "✅" if is_correct else "❌"
            print(f"  [{idx}/10] {prob['problem']} = {generator_output.answer} {status}")

        except Exception as e:
            print(f"❌ Error on problem {idx}: {e}")
            continue

    treatment_accuracy = (treatment_correct / len(test_problems) * 100) if test_problems else 0

    print(f"\n  Treatment Accuracy: {treatment_correct}/{len(test_problems)} = {treatment_accuracy:.1f}%\n")

    # Results Analysis
    print_section("Hypothesis Test Results")

    print(f"  Control (original):      {control_accuracy:.1f}% ({control_correct}/10)")
    print(f"  Treatment (formatted):   {treatment_accuracy:.1f}% ({treatment_correct}/10)")

    delta = treatment_accuracy - control_accuracy

    print(f"\n  Improvement: {delta:+.1f}%")

    if delta > 20:
        print(f"\n  ✨ HYPOTHESIS CONFIRMED!")
        print(f"     Format instructions significantly improved accuracy")
        print(f"     Strategies CAN transfer with proper output constraints")
    elif delta > 10:
        print(f"\n  ✅ HYPOTHESIS SUPPORTED")
        print(f"     Format instructions helped transfer")
    elif delta > 0:
        print(f"\n  ⚡ WEAK SUPPORT")
        print(f"     Slight improvement from format instructions")
    else:
        print(f"\n  ❌ HYPOTHESIS REJECTED")
        print(f"     Format instructions didn't help")
        print(f"     Transfer failure is deeper than formatting")

    # Show detailed comparison
    print_section("Detailed Comparison")

    print(f"  {'Problem':<20} {'Control':<15} {'Treatment':<15} {'Expected':<15}")
    print("  " + "-" * 70)

    for i in range(min(len(control_results), len(treatment_results))):
        ctrl = control_results[i]
        treat = treatment_results[i]

        ctrl_mark = "✓" if ctrl["correct"] else "✗"
        treat_mark = "✓" if treat["correct"] else "✗"

        print(f"  {ctrl['problem']:<20} {ctrl['answer']:<13}{ctrl_mark} {treat['answer']:<13}{treat_mark} {ctrl['expected']:<15}")

    print("\n" + "="*70)
    print("✅ Formatting Hypothesis Test Complete")
    print("="*70 + "\n")

    session.close()


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Format Hypothesis Test\n")

    try:
        test_formatting_hypothesis()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
