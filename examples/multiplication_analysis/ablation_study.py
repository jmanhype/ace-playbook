#!/usr/bin/env python3
"""
Ablation Study - Evaluation 2

Measures the impact of the playbook by comparing performance:
- Control: Generator WITHOUT playbook strategies (empty context)
- Treatment: Generator WITH playbook strategies (PROD bullets)

Uses the SAME problems to isolate the effect of the playbook.

This directly answers: "Do the learned strategies actually help?"
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


def run_ablation_study(
    session,
    generator: CoTGenerator,
    num_problems: int = 20,
    test_seed: int = 777
):
    """
    Run ablation study comparing with/without playbook.

    Args:
        session: Database session
        generator: CoT generator
        num_problems: Number of problems to test
        test_seed: Seed for test problems
    """
    print_header("Evaluation 2: Ablation Study")

    domain_id = "multiplication-large"
    control_domain = "multiplication-ablation-control"
    treatment_domain = "multiplication-ablation-treatment"

    # Get playbook bullets
    bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
    bullet_contents = [b.content for b in bullets]

    print(f"📚 Playbook: {len(bullets)} PROD strategies")
    if bullets:
        for i, bullet in enumerate(bullets, 1):
            print(f"   {i}. {bullet.content[:70]}...")

    # Generate test problems
    print(f"\n🎲 Generating {num_problems} test problems (seed={test_seed})...")
    test_problems = generate_multiplication_problems(num_problems, seed=test_seed)
    print(f"✓ Generated {len(test_problems)} problems\n")

    # Run CONTROL (no playbook)
    print_section("CONTROL: Generator WITHOUT Playbook")
    print("Running same problems with empty playbook context...\n")

    control_correct = 0
    control_total = 0
    control_results = {}

    for idx, prob in enumerate(test_problems, 1):
        task_db = create_task_db(session, prob["problem"], prob["answer"], control_domain)

        # NO PLAYBOOK BULLETS
        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=[],  # EMPTY
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)
            is_correct = generator_output.answer.strip() == prob["answer"]

            if is_correct:
                control_correct += 1
            control_total += 1

            control_results[prob["problem"]] = {
                "problem": prob["problem"],
                "answer": generator_output.answer,
                "correct": is_correct
            }

            save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": [],
                "latency_ms": generator_output.latency_ms or 0,
                "prompt_tokens": generator_output.prompt_tokens or 0,
                "completion_tokens": generator_output.completion_tokens or 0
            })

            if idx % 5 == 0 or idx == 1:
                status = "✅" if is_correct else "❌"
                print(f"  [{idx}/{num_problems}] {prob['problem']} {status}")

        except Exception as e:
            print(f"\n❌ Error on problem {idx}: {e}")
            continue

    control_accuracy = (control_correct / control_total * 100) if control_total > 0 else 0

    print(f"\n  Control Accuracy: {control_accuracy:.1f}% ({control_correct}/{control_total})")

    # Run TREATMENT (with playbook)
    print_section("TREATMENT: Generator WITH Playbook")
    print("Running same problems with learned strategies...\n")

    treatment_correct = 0
    treatment_total = 0
    treatment_results = {}

    for idx, prob in enumerate(test_problems, 1):
        task_db = create_task_db(session, prob["problem"], prob["answer"], treatment_domain)

        # WITH PLAYBOOK BULLETS
        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=bullet_contents,  # WITH STRATEGIES
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)
            is_correct = generator_output.answer.strip() == prob["answer"]

            if is_correct:
                treatment_correct += 1
            treatment_total += 1

            treatment_results[prob["problem"]] = {
                "problem": prob["problem"],
                "answer": generator_output.answer,
                "correct": is_correct
            }

            save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": generator_output.bullets_referenced,
                "latency_ms": generator_output.latency_ms or 0,
                "prompt_tokens": generator_output.prompt_tokens or 0,
                "completion_tokens": generator_output.completion_tokens or 0
            })

            if idx % 5 == 0 or idx == 1:
                status = "✅" if is_correct else "❌"
                print(f"  [{idx}/{num_problems}] {prob['problem']} {status}")

        except Exception as e:
            print(f"\n❌ Error on problem {idx}: {e}")
            continue

    treatment_accuracy = (treatment_correct / treatment_total * 100) if treatment_total > 0 else 0

    print(f"\n  Treatment Accuracy: {treatment_accuracy:.1f}% ({treatment_correct}/{treatment_total})")

    # Calculate lift
    print_section("Ablation Study Results")

    lift = treatment_accuracy - control_accuracy
    relative_improvement = (lift / control_accuracy * 100) if control_accuracy > 0 else 0

    print(f"  Control (no playbook):    {control_accuracy:>6.1f}%")
    print(f"  Treatment (with playbook): {treatment_accuracy:>6.1f}%")
    print(f"  {'─'*40}")
    print(f"  Absolute lift:             {lift:>+6.1f}%")
    print(f"  Relative improvement:      {relative_improvement:>+6.1f}%")

    # Interpretation
    print_section("Interpretation")

    if lift > 10:
        print(f"  ✨ STRONG POSITIVE EFFECT")
        print(f"     Playbook strategies significantly improve accuracy!")
    elif lift > 5:
        print(f"  ✅ MODERATE POSITIVE EFFECT")
        print(f"     Playbook strategies provide meaningful improvement")
    elif lift > 0:
        print(f"  ⚡ SMALL POSITIVE EFFECT")
        print(f"     Playbook strategies help slightly")
    elif lift > -5:
        print(f"  ⚠️  NEGLIGIBLE EFFECT")
        print(f"     Playbook strategies have minimal impact")
    else:
        print(f"  ❌ NEGATIVE EFFECT")
        print(f"     Playbook strategies may be interfering")

    # Problem-by-problem analysis
    print_section("Problem-Level Analysis")

    improved = 0
    regressed = 0
    unchanged = 0

    for prob in test_problems:
        problem_str = prob["problem"]
        control_result = control_results.get(problem_str)
        treatment_result = treatment_results.get(problem_str)

        if not control_result or not treatment_result:
            continue  # Skip if a result is missing for this problem

        control_correct = control_result["correct"]
        treatment_correct = treatment_result["correct"]

        if not control_correct and treatment_correct:
            improved += 1
        elif control_correct and not treatment_correct:
            regressed += 1
        else:
            unchanged += 1

    print(f"  Improved (wrong → right):   {improved} problems")
    print(f"  Regressed (right → wrong):  {regressed} problems")
    print(f"  Unchanged:                  {unchanged} problems")

    net_improvement = improved - regressed
    print(f"  {'─'*40}")
    print(f"  Net improvement:            {net_improvement:+d} problems")

    print("\n" + "="*70)
    print("✅ Ablation Study Complete")
    print("="*70 + "\n")

    print("💡 Key Takeaway:")
    if lift > 5:
        print("   The playbook IS helping! Strategies provide measurable value.")
    else:
        print("   The playbook effect is small - may need better strategies.")
    print()


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Ablation Study\n")

    try:
        # Configure DSPy
        lm, model_name = configure_dspy_lm()
        dspy.configure(lm=lm)

        # Initialize database
        session = setup_database()

        # Initialize generator
        generator = CoTGenerator(model=model_name, temperature=0.7)
        print(f"✓ Initialized Generator with {model_name}\n")

        # Run ablation study
        run_ablation_study(
            session=session,
            generator=generator,
            num_problems=20,
            test_seed=777
        )

        session.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
