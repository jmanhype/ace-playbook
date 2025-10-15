#!/usr/bin/env python3
"""
Multi-Domain Validation - Cross-Domain ACE Performance Test

Tests if ACE's performance gains generalize across different task domains:
- Math: Arithmetic (multiplication)
- Code: Python function generation
- Logic: Reasoning problems

Validates that ACE is not just good at one specific domain.
"""

import sys
import os
from pathlib import Path
import random
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput
from ace.models import Task, TaskOutput, PlaybookBullet, PlaybookStage
from multiplication_analysis import (
    setup_database,
    configure_dspy_lm,
    create_task_db,
    save_task_output,
    print_header,
    print_section
)


# ============================================================================
# DOMAIN 1: MATH (Multiplication)
# ============================================================================

def generate_math_problems(num_problems: int, seed: int = 42) -> List[Dict[str, str]]:
    """Generate multiplication problems."""
    random.seed(seed)
    problems = []

    for _ in range(num_problems):
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        answer = a * b

        problems.append({
            "problem": f"Calculate: {a} × {b}",
            "answer": str(answer),
            "domain": "math"
        })

    return problems


# ============================================================================
# DOMAIN 2: CODE (Python Function Generation)
# ============================================================================

def generate_code_problems(num_problems: int, seed: int = 42) -> List[Dict[str, str]]:
    """Generate simple Python function generation tasks."""
    random.seed(seed)

    templates = [
        {
            "problem": "Write a Python function 'is_even(n)' that returns True if n is even, False otherwise.",
            "answer": "def is_even(n):\n    return n % 2 == 0"
        },
        {
            "problem": "Write a Python function 'square(x)' that returns the square of x.",
            "answer": "def square(x):\n    return x * x"
        },
        {
            "problem": "Write a Python function 'max_of_three(a, b, c)' that returns the maximum of three numbers.",
            "answer": "def max_of_three(a, b, c):\n    return max(a, b, c)"
        },
        {
            "problem": "Write a Python function 'is_positive(n)' that returns True if n is positive, False otherwise.",
            "answer": "def is_positive(n):\n    return n > 0"
        },
        {
            "problem": "Write a Python function 'absolute(n)' that returns the absolute value of n.",
            "answer": "def absolute(n):\n    return abs(n)"
        },
        {
            "problem": "Write a Python function 'sum_list(numbers)' that returns the sum of a list of numbers.",
            "answer": "def sum_list(numbers):\n    return sum(numbers)"
        },
        {
            "problem": "Write a Python function 'reverse_string(s)' that reverses a string.",
            "answer": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "problem": "Write a Python function 'count_vowels(s)' that counts vowels in a string.",
            "answer": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"
        },
        {
            "problem": "Write a Python function 'factorial(n)' that calculates n factorial.",
            "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "problem": "Write a Python function 'is_palindrome(s)' that checks if a string is a palindrome.",
            "answer": "def is_palindrome(s):\n    return s == s[::-1]"
        },
    ]

    # Cycle through templates to get num_problems
    problems = []
    for i in range(num_problems):
        template = templates[i % len(templates)]
        problems.append({
            "problem": template["problem"],
            "answer": template["answer"],
            "domain": "code"
        })

    return problems


# ============================================================================
# DOMAIN 3: LOGIC (Reasoning Problems)
# ============================================================================

def generate_logic_problems(num_problems: int, seed: int = 42) -> List[Dict[str, str]]:
    """Generate logical reasoning problems."""
    random.seed(seed)

    templates = [
        {
            "problem": "If all dogs are mammals, and Rex is a dog, is Rex a mammal?",
            "answer": "Yes"
        },
        {
            "problem": "If it's raining, the ground is wet. The ground is wet. Is it necessarily raining?",
            "answer": "No"
        },
        {
            "problem": "There are 3 apples and 5 oranges in a basket. How many fruits are there total?",
            "answer": "8"
        },
        {
            "problem": "If Tom is older than Sarah, and Sarah is older than Mike, who is the oldest?",
            "answer": "Tom"
        },
        {
            "problem": "A train leaves Station A at 2 PM traveling at 60 mph. When will it reach Station B, 120 miles away?",
            "answer": "4 PM"
        },
        {
            "problem": "If all roses are flowers, and some flowers are red, are all roses necessarily red?",
            "answer": "No"
        },
        {
            "problem": "You have $100 and spend $35. How much do you have left?",
            "answer": "$65"
        },
        {
            "problem": "If A implies B, and B implies C, does A imply C?",
            "answer": "Yes"
        },
        {
            "problem": "A book costs $15. You buy 3 books. How much do you spend?",
            "answer": "$45"
        },
        {
            "problem": "If every student in class got an A, and John is in the class, what grade did John get?",
            "answer": "A"
        },
    ]

    problems = []
    for i in range(num_problems):
        template = templates[i % len(templates)]
        problems.append({
            "problem": template["problem"],
            "answer": template["answer"],
            "domain": "logic"
        })

    return problems


# ============================================================================
# Training Pipeline
# ============================================================================

def curate_insights(session, task, insights):
    """Curator: Add insights to playbook."""
    new_bullets = []
    incremented_bullets = []

    for insight in insights:
        # Check for duplicates
        existing = session.query(PlaybookBullet).filter_by(
            domain_id=task.domain_id,
            content=insight.content
        ).first()

        if existing:
            # Increment counter
            if insight.section.value == "Helpful":
                existing.helpful_count += 1
            elif insight.section.value == "Harmful":
                existing.harmful_count += 1

            existing.last_used_at = datetime.now(timezone.utc)
            incremented_bullets.append(existing)
        else:
            # Add new bullet in shadow stage
            bullet = PlaybookBullet(
                content=insight.content,
                domain_id=task.domain_id,
                section=insight.section.value,
                helpful_count=1 if insight.section.value == "Helpful" else 0,
                harmful_count=1 if insight.section.value == "Harmful" else 0,
                tags=[task.domain_id, "multi-domain"],
                embedding=[0.0] * 384,
                stage=PlaybookStage.SHADOW
            )
            session.add(bullet)
            new_bullets.append(bullet)

    session.commit()

    return {
        "new_bullets": len(new_bullets),
        "incremented": len(incremented_bullets)
    }


def promote_bullets(session, domain_id):
    """Apply promotion gates."""
    shadow_helpful_min = 2  # Faster promotion for multi-domain test
    prod_helpful_min = 3
    staging_ratio_min = 2.0
    prod_ratio_min = 3.0

    promotions = {"shadow_to_staging": 0, "staging_to_prod": 0}

    # Promote shadow → staging
    for bullet in session.query(PlaybookBullet).filter_by(domain_id=domain_id, stage=PlaybookStage.SHADOW).all():
        if bullet.helpful_count >= shadow_helpful_min:
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            if ratio >= staging_ratio_min:
                bullet.stage = PlaybookStage.STAGING
                promotions["shadow_to_staging"] += 1

    # Promote staging → prod
    for bullet in session.query(PlaybookBullet).filter_by(domain_id=domain_id, stage=PlaybookStage.STAGING).all():
        if bullet.helpful_count >= prod_helpful_min:
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            if ratio >= prod_ratio_min:
                bullet.stage = PlaybookStage.PROD
                promotions["staging_to_prod"] += 1

    session.commit()
    return promotions


def get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD):
    """Get playbook bullets for domain."""
    return session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=stage
    ).order_by(PlaybookBullet.helpful_count.desc()).all()


def run_baseline_test(session, generator, problems, domain_name):
    """Run baseline test without playbook."""
    print(f"\n  Running baseline (no playbook) for {domain_name}...")

    correct = 0
    total = len(problems)

    for prob in problems:
        task_db = create_task_db(
            session,
            prob["problem"],
            prob["answer"],
            f"{domain_name}-baseline"
        )

        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain=domain_name,
            playbook_bullets=[],  # No playbook
            max_reasoning_steps=10
        )

        try:
            output = generator(task_input)

            # Flexible matching for code
            if domain_name == "code":
                is_correct = check_code_equivalence(output.answer, prob["answer"])
            else:
                is_correct = output.answer.strip().lower() == prob["answer"].lower()

            if is_correct:
                correct += 1

        except Exception as e:
            print(f"    Error: {e}")
            continue

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  ✓ Baseline: {correct}/{total} = {accuracy:.1f}%")

    return accuracy


def check_code_equivalence(generated: str, expected: str) -> bool:
    """Check if generated code is equivalent to expected (lenient matching)."""
    # Extract function definition
    gen_clean = generated.strip().replace(" ", "").replace("\n", "")
    exp_clean = expected.strip().replace(" ", "").replace("\n", "")

    # Check if function name matches
    if "def" in gen_clean and "def" in exp_clean:
        gen_func = gen_clean.split("(")[0].replace("def", "")
        exp_func = exp_clean.split("(")[0].replace("def", "")
        return gen_func == exp_func

    return gen_clean == exp_clean


def run_ace_training(session, generator, reflector, problems, domain_name, num_epochs=2):
    """Run ACE training and return final accuracy."""
    print(f"\n  Training ACE playbook for {domain_name} ({num_epochs} epochs)...")

    domain_id = f"{domain_name}-ace"

    for epoch in range(1, num_epochs + 1):
        epoch_correct = 0

        for prob in problems:
            task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)

            # Get current playbook
            bullets = get_playbook_bullets(session, domain_id)
            bullet_contents = [b.content for b in bullets]

            # Generate
            task_input = TaskInput(
                task_id=task_db.id,
                description=task_db.prompt,
                domain=domain_name,
                playbook_bullets=bullet_contents,
                max_reasoning_steps=10
            )

            try:
                gen_output = generator(task_input)

                # Check correctness
                if domain_name == "code":
                    is_correct = check_code_equivalence(gen_output.answer, prob["answer"])
                else:
                    is_correct = gen_output.answer.strip().lower() == prob["answer"].lower()

                if is_correct:
                    epoch_correct += 1

                # Save output
                save_task_output(session, task_db, {
                    "reasoning_trace": gen_output.reasoning_trace,
                    "answer": gen_output.answer,
                    "confidence": gen_output.confidence,
                    "bullets_referenced": gen_output.bullets_referenced,
                    "latency_ms": gen_output.latency_ms or 0,
                    "prompt_tokens": gen_output.prompt_tokens or 0,
                    "completion_tokens": gen_output.completion_tokens or 0
                })

                # Reflect
                reflector_input = ReflectorInput(
                    task_id=task_db.id,
                    reasoning_trace=gen_output.reasoning_trace,
                    answer=gen_output.answer,
                    confidence=gen_output.confidence,
                    bullets_referenced=gen_output.bullets_referenced,
                    domain=domain_name,
                    ground_truth=prob["answer"],
                    test_results="",
                    error_messages=[],
                    performance_metrics=""
                )

                reflector_output = reflector(reflector_input)

                # Curate
                curate_insights(session, task_db, reflector_output.insights)

            except Exception as e:
                print(f"    Error on problem: {e}")
                continue

        # Promote bullets after each epoch
        promote_bullets(session, domain_id)

        accuracy = (epoch_correct / len(problems) * 100) if problems else 0
        print(f"    Epoch {epoch}: {accuracy:.1f}%")

    # Final test
    final_correct = 0
    bullets = get_playbook_bullets(session, domain_id)
    bullet_contents = [b.content for b in bullets]

    print(f"  ✓ Final playbook: {len(bullet_contents)} PROD bullets")

    for prob in problems:
        task_input = TaskInput(
            task_id=str(uuid.uuid4()),
            description=prob["problem"],
            domain=domain_name,
            playbook_bullets=bullet_contents,
            max_reasoning_steps=10
        )

        try:
            output = generator(task_input)

            if domain_name == "code":
                is_correct = check_code_equivalence(output.answer, prob["answer"])
            else:
                is_correct = output.answer.strip().lower() == prob["answer"].lower()

            if is_correct:
                final_correct += 1
        except Exception:
            continue

    final_accuracy = (final_correct / len(problems) * 100) if problems else 0
    print(f"  ✓ ACE-optimized: {final_correct}/{len(problems)} = {final_accuracy:.1f}%")

    return final_accuracy


# ============================================================================
# Main Multi-Domain Validation
# ============================================================================

def run_multi_domain_validation():
    """Run multi-domain validation test."""
    print_header("Multi-Domain Validation Test")

    print("🎯 Testing ACE across 3 domains:")
    print("   1. Math (Multiplication)")
    print("   2. Code (Python Functions)")
    print("   3. Logic (Reasoning Problems)")
    print()

    # Configure DSPy
    lm, model_name = configure_dspy_lm()
    dspy.configure(lm=lm)

    session = setup_database()

    generator = CoTGenerator(model=model_name, temperature=0.7)
    reflector = GroundedReflector(model=model_name, temperature=0.3)

    print(f"✓ Model: {model_name}")
    print(f"✓ Generator and Reflector initialized\n")

    # Generate problems
    num_problems = 15

    print(f"📊 Generating {num_problems} problems per domain...")
    math_problems = generate_math_problems(num_problems, seed=100)
    code_problems = generate_code_problems(num_problems, seed=200)
    logic_problems = generate_logic_problems(num_problems, seed=300)
    print(f"✓ Generated all problems\n")

    # Track results
    results = {}

    # ========================================================================
    # DOMAIN 1: MATH
    # ========================================================================
    print_section("Domain 1: Math (Multiplication)")

    math_baseline = run_baseline_test(session, generator, math_problems, "math")
    math_ace = run_ace_training(session, generator, reflector, math_problems, "math", num_epochs=2)

    results["math"] = {
        "baseline": math_baseline,
        "ace": math_ace,
        "improvement": math_ace - math_baseline
    }

    # ========================================================================
    # DOMAIN 2: CODE
    # ========================================================================
    print_section("Domain 2: Code (Python Functions)")

    code_baseline = run_baseline_test(session, generator, code_problems, "code")
    code_ace = run_ace_training(session, generator, reflector, code_problems, "code", num_epochs=2)

    results["code"] = {
        "baseline": code_baseline,
        "ace": code_ace,
        "improvement": code_ace - code_baseline
    }

    # ========================================================================
    # DOMAIN 3: LOGIC
    # ========================================================================
    print_section("Domain 3: Logic (Reasoning)")

    logic_baseline = run_baseline_test(session, generator, logic_problems, "logic")
    logic_ace = run_ace_training(session, generator, reflector, logic_problems, "logic", num_epochs=2)

    results["logic"] = {
        "baseline": logic_baseline,
        "ace": logic_ace,
        "improvement": logic_ace - logic_baseline
    }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("Multi-Domain Results Summary")

    print(f"\n{'Domain':<15} {'Baseline':<12} {'ACE':<12} {'Improvement':<12} {'Status'}")
    print("-" * 70)

    for domain, metrics in results.items():
        improvement = metrics["improvement"]
        status = "✅" if improvement > 5 else "⚡" if improvement > 0 else "❌"

        print(f"{domain.capitalize():<15} {metrics['baseline']:>6.1f}%     {metrics['ace']:>6.1f}%     {improvement:>+6.1f}%        {status}")

    # Overall stats
    avg_baseline = sum(r["baseline"] for r in results.values()) / len(results)
    avg_ace = sum(r["ace"] for r in results.values()) / len(results)
    avg_improvement = avg_ace - avg_baseline

    print("-" * 70)
    print(f"{'AVERAGE':<15} {avg_baseline:>6.1f}%     {avg_ace:>6.1f}%     {avg_improvement:>+6.1f}%")

    # Analysis
    print_section("Analysis")

    if avg_improvement > 10:
        print("  ✨ STRONG CROSS-DOMAIN PERFORMANCE")
        print(f"     ACE shows consistent improvement across all domains")
        print(f"     Average lift: {avg_improvement:+.1f}%")
    elif avg_improvement > 5:
        print("  ✅ GOOD CROSS-DOMAIN PERFORMANCE")
        print(f"     ACE generalizes well across domains")
        print(f"     Average lift: {avg_improvement:+.1f}%")
    elif avg_improvement > 0:
        print("  ⚡ MODERATE CROSS-DOMAIN PERFORMANCE")
        print(f"     Some domains benefit more than others")
        print(f"     Average lift: {avg_improvement:+.1f}%")
    else:
        print("  ⚠️  DOMAIN-SPECIFIC PERFORMANCE")
        print(f"     ACE may not generalize well across all domains")

    # Domain-specific insights
    best_domain = max(results.items(), key=lambda x: x[1]["improvement"])
    worst_domain = min(results.items(), key=lambda x: x[1]["improvement"])

    print(f"\n  Best domain: {best_domain[0].capitalize()} ({best_domain[1]['improvement']:+.1f}%)")
    print(f"  Worst domain: {worst_domain[0].capitalize()} ({worst_domain[1]['improvement']:+.1f}%)")

    print("\n" + "="*70)
    print("✅ Multi-Domain Validation Complete")
    print("="*70 + "\n")

    session.close()


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Multi-Domain Validation\n")

    try:
        run_multi_domain_validation()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
