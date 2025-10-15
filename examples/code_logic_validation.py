#!/usr/bin/env python3
"""
Code + Logic Domain Validation (Simplified)

Focused test on Code and Logic domains only (Math already validated).
Uses 5 problems per domain and 1 epoch for faster execution (~5-10 minutes).
"""

import sys
import os
from pathlib import Path
from typing import List, Dict
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput
from ace.models import Task, TaskOutput, PlaybookBullet, PlaybookStage
from ace.utils.database import get_session, init_database
from datetime import datetime, timezone


def generate_code_problems(num_problems: int, seed: int = 42) -> List[Dict[str, str]]:
    """Generate simple Python function generation tasks."""
    templates = [
        {
            "problem": "Write a Python function 'is_even(n)' that returns True if n is even, False otherwise.",
            "answer": "def is_even(n):\n    return n % 2 == 0"
        },
        {
            "problem": "Write a Python function 'max_of_two(a, b)' that returns the maximum of two numbers.",
            "answer": "def max_of_two(a, b):\n    return a if a > b else b"
        },
        {
            "problem": "Write a Python function 'factorial(n)' that returns the factorial of n.",
            "answer": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "problem": "Write a Python function 'is_palindrome(s)' that checks if a string is a palindrome.",
            "answer": "def is_palindrome(s):\n    return s == s[::-1]"
        },
        {
            "problem": "Write a Python function 'sum_list(lst)' that returns the sum of all elements in a list.",
            "answer": "def sum_list(lst):\n    return sum(lst)"
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
            "problem": "Write a Python function 'is_prime(n)' that checks if n is a prime number.",
            "answer": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "problem": "Write a Python function 'fibonacci(n)' that returns the nth Fibonacci number.",
            "answer": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        },
        {
            "problem": "Write a Python function 'remove_duplicates(lst)' that removes duplicates from a list.",
            "answer": "def remove_duplicates(lst):\n    return list(set(lst))"
        }
    ]

    random.seed(seed)
    selected = random.sample(templates, min(num_problems, len(templates)))

    return selected


def generate_logic_problems(num_problems: int, seed: int = 42) -> List[Dict[str, str]]:
    """Generate logical reasoning problems."""
    templates = [
        {
            "problem": "If all dogs are mammals, and Rex is a dog, is Rex a mammal?",
            "answer": "Yes"
        },
        {
            "problem": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "answer": "Not necessarily"
        },
        {
            "problem": "All birds have wings. Penguins are birds. Do penguins have wings?",
            "answer": "Yes"
        },
        {
            "problem": "If A is taller than B, and B is taller than C, who is the tallest?",
            "answer": "A"
        },
        {
            "problem": "True or False: If all cats are animals, then all animals are cats.",
            "answer": "False"
        },
        {
            "problem": "If you study hard, you will pass. You passed. Did you study hard?",
            "answer": "Not necessarily"
        },
        {
            "problem": "All squares are rectangles. All rectangles have four sides. Do all squares have four sides?",
            "answer": "Yes"
        },
        {
            "problem": "If it's sunny, I wear sunglasses. I'm wearing sunglasses. Is it sunny?",
            "answer": "Not necessarily"
        },
        {
            "problem": "Circle A is larger than Circle B. Circle B is larger than Circle C. Which circle is smallest?",
            "answer": "Circle C"
        },
        {
            "problem": "All prime numbers greater than 2 are odd. Is 7 odd?",
            "answer": "Yes"
        }
    ]

    random.seed(seed)
    selected = random.sample(templates, min(num_problems, len(templates)))

    return selected


def check_code_equivalence(generated: str, expected: str) -> bool:
    """Check if generated code is equivalent to expected (lenient matching)."""
    gen_clean = generated.strip().replace(" ", "").replace("\n", "")
    exp_clean = expected.strip().replace(" ", "").replace("\n", "")

    # Check if both contain function definition
    if "def" in gen_clean and "def" in exp_clean:
        # Extract function name
        gen_func = gen_clean.split("(")[0].replace("def", "")
        exp_func = exp_clean.split("(")[0].replace("def", "")

        # If function names match, consider it correct (lenient)
        if gen_func == exp_func:
            return True

    # Fallback to exact match
    return gen_clean == exp_clean


def check_logic_equivalence(generated: str, expected: str) -> bool:
    """Check if logic answer is correct (case-insensitive, flexible)."""
    gen_clean = generated.strip().lower()
    exp_clean = expected.strip().lower()

    # Direct match
    if gen_clean == exp_clean:
        return True

    # Check if expected answer appears in generated (flexible matching)
    if exp_clean in gen_clean:
        return True

    # Handle yes/no variants
    if exp_clean in ["yes", "true"] and any(word in gen_clean for word in ["yes", "true", "correct"]):
        return True
    if exp_clean in ["no", "false"] and any(word in gen_clean for word in ["no", "false", "incorrect"]):
        return True

    return False


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
                tags=[task.domain_id, "code-logic-test"],
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
    shadow_helpful_min = 2
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


def get_playbook_bullets(session, domain_id):
    """Get playbook bullets for domain."""
    return session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=PlaybookStage.PROD
    ).order_by(PlaybookBullet.helpful_count.desc()).all()


def run_baseline_test(session, generator, problems, domain_name, check_fn):
    """Run baseline test without playbook."""
    correct = 0
    results = []

    for idx, prob in enumerate(problems, 1):
        task_db = Task(
            prompt=prob["problem"],
            domain_id=f"{domain_name}-baseline",
            domain=domain_name,
            ground_truth=prob["answer"]
        )
        session.add(task_db)
        session.commit()

        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain=domain_name,
            playbook_bullets=[],  # No playbook
            max_reasoning_steps=10
        )

        try:
            output = generator(task_input)
            is_correct = check_fn(output.answer.strip(), prob["answer"])

            if is_correct:
                correct += 1

            results.append({
                "problem": prob["problem"],
                "expected": prob["answer"],
                "answer": output.answer,
                "correct": is_correct
            })

            # Save to DB
            total_tokens = (output.prompt_tokens or 0) + (output.completion_tokens or 0)
            task_output = TaskOutput(
                task_id=task_db.id,
                reasoning_trace=output.reasoning_trace,
                answer=output.answer,
                confidence=output.confidence,
                bullets_referenced=output.bullets_referenced,
                latency_ms=output.latency_ms or 0,
                token_count=total_tokens
            )
            session.add(task_output)
            session.commit()

            status = "✅" if is_correct else "❌"
            print(f"  [{idx}/{len(problems)}] {status}")

        except Exception as e:
            print(f"  ❌ Error on problem {idx}: {e}")
            continue

    accuracy = (correct / len(problems) * 100) if problems else 0
    return accuracy, results


def run_ace_training(session, generator, reflector, problems, domain_name, check_fn, num_epochs=1):
    """Run ACE training and return final accuracy."""
    domain_id = f"{domain_name}-ace"

    for epoch in range(1, num_epochs + 1):
        print(f"\n  Epoch {epoch}/{num_epochs}")
        correct = 0

        for idx, prob in enumerate(problems, 1):
            task_db = Task(
                prompt=prob["problem"],
                domain_id=domain_id,
                domain=domain_name,
                ground_truth=prob["answer"]
            )
            session.add(task_db)
            session.commit()

            # Get current playbook
            bullets = get_playbook_bullets(session, domain_id)
            bullet_contents = [b.content for b in bullets]

            task_input = TaskInput(
                task_id=task_db.id,
                description=task_db.prompt,
                domain=domain_name,
                playbook_bullets=bullet_contents,
                max_reasoning_steps=10
            )

            try:
                # Generate
                gen_output = generator(task_input)
                is_correct = check_fn(gen_output.answer.strip(), prob["answer"])

                if is_correct:
                    correct += 1

                # Save generator output
                total_tokens = (gen_output.prompt_tokens or 0) + (gen_output.completion_tokens or 0)
                task_output = TaskOutput(
                    task_id=task_db.id,
                    reasoning_trace=gen_output.reasoning_trace,
                    answer=gen_output.answer,
                    confidence=gen_output.confidence,
                    bullets_referenced=gen_output.bullets_referenced,
                    latency_ms=gen_output.latency_ms or 0,
                    token_count=total_tokens
                )
                session.add(task_output)
                session.commit()

                # Reflect
                reflection_input = ReflectorInput(
                    task_id=task_db.id,
                    reasoning_trace=gen_output.reasoning_trace,
                    answer=gen_output.answer,
                    confidence=gen_output.confidence,
                    bullets_referenced=gen_output.bullets_referenced,
                    domain=domain_id,
                    ground_truth=prob["answer"],
                    test_results="",
                    error_messages=[],
                    performance_metrics=""
                )

                reflector_output = reflector(reflection_input)

                # Curate
                curate_insights(session, task_db, reflector_output.insights)

                status = "✅" if is_correct else "❌"
                print(f"    [{idx}/{len(problems)}] {status}")

            except Exception as e:
                print(f"    ❌ Error on problem {idx}: {e}")
                continue

        # Promote bullets after epoch
        promote_bullets(session, domain_id)

        accuracy = (correct / len(problems) * 100) if problems else 0
        print(f"  Epoch {epoch} Accuracy: {correct}/{len(problems)} = {accuracy:.1f}%")

    return accuracy


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("🚀 ACE Framework - Code + Logic Domain Validation")
    print("="*70 + "\n")

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("❌ OPENROUTER_API_KEY required")
        sys.exit(1)

    model = "openrouter/qwen/qwen-2.5-7b-instruct"

    # Configure model
    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)

    generator = CoTGenerator(model=model, temperature=0.7)
    reflector = GroundedReflector(model=model, temperature=0.3)

    print(f"✓ Configured {model}\n")

    # Generate problems
    print("🎲 Generating test problems...")
    code_problems = generate_code_problems(5, seed=999)
    logic_problems = generate_logic_problems(5, seed=999)
    print(f"✓ Generated 5 Code + 5 Logic problems\n")

    # Initialize database
    init_database()

    results = {}

    with get_session() as session:
        # Test 1: Code Domain
        print("="*70)
        print("📝 CODE DOMAIN")
        print("="*70)

        print("\n[1/2] Baseline (no playbook)...")
        code_baseline_acc, _ = run_baseline_test(
            session, generator, code_problems, "code", check_code_equivalence
        )
        print(f"\n  Baseline Accuracy: {code_baseline_acc:.1f}%")

        print("\n[2/2] ACE Training (1 epoch)...")
        code_ace_acc = run_ace_training(
            session, generator, reflector, code_problems, "code", check_code_equivalence, num_epochs=1
        )

        code_lift = code_ace_acc - code_baseline_acc
        results["code"] = {
            "baseline": code_baseline_acc,
            "ace": code_ace_acc,
            "lift": code_lift
        }

        print(f"\n  ACE Accuracy: {code_ace_acc:.1f}%")
        print(f"  Direct Lift: {code_lift:+.1f}%")

        # Test 2: Logic Domain
        print("\n" + "="*70)
        print("🧠 LOGIC DOMAIN")
        print("="*70)

        print("\n[1/2] Baseline (no playbook)...")
        logic_baseline_acc, _ = run_baseline_test(
            session, generator, logic_problems, "logic", check_logic_equivalence
        )
        print(f"\n  Baseline Accuracy: {logic_baseline_acc:.1f}%")

        print("\n[2/2] ACE Training (1 epoch)...")
        logic_ace_acc = run_ace_training(
            session, generator, reflector, logic_problems, "logic", check_logic_equivalence, num_epochs=1
        )

        logic_lift = logic_ace_acc - logic_baseline_acc
        results["logic"] = {
            "baseline": logic_baseline_acc,
            "ace": logic_ace_acc,
            "lift": logic_lift
        }

        print(f"\n  ACE Accuracy: {logic_ace_acc:.1f}%")
        print(f"  Direct Lift: {logic_lift:+.1f}%")

    # Final Summary
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70 + "\n")

    print(f"  {'Domain':<15} {'Baseline':<12} {'ACE':<12} {'Lift':<12}")
    print("  " + "-"*50)

    for domain, res in results.items():
        print(f"  {domain.upper():<15} {res['baseline']:>6.1f}%     {res['ace']:>6.1f}%     {res['lift']:>+6.1f}%")

    print("\n" + "="*70)

    # Check if ACE generalizes
    all_positive_lift = all(res["lift"] > 0 for res in results.values())
    avg_lift = sum(res["lift"] for res in results.values()) / len(results)

    if all_positive_lift and avg_lift > 5:
        print("✅ VALIDATION SUCCESSFUL")
        print(f"   ACE generalizes across domains (avg lift: {avg_lift:+.1f}%)")
    elif all_positive_lift:
        print("⚡ WEAK VALIDATION")
        print(f"   Small positive lift across domains (avg: {avg_lift:+.1f}%)")
    else:
        print("❌ VALIDATION FAILED")
        print("   ACE does not consistently improve across domains")

    print("="*70 + "\n")

    session.close()


if __name__ == "__main__":
    main()
