#!/usr/bin/env python3
"""
Difficulty Analysis - Evaluation 3

Analyzes how accuracy varies with problem difficulty.

Categories:
- Easy: Both operands < 100
- Medium: One operand < 100, or both < 1000
- Hard: Both operands >= 1000

Analyzes existing task outputs from the database.
"""

import sys
from collections import defaultdict
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiplication_analysis import (
    setup_database,
    print_header,
    print_section,
    calculate_problem_difficulty
)
from ace.models import Task, TaskOutput


def analyze_difficulty_distribution(session):
    """Analyze accuracy by problem difficulty."""

    print_header("Evaluation 3: Problem Difficulty Analysis")

    domain_id = "multiplication-large"

    # Get all tasks and outputs
    tasks = session.query(Task).filter_by(domain_id=domain_id).all()

    if not tasks:
        print("❌ No tasks found for domain 'multiplication-large'")
        print("   Run the main multiplication experiment first.")
        return

    print(f"📊 Analyzing {len(tasks)} tasks from database...\n")

    # Group by difficulty
    difficulty_stats = defaultdict(lambda: {"total": 0, "correct": 0, "problems": []})

    for task in tasks:
        # Parse problem from metadata
        if not task.metadata_json or "problem" not in task.metadata_json:
            continue

        problem = task.metadata_json["problem"]

        # Extract operands
        try:
            parts = problem.replace("×", "*").split("*")
            a = int(parts[0].strip())
            b = int(parts[1].strip())
        except (ValueError, IndexError, AttributeError):
            continue

        difficulty = calculate_problem_difficulty(a, b)

        # Get task output
        output = session.query(TaskOutput).filter_by(task_id=task.id).first()
        if not output:
            continue

        # Check correctness
        is_correct = output.answer.strip() == task.ground_truth.strip()

        difficulty_stats[difficulty]["total"] += 1
        if is_correct:
            difficulty_stats[difficulty]["correct"] += 1

        difficulty_stats[difficulty]["problems"].append({
            "problem": problem,
            "a": a,
            "b": b,
            "answer": output.answer,
            "expected": task.ground_truth,
            "correct": is_correct
        })

    # Print results
    print_section("Accuracy by Difficulty")

    print(f"{'Difficulty':<12} {'Total':<8} {'Correct':<10} {'Accuracy':<12} {'Details'}")
    print("-" * 70)

    for difficulty in ["easy", "medium", "hard"]:
        stats = difficulty_stats[difficulty]
        if stats["total"] == 0:
            print(f"{difficulty.capitalize():<12} {0:<8} {0:<10} {'N/A':<12}")
            continue

        accuracy = stats["correct"] / stats["total"] * 100
        details = f"({stats['correct']}/{stats['total']})"

        print(f"{difficulty.capitalize():<12} {stats['total']:<8} {stats['correct']:<10} {accuracy:>6.1f}%      {details}")

    # Show sample problems per difficulty
    print_section("Sample Problems by Difficulty")

    for difficulty in ["easy", "medium", "hard"]:
        stats = difficulty_stats[difficulty]
        if stats["total"] == 0:
            continue

        print(f"\n{difficulty.upper()} Problems:")
        print(f"  Criteria: {get_difficulty_criteria(difficulty)}\n")

        # Show first 3 correct and first 3 incorrect
        correct_samples = [p for p in stats["problems"] if p["correct"]][:3]
        incorrect_samples = [p for p in stats["problems"] if not p["correct"]][:3]

        if correct_samples:
            print("  ✅ Correct:")
            for p in correct_samples:
                print(f"     {p['problem']} = {p['answer']}")

        if incorrect_samples:
            print("  ❌ Incorrect:")
            for p in incorrect_samples:
                print(f"     {p['problem']}")
                print(f"        Expected: {p['expected']}, Got: {p['answer']}")

    # Statistical summary
    print_section("Statistical Summary")

    total_easy = difficulty_stats["easy"]["total"]
    total_medium = difficulty_stats["medium"]["total"]
    total_hard = difficulty_stats["hard"]["total"]

    print(f"  Distribution:")
    print(f"    Easy:   {total_easy} problems ({total_easy/len(tasks)*100:.1f}%)")
    print(f"    Medium: {total_medium} problems ({total_medium/len(tasks)*100:.1f}%)")
    print(f"    Hard:   {total_hard} problems ({total_hard/len(tasks)*100:.1f}%)")

    # Calculate correlation
    if total_easy > 0 and total_hard > 0:
        easy_acc = difficulty_stats["easy"]["correct"] / total_easy * 100
        hard_acc = difficulty_stats["hard"]["correct"] / total_hard * 100
        diff = easy_acc - hard_acc

        print(f"\n  Difficulty Impact:")
        print(f"    Easy vs Hard accuracy gap: {diff:+.1f}%")

        if diff > 15:
            print(f"    ⚠️  Large difficulty gap - model struggles with hard problems")
        elif diff > 5:
            print(f"    ⚡ Moderate difficulty gap - some improvement possible")
        else:
            print(f"    ✨ Small difficulty gap - relatively consistent performance")

    print("\n" + "="*70)
    print("✅ Difficulty Analysis Complete")
    print("="*70 + "\n")


def get_difficulty_criteria(difficulty: str) -> str:
    """Get human-readable criteria for difficulty level."""
    criteria = {
        "easy": "Both operands < 100",
        "medium": "One operand < 100, or both < 1000",
        "hard": "Both operands >= 1000"
    }
    return criteria.get(difficulty, "Unknown")


def main():
    """Main entry point."""
    try:
        session = setup_database()
        analyze_difficulty_distribution(session)
        session.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
