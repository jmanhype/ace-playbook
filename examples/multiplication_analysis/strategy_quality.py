#!/usr/bin/env python3
"""
Strategy Quality Analysis - Evaluation 4

Analyzes which playbook strategies actually helped improve accuracy.

Examines:
- Which bullets were referenced in correct vs incorrect answers
- Correlation between bullet usage and success
- Most effective strategies
- Strategies that need improvement
"""

import sys
from collections import defaultdict, Counter
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiplication_analysis import (
    setup_database,
    print_header,
    print_section
)
from ace.models import Task, TaskOutput, PlaybookBullet, PlaybookStage


def analyze_strategy_effectiveness(session):
    """Analyze which strategies correlate with correct answers."""

    print_header("Evaluation 4: Strategy Quality Analysis")

    domain_id = "multiplication-large"

    # Get all bullets
    all_bullets = session.query(PlaybookBullet).filter_by(
        domain_id=domain_id
    ).all()

    if not all_bullets:
        print("❌ No playbook bullets found for domain 'multiplication-large'")
        print("   Run the main multiplication experiment first.")
        return

    print(f"📚 Analyzing {len(all_bullets)} playbook strategies...\n")

    # Map bullet ID to bullet
    bullet_map = {b.id: b for b in all_bullets}

    # Track bullet usage in correct vs incorrect
    bullet_stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})

    # Get all tasks and outputs
    tasks = session.query(Task).filter_by(domain_id=domain_id).all()

    correct_total = 0
    incorrect_total = 0

    for task in tasks:
        output = session.query(TaskOutput).filter_by(task_id=task.id).first()
        if not output:
            continue

        is_correct = output.answer.strip() == task.ground_truth.strip()

        if is_correct:
            correct_total += 1
        else:
            incorrect_total += 1

        # Track which bullets were referenced
        for bullet_id in output.bullets_referenced:
            if bullet_id in bullet_map:
                bullet_stats[bullet_id]["total"] += 1
                if is_correct:
                    bullet_stats[bullet_id]["correct"] += 1
                else:
                    bullet_stats[bullet_id]["incorrect"] += 1

    total_tasks = correct_total + incorrect_total

    # Calculate effectiveness scores
    bullet_effectiveness = []

    for bullet_id, stats in bullet_stats.items():
        if bullet_id not in bullet_map:
            continue

        bullet = bullet_map[bullet_id]

        # Calculate success rate when this bullet was used
        success_rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0

        # Calculate baseline (overall accuracy)
        baseline_rate = correct_total / total_tasks * 100 if total_tasks > 0 else 0

        # Effectiveness = how much better than baseline
        effectiveness = success_rate - baseline_rate

        bullet_effectiveness.append({
            "bullet": bullet,
            "stats": stats,
            "success_rate": success_rate,
            "baseline": baseline_rate,
            "effectiveness": effectiveness
        })

    # Sort by effectiveness
    bullet_effectiveness.sort(key=lambda x: x["effectiveness"], reverse=True)

    # Print results
    print_section(f"Overall Performance Baseline")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Correct: {correct_total} ({correct_total/total_tasks*100:.1f}%)")
    print(f"  Incorrect: {incorrect_total} ({incorrect_total/total_tasks*100:.1f}%)")

    # Most effective strategies
    print_section("Most Effective Strategies (Better than Baseline)")

    effective = [b for b in bullet_effectiveness if b["effectiveness"] > 0]

    if effective:
        print(f"\n{'#':<4} {'Uses':<7} {'Success':<10} {'Baseline':<10} {'Lift':<10} {'Stage':<10} {'Strategy'}")
        print("-" * 120)

        for i, item in enumerate(effective[:10], 1):
            bullet = item["bullet"]
            stats = item["stats"]
            success = item["success_rate"]
            baseline = item["baseline"]
            lift = item["effectiveness"]

            strategy_preview = bullet.content[:60] + "..." if len(bullet.content) > 60 else bullet.content

            print(f"{i:<4} {stats['total']:<7} {success:>6.1f}%    {baseline:>6.1f}%    {lift:>+6.1f}%    {bullet.stage.value:<10} {strategy_preview}")
    else:
        print("  No strategies performed better than baseline")

    # Least effective / harmful strategies
    print_section("Strategies Needing Improvement (Below Baseline)")

    ineffective = [b for b in bullet_effectiveness if b["effectiveness"] < -5]

    if ineffective:
        print(f"\n{'#':<4} {'Uses':<7} {'Success':<10} {'Baseline':<10} {'Drop':<10} {'Stage':<10} {'Strategy'}")
        print("-" * 120)

        for i, item in enumerate(ineffective[:10], 1):
            bullet = item["bullet"]
            stats = item["stats"]
            success = item["success_rate"]
            baseline = item["baseline"]
            drop = item["effectiveness"]

            strategy_preview = bullet.content[:60] + "..." if len(bullet.content) > 60 else bullet.content

            print(f"{i:<4} {stats['total']:<7} {success:>6.1f}%    {baseline:>6.1f}%    {drop:>6.1f}%    {bullet.stage.value:<10} {strategy_preview}")
    else:
        print("  All strategies performed at or near baseline")

    # Stage-level analysis
    print_section("Effectiveness by Stage")

    stage_stats = defaultdict(lambda: {"total_uses": 0, "correct": 0, "strategies": 0})

    for item in bullet_effectiveness:
        stage = item["bullet"].stage.value
        stats = item["stats"]

        stage_stats[stage]["total_uses"] += stats["total"]
        stage_stats[stage]["correct"] += stats["correct"]
        stage_stats[stage]["strategies"] += 1

    print(f"\n{'Stage':<12} {'Strategies':<15} {'Total Uses':<15} {'Success Rate'}")
    print("-" * 70)

    for stage in ["prod", "staging", "shadow"]:
        stats = stage_stats[stage]
        if stats["total_uses"] == 0:
            continue

        success_rate = stats["correct"] / stats["total_uses"] * 100

        print(f"{stage.upper():<12} {stats['strategies']:<15} {stats['total_uses']:<15} {success_rate:>6.1f}%")

    # Recommendations
    print_section("Recommendations")

    if effective:
        print(f"  ✅ {len(effective)} strategies show positive impact")
        print(f"     Consider promoting top performers to higher stages")

    if ineffective:
        print(f"  ⚠️  {len(ineffective)} strategies underperforming")
        print(f"     Consider quarantining or refining these strategies")

    # Check if prod strategies are actually helping
    prod_strategies = [b for b in bullet_effectiveness if b["bullet"].stage == PlaybookStage.PROD]
    if prod_strategies:
        avg_prod_lift = sum(b["effectiveness"] for b in prod_strategies) / len(prod_strategies)
        print(f"\n  📊 Production strategies average lift: {avg_prod_lift:+.1f}%")

        if avg_prod_lift > 5:
            print(f"     ✨ PROD strategies are significantly helping!")
        elif avg_prod_lift > 0:
            print(f"     ⚡ PROD strategies provide modest improvement")
        else:
            print(f"     ⚠️  PROD strategies may need refinement")

    print("\n" + "="*70)
    print("✅ Strategy Quality Analysis Complete")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    try:
        session = setup_database()
        analyze_strategy_effectiveness(session)
        session.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
