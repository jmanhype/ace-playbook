#!/usr/bin/env python3
"""
Longer Training - Evaluation 5

Runs extended training with 10 epochs to test:
- Does accuracy continue improving beyond 3 epochs?
- When do strategies plateau?
- How many production strategies emerge?

Note: This takes significantly longer (~30+ minutes)
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
from ace.reflector import GroundedReflector, ReflectorInput
from ace.models import PlaybookStage, PlaybookBullet
from datetime import datetime, timezone


def curate_insights(session, task, insights):
    """
    Curator: Deduplicate and add insights to playbook.
    (Simplified version from main script)
    """
    new_bullets = []
    incremented_bullets = []

    for insight in insights:
        # Check for duplicates
        existing = session.query(PlaybookBullet).filter_by(
            domain_id=task.domain_id,
            content=insight["content"]
        ).first()

        if existing:
            # Increment counter
            if insight["section"] == "Helpful":
                existing.helpful_count += 1
            elif insight["section"] == "Harmful":
                existing.harmful_count += 1

            existing.last_used_at = datetime.now(timezone.utc)
            incremented_bullets.append(existing)
        else:
            # Add new bullet in shadow stage
            bullet = PlaybookBullet(
                content=insight["content"],
                domain_id=task.domain_id,
                section=insight["section"],
                helpful_count=1 if insight["section"] == "Helpful" else 0,
                harmful_count=1 if insight["section"] == "Harmful" else 0,
                tags=["multiplication", "large-numbers", "extended-training"],
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
    shadow_helpful_min = int(os.getenv("STAGING_HELPFUL_MIN", "3"))
    prod_helpful_min = int(os.getenv("PROD_HELPFUL_MIN", "5"))
    staging_ratio_min = float(os.getenv("STAGING_RATIO_MIN", "3.0"))
    prod_ratio_min = float(os.getenv("PROD_RATIO_MIN", "5.0"))

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


def run_longer_training(
    session,
    generator: CoTGenerator,
    reflector: GroundedReflector,
    num_problems: int = 20,
    num_epochs: int = 10
):
    """
    Run extended training experiment.

    Args:
        session: Database session
        generator: CoT generator
        reflector: Grounded reflector
        num_problems: Number of problems per epoch
        num_epochs: Number of epochs (default: 10)
    """
    print_header(f"Evaluation 5: Extended Training ({num_epochs} Epochs)")

    domain_id = "multiplication-extended"

    # Generate problems
    print(f"📊 Generating {num_problems} multiplication problems...")
    problems = generate_multiplication_problems(num_problems, seed=100)
    print(f"✓ Generated {num_problems} problems\n")

    print(f"⏱️  This will take ~{num_epochs * 3} minutes (3 min/epoch)\n")

    # Track metrics across epochs
    epoch_metrics = []
    prod_strategy_counts = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*70}\n")

        correct_count = 0
        total_count = 0

        for idx, prob in enumerate(problems, 1):
            # Create task
            task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)

            # Get current playbook bullets
            bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
            bullet_contents = [b.content for b in bullets]

            # Generator
            task_input = TaskInput(
                task_id=task_db.id,
                description=task_db.prompt,
                domain="multiplication",
                playbook_bullets=bullet_contents,
                max_reasoning_steps=10
            )

            try:
                generator_output = generator(task_input)

                is_correct = generator_output.answer.strip() == prob["answer"]
                if is_correct:
                    correct_count += 1
                total_count += 1

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

                # Reflector
                reflector_input = ReflectorInput(
                    task_id=task_db.id,
                    reasoning_trace=generator_output.reasoning_trace,
                    answer=generator_output.answer,
                    confidence=generator_output.confidence,
                    bullets_referenced=generator_output.bullets_referenced,
                    domain="multiplication",
                    ground_truth=prob["answer"],
                    test_results="",
                    error_messages=[],
                    performance_metrics=""
                )

                reflector_output = reflector(reflector_input)

                # Curator
                insights_for_curator = [
                    {
                        "content": insight.content,
                        "section": insight.section.value,
                        "confidence": insight.confidence
                    }
                    for insight in reflector_output.insights
                ]

                curate_insights(session, task_db, insights_for_curator)

            except Exception as e:
                print(f"❌ Error on problem {idx}: {e}")
                continue

        # Epoch statistics
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        epoch_metrics.append({
            "epoch": epoch,
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy
        })

        # Promote bullets
        promotions = promote_bullets(session, domain_id)

        # Count prod strategies
        prod_count = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id,
            stage=PlaybookStage.PROD
        ).count()
        prod_strategy_counts.append(prod_count)

        print(f"\n{'─'*70}")
        print(f"EPOCH {epoch} RESULTS:")
        print(f"  Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")
        print(f"  PROD strategies: {prod_count}")
        print(f"  Promotions: Shadow→Staging: {promotions['shadow_to_staging']}, Staging→Prod: {promotions['staging_to_prod']}")
        print(f"{'─'*70}")

    # Final analysis
    print_section("Extended Training Results")

    print("📊 Accuracy Progression:\n")
    print(f"{'Epoch':<8} {'Correct':<10} {'Total':<8} {'Accuracy':<12} {'PROD Strategies'}")
    print("-" * 70)

    for i, metric in enumerate(epoch_metrics):
        prod_count = prod_strategy_counts[i]
        print(f"{metric['epoch']:<8} {metric['correct']:<10} {metric['total']:<8} {metric['accuracy']:>6.1f}%      {prod_count}")

    # Analyze trends
    print_section("Learning Curve Analysis")

    first_third = epoch_metrics[:3]
    middle_third = epoch_metrics[3:7]
    last_third = epoch_metrics[7:]

    first_avg = sum(m["accuracy"] for m in first_third) / len(first_third)
    middle_avg = sum(m["accuracy"] for m in middle_third) / len(middle_third)
    last_avg = sum(m["accuracy"] for m in last_third) / len(last_third)

    print(f"  Epochs 1-3 average:   {first_avg:.1f}%")
    print(f"  Epochs 4-7 average:   {middle_avg:.1f}%")
    print(f"  Epochs 8-10 average:  {last_avg:.1f}%")

    if last_avg > middle_avg + 5:
        print(f"\n  ✨ Still improving! Consider more epochs")
    elif last_avg > first_avg + 10:
        print(f"\n  ✅ Significant learning occurred")
        print(f"     Total improvement: {last_avg - first_avg:+.1f}%")
    else:
        print(f"\n  ⚠️  Learning may have plateaued")

    print("\n" + "="*70)
    print("✅ Extended Training Complete")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Extended Training (10 Epochs)\n")

    try:
        # Configure DSPy
        lm, model_name = configure_dspy_lm()
        dspy.configure(lm=lm)

        # Initialize database
        session = setup_database()

        # Initialize ACE components
        generator = CoTGenerator(model=model_name, temperature=0.7)
        reflector = GroundedReflector(model=model_name, temperature=0.3)

        print(f"✓ Initialized Generator with {model_name}")
        print(f"✓ Initialized Reflector with {model_name}\n")

        # Run extended training
        run_longer_training(
            session=session,
            generator=generator,
            reflector=reflector,
            num_problems=20,
            num_epochs=10
        )

        session.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
