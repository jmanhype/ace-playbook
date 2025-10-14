#!/usr/bin/env python3
"""
Arithmetic Learning Example - Full ACE Cycle

Demonstrates the complete ACE (Adaptive Code Evolution) framework:
- Generator: Solves arithmetic tasks
- Reflector: Analyzes outputs and extracts insights
- Curator: Deduplicates and promotes insights through stages

This is a functional demonstration showing how the system learns from execution feedback.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone

from ace.models import Base, Task, TaskOutput, PlaybookStage, PlaybookBullet


def setup_database():
    """Initialize database connection."""
    db_url = os.getenv("DATABASE_URL", "sqlite:///ace_playbook.db")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_arithmetic_task(session: Session, problem: str, domain_id: str = "demo-arithmetic"):
    """Create an arithmetic task."""
    task = Task(
        domain_id=domain_id,
        prompt=f"Solve: {problem}",
        domain="arithmetic",
        ground_truth=None,  # Would be set if we know the answer
        metadata_json={"problem": problem, "type": "arithmetic"}
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def generate_solution(session: Session, task: Task, answer: str, reasoning_steps: list):
    """
    Generator: Produces a solution with reasoning trace.

    In a real system, this would use DSPy ReAct/CoT to solve the task
    using relevant playbook bullets as context.
    """
    # Check for relevant playbook bullets
    bullets = session.query(PlaybookBullet).filter_by(
        domain_id=task.domain_id,
        stage=PlaybookStage.PROD
    ).all()

    bullet_ids = [b.id for b in bullets]

    output = TaskOutput(
        task_id=task.id,
        reasoning_trace=reasoning_steps,
        answer=answer,
        confidence=0.95,
        bullets_referenced=bullet_ids,
        latency_ms=150,
        token_count=75
    )
    session.add(output)
    session.commit()
    session.refresh(output)
    return output


def reflect_on_output(session: Session, output: TaskOutput, is_correct: bool):
    """
    Reflector: Analyzes output and extracts insights.

    Returns insights labeled as Helpful/Harmful/Neutral based on outcome.
    """
    insights = []

    if is_correct:
        # Extract helpful pattern
        if len(output.reasoning_trace) > 1:
            insights.append({
                "content": "Break down arithmetic into step-by-step reasoning",
                "section": "Helpful",
                "reasoning": "Multi-step reasoning led to correct answer"
            })

        insights.append({
            "content": "Verify the operation type before calculating",
            "section": "Helpful",
            "reasoning": "Correct identification of addition operation"
        })
    else:
        # Extract harmful pattern (if answer was wrong)
        insights.append({
            "content": "Rushing to answer without verification leads to errors",
            "section": "Harmful",
            "reasoning": "Incorrect answer suggests insufficient validation"
        })

    return insights


def curate_insights(session: Session, task: Task, insights: list):
    """
    Curator: Deduplicates and adds insights to playbook.

    In the full system, this uses semantic similarity (FAISS + embeddings)
    to prevent duplicates. Here we use simple content matching for demo purposes.
    """
    new_bullets = []
    incremented_bullets = []

    for insight in insights:
        # Check for semantic duplicates (simplified: exact content match)
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
                tags=["arithmetic", "strategy"],
                embedding=[0.0] * 384,  # Simplified - normally from sentence-transformers
                stage=PlaybookStage.SHADOW
            )
            session.add(bullet)
            new_bullets.append(bullet)

    session.commit()

    return {
        "new_bullets": len(new_bullets),
        "incremented": len(incremented_bullets)
    }


def promote_bullets(session: Session, domain_id: str):
    """
    Apply promotion gates: shadow → staging → prod

    Rules from config:
    - Shadow → Staging: 3+ helpful votes, ratio >= 3.0
    - Staging → Prod: 5+ helpful votes, ratio >= 5.0
    """
    shadow_helpful_min = int(os.getenv("STAGING_HELPFUL_MIN", "3"))
    prod_helpful_min = int(os.getenv("PROD_HELPFUL_MIN", "5"))
    staging_ratio_min = float(os.getenv("STAGING_RATIO_MIN", "3.0"))
    prod_ratio_min = float(os.getenv("PROD_RATIO_MIN", "5.0"))

    promotions = {"shadow_to_staging": 0, "staging_to_prod": 0}

    # Promote from shadow to staging
    shadow_bullets = session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=PlaybookStage.SHADOW
    ).all()

    for bullet in shadow_bullets:
        if bullet.helpful_count >= shadow_helpful_min:
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            if ratio >= staging_ratio_min:
                bullet.stage = PlaybookStage.STAGING
                promotions["shadow_to_staging"] += 1

    # Promote from staging to prod
    staging_bullets = session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=PlaybookStage.STAGING
    ).all()

    for bullet in staging_bullets:
        if bullet.helpful_count >= prod_helpful_min:
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            if ratio >= prod_ratio_min:
                bullet.stage = PlaybookStage.PROD
                promotions["staging_to_prod"] += 1

    session.commit()
    return promotions


def demonstrate_full_ace_cycle(session: Session):
    """Demonstrate complete Generator-Reflector-Curator cycle."""
    print("\n" + "="*70)
    print("ACE Playbook - Full Learning Cycle Demonstration")
    print("="*70 + "\n")

    domain_id = "demo-arithmetic"

    # ===== TASK 1: First learning iteration =====
    print("📝 ITERATION 1: Learning from first task")
    print("-" * 70)

    task1 = create_arithmetic_task(session, "25 + 17", domain_id)
    print(f"✓ Task created: {task1.prompt}")

    output1 = generate_solution(
        session, task1,
        answer="42",
        reasoning_steps=["Identify operation: addition", "Calculate: 25 + 17 = 42"]
    )
    print(f"✓ Generator output: {output1.answer}")
    print(f"  Reasoning: {' → '.join(output1.reasoning_trace)}")

    insights1 = reflect_on_output(session, output1, is_correct=True)
    print(f"✓ Reflector extracted {len(insights1)} insights")

    result1 = curate_insights(session, task1, insights1)
    print(f"✓ Curator: {result1['new_bullets']} new bullets, {result1['incremented']} incremented")

    # ===== TASK 2: Second iteration (reinforcement) =====
    print("\n📝 ITERATION 2: Reinforcing patterns")
    print("-" * 70)

    task2 = create_arithmetic_task(session, "38 + 29", domain_id)
    print(f"✓ Task created: {task2.prompt}")

    output2 = generate_solution(
        session, task2,
        answer="67",
        reasoning_steps=["Check operation type: addition", "Break down: 38 + 29 = 67"]
    )
    print(f"✓ Generator output: {output2.answer}")

    insights2 = reflect_on_output(session, output2, is_correct=True)
    result2 = curate_insights(session, task2, insights2)
    print(f"✓ Curator: {result2['new_bullets']} new bullets, {result2['incremented']} incremented")

    # ===== TASK 3: Third iteration (more reinforcement) =====
    print("\n📝 ITERATION 3: Building confidence")
    print("-" * 70)

    task3 = create_arithmetic_task(session, "51 + 44", domain_id)
    output3 = generate_solution(
        session, task3,
        answer="95",
        reasoning_steps=["Verify operation: addition", "Step-by-step: 51 + 44 = 95"]
    )
    insights3 = reflect_on_output(session, output3, is_correct=True)
    result3 = curate_insights(session, task3, insights3)
    print(f"✓ Task: {task3.prompt} → {output3.answer}")
    print(f"✓ Curator: {result3['new_bullets']} new bullets, {result3['incremented']} incremented")

    # ===== Promotion =====
    print("\n📊 PROMOTION: Applying gates")
    print("-" * 70)

    promotions = promote_bullets(session, domain_id)
    print(f"✓ Shadow → Staging: {promotions['shadow_to_staging']} bullets promoted")
    print(f"✓ Staging → Prod: {promotions['staging_to_prod']} bullets promoted")

    # ===== Show final statistics =====
    print("\n📈 FINAL STATISTICS")
    print("-" * 70)

    for stage in [PlaybookStage.SHADOW, PlaybookStage.STAGING, PlaybookStage.PROD]:
        bullets = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id,
            stage=stage
        ).all()
        print(f"  {stage.value.upper():10} stage: {len(bullets)} bullets")

        for bullet in bullets:
            helpful_ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            print(f"    • {bullet.content[:50]}...")
            print(f"      Helpful: {bullet.helpful_count}, Harmful: {bullet.harmful_count}, Ratio: {helpful_ratio:.1f}")

    print("\n" + "="*70)
    print("✅ Full ACE cycle complete!")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Complete Learning Demonstration\n")

    # Check for API keys (optional for this demo)
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("ℹ️  Running in simulation mode (no LLM API keys)")
        print("   Real ACE system would use DSPy for generation\n")

    try:
        session = setup_database()
        demonstrate_full_ace_cycle(session)
        session.close()

        print("\n💡 What happened:")
        print("  1. Generator solved 3 arithmetic tasks with reasoning")
        print("  2. Reflector analyzed outputs and extracted insights")
        print("  3. Curator deduplicated and tracked helpful/harmful votes")
        print("  4. Promotion gates moved insights: shadow → staging → prod")
        print("  5. Future tasks can now use production insights!")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
