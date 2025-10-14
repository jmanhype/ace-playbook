#!/usr/bin/env python3
"""
Arithmetic Learning Example

Demonstrates the ACE (Adaptive Code Evolution) framework using a simple
arithmetic task that learns from feedback.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ace.models import Base, Task, TaskOutput, PlaybookStage
from ace.repositories import PlaybookRepository, TaskRepository
from ace.utils.embeddings import EmbeddingGenerator


def setup_database():
    """Initialize database connection."""
    db_url = os.getenv("DATABASE_URL", "sqlite:///ace_playbook.db")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def create_arithmetic_task(session, problem: str, domain: str = "arithmetic"):
    """Create a simple arithmetic task."""
    task_repo = TaskRepository(session)

    task = Task(
        task_type="arithmetic",
        description=f"Solve: {problem}",
        domain=domain,
        metadata_={"problem": problem}
    )

    return task_repo.create(task)


def simulate_generator_output(task: Task, answer: int, reasoning: str) -> TaskOutput:
    """Simulate generator producing an output."""
    return TaskOutput(
        task_id=task.id,
        output_text=str(answer),
        metadata_={"reasoning": reasoning}
    )


def demonstrate_ace_cycle(session):
    """Demonstrate one complete ACE learning cycle."""
    print("\n" + "="*60)
    print("ACE Playbook - Arithmetic Learning Demo")
    print("="*60 + "\n")

    # Step 1: Create a task
    print("📋 Step 1: Creating arithmetic task...")
    task = create_arithmetic_task(session, "15 + 27", domain="demo-arithmetic")
    print(f"   Task created: {task.description}")

    # Step 2: Simulate Generator output
    print("\n🤖 Step 2: Generator solving task...")
    output = simulate_generator_output(
        task,
        answer=42,
        reasoning="Adding 15 and 27: 15 + 27 = 42"
    )
    print(f"   Generated answer: {output.output_text}")
    print(f"   Reasoning: {output.metadata_['reasoning']}")

    # Step 3: Check playbook for existing insights
    print("\n📖 Step 3: Checking playbook for insights...")
    playbook_repo = PlaybookRepository(session)

    bullets = playbook_repo.get_bullets_by_stage_and_domain(
        PlaybookStage.PROD,
        domain="demo-arithmetic",
        limit=5
    )

    if bullets:
        print(f"   Found {len(bullets)} production insights:")
        for bullet in bullets:
            print(f"   - {bullet.content} (used {bullet.access_count} times)")
    else:
        print("   No production insights yet (first run)")

    # Step 4: Show the learning cycle
    print("\n🔄 Step 4: Learning cycle overview...")
    print("   In a real scenario:")
    print("   1. Reflector would analyze the output")
    print("   2. Curator would deduplicate insights")
    print("   3. Insights move through: shadow → staging → prod")
    print("   4. Future tasks benefit from learned patterns")

    # Step 5: Check promotion criteria
    print("\n📊 Step 5: Promotion gates (from config)...")
    print(f"   Shadow → Staging: {os.getenv('SHADOW_HELPFUL_MIN', '0')} helpful votes")
    print(f"   Staging → Prod: {os.getenv('STAGING_HELPFUL_MIN', '3')} helpful votes")
    print(f"   Prod requirement: {os.getenv('PROD_HELPFUL_MIN', '5')} helpful votes")

    session.commit()

    print("\n✅ Demo complete!")
    print("="*60 + "\n")


def show_statistics(session):
    """Show playbook statistics."""
    playbook_repo = PlaybookRepository(session)

    print("\n📊 Playbook Statistics:")
    print("-" * 40)

    for stage in [PlaybookStage.SHADOW, PlaybookStage.STAGING, PlaybookStage.PROD]:
        bullets = playbook_repo.get_bullets_by_stage_and_domain(
            stage,
            domain="demo-arithmetic"
        )
        print(f"   {stage.value}: {len(bullets)} bullets")

    print("-" * 40 + "\n")


def main():
    """Main entry point."""
    print("\n🚀 Starting ACE Arithmetic Learning Example...")

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: No API keys found in .env file")
        print("   This demo will run in simulation mode only")
        print("   To use real LLMs, set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env\n")

    try:
        # Initialize database session
        session = setup_database()

        # Run demonstration
        demonstrate_ace_cycle(session)

        # Show statistics
        show_statistics(session)

        session.close()

    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        print(f"   Make sure the database is initialized with: alembic upgrade head\n")
        raise


if __name__ == "__main__":
    main()
