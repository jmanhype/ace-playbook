#!/usr/bin/env python3
"""
Simple ACE Demo - No Model Loading

Quick demonstration of ACE framework core concepts without heavy dependencies.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ace.models import (
    Base, Task, TaskOutput,
    PlaybookBullet, PlaybookStage
)


def main():
    """Demonstrate ACE framework basics."""
    print("\n" + "="*60)
    print("ACE Playbook - Simple Demo")
    print("="*60 + "\n")

    # Setup database
    print("📋 Setting up database...")
    db_url = os.getenv("DATABASE_URL", "sqlite:///ace_playbook.db")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Create a simple task
    print("\n1️⃣  Creating a task...")
    task = Task(
        domain_id="demo-user",
        prompt="Calculate 10 + 5",
        domain="arithmetic",
        ground_truth="15"
    )
    session.add(task)
    session.commit()
    print(f"   ✓ Task {task.id}: {task.prompt}")

    # Create task output
    print("\n2️⃣  Generating output...")
    output = TaskOutput(
        task_id=task.id,
        reasoning_trace=["Add 10 and 5", "10 + 5 = 15"],
        answer="15",
        confidence=0.95,
        bullets_referenced=[],
        latency_ms=100,
        token_count=50
    )
    session.add(output)
    session.commit()
    print(f"   ✓ Output: {output.answer}")

    # Create playbook bullet directly (simplified demo)
    print("\n3️⃣  Adding insight to playbook (shadow stage)...")
    bullet = PlaybookBullet(
        content="For arithmetic problems, break down step by step",
        domain_id="demo-user",
        section="Helpful",
        helpful_count=1,
        harmful_count=0,
        tags=["arithmetic", "strategy"],
        embedding=[0.0] * 384,  # Simplified - normally from sentence-transformers
        stage=PlaybookStage.SHADOW
    )
    session.add(bullet)
    session.commit()
    print(f"   ✓ Bullet {bullet.id} in {bullet.stage.value} stage")
    print(f"   ✓ Stats: {bullet.helpful_count} helpful, {bullet.harmful_count} harmful")

    # Show promotion path
    print("\n4️⃣  Promotion path (from config)...")
    print(f"   Shadow → Staging: {os.getenv('STAGING_HELPFUL_MIN', '3')} helpful votes needed")
    print(f"   Staging → Prod: {os.getenv('PROD_HELPFUL_MIN', '5')} helpful votes needed")
    print(f"   Current: {bullet.helpful_count} helpful votes (in shadow)")

    # Show statistics
    print("\n📊 Current Statistics:")
    print("-" * 40)

    for stage in [PlaybookStage.SHADOW, PlaybookStage.STAGING, PlaybookStage.PROD]:
        count = session.query(PlaybookBullet).filter_by(
            stage=stage,
            domain_id="demo-user"
        ).count()
        print(f"   {stage.value}: {count} bullets")

    print("-" * 40)

    session.close()

    print("\n✅ Demo complete!")
    print("\n💡 Key Concepts Demonstrated:")
    print("   • Task: Problem to solve (with domain_id, prompt, domain)")
    print("   • TaskOutput: Solution with reasoning trace")
    print("   • PlaybookBullet: Learned insights in the playbook")
    print("   • Stage promotion: shadow → staging → prod")
    print("   • Multi-domain: Isolated namespaces per tenant")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
