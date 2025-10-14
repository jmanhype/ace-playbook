#!/usr/bin/env python3
"""
Arithmetic Learning with DSPy - Full ACE Cycle

Demonstrates the complete ACE framework with REAL DSPy LLM integration:
- Generator: Uses DSPy ChainOfThought with playbook bullet injection
- Reflector: Analyzes outcomes with DSPy and extracts insights
- Curator: Deduplicates and promotes insights through stages
- Learning: System improves accuracy over iterations using learned strategies

This example shows how ACE automatically discovers effective prompting strategies,
similar to what Matt Mazur is trying to do manually.

Requirements:
- Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
- Run: python examples/arithmetic_learning_dspy.py
"""

import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from ace.models import Base, Task, TaskOutput as TaskOutputDB, PlaybookStage, PlaybookBullet
from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput


def setup_database():
    """Initialize database connection."""
    db_url = os.getenv("DATABASE_URL", "sqlite:///ace_playbook.db")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def configure_dspy_lm():
    """
    Configure DSPy with LLM backend.

    Supports OpenAI and Anthropic models.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        # Use OpenAI GPT models
        lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        print("✓ Configured DSPy with OpenAI GPT-4o-mini")
        return lm, "gpt-4o-mini"
    elif anthropic_key:
        # Use Anthropic Claude models
        lm = dspy.LM("anthropic/claude-3-haiku-20240307", api_key=anthropic_key)
        print("✓ Configured DSPy with Anthropic Claude-3-Haiku")
        return lm, "claude-3-haiku"
    else:
        raise ValueError(
            "No API keys found! Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file"
        )


def create_task_db(session: Session, problem: str, ground_truth: str, domain_id: str) -> Task:
    """Create a task in the database."""
    task = Task(
        domain_id=domain_id,
        prompt=f"Solve: {problem}",
        domain="arithmetic",
        ground_truth=ground_truth,
        metadata_json={"problem": problem, "type": "arithmetic"}
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def get_playbook_bullets(session: Session, domain_id: str, stage: PlaybookStage = PlaybookStage.PROD) -> List[PlaybookBullet]:
    """Get playbook bullets for context injection."""
    return session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=stage
    ).all()


def save_task_output(session: Session, task: Task, generator_output: Dict) -> TaskOutputDB:
    """Save generator output to database."""
    output = TaskOutputDB(
        task_id=task.id,
        reasoning_trace=generator_output["reasoning_trace"],
        answer=generator_output["answer"],
        confidence=generator_output["confidence"],
        bullets_referenced=generator_output["bullets_referenced"],
        latency_ms=generator_output.get("latency_ms", 0),
        token_count=generator_output.get("prompt_tokens", 0) + generator_output.get("completion_tokens", 0)
    )
    session.add(output)
    session.commit()
    session.refresh(output)
    return output


def curate_insights(session: Session, task: Task, insights: List[Dict]):
    """
    Curator: Deduplicate and add insights to playbook.

    Uses simple content matching for deduplication.
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
                tags=["arithmetic", "dspy-learned"],
                embedding=[0.0] * 384,  # Simplified
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
    """Apply promotion gates: shadow → staging → prod"""
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


def demonstrate_dspy_learning(session: Session, generator: CoTGenerator, reflector: GroundedReflector):
    """Demonstrate full ACE learning cycle with DSPy."""

    print("\n" + "="*75)
    print("ACE Playbook - DSPy-Powered Learning Demonstration")
    print("="*75 + "\n")

    domain_id = "dspy-arithmetic"

    # Arithmetic problems with ground truth
    problems = [
        {"problem": "137 + 248", "answer": "385"},
        {"problem": "456 - 189", "answer": "267"},
        {"problem": "23 × 17", "answer": "391"},
        {"problem": "892 + 147", "answer": "1039"},
        {"problem": "500 - 278", "answer": "222"},
    ]

    for iteration, prob in enumerate(problems, 1):
        print(f"\n{'='*75}")
        print(f"ITERATION {iteration}: {prob['problem']}")
        print(f"{'='*75}\n")

        # Create task
        task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)
        print(f"📋 Task: {task_db.prompt}")
        print(f"   Ground truth: {task_db.ground_truth}")

        # Get current playbook bullets for context
        bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
        bullet_contents = [b.content for b in bullets]
        bullet_ids = [b.id for b in bullets]

        if bullets:
            print(f"\n📖 Using {len(bullets)} production strategies:")
            for i, bullet in enumerate(bullets[:3], 1):
                print(f"   {i}. {bullet.content[:60]}...")
        else:
            print("\n📖 No production strategies yet (learning from scratch)")

        # GENERATOR: Use DSPy to solve with playbook context
        print(f"\n🤖 Generator: Solving with DSPy...")
        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="arithmetic",
            playbook_bullets=bullet_contents,
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)

            print(f"   Answer: {generator_output.answer}")
            print(f"   Confidence: {generator_output.confidence:.2f}")
            print(f"   Reasoning steps: {len(generator_output.reasoning_trace)}")

            # Save to database
            output_db = save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": generator_output.bullets_referenced,
                "latency_ms": generator_output.latency_ms,
                "prompt_tokens": generator_output.prompt_tokens,
                "completion_tokens": generator_output.completion_tokens
            })

            # Check correctness
            is_correct = generator_output.answer.strip() == prob["answer"]
            print(f"   Correct: {'✅ YES' if is_correct else '❌ NO'}")

            # REFLECTOR: Analyze with DSPy
            print(f"\n🔍 Reflector: Analyzing outcome...")
            reflector_input = ReflectorInput(
                task_id=task_db.id,
                reasoning_trace=generator_output.reasoning_trace,
                answer=generator_output.answer,
                confidence=generator_output.confidence,
                bullets_referenced=generator_output.bullets_referenced,
                domain="arithmetic",
                ground_truth=prob["answer"],
                test_results="",
                error_messages=[],
                performance_metrics=""
            )

            reflector_output = reflector(reflector_input)
            print(f"   Extracted {len(reflector_output.insights)} insights")
            print(f"   Confidence: {reflector_output.confidence_score:.2f}")

            # Convert insights to dict format for curator
            insights_for_curator = [
                {
                    "content": insight.content,
                    "section": insight.section.value,
                    "confidence": insight.confidence
                }
                for insight in reflector_output.insights
            ]

            # CURATOR: Deduplicate and store
            print(f"\n📚 Curator: Storing insights...")
            result = curate_insights(session, task_db, insights_for_curator)
            print(f"   New bullets: {result['new_bullets']}")
            print(f"   Incremented: {result['incremented']}")

        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Apply promotion gates
    print(f"\n{'='*75}")
    print("PROMOTION: Applying gates")
    print(f"{'='*75}\n")

    promotions = promote_bullets(session, domain_id)
    print(f"✓ Shadow → Staging: {promotions['shadow_to_staging']} bullets")
    print(f"✓ Staging → Prod: {promotions['staging_to_prod']} bullets")

    # Show final statistics
    print(f"\n{'='*75}")
    print("FINAL STATISTICS")
    print(f"{'='*75}\n")

    for stage in [PlaybookStage.SHADOW, PlaybookStage.STAGING, PlaybookStage.PROD]:
        bullets = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id,
            stage=stage
        ).all()
        print(f"  {stage.value.upper():10} stage: {len(bullets)} bullets")

        for bullet in bullets[:5]:  # Show top 5
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            print(f"    • {bullet.content[:55]}...")
            print(f"      H:{bullet.helpful_count} / Harm:{bullet.harmful_count} / Ratio:{ratio:.1f}")

    print(f"\n{'='*75}")
    print("✅ DSPy Learning Cycle Complete!")
    print(f"{'='*75}\n")


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - DSPy-Powered Learning\n")

    try:
        # Configure DSPy
        lm, model_name = configure_dspy_lm()
        dspy.configure(lm=lm)

        # Initialize database
        session = setup_database()

        # Initialize ACE components with DSPy
        generator = CoTGenerator(model=model_name, temperature=0.7)
        reflector = GroundedReflector(model=model_name, temperature=0.3)

        print(f"✓ Initialized Generator with {model_name}")
        print(f"✓ Initialized Reflector with {model_name}")

        # Run demonstration
        demonstrate_dspy_learning(session, generator, reflector)

        session.close()

        print("\n💡 What happened:")
        print("  1. Generator used DSPy with playbook strategies as context")
        print("  2. LLM generated solutions with explicit reasoning traces")
        print("  3. Reflector analyzed outcomes and extracted insights")
        print("  4. Curator deduplicated and promoted successful strategies")
        print("  5. Future tasks automatically benefit from learned patterns!")
        print("\n  This is exactly what Matt Mazur is trying to achieve - automated")
        print("  discovery of effective prompting strategies through execution feedback.\n")

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nTo run this example:")
        print("  1. Set OPENAI_API_KEY in .env file, OR")
        print("  2. Set ANTHROPIC_API_KEY in .env file")
        print("\nExample .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
