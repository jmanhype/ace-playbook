#!/usr/bin/env python3
"""
Large Integer Multiplication - Matt Mazur Challenge with ACE

This example demonstrates ACE on the exact challenge Matt Mazur posed:
getting LLMs to accurately multiply two integers in the range 1-10,000
through context learning (not tool calling).

The goal is to discover prompting strategies that improve accuracy on
large integer multiplication through iterative learning from execution feedback.

Requirements:
- Set OPENROUTER_API_KEY in .env (or OPENAI_API_KEY/ANTHROPIC_API_KEY)
- Run: python examples/arithmetic_learning_multiplication.py

Expected Behavior:
- Initial accuracy: ~10-20% (LLMs struggle with large multiplication)
- After 5 epochs: Discovering strategies like "break into chunks",
  "verify with digit checks", "show intermediate steps"
- Goal: Demonstrate ACE can discover sophisticated prompting strategies
"""

import os
import sys
import random
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

    Supports OpenRouter, OpenAI, and Anthropic models.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openrouter_key:
        # Use OpenRouter with Qwen model (fast and capable)
        lm = dspy.LM(
            "openrouter/qwen/qwen-2.5-7b-instruct",
            api_key=openrouter_key,
            api_base="https://openrouter.ai/api/v1"
        )
        print("✓ Configured DSPy with OpenRouter (Qwen 2.5 7B Instruct)")
        return lm, "qwen-2.5-7b-instruct"
    elif openai_key and not openai_key.startswith("your_"):
        # Use OpenAI GPT models
        lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        print("✓ Configured DSPy with OpenAI GPT-4o-mini")
        return lm, "gpt-4o-mini"
    elif anthropic_key and not anthropic_key.startswith("your_"):
        # Use Anthropic Claude models
        lm = dspy.LM("anthropic/claude-3-haiku-20240307", api_key=anthropic_key)
        print("✓ Configured DSPy with Anthropic Claude-3-Haiku")
        return lm, "claude-3-haiku"
    else:
        raise ValueError(
            "No API keys found! Please set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env file"
        )


def generate_multiplication_problems(num_problems: int, min_val: int = 1, max_val: int = 10000, seed: int = 42) -> List[Dict]:
    """
    Generate random multiplication problems in the specified range.

    Args:
        num_problems: Number of problems to generate
        min_val: Minimum value for each operand
        max_val: Maximum value for each operand
        seed: Random seed for reproducibility

    Returns:
        List of problem dicts with 'problem' and 'answer' keys
    """
    random.seed(seed)
    problems = []

    for _ in range(num_problems):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        answer = a * b
        problems.append({
            "problem": f"{a} × {b}",
            "answer": str(answer)
        })

    return problems


def create_task_db(session: Session, problem: str, ground_truth: str, domain_id: str) -> Task:
    """Create a task in the database."""
    task = Task(
        domain_id=domain_id,
        prompt=f"Calculate: {problem}",
        domain="multiplication",
        ground_truth=ground_truth,
        metadata_json={"problem": problem, "type": "large_multiplication"}
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
        latency_ms=generator_output.get("latency_ms", 0) or 0,
        token_count=(generator_output.get("prompt_tokens", 0) or 0) + (generator_output.get("completion_tokens", 0) or 0)
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
                tags=["multiplication", "large-numbers", "ace-learned"],
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


def run_multiplication_experiment(
    session: Session,
    generator: CoTGenerator,
    reflector: GroundedReflector,
    num_problems: int = 50,
    num_epochs: int = 5
):
    """
    Run the multiplication learning experiment with multiple epochs.

    Args:
        session: Database session
        generator: CoT generator
        reflector: Grounded reflector
        num_problems: Number of multiplication problems
        num_epochs: Number of training epochs
    """
    print("\n" + "="*80)
    print("ACE Framework - Large Integer Multiplication Challenge")
    print("Matt Mazur's Challenge: Multiply integers in range 1-10,000")
    print("="*80 + "\n")

    domain_id = "multiplication-large"

    # Generate problems
    print(f"📊 Generating {num_problems} random multiplication problems...")
    problems = generate_multiplication_problems(num_problems)
    print(f"✓ Generated {num_problems} problems (range: 1-10,000)\n")

    # Track metrics across epochs
    epoch_metrics = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*80}\n")

        correct_count = 0
        total_count = 0

        for idx, prob in enumerate(problems, 1):
            # Create task
            task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)

            # Get current playbook bullets for context
            bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
            bullet_contents = [b.content for b in bullets]

            # Show progress every 10 problems
            if idx % 10 == 0 or idx == 1:
                print(f"\n[{idx}/{num_problems}] Problem: {prob['problem']}")
                if bullets:
                    print(f"  Using {len(bullets)} production strategies")
                else:
                    print(f"  No production strategies yet")

            # GENERATOR: Use DSPy to solve with playbook context
            task_input = TaskInput(
                task_id=task_db.id,
                description=task_db.prompt,
                domain="multiplication",
                playbook_bullets=bullet_contents,
                max_reasoning_steps=10
            )

            try:
                generator_output = generator(task_input)

                # Check correctness
                is_correct = generator_output.answer.strip() == prob["answer"]
                if is_correct:
                    correct_count += 1
                total_count += 1

                if idx % 10 == 0 or idx == 1:
                    print(f"  Answer: {generator_output.answer} (Expected: {prob['answer']})")
                    print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")

                # Save to database
                output_db = save_task_output(session, task_db, {
                    "reasoning_trace": generator_output.reasoning_trace,
                    "answer": generator_output.answer,
                    "confidence": generator_output.confidence,
                    "bullets_referenced": generator_output.bullets_referenced,
                    "latency_ms": generator_output.latency_ms or 0,
                    "prompt_tokens": generator_output.prompt_tokens or 0,
                    "completion_tokens": generator_output.completion_tokens or 0
                })

                # REFLECTOR: Analyze with DSPy
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
                curate_insights(session, task_db, insights_for_curator)

            except Exception as e:
                print(f"\n❌ Error on problem {idx}: {e}")
                continue

        # Epoch statistics
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        epoch_metrics.append({
            "epoch": epoch,
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy
        })

        print(f"\n{'─'*80}")
        print(f"EPOCH {epoch} RESULTS:")
        print(f"  Accuracy: {correct_count}/{total_count} = {accuracy:.1f}%")
        print(f"{'─'*80}")

        # Apply promotion gates after each epoch
        promotions = promote_bullets(session, domain_id)
        print(f"\n📈 Promotions:")
        print(f"  Shadow → Staging: {promotions['shadow_to_staging']}")
        print(f"  Staging → Prod: {promotions['staging_to_prod']}")

    # Final statistics
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")

    print("📊 Accuracy Progression:")
    for metric in epoch_metrics:
        print(f"  Epoch {metric['epoch']}: {metric['accuracy']:.1f}% ({metric['correct']}/{metric['total']})")

    print(f"\n📚 Playbook Statistics:")
    for stage in [PlaybookStage.SHADOW, PlaybookStage.STAGING, PlaybookStage.PROD]:
        bullets = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id,
            stage=stage
        ).all()
        print(f"\n  {stage.value.upper()} stage: {len(bullets)} bullets")

        # Show top strategies by helpful count
        top_bullets = sorted(bullets, key=lambda b: b.helpful_count, reverse=True)[:5]
        for bullet in top_bullets:
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            print(f"    • {bullet.content[:70]}...")
            print(f"      Helpful:{bullet.helpful_count} Harmful:{bullet.harmful_count} Ratio:{ratio:.1f}")

    print(f"\n{'='*80}")
    print("✅ Multiplication Learning Experiment Complete!")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    print("\n🚀 ACE Framework - Large Multiplication Challenge\n")

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

        # Run experiment
        num_problems = int(os.getenv("NUM_PROBLEMS", "50"))
        num_epochs = int(os.getenv("NUM_EPOCHS", "5"))

        print(f"\n📋 Configuration:")
        print(f"  Problems: {num_problems}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Model: {model_name}")

        run_multiplication_experiment(
            session=session,
            generator=generator,
            reflector=reflector,
            num_problems=num_problems,
            num_epochs=num_epochs
        )

        session.close()

        print("\n💡 What This Demonstrates:")
        print("  1. ACE learns from execution feedback on large multiplication")
        print("  2. System discovers prompting strategies that improve accuracy")
        print("  3. Multi-epoch training allows progressive refinement")
        print("  4. Playbook evolves from shadow → staging → prod based on effectiveness")
        print("  5. This is Matt Mazur's challenge - improving via context learning!\n")

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nTo run this example:")
        print("  1. Set OPENAI_API_KEY in .env file, OR")
        print("  2. Set ANTHROPIC_API_KEY in .env file, OR")
        print("  3. Set OPENROUTER_API_KEY in .env file")
        print("\nExample .env file:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        print("  NUM_PROBLEMS=50")
        print("  NUM_EPOCHS=5\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
