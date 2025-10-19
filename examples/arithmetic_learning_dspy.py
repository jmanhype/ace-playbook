#!/usr/bin/env python3
"""
Arithmetic Learning with DSPy - Full ACE Cycle

Demonstrates the complete ACE framework with REAL DSPy LLM integration:
- Generator: Uses DSPy ChainOfThought with playbook bullet injection
- Reflector: Analyzes outcomes with DSPy and extracts insights
- Curator: Deduplicates and promotes insights through stages
- Learning: System improves accuracy over iterations using learned strategies

This example shows how ACE automatically discovers effective prompting strategies
across multiple epochs on randomized arithmetic tasks, mirroring the reporting
style from the multiplication learning suite while staying in the arithmetic
domain.

Requirements:
- Set OPENROUTER_API_KEY in .env (preferred) or OPENAI/ANTHROPIC fallbacks
- Optional: configure NUM_PROBLEMS and NUM_EPOCHS in .env
- Run: python examples/arithmetic_learning_dspy.py
"""

import os
import sys
import random
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
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
from ace.curator import CuratorService
from ace.ops.stage_manager import StageManager


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
        default_model = "openrouter/qwen/qwen-2.5-7b-instruct"
        openrouter_model = os.getenv("OPENROUTER_MODEL", default_model)
        lm = dspy.LM(
            openrouter_model,
            api_key=openrouter_key,
            api_base="https://openrouter.ai/api/v1"
        )
        model_label = openrouter_model.split("/")[-1]
        print(f"✓ Configured DSPy with OpenRouter ({model_label})")
        return lm, model_label
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


def generate_arithmetic_problems(
    num_problems: int,
    min_val: int = 10,
    max_val: int = 999,
    seed: int = 42
) -> List[Dict]:
    """Generate a list of random arithmetic problems."""

    random.seed(seed)
    problems: List[Dict] = []
    operations = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b if a >= b else b - a),
        ("×", lambda a, b: a * b),
    ]

    for _ in range(num_problems):
        op_symbol, op_fn = random.choice(operations)
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)

        if op_symbol == "-" and b > a:
            a, b = b, a

        result = op_fn(a, b)
        problems.append(
            {
                "problem": f"{a} {op_symbol} {b}",
                "answer": str(result),
                "operation": op_symbol,
            }
        )

    return problems


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

def save_task_output(session: Session, task: Task, generator_output: Dict) -> TaskOutputDB:
    """Save generator output to database."""
    output = TaskOutputDB(
        task_id=task.id,
        reasoning_trace=generator_output["reasoning_trace"],
        answer=generator_output["answer"],
        confidence=generator_output["confidence"],
        bullets_referenced=generator_output["bullets_referenced"],
        latency_ms=generator_output.get("latency_ms") or 0,
        token_count=(generator_output.get("prompt_tokens") or 0) + (generator_output.get("completion_tokens") or 0)
    )
    session.add(output)
    session.commit()
    session.refresh(output)
    return output


def get_active_playbook_context(stage_manager: StageManager, domain_id: str, max_bullets: int = 40) -> List[str]:
    """Retrieve active playbook strategies for context injection."""
    bullets = stage_manager.playbook_repo.get_active_playbook(
        domain_id=domain_id,
        exclude_quarantined=True
    )

    eligible = [
        bullet for bullet in bullets
        if bullet.stage in (PlaybookStage.PROD, PlaybookStage.STAGING)
    ]

    prioritized = sorted(
        eligible,
        key=lambda bullet: (
            0 if bullet.stage == PlaybookStage.PROD else 1,
            -bullet.helpful_count,
            bullet.created_at
        )
    )

    return [bullet.content for bullet in prioritized[:max_bullets]]


def run_arithmetic_experiment(
    session: Session,
    generator: CoTGenerator,
    reflector: GroundedReflector,
    stage_manager: StageManager,
    curator_service: CuratorService,
    num_problems: int = 20,
    num_epochs: int = 3
):
    """Run multi-epoch arithmetic learning with reporting akin to multiplication suite."""

    print("\n" + "=" * 80)
    print("ACE Framework - Arithmetic Learning Challenge")
    print("Discovering prompting strategies for random arithmetic problems")
    print("=" * 80 + "\n")

    domain_id = "arithmetic-learning"

    print(f"📊 Generating {num_problems} arithmetic problems...")
    problems = generate_arithmetic_problems(num_problems)
    print("✓ Generated problem set\n")

    epoch_metrics: List[Dict] = []
    total_insights = 0
    helpful_insights = 0
    harmful_insights = 0
    curator_totals = {
        "new_bullets": 0,
        "increments": 0,
        "quarantined": 0
    }

    for epoch in range(1, num_epochs + 1):
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch}/{num_epochs}")
        print("=" * 80 + "\n")

        correct_count = 0
        total_count = 0

        for idx, prob in enumerate(problems, 1):
            task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)

            playbook_context = get_active_playbook_context(stage_manager, domain_id)

            if idx == 1 or idx % 10 == 0:
                print(f"[{idx}/{num_problems}] {prob['problem']}")
                if playbook_context:
                    print(f"  Using {len(playbook_context)} live strategies (prod + staging)")
                else:
                    print("  No staged strategies yet")

            task_input = TaskInput(
                task_id=task_db.id,
                description=task_db.prompt,
                domain="arithmetic",
                playbook_bullets=playbook_context,
                max_reasoning_steps=10
            )

            try:
                generator_output = generator(task_input)

                is_correct = generator_output.answer.strip() == prob["answer"]
                if is_correct:
                    correct_count += 1
                total_count += 1

                if idx == 1 or idx % 10 == 0:
                    print(
                        f"  Answer: {generator_output.answer} (Expected: {prob['answer']})\n"
                        f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}"
                    )

                save_task_output(session, task_db, {
                    "reasoning_trace": generator_output.reasoning_trace,
                    "answer": generator_output.answer,
                    "confidence": generator_output.confidence,
                    "bullets_referenced": generator_output.bullets_referenced,
                    "latency_ms": generator_output.latency_ms or 0,
                    "prompt_tokens": generator_output.prompt_tokens or 0,
                    "completion_tokens": generator_output.completion_tokens or 0
                })

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

                for insight in reflector_output.insights:
                    total_insights += 1
                    if insight.section.value == "Helpful":
                        helpful_insights += 1
                    elif insight.section.value == "Harmful":
                        harmful_insights += 1

                insights_for_curator = [
                    {
                        "content": insight.content,
                        "section": insight.section.value,
                        "confidence": insight.confidence,
                        "tags": insight.tags,
                        "source_task_id": task_db.id
                    }
                    for insight in reflector_output.insights
                ]
                if insights_for_curator:
                    curator_output = curator_service.merge_insights(
                        task_id=task_db.id,
                        domain_id=domain_id,
                        insights=insights_for_curator,
                        target_stage=PlaybookStage.SHADOW
                    )

                    curator_totals["new_bullets"] += curator_output.new_bullets_added
                    curator_totals["increments"] += curator_output.existing_bullets_incremented
                    curator_totals["quarantined"] += curator_output.bullets_quarantined

                    # Ensure the local session sees updates performed by the curator service
                    session.expire_all()

            except Exception as e:
                print(f"  ❌ Error on problem {idx}: {e}")
                continue

        accuracy = (correct_count / total_count * 100) if total_count else 0
        epoch_metrics.append(
            {
                "epoch": epoch,
                "correct": correct_count,
                "total": total_count,
                "accuracy": accuracy,
            }
        )

        print("\n" + "─" * 80)
        print(f"EPOCH {epoch} RESULTS: {correct_count}/{total_count} = {accuracy:.1f}%")
        print("─" * 80)

        promotion_result = stage_manager.check_all_promotions(domain_id)
        session.expire_all()

        curator_totals["quarantined"] += len(promotion_result["quarantined"])

        print("📈 Promotions:")
        print(f"  Promoted: {len(promotion_result['promoted'])}")
        print(f"  Quarantined: {len(promotion_result['quarantined'])}")
        print(f"  No Action: {len(promotion_result['no_action'])}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80 + "\n")

    baseline_accuracy = epoch_metrics[0]["accuracy"] if epoch_metrics else 0.0
    header = f"  {'Epoch':<5}{'Correct':<10}{'Total':<10}{'Accuracy':<12}{'Improvement':<12}"
    print("📊 Accuracy Progression:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for metric in epoch_metrics:
        improvement = metric["accuracy"] - baseline_accuracy
        print(
            f"  {metric['epoch']:<5}{metric['correct']:<10}{metric['total']:<10}"
            f"{metric['accuracy']:<12.1f}{improvement:<+12.1f}"
        )

    print(
        f"\n🧠 Insights Discovered: {total_insights} "
        f"(Helpful: {helpful_insights}, Harmful: {harmful_insights})"
    )

    print(
        f"🧮 Curator Updates: New {curator_totals['new_bullets']} | "
        f"Reinforced {curator_totals['increments']} | "
        f"Quarantined {curator_totals['quarantined']}"
    )

    stage_labels = {
        PlaybookStage.PROD: "Production (active)",
        PlaybookStage.STAGING: "Staging (candidate)",
        PlaybookStage.SHADOW: "Shadow (emerging)",
    }
    stage_icons = {
        PlaybookStage.PROD: "✅",
        PlaybookStage.STAGING: "🔄",
        PlaybookStage.SHADOW: "•",
    }

    session.expire_all()

    for stage in [PlaybookStage.PROD, PlaybookStage.STAGING, PlaybookStage.SHADOW]:
        bullets = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id,
            stage=stage
        ).all()
        print(f"\n{stage_labels[stage]} — {len(bullets)} strategies")

        top_bullets = sorted(bullets, key=lambda b: b.helpful_count, reverse=True)[:5]
        for bullet in top_bullets:
            ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
            icon = stage_icons[stage]
            print(f"  {icon} {bullet.content[:90]}...")
            print(
                f"    Helpful: {bullet.helpful_count}  "
                f"Harmful: {bullet.harmful_count}  Ratio: {ratio:.1f}"
            )

    print("\n" + "=" * 80)
    print("✅ Arithmetic Learning Experiment Complete!")
    print("=" * 80 + "\n")


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
        stage_manager = StageManager(session)
        curator_service = CuratorService()

        print(f"✓ Initialized Generator with {model_name}")
        print(f"✓ Initialized Reflector with {model_name}")

        num_problems = int(os.getenv("NUM_PROBLEMS", "20"))
        num_epochs = int(os.getenv("NUM_EPOCHS", "3"))

        print("\n📋 Configuration:")
        print(f"  Problems: {num_problems}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Model: {model_name}")

        run_arithmetic_experiment(
            session=session,
            generator=generator,
            reflector=reflector,
            stage_manager=stage_manager,
            curator_service=curator_service,
            num_problems=num_problems,
            num_epochs=num_epochs
        )

        session.close()

        print("\n💡 What happened:")
        print("  1. Generator used DSPy with playbook strategies as context")
        print("  2. Multi-epoch runs let strategies accrue helpful votes")
        print("  3. Reflector analyzed outcomes and extracted insights")
        print("  4. Curator deduplicated and promoted successful strategies")
        print("  5. Final summary highlights accuracy and stage distribution\n")

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
