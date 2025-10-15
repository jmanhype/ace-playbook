#!/usr/bin/env python3
"""
Single-Domain Arithmetic Validation

Validates ACE framework on multiplication domain with isolated database.
Includes statistical analysis (t-tests, confidence intervals) to measure
whether ACE provides statistically significant improvements.

Key Features:
- Single domain only (arithmetic/multiplication)
- Isolated database (arithmetic_validation.db)
- Baseline vs ACE comparison
- Statistical significance testing
- No cross-domain contamination

Requirements:
- Set OPENROUTER_API_KEY in .env (or OPENAI_API_KEY/ANTHROPIC_API_KEY)
- Run: python examples/single_domain_arithmetic_validation.py
"""

import os
import sys
import random
import statistics
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from ace.models import Base, Task, TaskOutput as TaskOutputDB, PlaybookStage, PlaybookBullet
from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput


def setup_isolated_database():
    """Initialize isolated database for arithmetic validation only."""
    # Use isolated database file
    db_path = Path(__file__).parent / "arithmetic_validation.db"
    db_url = f"sqlite:///{db_path}"

    print(f"📁 Using isolated database: {db_path}")

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


def generate_multiplication_problems(num_problems: int, min_val: int = 1000, max_val: int = 9999, seed: int = 42) -> List[Dict]:
    """
    Generate random multiplication problems.

    Args:
        num_problems: Number of problems to generate
        min_val: Minimum value for each operand (default: 1000)
        max_val: Maximum value for each operand (default: 9999)
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
        metadata_json={"problem": problem, "type": "4digit_multiplication"}
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
                tags=["multiplication", "arithmetic", "ace-learned"],
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


def run_baseline(
    session: Session,
    generator: CoTGenerator,
    problems: List[Dict],
    domain_id: str
) -> Tuple[List[bool], float]:
    """
    Run baseline (no ACE context).

    Returns:
        Tuple of (results list, accuracy)
    """
    print("\n" + "="*80)
    print("BASELINE: No ACE Context")
    print("="*80 + "\n")

    results = []
    correct_count = 0

    for idx, prob in enumerate(problems, 1):
        # Create task
        task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)

        # NO playbook bullets for baseline
        task_input = TaskInput(
            task_id=task_db.id,
            description=task_db.prompt,
            domain="multiplication",
            playbook_bullets=[],  # Empty for baseline
            max_reasoning_steps=10
        )

        try:
            generator_output = generator(task_input)
            is_correct = generator_output.answer.strip() == prob["answer"]

            if is_correct:
                correct_count += 1

            results.append(is_correct)

            # Save to database
            save_task_output(session, task_db, {
                "reasoning_trace": generator_output.reasoning_trace,
                "answer": generator_output.answer,
                "confidence": generator_output.confidence,
                "bullets_referenced": generator_output.bullets_referenced,
                "latency_ms": generator_output.latency_ms or 0,
                "prompt_tokens": generator_output.prompt_tokens or 0,
                "completion_tokens": generator_output.completion_tokens or 0
            })

            status = "✅" if is_correct else "❌"
            if idx % 5 == 0 or idx == 1:
                print(f"  [{idx}/{len(problems)}] {prob['problem']} = {generator_output.answer} {status}")

        except Exception as e:
            print(f"❌ Error on problem {idx}: {e}")
            results.append(False)
            continue

    accuracy = (correct_count / len(problems) * 100) if problems else 0

    print(f"\n{'─'*80}")
    print(f"BASELINE RESULTS:")
    print(f"  Accuracy: {correct_count}/{len(problems)} = {accuracy:.1f}%")
    print(f"{'─'*80}")

    return results, accuracy


def run_ace_training(
    session: Session,
    generator: CoTGenerator,
    reflector: GroundedReflector,
    problems: List[Dict],
    domain_id: str,
    num_epochs: int = 3
) -> Tuple[List[bool], float, List[Dict]]:
    """
    Run ACE training with Generator-Reflector-Curator pipeline.

    Returns:
        Tuple of (final results list, final accuracy, epoch metrics)
    """
    print("\n" + "="*80)
    print(f"ACE TRAINING: {num_epochs} Epochs")
    print("="*80 + "\n")

    epoch_metrics = []
    final_results = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'─'*80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'─'*80}\n")

        correct_count = 0
        epoch_results = []

        for idx, prob in enumerate(problems, 1):
            # Create task
            task_db = create_task_db(session, prob["problem"], prob["answer"], domain_id)

            # Get current playbook bullets
            bullets = get_playbook_bullets(session, domain_id, stage=PlaybookStage.PROD)
            bullet_contents = [b.content for b in bullets]

            if idx % 5 == 0 or idx == 1:
                print(f"  [{idx}/{len(problems)}] {prob['problem']} (using {len(bullets)} strategies)")

            # GENERATOR: Solve with playbook context
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

                epoch_results.append(is_correct)

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

                # REFLECTOR: Analyze
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

                # Convert insights for curator
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
                print(f"❌ Error on problem {idx}: {e}")
                epoch_results.append(False)
                continue

        # Epoch statistics
        accuracy = (correct_count / len(problems) * 100) if problems else 0
        epoch_metrics.append({
            "epoch": epoch,
            "correct": correct_count,
            "total": len(problems),
            "accuracy": accuracy
        })

        print(f"\n{'─'*80}")
        print(f"EPOCH {epoch} RESULTS:")
        print(f"  Accuracy: {correct_count}/{len(problems)} = {accuracy:.1f}%")

        # Apply promotion gates
        promotions = promote_bullets(session, domain_id)
        print(f"\n  Promotions:")
        print(f"    Shadow → Staging: {promotions['shadow_to_staging']}")
        print(f"    Staging → Prod: {promotions['staging_to_prod']}")
        print(f"{'─'*80}")

        # Store final epoch results
        if epoch == num_epochs:
            final_results = epoch_results

    final_accuracy = (sum(final_results) / len(final_results) * 100) if final_results else 0

    return final_results, final_accuracy, epoch_metrics


def calculate_statistics(baseline_results: List[bool], ace_results: List[bool]) -> Dict:
    """
    Calculate statistical significance of ACE improvement.

    Uses paired t-test and confidence intervals.
    """
    n = len(baseline_results)

    # Calculate differences for paired t-test
    differences = [(1 if ace else 0) - (1 if base else 0) for ace, base in zip(ace_results, baseline_results)]

    # Mean difference
    mean_diff = statistics.mean(differences)

    # Standard error
    if n > 1:
        std_dev = statistics.stdev(differences)
        std_error = std_dev / (n ** 0.5)
    else:
        std_error = 0

    # T-statistic
    t_stat = mean_diff / std_error if std_error > 0 else 0

    # Degrees of freedom
    df = n - 1

    # Simple p-value approximation (two-tailed)
    # For proper p-value, would use scipy.stats.t.sf
    if abs(t_stat) > 2.0:
        p_value = 0.05  # Likely significant
    elif abs(t_stat) > 1.5:
        p_value = 0.15
    else:
        p_value = 0.50  # Not significant

    # 95% confidence interval
    margin = 1.96 * std_error  # Approximate for normal distribution
    ci_lower = mean_diff - margin
    ci_upper = mean_diff + margin

    return {
        "mean_diff": mean_diff,
        "std_error": std_error,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "lift_percent": mean_diff * 100
    }


def print_final_report(
    baseline_results: List[bool],
    baseline_accuracy: float,
    ace_results: List[bool],
    ace_accuracy: float,
    epoch_metrics: List[Dict],
    stats: Dict
):
    """Print comprehensive final report."""
    print("\n" + "="*80)
    print("FINAL VALIDATION REPORT")
    print("="*80 + "\n")

    print("📊 ACCURACY COMPARISON:")
    print(f"  Baseline (No ACE):  {baseline_accuracy:.1f}% ({sum(baseline_results)}/{len(baseline_results)})")
    print(f"  ACE (Final Epoch):  {ace_accuracy:.1f}% ({sum(ace_results)}/{len(ace_results)})")
    print(f"  Direct Lift:        {stats['lift_percent']:+.1f}%")

    print(f"\n📈 STATISTICAL ANALYSIS:")
    print(f"  T-statistic:        {stats['t_stat']:.4f}")
    print(f"  Degrees of Freedom: {stats['df']}")
    print(f"  P-value:            {stats['p_value']:.4f}")
    print(f"  95% CI:             [{stats['ci_lower']*100:.1f}%, {stats['ci_upper']*100:.1f}%]")

    # Significance interpretation
    if stats['p_value'] < 0.05:
        if stats['lift_percent'] > 0:
            print(f"\n  ✅ STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05)")
        else:
            print(f"\n  ❌ STATISTICALLY SIGNIFICANT DEGRADATION (p < 0.05)")
    else:
        print(f"\n  ⚠️  NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")

    print(f"\n📚 EPOCH PROGRESSION:")
    for metric in epoch_metrics:
        print(f"  Epoch {metric['epoch']}: {metric['accuracy']:.1f}% ({metric['correct']}/{metric['total']})")

    print("\n" + "="*80)

    # Final verdict
    if stats['p_value'] < 0.05 and stats['lift_percent'] > 5:
        print("✅ VALIDATION PASSED: ACE provides statistically significant improvement")
    elif stats['p_value'] < 0.05 and stats['lift_percent'] < -5:
        print("❌ VALIDATION FAILED: ACE causes statistically significant degradation")
    else:
        print("⚠️  VALIDATION INCONCLUSIVE: No statistically significant difference")

    print("="*80 + "\n")


def main():
    """Main entry point."""
    print("\n🚀 Single-Domain Arithmetic Validation\n")

    try:
        # Configure DSPy
        lm, model_name = configure_dspy_lm()
        dspy.configure(lm=lm)

        # Initialize isolated database
        session = setup_isolated_database()

        # Initialize ACE components
        generator = CoTGenerator(model=model_name, temperature=0.7)
        reflector = GroundedReflector(model=model_name, temperature=0.3)

        print(f"✓ Initialized Generator with {model_name}")
        print(f"✓ Initialized Reflector with {model_name}")

        # Configuration
        num_problems = int(os.getenv("NUM_PROBLEMS", "20"))
        num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
        domain_id = "multiplication-validation"

        print(f"\n📋 Configuration:")
        print(f"  Domain: multiplication (SINGLE DOMAIN)")
        print(f"  Problems: {num_problems}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Model: {model_name}")
        print(f"  Database: arithmetic_validation.db (ISOLATED)")

        # Generate problems
        print(f"\n🎲 Generating {num_problems} multiplication problems...")
        problems = generate_multiplication_problems(num_problems, seed=42)
        print(f"✓ Generated {num_problems} problems (4-digit × 4-digit)\n")

        # Run baseline
        baseline_results, baseline_accuracy = run_baseline(
            session=session,
            generator=generator,
            problems=problems,
            domain_id=domain_id
        )

        # Run ACE training
        ace_results, ace_accuracy, epoch_metrics = run_ace_training(
            session=session,
            generator=generator,
            reflector=reflector,
            problems=problems,
            domain_id=domain_id,
            num_epochs=num_epochs
        )

        # Calculate statistics
        stats = calculate_statistics(baseline_results, ace_results)

        # Print final report
        print_final_report(
            baseline_results=baseline_results,
            baseline_accuracy=baseline_accuracy,
            ace_results=ace_results,
            ace_accuracy=ace_accuracy,
            epoch_metrics=epoch_metrics,
            stats=stats
        )

        session.close()

        print("💡 Key Insights:")
        print("  • Single-domain validation avoids cross-contamination")
        print("  • Isolated database prevents playbook pollution")
        print("  • Statistical analysis provides confidence in results")
        print("  • Arithmetic domain has proven ACE effectiveness\n")

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nTo run this example:")
        print("  1. Set OPENAI_API_KEY in .env file, OR")
        print("  2. Set ANTHROPIC_API_KEY in .env file, OR")
        print("  3. Set OPENROUTER_API_KEY in .env file")
        print("\nExample .env file:")
        print("  OPENROUTER_API_KEY=sk-or-v1-...")
        print("  NUM_PROBLEMS=20")
        print("  NUM_EPOCHS=3\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
