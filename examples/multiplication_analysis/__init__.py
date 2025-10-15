"""
Shared utilities for multiplication learning analysis suite.

Common functions used across all evaluation scripts.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dspy
from ace.models import Base, Task, TaskOutput as TaskOutputDB, PlaybookStage, PlaybookBullet
from ace.generator import CoTGenerator, TaskInput
from ace.reflector import GroundedReflector, ReflectorInput


def setup_database(db_url: str = None) -> Session:
    """Initialize database connection."""
    if db_url is None:
        db_url = os.getenv("DATABASE_URL", "sqlite:///ace_playbook.db")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def configure_dspy_lm(model_override: str = None) -> Tuple[object, str]:
    """
    Configure DSPy with LLM backend.

    Args:
        model_override: Optional model name to use instead of auto-detection

    Returns:
        Tuple of (lm, model_name)
    """
    if model_override:
        # Use specific model if provided
        if "anthropic" in model_override or "claude" in model_override:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key or api_key.startswith("your_"):
                raise ValueError("ANTHROPIC_API_KEY required for Claude models")
            lm = dspy.LM(model_override, api_key=api_key)
            print(f"✓ Configured DSPy with {model_override}")
            return lm, model_override
        elif "gpt" in model_override or "openai" in model_override:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key.startswith("your_"):
                raise ValueError("OPENAI_API_KEY required for OpenAI models")
            lm = dspy.LM(model_override, api_key=api_key)
            print(f"✓ Configured DSPy with {model_override}")
            return lm, model_override
        elif "openrouter" in model_override:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY required for OpenRouter models")
            lm = dspy.LM(model_override, api_key=api_key, api_base="https://openrouter.ai/api/v1")
            print(f"✓ Configured DSPy with {model_override}")
            return lm, model_override

    # Auto-detect from environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openrouter_key:
        lm = dspy.LM(
            "openrouter/qwen/qwen-2.5-7b-instruct",
            api_key=openrouter_key,
            api_base="https://openrouter.ai/api/v1"
        )
        print("✓ Configured DSPy with OpenRouter (Qwen 2.5 7B Instruct)")
        return lm, "qwen-2.5-7b-instruct"
    elif openai_key and not openai_key.startswith("your_"):
        lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        print("✓ Configured DSPy with OpenAI GPT-4o-mini")
        return lm, "gpt-4o-mini"
    elif anthropic_key and not anthropic_key.startswith("your_"):
        lm = dspy.LM("anthropic/claude-3-haiku-20240307", api_key=anthropic_key)
        print("✓ Configured DSPy with Anthropic Claude-3-Haiku")
        return lm, "claude-3-haiku"
    else:
        raise ValueError(
            "No API keys found! Please set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env file"
        )


def generate_multiplication_problems(
    num_problems: int,
    min_val: int = 1,
    max_val: int = 10000,
    seed: int = 42
) -> List[Dict]:
    """
    Generate random multiplication problems.

    Args:
        num_problems: Number of problems to generate
        min_val: Minimum value for operands
        max_val: Maximum value for operands
        seed: Random seed for reproducibility

    Returns:
        List of dicts with 'problem', 'answer', 'a', 'b' keys
    """
    random.seed(seed)
    problems = []

    for _ in range(num_problems):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        answer = a * b
        problems.append({
            "problem": f"{a} × {b}",
            "answer": str(answer),
            "a": a,
            "b": b
        })

    return problems


def get_playbook_bullets(
    session: Session,
    domain_id: str,
    stage: PlaybookStage = PlaybookStage.PROD
) -> List[PlaybookBullet]:
    """Get playbook bullets for a domain and stage."""
    return session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=stage
    ).all()


def create_task_db(
    session: Session,
    problem: str,
    ground_truth: str,
    domain_id: str
) -> Task:
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


def save_task_output(
    session: Session,
    task: Task,
    generator_output: Dict
) -> TaskOutputDB:
    """Save generator output to database."""
    output = TaskOutputDB(
        task_id=task.id,
        reasoning_trace=generator_output["reasoning_trace"],
        answer=generator_output["answer"],
        confidence=generator_output["confidence"],
        bullets_referenced=generator_output["bullets_referenced"],
        latency_ms=generator_output.get("latency_ms", 0) or 0,
        token_count=(generator_output.get("prompt_tokens", 0) or 0) +
                    (generator_output.get("completion_tokens", 0) or 0)
    )
    session.add(output)
    session.commit()
    session.refresh(output)
    return output


def calculate_problem_difficulty(a: int, b: int) -> str:
    """
    Categorize problem difficulty based on operand sizes.

    Returns:
        "easy", "medium", or "hard"
    """
    max_operand = max(a, b)
    min_operand = min(a, b)

    if max_operand < 100:
        return "easy"
    elif max_operand < 1000 or min_operand < 100:
        return "medium"
    else:
        return "hard"


def print_header(title: str, width: int = 80):
    """Print a formatted header."""
    print(f"\n{'='*width}")
    print(title.center(width))
    print(f"{'='*width}\n")


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print(f"\n{'-'*width}")
    print(title)
    print(f"{'-'*width}")
