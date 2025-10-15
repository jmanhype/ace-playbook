# Multi-Model ACE Deployment Guide

**Practical guide for deploying ACE across multiple LLM models**

Based on research findings documented in [MODEL_TRANSFER_ANALYSIS.md](./MODEL_TRANSFER_ANALYSIS.md), this guide provides **actionable implementation patterns** for organizations using ACE with multiple LLM providers.

---

## Key Insight

**Strategies are model-specific, not model-agnostic.**

Don't try to create universal playbooks that work across all models. Instead, maintain separate playbooks per model family and use ACE's fast adaptation (86.9% latency reduction) to build them efficiently.

---

## Table of Contents

1. [Architecture Patterns](#1-architecture-patterns)
2. [Domain Organization](#2-domain-organization)
3. [Database Schema](#3-database-schema)
4. [Code Examples](#4-code-examples)
5. [Testing Strategy](#5-testing-strategy)
6. [Migration Guide](#6-migration-guide)
7. [Cost Analysis](#7-cost-analysis)

---

## 1. Architecture Patterns

### Pattern A: Model-Scoped Domains (Recommended)

Treat model family as part of the domain identifier.

```python
# Instead of generic domain
domain_id = "multiplication"

# Use model-scoped domain
domain_id = f"multiplication-{model_family}"
# Examples:
#   "multiplication-qwen"
#   "multiplication-claude"
#   "multiplication-llama"
```

**Advantages**:
- Explicit model-specificity
- No cross-contamination
- Clear playbook ownership
- Easy to query and analyze

**Disadvantages**:
- More database entries
- Need to manage model family taxonomy

### Pattern B: Model-Specific Databases

Separate databases per model family.

```python
# Database routing
def get_session(model_family: str):
    db_url = f"sqlite:///ace_playbooks_{model_family}.db"
    engine = create_engine(db_url)
    return Session(engine)

# Usage
qwen_session = get_session("qwen")
claude_session = get_session("claude")
```

**Advantages**:
- Complete isolation
- Independent scaling
- Simpler schema (no model_family column)

**Disadvantages**:
- More operational complexity
- Harder to do cross-model analysis
- Database proliferation

### Pattern C: Hybrid (Model Family + Subdomain)

Best of both: family-scoped domains with subdomain granularity.

```python
def create_domain_id(base_domain: str, model_family: str, subdomain: Optional[str] = None) -> str:
    """
    Create model-specific domain identifier.

    Examples:
        create_domain_id("math", "qwen") → "math-qwen"
        create_domain_id("math", "qwen", "multiplication") → "math-qwen-multiplication"
    """
    parts = [base_domain, model_family]
    if subdomain:
        parts.append(subdomain)
    return "-".join(parts)
```

**Advantages**:
- Hierarchical organization
- Fine-grained control
- Easy filtering

---

## 2. Domain Organization

### Recommended Structure

```
domains/
  math/
    qwen/
      multiplication.json      # 4-digit × 4-digit
      division.json
      fractions.json
    claude/
      multiplication.json
      division.json
    llama/
      multiplication.json

  coding/
    qwen/
      python.json
      javascript.json
    claude/
      python.json
      javascript.json

  writing/
    qwen/
      technical_docs.json
    claude/
      technical_docs.json
```

### Domain Metadata

Track model-specific characteristics per domain:

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class DomainConfig:
    """Configuration for a model-specific domain."""
    domain_id: str
    base_domain: str              # "multiplication"
    model_family: str             # "qwen", "claude", "llama"
    model_name: str               # "qwen/qwen-2.5-7b-instruct"

    # Training metadata
    training_problems: int
    training_epochs: int
    current_accuracy: float

    # Promotion gates (model-specific tuning)
    shadow_helpful_min: int = 3
    staging_helpful_min: int = 5
    prod_helpful_min: int = 8
    staging_ratio_min: float = 3.0
    prod_ratio_min: float = 5.0

    # Transfer testing
    tested_transfers: List[str] = None  # Models tested for transfer
    transfer_results: Dict[str, float] = None  # Model → accuracy
```

---

## 3. Database Schema

### Option 1: Add model_family Column

Add `model_family` to existing schema:

```python
from sqlalchemy import Column, String, Index

class PlaybookBullet(Base):
    __tablename__ = "playbook_bullets"

    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    domain_id = Column(String, nullable=False)
    model_family = Column(String, nullable=False)  # NEW
    section = Column(String, nullable=False)
    stage = Column(Enum(PlaybookStage), default=PlaybookStage.SHADOW)

    # Composite index for efficient queries
    __table_args__ = (
        Index('ix_domain_model', 'domain_id', 'model_family'),
        Index('ix_model_stage', 'model_family', 'stage'),
    )
```

### Option 2: Model-Scoped domain_id (No Schema Change)

Use `domain_id` convention:

```python
# No schema changes needed!
# Just use naming convention:

domain_id = f"{base_domain}-{model_family}"

# Queries work naturally:
bullets = session.query(PlaybookBullet).filter_by(
    domain_id="multiplication-qwen"
).all()
```

**Recommendation**: Start with **Option 2** (convention-based) for simplicity. Migrate to **Option 1** (explicit column) if you need complex cross-model queries.

---

## 4. Code Examples

### Example 1: Model-Aware Task Creation

```python
def create_task_with_model(
    session,
    problem: str,
    ground_truth: str,
    base_domain: str,
    model_name: str
) -> Task:
    """
    Create task with model-specific domain.

    Args:
        session: DB session
        problem: Problem description
        ground_truth: Expected answer
        base_domain: Base domain (e.g., "multiplication")
        model_name: Full model name (e.g., "openrouter/qwen/qwen-2.5-7b-instruct")

    Returns:
        Task with model-scoped domain_id
    """
    # Extract model family from full name
    model_family = extract_model_family(model_name)

    # Create model-scoped domain
    domain_id = f"{base_domain}-{model_family}"

    task = Task(
        id=str(uuid.uuid4()),
        prompt=problem,
        ground_truth=ground_truth,
        domain_id=domain_id,
        metadata={
            "model_name": model_name,
            "model_family": model_family,
            "base_domain": base_domain
        }
    )

    session.add(task)
    session.commit()
    return task


def extract_model_family(model_name: str) -> str:
    """
    Extract model family from full model name.

    Examples:
        "openrouter/qwen/qwen-2.5-7b-instruct" → "qwen"
        "anthropic/claude-3-haiku" → "claude"
        "openai/gpt-4o-mini" → "gpt4o"
    """
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        return "qwen"
    elif "claude" in model_lower:
        return "claude"
    elif "llama" in model_lower:
        return "llama"
    elif "gpt-4" in model_lower:
        return "gpt4"
    elif "gpt-3" in model_lower:
        return "gpt3"
    elif "gemini" in model_lower:
        return "gemini"
    else:
        # Fallback: use provider prefix
        return model_name.split("/")[0].replace("openrouter/", "")
```

### Example 2: Model-Aware Bullet Retrieval

```python
def get_playbook_for_model(
    session,
    base_domain: str,
    model_name: str,
    stage: PlaybookStage = PlaybookStage.PROD
) -> List[PlaybookBullet]:
    """
    Get model-specific playbook bullets.

    Args:
        session: DB session
        base_domain: Base domain (e.g., "multiplication")
        model_name: Full model name
        stage: Playbook stage to retrieve

    Returns:
        List of bullets for this model and domain
    """
    model_family = extract_model_family(model_name)
    domain_id = f"{base_domain}-{model_family}"

    bullets = session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=stage
    ).order_by(
        PlaybookBullet.helpful_count.desc()
    ).all()

    return bullets
```

### Example 3: Transfer Testing Before Deployment

```python
def test_transfer_before_deploy(
    session,
    source_model: str,
    target_model: str,
    base_domain: str,
    test_problems: List[Dict],
    threshold: float = 0.5
) -> Dict:
    """
    Test if strategies transfer before committing to target model.

    Args:
        session: DB session
        source_model: Model that trained the playbook
        target_model: Model to test transfer to
        base_domain: Domain to test
        test_problems: Sample problems for transfer test
        threshold: Minimum accuracy to consider transfer successful

    Returns:
        Transfer test results with recommendation
    """
    # Get source model's playbook
    source_family = extract_model_family(source_model)
    source_domain = f"{base_domain}-{source_family}"

    bullets = session.query(PlaybookBullet).filter_by(
        domain_id=source_domain,
        stage=PlaybookStage.PROD
    ).all()

    bullet_contents = [b.content for b in bullets]

    # Test on target model
    target_generator = CoTGenerator(model=target_model, temperature=0.7)

    correct = 0
    total = len(test_problems)

    for prob in test_problems:
        task_input = TaskInput(
            task_id=str(uuid.uuid4()),
            description=prob["problem"],
            domain=base_domain,
            playbook_bullets=bullet_contents,
            max_reasoning_steps=10
        )

        try:
            output = target_generator(task_input)
            if output.answer.strip() == prob["answer"]:
                correct += 1
        except Exception:
            pass  # Count as incorrect

    accuracy = correct / total

    return {
        "source_model": source_model,
        "target_model": target_model,
        "domain": base_domain,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "transfer_works": accuracy >= threshold,
        "recommendation": (
            "✅ Transfer successful - can reuse playbook"
            if accuracy >= threshold
            else "❌ Transfer failed - train new playbook for target model"
        )
    }
```

### Example 4: Unified Training Pipeline

```python
def train_playbook_for_model(
    session,
    base_domain: str,
    model_name: str,
    training_problems: List[Dict],
    num_epochs: int = 3,
    check_existing: bool = True
) -> Dict:
    """
    Train model-specific playbook with optional reuse check.

    Args:
        session: DB session
        base_domain: Base domain
        model_name: Full model name
        training_problems: Problems for training
        num_epochs: Number of training epochs
        check_existing: If True, check for existing playbook first

    Returns:
        Training results
    """
    model_family = extract_model_family(model_name)
    domain_id = f"{base_domain}-{model_family}"

    # Check for existing playbook
    if check_existing:
        existing_bullets = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id,
            stage=PlaybookStage.PROD
        ).count()

        if existing_bullets > 0:
            print(f"ℹ️  Found {existing_bullets} existing PROD bullets for {domain_id}")
            print(f"   Skipping training (already trained)")
            return {"status": "skipped", "reason": "existing_playbook"}

    # Initialize components
    generator = CoTGenerator(model=model_name, temperature=0.7)
    reflector = GroundedReflector(model=model_name, temperature=0.3)

    results = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        epoch_correct = 0

        for prob in training_problems:
            # Create task
            task = create_task_with_model(
                session, prob["problem"], prob["answer"],
                base_domain, model_name
            )

            # Get current playbook
            bullets = get_playbook_for_model(session, base_domain, model_name)
            bullet_contents = [b.content for b in bullets]

            # Generate
            task_input = TaskInput(
                task_id=task.id,
                description=task.prompt,
                domain=base_domain,
                playbook_bullets=bullet_contents,
                max_reasoning_steps=10
            )

            gen_output = generator(task_input)

            is_correct = gen_output.answer.strip() == prob["answer"]
            if is_correct:
                epoch_correct += 1

            # Save output
            save_task_output(session, task, {
                "reasoning_trace": gen_output.reasoning_trace,
                "answer": gen_output.answer,
                "confidence": gen_output.confidence,
                "bullets_referenced": gen_output.bullets_referenced,
                "latency_ms": gen_output.latency_ms or 0,
                "prompt_tokens": gen_output.prompt_tokens or 0,
                "completion_tokens": gen_output.completion_tokens or 0
            })

            # Reflect
            reflector_input = ReflectorInput(
                task_id=task.id,
                reasoning_trace=gen_output.reasoning_trace,
                answer=gen_output.answer,
                confidence=gen_output.confidence,
                bullets_referenced=gen_output.bullets_referenced,
                domain=base_domain,
                ground_truth=prob["answer"],
                test_results="",
                error_messages=[],
                performance_metrics=""
            )

            reflector_output = reflector(reflector_input)

            # Curate
            curate_insights(session, task, reflector_output.insights)

        # Promote bullets
        promote_bullets(session, domain_id)

        accuracy = epoch_correct / len(training_problems) * 100
        results.append({"epoch": epoch, "accuracy": accuracy})

        print(f"  Accuracy: {accuracy:.1f}%")

    return {
        "status": "completed",
        "domain_id": domain_id,
        "model_family": model_family,
        "epochs": num_epochs,
        "results": results
    }
```

---

## 5. Testing Strategy

### Step 1: Train Source Model

```python
# Train on your primary model
train_playbook_for_model(
    session=session,
    base_domain="multiplication",
    model_name="openrouter/qwen/qwen-2.5-7b-instruct",
    training_problems=training_problems,
    num_epochs=3
)
```

### Step 2: Test Transfer on Sample

```python
# Before training separate playbook, test if transfer works
transfer_result = test_transfer_before_deploy(
    session=session,
    source_model="openrouter/qwen/qwen-2.5-7b-instruct",
    target_model="anthropic/claude-3-haiku",
    base_domain="multiplication",
    test_problems=sample_problems[:10],  # Small sample
    threshold=0.5  # 50% accuracy threshold
)

print(transfer_result["recommendation"])
```

### Step 3: Conditional Training

```python
if not transfer_result["transfer_works"]:
    # Transfer failed - train separate playbook
    print("Training new playbook for Claude...")
    train_playbook_for_model(
        session=session,
        base_domain="multiplication",
        model_name="anthropic/claude-3-haiku",
        training_problems=training_problems,
        num_epochs=3
    )
else:
    # Transfer worked - reuse playbook
    print("Transfer successful! Reusing Qwen playbook for Claude.")
```

---

## 6. Migration Guide

### Migrating from Single-Model to Multi-Model

If you have existing playbooks without model-specificity:

#### Step 1: Audit Current Playbooks

```python
def audit_existing_playbooks(session) -> Dict:
    """Audit current playbooks to understand scope."""
    domains = session.query(PlaybookBullet.domain_id).distinct().all()

    report = {}
    for (domain_id,) in domains:
        bullet_count = session.query(PlaybookBullet).filter_by(
            domain_id=domain_id
        ).count()

        report[domain_id] = {
            "total_bullets": bullet_count,
            "prod_bullets": session.query(PlaybookBullet).filter_by(
                domain_id=domain_id,
                stage=PlaybookStage.PROD
            ).count()
        }

    return report
```

#### Step 2: Tag Existing Playbooks with Model

```python
def tag_existing_playbooks_with_model(
    session,
    domain_id: str,
    model_family: str
) -> None:
    """
    Migrate existing playbook to model-specific domain.

    WARNING: This modifies domain_id for all bullets in the domain.
    """
    new_domain_id = f"{domain_id}-{model_family}"

    # Update all bullets
    bullets = session.query(PlaybookBullet).filter_by(
        domain_id=domain_id
    ).all()

    print(f"Migrating {len(bullets)} bullets from '{domain_id}' to '{new_domain_id}'")

    for bullet in bullets:
        bullet.domain_id = new_domain_id

    session.commit()

    print(f"✅ Migration complete")

# Usage
tag_existing_playbooks_with_model(
    session,
    domain_id="multiplication",
    model_family="qwen"  # The model you used for original training
)
```

#### Step 3: Update Application Code

```python
# OLD: Generic domain
bullets = session.query(PlaybookBullet).filter_by(
    domain_id="multiplication"
).all()

# NEW: Model-specific domain
model_family = extract_model_family(model_name)
domain_id = f"multiplication-{model_family}"
bullets = session.query(PlaybookBullet).filter_by(
    domain_id=domain_id
).all()
```

---

## 7. Cost Analysis

### Training Cost: Separate vs Shared Playbooks

**Scenario**: Support 3 models (Qwen, Claude, Llama) across 5 domains

| Approach | Playbooks to Train | Training Cost | Transfer Testing Cost | Total |
|----------|-------------------|---------------|---------------------|--------|
| **Naive (try shared)** | 5 base + 10 failed transfers | High | High | **Very High** |
| **Model-Specific (recommended)** | 15 (3 models × 5 domains) | Medium | Low (small samples) | **Medium** |
| **Hybrid** | 5 base + 7 adapted | Medium-High | Medium | **Medium-High** |

### ACE Adaptation Efficiency

**Key insight from ACE paper**: 86.9% reduction in adaptation latency

This means training a new model-specific playbook is **fast enough** that maintaining separate playbooks is practical.

**Example**:
- Traditional prompt engineering: 2-4 hours per domain per model
- ACE with 3 epochs: 10-15 minutes per domain per model

**Recommendation**: The cost of training separate playbooks (15 × 15 min = ~4 hours total) is **far less** than the cost of failed transfers in production.

---

## 8. Best Practices Summary

### Do ✅

1. **Treat model family as domain dimension** - use `domain_id = f"{base}-{model}"`
2. **Test transfer on small samples** before assuming portability
3. **Train separate playbooks** when transfer fails (>50% accuracy drop)
4. **Use format-explicit strategies** as cheap optimization (+10% recovery)
5. **Monitor model-specific accuracy** separately
6. **Document model assumptions** in playbook metadata

### Don't ❌

1. **Assume universal strategies** work across all models
2. **Skip transfer testing** before production deployment
3. **Ignore formatting instructions** (easy +10% win)
4. **Mix model-specific bullets** in same domain
5. **Train on Model A, deploy on Model B** without testing
6. **Use generic domain IDs** when models are different

---

## 9. Quick Start Template

```python
from ace.generator import CoTGenerator
from ace.models import PlaybookStage

# Configuration
BASE_DOMAIN = "multiplication"
MODELS = {
    "qwen": "openrouter/qwen/qwen-2.5-7b-instruct",
    "claude": "anthropic/claude-3-haiku",
    "llama": "openrouter/meta-llama/llama-3.1-8b-instruct"
}

# 1. Train playbooks for each model
for model_family, model_name in MODELS.items():
    print(f"\n🚀 Training playbook for {model_family}...")

    train_playbook_for_model(
        session=session,
        base_domain=BASE_DOMAIN,
        model_name=model_name,
        training_problems=training_problems,
        num_epochs=3,
        check_existing=True  # Skip if already trained
    )

# 2. Use model-specific playbooks in production
def solve_with_best_model(problem: str, preferred_model: str = "qwen"):
    """Solve problem using model-specific playbook."""

    model_name = MODELS[preferred_model]
    model_family = extract_model_family(model_name)
    domain_id = f"{BASE_DOMAIN}-{model_family}"

    # Get model-specific bullets
    bullets = session.query(PlaybookBullet).filter_by(
        domain_id=domain_id,
        stage=PlaybookStage.PROD
    ).all()

    bullet_contents = [b.content for b in bullets]

    # Generate with correct playbook
    generator = CoTGenerator(model=model_name, temperature=0.7)

    task_input = TaskInput(
        task_id=str(uuid.uuid4()),
        description=problem,
        domain=BASE_DOMAIN,
        playbook_bullets=bullet_contents,
        max_reasoning_steps=10
    )

    return generator(task_input)

# Usage
result = solve_with_best_model("1234 × 5678", preferred_model="qwen")
print(f"Answer: {result.answer}")
```

---

## 10. FAQ

**Q: Can I ever share playbooks across models?**

A: Test first with a small sample (10 problems). If accuracy is >50%, you *might* get away with it. But expect degradation.

**Q: What about within-family transfer (e.g., Llama 8B → Llama 70B)?**

A: Not tested yet, but likely works better than cross-family. Still test before deploying.

**Q: Is the extra training cost worth it?**

A: Yes. ACE trains fast (~15 min/domain/model). Failed transfers in production cost far more.

**Q: Can I use hybrid approaches?**

A: Yes, but add complexity. Start simple (separate playbooks), optimize later if needed.

**Q: What if I add a new model later?**

A: Train a new playbook for it. ACE's fast adaptation makes this practical.

---

**End of Guide**

For research background, see [MODEL_TRANSFER_ANALYSIS.md](./MODEL_TRANSFER_ANALYSIS.md)
