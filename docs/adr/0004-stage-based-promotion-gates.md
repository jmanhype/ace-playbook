# ADR 0004: Stage-Based Promotion Gates for Bullet Lifecycle

**Date**: 2025-10-13
**Status**: Accepted
**Deciders**: ACE Team
**Tags**: lifecycle, quality-gates, shadow-learning

## Context

ACE Playbook uses online learning: new insights are continuously generated and merged into playbooks. However, not all insights are immediately production-ready. We need a staged rollout to validate quality before exposing insights to users.

### Problem Statement

Without staged rollout:
- **Risky Deployments**: Unvalidated insights go directly to production
- **No Validation**: No empirical evidence that insights improve outcomes
- **Hard Rollback**: Difficult to isolate and remove bad insights
- **User Impact**: Poor insights degrade Generator quality immediately

### Requirements

1. **Shadow Testing**: Test new insights without production impact
2. **Gradual Rollout**: SHADOW → STAGING → PROD progression
3. **Objective Criteria**: Data-driven promotion (not manual approval)
4. **Safety**: Automatically quarantine harmful insights
5. **Auditability**: Track promotion history in DiffJournal

## Alternatives Considered

1. **No Staging (Direct to Production)**
   ```python
   curator.apply_delta(delta)  # Immediately in PROD
   ```
   - Pros: Simple, fast iteration
   - Cons: Risky, no validation, hard to rollback

2. **Manual Approval Required**
   ```python
   if stage == Stage.SHADOW and admin_approved:
       bullet.stage = Stage.PROD
   ```
   - Pros: Human oversight, maximum safety
   - Cons: Bottleneck, not scalable, subjective

3. **Time-Based Promotion**
   ```python
   if bullet.age > 7 days:
       bullet.stage = Stage.PROD  # Automatic after 1 week
   ```
   - Pros: Predictable timeline
   - Cons: Ignores quality, promotes bad insights equally

4. **Counter-Based Promotion Gates (CHOSEN)**
   ```python
   if bullet.helpful_count >= 3 and bullet.helpful_count / bullet.harmful_count >= 3.0:
       bullet.stage = Stage.STAGING  # Data-driven
   ```
   - Pros: Objective, empirical validation, automatic
   - Cons: Requires feedback signals, slower for rare patterns

5. **A/B Testing with Statistical Significance**
   ```python
   if statistical_test(shadow_metrics, prod_metrics).p_value < 0.05:
       promote(bullet)
   ```
   - Pros: Rigorous, scientifically sound
   - Cons: Complex, requires large sample size, slow

## Decision

We will use **counter-based promotion gates** with three stages: SHADOW, STAGING, PROD.

### Stage Definitions

| Stage      | Visibility              | Purpose                          | Promotion Criteria                                |
|------------|-------------------------|----------------------------------|---------------------------------------------------|
| **SHADOW** | Not visible to users    | Parallel testing without impact  | `helpful_count ≥ 3`, `ratio ≥ 3.0`               |
| **STAGING**| Visible to 10% of users | Limited rollout for validation   | `helpful_count ≥ 5`, `ratio ≥ 5.0`               |
| **PROD**   | Visible to all users    | Fully deployed                   | N/A (final stage)                                 |
| **QUARANTINED** | Never visible      | Harmful insights removed         | `harmful_count ≥ helpful_count` (auto-detected)   |

### Promotion Criteria

```python
# SHADOW → STAGING
if bullet.stage == Stage.SHADOW:
    if bullet.helpful_count >= 3 and bullet.helpful_count / bullet.harmful_count >= 3.0:
        stage_manager.promote(bullet.id, Stage.STAGING)

# STAGING → PROD
if bullet.stage == Stage.STAGING:
    if bullet.helpful_count >= 5 and bullet.helpful_count / bullet.harmful_count >= 5.0:
        stage_manager.promote(bullet.id, Stage.PROD)

# Any stage → QUARANTINED (safety)
if bullet.harmful_count >= bullet.helpful_count and bullet.harmful_count > 0:
    stage_manager.quarantine(bullet.id, reason="harmful_ratio_exceeded")
```

### Shadow Execution Flow

```python
class OnlineLearningLoop:
    def process_task(self, task: str) -> str:
        # PROD path (user-facing)
        prod_bullets = stage_manager.get_production_bullets(domain_id)
        prod_response = generator.generate(task, context=prod_bullets)

        # SHADOW path (testing, parallel)
        shadow_bullets = stage_manager.get_all_bullets(domain_id)  # Includes SHADOW
        shadow_response = generator.generate(task, context=shadow_bullets)

        # Reflect on BOTH outcomes
        prod_reflection = reflector.reflect(task, prod_response)
        shadow_reflection = reflector.reflect(task, shadow_response)

        # Merge insights (SHADOW stage initially)
        curator.apply_delta(shadow_reflection, target_stage=Stage.SHADOW)

        return prod_response  # User sees PROD response only
```

## Consequences

### Positive

- **Safe Rollout**: Insights validated before production exposure
- **Automatic Promotion**: No manual bottleneck, scales with usage
- **Empirical Validation**: Counters reflect actual performance
- **Fast Quarantine**: Harmful insights removed automatically
- **Auditability**: DiffJournal tracks all promotions with timestamps
- **Parallel Testing**: SHADOW path runs without user impact

### Negative

- **Cold Start**: New insights require 3 helpful validations before STAGING
  - Mitigation: Start in SHADOW, accumulate evidence over time
  - Trade-off: Safety > speed

- **Rare Patterns Stuck in SHADOW**: Infrequently used insights may never reach threshold
  - Example: "Handle edge case X" only seen once per month
  - Mitigation: Lower threshold for low-frequency domains, manual override for critical insights

- **Counter Inflation**: High-frequency patterns accumulate counts faster
  - Example: "Use CoT" seen 100 times → promotes quickly
  - Mitigation: Acceptable (frequent patterns ARE important), ratio prevents spam

### Risks

- **False Positives**: Harmful insight promoted due to noisy feedback
  - Mitigation: Ratio threshold (3.0, 5.0) requires strong signal
  - Guardrails: GuardrailMonitor detects performance regressions

- **Gaming**: Malicious actors could artificially inflate helpful_count
  - Mitigation: Feedback signals tied to actual task outcomes (not user votes)
  - Future: Anomaly detection on counter velocity

- **Stuck in SHADOW**: Bug in promotion logic prevents advancement
  - Mitigation: Manual promotion endpoint (`/admin/bullets/{id}/promote`)
  - Monitoring: Alert on "bullets in SHADOW > 7 days with count ≥ 10"

## Validation

Tests (T059, T064):
- `test_promote_shadow_to_staging`: Validates promotion criteria
- `test_promotion_gate_not_met`: Ensures thresholds enforced
- `test_quarantine_harmful_bullets`: Auto-quarantine harmful insights
- `test_get_production_bullets_excludes_shadow`: Stage filtering

Integration tests (T063):
- `test_shadow_learning_parallel_execution`: PROD and SHADOW paths independent
- `test_shadow_learning_no_user_impact`: User sees PROD response only

## Alternative Promotion Strategies (Future)

If counter-based gates prove insufficient, consider:

1. **Confidence-Weighted Promotion**:
   ```python
   weighted_helpful = sum(count * confidence for count, confidence in helpful_feedback)
   if weighted_helpful >= 3.0:
       promote()
   ```

2. **Multi-Armed Bandit**:
   - Thompson Sampling to balance exploration (SHADOW) vs exploitation (PROD)
   - Gradually shift traffic based on empirical reward

3. **Staged Rollout Percentages**:
   - SHADOW: 0% of users
   - STAGING: 10% of users
   - PROD: 100% of users

## References

- Implementation: `ace/stage/stage_manager.py`
- Online Learning: `ace/online/learning_loop.py`
- Tests: `tests/unit/test_stage_manager.py`, `tests/integration/test_shadow_learning.py`
- Audit Trail: `ace/repository/journal_repository.py`
- Metrics: `ace/ops/metrics.py` (`promotion_events_total`, `quarantine_events_total`)

## Further Reading

- [Google: Canary Deployments](https://cloud.google.com/architecture/application-deployment-and-testing-strategies#canary)
- [AWS: Progressive Delivery](https://docs.aws.amazon.com/wellarchitected/latest/devops-guidance/progressive-delivery.html)
- [Feature Flags and Staged Rollouts](https://martinfowler.com/articles/feature-toggles.html)
