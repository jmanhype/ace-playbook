# ADR 0002: Append-Only Playbook Updates

**Date**: 2025-10-11
**Status**: Accepted
**Deciders**: ACE Team
**Tags**: data-model, consistency, auditability

## Context

The SemanticCurator must merge new insights into existing playbook bullets. When a new insight is semantically similar to an existing bullet (cosine similarity ≥ 0.8), we need to decide how to handle the update.

### Problem Statement

Given:
- Existing bullet: `content="Use careful prompt engineering"`, `helpful_count=2`, `harmful_count=0`
- New insight: `"Prompt engineering improves accuracy"` (similar=0.85)

What should the update semantics be?

### Alternatives Considered

1. **Replace content entirely**
   ```python
   bullet.content = new_insight.content  # Overwrites original
   bullet.helpful_count += 1
   ```
   - Pros: Simple, always reflects latest insight
   - Cons: Loses original wording, breaks audit trail, violates immutability

2. **Append with separator**
   ```python
   bullet.content = f"{bullet.content} | {new_insight.content}"
   bullet.helpful_count += 1
   ```
   - Pros: Preserves history, no data loss
   - Cons: Unbounded growth, messy formatting, duplicates

3. **Merge intelligently with LLM**
   ```python
   bullet.content = llm.merge(bullet.content, new_insight.content)
   bullet.helpful_count += 1
   ```
   - Pros: High-quality consolidation, removes redundancy
   - Cons: Expensive (LLM call per merge), non-deterministic, latency

4. **Append-only: Increment counters, preserve content** (CHOSEN)
   ```python
   # Content never changes
   bullet.helpful_count += 1  # Only counters update
   ```
   - Pros: Fast, deterministic, audit-friendly, immutable
   - Cons: First insight "wins", later insights not reflected in text

## Decision

We will use **append-only updates**: increment counters, never rewrite content.

### Update Rules

When a new insight matches an existing bullet (similarity ≥ 0.8):

1. **Content**: NEVER modify `bullet.content` (immutable)
2. **Counters**: Increment `helpful_count`, `harmful_count`, or `neutral_count` based on section
3. **Metadata**: Update `last_seen_at` timestamp
4. **Audit**: Log DeltaUpdate with before/after hashes

### Rationale

1. **Auditability**: Every bullet's content is traceable to a specific reflection
   - DiffJournal tracks which task created the bullet
   - SHA-256 hash in DeltaUpdate prevents tampering
   - Rollback requires no content restoration (counters only)

2. **Determinism**: No LLM involvement → predictable behavior
   - Same input always produces same output
   - Easier to debug and test
   - No API costs for merging

3. **Performance**: O(1) update (no LLM call, no string manipulation)
   - Curator latency stays ≤ 20ms
   - No backpressure from merge operations

4. **Simplicity**: Counters are sufficient signals for promotion gates
   - StageManager uses `helpful_count ≥ 3` (SHADOW → STAGING)
   - GuardrailMonitor uses `harmful_count ≥ helpful_count` (quarantine)
   - Ratio-based thresholds work without content analysis

5. **"First insight wins" is acceptable**:
   - High-quality reflections should be generated upfront
   - Similar insights reinforce (increment counter) rather than refine
   - Edge case: If first insight is poor quality → manual review or quarantine

## Consequences

### Positive

- **Fast**: No LLM calls, no string merging → <5ms per update
- **Predictable**: Same similarity → same merge behavior every time
- **Audit-Friendly**: DiffJournal tracks all changes, SHA-256 prevents tampering
- **Simple**: No complex merge logic, easy to understand and test
- **Rollback-Safe**: Decrement counters to undo, no content restoration needed

### Negative

- **First Insight Bias**: Initial wording locked in, later improvements not reflected
  - Example: First insight "use prompts" vs later "use chain-of-thought prompting"
  - Mitigation: High-quality reflections upfront, manual review for low-confidence

- **No Consolidation**: Multiple similar insights → multiple bullets (if below threshold)
  - Example: "use CoT" (0.78 similar) and "use chain-of-thought" (0.79 similar) → 2 bullets
  - Mitigation: Tune threshold to 0.8 (empirically validated), batch deduplication

- **Counter Inflation**: Highly frequent patterns accumulate large counts
  - Example: "use careful prompts" seen 100 times → `helpful_count=100`
  - Mitigation: Acceptable (signals strong consensus), promotion gates use ratios not absolute counts

### Risks

- **Poor First Insight**: If first insight is low quality, stuck with it
  - Mitigation: Require confidence ≥ 0.7 for PROD promotion (T055 review queue)
  - Manual override: admins can edit content directly (outside curator)

- **Semantic Drift**: Similar insights may diverge over time (0.85 → 0.75)
  - Mitigation: Embeddings are stable (frozen model), deduplication threshold conservative (0.8)

## Alternatives for Future Consideration

If append-only proves insufficient, consider:

1. **Periodic LLM consolidation** (offline batch job):
   - Merge bullets with `helpful_count ≥ 10` weekly
   - Human-in-the-loop approval before replacing content

2. **Versioned bullets**:
   - Track `bullet.version` (e.g., v1, v2, v3)
   - Allow refinement but preserve lineage

3. **Content variants**:
   - Store multiple phrasings: `bullet.content_variants = ["phrasing 1", "phrasing 2"]`
   - Generator randomly samples variant for diversity

## Validation

- Unit tests: `tests/unit/test_semantic_curator.py::test_duplicate_increments_helpful_count`
- E2E tests: `tests/e2e/test_smoke.py::test_deduplication_works`
- Audit trail: DiffJournal captures all delta updates with SHA-256 hashes

## References

- SemanticCurator: `ace/curator/semantic_curator.py:apply_delta()`
- DeltaUpdate model: `ace/models.py:DeltaUpdate`
- Audit trail: `ace/repository/journal_repository.py`
- Promotion gates: `ace/stage/stage_manager.py`
