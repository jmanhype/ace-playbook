# ACE Review Queue Guide

This guide explains the review queue system for managing low-confidence insights that require human approval before entering the playbook.

## Overview

The Review Queue implements T062 from the ACE specification. When the Reflector generates insights with confidence scores below 0.6, these insights are automatically queued for manual review rather than being merged directly into the playbook.

This provides a safety mechanism to:
- Prevent low-quality insights from entering the playbook automatically
- Enable human oversight for uncertain patterns
- Build confidence in insights before production deployment
- Maintain playbook quality standards

## Architecture

### Components

1. **ReviewQueueItem Model** (`ace/models/review_queue.py`)
   - Database model for queued insights
   - Tracks status: pending, approved, rejected
   - Records reviewer decisions and timestamps

2. **ReviewQueueRepository** (`ace/repositories/review_queue_repository.py`)
   - Database access layer for review queue operations
   - Methods: add_to_queue, get_pending, approve, reject

3. **ReviewService** (`ace/ops/review_service.py`)
   - Service layer integrating with curator
   - Handles approval workflow with bullet creation
   - Manages rejection and statistics

4. **CLI Interface** (`scripts/ace_review.py`)
   - Command-line tool for review management
   - Commands: list, approve, reject, stats

### Integration with Online Loop

The OnlineLearningLoop automatically routes insights based on confidence:

```python
# Confidence threshold: 0.6
if insight.confidence < 0.6:
    # Queue for review
    review_service.queue_insight(insight, task_id, domain_id)
else:
    # Process directly with curator
    curator.batch_merge([insight], ...)
```

## CLI Usage

### List Pending Reviews

View all insights awaiting review:

```bash
python scripts/ace_review.py list
```

Output:
```
ID       Domain      Section   Conf  Content                                           Task     Created
------------------------------------------------------------------------
abc12345 arithmetic  helpful   0.55  Break complex problems into smaller steps...     t001     2025-10-13 15:30
def67890 geometry    harmful   0.42  Assume all angles are right angles...            t002     2025-10-13 15:35

Total pending: 2
```

### Filter by Domain

```bash
python scripts/ace_review.py list --domain arithmetic
```

### Limit Results

```bash
python scripts/ace_review.py list --limit 10
```

### Approve Review Item

Approve an insight and promote to shadow stage:

```bash
python scripts/ace_review.py approve abc12345 \
  --reviewer john \
  --notes "Valid decomposition strategy"
```

Output:
```
Approving review item: abc12345
Content: Break complex problems into smaller steps
Confidence: 0.55
Domain: arithmetic

✓ Approved and promoted to shadow stage.
Created bullet ID: B123
```

The approved insight:
1. Creates a new PlaybookBullet in shadow stage
2. Records the approval in review_queue table
3. Logs the decision with reviewer and notes

### Reject Review Item

Reject and discard an insight:

```bash
python scripts/ace_review.py reject def67890 \
  --reviewer jane \
  --notes "Too broad and likely to cause errors"
```

Output:
```
Rejecting review item: def67890
Content: Assume all angles are right angles
Confidence: 0.42
Domain: geometry

✓ Review item rejected and discarded.
```

The rejected insight:
1. Marked as rejected (not added to playbook)
2. Records the rejection reason
3. Available in statistics for analysis

### View Statistics

Show review queue statistics:

```bash
python scripts/ace_review.py stats
```

Output:
```
Review Queue Statistics:

Total items:    25
Pending:        8
Approved:       12
Rejected:       5
```

Filter by domain:

```bash
python scripts/ace_review.py stats --domain arithmetic
```

## Programmatic Usage

### Queue Insight for Review

```python
from ace.ops import create_review_service
from ace.reflector import InsightCandidate, InsightSection
from ace.utils.database import get_session

with get_session() as session:
    review_service = create_review_service(session)

    # Check if should be queued
    if review_service.should_queue_for_review(confidence=0.55):
        # Queue low-confidence insight
        item = review_service.queue_insight(
            insight=InsightCandidate(
                content="Break problems into steps",
                section=InsightSection.HELPFUL,
                confidence=0.55,
                rationale="Answer matched ground truth"
            ),
            source_task_id="task-001",
            domain_id="arithmetic"
        )

        print(f"Queued for review: {item.id}")
```

### List Pending Items

```python
with get_session() as session:
    review_service = create_review_service(session)

    # Get pending reviews
    pending = review_service.list_pending(
        domain_id="arithmetic",
        limit=10
    )

    for item in pending:
        print(f"{item.id}: {item.content[:50]} (conf: {item.confidence:.2f})")
```

### Approve with Promotion

```python
with get_session() as session:
    review_service = create_review_service(session)

    # Approve and create bullet in shadow stage
    bullet_id = review_service.approve_and_promote(
        item_id="abc12345",
        reviewer_id="john",
        review_notes="Valid strategy"
    )

    if bullet_id:
        print(f"Created bullet: {bullet_id}")
```

### Reject Item

```python
with get_session() as session:
    review_service = create_review_service(session)

    # Reject and discard
    success = review_service.reject(
        item_id="def67890",
        reviewer_id="jane",
        review_notes="Too risky"
    )

    if success:
        print("Item rejected")
```

## Confidence Threshold

The default confidence threshold is **0.6**:

```python
from ace.ops import REVIEW_CONFIDENCE_THRESHOLD

print(REVIEW_CONFIDENCE_THRESHOLD)  # 0.6
```

Insights with confidence < 0.6 are queued automatically.

To modify the threshold, update `REVIEW_CONFIDENCE_THRESHOLD` in `ace/ops/review_service.py`.

## Database Schema

The `review_queue` table structure:

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR | Primary key (UUID) |
| content | TEXT | Insight content |
| section | VARCHAR | "helpful" or "harmful" |
| confidence | REAL | Confidence score (< 0.6) |
| rationale | TEXT | Insight rationale |
| source_task_id | VARCHAR | Task that generated insight |
| domain_id | VARCHAR | Domain namespace |
| status | VARCHAR | "pending", "approved", "rejected" |
| created_at | TIMESTAMP | When queued |
| reviewed_at | TIMESTAMP | When reviewed |
| reviewer_id | VARCHAR | Reviewer identifier |
| review_notes | TEXT | Review decision notes |
| promoted_bullet_id | VARCHAR | Created bullet ID (if approved) |

Indexes:
- `idx_review_status_created`: (status, created_at) - List pending by date
- `idx_review_domain_status`: (domain_id, status) - Filter by domain
- `idx_review_source_task`: (source_task_id) - Track task insights

## Integration with Online Loop

When running the OnlineLearningLoop, review queue metrics are tracked:

```python
from ace.ops import create_online_loop

loop = create_online_loop(
    domain_id="production",
    use_shadow_mode=True
)

metrics = loop.run()

print(f"Insights queued for review: {metrics.insights_queued_for_review}")
```

The loop automatically:
1. Separates high-confidence (≥0.6) and low-confidence (<0.6) insights
2. Merges high-confidence insights directly to shadow stage
3. Queues low-confidence insights for review
4. Tracks counts in metrics

## Best Practices

1. **Review Regularly**: Check pending reviews daily to avoid backlog
2. **Domain Filtering**: Focus on one domain at a time for context
3. **Document Decisions**: Use `--notes` to explain approve/reject rationale
4. **Monitor Statistics**: Track approval/rejection rates to tune threshold
5. **Reviewer IDs**: Use consistent identifiers for accountability
6. **Batch Review**: Use `--limit` to process manageable batches

## Workflow Example

### Daily Review Workflow

```bash
# 1. Check statistics
python scripts/ace_review.py stats

# 2. List pending reviews
python scripts/ace_review.py list --limit 20

# 3. Review arithmetic domain
python scripts/ace_review.py list --domain arithmetic

# 4. Approve valid insights
python scripts/ace_review.py approve abc123 --reviewer john --notes "Good strategy"

# 5. Reject problematic insights
python scripts/ace_review.py reject def456 --reviewer john --notes "Too vague"

# 6. Check updated statistics
python scripts/ace_review.py stats --domain arithmetic
```

### Automated Review Workflow

```python
from ace.ops import create_review_service
from ace.utils.database import get_session

def review_workflow():
    with get_session() as session:
        review_service = create_review_service(session)

        # Get pending items
        pending = review_service.list_pending(limit=50)

        for item in pending:
            # Example: Auto-approve if confidence is close to threshold
            if item.confidence >= 0.55:
                review_service.approve_and_promote(
                    item_id=item.id,
                    reviewer_id="auto_reviewer",
                    review_notes="Auto-approved: confidence near threshold"
                )
            # Example: Auto-reject harmful insights with very low confidence
            elif item.section == "harmful" and item.confidence < 0.4:
                review_service.reject(
                    item_id=item.id,
                    reviewer_id="auto_reviewer",
                    review_notes="Auto-rejected: harmful with very low confidence"
                )
```

## Troubleshooting

### No Pending Items

If no items appear in queue:
1. Check that OnlineLearningLoop is running
2. Verify confidence threshold is set correctly
3. Ensure insights are being generated with low confidence
4. Check database connectivity

### Approval Fails

If approval fails:
1. Verify item exists and is pending
2. Check database session is active
3. Ensure curator is properly initialized
4. Review logs for embedding generation errors

### CLI Import Errors

```bash
# Ensure tabulate is installed
pip install tabulate
```

### Database Migration

Run migration to create review_queue table:

```bash
sqlite3 ace.db < migrations/add_review_queue.sql
```

Or use SQLAlchemy to create tables:

```python
from ace.utils.database import Base, engine
Base.metadata.create_all(engine)
```

## See Also

- [Online Learning Documentation](ONLINE_LEARNING.md)
- [Stage Management Guide](STAGE_MANAGEMENT.md)
- [Curator Documentation](../README.md#curator)
- [Architecture Documentation](../README.md)
