# Edge Cases and Error Handling

This document catalogs edge cases, error conditions, and recovery procedures for all ACE Playbook components.

## Table of Contents

1. [Curator Edge Cases](#curator-edge-cases)
2. [FAISS Edge Cases](#faiss-edge-cases)
3. [Embedding Edge Cases](#embedding-edge-cases)
4. [Repository Edge Cases](#repository-edge-cases)
5. [Promotion Gates Edge Cases](#promotion-gates-edge-cases)
6. [Circuit Breaker Edge Cases](#circuit-breaker-edge-cases)
7. [Error Handling Patterns](#error-handling-patterns)

## Curator Edge Cases

### Empty Insights List

**Condition**: `curator_input.insights` is empty

**Error**: `ValueError`

**Example**:
```python
curator_input = CuratorInput(
    insights=[],  # Empty!
    current_playbook=playbook
)

# Raises: ValueError("Insights list cannot be empty")
```

**Recovery**: Validate before calling curator
```python
if not insights:
    logger.warning("No insights to curate, skipping")
    return current_playbook
```

---

### Domain ID Mismatch

**Condition**: Playbook bullets have different `domain_id` than input

**Error**: `ValueError("Cross-domain access violation")`

**Example**:
```python
# Playbook has bullets from "domain-A"
playbook = [bullet1, bullet2]  # domain_id="domain-A"

# Try to curate with "domain-B"
curator_input = CuratorInput(
    domain_id="domain-B",  # Mismatch!
    insights=insights,
    current_playbook=playbook
)

# Raises: ValueError("Cross-domain access violation")
```

**Recovery**: Validate domain consistency
```python
playbook_domains = {b.domain_id for b in playbook}
if curator_input.domain_id not in playbook_domains and playbook_domains:
    raise ValueError("Domain mismatch")
```

---

### Batch Size Zero

**Condition**: `batch_merge()` called with empty task list

**Error**: `ValueError("task_insights cannot be empty")`

**Example**:
```python
service.batch_merge([])  # Empty list!
# Raises: ValueError
```

**Recovery**: Check before calling
```python
if not task_insights:
    return CuratorOutput(updated_playbook=[], stats={})
```

---

### Mixed Domain Batch

**Condition**: Batch contains tasks from different domains

**Error**: `ValueError("Mixed domains in batch")`

**Example**:
```python
batch = [
    ("task-1", "domain-A", insights1),
    ("task-2", "domain-B", insights2),  # Different domain!
]

service.batch_merge(batch)
# Raises: ValueError
```

**Recovery**: Group by domain before batching
```python
from itertools import groupby

for domain_id, group in groupby(tasks, key=lambda t: t[1]):
    service.batch_merge(list(group))
```

---

### No Playbook Fallback

**Condition**: `DEFAULT_DOMAIN_ID` used when domain not found

**Error**: **REMOVED** - This is a security risk

**Old (vulnerable)**:
```python
domain_id = task_insights[0][1] if task_insights else "default"  # BAD!
```

**New (secure)**:
```python
if not task_insights:
    raise ValueError("task_insights cannot be empty")

domain_id = task_insights[0][1]
validate_domain_id(domain_id)  # Strict validation
```

**Recovery**: Always provide explicit domain_id

---

## FAISS Edge Cases

### Zero Vectors

**Condition**: Adding empty vector list to FAISS

**Error**: `RuntimeError("Cannot add 0 vectors")`

**Example**:
```python
faiss_manager.add_vectors(domain_id, [], [])
# Raises: RuntimeError
```

**Recovery**: Check vector list size
```python
if not vectors:
    logger.info("No vectors to add, skipping FAISS update")
    return
```

---

### Dimension Mismatch

**Condition**: Vector dimension doesn't match FAISS index

**Error**: `RuntimeError(f"Expected dimension {expected}, got {actual}")`

**Example**:
```python
# Index expects 384-dim
faiss_manager = FAISSIndexManager(dimension=384)

# Try to add 768-dim vector
vector = [0.1] * 768  # Wrong dimension!
faiss_manager.add_vectors(domain_id, [vector], ["id1"])

# Raises: RuntimeError("Expected dimension 384, got 768")
```

**Recovery**: Validate dimensions
```python
expected_dim = faiss_manager.dimension
actual_dim = len(vectors[0])

if actual_dim != expected_dim:
    raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")
```

---

### Index Not Initialized

**Condition**: Searching before any vectors added

**Error**: Returns empty results (not an error)

**Example**:
```python
# Search on empty index
results = faiss_manager.search(domain_id, query, k=5)
# Returns: []
```

**Recovery**: Check index size before searching
```python
if faiss_manager.get_index_size(domain_id) == 0:
    logger.warning(f"FAISS index for {domain_id} is empty")
    return []  # No results possible
```

---

### Cross-Domain Index Contamination

**Condition**: Wrong domain_id used in FAISS operations

**Error**: Incorrect results (silent failure!)

**Example**:
```python
# Add vectors to "domain-A"
faiss_manager.add_vectors("domain-A", vectors_a, ids_a)

# Accidentally search "domain-B"
results = faiss_manager.search("domain-B", query, k=5)
# Returns: [] (should have raised error!)
```

**Recovery**: Strict domain validation
```python
def search(self, domain_id: str, query_vector, k: int):
    validate_domain_id(domain_id)  # Validate format

    if domain_id not in self.indices:
        raise ValueError(f"No FAISS index for domain {domain_id}")

    return self.indices[domain_id].search(query_vector, k)
```

---

## Embedding Edge Cases

### Empty String

**Condition**: Embedding empty text

**Error**: Returns zero vector (dimension-sized)

**Example**:
```python
embedding = embedding_service.embed("")
# Returns: [0.0] * 384  # All zeros!
```

**Recovery**: Validate input
```python
if not text or not text.strip():
    raise ValueError("Cannot embed empty string")
```

---

### Unicode and Special Characters

**Condition**: Text with emoji, special chars, non-Latin scripts

**Error**: No error (model handles it)

**Example**:
```python
embedding = embedding_service.embed("Hello 👋 世界 🌍")
# Works fine, returns 384-dim vector
```

**Recovery**: No action needed (sentence-transformers handles this)

---

### Text Length Exceeds Model Limit

**Condition**: Text longer than 512 tokens (model limit)

**Error**: Truncated silently (no error)

**Example**:
```python
long_text = "word " * 1000  # Very long
embedding = embedding_service.embed(long_text)
# Only first ~512 tokens embedded, rest ignored
```

**Recovery**: Validate or truncate explicitly
```python
MAX_CONTENT_LENGTH = 500  # chars

if len(content) > MAX_CONTENT_LENGTH:
    logger.warning(f"Content truncated: {len(content)} > {MAX_CONTENT_LENGTH}")
    content = content[:MAX_CONTENT_LENGTH]
```

---

### Batch Size Too Large

**Condition**: Embedding >1000 texts in single batch

**Error**: OOM or slow performance

**Example**:
```python
texts = ["text"] * 10000
embeddings = embedding_service.embed_batch(texts)
# May cause OOM or take minutes
```

**Recovery**: Batch in chunks
```python
BATCH_SIZE = 100

def embed_large_batch(texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embeddings.extend(embedding_service.embed_batch(batch))
    return embeddings
```

---

## Repository Edge Cases

### Database Connection Lost

**Condition**: SQLite file deleted or corrupted mid-operation

**Error**: `OperationalError("database is locked")`

**Example**:
```python
# Mid-transaction, database file deleted
repo.create(bullet)
# Raises: OperationalError
```

**Recovery**: Retry with exponential backoff
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def create_with_retry(bullet):
    return repo.create(bullet)
```

---

### Concurrent Write Conflict

**Condition**: Two processes updating same bullet simultaneously

**Error**: One succeeds, one gets stale data

**Example**:
```python
# Process A: Read bullet (helpful=5)
bullet_a = repo.get(bullet_id)

# Process B: Read bullet (helpful=5)
bullet_b = repo.get(bullet_id)

# Process A: Increment to 6, commit
bullet_a.helpful_count += 1
repo.update(bullet_a)

# Process B: Increment to 6 (stale!), commit
bullet_b.helpful_count += 1
repo.update(bullet_b)  # Overwrites A's change!

# Result: helpful=6 (should be 7)
```

**Recovery**: Use atomic increment
```python
def increment_helpful(self, bullet_id: str):
    """Atomic counter increment."""
    with self.session.begin():
        stmt = (
            update(PlaybookBullet)
            .where(PlaybookBullet.id == bullet_id)
            .values(helpful_count=PlaybookBullet.helpful_count + 1)
        )
        self.session.execute(stmt)
```

---

### Invalid Domain ID Format

**Condition**: Domain ID with special chars or too long

**Error**: `ValueError("Invalid domain_id format")`

**Example**:
```python
repo.get_all("domain@#$%")  # Invalid chars
# Raises: ValueError
```

**Recovery**: Validate format
```python
import re

def validate_domain_id(domain_id: str):
    if not re.match(r"^[a-z0-9-]{3,50}$", domain_id):
        raise ValueError(f"Invalid domain_id format: {domain_id}")
```

---

## Promotion Gates Edge Cases

### Zero Helpful Count

**Condition**: Bullet has helpful=0, harmful=0

**Error**: No error (stays in SHADOW)

**Example**:
```python
bullet = PlaybookBullet(helpful_count=0, harmful_count=0)
policy.check_promotion(bullet)
# Returns: None (not eligible)
```

**Recovery**: Expected behavior (bullets start at 0)

---

### Division by Zero in Ratio

**Condition**: Calculating ratio with harmful=0

**Error**: Could cause `ZeroDivisionError` if not handled

**Example**:
```python
# Vulnerable code
ratio = bullet.helpful_count / bullet.harmful_count  # ZeroDivisionError!
```

**Recovery**: Use max(harmful, 1)
```python
ratio = bullet.helpful_count / max(bullet.harmful_count, 1)
# When harmful=0, ratio = helpful/1 = helpful (effectively infinite)
```

---

### Equal Helpful and Harmful

**Condition**: Bullet has helpful=5, harmful=5

**Error**: Triggers quarantine

**Example**:
```python
bullet = PlaybookBullet(helpful_count=5, harmful_count=5)

if bullet.harmful_count >= bullet.helpful_count:
    policy.quarantine(bullet)
```

**Recovery**: Expected behavior (ambiguous bullets quarantined)

---

## Circuit Breaker Edge Cases

### Half-Open State Flapping

**Condition**: Circuit repeatedly opens/closes

**Error**: Unstable behavior

**Example**:
```python
# Circuit opens after 5 failures
# After 60s, tries request → succeeds
# Circuit closes
# Next request fails
# Circuit opens again (flapping)
```

**Recovery**: Increase success threshold
```python
breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    success_threshold=3  # Require 3 successes before fully closing
)
```

---

### Timeout vs Connection Error

**Condition**: Different error types need different handling

**Error**: Timeout may recover, connection error may not

**Example**:
```python
try:
    result = call_llm_api()
except Timeout:
    # Retry (temporary issue)
    pass
except ConnectionError:
    # Don't retry immediately (API down)
    open_circuit()
```

**Recovery**: Distinguish error types
```python
@breaker.call
def call_api():
    try:
        return api.request()
    except Timeout:
        # Transient - count as soft failure
        breaker.record_soft_failure()
    except ConnectionError:
        # Hard failure - open circuit immediately
        breaker.open()
```

---

## Error Handling Patterns

### Exception Hierarchy

```
Exception
├── ValueError (invalid inputs)
│   ├── DomainIsolationError (cross-domain violation)
│   ├── InvalidDomainIDError (format validation)
│   └── EmptyInsightsError (empty list)
├── RuntimeError (system failures)
│   ├── FAISSError (index failures)
│   ├── CircuitBreakerOpen (service unavailable)
│   └── DatabaseError (connection issues)
└── OperationalError (database-specific)
```

### Custom Exceptions

```python
# ace/exceptions.py
class DomainIsolationError(ValueError):
    """Raised when cross-domain access attempted."""
    pass

class CircuitBreakerOpen(RuntimeError):
    """Raised when circuit breaker is open."""
    pass

class FAISSError(RuntimeError):
    """Raised for FAISS operation failures."""
    pass
```

### Error Response Format

```python
{
    "error": "ValueError",
    "message": "Insights list cannot be empty",
    "context": {
        "domain_id": "arithmetic",
        "task_id": "task-001",
        "timestamp": "2025-01-15T10:30:00Z"
    },
    "recovery": "Ensure insights list is not empty before calling curator"
}
```

### Logging Pattern

```python
import structlog

logger = structlog.get_logger()

try:
    output = curator.apply_delta(curator_input)
except ValueError as e:
    logger.error(
        "curator_validation_failed",
        error=str(e),
        domain_id=curator_input.domain_id,
        task_id=curator_input.task_id,
        insights_count=len(curator_input.insights)
    )
    raise
```

### Retry Pattern

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, Timeout))
)
def call_with_retry():
    return api.request()
```

### Graceful Degradation

```python
try:
    bullets = repo.get_top_k(domain_id, k=40)
except DatabaseError:
    logger.warning("Database unavailable, using empty playbook")
    bullets = []  # Continue with empty playbook

# Still execute task (without playbook context)
output = generator.forward(task_input)
```

## Cross-Reference

- [Architecture](architecture.md) - System design context
- [Runbook](runbook.md) - Operational procedures
- [API Reference](api/index.rst) - Detailed method signatures
- [Onboarding](onboarding.md) - Development guidelines
