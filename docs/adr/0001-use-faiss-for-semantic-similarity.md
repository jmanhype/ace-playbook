# ADR 0001: Use FAISS for Semantic Similarity Search

**Date**: 2025-10-11
**Status**: Accepted
**Deciders**: ACE Team
**Tags**: storage, performance, ml

## Context

ACE Playbook requires efficient semantic deduplication and similarity search for playbook bullets. The system needs to:

1. **Deduplicate similar insights** before adding to playbook (cosine similarity ≥ 0.8)
2. **Retrieve relevant context** for Generator (top-k similar bullets for a given task)
3. **Scale to 1000+ bullets** per domain with P95 latency ≤ 15ms
4. **Support per-domain isolation** (no cross-domain leakage)

### Alternatives Considered

1. **PostgreSQL with pgvector**
   - Pros: Single storage backend, transactional consistency
   - Cons: Additional extension dependency, slower for large indices, not optimized for in-memory search

2. **Elasticsearch/OpenSearch**
   - Pros: Full-text + vector search, distributed scaling
   - Cons: Heavy operational overhead, overpowered for single-node use case, higher latency

3. **Chromadb/Pinecone**
   - Pros: Purpose-built for vector search, managed options available
   - Cons: External service dependencies, cost, network latency

4. **FAISS (Facebook AI Similarity Search)**
   - Pros: Industry-standard, optimized for speed (P50 ≤10ms), in-process (no network), supports multiple indices
   - Cons: In-memory only (requires persistence layer), no native transactionality

5. **Naive Python cosine similarity (no index)**
   - Pros: Simple, no dependencies
   - Cons: O(n) scan per query, unacceptable latency at scale

## Decision

We will use **FAISS IndexFlatIP** (inner product index) for semantic similarity search.

### Key Design Choices

1. **Index Type**: `IndexFlatIP` (exact inner product search)
   - L2-normalized embeddings → inner product ≡ cosine similarity
   - Exact results (no approximation error)
   - Fast enough for ≤1000 bullets per domain

2. **Per-Domain Indices**: Each domain gets its own FAISS index
   - Enforces domain isolation (CHK081)
   - Parallel search across domains if needed
   - Smaller index → faster search

3. **Persistence Strategy**: Store FAISS indices + bullet ID mappings to disk
   - FAISS binary format (`.index` files)
   - Pickle mappings (`.pkl` files)
   - Lazy loading on first access

4. **Embedding Model**: sentence-transformers `all-MiniLM-L6-v2`
   - 384-dimensional embeddings
   - Fast inference (CPU-optimized)
   - Good quality/speed tradeoff

## Consequences

### Positive

- **Performance**: Achieved P50 ~5-10ms, P95 ~12-15ms (meets SLA)
- **Simplicity**: No external services, minimal operational overhead
- **Domain Isolation**: Per-domain indices prevent cross-domain leakage
- **Memory Efficiency**: In-process cache, no serialization overhead
- **Battle-Tested**: FAISS used in production at Meta, industry standard

### Negative

- **In-Memory Limitation**: Full index must fit in RAM
  - Mitigated by per-domain sharding (1000 bullets × 384 dims × 4 bytes = 1.5MB per domain)
  - Not a concern until 100k+ bullets per domain

- **No Native Persistence**: Must implement save/load logic
  - Mitigated by `FAISSIndexManager` with automatic persistence

- **No Transactionality**: FAISS updates not atomic with SQLAlchemy
  - Mitigated by append-only updates (never delete bullets)
  - WAL-style recovery: rebuild index from database on corruption

- **Single-Node Limitation**: Cannot distribute across machines
  - Acceptable for v1.x (single-node deployment)
  - Future: consider Milvus/Qdrant for multi-node

### Risks

- **Memory Growth**: Unbounded index growth could exhaust RAM
  - Mitigation: Monitor per-domain bullet counts, alert at 10k bullets
  - Archival strategy: move old bullets to cold storage

- **Index Corruption**: Disk writes could fail mid-operation
  - Mitigation: Atomic file writes (tmp + rename), checksums
  - Recovery: Rebuild from database (journaled inserts)

## Validation

Performance benchmarks (T069):
```
Small playbook (10 bullets):   P50 ~5ms,  P95 ~8ms
Medium playbook (100 bullets):  P50 ~8ms,  P95 ~12ms
Large playbook (1000 bullets):  P50 ~10ms, P95 ~15ms
```

All metrics meet target SLA (P95 ≤ 15ms).

## References

- FAISS Documentation: https://github.com/facebookresearch/faiss
- ACE Playbook: `ace/utils/faiss_index.py` (FAISSIndexManager)
- Benchmarks: `tests/benchmarks/test_retrieval_performance.py`
- Performance Analysis: CHANGELOG.md v1.10.0 (T069)
