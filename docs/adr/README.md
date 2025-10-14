# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting key design decisions in the ACE Playbook project.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences. ADRs help teams:

- **Understand why decisions were made** (not just what was implemented)
- **Onboard new team members** by providing historical context
- **Avoid revisiting settled debates** unless circumstances change
- **Track technical debt** and alternatives considered

## ADR Format

Each ADR follows the [Michael Nygard format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):

- **Title**: Short noun phrase (e.g., "Use FAISS for Semantic Similarity")
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: Problem statement and constraints
- **Decision**: What was decided and why
- **Consequences**: Positive, negative, and risks

## Index

| ADR | Title | Status | Date | Tags |
|-----|-------|--------|------|------|
| [0001](./0001-use-faiss-for-semantic-similarity.md) | Use FAISS for Semantic Similarity Search | Accepted | 2025-10-11 | storage, performance, ml |
| [0002](./0002-append-only-playbook-updates.md) | Append-Only Playbook Updates | Accepted | 2025-10-11 | data-model, consistency, auditability |
| [0003](./0003-circuit-breaker-for-llm-apis.md) | Circuit Breaker for LLM API Calls | Accepted | 2025-10-14 | reliability, fault-tolerance, llm |
| [0004](./0004-stage-based-promotion-gates.md) | Stage-Based Promotion Gates for Bullet Lifecycle | Accepted | 2025-10-13 | lifecycle, quality-gates, shadow-learning |

## Decision Log

### Phase 1-2: Data Model and Embedding (v1.6.0)
- **ADR 0001**: FAISS for semantic similarity (over pgvector, Elasticsearch)
- **ADR 0002**: Append-only updates (over LLM merging, content replacement)

### Phase 10: Production Hardening (v1.10.0)
- **ADR 0003**: Circuit breaker for LLM API fault tolerance

### Phase 8: Online Adaptation (v1.8.0)
- **ADR 0004**: Stage-based promotion gates (SHADOW → STAGING → PROD)

## Creating a New ADR

1. **Copy template**:
   ```bash
   cp docs/adr/template.md docs/adr/NNNN-short-title.md
   ```

2. **Fill in sections**:
   - Date, Status, Deciders, Tags
   - Context: Why is this decision needed?
   - Decision: What did we decide?
   - Consequences: What are the trade-offs?

3. **Update index**: Add entry to this README.md

4. **Commit**: `git commit -m "docs: Add ADR NNNN - Short Title"`

## When to Write an ADR

Write an ADR when:
- **Choosing between alternatives** (e.g., database, library, architecture)
- **Making irreversible decisions** (e.g., API contracts, data model)
- **Accepting significant trade-offs** (e.g., performance vs simplicity)
- **Deviating from standards** (e.g., why not REST, why not microservices)

**Do NOT** write ADRs for:
- Minor implementation details (e.g., variable names)
- Obvious choices (e.g., using Python for a Python project)
- Reversible decisions (e.g., changing log levels)

## ADR Lifecycle

```
PROPOSED ──> ACCEPTED ──> DEPRECATED
                   │           ▲
                   └─────────┬─┘
                   SUPERSEDED BY ADR XXXX
```

- **PROPOSED**: Under discussion, not yet implemented
- **ACCEPTED**: Implemented and in use
- **DEPRECATED**: No longer recommended, but not replaced
- **SUPERSEDED**: Replaced by a newer ADR (link to successor)

## References

- [Michael Nygard: Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Organization](https://adr.github.io/)
- [ThoughtWorks: Lightweight ADRs](https://www.thoughtworks.com/en-us/radar/techniques/lightweight-architecture-decision-records)
