# Research: BLACKICE 3.0 Technology Decisions

**Feature**: BLACKICE 3.0 Agentic Software Factory
**Date**: 2026-01-18
**Branch**: `007-blackice-3`

## 1. Event Sourcing Implementation

### Decision: Beads + JSONL

**Rationale**: The white paper specifies Beads as the event log format, and BLACKICE already has Beads integration. JSONL provides human-readable, append-only storage that is git-trackable.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| PostgreSQL event store | Requires database server; not suitable for offline/local runs |
| SQLite WAL | Less human-readable; harder to debug event sequences |
| Custom binary format | Not git-trackable; requires tooling to inspect |
| EventStoreDB | External dependency; overkill for single-machine runs |

**Implementation Notes**:
- Event schema uses Pydantic for validation
- Events are immutable once appended
- Snapshots stored as separate JSONL files for fast resume
- Hash chain for tamper detection (Enterprise)

## 2. Multi-Agent Coordination

### Decision: Internal Messaging with File Reservations

**Rationale**: Specialist agents (planner, implementer, reviewer, tester, security) need durable handoffs without external message brokers. File reservations prevent concurrent edit conflicts.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| Redis pub/sub | External dependency; not suitable for offline runs |
| ZeroMQ | Complex setup; overkill for single-process coordination |
| Actor model (Pykka) | Adds complexity; agents are relatively coarse-grained |
| Shared database locks | Less flexible than file-level reservations |

**Implementation Notes**:
- Threaded messaging uses in-memory queues with persistence to event log
- Reservations implemented as lease files with TTL
- Voting policies configurable per-task (majority, supermajority, unanimous, quorum, weighted)
- Escalation to cross-model attack patterns on low confidence

## 3. Model Provider Abstraction

### Decision: Protocol-Based Adapters with Failover Chain

**Rationale**: Different model providers have different APIs, capabilities, and availability. A unified interface with failover ensures resilience.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| LiteLLM wrapper | Additional dependency; some features not exposed |
| Direct API calls | No abstraction; hard to add new providers |
| LangChain | Too heavy; brings unwanted abstractions |
| Single provider only | No resilience; vendor lock-in |

**Implementation Notes**:
- Base ModelProvider protocol defines generate(), chat(), embed()
- Adapters: Claude, OpenAI, Ollama (local)
- Failover order configurable via environment/config
- Health checks determine provider availability
- Cost tracking per provider for budget management

## 4. Execution Environment

### Decision: Provider Pattern with Local/Container/Sandbox Options

**Rationale**: Execution environments must be interchangeable. Local is fastest for development, containers provide isolation for CI, ephemeral sandboxes for untrusted code.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| Docker only | Not available everywhere; overhead for simple runs |
| Local only | No isolation; security risk for untrusted code |
| Cloud functions | Latency; external dependency; cost |
| VM-based sandboxes | Too heavyweight; slow startup |

**Implementation Notes**:
- ExecutionProvider interface: execute(), health(), attach()
- Safety pipeline: shell unwrap → semantic parse → stack allowlist → policy check → execute
- Idempotency keys for external side effects
- Secrets injected via environment variables, never in command strings

## 5. Memory and Learning

### Decision: Letta Primary with Local JSONL Fallback

**Rationale**: Letta provides stateful agents with archival memory and semantic search. Local fallback ensures the system works when Letta is unavailable.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| Chroma/Weaviate | Vector DB only; no agent state management |
| SQLite FTS | No semantic search; limited cross-session context |
| Redis with RedisSearch | External dependency; not always available |
| In-memory only | No persistence across sessions |

**Implementation Notes**:
- MemoryProvider interface: put(), search(), load_context(), retention_policy()
- Letta adapter connects to Letta server for archival memory + embeddings
- Local JSONL fallback: keyword search, no embeddings (degraded mode)
- Memory is advisory, not authoritative; ground truth is events + artifacts
- Retention policies: redact, hash-only, TTL

## 6. Secrets Management

### Decision: Environment Variables with Vault Optional

**Rationale**: Environment variables are universally supported and simple. Vault integration available for enterprise deployments.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| Vault only | Not always available; complex setup |
| SOPS encrypted files | Requires decryption tooling; key management |
| AWS Secrets Manager | Cloud-specific; vendor lock-in |
| Hardcoded (masked) | Security risk; prohibited by constitution |

**Implementation Notes**:
- SecretsProvider interface: get(), inject_env(), redact()
- Secrets never appear in prompts, logs, or event payloads
- Receipts use hash-only mode for secret references
- Environment variables loaded at startup, validated for presence

## 7. Observability Stack

### Decision: Structlog + OpenTelemetry + Prometheus Metrics

**Rationale**: Constitution requires health checks, structured logs, and distributed tracing. This stack is production-proven and integrates well with existing tooling.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| Print statements | Not structured; hard to query |
| Logging module only | No correlation IDs; limited formatting |
| DataDog/Honeycomb | External dependency; cost |
| Custom telemetry | Reinventing the wheel |

**Implementation Notes**:
- Structlog with JSON output for machine parsing
- Correlation IDs (trace_id, run_id, task_id) in every log
- OpenTelemetry spans for distributed tracing
- Prometheus metrics exposed on /metrics endpoint
- Health checks: /health (liveness), /ready (readiness), /live (startup)

## 8. API Design

### Decision: REST with OpenAPI 3.1 Specification

**Rationale**: REST is widely understood and tooling-rich. OpenAPI enables contract testing and client generation.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| GraphQL | Overkill for this use case; learning curve |
| gRPC | Requires protobuf; less accessible |
| CLI only | No programmatic access; limits integrations |
| WebSocket only | Stateful; complex to debug |

**Implementation Notes**:
- FastAPI for automatic OpenAPI generation
- Versioned endpoints: /api/v1/
- Contract tests verify spec matches implementation
- WebSocket available for streaming run progress (optional)

## 9. TaskSpec Validation (Enterprise)

### Decision: Pydantic/JSON Schema with Strictness Tiers

**Rationale**: The white paper specifies pragmatic tooling that engineering teams can maintain. RDF/SHACL deferred to optional future module.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| RDF/SHACL/SPARQL | Complex; steep learning curve; limited tooling |
| Custom DSL | Requires parser; maintenance burden |
| YAML-only | No validation; error-prone |
| TypeScript types | Language-specific; not portable |

**Implementation Notes**:
- TaskSpec as Pydantic model with JSON Schema export
- Strictness tiers: learning (warn only), permissive (allow with logging), strict (block violations)
- Optional RDF export behind clean interfaces (future)
- Validation runs before execution; receipts include spec hash

## 10. Cryptographic Receipts (Enterprise)

### Decision: SHA-256 Hashes with Optional Signing

**Rationale**: Receipts must prove what happened without being forgeable. Standard cryptographic primitives ensure auditability.

**Alternatives Considered**:
| Alternative | Rejected Because |
|------------|------------------|
| No receipts | Fails Enterprise audit requirements |
| Blockchain | Overkill; slow; expensive |
| Merkle trees | More complex than needed for single receipts |
| Custom hashing | Security risk; use proven algorithms |

**Implementation Notes**:
- Receipt contains: spec_hash, artifact_hashes, verification_evidence, provenance
- Derived from event log (single source of truth)
- Redaction policy: secrets hashed, not stored
- Optional signing with Ed25519 for non-repudiation

## Summary of Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.11+ | Primary implementation |
| Validation | Pydantic | Schemas, config, TaskSpec |
| HTTP Client | httpx | Async model provider calls |
| API Framework | FastAPI | REST API with OpenAPI |
| CLI Framework | Typer | Command-line interface |
| Logging | structlog | Structured JSON logs |
| Tracing | OpenTelemetry | Distributed tracing |
| Metrics | Prometheus | Operational metrics |
| Testing | pytest, hypothesis | Unit, integration, property tests |
| Containers | Docker | Isolated execution |
| Event Store | JSONL (Beads) | Durable event log |
| Memory | Letta + JSONL fallback | Cross-session learning |
| Secrets | Env vars + Vault | Secure credential handling |
