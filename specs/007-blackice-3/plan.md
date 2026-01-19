# Implementation Plan: BLACKICE 3.0 Agentic Software Factory

**Branch**: `007-blackice-3` | **Date**: 2026-01-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/007-blackice-3/spec.md`

## Summary

BLACKICE 3.0 is an agentic software factory that converts a single human vision into working software with tests, documentation, and a reproducible artifact trail. The system implements an 11-layer architecture with the Ralph pattern execution loop (try-fail-reflect-learn-retry), multi-agent consensus with specialist roles, event sourcing via Beads for crash recovery, and provider-based abstractions for execution, memory, connectivity, and secrets. Three editions (Lite, Core, Enterprise) are additive layers providing progressively stronger guarantees.

## Architectural Vision *(mandatory)*

1. **L1-L3 Foundation Layer**: Primitives, adapters, and core loop provide the base execution substrate with retry logic, cancellation, and budget management - keeping model providers and execution backends swappable behind stable contracts.

2. **L4-L5 Service Colony + Instrumentation**: Multi-agent coordination through supervisor, consensus, messaging, and registry services, with comprehensive observability (tracing, metrics, structured logs, safety, cost tracking).

3. **L6-L7 Persistence + Recovery**: Event sourcing via Beads (immutable append-only log) enables durable state, crash resume, and deterministic reconstruction. Dead letter handling and checkpoints ensure no work is lost.

4. **L8-L9 Reflexion + Flywheel**: Evaluation and learning loops implement the Ralph pattern - quality improves through iteration rather than requiring perfection on first attempt. The unified flywheel drives end-to-end pipeline execution.

5. **L10-L11 Orchestrator + CLI/API**: Single entry point triggers full build workflow. Run state machine coordinates phases while keeping all complexity internal - the user interface remains simple.

## Technical Context

**Language/Version**: Python 3.11+ (primary), TypeScript (optional CLI wrapper)
**Primary Dependencies**: Pydantic (validation/schemas), httpx (async HTTP), structlog (logging), OpenTelemetry (tracing)
**Storage**: JSONL files (Beads event log), SQLite (local metadata), S3-compatible (artifacts)
**Testing**: pytest, pytest-asyncio, hypothesis (property-based), testcontainers
**Target Platform**: Linux server (primary), macOS (development), Docker containers
**Project Type**: Single project with plugin architecture for providers
**Performance Goals**: 1000 concurrent runs, <100ms API response for status queries, <5s run startup
**Constraints**: <500MB memory per run, offline-capable (degraded mode), secrets never in logs
**Scale/Scope**: 10+ concurrent runs, 100+ tasks per run, cross-session memory spanning months

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Implementation Notes |
|-----------|--------|---------------------|
| **I. Beads Integration** | PASS | FR-006 requires immutable event log; event sourcing is core to architecture |
| **II. Test-First Development** | PASS | IT-001 through IT-008 define integration tests; TDD enforced in tasks |
| **II.A Test Pass Gate** | PASS | SC-001 requires working software with passing tests |
| **II.B SOLID Architecture** | PASS | Provider pattern (ExecutionProvider, MemoryProvider, etc.) enables DI/ISP |
| **III. Security-First Design** | PASS | FR-019 mandates secrets never in prompts/logs; FR-003 safety pipeline |
| **IV. Compliance by Design** | PASS | FR-011/FR-012 cryptographic receipts with verification evidence |
| **V. API Versioning** | PASS | OpenAPI contracts in /contracts/ directory |
| **VI. Observability** | PASS | FR-024/FR-025/FR-026 health checks, structured logs, distributed tracing |
| **VII. Graceful Degradation** | PASS | Fallback strategies defined below |
| **VIII. Code Quality** | PASS | Type hints, Pydantic validation, linting enforced |

### Graceful Degradation Strategies

| Dependency | Failure Mode | Fallback Strategy |
|------------|--------------|-------------------|
| Model Provider (Claude/OpenAI) | API timeout/error | Failover chain: Claude - OpenAI - Ollama - fail with checkpoint |
| Memory Backend (Letta) | Connection failure | Local JSONL cache (degraded: no semantic search) |
| Container Runtime | Docker unavailable | Local execution with filesystem isolation |
| Network Connectivity | Egress blocked | Offline mode with cached dependencies, queue outbound ops |
| Secrets Provider | Vault unavailable | Environment variables fallback, warn on startup |

## Project Structure

### Documentation (this feature)

```text
specs/007-blackice-3/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output - OpenAPI specs
│   └── openapi.yaml
├── checklists/          # Quality gates
│   └── requirements.md
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
blackice/
├── __init__.py
├── cli/                      # L11: CLI entry point
│   ├── __init__.py
│   ├── main.py               # Click/Typer CLI
│   └── commands/             # build, resume, status, doctor
├── api/                      # L11: API entry point (optional)
│   ├── __init__.py
│   ├── app.py                # FastAPI app
│   └── routes/               # REST endpoints
├── orchestrator/             # L10: Run state machine
│   ├── __init__.py
│   ├── state_machine.py
│   └── phases.py
├── flywheel/                 # L9: End-to-end pipeline driver
│   ├── __init__.py
│   └── unified.py
├── reflexion/                # L8: Evaluation + learning
│   ├── __init__.py
│   ├── evaluator.py
│   └── ralph_loop.py
├── recovery/                 # L7: Resume, checkpoints, dead letters
│   ├── __init__.py
│   ├── checkpoint.py
│   └── resume.py
├── persistence/              # L6: Event store + memory + artifacts
│   ├── __init__.py
│   ├── event_store.py        # Beads integration
│   ├── memory_provider.py    # MemoryProvider interface
│   └── artifact_store.py
├── instrumentation/          # L5: Observability
│   ├── __init__.py
│   ├── tracing.py
│   ├── metrics.py
│   └── logger.py
├── colony/                   # L4: Service colony (agents)
│   ├── __init__.py
│   ├── supervisor.py
│   ├── consensus.py
│   ├── messaging.py
│   ├── registry.py
│   └── agents/               # Specialist roles
│       ├── planner.py
│       ├── implementer.py
│       ├── reviewer.py
│       ├── tester.py
│       └── security.py
├── core/                     # L3: Core loop
│   ├── __init__.py
│   ├── retry.py
│   ├── budget.py
│   └── cancellation.py
├── adapters/                 # L2: External integrations
│   ├── __init__.py
│   ├── models/               # Model providers
│   │   ├── base.py
│   │   ├── claude.py
│   │   ├── openai.py
│   │   └── ollama.py
│   ├── execution/            # Execution providers
│   │   ├── base.py
│   │   ├── local.py
│   │   ├── container.py
│   │   └── sandbox.py
│   ├── memory/               # Memory providers
│   │   ├── base.py
│   │   ├── letta.py
│   │   └── local_jsonl.py
│   ├── connectivity/         # Connectivity providers
│   │   ├── base.py
│   │   ├── ssh.py
│   │   └── wireguard.py
│   └── secrets/              # Secrets providers
│       ├── base.py
│       ├── env.py
│       └── vault.py
├── primitives/               # L1: Utils, types, patterns
│   ├── __init__.py
│   ├── types.py
│   ├── errors.py
│   └── patterns.py
├── schemas/                  # Pydantic models (shared)
│   ├── __init__.py
│   ├── run.py
│   ├── task.py
│   ├── event.py
│   ├── agent.py
│   ├── taskspec.py           # Enterprise
│   └── receipt.py            # Enterprise
└── workspace/                # Run workspace management
    ├── __init__.py
    └── builder.py

tests/
├── conftest.py
├── contract/                 # Contract tests
│   └── test_api_contracts.py
├── integration/              # Integration tests
│   ├── test_end_to_end.py
│   ├── test_crash_recovery.py
│   ├── test_multi_agent.py
│   └── test_memory.py
└── unit/                     # Unit tests
    ├── test_ralph_loop.py
    ├── test_event_store.py
    ├── test_consensus.py
    └── test_safety_pipeline.py
```

**Structure Decision**: Single project with plugin architecture. The 11-layer stack maps directly to directory structure. Provider pattern enables swappable backends for models, execution, memory, connectivity, and secrets.

## Complexity Tracking

No constitution violations requiring justification. The architecture follows SOLID principles with clear separation of concerns across layers.

## Phase Outputs

### Phase 0: Research
- [research.md](./research.md) - Technology decisions and alternatives

### Phase 1: Design
- [data-model.md](./data-model.md) - Entity schemas and relationships
- [contracts/openapi.yaml](./contracts/openapi.yaml) - API specification
- [quickstart.md](./quickstart.md) - Getting started guide

### Phase 2: Tasks
- [tasks.md](./tasks.md) - Implementation tasks (created by /speckit.tasks)
