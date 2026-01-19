# Tasks: BLACKICE 3.0 Agentic Software Factory

**Input**: Design documents from `/specs/007-blackice-3/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/openapi.yaml ✓

**Tests**: Integration tests are included as specified in IT-001 through IT-008 in spec.md.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

**⚠️ IMPORTANT**: After generating tasks.md, ALWAYS run `/speckit.analyze` to validate consistency between spec, plan, and tasks before implementation begins.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Per plan.md, this is a single project with 11-layer architecture:
- **Source**: `blackice/` (cli/, api/, orchestrator/, flywheel/, reflexion/, recovery/, persistence/, instrumentation/, colony/, core/, adapters/, primitives/, schemas/, workspace/)
- **Tests**: `tests/` (contract/, integration/, unit/)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per plan.md 11-layer architecture in blackice/
- [ ] T002 Initialize Python 3.11+ project with pyproject.toml and dependencies (Pydantic, httpx, structlog, OpenTelemetry, FastAPI, Typer, pytest)
- [ ] T003 [P] Configure linting (ruff) and formatting (black) in pyproject.toml
- [ ] T004 [P] Configure pytest with pytest-asyncio and hypothesis in pyproject.toml
- [ ] T005 [P] Create blackice/primitives/types.py with base types (RunId, TaskId, EventId, AgentRole enum)
- [ ] T006 [P] Create blackice/primitives/errors.py with exception hierarchy (BlackiceError, ConfigError, ExecutionError, ProviderError)
- [ ] T007 [P] Create blackice/primitives/patterns.py with common patterns (Result, Either, retry decorator)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Schemas (shared across all stories)

- [ ] T008 [P] Create blackice/schemas/run.py with Run Pydantic model (id, vision, status, edition, created_at, completed_at, workspace_path, config)
- [ ] T009 [P] Create blackice/schemas/task.py with Task Pydantic model (id, run_id, name, status, attempt, dependencies, plan_file, state_file, notes_file, idempotency_key)
- [ ] T010 [P] Create blackice/schemas/event.py with Event Pydantic model (id, run_id, type, payload, timestamp, correlation_id, sequence, hash)
- [ ] T011 [P] Create blackice/schemas/agent.py with Agent Pydantic model (id, model_provider, capabilities, system_prompt) and AgentExecution model
- [ ] T012 [P] Create blackice/schemas/taskspec.py with TaskSpec Pydantic model (id, name, version, strictness, input_schema, output_schema, validation_rules) - Enterprise
- [ ] T013 [P] Create blackice/schemas/receipt.py with Receipt Pydantic model (id, run_id, spec_hash, artifact_hashes, verification, provenance, signature) - Enterprise

### Provider Base Interfaces

- [ ] T014 [P] Create blackice/adapters/models/base.py with ModelProvider protocol (generate, chat, embed, health)
- [ ] T015 [P] Create blackice/adapters/execution/base.py with ExecutionProvider protocol (execute, health, attach)
- [ ] T015a [P] Implement ProviderSelector with capability negotiation for execution environments in blackice/adapters/execution/selector.py (FR-018)
- [ ] T015b [P] Add provider interface contract tests in tests/contract/test_execution_providers.py (FR-018)
- [ ] T016 [P] Create blackice/adapters/memory/base.py with MemoryProvider protocol (put, search, load_context, retention_policy)
- [ ] T017 [P] Create blackice/adapters/connectivity/base.py with ConnectivityProvider protocol (attach, rescue, port_forward)
- [ ] T018 [P] Create blackice/adapters/secrets/base.py with SecretsProvider protocol (get, inject_env, redact)

### Connectivity Provider Adapters (FR-020)

- [ ] T018a [P] Implement SSHConnectivityProvider in blackice/adapters/connectivity/ssh.py with attach, rescue, port_forward
- [ ] T018b [P] Implement WireGuardConnectivityProvider in blackice/adapters/connectivity/wireguard.py for secure tunnels

### Core Loop Infrastructure

- [ ] T019 Create blackice/core/retry.py with exponential backoff retry logic
- [ ] T020 [P] Create blackice/core/budget.py with token/cost budget management
- [ ] T021 [P] Create blackice/core/cancellation.py with cancellation token support

### Instrumentation Layer (L5)

- [ ] T022 [P] Create blackice/instrumentation/logger.py with structlog JSON configuration and correlation IDs
- [ ] T023 [P] Create blackice/instrumentation/tracing.py with OpenTelemetry setup and span creation
- [ ] T024 [P] Create blackice/instrumentation/metrics.py with Prometheus metrics (run_count, task_duration, model_calls)

### Workspace Builder

- [ ] T025 Create blackice/workspace/builder.py with run workspace creation (repo/, tests/, docs/, decisions/, logs/)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Vision to Working Software (Priority: P1) 🎯 MVP

**Goal**: Single command converts vision description into working software with tests and documentation

**Independent Test**: Provide a feature description and verify output contains working code, passing tests, and documentation in a git-trackable workspace.

**Acceptance Criteria**: FR-001, FR-002, FR-003, FR-004, FR-005

### Integration Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T026 [P] [US1] Integration test IT-001: End-to-end vision-to-software in tests/integration/test_end_to_end.py
- [ ] T027 [P] [US1] Unit test for Ralph loop (try-fail-reflect-learn-retry) in tests/unit/test_ralph_loop.py
- [ ] T028 [P] [US1] Unit test for safety pipeline in tests/unit/test_safety_pipeline.py
- [ ] T028a [P] [US1] Integration test IT-007: Secret handling in tests/integration/test_secrets.py

### Model Provider Adapters (L2)

- [ ] T029 [P] [US1] Implement ClaudeProvider in blackice/adapters/models/claude.py with httpx async client
- [ ] T030 [P] [US1] Implement OpenAIProvider in blackice/adapters/models/openai.py with httpx async client
- [ ] T031 [P] [US1] Implement OllamaProvider in blackice/adapters/models/ollama.py for local inference

### Execution Provider Adapters (L2)

- [ ] T032 [P] [US1] Implement LocalExecutionProvider in blackice/adapters/execution/local.py with subprocess execution
- [ ] T033 [P] [US1] Implement ContainerExecutionProvider in blackice/adapters/execution/container.py with Docker support
- [ ] T033a [P] [US1] Implement SandboxExecutionProvider in blackice/adapters/execution/sandbox.py for ephemeral isolated execution (FR-018)
- [ ] T034 [US1] Implement safety pipeline (shell unwrap, semantic parse, allowlist, policy check) in blackice/adapters/execution/safety.py

### Secrets Provider Adapters (L2)

- [ ] T035 [P] [US1] Implement EnvSecretsProvider in blackice/adapters/secrets/env.py with environment variable lookup
- [ ] T036 [US1] Implement secret redaction in blackice/adapters/secrets/redaction.py (never in prompts/logs)

### Reflexion Layer (L8)

- [ ] T037 [US1] Implement RalphLoop in blackice/reflexion/ralph_loop.py (try-fail-reflect-learn-retry pattern)
- [ ] T038 [US1] Implement Evaluator in blackice/reflexion/evaluator.py for test verification and repair

### Flywheel Layer (L9)

- [ ] T039 [US1] Implement UnifiedFlywheel in blackice/flywheel/unified.py for end-to-end pipeline execution

### Orchestrator Layer (L10)

- [ ] T040 [US1] Implement RunStateMachine in blackice/orchestrator/state_machine.py (pending→planning→executing→verifying→completed/failed)
- [ ] T041 [US1] Implement phase handlers in blackice/orchestrator/phases.py (plan, implement, test, verify)

### CLI Layer (L11)

- [ ] T042 [US1] Implement CLI entry point in blackice/cli/main.py with Typer
- [ ] T043 [US1] Implement `build` command in blackice/cli/commands/build.py (accepts vision, triggers pipeline)
- [ ] T044 [P] [US1] Implement `status` command in blackice/cli/commands/status.py (list runs, show details)
- [ ] T045 [P] [US1] Implement `doctor` command in blackice/cli/commands/doctor.py (health checks)

**Checkpoint**: User Story 1 complete - single command produces working software with tests

---

## Phase 4: User Story 2 - Crash Recovery and Resume (Priority: P2)

**Goal**: Long-running builds can be resumed after crashes without repeating completed work or duplicating side effects

**Independent Test**: Terminate a build mid-run, invoke resume, verify completed tasks are skipped and in-flight tasks are retried with fresh attempt IDs.

**Acceptance Criteria**: FR-006, FR-007, FR-008, FR-009

### Integration Tests for User Story 2

- [ ] T046 [P] [US2] Integration test IT-002: Crash recovery in tests/integration/test_crash_recovery.py
- [ ] T047 [P] [US2] Unit test for event store in tests/unit/test_event_store.py

### Persistence Layer (L6)

- [ ] T048 [US2] Implement EventStore in blackice/persistence/event_store.py with Beads JSONL format
- [ ] T049 [US2] Implement hash chain for tamper detection in blackice/persistence/event_store.py
- [ ] T049a [US2] Implement event schema versioning and migration strategy in blackice/persistence/event_schema.py (FR-009)
- [ ] T049b [US2] Implement RunProjection and TaskProjection for deterministic state reconstruction from events (FR-009)
- [ ] T049c [US2] Implement `replay` command in blackice/cli/commands/replay.py to prove deterministic reconstruction (FR-009)
- [ ] T050 [P] [US2] Implement ArtifactStore in blackice/persistence/artifact_store.py for workspace artifacts

### Recovery Layer (L7)

- [ ] T051 [US2] Implement Checkpoint in blackice/recovery/checkpoint.py for snapshot creation
- [ ] T052 [US2] Implement Resume in blackice/recovery/resume.py (skip completed, retry in-flight with new attempt IDs)
- [ ] T053 [US2] Implement idempotency key handling in blackice/recovery/idempotency.py for external effects
- [ ] T053a [US2] Wire idempotency keys into ExecutionProvider (generation, persistence, check, enforcement per effect type) (FR-008)
- [ ] T054 [P] [US2] Implement dead letter handling in blackice/recovery/dead_letter.py

### CLI Extension

- [ ] T055 [US2] Implement `resume` command in blackice/cli/commands/resume.py (resumes crashed run)
- [ ] T056 [US2] Implement `watch` command in blackice/cli/commands/watch.py (stream run progress)

**Checkpoint**: User Story 2 complete - crashed runs resume without data loss

---

## Phase 5: User Story 4 - Multi-Agent Consensus (Priority: P2)

**Goal**: Specialist agents coordinate through voting to reduce single-model brittleness

**Independent Test**: Initiate a build requiring security review, verify specialist agents are invoked, and consensus voting produces final decisions.

**Acceptance Criteria**: FR-014, FR-015, FR-016, FR-017

### Integration Tests for User Story 4

- [ ] T057 [P] [US4] Integration test IT-003: Multi-agent consensus in tests/integration/test_multi_agent.py
- [ ] T058 [P] [US4] Integration test IT-008: Reservation conflicts in tests/integration/test_reservations.py
- [ ] T059 [P] [US4] Unit test for consensus voting in tests/unit/test_consensus.py

### Colony Layer (L4) - Agent Implementations

- [ ] T060 [P] [US4] Implement PlannerAgent in blackice/colony/agents/planner.py (decompose, prioritize, estimate)
- [ ] T061 [P] [US4] Implement ImplementerAgent in blackice/colony/agents/implementer.py (code, document, refactor)
- [ ] T062 [P] [US4] Implement ReviewerAgent in blackice/colony/agents/reviewer.py (analyze, critique, suggest)
- [ ] T063 [P] [US4] Implement TesterAgent in blackice/colony/agents/tester.py (test, verify, validate)
- [ ] T064 [P] [US4] Implement SecurityAgent in blackice/colony/agents/security.py (audit, scan, harden)

### Colony Layer (L4) - Coordination

- [ ] T065 [US4] Implement Supervisor in blackice/colony/supervisor.py (agent lifecycle management)
- [ ] T066 [US4] Implement Consensus in blackice/colony/consensus.py (voting policies: majority, supermajority, unanimous, quorum, weighted)
- [ ] T067 [US4] Implement Messaging in blackice/colony/messaging.py (threaded internal messaging, durable handoffs)
- [ ] T067a [US4] Implement message durability (messages as events, replayable threads, correlation IDs) in blackice/colony/messaging.py (FR-017)
- [ ] T068 [US4] Implement Registry in blackice/colony/registry.py (agent registration and discovery)
- [ ] T069 [US4] Implement Reservation system in blackice/colony/reservation.py (file/directory leases with TTL)

**Checkpoint**: User Story 4 complete - multi-agent consensus reduces single-model brittleness

---

## Phase 6: User Story 3 - Spec-Validated Execution (Priority: P3) - Enterprise

**Goal**: Pre-execution validation ensures builds conform to organizational policies with cryptographic receipts

**Independent Test**: Define a TaskSpec with strict validation, run a build, verify the receipt contains spec hash, artifact hashes, and verification evidence.

**Acceptance Criteria**: FR-010, FR-011, FR-012, FR-013

### Integration Tests for User Story 3

- [ ] T070 [P] [US3] Integration test IT-005: TaskSpec validation in tests/integration/test_taskspec_validation.py
- [ ] T071 [P] [US3] Integration test IT-006: Receipt generation in tests/integration/test_receipt_generation.py

### TaskSpec Validation

- [ ] T072 [US3] Implement TaskSpecValidator in blackice/schemas/taskspec.py with strictness tiers (learning, permissive, strict)
- [ ] T073 [US3] Implement policy violation messaging in blackice/schemas/taskspec.py

### Receipt Generation

- [ ] T074 [US3] Implement ReceiptGenerator in blackice/schemas/receipt.py deriving from event log
- [ ] T074a [US3] Implement EvidenceModel (test reports, scan outputs, exit codes, stdout/stderr digests) in blackice/schemas/evidence.py (FR-012)
- [ ] T074b [US3] Implement evidence artifact persistence and event emission in blackice/persistence/artifact_store.py (FR-012)
- [ ] T074c [US3] Add contract test validating evidence presence in receipts for IT-006 in tests/contract/test_receipt_evidence.py (FR-012)
- [ ] T075 [US3] Implement spec hash generation (SHA-256) in blackice/schemas/receipt.py
- [ ] T076 [US3] Implement artifact hash generation in blackice/schemas/receipt.py
- [ ] T077 [US3] Implement redaction policy (hash-only, scrubbed fields) in blackice/schemas/receipt.py
- [ ] T078 [P] [US3] Implement optional Ed25519 signing for non-repudiation in blackice/schemas/receipt.py

### CLI Extension

- [ ] T079 [US3] Implement `receipt` command in blackice/cli/commands/receipt.py (get run receipt)

**Checkpoint**: User Story 3 complete - Enterprise governance with cryptographic receipts

---

## Phase 7: User Story 5 - Persistent Cross-Session Learning (Priority: P3)

**Goal**: System remembers successful patterns, safe commands, and good outputs to accelerate future runs

**Independent Test**: Run a build, store patterns to memory, run a similar build and verify memory retrieval accelerates planning.

**Acceptance Criteria**: FR-021, FR-022, FR-023

### Integration Tests for User Story 5

- [ ] T080 [P] [US5] Integration test IT-004: Memory integration in tests/integration/test_memory.py

### Memory Provider Adapters (L2)

- [ ] T082 [P] [US5] Implement LettaMemoryProvider in blackice/adapters/memory/letta.py with semantic search
- [ ] T083 [P] [US5] Implement LocalJsonlMemoryProvider in blackice/adapters/memory/local_jsonl.py (fallback, keyword search only)

### Persistence Layer (L6) - Memory

- [ ] T084 [US5] Implement MemoryStore in blackice/persistence/memory_provider.py with retention policies (redact, hash-only, TTL)
- [ ] T085 [US5] Implement memory query during planning in blackice/flywheel/unified.py
- [ ] T086 [US5] Implement verified pattern storage (write only after verification) in blackice/persistence/memory_provider.py

**Checkpoint**: User Story 5 complete - cross-session learning accelerates future runs

---

## Phase 8: API Layer (Required - FR-024/025/026)

**Goal**: REST API for programmatic access to BLACKICE functionality

**⚠️ REQUIRED**: FR-024 health endpoints (/health, /ready, /live) are MUST requirements, not optional

**Acceptance Criteria**: FR-024, FR-025, FR-026

### API Implementation

- [ ] T087 [P] Create blackice/api/app.py with FastAPI application and OpenAPI generation
- [ ] T088 [P] Implement health endpoints (/health, /ready, /live) in blackice/api/routes/health.py
- [ ] T089 [P] Implement runs endpoints (POST /runs, GET /runs, GET /runs/{id}) in blackice/api/routes/runs.py
- [ ] T090 [P] Implement tasks endpoints (GET /runs/{id}/tasks, GET /runs/{id}/tasks/{taskId}) in blackice/api/routes/tasks.py
- [ ] T091 [P] Implement events endpoints (GET /runs/{id}/events) in blackice/api/routes/events.py
- [ ] T092 [P] Implement memory endpoints (POST /memory/search, POST /memory/entries) in blackice/api/routes/memory.py
- [ ] T093 [P] Implement receipt endpoint (GET /runs/{id}/receipt) in blackice/api/routes/receipt.py
- [ ] T094 [P] Implement taskspecs endpoints (GET /taskspecs, POST /taskspecs) in blackice/api/routes/taskspecs.py
- [ ] T095 Create contract tests for OpenAPI spec in tests/contract/test_api_contracts.py

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T096 [P] Create README.md with installation and usage instructions
- [ ] T097 [P] Create CONTRIBUTING.md with development guidelines
- [ ] T098 [P] Create CHANGELOG.md with initial release notes
- [ ] T099 [P] Add docstrings to all public modules and functions
- [ ] T100 [P] Create Dockerfile for containerized deployment
- [ ] T101 [P] Create docker-compose.yml for local development stack
- [ ] T102 Implement graceful degradation tests for all fallback strategies
- [ ] T102a [P] Integration test for provider failover chain (Claude→OpenAI→Ollama→fail with checkpoint)
- [ ] T103 Performance testing: verify <100ms API response, <5s run startup
- [ ] T104 Security audit: verify secrets never in logs, prompts, or unredacted receipts
- [ ] T105 Run quickstart.md validation with sample builds

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **API Layer (Phase 8)**: Depends on User Story 1 completion
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Extends US1 with durability
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Independent of US2
- **User Story 3 (P3)**: Depends on US2 (event sourcing) for receipts derivation
- **User Story 5 (P3)**: Can start after Foundational (Phase 2) - Independent of other stories

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Adapters/providers before higher layers
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes:
  - US1, US4, US5 can start in parallel
  - US2 can start after or with US1
  - US3 must wait for US2 (event sourcing required for receipts)
- All agent implementations (T060-T064) can run in parallel
- All API route implementations (T088-T094) can run in parallel

---

## Parallel Example: User Story 4 (Multi-Agent)

```bash
# Launch all agent implementations together:
Task: "Implement PlannerAgent in blackice/colony/agents/planner.py"
Task: "Implement ImplementerAgent in blackice/colony/agents/implementer.py"
Task: "Implement ReviewerAgent in blackice/colony/agents/reviewer.py"
Task: "Implement TesterAgent in blackice/colony/agents/tester.py"
Task: "Implement SecurityAgent in blackice/colony/agents/security.py"

# Then sequentially:
Task: "Implement Supervisor in blackice/colony/supervisor.py"
Task: "Implement Consensus in blackice/colony/consensus.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test vision-to-software pipeline independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Demo (MVP!)
3. Add User Story 2 → Crash recovery works → Demo
4. Add User Story 4 → Multi-agent consensus → Demo
5. Add User Story 3 → Enterprise receipts → Demo (Enterprise release)
6. Add User Story 5 → Cross-session learning → Demo (Complete)

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (MVP)
   - Developer B: User Story 4 (Multi-Agent)
   - Developer C: User Story 5 (Memory)
3. After US1 complete:
   - Developer A: User Story 2 (Event Sourcing)
4. After US2 complete:
   - Developer A: User Story 3 (Receipts)
5. Stories complete and integrate independently

---

## Notes

- **[P] tasks** = different files, no dependencies
- **[Story] label** maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (TDD)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

---

## Task Completion Criteria

**A task is NOT complete until ALL of the following are true:**

| Criterion | Requirement |
|-----------|-------------|
| Implementation | Code is written and compiles/runs without errors |
| Unit tests | 100% pass rate (all relevant unit tests pass) |
| Integration tests | 100% pass rate (if task affects API/data flow) |
| Smoke tests | 100% pass rate (if task affects critical paths) |
| No regressions | All previously passing tests still pass |
| Marked complete | Task checkbox changed from `[ ]` to `[x]` |

**Failure Protocol:**
1. If any test fails → FIX before marking complete
2. If fix is non-trivial → Create blocking issue, do NOT proceed
3. NEVER skip tests or mark task complete with failures

## User Story Completion Criteria

**A user story is NOT complete until:**

| Criterion | Requirement |
|-----------|-------------|
| All tasks | Every task in the story is `[x]` completed |
| Unit tests | 100% pass for all story code |
| Integration tests | 100% pass for story's user journeys |
| Independent test | Story verified working in isolation |
| Checkpoint passed | Story validated at its checkpoint |

## Feature Completion Criteria

**A feature is NOT shippable until:**

| Criterion | Requirement |
|-----------|-------------|
| All stories | Every user story complete (per above) |
| Smoke tests | 100% pass (all critical paths work) |
| No regressions | All existing tests still pass |
| Cross-story | Stories work together correctly |

---

## Summary

| Phase | Tasks | Parallelizable | Key Deliverables |
|-------|-------|----------------|------------------|
| Phase 1: Setup | 7 | 5 | Project structure, primitives |
| Phase 2: Foundational | 22 | 20 | Schemas, providers, selector, connectivity, instrumentation |
| Phase 3: US1 Vision-to-Software | 22 | 13 | MVP - single command build, sandbox, secrets |
| Phase 4: US2 Crash Recovery | 15 | 5 | Event sourcing, projections, replay, resume, idempotency |
| Phase 5: US4 Multi-Agent | 14 | 9 | Specialist agents, consensus, durable messaging |
| Phase 6: US3 Spec-Validated | 13 | 5 | TaskSpec, evidence model, receipts (Enterprise) |
| Phase 7: US5 Learning | 6 | 3 | Memory, cross-session patterns |
| Phase 8: API (Required) | 9 | 8 | REST API endpoints, health checks |
| Phase 9: Polish | 11 | 8 | Docs, deployment, failover tests |
| **Total** | **119** | **76** | Full BLACKICE 3.0 |

**MVP Scope**: Phases 1-3 (51 tasks) → Functional vision-to-software pipeline
**Core Scope**: Phases 1-5 (80 tasks) → Add crash recovery + multi-agent
**Enterprise Scope**: Phases 1-7 (99 tasks) → Add governance + learning
**Complete Scope**: All phases (119 tasks) → Full system with API
