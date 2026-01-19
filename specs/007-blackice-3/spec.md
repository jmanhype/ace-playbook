# Feature Specification: BLACKICE 3.0 Agentic Software Factory

**Feature Branch**: `007-blackice-3`
**Created**: 2026-01-18
**Status**: Draft
**Input**: BLACKICE 3.0 White Paper - Agentic software factory architecture

## Problem Statement *(mandatory)*

Most software development is constrained by coordination overhead: translating intent into tickets, tickets into code, code into reviews, reviews into deployments, and deployments into operations. Current AI coding tools help at the micro-level (snippets and completions), but fail at the macro-level: multi-step delivery, safe execution, recovery from failures, and consistent quality.

**Current Pain Points**:
- Fragmented tooling requires manual orchestration between planning, implementation, testing, and deployment
- Long-running builds crash without recovery, requiring complete restarts
- No reproducible artifact trail to explain what happened during automated builds
- Model outputs are unpredictable and lack verification mechanisms
- Unsafe command execution can damage environments or leak secrets
- Cross-session learning is lost, forcing repeated discovery of patterns
- Multi-agent coordination lacks standardized handoff and conflict resolution

## Business Value *(mandatory)*

- **Reduced coordination overhead**: Single command triggers full vision-to-software pipeline, eliminating manual task decomposition
- **Crash-resilient operations**: Durable run logs enable resume after failures without repeating unsafe actions
- **Auditable software production**: Complete artifact trail and cryptographic receipts for compliance and debugging
- **Consistent quality**: Evaluation and repair loops improve output through iteration rather than requiring perfection
- **Persistent organizational learning**: Memory system captures successful patterns across runs and repositories
- **Enterprise trust**: Spec-first validation and verifiable receipts meet policy and audit requirements

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Vision to Working Software (Priority: P1)

A software architect has a clear vision for a new feature but limited time for detailed task decomposition. They want to describe their intent in natural language and receive a working implementation with tests and documentation.

**Why this priority**: This is the core value proposition - converting human vision into working software with minimal friction.

**Independent Test**: Can be tested by providing a feature description and verifying the output contains working code, passing tests, and documentation in a git-trackable workspace.

**Acceptance Scenarios**:

1. **Given** a feature description in natural language, **When** the user invokes the build command, **Then** the system produces a working codebase with tests and documentation in a run workspace
2. **Given** a build in progress, **When** tests fail, **Then** the system automatically attempts repair and re-verification before declaring failure
3. **Given** a completed build, **When** the user inspects the run folder, **Then** they can trace every decision and artifact back to its source

---

### User Story 2 - Crash Recovery and Resume (Priority: P2)

A developer initiates a long-running build that crashes due to terminal disconnect, model failure, or system error. They want to resume from where the build left off without repeating completed work or re-executing unsafe side effects.

**Why this priority**: Long-running builds are common and crashes are inevitable; recovery without data loss is essential for trust.

**Independent Test**: Can be tested by intentionally terminating a build mid-run, then invoking resume and verifying completed tasks are skipped while in-flight tasks are retried.

**Acceptance Scenarios**:

1. **Given** a crashed run with completed and in-flight tasks, **When** the user invokes resume, **Then** completed tasks are not re-run and in-flight tasks resume with fresh attempt IDs
2. **Given** a run that executed external side effects, **When** resuming, **Then** the system uses idempotency keys to avoid duplicate effects
3. **Given** a resumed run, **When** it completes, **Then** the final artifacts are identical to what would have been produced without the crash

---

### User Story 3 - Spec-Validated Execution (Priority: P3)

An enterprise engineering team requires pre-execution validation to ensure builds conform to organizational policies. They want to define TaskSpecs with strictness tiers and receive cryptographic receipts proving what was built.

**Why this priority**: Enterprise adoption requires governance, policy enforcement, and audit trails.

**Independent Test**: Can be tested by defining a TaskSpec with strict validation, running a build, and verifying the receipt contains spec hash, artifact hashes, and verification evidence.

**Acceptance Scenarios**:

1. **Given** a TaskSpec with "strict" validation tier, **When** the build deviates from spec, **Then** execution stops with clear policy violation messages
2. **Given** a completed build, **When** the user requests the receipt, **Then** they receive cryptographic proof including spec hash, artifact hashes, and verification evidence
3. **Given** a receipt with redaction policy enabled, **When** inspecting the receipt, **Then** secrets and PII are hashed rather than stored in plaintext

---

### User Story 4 - Multi-Agent Consensus (Priority: P2)

A complex feature requires multiple perspectives: planning, implementation, review, testing, and security analysis. The system should coordinate specialist agents to reduce single-model brittleness.

**Why this priority**: Multi-agent consensus improves reliability and catches issues that single-model approaches miss.

**Independent Test**: Can be tested by initiating a build requiring security review, verifying specialist agents (planner, implementer, reviewer, tester, security) are invoked, and checking consensus voting produces the final decision.

**Acceptance Scenarios**:

1. **Given** a task requiring multiple perspectives, **When** the build runs, **Then** specialist agents generate parallel proposals that are voted on according to configured policy
2. **Given** low confidence in a consensus decision, **When** the threshold is not met, **Then** the system escalates to cross-model attack patterns for additional validation
3. **Given** agent coordination in progress, **When** two agents attempt to modify the same file, **Then** the reservation system prevents conflicts

---

### User Story 5 - Persistent Cross-Session Learning (Priority: P3)

An organization uses BLACKICE repeatedly on the same codebase. They want the system to remember successful patterns, safe commands, and "good" outputs to accelerate future runs.

**Why this priority**: Learning from history dramatically improves efficiency and consistency over time.

**Independent Test**: Can be tested by running a build, storing patterns to memory, then running a similar build and verifying memory retrieval accelerates planning.

**Acceptance Scenarios**:

1. **Given** a completed verified build, **When** patterns are stored to memory, **Then** only validated patterns and outcomes are persisted
2. **Given** a new build similar to prior runs, **When** planning begins, **Then** the system retrieves relevant prior runs and repo-specific SOPs
3. **Given** memory with sensitive content, **When** retention policy is set to "hash-only", **Then** secrets and PII are redacted before storage

---

### Edge Cases

- What happens when all model providers fail? System should fail gracefully with clear messaging and checkpoint state for manual recovery.
- How does the system handle network isolation during sandbox execution? Commands should fail safely, and the run should be resumable when connectivity is restored.
- What happens when memory storage reaches capacity? Oldest non-critical entries should be pruned according to retention policy.
- How does the system handle conflicting agent recommendations? Voting policy (majority/supermajority/unanimous) determines resolution; ties escalate to human operator.
- What happens when a TaskSpec version is updated mid-run? Current run continues with original spec; new runs use updated spec.

## Requirements *(mandatory)*

### Functional Requirements

**Core Pipeline (Lite)**
- **FR-001**: System MUST accept a vision description and produce a working codebase with tests and documentation
- **FR-002**: System MUST emit a git-trackable run workspace containing all artifacts, decisions, and logs
- **FR-003**: System MUST execute commands through a safety pipeline (shell unwrap, semantic parse, stack allowlist, policy check)
- **FR-004**: System MUST support iterative repair through the Ralph loop (try, fail, reflect, learn, retry)
- **FR-005**: System MUST provide a single CLI/API entry point that triggers the full build workflow

**Durability (Core)**
- **FR-006**: System MUST persist an immutable event log (Beads) for every significant action
- **FR-007**: System MUST support run resume from checkpoint without re-running completed tasks
- **FR-008**: System MUST require idempotency keys for external side effects to prevent duplicate execution on resume
- **FR-009**: System MUST support deterministic reconstruction of run state from event log replay

**Governance (Enterprise)**
- **FR-010**: System MUST validate TaskSpecs with configurable strictness tiers (learning, permissive, strict)
- **FR-011**: System MUST generate cryptographic receipts derived from the durable event log
- **FR-012**: System MUST include verification evidence (tests, scans, policy checks) in receipts
- **FR-013**: System MUST support redaction policies (hash-only, scrubbed fields) for receipts

**Multi-Agent Coordination**
- **FR-014**: System MUST support specialist agent roles (planner, implementer, reviewer, tester, security)
- **FR-015**: System MUST implement voting policies for agent consensus (majority, supermajority, unanimous, quorum, weighted)
- **FR-016**: System MUST provide file/directory reservations to prevent concurrent conflicting edits
- **FR-017**: System MUST support threaded internal messaging for durable agent handoffs

**Execution Substrate**
- **FR-018**: System MUST support interchangeable execution environments (local, container, ephemeral sandbox)
- **FR-019**: System MUST inject secrets into tools without exposing them in prompts or logs
- **FR-020**: System MUST support connectivity operations (attach, rescue, port-forward) for remote sandboxes

**Memory and Learning**
- **FR-021**: System MUST query memory during planning to retrieve similar prior runs and SOPs
- **FR-022**: System MUST write to memory only after verification to store validated patterns
- **FR-023**: System MUST enforce retention policies (redact, hash-only, TTL) on memory storage

**Observability**
- **FR-024**: System MUST implement health checks (/health, /ready, /live)
- **FR-025**: System MUST emit structured logs with correlation IDs for all operations
- **FR-026**: System MUST support distributed tracing for end-to-end request visibility

### Key Entities

- **Run**: A complete execution from vision to artifacts, containing workspace, event log, and receipts
- **Task**: A unit of work within a run with plan, state, and notes files
- **Event**: An immutable record of a significant action in the event log (Beads)
- **Agent**: A specialist role (planner, implementer, reviewer, tester, security) with defined capabilities
- **TaskSpec**: A schema defining expected inputs, outputs, and validation rules (Enterprise)
- **Receipt**: Cryptographic proof of run completion with hashes and verification evidence
- **Reservation**: A lease on a file/directory to prevent concurrent conflicting edits
- **Memory Entry**: A persisted pattern, outcome, or SOP retrieved during planning

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can convert a vision description into working software with a single command invocation
- **SC-002**: 95% of crashed runs can be resumed successfully without data loss or duplicate side effects
- **SC-003**: Run workspaces are fully inspectable - every artifact traces back to source decision
- **SC-004**: Multi-agent consensus reduces single-model error rate by at least 30% compared to single-agent execution
- **SC-005**: Memory-assisted runs complete 40% faster than cold runs on similar codebases
- **SC-006**: Enterprise receipts pass third-party audit validation for provenance and verification
- **SC-007**: System supports concurrent execution of 10+ runs without resource contention
- **SC-008**: All unsafe commands are blocked or escalated according to policy with zero bypasses

## Integration Tests *(mandatory)*

- **IT-001**: End-to-end: Vision description -> Planning -> Implementation -> Testing -> Verification -> Run workspace with passing tests
- **IT-002**: Crash recovery: Initiate run -> Force terminate mid-execution -> Resume -> Verify identical final artifacts
- **IT-003**: Multi-agent consensus: Submit task -> Verify specialist proposals -> Verify voting produces valid decision
- **IT-004**: Memory integration: Complete run -> Store patterns -> Start similar run -> Verify memory retrieval accelerates planning
- **IT-005**: TaskSpec validation: Define strict spec -> Submit non-compliant input -> Verify rejection with policy violation message
- **IT-006**: Receipt generation: Complete Enterprise run -> Generate receipt -> Verify spec hash, artifact hashes, and verification evidence present
- **IT-007**: Secret handling: Configure secrets provider -> Execute command requiring secrets -> Verify secrets not in logs or prompts
- **IT-008**: Reservation conflicts: Start two agents -> Both attempt same file -> Verify reservation prevents conflict

## Acceptance Criteria *(mandatory)*

1. Single CLI/API command triggers complete vision-to-software pipeline without intermediate user interaction
2. Run workspace contains human-readable, git-trackable artifacts explaining every decision and outcome
3. Crashed runs resume from checkpoint with completed tasks skipped and in-flight tasks retried
4. Multi-agent specialist roles coordinate through voting without file conflicts
5. Memory system accelerates similar runs by retrieving validated patterns from prior executions
6. TaskSpec validation blocks non-compliant builds at Enterprise tier with clear policy violation messages
7. Receipts derive from event log (single source of truth) and include verification evidence
8. Secrets never appear in prompts, logs, or unredacted receipts
9. Execution environments (local, container, remote) are interchangeable behind stable provider interfaces
10. System degrades gracefully when dependencies fail (model providers, memory backend, connectivity)

## Non-Functional Requirements

### Security Requirements (NFR-SEC)

**Authentication & Authorization**
- **NFR-SEC-001**: API endpoints MUST require Bearer token authentication (JWT or API key)
- **NFR-SEC-002**: Health endpoints (/health, /ready, /live) MAY be unauthenticated for load balancer probes
- **NFR-SEC-003**: Edition-based authorization MUST restrict Enterprise features (TaskSpecs, receipts) to Enterprise tokens
- **NFR-SEC-004**: Role-based access MUST support: operator (full access), viewer (read-only), auditor (receipts only)

**Safety Pipeline (FR-003 Clarification)**
- **NFR-SEC-005**: Command allowlist MUST include: git, npm, pip, cargo, make, pytest, standard POSIX utilities
- **NFR-SEC-006**: Command blocklist MUST include: rm -rf /, sudo, chmod 777, curl | bash, eval with untrusted input
- **NFR-SEC-007**: Network commands (curl, wget, ssh) MUST be validated against egress policy before execution
- **NFR-SEC-008**: Policy escalation MUST: (1) block command, (2) emit warning event, (3) notify operator if real-time channel available, (4) continue run in degraded mode

**Threat Model**
- **NFR-SEC-009**: Threat model covers: prompt injection, command injection, secret exfiltration, model jailbreaking, supply chain attacks
- **NFR-SEC-010**: All model outputs MUST be treated as untrusted input - validated before execution
- **NFR-SEC-011**: Sandbox isolation MUST provide: separate filesystem namespace, no host network access, capped CPU/memory, no privileged operations

**Input Validation**
- **NFR-SEC-012**: Vision description MUST be validated: max 10,000 characters, UTF-8 encoded, no null bytes, sanitized for logging
- **NFR-SEC-013**: All user inputs MUST be validated against injection patterns before processing

**Cryptography**
- **NFR-SEC-014**: Receipt signatures MUST use Ed25519 with 256-bit keys
- **NFR-SEC-015**: Content hashes MUST use SHA-256
- **NFR-SEC-016**: Secret rotation MUST be supported via provider refresh without run interruption

**Network Security**
- **NFR-SEC-017**: Sandbox egress policy: deny-by-default, allowlist for package registries (npm, pypi, crates.io)
- **NFR-SEC-018**: Inter-agent communication MUST use authenticated channels with correlation ID validation

**Code Security**
- **NFR-SEC-019**: Generated code MUST be scanned for: hardcoded secrets, known vulnerable patterns, SQL injection, XSS vectors

### Privacy Requirements (NFR-PRIV)

**PII Handling**
- **NFR-PRIV-001**: PII detection MUST identify: email addresses, phone numbers, SSN patterns, credit card numbers, API keys, passwords
- **NFR-PRIV-002**: Redact policy MUST replace PII with deterministic hashes (SHA-256 truncated to 8 chars)
- **NFR-PRIV-003**: Hash-only mode MUST hash: secrets, credentials, personally identifiable text; preserve: code structure, error types, timestamps

**Data Retention**
- **NFR-PRIV-004**: Memory entry TTL defaults: patterns (365 days), outcomes (90 days), SOPs (indefinite unless manual delete)
- **NFR-PRIV-005**: Event logs MUST be retained for minimum 30 days, maximum configurable up to 7 years for compliance

**User Rights**
- **NFR-PRIV-006**: User consent MUST be obtained before storing any cross-session patterns (opt-in, not opt-out)
- **NFR-PRIV-007**: Data export MUST provide full memory dump in portable JSON format within 48 hours of request
- **NFR-PRIV-008**: Data deletion MUST remove all user-associated memory entries within 72 hours of request

**Secrets Definition**
- **NFR-PRIV-009**: "Secrets" scope includes: API keys, OAuth tokens, passwords, private keys, connection strings, bearer tokens

**Consistency**
- **NFR-PRIV-010**: Redaction MUST be consistent: same redaction rules apply to memory, logs, receipts, and error messages

### Performance Requirements (NFR-PERF)

**Startup & Latency**
- **NFR-PERF-001**: Run startup (vision to first task execution) MUST complete in <5 seconds
- **NFR-PERF-002**: API status queries MUST respond in <100ms at P99
- **NFR-PERF-003**: Event log append MUST complete in <10ms at P99

**Resource Limits**
- **NFR-PERF-004**: Memory consumption per run MUST NOT exceed 500MB baseline (excluding model context)
- **NFR-PERF-005**: Individual task resource limit: 60 seconds wall time, 256MB memory (configurable per TaskSpec)
- **NFR-PERF-006**: SC-007 clarification: 10+ concurrent runs assumes 500MB/run, 4 vCPU shared, 50GB workspace disk

**Timeouts**
- **NFR-PERF-007**: Model provider call timeout: 120 seconds default, configurable per provider
- **NFR-PERF-008**: Ralph loop budget: max 5 iterations, 10 minutes total per task before escalation

**Baselines**
- **NFR-PERF-009**: SC-005 "cold run" baseline: no memory cache, fresh workspace, first run on repository
- **NFR-PERF-010**: "40% faster" measured as: (cold run time - warm run time) / cold run time >= 0.40

**Degradation Thresholds**
- **NFR-PERF-011**: Enter degraded mode when: API latency >500ms for 60s, memory usage >80%, model errors >3 consecutive
- **NFR-PERF-012**: Reservation TTL: 300 seconds default, extendable via heartbeat, rationale: balance conflict prevention with deadlock recovery

### Observability Requirements (NFR-OBS)

**Structured Logging**
- **NFR-OBS-001**: Log schema required fields: timestamp (ISO 8601), level, correlation_id, run_id, component, message
- **NFR-OBS-002**: Log levels: DEBUG, INFO, WARN, ERROR, FATAL with appropriate severity filtering

**Correlation & Tracing**
- **NFR-OBS-003**: Correlation ID format: UUIDv4, propagated across all service boundaries
- **NFR-OBS-004**: Trace sampling: 100% for errors/warnings, 10% for successful operations (configurable)

**Health Checks**
- **NFR-OBS-005**: /health returns: {"status": "healthy", "version": "X.Y.Z"} - basic liveness
- **NFR-OBS-006**: /ready returns: {"ready": true/false, "checks": {"db": true, "model": true, ...}} - dependency health
- **NFR-OBS-007**: /live returns: {"alive": true} - simple liveness for k8s probes

**Metrics**
- **NFR-OBS-008**: Metrics export format: Prometheus exposition format on /metrics endpoint
- **NFR-OBS-009**: Required metrics: run_duration_seconds, task_completion_total, model_latency_seconds, error_count_total

**Alerting**
- **NFR-OBS-010**: Alert thresholds: error rate >5% triggers warning, >10% triggers critical
- **NFR-OBS-011**: Log retention: 30 days hot storage, 90 days cold storage (compressed)

**Progress & Errors**
- **NFR-OBS-012**: Long-running builds MUST stream progress events every 30 seconds minimum
- **NFR-OBS-013**: Errors categorized as: recoverable (retry), fatal (abort), degraded (continue with warning)

### Compliance Requirements (NFR-COMP)

**Audit Trail**
- **NFR-COMP-001**: Event log provides complete audit trail - no additional audit mechanism required
- **NFR-COMP-002**: Hash chain format: SHA-256(previous_hash + event_json) for tamper detection

**Receipt Verification**
- **NFR-COMP-003**: Receipt verification: (1) validate signature with public key, (2) verify artifact hashes match, (3) confirm spec_hash matches TaskSpec
- **NFR-COMP-004**: WORM storage: optional S3 Object Lock integration for compliance-sensitive deployments

**TaskSpec Behavior**
- **NFR-COMP-005**: Strictness tiers: learning (warn only), permissive (block critical deviations), strict (block all deviations)
- **NFR-COMP-006**: SC-006 audit validation: receipts MUST be independently verifiable by third-party tools using published schema

**Provenance**
- **NFR-COMP-007**: Model provenance MUST include: model name, version, prompt template hash, temperature/settings
- **NFR-COMP-008**: Compliance reports MAY be generated on demand from event log replay

**Version Control**
- **NFR-COMP-009**: TaskSpecs MUST be versioned (semver); active runs use locked version, new runs use latest
- **NFR-COMP-010**: Non-repudiation: Enterprise receipt signatures prove origin; public key published for verification

### UX Requirements (NFR-UX)

**CLI Interface**
- **NFR-UX-001**: "Single command" SC-001: `blackice build "<vision>"` with optional flags for edition, config, TaskSpec
- **NFR-UX-002**: Non-interactive by default: all required inputs via command line; prompts only for security confirmations

**Output Formats**
- **NFR-UX-003**: CLI progress output: spinner during work, checkmarks for completed tasks, clear error summary at end
- **NFR-UX-004**: Error messages MUST be user-actionable: what failed, why, suggested fix, relevant log location

**Workspace**
- **NFR-UX-005**: Run workspace structure: repo/, tests/, docs/, decisions/, logs/ - each human-readable without tooling

**Cancellation & Resume**
- **NFR-UX-006**: Keyboard interrupt (Ctrl+C) MUST: checkpoint current state, emit event, exit cleanly within 5 seconds
- **NFR-UX-007**: Resume UX: `blackice resume <run-id>` shows "Resuming from task N of M, skipping N-1 completed tasks"

**Policy & Progress**
- **NFR-UX-008**: Policy violations MUST show: violated rule, offending input, allowed alternatives, escalation path
- **NFR-UX-009**: Progress indicators: percentage complete, current task name, elapsed/estimated time for runs >60s

**Troubleshooting**
- **NFR-UX-010**: Doctor command output: check name, status (OK/WARN/FAIL), diagnostic message, fix suggestion

### Edge Case Handling (NFR-EDGE)

- **NFR-EDGE-001**: Model provider fails mid-task: checkpoint state, attempt next provider in chain, resume if recovered
- **NFR-EDGE-002**: Memory backend unavailable during write: queue to local buffer, retry with exponential backoff, warn user
- **NFR-EDGE-003**: Event log corruption: detect via hash chain, mark corrupted range, attempt recovery from last valid checkpoint
- **NFR-EDGE-004**: Workspace collision: second run attempting same workspace MUST fail fast with clear error, no corruption
- **NFR-EDGE-005**: Empty/malformed vision: validate before run creation, return 400 with validation errors
- **NFR-EDGE-006**: Consensus tie: escalate to human operator if real-time channel available; else use weighted fallback (planner breaks tie)
- **NFR-EDGE-007**: Network partition during remote sandbox: timeout after 60s, checkpoint, mark run as recoverable, notify user

### Exception Flow Handling (NFR-EXC)

- **NFR-EXC-001**: Task fails after partial completion: rollback filesystem changes via workspace snapshot, emit rollback event
- **NFR-EXC-002**: Forcible termination: SIGTERM triggers graceful shutdown (5s), SIGKILL leaves recovery checkpoint
- **NFR-EXC-003**: Expired reservation with uncommitted changes: warn holder agent, force-release after grace period (60s), log conflict
- **NFR-EXC-004**: TaskSpec validation fails after tasks started: complete in-progress task, halt new tasks, mark run as policy-failed
- **NFR-EXC-005**: Event log append fails: retry 3 times with backoff, if still fails: halt run, preserve in-memory state, emit recovery instructions

## Assumptions

- Model providers (Claude, OpenAI, Ollama) expose compatible APIs for multi-agent execution
- Letta is available as the primary MemoryProvider implementation with local JSONL fallback
- Event sourcing uses Beads format compatible with existing BLACKICE infrastructure
- Container runtime (Docker or equivalent) is available for isolated execution environments
- SSH/tmux available for attach/rescue connectivity operations
- Standard cryptographic libraries available for receipt generation (SHA-256, signing)

## Non-Goals

- This specification does not cover the web-based operator console (CLI-first design)
- This specification does not cover pricing or licensing tiers
- This specification does not cover cloud deployment infrastructure (focus is on the core engine)
- This specification does not cover IDE integrations (future scope)
