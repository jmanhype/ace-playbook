# Data Model: BLACKICE 3.0

**Feature**: BLACKICE 3.0 Agentic Software Factory
**Date**: 2026-01-18
**Branch**: `007-blackice-3`

## Entity Relationship Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                              Run                                     │
│  (Complete execution from vision to artifacts)                       │
├─────────────────────────────────────────────────────────────────────┤
│  id, vision, status, edition, created_at, completed_at               │
│  workspace_path, config                                              │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                     │
         │ 1:N                │ 1:N                 │ 1:1 (Enterprise)
         ▼                    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│      Task       │  │      Event      │  │        Receipt          │
│  (Unit of work) │  │  (Immutable     │  │  (Cryptographic proof)  │
│                 │  │   action record)│  │                         │
├─────────────────┤  ├─────────────────┤  ├─────────────────────────┤
│  id, run_id,    │  │  id, run_id,    │  │  id, run_id, spec_hash, │
│  name, status,  │  │  type, payload, │  │  artifact_hashes,       │
│  attempt,       │  │  timestamp,     │  │  verification,          │
│  dependencies   │  │  correlation_id │  │  provenance, signature  │
└─────────────────┘  └─────────────────┘  └─────────────────────────┘
         │                                          ▲
         │ 1:N                                      │
         ▼                                          │
┌─────────────────────────────────────┐             │
│           AgentExecution            │             │
│  (Agent work on a task)             │             │
├─────────────────────────────────────┤             │
│  id, task_id, agent_role, proposal, │─────────────┘
│  confidence, voted_at               │  (contributes to)
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│          Reservation                │
│  (File/directory lease)             │
├─────────────────────────────────────┤
│  id, path, holder_agent, ttl,       │
│  acquired_at, released_at           │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│          MemoryEntry                │
│  (Persisted pattern/SOP)            │
├─────────────────────────────────────┤
│  id, namespace, key, value,         │
│  tags, embedding, created_at, ttl   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│          TaskSpec (Enterprise)      │
│  (Schema for validation)            │
├─────────────────────────────────────┤
│  id, name, version, strictness,     │
│  input_schema, output_schema,       │
│  validation_rules                   │
└─────────────────────────────────────┘
```

## Entity Definitions

### Run

A complete execution from vision to artifacts.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key, auto-generated |
| vision | string | Natural language description of desired software | Required, max 10000 chars |
| status | enum | Current state of the run | pending, planning, executing, verifying, completed, failed, paused |
| edition | enum | Product tier | lite, core, enterprise |
| created_at | datetime | When run was initiated | Auto-set, immutable |
| completed_at | datetime | When run finished | Null until completion |
| workspace_path | string | Path to run folder | Required, valid directory |
| config | object | Run configuration | Edition-specific settings |

**State Transitions**:
```
pending → planning → executing → verifying → completed
                  ↘           ↗
                    → failed
planning/executing/verifying → paused → planning/executing/verifying
```

### Task

A unit of work within a run.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key, auto-generated |
| run_id | UUID | Parent run | Foreign key to Run |
| name | string | Task description | Required, max 500 chars |
| status | enum | Current state | pending, in_progress, completed, failed, skipped |
| attempt | integer | Current attempt number | >= 1, increments on retry |
| dependencies | UUID[] | Tasks that must complete first | Valid task IDs in same run |
| plan_file | string | Path to task plan | Relative to workspace |
| state_file | string | Path to task state | Relative to workspace |
| notes_file | string | Path to task notes | Relative to workspace |
| idempotency_key | string | Key for external effects | Optional, unique per run |

### Event

An immutable record of a significant action.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key, auto-generated |
| run_id | UUID | Parent run | Foreign key to Run |
| type | string | Event type | Required, from allowed list |
| payload | object | Event-specific data | JSON, secrets redacted |
| timestamp | datetime | When event occurred | Auto-set, immutable |
| correlation_id | UUID | Links related events | Optional |
| sequence | integer | Order in event log | Auto-increment per run |
| hash | string | SHA-256 of previous event + this event | For tamper detection |

**Event Types**:
- `run.started`, `run.completed`, `run.failed`, `run.paused`, `run.resumed`
- `task.started`, `task.completed`, `task.failed`, `task.skipped`
- `agent.proposal`, `agent.vote`, `agent.decision`
- `command.executed`, `command.blocked`, `command.failed`
- `memory.query`, `memory.store`
- `checkpoint.created`, `checkpoint.restored`

### Agent

A specialist role with defined capabilities.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | string | Agent identifier | planner, implementer, reviewer, tester, security |
| model_provider | string | Backing model | claude, openai, ollama |
| capabilities | string[] | What the agent can do | Role-specific |
| system_prompt | string | Agent instructions | Required |

**Agent Roles**:
| Role | Capabilities | Purpose |
|------|--------------|---------|
| planner | decompose, prioritize, estimate | Break vision into tasks |
| implementer | code, document, refactor | Write and modify code |
| reviewer | analyze, critique, suggest | Review code for issues |
| tester | test, verify, validate | Ensure correctness |
| security | audit, scan, harden | Identify security issues |

### AgentExecution

Record of agent work on a task.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key |
| task_id | UUID | Parent task | Foreign key to Task |
| agent_role | string | Which agent | From Agent roles |
| proposal | object | Agent's proposed action/output | JSON |
| confidence | float | Agent's confidence score | 0.0 to 1.0 |
| voted_at | datetime | When vote was cast | Null if not voted |
| vote | enum | Agent's vote | approve, reject, abstain |

### Reservation

A lease on a file or directory.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key |
| path | string | File or directory path | Absolute or workspace-relative |
| holder_agent | string | Agent holding the lease | From Agent roles |
| ttl | integer | Lease duration in seconds | Default 300 |
| acquired_at | datetime | When lease was acquired | Auto-set |
| released_at | datetime | When lease was released | Null if active |

### MemoryEntry

A persisted pattern, outcome, or SOP.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key |
| namespace | string | Grouping for entries | Required, e.g., "sop", "pattern", "outcome" |
| key | string | Lookup key | Required, unique within namespace |
| value | object | Stored content | JSON, secrets redacted |
| tags | string[] | Searchable tags | Optional |
| embedding | float[] | Vector embedding | Optional, for semantic search |
| created_at | datetime | When stored | Auto-set |
| ttl | integer | Time-to-live in days | Optional, null = forever |
| retention | enum | How to handle sensitive data | redact, hash_only, full |

### TaskSpec (Enterprise)

A schema defining expected inputs, outputs, and validation rules.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key |
| name | string | Spec name | Required, unique |
| version | string | Semantic version | Required, e.g., "1.0.0" |
| strictness | enum | Validation tier | learning, permissive, strict |
| input_schema | object | JSON Schema for inputs | Required |
| output_schema | object | JSON Schema for outputs | Required |
| validation_rules | object[] | Additional validation rules | Optional |

### Receipt (Enterprise)

Cryptographic proof of run completion.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| id | UUID | Unique identifier | Primary key |
| run_id | UUID | Parent run | Foreign key to Run |
| spec_hash | string | SHA-256 of TaskSpec | Required |
| artifact_hashes | object | Hash of each artifact | Required, keyed by path |
| verification | object | Test results, scans, checks | Required |
| provenance | object | Models, prompts, tool versions | Required, hashed |
| signature | string | Ed25519 signature | Optional |
| created_at | datetime | When receipt generated | Auto-set |

## Validation Rules

### Run
- `vision` must be non-empty
- `status` transitions follow state machine
- `completed_at` only set when status is `completed` or `failed`

### Task
- `dependencies` must not create cycles
- `attempt` increments only on retry after failure
- `idempotency_key` required for tasks with external side effects

### Event
- `payload` must not contain unredacted secrets
- `hash` must chain correctly from previous event
- `sequence` must be monotonically increasing

### Reservation
- Only one active reservation per path
- `ttl` must be positive
- Expired reservations automatically released

### MemoryEntry
- `key` unique within `namespace`
- `embedding` dimension must match configured model
- `value` redacted according to `retention` policy

### TaskSpec
- `input_schema` and `output_schema` must be valid JSON Schema
- `version` must follow semantic versioning
- `strictness` determines enforcement behavior

### Receipt
- `spec_hash` must match actual TaskSpec used
- `artifact_hashes` must match actual artifact contents
- `verification` must include all required checks per edition
