# Implementation Tasks: ACE Framework

**Generated**: 2025-10-13 (Updated with Phase 13)
**Feature**: Agentic Context Engineering (Generator-Reflector-Curator Pattern)
**Total Tasks**: 88
**Priority Distribution**: P1 (32 tasks), P2 (34 tasks), P3 (22 tasks)
**Total Estimated Effort**: ~150 hours
**New in v2.0**:
- Phase 10-12 added based on senior engineering review recommendations (production hardening)
- Phase 13 added for comprehensive documentation (onboarding & accessibility)

---

## Task Organization

Tasks are organized by **user story** to enable independent implementation and testing. Each story phase includes:
- Story goal and acceptance criteria
- Independent test verification
- All components needed to complete JUST that story
- Dependencies and parallelization opportunities

**Legend**:
- `[P]` = Parallelizable (can run concurrently with other [P] tasks)
- `[BLOCKING]` = Must complete before dependent tasks
- `[OPTIONAL]` = Nice-to-have, not required for MVP

---

## Phase 1: Project Setup (7 tasks)

**Goal**: Initialize project infrastructure needed by all user stories.

### T001: Initialize Python 3.11 Project with uv [BLOCKING]
**Description**: Set up Python 3.11 project structure with uv package manager.
**Acceptance**:
- Create project root directory: `ace/`
- Initialize git repository
- Install uv package manager
- Verify Python 3.11+ available
**Deliverables**: Project directory structure, `.git/` initialized
**Estimated Effort**: 30 min
**Dependencies**: None

### T002: Create pyproject.toml with Core Dependencies [BLOCKING]
**Description**: Define project metadata and pin dependency versions.
**Acceptance**:
- `pyproject.toml` with PEP 621 format
- Dependencies: dspy-ai, sentence-transformers, faiss-cpu, pydantic>=2.0
- Dev dependencies: pytest, pytest-cov, black, ruff
- Pin major versions (e.g., `dspy-ai==2.4.*`)
**Deliverables**: `pyproject.toml`, `uv.lock`
**Estimated Effort**: 45 min
**Dependencies**: T001
**References**: research.md:41-67

### T003: Create Project Directory Structure [BLOCKING]
**Description**: Scaffold directories per plan.md architecture.
**Acceptance**:
```
ace/
  __init__.py
  core/         # Generator, Reflector, Curator modules
  runner/       # Workflow orchestration
  store/        # Persistence layer
  tools/        # Code execution, HTTP tools
  ops/          # Metrics, guardrails
  cli.py
tests/
  contract/
  integration/
  unit/
  performance/
```
**Deliverables**: Directory structure with `__init__.py` files
**Estimated Effort**: 20 min
**Dependencies**: T001
**References**: plan.md:255-270

### T004: Set Up SQLite Database Schema [BLOCKING]
**Description**: Create SQLite database with bullets and diff_journal tables.
**Acceptance**:
- Schema matches data-model.md entities
- Indexes on (helpful_count, harmful_count, last_used_at)
- JSON1 extension enabled for tags
- WAL mode enabled for concurrency
**Deliverables**: `ace/store/schema.sql`, migration script
**Estimated Effort**: 1.5 hours
**Dependencies**: T003
**References**: data-model.md:21-114, research.md:94-120

### T005: Create Docker Code Execution Sandbox [P]
**Description**: Set up Docker container for safe Python code execution.
**Acceptance**:
- `python:3.11-slim` base image
- 256MB memory limit, 10s timeout
- `network_mode="none"` for isolation
- Volume mount for code injection
- Returns stdout/stderr/exit_code
**Deliverables**: `ace/tools/code_executor.py`, Docker configuration
**Estimated Effort**: 2 hours
**Dependencies**: T002
**References**: research.md:188-231

### T006: Set Up Pytest Test Infrastructure [P]
**Description**: Configure pytest with coverage and fixtures.
**Acceptance**:
- `pytest.ini` with coverage settings (≥80% target)
- Fixture for SQLite test database
- Fixture for mock playbook data
- Fixture for DSPy test signatures
**Deliverables**: `pytest.ini`, `tests/conftest.py`
**Estimated Effort**: 1 hour
**Dependencies**: T002
**References**: plan.md:239-253

### T007: Set Up Pre-commit Hooks and Linting [P] [OPTIONAL]
**Description**: Configure black, ruff, mypy for code quality.
**Acceptance**:
- `.pre-commit-config.yaml` with black, ruff
- `pyproject.toml` mypy configuration
- Type hints enforced for public APIs
**Deliverables**: `.pre-commit-config.yaml`, linting configuration
**Estimated Effort**: 45 min
**Dependencies**: T002

---

## Phase 2: Foundational Infrastructure (12 tasks)

**Goal**: Build shared components that ALL user stories depend on.

### T008: Implement PlaybookBullet Pydantic Model [BLOCKING]
**Description**: Create type-safe model for playbook bullets.
**Acceptance**:
- Pydantic model matching data-model.md:21-68
- Validators: content length 10-500 chars, section in {Helpful, Harmful, Neutral}
- `compute_hash()` method using SHA-256
- `to_dict()` and `from_dict()` serialization
**Deliverables**: `ace/core/models.py` with `PlaybookBullet` class
**Estimated Effort**: 1.5 hours
**Dependencies**: T002
**References**: data-model.md:21-68, contracts/curator.py:42-93

### T009: Implement Playbook Container Class [BLOCKING]
**Description**: Create Playbook class managing bullet collections.
**Acceptance**:
- Pydantic model with bullets: List[PlaybookBullet]
- Validation: 0-300 bullets max
- Computed field: `bullet_count` property
- Methods: `add_bullet()`, `get_bullet()`, `get_by_section()`
**Deliverables**: `Playbook` class in `ace/core/models.py`
**Estimated Effort**: 1 hour
**Dependencies**: T008
**References**: data-model.md:70-114

### T010: Implement SQLite CRUD Operations [BLOCKING]
**Description**: Database persistence layer for bullets and playbooks.
**Acceptance**:
- `PlaybookDB` class with connection pooling
- Methods: `create_bullet()`, `update_counters()`, `get_bullets()`
- Transactional batch updates (BEGIN/COMMIT)
- Exception handling with rollback
**Deliverables**: `ace/store/playbook_db.py`
**Estimated Effort**: 2.5 hours
**Dependencies**: T004, T008
**References**: research.md:94-120

### T011: Implement Diff Journal Persistence [BLOCKING]
**Description**: Append-only audit trail for playbook changes.
**Acceptance**:
- `DiffJournal` class writing to diff_journal table
- Methods: `append()`, `get_history()`, `get_by_bullet_id()`
- SHA-256 before/after hashes
- Transaction-safe appends
**Deliverables**: `ace/store/diff_journal.py`
**Estimated Effort**: 2 hours
**Dependencies**: T004, T008
**References**: data-model.md:304-370, contracts/curator.py:112-135

### T012: Set Up sentence-transformers Embedding Model [BLOCKING]
**Description**: Initialize embedding model for semantic similarity.
**Acceptance**:
- Lazy loading of `all-MiniLM-L6-v2` model
- `embed_text()` method returning 384-dim vectors
- Batch embedding support (≤100 texts/batch)
- Model caching to avoid re-downloads
**Deliverables**: `ace/core/embeddings.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T002
**References**: research.md:122-148

### T013: Implement FAISS Index Manager [BLOCKING]
**Description**: Fast similarity search for semantic deduplication.
**Acceptance**:
- `IndexFlatIP` with L2-normalized embeddings
- Methods: `build_index()`, `search_similar()`, `add_vectors()`
- Returns top-K with similarity scores
- <10ms P50 search on 100-bullet index
**Deliverables**: `ace/core/faiss_index.py`
**Estimated Effort**: 2.5 hours
**Dependencies**: T012
**References**: research.md:122-148, contracts/curator.py:338-351

### T014: Implement Semantic Similarity Utilities [P]
**Description**: Cosine similarity computation and threshold checks.
**Acceptance**:
- `cosine_similarity()` function using numpy
- `is_duplicate()` helper with ≥0.8 threshold
- Unit tests with known similar/dissimilar pairs
**Deliverables**: `ace/core/similarity.py`
**Estimated Effort**: 1 hour
**Dependencies**: T012
**References**: contracts/curator.py:230-241

### T015: Create Task and TaskOutput Models [P]
**Description**: Pydantic models for task execution.
**Acceptance**:
- `Task` model matching data-model.md:116-161
- `TaskOutput` model with reasoning_trace: List[str]
- Validators: trace length 1-20 steps, confidence 0.0-1.0
**Deliverables**: Task/TaskOutput classes in `ace/core/models.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T008
**References**: data-model.md:116-214

### T016: Create ExecutionFeedback Model [P]
**Description**: Pydantic model for execution signals.
**Acceptance**:
- Model matching data-model.md:216-259
- Optional fields: tool_results, errors, performance_metrics
- Validator: at least one feedback field must be populated
**Deliverables**: ExecutionFeedback class in `ace/core/models.py`
**Estimated Effort**: 1 hour
**Dependencies**: T008
**References**: data-model.md:216-259

### T017: Create Reflection Model [P]
**Description**: Pydantic model for Reflector output.
**Acceptance**:
- Model matching data-model.md:261-302
- `insights` field with 0-10 InsightCandidate objects
- Validator: confidence_score 0.0-1.0
**Deliverables**: Reflection class in `ace/core/models.py`
**Estimated Effort**: 1 hour
**Dependencies**: T008
**References**: data-model.md:261-302

### T018: Implement DSPy Signature Base Classes [P]
**Description**: Create reusable DSPy signature types.
**Acceptance**:
- Import dspy, define custom InputField/OutputField shortcuts
- Base signature class with version tracking
- Type hints for all fields
**Deliverables**: `ace/core/signatures.py`
**Estimated Effort**: 45 min
**Dependencies**: T002
**References**: contracts/generator.py, contracts/reflector.py

### T019: Write Unit Tests for Foundational Models [P]
**Description**: Test coverage for Pydantic models and utilities.
**Acceptance**:
- Tests for all validators (content length, confidence range, etc.)
- Tests for serialization (to_dict, from_dict)
- Tests for hash computation determinism
- ≥90% coverage on `ace/core/models.py`
**Deliverables**: `tests/unit/test_models.py`
**Estimated Effort**: 2 hours
**Dependencies**: T008-T017

---

## Phase 3: User Story 1 - Accumulating Domain Knowledge (P1) (9 tasks)

**Story Goal**: Accumulate domain-specific strategies in a persistent playbook without fine-tuning the base LLM.

**Acceptance Criteria**:
- Start with empty playbook
- Execute 10 tasks in a domain (e.g., arithmetic)
- Playbook grows to ≥5 bullets with helpful/harmful counters
- Bullets persist across runs

**Independent Test**: Run offline training on GSM8K subset → verify playbook JSON contains 5+ bullets with non-zero counters.

### T020: Implement Basic Playbook Retrieval [BLOCKING]
**Description**: Retrieve top-K playbook bullets by recency/relevance.
**Acceptance**:
- `PlaybookRetriever` class
- Methods: `get_top_k()`, `get_by_section()`
- Ranking: helpful_count DESC, last_used_at DESC
- Default K=40 (context budget limit)
**Deliverables**: `ace/core/retriever.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T010
**References**: plan.md:48-65

### T021: Implement Playbook Bullet Addition [BLOCKING]
**Description**: Add new bullets to playbook with deduplication.
**Acceptance**:
- `add_bullet()` method in PlaybookDB
- Checks for duplicates using FAISS similarity
- If cosine ≥0.8: increment existing counter
- If cosine <0.8: insert new bullet
- Append to diff journal
**Deliverables**: Method in `ace/store/playbook_db.py`
**Estimated Effort**: 2 hours
**Dependencies**: T010, T013, T011
**References**: contracts/curator.py:215-228

### T022: Implement Counter Update Operations [BLOCKING]
**Description**: Increment helpful/harmful counters for existing bullets.
**Acceptance**:
- `increment_helpful()` and `increment_harmful()` methods
- Update last_used_at timestamp
- Compute before/after SHA-256 hashes
- Atomic transaction (read-modify-write)
**Deliverables**: Methods in `ace/store/playbook_db.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T010, T011
**References**: contracts/curator.py:34-39

### T023: Create Playbook Inspection CLI [P]
**Description**: Command-line interface to view playbook contents.
**Acceptance**:
- `ace playbook show` displays all bullets
- `ace playbook stats` shows counts by section
- `ace playbook history <bullet_id>` shows diff journal
- JSON output option: `--format=json`
**Deliverables**: Commands in `ace/cli.py`
**Estimated Effort**: 2 hours
**Dependencies**: T010, T011
**References**: spec.md:124 (FR-009)

### T024: Write Integration Test for Bullet Addition [P]
**Description**: End-to-end test for adding bullets to empty playbook.
**Acceptance**:
- Start with empty SQLite database
- Add 5 bullets sequentially
- Verify bullet_count increases to 5
- Verify diff_journal has 5 "add" entries
**Deliverables**: `tests/integration/test_playbook_growth.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T020-T022
**References**: US1 acceptance criteria

### T025: Write Integration Test for Counter Increments [P]
**Description**: Test that counters increment instead of duplicating.
**Acceptance**:
- Add bullet "Strategy A" with helpful=1
- Add similar bullet (cosine ≥0.8)
- Verify bullet_count stays at 1
- Verify helpful_count increments to 2
**Deliverables**: Test in `tests/integration/test_playbook_growth.py`
**Estimated Effort**: 1 hour
**Dependencies**: T020-T022
**References**: US1 acceptance criteria

### T026: Implement Playbook JSON Export [P]
**Description**: Export playbook to JSON for portability.
**Acceptance**:
- `export_to_json()` method in Playbook class
- Schema version in output
- Includes all bullets with metadata
- `ace playbook export output.json` CLI command
**Deliverables**: Export method, CLI command
**Estimated Effort**: 1 hour
**Dependencies**: T009, T023
**References**: plan.md:130-143

### T027: Implement Playbook JSON Import [P]
**Description**: Import playbook from JSON file.
**Acceptance**:
- `import_from_json()` validates schema version
- Overwrites existing playbook or merges (flag option)
- `ace playbook import input.json` CLI command
**Deliverables**: Import method, CLI command
**Estimated Effort**: 1.5 hours
**Dependencies**: T026
**References**: plan.md:130-143

### T028: Write Performance Benchmark for Playbook Operations [P] [OPTIONAL]
**Description**: Measure retrieval and update latency.
**Acceptance**:
- Benchmark: 100 bullets → measure get_top_k() P50/P95
- Benchmark: 100 bullets → measure add_bullet() latency
- Assert: retrieval ≤10ms P50, ≤25ms P95
- Assert: addition ≤50ms P50
**Deliverables**: `tests/performance/test_playbook_perf.py`
**Estimated Effort**: 2 hours
**Dependencies**: T020-T022
**References**: plan.md:48-65

---

## Phase 4: User Story 2 - Incremental Updates Without Collapse (P1) (8 tasks)

**Story Goal**: Prevent context collapse through semantic deduplication and append-only updates.

**Acceptance Criteria**:
- Cosine similarity ≥0.8 triggers counter increment (not new bullet)
- Monitor 50+ task iterations
- Playbook content never rewritten (verified via SHA-256 diff journal)
- Playbook size stabilizes (not linear growth)

**Independent Test**: Run 50 tasks → verify diff_journal shows 0 "update" operations (only "add" and "increment_*").

### T029: Implement SemanticCurator Base Class [BLOCKING]
**Description**: Core deduplication logic using embeddings and FAISS.
**Acceptance**:
- `SemanticCurator` class implementing CuratorInterface
- `apply_delta()` method processes InsightCandidate list
- Computes embeddings for new candidates
- Searches FAISS for duplicates (threshold=0.8)
**Deliverables**: `ace/core/curator.py` with SemanticCurator
**Estimated Effort**: 3 hours
**Dependencies**: T013, T014
**References**: contracts/curator.py:269-351

### T030: Implement Deduplication with Counter Logic [BLOCKING]
**Description**: Merge logic: increment counters vs add new bullets.
**Acceptance**:
- If similarity ≥0.8: create INCREMENT_HELPFUL/HARMFUL operation
- If similarity <0.8: create ADD_NEW operation
- Returns CuratorOutput with DeltaUpdate list
**Deliverables**: Logic in `ace/core/curator.py`
**Estimated Effort**: 2.5 hours
**Dependencies**: T029
**References**: contracts/curator.py:34-40, 112-135

### T031: Implement SHA-256 Diff Journal Generation [BLOCKING]
**Description**: Generate audit trail entries for all playbook changes.
**Acceptance**:
- Compute SHA-256 before/after for each DeltaUpdate
- Create DiffJournalEntry for each operation
- Batch write to diff_journal table
- Transactional: all-or-nothing commits
**Deliverables**: Logic in `ace/core/curator.py`
**Estimated Effort**: 2 hours
**Dependencies**: T030, T011
**References**: contracts/curator.py:112-135

### T032: Implement Curator apply_delta() Method [BLOCKING]
**Description**: Main entry point for merging insights into playbook.
**Acceptance**:
- Accepts CuratorInput (insights + current playbook)
- Computes embeddings for all insights
- Performs similarity search for each insight
- Returns CuratorOutput with updated playbook + stats
**Deliverables**: Complete apply_delta() implementation
**Estimated Effort**: 2 hours
**Dependencies**: T029-T031
**References**: contracts/curator.py:215-228

### T033: Write Unit Tests for Semantic Deduplication [P]
**Description**: Test similarity threshold behavior.
**Acceptance**:
- Test: cosine=0.85 → triggers increment (not add)
- Test: cosine=0.75 → triggers add (not increment)
- Test: exact match (cosine=1.0) → increment
- Mock embeddings to control similarity scores
**Deliverables**: `tests/unit/test_curator.py`
**Estimated Effort**: 2 hours
**Dependencies**: T029-T032

### T034: Write Integration Test for Deduplication Workflow [P]
**Description**: End-to-end test preventing duplicate bullets.
**Acceptance**:
- Start with 1 bullet: "Break problems into steps"
- Add similar insight: "Decompose into smaller steps"
- Verify bullet_count stays at 1
- Verify helpful_count increments
- Verify diff_journal has 1 "add" + 1 "increment" entry
**Deliverables**: `tests/integration/test_deduplication.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T029-T032
**References**: US2 acceptance criteria

### T035: Write Integration Test for Append-Only Guarantee [P]
**Description**: Verify no content rewrites occur.
**Acceptance**:
- Run 50 task iterations
- Check diff_journal for all operations
- Assert: 0 operations with type="update" or "rewrite"
- Assert: all bullets have stable content (SHA-256 unchanged)
**Deliverables**: Test in `tests/integration/test_append_only.py`
**Estimated Effort**: 2 hours
**Dependencies**: T029-T032
**References**: US2 acceptance criteria

### T036: Implement Playbook Consolidation Trigger [P] [OPTIONAL]
**Description**: Automatic consolidation at 150 bullets threshold.
**Acceptance**:
- Monitor bullet_count in Curator
- When count ≥150: trigger consolidation
- Merge similar bullets (cosine ≥0.85)
- Prune bullets with helpful=0, harmful=0, unused ≥10 days
- Target: reduce to ≤100 bullets
**Deliverables**: Consolidation logic in `ace/core/curator.py`
**Estimated Effort**: 3 hours
**Dependencies**: T032
**References**: contracts/curator.py:377-382, plan.md:166-188

---

## Phase 5: User Story 3 - Explicit Reasoning Traces (P2) (7 tasks)

**Story Goal**: Generate step-by-step reasoning traces that show strategy consultation and decision-making process.

**Acceptance Criteria**:
- Reasoning trace shows: problem decomposition, strategy application, intermediate steps
- Trace explicitly references playbook bullet IDs consulted
- Trace is structured (not free-form prose)

**Independent Test**: Parse TaskOutput.reasoning_trace → verify array of strings with labeled steps (e.g., "Step 1: ...", "Applying strategy bullet-123: ...").

### T037: Define TaskInput DSPy Signature [BLOCKING]
**Description**: Formal interface for Generator input.
**Acceptance**:
- DSPy Signature matching contracts/generator.py:16-52
- Fields: task_id, description, domain, playbook_bullets, context
- Default values for optional fields
**Deliverables**: `TaskInput` signature in `ace/core/signatures.py`
**Estimated Effort**: 1 hour
**Dependencies**: T018
**References**: contracts/generator.py:16-52

### T038: Define TaskOutput DSPy Signature [BLOCKING]
**Description**: Formal interface for Generator output.
**Acceptance**:
- DSPy Signature matching contracts/generator.py:54-100
- Fields: task_id, reasoning_trace, answer, confidence, bullets_referenced
- Performance fields: latency_ms, model_name, tokens
**Deliverables**: `TaskOutput` signature in `ace/core/signatures.py`
**Estimated Effort**: 1 hour
**Dependencies**: T018
**References**: contracts/generator.py:54-100

### T039: Implement CoTGenerator Base Module [BLOCKING]
**Description**: Chain-of-Thought Generator using DSPy.
**Acceptance**:
- Inherits from GeneratorModule
- Uses `dspy.ChainOfThought` predictor
- Injects playbook bullets into prompt context
- Instructs model to output structured trace
**Deliverables**: `ace/core/generator.py` with CoTGenerator class
**Estimated Effort**: 3 hours
**Dependencies**: T037, T038
**References**: contracts/generator.py:179-212

### T040: Implement Reasoning Trace Formatter [BLOCKING]
**Description**: Post-process LLM output into structured trace.
**Acceptance**:
- Parse LLM response into List[str] steps
- Detect strategy references (e.g., "Using bullet-abc123")
- Extract bullet IDs into bullets_referenced list
- Ensure trace has 1-20 steps (truncate if needed)
**Deliverables**: `format_trace()` method in generator.py
**Estimated Effort**: 2 hours
**Dependencies**: T039
**References**: data-model.md:163-214 (reasoning_trace validation)

### T041: Implement Playbook Context Injection [BLOCKING]
**Description**: Inject top-K bullets into Generator prompt.
**Acceptance**:
- Retrieve top-K=40 bullets from playbook
- Format as numbered list with IDs
- Append to task description in prompt
- Stay within context budget (≤300 tokens)
**Deliverables**: `inject_playbook_context()` in generator.py
**Estimated Effort**: 1.5 hours
**Dependencies**: T020, T039
**References**: plan.md:48-65

### T042: Write Unit Tests for Trace Formatting [P]
**Description**: Test trace parsing and validation.
**Acceptance**:
- Test: multi-line LLM output → structured List[str]
- Test: detect bullet references → populate bullets_referenced
- Test: enforce 1-20 step limit
- Mock LLM responses
**Deliverables**: `tests/unit/test_generator.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T039-T041

### T043: Write Integration Test for Reasoning Traces [P]
**Description**: End-to-end test generating structured traces.
**Acceptance**:
- Initialize playbook with 3 bullets
- Execute task with CoTGenerator
- Verify TaskOutput.reasoning_trace is List[str] with ≥3 steps
- Verify bullets_referenced contains at least 1 ID
- Verify trace shows strategy consultation
**Deliverables**: `tests/integration/test_reasoning_traces.py`
**Estimated Effort**: 2 hours
**Dependencies**: T039-T041
**References**: US3 acceptance criteria

---

## Phase 6: User Story 4 - Learning from Execution Feedback (P2) (8 tasks)

**Story Goal**: Extract labeled insights from execution feedback without requiring human annotation.

**Acceptance Criteria**:
- Generate InsightCandidate objects from ground-truth comparisons
- Generate insights from test pass/fail results
- Generate insights from error messages
- No manual labeling required

**Independent Test**: Run task with ground_truth → verify Reflection.insights contains Helpful/Harmful labels based on correctness signal.

### T044: Define ReflectorInput DSPy Signature [BLOCKING]
**Description**: Formal interface for Reflector input.
**Acceptance**:
- DSPy Signature matching contracts/reflector.py:35-93
- Fields: task_id, reasoning_trace, answer, confidence, bullets_referenced
- Optional feedback: ground_truth, test_results, error_messages
**Deliverables**: `ReflectorInput` signature in `ace/core/signatures.py`
**Estimated Effort**: 1 hour
**Dependencies**: T018
**References**: contracts/reflector.py:35-93

### T045: Define InsightCandidate DSPy Signature [BLOCKING]
**Description**: Structure for single extracted insight.
**Acceptance**:
- DSPy Signature matching contracts/reflector.py:95-124
- Fields: content, section (Helpful/Harmful/Neutral), confidence, rationale
- Validation: content 10-500 chars, section enum, confidence 0.0-1.0
**Deliverables**: `InsightCandidate` signature in `ace/core/signatures.py`
**Estimated Effort**: 45 min
**Dependencies**: T018
**References**: contracts/reflector.py:95-124

### T046: Define ReflectorOutput DSPy Signature [BLOCKING]
**Description**: Formal interface for Reflector output.
**Acceptance**:
- DSPy Signature matching contracts/reflector.py:126-168
- Fields: task_id, insights (List[InsightCandidate]), analysis_summary
- Metadata: referenced_steps, confidence_score, feedback_types_used
**Deliverables**: `ReflectorOutput` signature in `ace/core/signatures.py`
**Estimated Effort**: 1 hour
**Dependencies**: T045
**References**: contracts/reflector.py:126-168

### T047: Implement GroundedReflector Module [BLOCKING]
**Description**: Reflector using ground-truth and execution feedback.
**Acceptance**:
- Inherits from ReflectorModule
- Uses `dspy.ChainOfThought` for analysis
- Prompt includes: reasoning_trace, answer, ground_truth, test_results
- Instructs model to label insights as Helpful/Harmful/Neutral
**Deliverables**: `ace/core/reflector.py` with GroundedReflector class
**Estimated Effort**: 3 hours
**Dependencies**: T044-T046
**References**: contracts/reflector.py:212-243

### T048: Implement Ground-Truth Comparison Logic [BLOCKING]
**Description**: Compare answer to ground_truth and generate labels.
**Acceptance**:
- If answer == ground_truth: strategies → Helpful
- If answer != ground_truth: strategies → Harmful
- If no ground_truth: use test_results or errors
- Generate rationale explaining label
**Deliverables**: Logic in `ace/core/reflector.py`
**Estimated Effort**: 2 hours
**Dependencies**: T047
**References**: contracts/reflector.py:19-25

### T049: Implement Test Result Analysis Logic [BLOCKING]
**Description**: Extract insights from test pass/fail outcomes.
**Acceptance**:
- Parse test_results dict: {test_name: passed}
- If all passed: reasoning steps → Helpful
- If any failed: identify failing steps → Harmful
- Generate rationale referencing specific tests
**Deliverables**: Logic in `ace/core/reflector.py`
**Estimated Effort**: 2 hours
**Dependencies**: T047
**References**: contracts/reflector.py:19-25

### T050: Write Unit Tests for Insight Labeling [P]
**Description**: Test label assignment logic.
**Acceptance**:
- Test: correct answer → Helpful label
- Test: wrong answer → Harmful label
- Test: test pass → Helpful label
- Test: test fail → Harmful label
- Mock ReflectorInput with feedback signals
**Deliverables**: `tests/unit/test_reflector.py`
**Estimated Effort**: 2 hours
**Dependencies**: T047-T049

### T051: Write Integration Test for Feedback Learning [P]
**Description**: End-to-end test extracting labeled insights.
**Acceptance**:
- Execute task with CoTGenerator
- Add ground_truth to ExecutionFeedback
- Run GroundedReflector
- Verify Reflection.insights contains ≥1 InsightCandidate
- Verify labels match correctness (Helpful if correct, Harmful if wrong)
**Deliverables**: `tests/integration/test_feedback_learning.py`
**Estimated Effort**: 2 hours
**Dependencies**: T047-T049
**References**: US4 acceptance criteria

---

## Phase 7: User Story 5 - Offline Pre-Training (P3) (6 tasks)

**Story Goal**: Bootstrap playbook from offline dataset before production deployment.

**Acceptance Criteria**:
- Process 100 problems from dataset (e.g., GSM8K)
- Generate playbook with ≥20 strategies
- Validation set shows ≥20% improvement over empty playbook

**Independent Test**: Run offline_train.py on GSM8K subset → measure accuracy on held-out set before/after.

### T052: Implement Offline Training Runner [BLOCKING]
**Description**: Batch processing workflow for dataset pre-training.
**Acceptance**:
- `offline_train.py` CLI script
- Arguments: --dataset, --num_examples, --output_playbook
- Loop: load example → execute Generator → run Reflector → merge with Curator
- Progress bar showing examples processed
**Deliverables**: `ace/runner/offline_train.py`
**Estimated Effort**: 3 hours
**Dependencies**: T039, T047, T032
**References**: plan.md:84-105

### T053: Implement Dataset Loader for GSM8K [BLOCKING]
**Description**: Load and parse GSM8K dataset.
**Acceptance**:
- Load from HuggingFace datasets library
- Parse question + ground_truth answer
- Convert to Task format
- Support train/validation split
**Deliverables**: `ace/runner/dataset_loader.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T015
**References**: plan.md:84-105

### T054: Implement Batch Insight Merging [BLOCKING]
**Description**: Efficient merging of multiple Reflections at once.
**Acceptance**:
- Batch CuratorInput with all insights from N tasks
- Single FAISS index build for all embeddings
- Deduplicate across entire batch
- Single transaction to commit all updates
**Deliverables**: `batch_merge()` method in curator.py
**Estimated Effort**: 2 hours
**Dependencies**: T032
**References**: plan.md:84-105

### T055: Implement GEPA Prompt Optimization [P] [OPTIONAL]
**Description**: Use DSPy GEPA teleprompter to optimize prompts.
**Acceptance**:
- Define evaluation metric (exact match accuracy)
- Run GEPA on Generator + Reflector prompts
- Save optimized prompts to file
- 5-50 evaluation examples
**Deliverables**: `ace/ops/optimize_prompts.py` script
**Estimated Effort**: 4 hours
**Dependencies**: T039, T047
**References**: research.md:233-267

### T056: Write Integration Test for Offline Training [P]
**Description**: End-to-end test of batch training workflow.
**Acceptance**:
- Start with empty playbook
- Run offline_train on 10 GSM8K examples
- Verify playbook contains ≥5 bullets
- Verify bullets have non-zero counters
- Verify diff_journal has ≥5 entries
**Deliverables**: `tests/integration/test_offline_training.py`
**Estimated Effort**: 2.5 hours
**Dependencies**: T052-T054
**References**: US5 acceptance criteria

### T057: Write Performance Test for Batch Processing [P] [OPTIONAL]
**Description**: Measure throughput of offline training.
**Acceptance**:
- Benchmark: 100 examples → measure total time
- Assert: ≤5 seconds per example (including LLM calls)
- Assert: final playbook has ≤150 bullets (dedup working)
**Deliverables**: `tests/performance/test_batch_training.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T052-T054

---

## Phase 8: User Story 6 - Online Adaptation in Production (P3) (7 tasks)

**Story Goal**: Continuously improve playbook from production traffic with safety gates and rollback.

**Acceptance Criteria**:
- Run for 7 days in shadow mode
- Generate ≥10 new strategies
- Confidence thresholds enforced (≥0.7 for staging)
- Human review queue for low-confidence insights (<0.6)

**Independent Test**: Run online_loop for 100 iterations → verify insights logged to shadow playbook, not applied to prod playbook.

### T058: Implement Shadow Learning Mode [BLOCKING]
**Description**: Log insights without applying to production retrieval.
**Acceptance**:
- Playbook stage enum: shadow, staging, prod
- Shadow bullets excluded from retrieval
- All bullets created with stage=shadow initially
- CLI: `ace playbook set-stage <bullet_id> staging`
**Deliverables**: Stage logic in `ace/store/playbook_db.py`
**Estimated Effort**: 2 hours
**Dependencies**: T010, T009
**References**: contracts/curator.py:26-30, plan.md:176-188

### T059: Implement Promotion Gate Logic [BLOCKING]
**Description**: Automated promotion based on helpful/harmful counters.
**Acceptance**:
- Check: helpful_count ≥3 AND helpful:harmful ≥3:1
- If true: promote shadow → staging
- Check: helpful_count ≥5 AND helpful:harmful ≥5:1
- If true: promote staging → prod
- Log promotion events to diff_journal
**Deliverables**: `promote_bullet()` in curator.py
**Estimated Effort**: 2 hours
**Dependencies**: T032, T058
**References**: contracts/curator.py:362-375, plan.md:176-188

### T060: Implement Quarantine Logic [BLOCKING]
**Description**: Automatically quarantine harmful strategies.
**Acceptance**:
- Check: harmful_count ≥ helpful_count AND helpful_count > 0
- If true: exclude bullet from retrieval
- Mark with quarantine flag in database
- CLI: `ace playbook quarantine <bullet_id>`
**Deliverables**: Quarantine logic in curator.py
**Estimated Effort**: 1.5 hours
**Dependencies**: T032, T058
**References**: contracts/curator.py:256-266, plan.md:176-188

### T061: Implement Online Learning Loop [BLOCKING]
**Description**: Continuous adaptation workflow for production.
**Acceptance**:
- `online_loop.py` daemon process
- Poll task queue or listen to events
- For each task: execute → get feedback → reflect → merge (shadow)
- Periodic promotion check (every N tasks or time interval)
**Deliverables**: `ace/runner/online_loop.py`
**Estimated Effort**: 3 hours
**Dependencies**: T039, T047, T032, T059
**References**: plan.md:106-127

### T062: Implement Human Review Queue [P]
**Description**: Queue low-confidence insights for manual review.
**Acceptance**:
- If insight confidence <0.6: write to review_queue table
- CLI: `ace review list` shows pending items
- CLI: `ace review approve <insight_id>` promotes to shadow
- CLI: `ace review reject <insight_id>` discards
**Deliverables**: Review queue in `ace/store/review_queue.py`, CLI commands
**Estimated Effort**: 2.5 hours
**Dependencies**: T047, T023
**References**: contracts/reflector.py:159-167

### T063: Write Integration Test for Shadow Learning [P]
**Description**: Verify shadow insights don't affect retrieval.
**Acceptance**:
- Add 5 bullets to shadow stage
- Retrieve top-K bullets
- Assert: 0 shadow bullets in results
- Promote 1 bullet to staging
- Assert: 1 staging bullet in results (if staging retrieval enabled)
**Deliverables**: `tests/integration/test_shadow_learning.py`
**Estimated Effort**: 1.5 hours
**Dependencies**: T058-T061
**References**: US6 acceptance criteria

### T064: Write Integration Test for Promotion Gates [P]
**Description**: Test automated promotion logic.
**Acceptance**:
- Create bullet with helpful=0, harmful=0, stage=shadow
- Increment helpful to 3, harmful to 0
- Run promotion check
- Assert: stage promoted to staging
- Increment helpful to 5
- Assert: stage promoted to prod
**Deliverables**: Test in `tests/integration/test_promotion_gates.py`
**Estimated Effort**: 2 hours
**Dependencies**: T059, T060
**References**: US6 acceptance criteria

---

## Phase 9: Polish & Cross-Cutting Concerns (4 tasks) ✅ COMPLETED

**Goal**: Production readiness features not tied to specific user stories.
**Status**: ✅ All tasks completed (2025-10-13)
**Release**: v1.9.0

### T065: Implement Observability Metrics [P] ✅ COMPLETED
**Description**: Prometheus-style metrics for monitoring.
**Acceptance**:
- ✅ Metrics: playbook_bullet_count, retrieval_latency_ms, reflection_count
- ✅ Metrics: promotion_events_total, quarantine_events_total
- ✅ Export endpoint: Prometheus text format via `export_prometheus()`
- ✅ Thread-safe counters with Lock
- ✅ LatencyTimer context manager
**Deliverables**: `ace/ops/metrics.py` (293 lines)
**Actual Effort**: 2 hours
**Dependencies**: T010, T032
**References**: plan.md:189-211
**Commit**: e44ff21

### T066: Implement Guardrail Monitoring [P] ✅ COMPLETED
**Description**: Automated rollback on performance regression.
**Acceptance**:
- ✅ Monitor: success rate delta, latency P95, error rate
- ✅ Thresholds: success_delta < -8%, latency_delta > +30%, error_rate > 15%
- ✅ Rollback: quarantine bullets added in last 24 hours
- ✅ PerformanceSnapshot for baseline comparison
- ✅ Alert: callback system for notifications
- ✅ RollbackTrigger audit trail
**Deliverables**: `ace/ops/guardrails.py` (299 lines)
**Actual Effort**: 3 hours
**Dependencies**: T061, T065
**References**: plan.md:176-188
**Commit**: e44ff21

### T067: Create Docker Compose Setup [P] ✅ COMPLETED
**Description**: Local development environment with all services.
**Acceptance**:
- ✅ `docker-compose.yml` with ace service + SQLite volume
- ✅ Environment variables for LLM API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
- ✅ Volume mount for playbook persistence (/data)
- ✅ Health check endpoint
- ✅ Automatic Alembic migrations on startup
- ✅ `.dockerignore` for optimized builds
**Deliverables**:
- `docker-compose.yml` (48 lines)
- `Dockerfile` (41 lines, Python 3.11-slim with uv)
- `.dockerignore` (49 lines)
**Actual Effort**: 2 hours
**Dependencies**: T001-T007
**References**: research.md:188-231
**Commit**: e44ff21

### T068: Write End-to-End Smoke Test [P] ✅ COMPLETED
**Description**: Complete workflow test covering all components.
**Acceptance**:
- ✅ Start with empty playbook
- ✅ Execute 5 tasks end-to-end (generate → reflect → curate)
- ✅ Verify playbook contains ≥3 bullets
- ✅ Verify bullet metadata (embedding dimension 384)
- ✅ Verify retrieval filters by stage (SHADOW vs PROD)
- ✅ Test semantic deduplication at 0.8 threshold
- ✅ All 3 E2E tests passing (100% pass rate)
**Deliverables**: `tests/e2e/test_smoke.py` (264 lines)
**Actual Effort**: 2.5 hours
**Dependencies**: T020-T064
**Commit**: e44ff21

**Bug Fixes in Phase 9**:
- ✅ Fixed FAISS integration in semantic_curator.py:
  - Replaced `build_index()` with `add_vectors(domain_id, embeddings, bullet_ids)`
  - Replaced `search_similar()` with `search(domain_id, query_embedding, k)`
  - Convert embedding lists to numpy arrays before FAISS calls
- ✅ Updated E2E tests to use correct repository methods

**Test Results**:
```
======================== 3 passed, 12 warnings in 7.84s ========================
✅ test_complete_workflow: PASSED
✅ test_deduplication_works: PASSED
✅ test_retrieval_excludes_shadow_bullets: PASSED
```

**Coverage**: 45% overall, semantic_curator.py: 45%, faiss_index.py: 70%

---

## Dependency Graph

### Critical Path (Blocking Dependencies)
```
T001 (Project Setup)
  → T002 (pyproject.toml)
    → T003 (Directory Structure)
      → T004 (SQLite Schema) → T010 (CRUD) → T020 (Retrieval) → ...
      → T008 (PlaybookBullet) → T009 (Playbook) → ...
      → T012 (Embeddings) → T013 (FAISS) → T029 (Curator) → ...
```

### Parallel Execution Opportunities

**Phase 1 (Setup)**: T005, T006, T007 can run in parallel after T002

**Phase 2 (Foundation)**:
- T008-T009 (Models) in parallel
- T012-T014 (Embeddings) in parallel
- T015-T017 (Additional models) in parallel
- T018 (Signatures) in parallel
- T019 (Tests) after models complete

**Phase 3 (US1)**:
- T023, T024, T025, T026, T027, T028 can run in parallel after T020-T022 complete

**Phase 4 (US2)**:
- T033, T034, T035, T036 can run in parallel after T029-T032 complete

**Phase 5 (US3)**:
- T042, T043 can run in parallel after T039-T041 complete

**Phase 6 (US4)**:
- T050, T051 can run in parallel after T047-T049 complete

**Phase 7 (US5)**:
- T055, T056, T057 can run in parallel after T052-T054 complete

**Phase 8 (US6)**:
- T062, T063, T064 can run in parallel after T058-T061 complete

**Phase 9 (Polish)**:
- T065, T066, T067, T068 can all run in parallel

**Maximum Parallelism**: Up to 8 tasks can run concurrently in later phases (e.g., Phase 9).

---

## MVP Scope Recommendation

**Minimum viable implementation** to validate core hypothesis (accumulating knowledge without fine-tuning):

**Must-Have (MVP)**: Phases 1-4 (T001-T036)
- Project setup and foundational models
- US1: Basic playbook accumulation
- US2: Semantic deduplication
- 36 tasks, ~45 hours estimated effort

**Should-Have (Beta)**: Add Phase 5-6 (T037-T051)
- US3: Reasoning traces
- US4: Feedback learning
- +16 tasks, +23 hours

**Nice-to-Have (v1.0)**: Add Phase 7-9 (T052-T068)
- US5: Offline pre-training
- US6: Online adaptation with safety
- Polish & observability
- +17 tasks, +32 hours

**Total Effort Estimate**: ~100 hours for full v1.0 release

---

## Success Metrics

**Phase 3 Complete (US1)**:
- Playbook grows from 0 → ≥5 bullets after 10 tasks
- Bullets persist across runs (SQLite storage)
- Independent test: `pytest tests/integration/test_playbook_growth.py`

**Phase 4 Complete (US2)**:
- 50 tasks → 0 duplicate bullets (all similar bullets merged)
- Diff journal shows 0 "update" operations (append-only guarantee)
- Independent test: `pytest tests/integration/test_append_only.py`

**Phase 5 Complete (US3)**:
- Reasoning traces show ≥3 steps with strategy references
- Bullets_referenced list populated with ≥1 ID
- Independent test: `pytest tests/integration/test_reasoning_traces.py`

**Phase 6 Complete (US4)**:
- Insights labeled as Helpful/Harmful based on ground-truth
- No manual annotation required
- Independent test: `pytest tests/integration/test_feedback_learning.py`

**Phase 7 Complete (US5)**:
- 100 examples → ≥20 strategies
- Validation accuracy ≥20% higher than baseline
- Independent test: `ace offline-train --dataset gsm8k --num_examples 100`

**Phase 8 Complete (US6)**:
- 100 iterations → ≥10 shadow insights
- Promotion gates enforced (helpful≥3, ratio≥3:1)
- Independent test: `pytest tests/integration/test_shadow_learning.py`

---

## Notes

1. **Test-Driven Development**: Each user story phase includes integration tests that validate acceptance criteria independently.

2. **Parallelization**: Mark tasks with `[P]` to indicate they can run concurrently. Maximum of 8 parallel tasks in later phases.

3. **Incremental Delivery**: Each phase delivers a testable user story. Can ship MVP (Phases 1-4) and iterate.

4. **Performance Validation**: Include performance benchmarks (T028, T057) to verify latency SLOs (≤10ms retrieval, ≤700ms P50 E2E).

5. **Optional Enhancements**: Tasks marked `[OPTIONAL]` provide additional value but not required for core functionality (e.g., GEPA optimization, consolidation trigger).

6. **Dependency Management**: Critical path focuses on blocking dependencies. Non-blocking tasks can be deferred or parallelized.

7. **Technology Choices**: All tasks assume Python 3.11, DSPy, FAISS, SQLite per research.md decisions. No alternative implementations needed for MVP.

---

## Phase 10: Production Hardening - Priority 1 (BLOCKING) (6 tasks)

**Goal**: Address critical gaps that block production deployment identified in senior engineering review.
**Status**: 🔴 NOT STARTED
**Priority**: P1 (BLOCKING - Must complete before production)
**Timeline**: 1-2 weeks

### T069: Add Performance Tests to Verify P50 ≤10ms Claim [BLOCKING]
**Description**: Implement performance benchmarks to validate playbook retrieval latency claims.
**Acceptance**:
- Create `tests/benchmarks/test_retrieval_perf.py` with pytest-benchmark
- Test: 100 bullets → measure `get_top_k()` P50/P95 latency
- Test: 100 bullets → measure `apply_delta()` latency
- Assert: retrieval P50 ≤10ms, P95 ≤25ms
- Assert: curator P50 ≤50ms
- Generate HTML benchmark report
**Deliverables**: `tests/benchmarks/test_retrieval_perf.py` (150 lines)
**Estimated Effort**: 3 hours
**Dependencies**: T020, T032
**References**: docs/ENGINEERING_REVIEW.md:341-363

### T070: Fix Input Validation in batch_merge() [BLOCKING]
**Description**: Add comprehensive input validation to prevent runtime errors.
**Acceptance**:
- Validate `task_insights` not empty before accessing `[0]`
- Validate all tasks have same `domain_id` (reject mixed batches)
- Call `validate_domain_id()` on extracted domain
- Remove unsafe `"default"` fallback that violates isolation
- Add schema validation for insight dict structure
- Return clear error messages with context
**Deliverables**: Updated `ace/curator/semantic_curator.py:430-460`
**Estimated Effort**: 2 hours
**Dependencies**: T032
**References**: docs/ENGINEERING_REVIEW.md:146-175

### T071: Add Circuit Breaker for OpenAI/Anthropic Calls [BLOCKING]
**Description**: Implement circuit breaker pattern to prevent cascading failures.
**Acceptance**:
- Install `circuitbreaker` dependency
- Wrap all LLM API calls with `@circuit` decorator
- Configure: `failure_threshold=5`, `recovery_timeout=60`
- Circuit opens after 5 consecutive failures
- Circuit half-opens after 60 seconds to test recovery
- Log circuit state changes
- Add metric: `circuit_breaker_state{provider}` (open/closed/half_open)
**Deliverables**: `ace/core/circuit_breaker.py` (80 lines)
**Estimated Effort**: 2.5 hours
**Dependencies**: T039, T047
**References**: docs/ENGINEERING_REVIEW.md:247-256

### T072: Fix FAISS Resource Leak in Batch Operations [BLOCKING]
**Description**: Prevent memory leak from unbounded FAISS index growth.
**Acceptance**:
- Add `clear_index(domain_id)` method to FAISSIndexManager
- Call cleanup in `finally` block after batch operations
- Implement context manager for FAISS operations
- Add metric: `faiss_index_size_bytes{domain_id}`
- Document index lifecycle in docstrings
**Deliverables**: Updated `ace/curator/semantic_curator.py:499`, `ace/utils/faiss_index.py`
**Estimated Effort**: 2 hours
**Dependencies**: T013, T032
**References**: docs/ENGINEERING_REVIEW.md:184-199

### T073: Document Rollback Procedures in RUNBOOK.md [BLOCKING]
**Description**: Create operational runbook with emergency procedures.
**Acceptance**:
- Create `docs/RUNBOOK.md` with incident response procedures
- Document: High latency → Check FAISS index size + consolidate
- Document: OOM errors → Reduce batch size in config
- Document: Failed promotions → Check promotion gate thresholds
- Document: Emergency rollback (≤5 min):
  1. `git revert <commit>`
  2. `alembic downgrade -1`
  3. `docker-compose restart ace`
  4. Verify health check passes
- Document: Playbook restoration from backup
**Deliverables**: `docs/RUNBOOK.md` (200 lines)
**Estimated Effort**: 2 hours
**Dependencies**: None
**References**: docs/ENGINEERING_REVIEW.md:564-575

### T074: Add Rate Limiting for LLM API Calls [BLOCKING]
**Description**: Implement rate limiting to prevent API quota exhaustion.
**Acceptance**:
- Install `ratelimit` dependency
- Wrap Generator/Reflector with `@sleep_and_retry` + `@limits`
- Configure: 10 calls per minute per provider
- Add per-domain rate limits (100 calls/hour)
- Queue requests when limit reached (backpressure)
- Add metric: `rate_limit_exceeded_total{provider,domain_id}`
- Log rate limit events with backoff time
**Deliverables**: `ace/core/rate_limiter.py` (120 lines)
**Estimated Effort**: 2.5 hours
**Dependencies**: T039, T047
**References**: docs/ENGINEERING_REVIEW.md:258-266

---

## Phase 11: Production Quality - Priority 2 (HIGH) (4 tasks)

**Goal**: Improve production quality and observability for reliable operations.
**Status**: 🔴 NOT STARTED
**Priority**: P2 (Required for production)
**Timeline**: 2-4 weeks

### T075: Add Dependency Security Scanning Hook [P]
**Description**: Automated vulnerability scanning for Python dependencies.
**Acceptance**:
- Add Safety hook to `.pre-commit-config.yaml`
- Scan on push stage (not every commit)
- Fail build on known CVEs (CRITICAL/HIGH severity)
- Generate vulnerability report: `safety check --json > security-report.json`
- Document remediation process in CONTRIBUTING.md
- Add CI job: security scan on pull requests
**Deliverables**: Updated `.pre-commit-config.yaml`, security docs
**Estimated Effort**: 1.5 hours
**Dependencies**: T007 (already completed)
**References**: docs/ENGINEERING_REVIEW.md:100-108

### T076: Implement Distributed Tracing with OpenTelemetry [P]
**Description**: End-to-end request tracing across all components.
**Acceptance**:
- Install `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-jaeger`
- Instrument all service calls with spans
- Trace: Generator → Reflector → Curator pipeline
- Trace: FAISS search operations
- Trace: Database queries
- Export to Jaeger for visualization
- Add trace ID to structured logs
- Capture: latency, errors, dependencies
**Deliverables**: `ace/ops/tracing.py` (150 lines)
**Estimated Effort**: 4 hours
**Dependencies**: T039, T047, T032
**References**: docs/ENGINEERING_REVIEW.md:466-476

### T077: Create Architecture Decision Records (ADRs) [P]
**Description**: Document key architectural decisions and rationale.
**Acceptance**:
- Create `docs/adr/` directory
- ADR-001: Choose FAISS over ChromaDB
  - Context: Need fast semantic search
  - Decision: FAISS for sub-10ms P50
  - Consequences: No built-in persistence
- ADR-002: Append-only playbook design
  - Context: Prevent context collapse
  - Decision: Never rewrite content, only increment counters
  - Consequences: Diff journal for audit
- ADR-003: Domain isolation strategy
  - Context: Multi-tenancy requirements
  - Decision: Per-domain FAISS indices
  - Consequences: Memory overhead
- Follow ADR template (status, context, decision, consequences)
**Deliverables**: `docs/adr/001-faiss.md`, `002-append-only.md`, `003-domain-isolation.md`
**Estimated Effort**: 3 hours
**Dependencies**: None
**References**: docs/ENGINEERING_REVIEW.md:507-513

### T078: Add Health Check Endpoint [P]
**Description**: HTTP health check for orchestration and monitoring.
**Acceptance**:
- Create `/health` endpoint
- Check: FAISS manager ready
- Check: Database connection alive
- Check: Embedding service loaded
- Return: `{"status": "healthy", "checks": {...}, "timestamp": "..."}`
- Return 503 if any check fails
- Add metric: `health_check_status{component}` (0/1)
- Document in API docs
**Deliverables**: `ace/api/health.py` (80 lines)
**Estimated Effort**: 1.5 hours
**Dependencies**: T012, T013, T010
**References**: docs/ENGINEERING_REVIEW.md:478-488

---

## Phase 12: Maintainability Improvements - Priority 3 (MEDIUM) (4 tasks)

**Goal**: Improve code maintainability and test robustness.
**Status**: 🔴 NOT STARTED
**Priority**: P3 (Quality improvements)
**Timeline**: 1-2 months

### T079: Refactor semantic_curator.py into Modules [P]
**Description**: Split large file into focused modules for maintainability.
**Acceptance**:
- Extract `BatchProcessor` class → `ace/curator/batch_processor.py`
- Extract `SimilarityEngine` class → `ace/curator/similarity_engine.py`
- Extract `DomainValidator` class → `ace/curator/domain_validator.py`
- Keep core `SemanticCurator` logic in `semantic_curator.py`
- Maintain 100% backward compatibility (public API unchanged)
- All existing tests pass without modification
- Target: each file ≤300 lines
**Deliverables**: 4 new files, updated `semantic_curator.py`
**Estimated Effort**: 4 hours
**Dependencies**: T032
**References**: docs/ENGINEERING_REVIEW.md:534-545

### T080: Add Property-Based Testing with Hypothesis [P]
**Description**: Fuzzy testing for edge cases in curator logic.
**Acceptance**:
- Install `hypothesis` dependency
- Test: `batch_merge()` handles any list size (0-1000 insights)
- Test: `batch_merge()` handles any content (unicode, special chars)
- Test: similarity threshold edge cases (0.799, 0.800, 0.801)
- Test: counter overflow scenarios (helpful_count > 1M)
- Generate 100+ test cases automatically
- Catch edge cases not covered by unit tests
**Deliverables**: `tests/property/test_curator_properties.py` (200 lines)
**Estimated Effort**: 3 hours
**Dependencies**: T032
**References**: docs/ENGINEERING_REVIEW.md:426-434

### T081: Add Mutation Testing with mutpy [P] [OPTIONAL]
**Description**: Verify tests actually catch bugs (test the tests).
**Acceptance**:
- Install `mutpy` dependency
- Run mutation testing on `ace/curator/` module
- Generate mutations: change operators, swap conditions, remove lines
- Measure: mutation score (% of mutations caught by tests)
- Target: mutation score ≥70%
- Add CI job: mutation testing on pull requests
- Document in CONTRIBUTING.md
**Deliverables**: Mutation testing CI job, report
**Estimated Effort**: 2 hours
**Dependencies**: T033-T035
**References**: docs/ENGINEERING_REVIEW.md:436-441

### T082: Add Complexity Monitoring Hook [P]
**Description**: Prevent code complexity from increasing over time.
**Acceptance**:
- Add Radon complexity hook to `.pre-commit-config.yaml`
- Check cyclomatic complexity on push stage
- Fail if any function has complexity grade C or worse
- Fail if maintainability index <B grade
- Generate complexity report: `radon cc ace/ --json > complexity.json`
- Document refactoring guidelines in CONTRIBUTING.md
**Deliverables**: Updated `.pre-commit-config.yaml`, complexity docs
**Estimated Effort**: 1 hour
**Dependencies**: T007 (already completed)
**References**: docs/ENGINEERING_REVIEW.md:81-86

---

## Phase 13: Comprehensive Documentation (FINAL) (6 tasks)

**Goal**: Generate complete, accessible documentation for the entire codebase for new developers.
**Status**: 🔴 NOT STARTED
**Priority**: P2 (Required for onboarding)
**Timeline**: 1 week

### T083: Generate API Reference Documentation [BLOCKING]
**Description**: Auto-generate comprehensive API documentation from docstrings.
**Acceptance**:
- Install `sphinx` and `sphinx-autodoc` for documentation generation
- Configure Sphinx with `docs/conf.py`
- Generate API docs for all public modules:
  - `ace.generator` (CoTGenerator, ReActGenerator)
  - `ace.reflector` (GroundedReflector)
  - `ace.curator` (SemanticCurator, BatchProcessor)
  - `ace.models` (PlaybookBullet, Task, Reflection)
  - `ace.repositories` (PlaybookRepository, JournalRepository)
  - `ace.utils` (EmbeddingService, FAISSIndexManager)
  - `ace.ops` (MetricsCollector, GuardrailMonitor)
- Include: purpose, parameters, return values, exceptions, examples
- Generate HTML docs: `docs/_build/html/`
- Host on GitHub Pages or ReadTheDocs
**Deliverables**: `docs/conf.py`, `docs/api/`, generated HTML docs
**Estimated Effort**: 4 hours
**Dependencies**: T001-T082 (all implementation complete)
**References**: User request for comprehensive documentation

### T084: Write Architecture Overview Documentation [BLOCKING]
**Description**: High-level architecture guide for new developers.
**Acceptance**:
- Create `docs/ARCHITECTURE.md` with system design
- Document Generator-Reflector-Curator pattern with diagrams
- Explain data flow: Task → Generator → Reflector → Curator → Playbook
- Document domain isolation architecture (per-tenant namespaces)
- Document staged rollout: SHADOW → STAGING → PROD
- Include sequence diagrams (using Mermaid):
  - Online learning loop
  - Offline training workflow
  - Promotion gate logic
- Document performance budgets and SLOs
- Explain append-only design and diff journal
**Deliverables**: `docs/ARCHITECTURE.md` (500+ lines with diagrams)
**Estimated Effort**: 4 hours
**Dependencies**: T083
**References**: plan.md, data-model.md, docs/adr/

### T085: Write Developer Onboarding Guide [P]
**Description**: Step-by-step guide for new contributors.
**Acceptance**:
- Create `docs/ONBOARDING.md` for new developers
- Section: Development environment setup (uv, Python 3.11, Docker)
- Section: Code structure walkthrough (what each module does)
- Section: Running tests locally (`pytest`, coverage reports)
- Section: Common workflows:
  - Adding a new playbook bullet type
  - Implementing a custom Reflector
  - Adding a new metric
  - Debugging FAISS issues
- Section: Troubleshooting guide (common errors + fixes)
- Section: How to add a new feature (TDD workflow)
- Include code examples for each workflow
**Deliverables**: `docs/ONBOARDING.md` (400+ lines)
**Estimated Effort**: 3 hours
**Dependencies**: T083, T084
**References**: CONTRIBUTING.md, README.md

### T086: Document All Edge Cases and Error Handling [P]
**Description**: Comprehensive edge case documentation for operational safety.
**Acceptance**:
- Create `docs/EDGE_CASES.md`
- Document edge cases per module:
  - **Curator**: Empty insights list, domain_id mismatch, batch size = 0
  - **FAISS**: Zero vectors, dimension mismatch, index not initialized
  - **Embeddings**: Unicode text, empty strings, text > 500 chars
  - **Promotion Gates**: Zero counters, division by zero, ratio edge cases
  - **Circuit Breaker**: Timeout vs connection error, half-open state
- Document error handling patterns:
  - ValueError for invalid input
  - RuntimeError for system failures
  - Custom exceptions (DomainIsolationError, DeduplicationError)
- Include recovery procedures for each error type
- Cross-reference with RUNBOOK.md
**Deliverables**: `docs/EDGE_CASES.md` (300+ lines)
**Estimated Effort**: 3 hours
**Dependencies**: T083
**References**: semantic_curator.py error handling

### T087: Create Interactive Usage Examples and Tutorials [P]
**Description**: Practical tutorials for common use cases.
**Acceptance**:
- Create `docs/tutorials/` directory
- Tutorial 1: `01-quick-start.md` (10 min read)
  - Bootstrap empty playbook
  - Run 5 tasks end-to-end
  - Inspect playbook growth
- Tutorial 2: `02-offline-training.md` (20 min)
  - Load GSM8K dataset
  - Pre-train on 100 examples
  - Evaluate accuracy improvement
- Tutorial 3: `03-domain-isolation.md` (15 min)
  - Create multiple domains (customer-acme, customer-beta)
  - Add domain-specific bullets
  - Verify isolation (no cross-domain retrieval)
- Tutorial 4: `04-shadow-promotion.md` (20 min)
  - Deploy new strategies to shadow
  - Monitor helpful/harmful counters
  - Promote to staging → prod
- Include: full code examples, expected output, verification commands
- Add Jupyter notebooks: `examples/tutorials/*.ipynb`
**Deliverables**: `docs/tutorials/*.md`, `examples/tutorials/*.ipynb`
**Estimated Effort**: 5 hours
**Dependencies**: T083, T084
**References**: examples/arithmetic_learning.py

### T088: Generate Comprehensive Docstrings for All Public APIs [P]
**Description**: Ensure 100% docstring coverage for all public functions/classes.
**Acceptance**:
- Use `interrogate` to measure current docstring coverage
- Target: 100% coverage (currently ~80%)
- Add docstrings following Google style guide:
  ```python
  def apply_delta(self, curator_input: CuratorInput) -> CuratorOutput:
      """Merge new insights into playbook with semantic deduplication.

      This method performs the core curation logic: compute embeddings for
      insights, search FAISS for duplicates (≥0.8 cosine similarity), and
      either increment existing bullet counters or add new bullets.

      Args:
          curator_input: Contains insights from Reflector and current playbook.
              Must have valid domain_id for isolation.

      Returns:
          CuratorOutput with updated playbook, delta statistics, and
          diff journal entries for audit trail.

      Raises:
          ValueError: If insights list is empty or domain_id invalid.
          RuntimeError: If FAISS index fails to initialize.

      Example:
          >>> curator = SemanticCurator(embedding_service, faiss_manager)
          >>> output = curator.apply_delta(curator_input)
          >>> print(f"Added {output.stats['new_bullets']} bullets")

      Note:
          This method is thread-safe and uses atomic transactions.
          All bullets start in SHADOW stage for safety.
      """
  ```
- Include: purpose, parameters, return values, exceptions, examples, notes
- Document architectural context (how component fits in system)
- Add type hints to all public APIs (already done via mypy)
**Deliverables**: Updated docstrings in all `ace/` modules
**Estimated Effort**: 6 hours
**Dependencies**: T083
**References**: interrogate configuration in pyproject.toml

---

## Post-Implementation Summary

**Total Implementation**: 88 tasks across 13 phases
- **Phase 1-2**: Project setup and foundation (19 tasks)
- **Phase 3-8**: Core user stories (38 tasks)
- **Phase 9**: Production polish (4 tasks) ✅ COMPLETED
- **Phase 10-12**: Production hardening from review (14 tasks)
- **Phase 13**: Comprehensive documentation (6 tasks)

**Effort Estimates**:
- Core MVP (Phase 1-4): ~45 hours
- Full v1.0 (Phase 1-9): ~100 hours
- Production-ready (Phase 1-12): ~125 hours
- Documentation complete (Phase 1-13): ~150 hours total

**Completion Criteria**:
- ✅ All 88 tasks completed
- ✅ Test coverage ≥80%
- ✅ Performance benchmarks passing (P50 ≤10ms)
- ✅ Security scans passing (no CRITICAL/HIGH CVEs)
- ✅ Documentation generated (API + guides + tutorials)
- ✅ Pre-commit hooks enforcing quality
- ✅ Production monitoring in place
- ✅ Rollback procedures documented

**Deployment Readiness**:
After Phase 13, the system is ready for:
- Production deployment with confidence
- New developer onboarding (≤1 day ramp-up)
- Customer documentation and demos
- Open source release (if applicable)
