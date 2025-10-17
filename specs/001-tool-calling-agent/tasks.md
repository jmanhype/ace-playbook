# Tasks: Tool-Calling Agent with ReAct Reasoning

**Input**: Design documents from `/specs/001-tool-calling-agent/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included per Constitution Principle II (Reflection-Based Testing)

**Organization**: Tasks grouped by user story for independent implementation and testing

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and tool-calling infrastructure

- [X] T001 Create playbook storage directory `.specify/memory/playbooks/tool-calling/` with subdirectories for `rag-database/` and `general/`
- [X] T002 [P] Install DSPy dependency for ReAct module support (`pip install dspy`)
- [X] T003 [P] Configure pytest for tool validation and performance testing
- [X] T004 [P] Create examples directory `examples/` for RAG agent demonstrations

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 Extend PlaybookBullet dataclass in `ace/playbook/playbook.py` with optional fields: `tool_sequence: Optional[List[str]]`, `tool_success_rate: Optional[float]`, `avg_iterations: Optional[int]`
- [X] T006 [P] Create ReasoningStep dataclass in `ace/generator/signatures.py` with fields: iteration, thought, action, tool_name, tool_args, observation, timestamp, duration_ms
- [X] T007 [P] Extend TaskInput dataclass in `ace/generator/signatures.py` with optional fields: `available_tools: Optional[List[str]]`, `max_iterations: Optional[int]`
- [X] T008 Extend TaskOutput dataclass in `ace/generator/signatures.py` with fields: `structured_trace: List[ReasoningStep]`, `tools_used: List[str]`, `total_iterations: int`, `iteration_limit_reached: bool`
- [X] T009 [P] Create tool validation utility function `validate_tool()` in `ace/generator/react_generator.py` using `inspect` module to check signatures, type annotations, and docstrings
- [X] T010 [P] Create custom exceptions in `ace/generator/react_generator.py`: ToolValidationError, ToolExecutionError, DuplicateToolError, ToolNotFoundError, MaxIterationsExceededError

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - RAG Application with Database Tools (Priority: P1) 🎯 MVP

**Goal**: Enable RAG developers to create agents that query multiple databases iteratively, learning which retrieval patterns work best over time

**Independent Test**: Provide a multi-database query scenario and verify the agent successfully retrieves information through iterative tool calls with 30% iteration reduction after learning

### Tests for User Story 1 (Red-Green-Refactor-Reflect)

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T011 [P] [US1] Unit test for tool validation in `tests/unit/test_tool_validation.py` - verify valid/invalid tool signatures are correctly identified
- [X] T012 [P] [US1] Unit test for ReActGenerator initialization in `tests/unit/test_react_generator.py` - verify tools are registered, max_iters configured, and DSPy ReAct module initialized
- [X] T013 [P] [US1] Integration test for ReAct + Reflector in `tests/integration/test_react_with_reflector.py` - verify tool usage patterns are extracted and analyzed
- [X] T014 [P] [US1] Integration test for full ACE cycle in `tests/integration/test_tool_calling_workflow.py` - verify Generator → Reflector → Curator → Playbook flow with tool strategies

### Implementation for User Story 1

- [X] T015 [US1] Implement ReActGenerator class in `ace/generator/react_generator.py` with __init__ method: accept tools list, model, max_iters; validate tools; initialize dspy.ReAct module
- [X] T016 [US1] Implement ReActGenerator.forward() in `ace/generator/react_generator.py`: retrieve playbook strategies, format as context, inject into DSPy ReAct, execute with hybrid max_iters (task > agent > system default 10)
- [X] T017 [US1] Implement ReActGenerator.register_tool() in `ace/generator/react_generator.py`: validate new tool, check for duplicates, add to tools list
- [X] T018 [US1] Implement ReActGenerator.validate_tools() in `ace/generator/react_generator.py`: return list of validation errors for all registered tools
- [X] T019 [US1] Implement tool execution wrapper with timeout/error handling in `ace/generator/react_generator.py`: catch timeouts (<30s), exceptions, format observations
- [X] T020 [US1] Implement structured trace generation in `ace/generator/react_generator.py`: convert DSPy ReAct iterations to List[ReasoningStep] with timing metadata
- [X] T021 [US1] Update `ace/generator/__init__.py` to export ReActGenerator class and related types
- [X] T022 [US1] Extend GroundedReflector in `ace/reflector/grounded_reflector.py`: add tool usage analysis method to extract tool sequences from TaskOutput.structured_trace and create ToolCallingStrategy bullets
- [X] T023 [US1] Extend SemanticCurator in `ace/curator/semantic_curator.py`: handle ToolCallingStrategy bullets with tool_sequence metadata, deduplicate by both content and tool sequence similarity (≥0.8)
- [X] T024 [US1] Implement playbook context injection in ReActGenerator.forward(): retrieve strategies with filter `{"has_tool_sequence": True}`, format as bullet list, inject into DSPy prompt
- [X] T025 [US1] Create example RAG agent in `examples/react_rag_agent.py`: demonstrate database tools (search_vector_db, search_sql_db, rank_results), show learning over 20+ tasks, validate 30% iteration reduction

**Checkpoint**: User Story 1 (RAG Database Agent) fully functional - agent queries databases iteratively, learns strategies, reduces iterations by 30-50%

---

## Phase 4: User Story 2 - Multi-Tool Workflow Agent (Priority: P2)

**Goal**: Enable agents to orchestrate multiple heterogeneous tools (APIs, calculators, search engines) to complete complex multi-step tasks

**Independent Test**: Provide a task requiring 3+ different tool types (e.g., search → calculate → format) and verify successful completion with appropriate tool sequencing

### Tests for User Story 2

- [X] T026 [P] [US2] Integration test for multi-tool orchestration in `tests/integration/test_multi_tool_workflow.py` - verify agent selects appropriate tools from heterogeneous set (API, calculator, formatter)
- [X] T027 [P] [US2] Integration test for tool failure handling in `tests/integration/test_multi_tool_workflow.py` - verify agent adapts strategy when tools return errors or unexpected results

### Implementation for User Story 2

- [X] T028 [P] [US2] Implement graceful degradation logic in `ace/generator/react_generator.py`: track failed tools, exclude from available_tools after 3+ failures, capture fallback patterns
- [X] T029 [P] [US2] Add error context formatting in `ace/generator/react_generator.py`: include tool name, error type, parameters attempted, suggestions for alternatives
- [X] T030 [US2] Extend Reflector tool analysis in `ace/reflector/grounded_reflector.py`: identify tool failure patterns (Harmful bullets), track tool adaptations (when agent switches tools after error)
- [X] T031 [US2] Add tool reliability metrics to ToolCallingStrategy in playbook: track tool success/failure rates, average execution time, recommend tools by reliability
- [X] T032 [US2] Create multi-tool workflow example in `examples/multi_tool_orchestration.py`: demonstrate API + calculator + formatter tools, show error handling and adaptation

**Checkpoint**: User Story 2 (Multi-Tool Orchestration) functional - agent handles heterogeneous tools, adapts to failures, learns tool reliability patterns

---

## Phase 5: User Story 3 - Tool Usage Learning and Optimization (Priority: P3)

**Goal**: Enable analysis of tool usage pattern evolution, understanding which tool combinations prove most effective for different task types

**Independent Test**: Run batch of similar tasks and verify playbook captures tool usage patterns with success metrics, strategies are observable and reusable

### Tests for User Story 3

- [X] T033 [P] [US3] Integration test for playbook strategy retrieval in `tests/integration/test_tool_strategy_learning.py` - verify strategies are organized by domain with success metrics after 100+ tasks
- [X] T034 [P] [US3] Integration test for strategy reuse in `tests/integration/test_tool_strategy_learning.py` - verify new tasks in learned domains use proven patterns from playbook (initial tool selections match strategies)

### Implementation for User Story 3

- [X] T035 [P] [US3] Implement LRU cache for playbook retrieval in `ace/generator/react_generator.py`: add @lru_cache(maxsize=128) to _get_tool_strategies method, clear cache on playbook updates
- [X] T036 [P] [US3] Add tool performance tracking to ReActGenerator: measure tool call overhead, playbook retrieval latency, total task duration; expose via TaskOutput.metadata
- [X] T037 [US3] Implement strategy effectiveness scoring in `ace/curator/semantic_curator.py`: calculate tool_success_rate from helpful_count/harmful_count, prioritize high-success strategies in retrieval
- [X] T038 [US3] Add domain-specific strategy organization in playbook: create separate indices for different domains, enable cross-domain pattern transfer for related tasks
- [X] T039 [US3] Create batch processing example in `examples/batch_tool_learning.py`: process 100+ tasks, track strategy evolution, visualize iteration reduction over time, demonstrate playbook analytics

**Checkpoint**: User Story 3 (Tool Learning & Optimization) functional - strategies organized by domain, success metrics tracked, cross-domain learning enabled

---

## Phase 6: Performance & Testing

**Purpose**: Validate performance budgets and comprehensive testing coverage

- [X] T040 [P] Performance benchmark for tool call overhead in `tests/performance/test_react_performance.py` - measure and assert <100ms overhead per iteration (excluding tool execution)
- [X] T041 [P] Performance benchmark for playbook retrieval in `tests/performance/test_react_performance.py` - measure and assert <10ms P50 latency for strategy retrieval
- [X] T042 [P] Performance benchmark for agent initialization in `tests/performance/test_react_performance.py` - measure and assert <500ms init time with 10-50 tools
- [X] T043 [P] End-to-end performance test in `tests/performance/test_react_performance.py` - measure total task completion for RAG queries, assert <10s for 95% of queries (SC-004)
- [X] T044 [P] Scaling test in `tests/performance/test_react_performance.py` - verify agent handles 10, 25, 50 tools without degradation
- [X] T045 [P] Create unit tests for reasoning trace generation in `tests/unit/test_react_generator.py` - verify structured_trace accuracy, timing metadata, iteration counting
- [X] T046 [P] Create unit tests for max iterations configuration in `tests/unit/test_react_generator.py` - verify hybrid override (task > agent > system default 10)
- [X] T047 [P] Add playbook integration tests in `tests/integration/test_tool_calling_workflow.py` - verify strategies are retrieved, deduplicated, and applied correctly

**Checkpoint**: All performance budgets validated, comprehensive test coverage achieved

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, examples, and cross-story improvements

- [X] T048 [P] Create comprehensive docstrings for all ReActGenerator methods in `ace/generator/react_generator.py` following ACE style guide
- [~] T049 [P] Add type hints and validation (partial) to all tool-related functions using mypy/pyright
- [X] T050 [P] Update README.md with ReActGenerator usage examples and quickstart link
- [X] T051 [P] Validate quickstart.md examples work end-to-end (30-minute setup time)
- [X] T052 Add backward compatibility tests in `tests/integration/test_backward_compat.py` - verify ReActGenerator works as drop-in replacement for CoTGenerator when no tools provided
- [X] T053 Create migration guide in `docs/MIGRATION_COT_TO_REACT.md` for existing CoTGenerator users
- [ ] T054 [P] Consolidate and deduplicate playbook bullets in `.specify/memory/playbooks/tool-calling/` using semantic similarity (≥0.8 threshold)
- [ ] T055 [P] Review playbook effectiveness metrics (helpful_count vs harmful_count) and prune stale bullets (helpful=0, harmful=0 after 10+ iterations)
- [ ] T056 Commit playbook updates with descriptive messages following convention: `playbook: add [category] strategy for [context]`
- [X] T057 [P] Add logging for all tool executions with structured context (tool name, args, result, duration) for debugging
- [X] T058 [P] Implement playbook archaeology feature: add attribution metadata (which task/domain generated each bullet) for traceability

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational completion
  - US1 (P1): No dependencies on other stories
  - US2 (P2): Can start in parallel with US1, independent testability
  - US3 (P3): Can start in parallel with US1/US2, independent testability
- **Performance & Testing (Phase 6)**: Depends on US1 minimum (can test incrementally as stories complete)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1 - RAG Agent)**: Independent - only needs Foundational
- **User Story 2 (P2 - Multi-Tool)**: Independent - only needs Foundational (builds on US1 concepts but separately testable)
- **User Story 3 (P3 - Learning)**: Independent - only needs Foundational (analyzes patterns from US1/US2 but separately testable)

### Within Each User Story

**TDD Order** (Red-Green-Refactor-Reflect):
1. Tests FIRST - ensure they FAIL
2. Data structures (dataclasses, signatures)
3. Core implementation (Generator, Reflector, Curator extensions)
4. Integration and examples
5. Reflect on test outcomes, update playbook

### Parallel Opportunities

**Setup (Phase 1)**: All tasks [P] can run in parallel

**Foundational (Phase 2)**: T005-T010 all [P] within their type:
- T005-T008: Data structure changes (4 parallel)
- T009-T010: Validation/exceptions (2 parallel)

**User Story 1 Tests**: T011-T014 all [P] (4 tests in parallel)

**User Story 1 Implementation**:
- T015-T018: Core ReActGenerator methods (can parallelize per method)
- T022-T023: Reflector/Curator extensions [P] (2 parallel)

**User Stories**: US1, US2, US3 can proceed in parallel with different developers after Foundational complete

**Performance Tests**: T040-T047 all [P] (8 benchmarks in parallel)

**Polish**: T048-T051, T054-T055, T057-T058 all [P] (documentation, cleanup tasks)

---

## Parallel Example: User Story 1

```bash
# Step 1: Launch all tests for US1 together (TDD - these should FAIL initially):
Task T011: "Unit test for tool validation in tests/unit/test_tool_validation.py"
Task T012: "Unit test for ReActGenerator init in tests/unit/test_react_generator.py"
Task T013: "Integration test ReAct + Reflector in tests/integration/test_react_with_reflector.py"
Task T014: "Integration test full ACE cycle in tests/integration/test_tool_calling_workflow.py"

# Step 2: Launch core ReActGenerator methods in parallel:
Task T015: "Implement ReActGenerator.__init__ in ace/generator/react_generator.py"
Task T017: "Implement ReActGenerator.register_tool in ace/generator/react_generator.py"
Task T018: "Implement ReActGenerator.validate_tools in ace/generator/react_generator.py"

# Step 3: Launch Reflector and Curator extensions in parallel:
Task T022: "Extend GroundedReflector tool analysis in ace/reflector/grounded_reflector.py"
Task T023: "Extend SemanticCurator for tool strategies in ace/curator/semantic_curator.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. ✅ Complete Phase 1: Setup (4 tasks)
2. ✅ Complete Phase 2: Foundational (6 tasks) - CRITICAL BLOCKING
3. ✅ Complete Phase 3: User Story 1 (15 tasks)
4. **STOP and VALIDATE**:
   - Run all US1 tests (T011-T014)
   - Execute RAG agent example (T025)
   - Verify 30% iteration reduction (SC-002)
   - Validate <10s query time (SC-004)
5. **Deploy/demo MVP** - fully functional RAG agent with learning

**MVP Scope**: 25 tasks total (Setup + Foundational + US1)
**Estimated Time**: 2-3 days

### Incremental Delivery

1. **Foundation** (Phase 1-2): 10 tasks → Setup complete
2. **MVP** (Phase 3): +15 tasks → RAG agent working → **Deploy**
3. **Multi-Tool** (Phase 4): +5 tasks → Error handling → **Deploy**
4. **Optimization** (Phase 5): +5 tasks → Analytics → **Deploy**
5. **Quality** (Phase 6-7): +18 tasks → Performance + Polish → **Production ready**

Each increment adds value without breaking previous functionality.

### Parallel Team Strategy

With 3 developers after Foundational complete:

1. **Team completes Setup + Foundational together** (Phases 1-2)
2. **Once Foundational done, parallel development**:
   - Developer A: User Story 1 (RAG Agent) - T011-T025
   - Developer B: User Story 2 (Multi-Tool) - T026-T032
   - Developer C: User Story 3 (Learning) - T033-T039
3. **Reconverge for Performance & Polish** (Phases 6-7)

---

## Success Criteria Mapping

Tasks mapped to success criteria from spec.md:

- **SC-001** (90% task completion): Validated by T014, T026, T043
- **SC-002** (30-50% iteration reduction): Validated by T014, T025, T034
- **SC-003** (Strategies captured within 3 tasks): Validated by T014, T022, T023
- **SC-004** (<10s RAG query time): Validated by T043
- **SC-005** (95% alternative solution success): Validated by T027, T028, T029
- **SC-006** (30-min developer setup): Validated by T051

---

## Task Statistics

**Total Tasks**: 58
**By Phase**:
- Phase 1 (Setup): 4 tasks
- Phase 2 (Foundational): 6 tasks
- Phase 3 (US1 - RAG Agent): 15 tasks (11 implementation + 4 tests)
- Phase 4 (US2 - Multi-Tool): 5 tasks (3 implementation + 2 tests)
- Phase 5 (US3 - Learning): 5 tasks (3 implementation + 2 tests)
- Phase 6 (Performance): 8 tasks (all tests)
- Phase 7 (Polish): 11 tasks (documentation, cleanup)

**By User Story**:
- US1 (P1 - RAG Agent): 15 tasks
- US2 (P2 - Multi-Tool): 5 tasks
- US3 (P3 - Learning): 5 tasks
- Shared/Infrastructure: 29 tasks

**Parallelization**:
- Setup: 3/4 tasks [P] (75%)
- Foundational: 5/6 tasks [P] (83%)
- US1: 7/15 tasks [P] (47%)
- US2: 4/5 tasks [P] (80%)
- US3: 4/5 tasks [P] (80%)
- Performance: 8/8 tasks [P] (100%)
- Polish: 8/11 tasks [P] (73%)

**MVP Scope (US1 only)**: 25 tasks (43% of total)

---

## Notes

- **[P] tasks** = different files, no dependencies, can parallelize
- **[Story] labels** (US1, US2, US3) map tasks to user stories for traceability
- **TDD approach**: Tests written FIRST (Red), then implementation (Green), then refactor with playbook guidance (Reflect)
- **Independent stories**: Each user story can be implemented and tested independently
- **Checkpoints**: Stop after each story phase to validate independently before proceeding
- **Constitution compliance**: All tasks follow ACE principles (playbook updates, reflection triggers, performance budgets)
- **Backward compatibility**: ReActGenerator extends CoTGenerator interface, no breaking changes
