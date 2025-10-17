# Implementation Plan: Tool-Calling Agent with ReAct Reasoning

**Branch**: `001-tool-calling-agent` | **Date**: October 16, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-tool-calling-agent/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement ReAct (Reasoning and Acting) agent support to enable tool-calling workflows where agents iteratively reason about tasks and execute tools to gather information. The system will integrate with ACE's existing playbook architecture to learn and optimize tool-calling strategies over time. Primary use case: RAG applications that need to query multiple databases iteratively, with the agent learning which retrieval patterns work best.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: DSPy (for ReAct module), existing ACE components (Reflector, Curator, Playbook)
**Storage**: FAISS (existing vector storage for playbook bullets), file-based playbook persistence
**Testing**: pytest (existing test infrastructure), DSPy assertions for agent validation
**Target Platform**: Linux/macOS (development), containerized deployment
**Project Type**: Single library/framework (extends existing ace/ package)
**Performance Goals**:
- Tool call overhead <100ms per iteration (excluding tool execution time)
- Playbook retrieval <10ms P50 for tool-calling strategies
- Agent initialization <500ms with 10-50 tools

**Constraints**:
- Must integrate with existing Generator-Reflector-Curator architecture
- Must maintain backward compatibility with CoTGenerator
- Tool functions must be synchronous (async support deferred)
- Maximum 100 iterations per agent execution (configurable with hybrid override)

**Scale/Scope**:
- Support 10-50 tools per agent instance
- Handle reasoning traces up to 100 iterations
- Playbook storage for 100+ domains with tool-calling strategies
- Initial target: 1-5 agent types (starting with RAG database query agent)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Reference: `.specify/memory/constitution.md`

**Principle I: Self-Improving Code Quality**
- [x] Component playbooks identified: `ReActGenerator` will maintain tool-calling strategy bullets; Reflector will analyze tool usage patterns
- [x] Code quality metrics defined: Track helpful tool sequences (successful task completion), harmful tool sequences (failures/timeouts), tool selection accuracy
- [x] Reflection triggers established: After each task completion (success/failure), after tool failures, when max iterations reached

**Principle II: Reflection-Based Testing**
- [x] Test playbooks planned: Unit tests (individual tool execution), integration tests (multi-tool workflows), contract tests (tool signature validation)
- [x] Execution feedback integration designed: Capture tool execution results, timing metrics, error messages as reflection inputs; test outcomes feed directly to Reflector
- [x] Red-Green-Refactor-Reflect cycle documented: Write failing test → implement ReActGenerator → refactor with playbook guidance → reflect on tool usage patterns

**Principle III: Context-Aware UX**
- [x] UX playbook reference points identified: Developer-facing API consistency (follows existing Generator pattern), error message clarity (tool failure explanations)
- [x] User feedback loops designed: Developer integration experience tracked, reasoning trace readability assessed, API usability feedback captured
- [x] Consistency validation approach defined: ReActGenerator follows same interface as CoTGenerator (TaskInput → TaskOutput), maintains ACE naming conventions

**Principle IV: Performance with Continuous Optimization**
- [x] Performance budgets established: <100ms tool call overhead, <10ms playbook retrieval P50, <500ms agent init, <10s total task completion (95%)
- [x] Performance playbook categories identified: Tool selection optimization, iteration reduction strategies, playbook retrieval caching, tool execution parallelization (future)
- [x] Automated performance testing planned: Benchmark suite measuring tool call overhead, playbook retrieval latency, end-to-end task completion time; failures trigger reflection

**Principle V: Modular Black-Box Architecture**
- [x] Generator-Reflector-Curator roles assigned:
  - **Generator**: ReActGenerator (produces tool-calling task outputs)
  - **Reflector**: GroundedReflector (analyzes tool usage, identifies successful patterns)
  - **Curator**: SemanticCurator (merges tool-calling strategies into playbook)
- [x] Interface contracts defined:
  - ReActGenerator: `TaskInput → TaskOutput` (v1.0.0, matches CoTGenerator)
  - Tool signature: `Callable[[...], Any]` with validation
  - Playbook bullet schema: extends existing with tool_sequence metadata
- [x] Forward compatibility plan documented: Semantic versioning for ReActGenerator, optional tool parameter preserves backward compat, reasoning trace format extensible

**ACE Framework Compliance**
- [x] Playbook storage locations determined: `.specify/memory/playbooks/tool-calling/` for tool strategies, domain-specific paths for RAG patterns
- [x] Delta update mechanism planned: Append-only bullet addition, semantic deduplication (≥0.8 similarity), counter increments for similar strategies
- [x] Adaptation mode selected: Hybrid - bootstrap with offline training on example tasks, then online learning from production usage with conservative thresholds

**Violations/Justifications**: No constitution principle violations. Feature fully aligns with ACE framework design.

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
ace/
├── generator/
│   ├── __init__.py              # Export ReActGenerator
│   ├── cot_generator.py         # Existing CoT implementation
│   ├── react_generator.py       # NEW: ReAct implementation
│   └── signatures.py            # Shared TaskInput/TaskOutput signatures
│
├── reflector/
│   └── grounded_reflector.py    # EXTENDS: Add tool usage analysis
│
├── curator/
│   └── semantic_curator.py      # EXTENDS: Handle tool-calling strategies
│
└── playbook/
    └── playbook.py               # EXTENDS: Tool sequence metadata support

tests/
├── unit/
│   ├── test_react_generator.py         # NEW: Unit tests for ReActGenerator
│   └── test_tool_validation.py         # NEW: Tool signature validation tests
│
├── integration/
│   ├── test_react_with_reflector.py    # NEW: ReAct + Reflector integration
│   └── test_tool_calling_workflow.py   # NEW: Full ACE cycle with tools
│
└── performance/
    └── test_react_performance.py       # NEW: Benchmarks for tool call overhead

examples/
└── react_rag_agent.py            # NEW: Example RAG agent with database tools

.specify/memory/playbooks/
└── tool-calling/                 # NEW: Tool-calling strategy storage
    ├── rag-database/
    └── general/
```

**Structure Decision**: Single project structure (Option 1) extending the existing `ace/` package. This maintains consistency with the current ACE architecture where all core components (generator, reflector, curator, playbook) reside in the `ace/` module. New ReActGenerator is added to `ace/generator/` alongside CoTGenerator, following the established pattern. Playbook storage follows constitution guidelines in `.specify/memory/playbooks/tool-calling/` with domain-specific subdirectories.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

**No complexity violations.** Feature fully aligns with ACE constitution - all 5 principles satisfied.

---

## Phase Completion Status

### ✅ Phase 0: Research (COMPLETE)

**Output**: `research.md`

**Decisions Made**:
1. DSPy ReAct module integration (use built-in `dspy.ReAct`)
2. Tool signature validation strategy (runtime validation using `inspect` module)
3. Maximum iterations configuration (hybrid: system → agent → task override)
4. Tool-calling strategy metadata schema (extend PlaybookBullet with optional fields)
5. Playbook context injection for tool selection (inject strategies as context)
6. Reasoning trace format (structured ReasoningStep list in TaskOutput)
7. Performance optimization strategy (LRU cache for playbook retrieval)
8. Tool failure handling strategy (graceful degradation with fallback)

**All unknowns resolved** - no NEEDS CLARIFICATION markers remain.

---

### ✅ Phase 1: Design & Contracts (COMPLETE)

**Outputs**:
- `data-model.md` - 6 core entities defined with relationships
- `contracts/react_generator_api.py` - Python API contract with interface specs
- `quickstart.md` - Developer onboarding guide (30-minute time-to-first-success)
- `CLAUDE.md` - Agent context updated with new tech stack

**Key Artifacts**:
- **Entities**: Tool, ReasoningStep, ToolCall, ToolCallingStrategy, TaskInput, TaskOutput
- **API Contracts**: ReActGeneratorInterface, Tool interface, exceptions
- **Developer Guide**: 6-step quickstart with code examples and troubleshooting

**Backward Compatibility**: ✅ Verified
- TaskInput/TaskOutput extend existing signatures (optional new fields)
- ReActGenerator compatible with CoTGenerator interface
- Existing playbooks work without migration

---

### ⏭️ Phase 2: Task Generation (PENDING)

**Next Command**: `/speckit.tasks`

This will generate `tasks.md` with implementation tasks based on:
- User stories from spec.md (P1, P2, P3 prioritization)
- Technical design from plan.md
- Data model entities
- API contracts

**Not included in /speckit.plan** - use separate command.

---

## Implementation Readiness

**Status**: ✅ **READY FOR IMPLEMENTATION**

**What's Ready**:
- [x] Feature specification (spec.md) - validated and clarified
- [x] Constitution compliance check - all 5 principles satisfied
- [x] Technical research complete - 8 key decisions documented
- [x] Data model designed - 6 entities with relationships
- [x] API contracts defined - interface specifications documented
- [x] Developer quickstart guide - 30-minute onboarding path
- [x] Agent context updated - tech stack propagated to Claude/Cursor

**What's Next**:
1. Run `/speckit.tasks` to generate implementation task breakdown
2. Implement tasks following TDD (Red-Green-Refactor-Reflect)
3. Validate against success criteria from spec.md
4. Deploy and enable online learning

**Estimated Implementation Time**: 2-3 days for P1 user story (RAG database agent)

---

## Key Files Summary

| File | Purpose | Status |
|------|---------|--------|
| `spec.md` | Feature requirements and success criteria | ✅ Complete |
| `plan.md` | Implementation plan (this file) | ✅ Complete |
| `research.md` | Technical decisions and rationale | ✅ Complete |
| `data-model.md` | Entity definitions and relationships | ✅ Complete |
| `contracts/react_generator_api.py` | Python API contract | ✅ Complete |
| `quickstart.md` | Developer onboarding guide | ✅ Complete |
| `tasks.md` | Implementation task breakdown | ⏭️ Run `/speckit.tasks` |
