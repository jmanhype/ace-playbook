<!--
SYNC IMPACT REPORT
==================
Version Change: N/A → 1.0.0 (Initial Constitution)
Change Type: MINOR (Initial creation with foundational principles)

Modified Principles:
- NEW: I. Self-Improving Code Quality (ACE-Driven)
- NEW: II. Reflection-Based Testing Standards
- NEW: III. Context-Aware User Experience
- NEW: IV. Performance with Continuous Optimization
- NEW: V. Modular Black-Box Architecture (Generator-Reflector-Curator)

Added Sections:
- Core Principles (5 principles integrating ACE framework)
- ACE Framework Implementation Standards
- Development Workflow & Quality Gates
- Governance

Removed Sections:
- None (initial creation)

Templates Requiring Updates:
- ✅ .specify/templates/plan-template.md (updated: added comprehensive ACE constitution check)
- ✅ .specify/templates/spec-template.md (updated: added reflection-based testing guidance)
- ✅ .specify/templates/tasks-template.md (updated: added playbook infrastructure and maintenance tasks)

Follow-up TODOs:
- None (all fields populated)

Rationale for Version 1.0.0:
This is the initial constitution establishing foundational principles for the ACE Playbook project. It integrates Agentic Context Engineering (ACE) framework with traditional software engineering practices, creating a self-improving system architecture with clear quality, testing, UX, performance, and modularity standards.
-->

# ACE Playbook Constitution

## Core Principles

### I. Self-Improving Code Quality (ACE-Driven)

Every component MUST maintain and evolve a "playbook" of coding strategies through the Generator-Reflector-Curator pattern.

**Requirements**:
- Code modules maintain context bullets documenting: successful patterns (marked "Helpful"), anti-patterns to avoid (marked "Harmful"), and observations (marked "Neutral")
- Each code review MUST generate reflection insights that update the component's playbook
- Playbook updates use **incremental delta additions** (append-only) rather than wholesale rewrites to prevent context collapse
- Code quality metrics track playbook utilization: % of code following helpful strategies, % avoiding harmful patterns
- Duplicate or redundant strategies MUST be merged using semantic similarity (embedding-based deduplication at threshold ≥ 0.8)

**Rationale**: Traditional code review captures lessons ephemerally. ACE's playbook approach ensures accumulated wisdom persists and compounds, enabling continuous quality improvement without fine-tuning or retraining developers.

### II. Reflection-Based Testing Standards

Test-Driven Development (TDD) MUST incorporate execution-based reflection to generate new test cases automatically.

**Requirements**:
- **Red-Green-Refactor with Reflection**: After each test cycle, run Reflector to analyze: what the passing test validated (Helpful), what edge cases were missed (Harmful), what assumptions were made (Neutral)
- Tests MUST include execution feedback integration: capture runtime errors, performance metrics, and environment signals as reflection inputs
- Test playbooks accumulate learned test strategies: boundary conditions to check, common failure modes, integration points requiring validation
- Contract tests, integration tests, and unit tests each maintain separate playbooks
- Test coverage MUST be validated against playbook completeness (all "Harmful" patterns have corresponding test cases)

**Rationale**: Static test suites decay over time as code evolves. Reflection-based testing creates self-expanding test coverage by learning from failures and successes, similar to how ACE improved agent performance by 10.6% on AppWorld through accumulated test strategies.

### III. Context-Aware User Experience

User interfaces and interactions MUST maintain consistency through shared UX playbooks that evolve based on user feedback.

**Requirements**:
- UX components reference a centralized playbook of interaction patterns, accessibility guidelines, and user journey strategies
- User feedback (explicit and behavioral) feeds into Reflector to generate UX insights: effective patterns (Helpful), confusing interactions (Harmful), neutral observations
- Design decisions MUST be documented as playbook bullets with usage counters tracking: frequency of pattern application, user satisfaction scores, accessibility compliance
- Context evolution prevents "UX drift": new features append patterns rather than creating inconsistent paradigms
- Playbook bullets include: navigation strategies, error message formats, loading state patterns, responsive design breakpoints

**Rationale**: UX consistency degrades as teams grow and features accumulate. A living playbook ensures new work builds on proven patterns while capturing lessons from user interactions, similar to ACE's grow-and-refine strategy.

### IV. Performance with Continuous Optimization

Performance requirements MUST be tracked through playbooks that accumulate optimization strategies and anti-patterns.

**Requirements**:
- Performance targets MUST be explicit and measurable: latency (p50, p95, p99), throughput (requests/sec), resource usage (memory, CPU), scalability limits
- Each performance test generates reflection: optimization techniques that succeeded (Helpful), bottlenecks encountered (Harmful), profiling insights (Neutral)
- Performance playbooks organize by category: database query optimization, caching strategies, algorithmic complexity, network efficiency, rendering performance
- **Execution-based feedback**: automated performance tests feed metrics directly into Reflector without manual labeling
- Regression prevention: any performance degradation triggers Reflector to identify harmful pattern and add to playbook
- Performance budgets enforced at build time, with playbook providing remediation strategies when budgets exceeded

**Rationale**: Performance optimization knowledge typically exists in tribal memory or scattered docs. Playbook-driven performance ensures every optimization lesson persists and compounds, similar to how ACE reduced agent rollout latency through accumulated efficiency strategies.

### V. Modular Black-Box Architecture (Generator-Reflector-Curator)

System architecture MUST follow the Generator-Reflector-Curator separation pattern with forward-compatible interfaces.

**Requirements**:
- **Generator modules**: Components that produce outputs (APIs, services, UIs) MUST expose clear input/output contracts with versioned schemas
- **Reflector modules**: Analysis components that critique outputs and generate insights MUST operate independently from Generators
- **Curator logic**: Deterministic, non-LLM code that merges insights into playbooks MUST be separated from both Generator and Reflector
- **Black-box principle**: Modules interact only through defined interfaces (signatures in DSPy terms), never through internal state access
- **Forward compatibility**: Interface versioning follows semantic versioning (MAJOR.MINOR.PATCH) with explicit deprecation policies:
  - MAJOR: Breaking changes (require migration)
  - MINOR: Backward-compatible additions (new optional fields, new endpoints)
  - PATCH: Bug fixes and clarifications (no interface changes)
- Inter-module communication serialized as structured data (JSON, Protocol Buffers, or typed DSPy Signatures)
- Each module maintains its own playbook; cross-module learnings propagate through shared playbook sections

**Rationale**: Monolithic architectures resist change and make testing difficult. The Generator-Reflector-Curator pattern from ACE provides natural separation of concerns: production logic, analysis logic, and knowledge management. This enables independent evolution and testing of each component while maintaining system-wide learning.

## ACE Framework Implementation Standards

The ACE Playbook operates as a self-improving system using Agentic Context Engineering principles.

### Playbook Structure

All playbooks MUST use the following data structure:

```python
@dataclass
class PlaybookBullet:
    content: str                    # The strategy/heuristic text
    section: str                    # Category: "Strategies", "Pitfalls", "Observations"
    helpful_count: int = 0          # Times this advice led to success
    harmful_count: int = 0          # Times following this caused failure
    tags: list = field(default_factory=list)  # Optional: domain, component
```

### Delta Update Policy

- **Incremental only**: Playbook updates MUST append new bullets or update existing ones; never rewrite entire playbook
- **Deduplication**: Before adding new bullet, check for semantic similarity (cosine similarity of embeddings) ≥ 0.8 with existing bullets
- **Counter updates**: When similar bullet found, increment appropriate counter (helpful_count or harmful_count) instead of duplicating
- **Pruning triggers**: When playbook exceeds 100 bullets, run consolidation: merge semantically similar bullets, remove bullets with (helpful_count = 0 and harmful_count = 0) after 10+ iterations

### Reflection Protocol

- **Success reflection**: When task succeeds, Reflector identifies: strategies that contributed (Helpful), potential optimizations (Neutral)
- **Failure reflection**: When task fails, Reflector identifies: mistakes made (Harmful), missing strategies (Helpful), context info (Neutral)
- **Execution feedback priority**: Prefer concrete execution signals (test results, error messages, metrics) over subjective assessment
- **Reflection frequency**: Run Reflector after every: failed test, performance regression, user-reported issue, successful deployment (to capture positive patterns)

### Adaptation Modes

- **Offline adaptation (pre-deployment)**: Use training dataset to accumulate initial playbook; run multiple epochs to refine strategies
- **Online adaptation (runtime)**: Continue evolving playbook based on production feedback; implement safeguards (human review for critical changes, confidence thresholds)
- **Hybrid mode**: Bootstrap with offline-trained playbook, then adapt online with conservative update policy

## Development Workflow & Quality Gates

### Phase 0: Constitution Compliance Check

Before starting ANY feature work:
- Verify feature aligns with all 5 core principles
- Identify which playbooks will be updated (code quality, testing, UX, performance, architecture)
- Document any principle violations and justification for deviation

### Phase 1: Generator Design

- Define clear input/output signatures for all components
- Establish performance budgets and success metrics
- Create initial playbook bullets based on similar past features
- Design black-box interfaces with versioning plan

### Phase 2: Test-First with Reflection

- Write tests that will fail (Red phase)
- Implement feature to pass tests (Green phase)
- Refactor with quality playbook guidance (Refactor phase)
- Run Reflector on test outcomes to generate insights
- Update testing playbook with new strategies/pitfalls

### Phase 3: Integration & Performance Validation

- Run integration tests with execution feedback capture
- Measure performance against budgets
- Generate performance reflection if regressions occur
- Update performance playbook with optimization insights

### Phase 4: Curator Review

- Review all playbook updates for this feature
- Deduplicate and merge similar bullets
- Validate no context collapse (details preserved, not summarized)
- Commit playbook changes alongside code changes

### Quality Gate Enforcement

**Block deployment if**:
- Performance budgets exceeded without documented playbook justification
- Tests fail without Reflector analysis
- New code violates "Harmful" patterns in quality playbook without explicit override
- Playbook changes deleted existing strategies (must append or update only)
- Interfaces changed without version bump

**Require approval if**:
- Playbook accumulated >20 new bullets in single PR (indicates insufficient deduplication)
- Feature deviates from constitution principles (must document in Complexity Tracking table)
- Performance reflection identifies new bottleneck class not previously in playbook

## Governance

### Amendment Authority

This constitution supersedes all other development practices and guidelines.

**Amendment process**:
1. Propose amendment via PR with justification
2. Document impact on existing playbooks
3. Update all dependent templates (plan, spec, tasks)
4. Requires approval from: project maintainer + 1 senior contributor
5. Include migration plan if amendment affects existing code

### Version Management

Constitution versioning follows semantic versioning:
- **MAJOR**: Principle removal, incompatible governance changes (requires codebase audit)
- **MINOR**: New principle added, significant guidance expansion (requires template updates)
- **PATCH**: Clarifications, typo fixes, non-semantic improvements (no code changes needed)

### Compliance Review

**Continuous compliance**:
- All PRs MUST reference applicable constitution principles in description
- CI pipeline enforces: performance budgets, playbook delta-only updates, interface versioning
- Monthly playbook audit: review bullet effectiveness (helpful_count vs harmful_count), prune stale bullets

**Complexity justification**:
- Any code violating constitution principles MUST be documented in feature's plan.md Complexity Tracking table
- Justification MUST explain: why violation necessary, what simpler alternative was rejected, how violation is contained

### Playbook Persistence

- Playbooks stored in `.specify/memory/playbooks/` directory, organized by domain
- Playbook changes committed with descriptive messages: `playbook: add [category] strategy for [context]`
- Playbooks versioned alongside code (not separate repo)
- Playbook archaeology encouraged: reference historical bullets to understand system evolution

**Version**: 1.0.0 | **Ratified**: 2025-10-13 | **Last Amended**: 2025-10-13
