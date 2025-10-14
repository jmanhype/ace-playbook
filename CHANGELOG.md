# Changelog

All notable changes to the ACE Playbook project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.13.0] - 2025-10-14

### Added - Phase 13: Complete Maintainability Suite (Priority 3)

All Priority 3 tasks from the engineering review are now complete, providing
comprehensive maintainability infrastructure for long-term code quality.

## [1.12.0] - 2025-10-14

### Added - Phase 12: Maintainability Improvements (Priority 3)

#### Mutation Testing Infrastructure
- **mutmut integration** for test suite quality verification:
  - Added `mutmut>=2.4.5` to dev dependencies
  - Mutation testing evaluates test robustness by introducing code mutations
  - Target: ≥90% mutation kill rate for critical modules
- **Mutation testing scripts**:
  - `scripts/run_mutation_tests.sh`: Automated mutation test execution
  - `Makefile` targets: `mutation-test`, `mutation-results`, `mutation-show`
  - Colored output with progress tracking and result summaries
- **Comprehensive documentation** in `CONTRIBUTING.md`:
  - What is mutation testing and why it matters
  - Running mutation tests with make/script/manual commands
  - Interpreting results and mutation score guide
  - Fixing surviving mutants with before/after examples
  - Common mutation types (arithmetic, comparison, boolean, numbers, strings)
  - Best practices for periodic testing and CI integration
  - Configuration and further reading resources
- **Makefile utilities** for development workflow:
  - Test commands: `test`, `test-unit`, `test-integration`, `test-cov`
  - Quality commands: `format`, `lint`, `security`, `complexity`
  - Mutation commands: `mutation-test`, `mutation-results`, `mutation-show`
  - Utility commands: `install`, `clean`, `ci`

#### Property-Based Testing with Hypothesis
- **Comprehensive property tests** in `tests/unit/test_curator_properties.py` (450+ lines):
  - **Idempotence**: Applying same insight twice only adds one bullet
  - **Monotonicity**: Counters never decrease during operations
  - **Domain Isolation**: Cross-domain operations always fail (ValueError)
  - **Completeness**: Every insight generates exactly one delta update
  - **Bounded Growth**: Playbook size grows sublinearly with duplicates
  - **Base Case**: First insight on empty playbook always adds bullet
  - **Symmetry**: Cosine similarity is symmetric (sim(A,B) = sim(B,A))
  - **Reflexivity**: Self-similarity is 1.0 for non-zero vectors
  - **Boundedness**: Similarity always in [-1, 1] range
- **Custom hypothesis strategies**:
  - `domain_ids`: Valid domain ID generator (regex-based)
  - `playbook_bullet`: Complete PlaybookBullet generator
  - `insight_dict`: Valid insight dictionary generator
  - `embeddings_384`: 384-dimensional embedding vectors
- **Test properties verified**:
  - Semantic deduplication invariants hold for ALL inputs
  - Counter operations preserve monotonicity
  - Domain isolation is absolute and enforceable
  - Similarity computation is mathematically sound
- **Configuration**: 20 examples in dev, scalable to 1000+ in CI

#### Complexity Monitoring with Radon
- **Pre-commit hooks** for automated complexity checks:
  - `radon-cc`: Cyclomatic complexity check (fails on C-grade or worse)
  - `radon-mi`: Maintainability index check (fails on B-grade or worse)
  - Runs automatically on `git push` to prevent complexity degradation
- **Makefile targets** for complexity analysis:
  - `make complexity`: Check complexity (informational)
  - `make complexity-strict`: Check with strict thresholds (fail on C-grade)
  - `make complexity-report`: Generate JSON reports for CI/CD
- **Complexity grades** and thresholds:
  - **Cyclomatic Complexity**: A (1-5), B (6-10), C (11-20), D (21-50), F (51+)
  - **Maintainability Index**: A (20-100), B (10-19), C (0-9)
- **Target standards**:
  - All modules ≤ B-grade average
  - No functions > C-grade (complexity 20)
  - Pre-commit blocks push if C-grade or worse detected

### Documentation

- Extended `CONTRIBUTING.md` with 130+ lines of mutation testing documentation
- Added practical examples of weak vs strong assertions
- Documented mutation testing workflow from detection to fix
- Included CI integration patterns for resource-intensive testing
- **Property-Based Testing Guide** (210+ lines in CONTRIBUTING.md):
  - What is property-based testing and why use it
  - Running property tests with hypothesis options
  - Writing property tests with idempotence, monotonicity, symmetry examples
  - Hypothesis strategies (built-in, custom, composite)
  - Common properties to test (invariants, relations, bounds)
  - Debugging property test failures with shrinking
  - Best practices and configuration
  - Further reading resources
- **Complexity Monitoring Guide** (210+ lines in CONTRIBUTING.md):
  - What is cyclomatic complexity and maintainability index
  - Running complexity checks with radon and make targets
  - Interpreting complexity grades and reports
  - Refactoring patterns for high-complexity code (extract methods, guard clauses, polymorphism, data structures)
  - Complexity targets and thresholds for ACE Playbook
  - Pre-commit integration and CI/CD patterns
  - Best practices for maintaining low complexity
  - Further reading resources

---

## [1.11.0] - 2025-10-14

### Added - Phase 11: Production Quality (Priority 2)

#### T075: Dependency Security Scanning (pip-audit)
- **pip-audit pre-commit hook** in `.pre-commit-config.yaml`:
  - Runs on push stage with `--desc on` and `--skip-editable` flags
  - Dual-layer security scanning (safety + pip-audit)
  - Comprehensive vulnerability detection across dependencies
- **Test results**: Detected 25 known vulnerabilities in 20 packages
- Added `pip-audit>=2.7.0` to dev dependencies

#### T076: Distributed Tracing (OpenTelemetry)
- **Tracing infrastructure** in `ace/utils/tracing.py` (269 lines):
  - `setup_tracing()`: Initialize OTLP exporter with configurable endpoint/service name
  - `@trace_operation()` decorator: Automatic span creation with error handling
  - Manual span control: `get_tracer()`, `add_span_attributes()`, `add_span_event()`
  - Graceful shutdown: `shutdown_tracing()` flushes pending spans
- **Auto-instrumentation**:
  - SQLAlchemy database queries
  - HTTP requests via requests library
- **Configuration via environment variables**:
  - `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP collector endpoint (default: http://localhost:4318)
  - `OTEL_SERVICE_NAME`: Service name (default: ace-playbook)
  - `OTEL_ENVIRONMENT`: Deployment environment (default: development)
  - `OTEL_TRACING_ENABLED`: Enable/disable tracing (default: true)
- **Dependencies added**:
  - opentelemetry-api>=1.27.0
  - opentelemetry-sdk>=1.27.0
  - opentelemetry-exporter-otlp-proto-http>=1.27.0
  - opentelemetry-instrumentation-sqlalchemy>=0.48b0
  - opentelemetry-instrumentation-requests>=0.48b0
- **Comprehensive tests** (16 tests, 97% coverage):
  - Setup with default and custom configuration
  - Decorator functionality and error handling
  - Enable/disable scenarios
  - Auto-instrumentation verification

#### T077: Architecture Decision Records (ADRs)
- **ADR documentation structure** in `docs/adr/`:
  - **ADR 0001: Use FAISS for Semantic Similarity Search**
    - Decision: FAISS IndexFlatIP over pgvector, Elasticsearch, Chromadb
    - Performance: P50 ~5-10ms, P95 ~12-15ms (meets SLA)
    - Rationale: In-process cache, per-domain indices, battle-tested
  - **ADR 0002: Append-Only Playbook Updates**
    - Decision: Increment counters, never rewrite content
    - Rationale: Auditability, determinism, O(1) performance
    - Trade-offs: First insight bias vs simplicity
  - **ADR 0003: Circuit Breaker for LLM API Calls**
    - Decision: Three-state circuit breaker (CLOSED/OPEN/HALF_OPEN)
    - Configuration: 5 failure threshold, 60s recovery timeout
    - Benefits: Fail-fast + self-healing, resource protection
  - **ADR 0004: Stage-Based Promotion Gates**
    - Decision: SHADOW → STAGING → PROD progression with counter-based gates
    - Criteria: SHADOW→STAGING (helpful≥3, ratio≥3.0), STAGING→PROD (helpful≥5, ratio≥5.0)
    - Benefits: Safe rollout, automatic promotion, empirical validation
- **Format**: Michael Nygard format with Context/Decision/Consequences
- **Total documentation**: 796 lines across 5 files (including README.md index)

#### T078: Refactor Large Files (semantic_curator.py)
- **Problem**: semantic_curator.py was 659 lines, violating single responsibility principle
- **Solution**: Extracted code into focused modules
- **New modules created**:
  - `curator_models.py` (99 lines): CuratorInput, CuratorOutput, DeltaUpdate data classes
  - `domain_validator.py` (104 lines): Domain validation and CHK081-CHK082 enforcement
  - `curator_utils.py` (65 lines): Similarity computation and bullet hashing utilities
  - `promotion_policy.py` (70 lines): Promotion gates and quarantine logic
- **Impact**: Main file reduced by 224 lines (34% reduction: 659 → 435 lines)
- **Benefits**:
  - Each module has single responsibility
  - Easier to test individual components
  - Improved code organization and discoverability
  - Better maintainability and extensibility
- **Test results**: 12/12 tests passing, no functionality changes

### Changed

- Updated `ace/curator/__init__.py` to export CuratorInput, CuratorOutput, DeltaUpdate from `curator_models`
- Refactored SemanticCurator to use extracted utility functions and validators

### Documentation

- Created comprehensive ADR index with decision log organized by phase
- Documented alternatives considered and trade-offs for key architectural decisions
- Added lifecycle management guidelines for ADR status (PROPOSED, ACCEPTED, DEPRECATED, SUPERSEDED)

---

## [1.10.0] - 2025-10-14

### Added - Phase 10: Production Hardening (Priority 1 - BLOCKING)

#### T069: Performance Benchmark Tests
- **Retrieval Performance Tests** in `tests/benchmarks/test_retrieval_performance.py`:
  - Validates P50 ≤10ms claim for semantic retrieval
  - Tests small (10 bullets), medium (100 bullets), and large (1000 bullets) playbooks
  - Verifies P95 ≤15ms across all playbook sizes
  - Includes bulk update performance tests (50 insights)
  - Uses pytest-benchmark for statistical analysis
- Performance results:
  - Small playbook: ~5ms P50, ~8ms P95
  - Medium playbook: ~8ms P50, ~12ms P95
  - Large playbook: ~10ms P50, ~15ms P95

#### T070: Input Validation Fixes
- **batch_merge() validation** in `ace/curator/semantic_curator.py`:
  - Added empty list check with ValueError
  - Validates all tasks have same domain_id
  - Enforces domain isolation in batch operations
  - Clear error messages for validation failures
- Prevents domain isolation violations and improves error diagnostics

#### T071: Circuit Breaker for LLM API Calls
- **CircuitBreaker** pattern in `ace/utils/circuit_breaker.py`:
  - Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
  - Configurable failure threshold (default: 5 consecutive failures)
  - Automatic recovery testing after timeout (default: 60s)
  - Metrics tracking: total failures, success rate, state transitions
  - Thread-safe implementation with explicit locking
- **LLM Circuit Breaker Wrapper** in `ace/utils/llm_circuit_breaker.py`:
  - `protected_predict()` wraps DSPy predictors with circuit breaker
  - Per-component breakers (generator, reflector) for isolation
  - Fail-fast behavior during LLM API outages
  - Structured logging for all state transitions
- **Comprehensive Tests** (18 tests) covering:
  - State transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
  - Failure counting and threshold enforcement
  - Recovery testing with success/failure scenarios
  - Thread safety with concurrent access
  - Metrics accuracy and decorator pattern

#### T072: FAISS Resource Cleanup
- **Memory leak fix** in `ace/curator/semantic_curator.py:batch_merge()`:
  - Wrapped FAISS operations in try-finally block
  - Ensures index cleanup even on exceptions
  - Prevents unbounded memory growth in long-running processes
- **FAISSIndexManager.clear_index()** in `ace/utils/faiss_index.py`:
  - Clears in-memory FAISS indices and bullet ID mappings
  - Preserves persisted files on disk
  - Explicit cleanup for batch operations

#### T073: Operations Runbook
- **Comprehensive RUNBOOK.md** (819 lines) in `docs/`:
  - Emergency rollback procedure (<5 minute target):
    - Git revert/checkout commands
    - Alembic database downgrade
    - Docker Compose and Kubernetes restart procedures
    - Health check verification steps
  - Common incidents troubleshooting:
    - High retrieval latency (P95 > 15ms)
    - Out of memory errors
    - Failed promotions (Shadow → Staging → Prod)
    - Circuit breaker stuck OPEN
  - Circuit breaker management:
    - Status checking commands
    - Manual reset procedures
    - Integration with metrics
  - Performance troubleshooting:
    - FAISS index optimization
    - Database query analysis
    - Connection pool tuning
  - Post-mortem template for incident review

#### T074: Rate Limiting for LLM API Calls
- **Token Bucket Rate Limiter** in `ace/utils/rate_limiter.py`:
  - Token bucket algorithm with configurable rates
  - Default limits: 60 calls/min, 1000 calls/hour, burst=10
  - Per-domain and global rate limiting modes
  - Thread-safe implementation with threading.Lock
  - Blocking and non-blocking acquire modes
  - Metrics: total_calls, total_throttled, throttle_rate
  - Automatic token refill based on elapsed time
- **Integration with Circuit Breaker** in `ace/utils/llm_circuit_breaker.py`:
  - Rate limiting applied BEFORE circuit breaker in `protected_predict()`
  - Multi-layered protection: Rate Limiter → Circuit Breaker → LLM API
  - Configurable via `rate_limit` parameter (default: True)
  - Timeout support for blocking mode (default: 30s)
- **Comprehensive Tests** (18 tests) covering:
  - Token bucket: initial state, refill, consume, wait time calculation
  - Burst handling (allows 10 rapid calls, throttles 11th)
  - Per-domain isolation (domain-a exhausted, domain-b still available)
  - Thread safety (20 concurrent threads: 10 succeed, 10 throttled)
  - Decorator pattern for function wrapping
  - Metrics accuracy and reset functionality

### Fixed

- **FAISS memory leak** in batch operations (T072)
- **Input validation** in batch_merge() for empty lists and mixed domains (T070)
- **Missing performance benchmarks** - now verifies P50 ≤10ms claim (T069)

### Security

- **Rate limiting** prevents DoS attacks and API quota exhaustion (T074)
- **Circuit breaker** prevents cascading failures from LLM API outages (T071)
- **Input validation** enforces domain isolation in batch operations (T070)

### Changed

- `protected_predict()` now includes rate limiting before circuit breaker protection
- `batch_merge()` enforces stricter validation and includes FAISS cleanup
- All LLM API calls now protected by both rate limiting and circuit breaker

---

## [1.9.0] - 2025-10-13

### Added - Phase 9: Polish and Production Readiness

#### T065: Observability Metrics
- **MetricsCollector** with thread-safe Prometheus format export (`ace/ops/metrics.py`)
- Metrics tracking:
  - `playbook_bullet_count{domain_id}` - Bullets per domain
  - `retrieval_latency_ms{quantile}` - P50/P95/P99 retrieval latency
  - `reflection_count` - Total reflections processed
  - `promotion_events_total` - Bullet promotions to staging/prod
  - `quarantine_events_total` - Bullets quarantined
  - `bullets_added_total` - New bullets added
  - `bullets_incremented_total` - Existing bullets incremented
- **LatencyTimer** context manager for timing operations
- Singleton pattern via `get_metrics_collector()`

#### T066: Guardrail Monitoring
- **GuardrailMonitor** with automated rollback on performance regression (`ace/ops/guardrails.py`)
- Regression detection thresholds:
  - Success rate: -8% delta triggers rollback
  - Latency P95: +30% increase triggers rollback
  - Error rate: 15% absolute triggers rollback
- **Automated Rollback**: Quarantines bullets added in last 24 hours
- **PerformanceSnapshot** for baseline vs current comparison
- Alert callback system for notifications
- **RollbackTrigger** audit trail with reason, metric, threshold, and actual value

#### T067: Docker Compose Setup
- **Dockerfile**: Python 3.11-slim with uv package manager
- **docker-compose.yml**: Volume persistence for SQLite database
- **.dockerignore**: Optimized build context (excludes venv, cache, .git)
- Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DSPY_LM`, `DSPY_MODEL`
- Health check endpoint for container orchestration
- Automatic Alembic migrations on startup via CMD
- Port 8000 exposed for metrics endpoint

#### T068: E2E Smoke Test
- Comprehensive smoke tests in `tests/e2e/test_smoke.py`:
  - `test_complete_workflow`: Empty playbook → 5 tasks → ≥3 bullets validation
  - `test_deduplication_works`: Semantic deduplication at 0.8 threshold
  - `test_retrieval_excludes_shadow_bullets`: Stage-based filtering (SHADOW vs PROD)
- Validates: playbook updates, bullet metadata (embedding dimension), retrieval filtering
- Test isolation with in-memory SQLite and global engine reset per test
- Sample arithmetic tasks with ground truth for realistic workflow testing

### Fixed

- **FAISS Integration Bug** in `ace/curator/semantic_curator.py`:
  - Replace non-existent `build_index()` with `add_vectors(domain_id, embeddings, bullet_ids)`
  - Replace `search_similar()` with `search(domain_id, query_embedding, k)`
  - Convert embedding lists to numpy arrays before FAISS calls (`np.array(embedding, dtype=np.float32)`)
  - Add bullet_id to bullet mapping for efficient lookup during search
- **E2E Test Fixes**:
  - Update `test_complete_workflow` to use `get_by_domain()` instead of non-existent `get_by_section()`
  - Remove journal assertion (journal writing is CuratorService responsibility, not SemanticCurator)
  - Manual section filtering: `[b for b in all_bullets if b.section == InsightSection.HELPFUL]`

### Changed

- Updated `ace/ops/__init__.py` to export new modules:
  - `MetricsCollector`, `get_metrics_collector`, `LatencyTimer`
  - `GuardrailMonitor`, `PerformanceSnapshot`, `RollbackTrigger`, `create_guardrail_monitor`
- Updated README.md with Phase 9 features, Docker Compose quickstart, observability examples
- Project structure documentation updated to include Docker files and E2E tests

## [1.8.1] - 2025-10-13

### Fixed

- **Shadow Learning Integration Tests** (100% pass rate achieved)
- Corrected journal repository method calls in `stage_manager.py`
- Fixed test parameter names in `test_stage_manager.py` and `test_shadow_learning.py`

## [1.8.0] - 2025-10-13

### Added - Phase 8: Online Adaptation in Production

#### T058: Shadow Learning Infrastructure
- Shadow mode for testing new bullets without production impact
- Parallel execution: shadow and production paths run concurrently
- **ShadowExecutionResult** with both outcomes for comparison
- Integration with `OnlineLearningLoop` for continuous learning

#### T059: Automated Promotion Gates
- **StageManager** with promotion criteria:
  - SHADOW → STAGING: helpful_count ≥ 3, ratio ≥ 3.0
  - STAGING → PROD: helpful_count ≥ 5, ratio ≥ 5.0
- `promote_bullets()` method with automatic gate validation
- `get_production_bullets()` excludes SHADOW/QUARANTINED bullets
- Audit trail via DiffJournal for all promotions

#### T060: Online Learning Loop
- **OnlineLearningLoop** orchestrates continuous adaptation:
  - Fetch production tasks from queue
  - Execute with current playbook (Generator → Reflector)
  - Curator merges insights into SHADOW stage
  - StageManager checks promotion gates
- **OnlineLoopConfig** with intervals, batch sizes, shadow mode toggle
- **OnlineLoopMetrics** tracks tasks processed, bullets promoted, errors
- Graceful shutdown with signal handling (SIGINT, SIGTERM)

#### T061: Performance Degradation Detection
- **GuardrailMonitor** baseline performance tracking:
  - Success rate delta threshold: -8%
  - Latency P95 threshold: +30%
  - Error rate absolute threshold: 15%
- **PerformanceSnapshot** captures metrics at intervals
- Automated rollback quarantines bullets added in last 24 hours

#### T062: Feedback Loop Metrics
- **MetricsCollector** with thread-safe counters:
  - Playbook bullet counts per domain
  - Retrieval/reflection/curator latency histograms
  - Promotion and quarantine event counters
- Prometheus text format export
- **LatencyTimer** context manager for consistent timing

### Fixed

- Integration tests for shadow learning (T063)
- Integration tests for promotion gates (T064)
- Test parameter alignment and method signatures

## [1.7.0] - 2025-10-12

### Added - Phase 7: Offline Training System

#### T052: Dataset Loader
- **DatasetLoader** supporting GSM8K and custom JSON formats
- **DatasetExample** dataclass for task structure
- Automatic dataset caching and parsing
- `create_dataset_loader()` factory method

#### T053: Offline Training Workflow
- **OfflineTrainer** orchestrates end-to-end training:
  - Load dataset → Generate responses → Reflect → Curate
  - Parallel task processing with ThreadPoolExecutor
  - **TrainingConfig** for customization (num_examples, domain_id, target_stage)
  - **TrainingMetrics** tracks success/failure/accuracy
- Integration with existing Generator, Reflector, Curator

#### T054: Batch Processing Optimization
- **SemanticCurator.batch_merge()** for efficient multi-task processing:
  - Single FAISS index build for all embeddings
  - Deduplicate across entire batch (not per-task)
  - Single transaction to commit all updates
- Reduces overhead from O(n) to O(1) index builds per batch

#### T055: Review Queue for Low-Confidence Insights
- **ReviewService** manages human review workflow:
  - Automatic queueing for confidence < 0.7 (configurable threshold)
  - `list_pending()`, `approve()`, `reject()` methods
  - Approved insights merged to staging, rejected insights discarded
  - DiffJournal audit trail for all review actions
- **ReviewQueueRepository** with persistence layer

#### T056: Training Metrics Tracking
- **MetricsCollector** in `ace/ops/metrics.py`:
  - Training-specific counters (tasks processed, success/failure rates)
  - Integration with offline trainer for real-time tracking
- **TrainingMetrics** summary after training completion

#### T057: Offline Training CLI
- Command-line interface via `ace/cli/train.py`:
  - `--dataset`: GSM8K or custom JSON path
  - `--num-examples`: Limit training samples
  - `--domain-id`: Target domain namespace
  - `--parallel`: Enable parallel processing
  - Progress reporting and final metrics display

### Changed

- `SemanticCurator` now supports both single-task (`apply_delta`) and batch (`batch_merge`) modes
- `PlaybookRepository` extended with batch operations
- Database schema updated for review queue (Alembic migration)

## [1.6.0] - 2025-10-11

### Added - Phases 1-6: Core Framework

#### Phase 1: Data Model and Contracts
- SQLAlchemy models: PlaybookBullet, DiffJournal, TaskHistory, ReflectionResult
- Domain isolation with namespace pattern: `^[a-z0-9-]+$`
- Stage enum: SHADOW, STAGING, PROD, QUARANTINED
- Alembic migrations for schema evolution

#### Phase 2: Embedding and FAISS Integration
- **EmbeddingService** with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **FAISSIndexManager** with per-domain IndexFlatIP indices
- Cosine similarity via L2-normalized inner product
- Singleton pattern for efficient resource management

#### Phase 3: Semantic Curator
- **SemanticCurator** with 0.8 cosine similarity threshold
- Pure Python deduplication (no DSPy dependencies)
- Append-only updates: increment counters, never rewrite content
- **DeltaUpdate** audit trail with before/after hashes
- Domain isolation enforcement (CHK081-CHK082)

#### Phase 4: Repository Layer
- **PlaybookRepository**: CRUD for playbook bullets with stage filtering
- **DiffJournalRepository**: Append-only change log with rollback metadata
- **ReviewQueueRepository**: Human review workflow for low-confidence insights
- Transaction management with SQLAlchemy sessions

#### Phase 5: Generator and Reflector
- **CoTGenerator**: DSPy Chain-of-Thought generator with playbook retrieval
- **GroundedReflector**: Analyzes outcomes and extracts labeled insights
- **InsightSection** enum: Helpful, Harmful, Neutral
- Confidence scoring for review queue filtering

#### Phase 6: Stage Management and Promotion Gates
- **StageManager**: Promotion criteria and gate validation
  - SHADOW → STAGING: helpful_count ≥ 3, ratio ≥ 3.0
  - STAGING → PROD: helpful_count ≥ 5, ratio ≥ 5.0
- Quarantine detection: harmful_count ≥ helpful_count
- Automated rollback procedures (<5 minutes)

### Fixed

- Thread safety in embedding service (lazy model loading)
- FAISS index persistence with bullet ID mappings
- Domain isolation cross-checks in curator
- Test flakiness with proper database isolation

### Security

- Input validation for domain IDs (regex pattern)
- Reserved domain namespace protection (system, admin, test)
- SQL injection prevention via parameterized queries

---

## Version Guidelines

- **MAJOR**: Breaking changes (API incompatibility, schema migrations requiring manual intervention)
- **MINOR**: New features, backward-compatible functionality additions
- **PATCH**: Bug fixes, performance improvements, documentation updates

[1.13.0]: https://github.com/jmanhype/ace-playbook/compare/v1.12.0...v1.13.0
[1.12.0]: https://github.com/jmanhype/ace-playbook/compare/v1.11.0...v1.12.0
[1.11.0]: https://github.com/jmanhype/ace-playbook/compare/v1.10.0...v1.11.0
[1.10.0]: https://github.com/jmanhype/ace-playbook/compare/v1.9.0...v1.10.0
[1.9.0]: https://github.com/jmanhype/ace-playbook/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/jmanhype/ace-playbook/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/jmanhype/ace-playbook/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/jmanhype/ace-playbook/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/jmanhype/ace-playbook/releases/tag/v1.6.0
