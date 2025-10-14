# Changelog

All notable changes to the ACE Playbook project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.9.0]: https://github.com/jmanhype/ace-playbook/compare/v1.8.1...v1.9.0
[1.8.1]: https://github.com/jmanhype/ace-playbook/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/jmanhype/ace-playbook/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/jmanhype/ace-playbook/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/jmanhype/ace-playbook/releases/tag/v1.6.0
