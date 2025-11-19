<!--
Sync Impact Report:
- Version: Template → 1.0.0 (Initial ratification)
- Principles Added: I. Test-First Development (TDD), II. SOLID Architecture, III. Security-First Design,
  IV. Compliance by Design, V. API Versioning & Stability, VI. Observability & Monitoring,
  VII. Graceful Degradation, VIII. Code Quality Standards
- Sections Added: Security Requirements, Compliance Standards, Performance & Scalability
- Templates Status:
  ✅ plan-template.md - Includes Constitution Check section
  ✅ spec-template.md - Aligned with compliance requirements
  ✅ tasks-template.md - Includes test-first task ordering
  ⚠ README.md - Manual update recommended to reference constitution
- Deferred: None
- Created: 2025-11-18
-->

# RCST v2 Constitution

## Core Principles

### I. Test-First Development (TDD) - NON-NEGOTIABLE

**All code MUST be developed using strict Test-Driven Development (TDD):**
- Tests MUST be written before implementation code
- Tests MUST fail initially (Red phase)
- Implementation MUST make tests pass (Green phase)
- Code MUST be refactored after passing (Refactor phase)
- Test coverage MUST be ≥90% for all modules
- Test coverage MUST be 100% for security-critical code (auth, authorization, data access)

**Rationale:** V1 had 70% coverage with 100+ tests written but not implemented. V2 enforces TDD to prevent technical debt and ensure correctness from the start.

**Test Requirements:**
- Unit tests for all business logic
- Integration tests for all API endpoints
- Contract tests for all external service adapters
- Property-based tests for complex algorithms (verification, gap analysis)
- Accessibility tests for all UI components (WCAG 2.1 AA)
- Performance tests for critical paths (gap analysis, vector search)

### II. SOLID Architecture - MANDATORY

**All code MUST adhere to SOLID principles:**

- **S**ingle Responsibility: Each class/module has one reason to change
- **O**pen/Closed: Open for extension, closed for modification (use interfaces/protocols)
- **L**iskov Substitution: Implementations must be substitutable for their abstractions
- **I**nterface Segregation: Many focused interfaces over one general interface
- **D**ependency Inversion: Depend on abstractions, inject implementations

**Architectural Patterns REQUIRED:**
- **Repository Pattern:** All data access via repository interfaces
- **Adapter Pattern:** All external dependencies isolated behind adapters
- **Factory Pattern:** Service creation via dependency injection
- **Strategy Pattern:** Algorithm selection (classification, verification, caching)
- **Facade Pattern:** Simplified interfaces to complex subsystems

**Rationale:** V1 successfully refactored from 1,233-line god object to SOLID architecture. V2 enforces this from the start.

### III. Security-First Design - NON-NEGOTIABLE

**Zero-trust security MUST be enforced at every layer:**

**Defense in Depth (Multiple Layers):**
- **Layer 1: Network** - Rate limiting, DDoS protection
- **Layer 2: Authentication** - JWT + OIDC, MFA support
- **Layer 3: Authorization** - Cedar/OPA policies, RBAC, ABAC
- **Layer 4: Data Access** - Row-Level Security (RLS), tenant isolation
- **Layer 5: Application** - Input validation, output encoding, SQL injection prevention
- **Layer 6: Audit** - Hash-chained audit logs, WORM storage

**Security Requirements:**
- All secrets MUST be in environment variables or secret managers (NO hardcoding)
- All authentication MUST use industry-standard protocols (OIDC, OAuth2)
- All authorization MUST use policy engines (Cedar or OPA)
- All data MUST be encrypted at rest (AES-256) and in transit (TLS 1.3)
- All inputs MUST be validated with Pydantic models (strict mode)
- All database queries MUST use ORM or parameterized queries (NO string interpolation)
- Row-Level Security (RLS) MUST be enabled in production
- All API endpoints MUST have rate limiting
- All errors MUST NOT leak sensitive information in production

**Rationale:** V1 had critical security fixes (tenant leakage, hardcoded secrets, mock auth). V2 prevents these by design.

### IV. Compliance by Design - MANDATORY

**System MUST be designed for compliance from day one:**

**Target Certifications:**
- **SOC 2 Type II:** Trust Services Criteria (Security, Availability, Confidentiality, Processing Integrity, Privacy)
- **HIPAA:** Business Associate Agreement (BAA) ready, PHI handling procedures
- **GDPR:** Data privacy, right to deletion, data portability

**Compliance Requirements:**
- All user actions MUST be logged in immutable audit trail
- Audit logs MUST be hash-chained for tamper detection
- Audit logs MUST be stored in WORM (Write-Once-Read-Many) storage
- Data retention policies MUST be configurable per tenant
- Data deletion MUST be verifiable and complete
- Provenance tracking MUST be complete (requirement → document → chunk → gap)
- License scope enforcement MUST be backend-enforced (not client-side)

**Rationale:** V1 implements ALCOA+ provenance and audit trails. V2 extends this to full SOC 2/HIPAA compliance.

### V. API Versioning & Stability - MANDATORY

**All APIs MUST be versioned and backward-compatible:**

**Versioning Strategy:**
- URL versioning: `/api/v1/`, `/api/v2/`
- Version in Accept header: `Accept: application/vnd.rcst.v1+json`
- Semantic versioning for API: MAJOR.MINOR.PATCH
  - MAJOR: Breaking changes
  - MINOR: New features (backward-compatible)
  - PATCH: Bug fixes (backward-compatible)

**Stability Requirements:**
- Breaking changes MUST be announced 6 months in advance
- Previous MAJOR version MUST be supported for 12 months after new version release
- Deprecation warnings MUST be included in responses
- Migration guides MUST be provided for breaking changes
- API contracts MUST be documented with OpenAPI 3.1

**Rationale:** V1 lacks API versioning. V2 prevents breaking changes for existing clients.

### VI. Observability & Monitoring - MANDATORY

**System MUST be fully observable in production:**

**Three Pillars of Observability:**
- **Metrics:** Prometheus metrics for all critical operations
  - Request rates, latencies (p50, p95, p99), error rates
  - Database connection pool usage, query latencies
  - Cache hit rates, Redis latency
  - LLM API calls, costs, latencies
  - Business metrics (gap detection accuracy, confidence scores)

- **Logging:** Structured JSON logging with correlation IDs
  - All requests logged with tenant_id, user_id, trace_id
  - All errors logged with full context (non-sensitive)
  - Log levels: DEBUG (dev), INFO (prod), WARN, ERROR, CRITICAL

- **Tracing:** OpenTelemetry distributed tracing
  - All API requests traced end-to-end
  - Database queries, cache operations, LLM calls instrumented
  - Trace sampling: 100% errors, 10% success (configurable)

**Monitoring Requirements:**
- Health checks MUST be implemented (`/health`, `/ready`, `/live`)
- Alerts MUST be configured for critical errors
- Dashboards MUST be created (Grafana or equivalent)
- SLOs MUST be defined and monitored (99.9% uptime, p95 < 500ms)

**Rationale:** V1 has basic observability. V2 requires production-grade monitoring for enterprise use.

### VII. Graceful Degradation - MANDATORY

**System MUST degrade gracefully when dependencies fail:**

**Degradation Strategy:**
- **Critical Path:** Core functionality continues with reduced features
- **Non-Critical Path:** Feature disabled with clear user messaging

**Fallback Requirements:**
- LLM provider failure → Failover chain (OpenAI → Anthropic → Ollama) → Heuristic classification
- Vector search failure → Full-text search → Manual review required
- Redis failure → In-memory cache → Performance degradation warning
- OIDC provider failure → Cached JWKS (max 5 minutes) → Fail closed
- Verification engine failure → Heuristic verification → Flag for manual review

**Circuit Breaker Pattern:**
- External service calls MUST have circuit breakers (open/half-open/closed states)
- Timeouts MUST be configured for all external calls
- Retries MUST use exponential backoff with jitter
- Max retries: 3 for idempotent operations, 0 for non-idempotent

**Rationale:** V1 has some graceful degradation. V2 requires comprehensive fallback strategies for enterprise reliability.

### VIII. Code Quality Standards - MANDATORY

**All code MUST meet high quality standards:**

**Static Analysis:**
- **Python:** mypy (strict mode), ruff, black, bandit (security)
- **TypeScript:** ESLint (strict), Prettier, no any types
- **SQL:** sqlfluff for migrations

**Code Review:**
- All changes MUST be peer-reviewed before merge
- Security-sensitive changes MUST be reviewed by security team
- Breaking changes MUST be reviewed by tech lead
- Code review checklist MUST be completed (security, performance, tests, docs)

**Documentation:**
- All public APIs MUST have docstrings (Google style)
- All complex algorithms MUST have inline comments explaining "why"
- All architectural decisions MUST be documented in ADRs (Architecture Decision Records)
- All breaking changes MUST have migration guides

**Performance:**
- All API endpoints MUST complete within 500ms p95 (excluding long-running tasks)
- All database queries MUST be optimized (no N+1, proper indexes)
- All critical paths MUST be profiled and optimized

**Rationale:** V1 has good code quality. V2 enforces these standards via CI/CD gates.

## Security Requirements

### Authentication & Authorization

- **Authentication:** OIDC with PKCE flow (Auth0, Okta, Azure AD, Keycloak)
- **Authorization:** Cedar or OPA policy engine (fine-grained, attribute-based)
- **Session Management:** JWT with short expiration (15 minutes), refresh tokens (7 days)
- **MFA:** Optional for users, MANDATORY for admins and privileged operations
- **Password Policy:** Min 12 chars, complexity requirements, breach check (HaveIBeenPwned)

### Data Protection

- **Encryption at Rest:** AES-256-GCM for all sensitive data (PII, credentials, secrets)
- **Encryption in Transit:** TLS 1.3 with strong cipher suites only
- **Key Management:** AWS KMS, Azure Key Vault, or HashiCorp Vault (NO local keys)
- **Secrets Management:** Environment variables or secret manager (NO hardcoding, NO .env in git)

### Tenant Isolation

- **Database:** Row-Level Security (RLS) MUST be enabled with `FORCE ROW LEVEL SECURITY`
- **Caching:** Tenant-scoped cache keys (`{tenant_id}:{namespace}:{key}`)
- **Storage:** Tenant-scoped prefixes in S3/blob storage
- **Queries:** ALL database queries MUST filter by tenant_id
- **Validation:** Integration tests MUST verify cross-tenant isolation

### Security Testing

- **SAST:** Bandit, Semgrep on every commit
- **DAST:** OWASP ZAP on staging deployments
- **Dependency Scanning:** Trivy, Safety on every build
- **Secret Scanning:** detect-secrets, GitGuardian on every commit
- **Penetration Testing:** Annual third-party pentest for production

## Compliance Standards

### Audit Trail Requirements (ALCOA+)

All audit events MUST meet ALCOA+ criteria:

- **Attributable:** Every action linked to authenticated user
- **Legible:** Human-readable event descriptions
- **Contemporaneous:** Logged in real-time (max 1 second delay)
- **Original:** First record preserved, no modifications
- **Accurate:** Event details complete and correct
- **Complete:** All relevant context captured
- **Consistent:** Uniform format across all events
- **Enduring:** Stored in WORM storage (S3 Object Lock, Azure Immutable Blobs)
- **Available:** Queryable for auditors

**Audit Events to Log:**
- Authentication (login, logout, failed attempts, MFA)
- Authorization (access grants, denials, policy changes)
- Data access (document views, downloads, exports)
- Data modifications (uploads, edits, deletes)
- Administrative actions (user management, tenant config, policy changes)
- System events (deployments, config changes, security incidents)

### Data Privacy (GDPR/CCPA)

- **Right to Access:** Users can export all their data in machine-readable format
- **Right to Deletion:** Users can request deletion, completed within 30 days
- **Right to Portability:** Data export in JSON format via API
- **Consent Management:** Explicit opt-in for data processing, granular consent
- **Data Minimization:** Collect only data necessary for core functionality
- **Purpose Limitation:** Data used only for stated purposes
- **Retention Policies:** Configurable per tenant (default: 7 years for compliance)

### License Enforcement

- **License Scopes:** FULL (complete regulatory text), CITATION_ONLY (references only)
- **Enforcement:** Backend validation on every API call
- **Per-Framework:** License scope per regulatory framework (SOX, ISO27001, GDPR, HIPAA, SOC2)
- **Content Redaction:** Automatic redaction for CITATION_ONLY licenses
- **Audit:** License usage tracked in audit logs

## Performance & Scalability

### Performance Targets

- **API Latency:**
  - p50: ≤ 100ms
  - p95: ≤ 500ms
  - p99: ≤ 1000ms (excluding long-running background tasks)

- **Background Tasks:**
  - Gap analysis (100 requirements): ≤ 5 minutes
  - Document upload + parsing (100 pages): ≤ 2 minutes
  - Verification (proposed edit): ≤ 5 seconds

- **Throughput:**
  - API requests: ≥ 1000 req/s sustained
  - Concurrent users: ≥ 10,000
  - Database queries: ≥ 5000 queries/s

- **Availability:**
  - Uptime: 99.9% (SLA)
  - Planned maintenance: ≤ 4 hours/month

### Scalability Requirements

- **Horizontal Scaling:** All services MUST be stateless (scale via load balancer)
- **Database Scaling:**
  - Read replicas for read-heavy workloads
  - Connection pooling (min: 10, max: 100 per instance)
  - Query optimization (all queries < 100ms, indexes on all foreign keys)

- **Caching Strategy:**
  - L1: In-memory cache (LRU, 1000 items, 60 seconds TTL)
  - L2: Redis distributed cache (configurable TTL per namespace)
  - Cache invalidation: Namespace-based, event-driven

- **Vector Search:**
  - Current: pgvector IVFFlat (good for < 500K vectors)
  - Migration trigger: p95 latency > 300ms
  - Migration target: HNSW index or Weaviate (at 500K vectors)

- **Background Tasks:**
  - Async task queue: Celery with Redis broker
  - Task priority levels: Critical, High, Normal, Low
  - Task timeout: Configurable per task type
  - Task retry: Exponential backoff with max 3 retries

### Resource Limits

- **File Upload:** Max 50MB per file
- **Request Body:** Max 10MB for JSON payloads
- **Rate Limiting:**
  - Unauthenticated: 10 req/min per IP
  - Authenticated: 100 req/min per user
  - Admin: 1000 req/min per user
  - Burst allowance: 2x rate limit for 10 seconds

## Development Workflow

### Git Workflow

- **Branching:** Feature branches from `main`, PRs for all changes
- **Branch Naming:** `###-feature-name` (e.g., `001-user-auth`)
- **Commits:** Conventional commits format (feat, fix, docs, chore, test, refactor)
- **Merge:** Squash merge to `main` after approval
- **Protection:** `main` branch protected, requires 1 approval + CI passing

### CI/CD Pipeline

**Required Gates (All MUST pass):**
- ✅ Linting (ruff, ESLint)
- ✅ Type checking (mypy strict, TypeScript strict)
- ✅ Unit tests (≥90% coverage)
- ✅ Integration tests (all endpoints)
- ✅ Security scan (Bandit, Trivy, detect-secrets)
- ✅ Performance tests (critical paths within SLO)
- ✅ Accessibility tests (WCAG 2.1 AA compliance)

**Deployment Stages:**
1. **Development:** Auto-deploy on merge to `develop` branch
2. **Staging:** Auto-deploy on merge to `main`, smoke tests required
3. **Production:** Manual approval required, canary deployment (10% → 50% → 100%)

### Code Review Checklist

**Every PR MUST be reviewed for:**
- [ ] Tests written first (TDD red-green-refactor cycle)
- [ ] Test coverage ≥90% (100% for security-critical code)
- [ ] No hardcoded secrets or credentials
- [ ] Input validation with Pydantic models
- [ ] SQL queries use ORM or parameterized queries
- [ ] Error handling comprehensive (no unhandled exceptions)
- [ ] Logging includes correlation IDs
- [ ] Metrics/tracing instrumented for new endpoints
- [ ] API changes backward-compatible (or versioned)
- [ ] Documentation updated (docstrings, ADRs, migration guides)
- [ ] Performance acceptable (profiled if critical path)

## Governance

### Constitution Authority

This constitution supersedes all other development practices and guidelines. When conflicts arise between this constitution and other documentation, **the constitution takes precedence**.

### Amendment Process

**Minor Amendments (Version 1.x.y → 1.x.y+1):**
- Clarifications, typo fixes, non-semantic refinements
- Requires: 1 tech lead approval + PR review
- Notification: Team announcement

**Feature Amendments (Version 1.x.y → 1.x+1.0):**
- New principles, expanded guidance, additional requirements
- Requires: 2 tech lead approvals + team discussion
- Notification: RFC (Request for Comments) process

**Breaking Amendments (Version x.y.z → x+1.0.0):**
- Principle removals, incompatible governance changes
- Requires: Tech lead + engineering manager approval + team vote (2/3 majority)
- Notification: 30-day notice, migration plan required

### Compliance Verification

**Every PR MUST verify:**
- [ ] TDD cycle followed (tests committed before implementation)
- [ ] SOLID principles applied (no god objects, clear separation)
- [ ] Security checklist complete (no hardcoded secrets, input validation, RLS)
- [ ] Test coverage meets thresholds (≥90% overall, 100% security-critical)
- [ ] Documentation updated (docstrings, ADRs if needed)

**Every Release MUST verify:**
- [ ] All CI/CD gates passing
- [ ] Security scan clean (no critical/high vulnerabilities)
- [ ] Performance tests passing (p95 latency within SLO)
- [ ] Accessibility tests passing (WCAG 2.1 AA)
- [ ] Audit trail functional (hash chain intact, WORM storage configured)

### Runtime Guidance

For detailed runtime development guidance, reference:
- `CLAUDE.md` - Claude Code-specific instructions
- `README.md` - Project setup and common commands
- `ARCHITECTURE.md` - System architecture and design decisions
- `API_DOCS.md` - API contracts and OpenAPI specification

**Version**: 1.0.0 | **Ratified**: 2025-11-18 | **Last Amended**: 2025-11-18
