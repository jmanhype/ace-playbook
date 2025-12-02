# Implementation Plan: Unified Management of Entitlements and Identity Subsystem (UMES)

**Branch**: `001-umes-identity-subsystem` | **Date**: 2025-12-02 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-umes-identity-subsystem/spec.md`

## Summary

UMES is a cloud-agnostic, multi-tenant identity and authorization subsystem that provides:
- Unified authentication across multiple IdP providers (Local, OIDC, SAML)
- Cloud-agnostic key management via adapters (6 KMS providers)
- Capability-based + role-based authorization with tenant isolation
- API key lifecycle management
- Comprehensive audit logging
- Zero-code configuration changes for provider swapping

**Technical Approach**: Adapter pattern for all external dependencies (KMS, IdP, storage), strict TDD with 100% coverage, SOLID architecture, container-based deployment for portability.

## Architectural Vision

1. **Adapter Layer**: All external dependencies (KMS, IdP, database) isolated behind adapters implementing common interfaces. Enables zero-code provider swapping - administrator changes environment variables only.

2. **Service Layer**: Core business logic (authentication, authorization, token generation, API key management) depends on adapter abstractions, not concrete implementations. Enforces SOLID principles and dependency inversion.

3. **Authorization Engine**: Capability-based + role-based evaluation with tenant-aware context. Authorization decisions are pure functions of (user, tenant, resource, action) - identical output across all deployment environments.

4. **Token Strategy**: JWT (stateless, signed by KMS) for distributed validation + opaque tokens (database-backed) for revocable sessions. Services choose strategy based on requirements (performance vs revocation immediacy).

5. **Audit Subsystem**: Immutable append-only event log for all identity operations. Decoupled from main service layer via event bus pattern - audit failures don't block critical operations.

6. **Graceful Degradation**: Circuit breakers for all external calls (KMS, IdP), cached authorization decisions (2-minute TTL), emergency local crypto mode if KMS fails. System fails closed when degradation exceeds safety thresholds.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript 5.x/React 18 (future admin UI - v1 is API-only)
**Primary Dependencies**: FastAPI 0.104+, SQLAlchemy 2.0, PyJWT, httpx (async HTTP), pydantic 2.0 (validation)
**Storage**: PostgreSQL 15+ (primary database with RLS), Redis 7+ (caching, session storage, rate limiting)
**Testing**: pytest 7.4+, pytest-asyncio, httpx (async test client), Testcontainers (integration tests)
**Target Platform**: Container (Docker), Kubernetes (all clouds), Docker Compose (single-host), VM deployment (on-prem)
**Project Type**: Backend API service (single project - no frontend in v1)
**Performance Goals**: Token validation ≤50ms p95, authorization ≤100ms p95, token generation ≤200ms p95, 10,000 req/s sustained
**Constraints**: 99.9% uptime SLA, multi-cloud deployment identical behavior, zero-code configuration changes, 100% test coverage
**Scale/Scope**: 100,000 concurrent users, 10,000 req/s token validation, 6 KMS adapters, 3 IdP adapters, 30 functional requirements

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Test-First Development (TDD) - ✅ PASS

- **100% coverage for UMES** (entire codebase is security-critical): Plan includes comprehensive test strategy
- **Test types**: Unit (all business logic), Integration (all endpoints + adapters), Smoke (critical paths), Contract (adapter interfaces), Multi-cloud (provider parity), Security (tenant isolation), Performance (latency SLOs)
- **Test-first workflow**: Tasks will be structured as: Write failing test → Implement minimum code → Refactor → Repeat

**Enforcement**: All tasks in tasks.md will follow strict red-green-refactor ordering. Test tasks always precede implementation tasks.

### I.A Test Pass Gate - ✅ PASS

- **100% pass rate before completion**: Plan includes pytest configuration with strict pass requirements, no flaky test tolerance
- **Failure protocol**: Documented in tasks - if test fails, create blocking Beads issue, do not proceed
- **CI/CD gates**: All tests must pass before merge (enforced by CI pipeline)

### II. SOLID Architecture - ✅ PASS

**Adapter Pattern**:
- KMS adapters: LocalKMS, GcpKMS, AwsKMS, AzureKeyVault, OracleKMS, OpenBaoKMS (all implement `KMSAdapter` protocol)
- IdP adapters: LocalIdP, OIDCIdP, SAMLIdP (all implement `IdPAdapter` protocol)
- Repository pattern: UserRepository, TenantRepository, AuditRepository (all implement `Repository` protocol)

**Dependency Inversion**: Services receive adapters via constructor injection (FastAPI dependency injection). No service imports concrete adapter implementations.

**Interface Segregation**: Separate interfaces for read vs write operations, token signing vs verification, authentication vs token refresh.

### III. Security-First Design - ✅ PASS

**Defense in Depth**:
- Layer 1 (Network): Rate limiting via slowapi + Redis
- Layer 2 (Authentication): JWT + OIDC/SAML, MFA support via IdP
- Layer 3 (Authorization): Capability + role evaluation with tenant context
- Layer 4 (Data Access): PostgreSQL RLS with `FORCE ROW LEVEL SECURITY`
- Layer 5 (Application): Pydantic strict validation, parameterized queries via SQLAlchemy
- Layer 6 (Audit): Immutable audit log with hash chain for integrity

**Security Requirements Met**:
- Secrets in environment variables / secret managers: ✅ (no `.env` in git, `.env.example` only)
- Industry-standard protocols: ✅ (OIDC, OAuth2, SAML)
- Policy-based authorization: ✅ (capability + role engine)
- KMS-backed encryption: ✅ (all secrets encrypted via KMS adapter)
- Pydantic validation: ✅ (strict mode, all API inputs validated)
- ORM queries only: ✅ (SQLAlchemy Core/ORM, no raw SQL strings)
- RLS enabled: ✅ (migration includes RLS policies for all tenant-scoped tables)
- Rate limiting: ✅ (slowapi middleware)
- Error sanitization: ✅ (production mode strips sensitive details)
- Tenant isolation tests: ✅ (integration test suite validates cross-tenant isolation)

### IV. Cloud-Agnostic Design - ✅ PASS

**Target Environments Supported**: GCP, AWS, Azure, Oracle, Hetzner, on-prem
**Adapters Required**: 6 KMS, 3 IdP, 1 storage (PostgreSQL - all clouds), 1 cache (Redis - optional)
**Configuration-Only Swapping**: All adapter selection via environment variables (`KMS_PROVIDER=gcp`, `IDP_PROVIDER=oidc`)
**Multi-Cloud Integration Tests**: Test matrix runs identical test suite on GCP, AWS, Azure, Oracle, Hetzner, local (Testcontainers)

### V. API Versioning & Stability - ✅ PASS

**Versioning Strategy**: URL versioning `/api/v1/` (all endpoints), semantic versioning for releases
**Stability Requirements**: OpenAPI 3.1 contract, deprecation warnings in responses (custom header `X-Deprecation-Warning`), migration guide for breaking changes
**Backward Compatibility**: v1 API locked after GA release (12-month support window for previous major version)

### VI. Observability & Monitoring - ✅ PASS

**Metrics**: Prometheus metrics via `prometheus-fastapi-instrumentator` (request rates, latencies p50/p95/p99, error rates, KMS latencies, IdP latencies, token generation rates, authorization check latencies)
**Logging**: Structlog JSON logging with correlation IDs (`trace_id`, `tenant_id`, `user_id` in all log entries)
**Tracing**: OpenTelemetry with Jaeger exporter (100% error traces, 10% success traces - configurable)
**Health Checks**: FastAPI `/health` (liveness), `/ready` (readiness - checks DB + Redis), `/live` (liveness - always 200)

### VII. Graceful Degradation - ✅ PASS

**Critical Path**: Authentication continues with cached tokens (5-minute max) when IdP fails
**Non-Critical Path**: Audit log buffering (in-memory queue) when database write fails temporarily

**Fallback Strategies**:
- KMS failure → Emergency local crypto mode (alerts triggered) → Fail closed after 10 minutes
- IdP failure → Cached tokens (5-minute max) → Fail closed for new authentication
- Database failure → Cached authorization decisions (2-minute TTL) → Fail closed for writes
- Redis failure → In-memory LRU cache → Performance degradation warning

**Circuit Breakers**: All external service calls have circuit breakers (open/half-open/closed), timeouts (KMS: 5s, IdP: 10s), exponential backoff with jitter (max 3 retries for idempotent operations)

⚠️ **CRITICAL**: Tasks will include specific implementation tasks for:
- Circuit breaker implementation for KMS adapters
- Circuit breaker implementation for IdP adapters
- Emergency local crypto mode for KMS failure
- Cached token validation for IdP failure
- In-memory fallback cache for Redis failure

### VIII. Code Quality Standards - ✅ PASS

**Static Analysis**: mypy (strict mode), ruff (linter + formatter), black (formatting), bandit (security), sqlfluff (SQL migrations)
**Documentation**: Google-style docstrings for all public APIs, ADRs for architectural decisions (stored in `docs/adr/`)
**Performance**: All critical paths profiled with py-spy, optimized to meet p95 latency SLOs

### IX. UX Consistency - ✅ PASS

**Error Response Format**: Standardized JSON error responses with `error.code`, `error.message`, `error.correlation_id`
**Error Codes**: Defined in spec (AUTHENTICATION_FAILED, AUTHORIZATION_DENIED, TENANT_ISOLATION_VIOLATION, etc.)

### X. Accessibility - ✅ N/A (v1 is API-only, no UI)

Future UI will require WCAG 2.1 AA compliance.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-umes-identity-subsystem/
├── plan.md              # This file (/speckit.plan output)
├── spec.md              # Feature specification
├── research.md          # Phase 0 output (technical research)
├── data-model.md        # Phase 1 output (entity/schema design)
├── quickstart.md        # Phase 1 output (getting started guide)
├── contracts/           # Phase 1 output (API contracts)
│   ├── openapi.yaml     # OpenAPI 3.1 specification
│   ├── auth.yaml        # Authentication endpoints
│   ├── authz.yaml       # Authorization endpoints
│   ├── tokens.yaml      # Token management endpoints
│   ├── api-keys.yaml    # API key management endpoints
│   ├── users.yaml       # User management endpoints
│   ├── tenants.yaml     # Tenant management endpoints
│   └── audit.yaml       # Audit log endpoints
├── checklists/          # Quality validation checklists
│   └── requirements.md  # Specification quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)

**Structure Decision**: Single backend API project (Python). UMES v1 is API-only with no frontend. Future admin UI will be separate project.

```text
backend/
├── src/
│   ├── umes/                  # Main package
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI application entry point
│   │   ├── config.py          # Configuration (Pydantic Settings)
│   │   ├── dependencies.py    # FastAPI dependency injection
│   │   │
│   │   ├── adapters/          # Adapter implementations (KMS, IdP, repositories)
│   │   │   ├── __init__.py
│   │   │   ├── kms/           # KMS adapters
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py    # KMSAdapter protocol
│   │   │   │   ├── local.py   # LocalKMS (dev)
│   │   │   │   ├── gcp.py     # GCP KMS
│   │   │   │   ├── aws.py     # AWS KMS
│   │   │   │   ├── azure.py   # Azure Key Vault
│   │   │   │   ├── oracle.py  # Oracle KMS
│   │   │   │   └── openbao.py # OpenBao (on-prem)
│   │   │   ├── idp/           # IdP adapters
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py    # IdPAdapter protocol
│   │   │   │   ├── local.py   # Local username/password
│   │   │   │   ├── oidc.py    # OIDC (Auth0/Okta/Keycloak/AAD/Cloudflare)
│   │   │   │   └── saml.py    # SAML
│   │   │   └── repositories/  # Data access repositories
│   │   │       ├── __init__.py
│   │   │       ├── base.py    # Repository base classes
│   │   │       ├── user.py    # UserRepository
│   │   │       ├── tenant.py  # TenantRepository
│   │   │       ├── role.py    # RoleRepository
│   │   │       ├── api_key.py # APIKeyRepository
│   │   │       └── audit.py   # AuditRepository
│   │   │
│   │   ├── models/            # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── tenant.py
│   │   │   ├── membership.py
│   │   │   ├── role.py
│   │   │   ├── entitlement.py
│   │   │   ├── api_key.py
│   │   │   └── audit_event.py
│   │   │
│   │   ├── schemas/           # Pydantic models (API request/response)
│   │   │   ├── __init__.py
│   │   │   ├── auth.py        # Authentication request/response schemas
│   │   │   ├── token.py       # Token schemas
│   │   │   ├── user.py        # User schemas
│   │   │   ├── tenant.py      # Tenant schemas
│   │   │   ├── role.py        # Role schemas
│   │   │   ├── api_key.py     # API key schemas
│   │   │   └── audit.py       # Audit event schemas
│   │   │
│   │   ├── services/          # Business logic services
│   │   │   ├── __init__.py
│   │   │   ├── auth.py        # Authentication service
│   │   │   ├── authz.py       # Authorization service
│   │   │   ├── token.py       # Token generation/validation service
│   │   │   ├── api_key.py     # API key management service
│   │   │   ├── user.py        # User management service
│   │   │   ├── tenant.py      # Tenant management service
│   │   │   └── audit.py       # Audit logging service
│   │   │
│   │   ├── api/               # FastAPI routers
│   │   │   ├── __init__.py
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── auth.py    # /api/v1/auth/*
│   │   │   │   ├── authz.py   # /api/v1/authz/*
│   │   │   │   ├── tokens.py  # /api/v1/tokens/*
│   │   │   │   ├── api_keys.py # /api/v1/api-keys/*
│   │   │   │   ├── users.py   # /api/v1/users/*
│   │   │   │   ├── tenants.py # /api/v1/tenants/*
│   │   │   │   └── audit.py   # /api/v1/audit/*
│   │   │   └── health.py      # /health, /ready, /live
│   │   │
│   │   ├── middleware/        # FastAPI middleware
│   │   │   ├── __init__.py
│   │   │   ├── auth.py        # JWT validation middleware
│   │   │   ├── tenant.py      # Tenant context middleware
│   │   │   ├── correlation.py # Correlation ID injection
│   │   │   └── rate_limit.py  # Rate limiting middleware
│   │   │
│   │   └── utils/             # Utilities
│   │       ├── __init__.py
│   │       ├── crypto.py      # Cryptography utilities (hashing, encryption)
│   │       ├── circuit_breaker.py # Circuit breaker implementation
│   │       ├── cache.py       # Cache abstraction (Redis + in-memory)
│   │       └── logging.py     # Structured logging setup
│   │
│   └── migrations/            # Alembic database migrations
│       ├── env.py
│       ├── script.py.mako
│       └── versions/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   │
│   ├── unit/                  # Unit tests (isolated, fast)
│   │   ├── __init__.py
│   │   ├── adapters/
│   │   │   ├── kms/           # KMS adapter unit tests
│   │   │   ├── idp/           # IdP adapter unit tests
│   │   │   └── repositories/  # Repository unit tests
│   │   ├── services/          # Service unit tests
│   │   ├── middleware/        # Middleware unit tests
│   │   └── utils/             # Utility unit tests
│   │
│   ├── integration/           # Integration tests (database, HTTP, adapters)
│   │   ├── __init__.py
│   │   ├── api/               # API endpoint integration tests
│   │   │   ├── test_auth.py
│   │   │   ├── test_authz.py
│   │   │   ├── test_tokens.py
│   │   │   ├── test_api_keys.py
│   │   │   ├── test_users.py
│   │   │   ├── test_tenants.py
│   │   │   └── test_audit.py
│   │   ├── adapters/          # Adapter integration tests (real services via Testcontainers)
│   │   │   ├── test_kms_adapters.py      # All KMS providers
│   │   │   └── test_idp_adapters.py      # All IdP providers
│   │   └── test_tenant_isolation.py      # Cross-tenant isolation tests
│   │
│   ├── smoke/                 # Smoke tests (critical user paths)
│   │   ├── __init__.py
│   │   ├── test_authentication_flow.py   # End-to-end auth
│   │   ├── test_authorization_flow.py    # End-to-end authz
│   │   └── test_api_key_lifecycle.py     # API key create/rotate/revoke
│   │
│   ├── contract/              # Contract tests (adapter interface compliance)
│   │   ├── __init__.py
│   │   ├── test_kms_contract.py          # All KMS adapters implement protocol
│   │   └── test_idp_contract.py          # All IdP adapters implement protocol
│   │
│   ├── security/              # Security tests (tenant isolation, authz enforcement)
│   │   ├── __init__.py
│   │   ├── test_tenant_isolation.py
│   │   ├── test_authorization_bypass.py
│   │   └── test_token_security.py
│   │
│   └── performance/           # Performance tests (latency SLOs)
│       ├── __init__.py
│       ├── test_token_validation_latency.py
│       ├── test_authorization_latency.py
│       └── test_throughput.py
│
├── docker/
│   ├── Dockerfile             # Production container image
│   ├── Dockerfile.dev         # Development container (hot reload)
│   └── docker-compose.yml     # Local development stack (UMES + PostgreSQL + Redis)
│
├── k8s/                       # Kubernetes manifests (Helm chart)
│   ├── Chart.yaml
│   ├── values.yaml
│   ├── values-gcp.yaml        # GCP-specific overrides
│   ├── values-aws.yaml        # AWS-specific overrides
│   ├── values-azure.yaml      # Azure-specific overrides
│   ├── values-oracle.yaml     # Oracle-specific overrides
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── configmap.yaml
│       ├── secret.yaml        # Placeholder - real secrets from cloud secret manager
│       └── ingress.yaml
│
├── docs/
│   ├── adr/                   # Architecture Decision Records
│   │   ├── 001-adapter-pattern.md
│   │   ├── 002-jwt-vs-opaque-tokens.md
│   │   ├── 003-graceful-degradation.md
│   │   └── 004-multi-cloud-testing.md
│   ├── deployment/
│   │   ├── gcp.md             # GCP deployment guide
│   │   ├── aws.md             # AWS deployment guide
│   │   ├── azure.md           # Azure deployment guide
│   │   ├── oracle.md          # Oracle deployment guide
│   │   ├── hetzner.md         # Hetzner deployment guide
│   │   └── on-prem.md         # On-prem deployment guide
│   └── integration/
│       └── sdk-guide.md       # Service integration guide (for other Hextropian services)
│
├── scripts/
│   ├── init-db.sh             # Database initialization (create DB, run migrations)
│   ├── run-dev.sh             # Run dev server with hot reload
│   └── run-tests.sh           # Run full test suite
│
├── .env.example               # Environment variable template (checked into git)
├── .gitignore
├── pyproject.toml             # Python project metadata (Poetry/pip-tools)
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── mypy.ini                   # mypy configuration
├── ruff.toml                  # ruff configuration
└── README.md                  # Project README
```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations. All constitution requirements are met by this plan.

---

## Phase 0: Research & Technical Decisions (NEXT)

Phase 0 will resolve all technical unknowns and generate `research.md` with decisions for:

1. **KMS Adapter Implementation Strategy**: How to implement 6 KMS adapters with consistent interface (research PyJWT, cryptography library, cloud SDK integration patterns)

2. **IdP Adapter Implementation Strategy**: OIDC vs SAML implementation libraries (research python-jose, python-saml, httpx for OIDC, assertion validation patterns)

3. **Row-Level Security (RLS) Implementation**: PostgreSQL RLS policy design for tenant isolation (research RLS patterns, `FORCE ROW LEVEL SECURITY`, session context variables)

4. **Circuit Breaker & Graceful Degradation Patterns**: Python circuit breaker libraries (research pybreaker, tenacity, aiobreaker for async support)

5. **Multi-Cloud Testing Strategy**: Testcontainers vs cloud emulators vs real cloud accounts (research Testcontainers Python, LocalStack, azurite, GCP emulators)

6. **Performance Optimization for p95 Latency SLOs**: Caching strategies, connection pooling, async I/O patterns (research aiocache, asyncpg vs psycopg3, Redis pipelining)

7. **Audit Log Integrity**: Hash chain implementation for tamper-evident audit logs (research Merkle tree patterns, append-only log structures)

8. **Token Revocation Strategy**: JWT revocation list vs opaque tokens trade-offs (research token introspection, revocation list storage patterns, cache invalidation)

**Output**: `research.md` with decisions, rationale, and alternatives considered for each topic.

---

## Phase 1: Design & Contracts (AFTER Phase 0)

Phase 1 will generate:

1. **data-model.md**: Entity-relationship diagram, database schema, RLS policies, indexes, constraints
2. **contracts/openapi.yaml**: Complete OpenAPI 3.1 specification for all endpoints
3. **quickstart.md**: Getting started guide for developers integrating with UMES

**Output**: Design artifacts ready for task generation in `/speckit.tasks` command.
