# UMES Backend

Unified Management of Entitlements and Identity Subsystem

## Project Structure

```
backend/
├── src/
│   └── umes/
│       ├── adapters/          # External service adapters
│       │   ├── kms/          # KMS adapters (AWS, GCP, Azure, Oracle, OpenBao, Local)
│       │   └── idp/          # IdP adapters (Local, OIDC, SAML)
│       ├── models/            # SQLAlchemy ORM models
│       ├── schemas/           # Pydantic schemas (request/response)
│       ├── services/          # Business logic services
│       ├── api/              # FastAPI routes
│       │   └── v1/
│       │       └── routes/   # API v1 route definitions
│       ├── middleware/        # FastAPI middleware
│       └── utils/            # Utility functions
└── tests/
    ├── unit/                 # Unit tests (per module)
    │   ├── adapters/
    │   ├── models/
    │   ├── schemas/
    │   ├── services/
    │   ├── api/
    │   ├── middleware/
    │   └── utils/
    ├── integration/          # Integration tests
    │   ├── auth/            # Authentication flows
    │   ├── authz/           # Authorization flows
    │   ├── audit/           # Audit logging
    │   └── multi_cloud/     # Multi-cloud deployment
    ├── contract/             # API contract tests
    ├── security/             # Security tests
    ├── smoke/                # Smoke tests
    └── performance/          # Performance/load tests
```

## Technology Stack

- **Framework**: FastAPI 0.104+
- **Database**: PostgreSQL 15+ with Row-Level Security (RLS)
- **ORM**: SQLAlchemy 2.0 (async)
- **Cache**: Redis 7+ (two-level caching with aiocache)
- **KMS**: Multi-provider support (AWS, GCP, Azure, Oracle, OpenBao, Local)
- **IdP**: Multi-provider support (Local/argon2, OIDC, SAML)
- **Testing**: pytest 7.4+, Testcontainers

## Development

### Setup

```bash
# Install dependencies
pip install -e .

# Run tests
pytest backend/tests/

# Run with coverage
pytest --cov=backend/src/umes --cov-report=html
```

### Architecture Principles

1. **Test-First Development (TDD)**: 100% test coverage requirement
2. **Cloud-Agnostic**: Runs identically on GCP, AWS, Azure, Oracle, Hetzner, on-prem
3. **Adapter Pattern**: Pluggable KMS and IdP providers
4. **Multi-Tenancy**: Row-Level Security for tenant isolation
5. **Audit Everything**: Immutable audit log with hash chaining

## Documentation

- **Specification**: `/specs/001-umes-identity-subsystem/spec.md`
- **Technical Plan**: `/specs/001-umes-identity-subsystem/plan.md`
- **Implementation Tasks**: `/specs/001-umes-identity-subsystem/tasks.md`
- **API Contract**: `/specs/001-umes-identity-subsystem/contracts/openapi.yaml`

## Status

**Phase**: 1 - Project Setup
**Current Task**: T001 - Initialize backend project structure ✅
**Next**: T002 - Create pyproject.toml with all dependencies
