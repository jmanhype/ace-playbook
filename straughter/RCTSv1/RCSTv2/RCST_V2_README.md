# RCST v2 - Enterprise Regulatory Compliance Platform

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.3+-blue.svg)](https://www.typescriptlang.org/)
[![Code Coverage](https://img.shields.io/badge/coverage-90%25+-green.svg)](backend/htmlcov/index.html)

RCST v2 is an enterprise regulatory compliance platform that enables organizations to analyze Standard Operating Procedures (SOPs) against regulatory frameworks (FDA 21 CFR Part 11, EU GMP, ISO 13485, HIPAA, GDPR), perform formal verification of proposed SOP edits, and maintain ALCOA+ compliant provenance tracking for audit purposes.

## Features

- **Automated Gap Analysis**: AI-powered analysis of SOPs against 50+ regulatory frameworks
- **Formal Verification**: Logic-based verification of proposed SOP edits using PyReason + Z3
- **ALCOA+ Provenance**: Complete audit trail with hash-chained, tamper-evident logs
- **Multi-Tenant Architecture**: Enterprise-ready SaaS with row-level security
- **SOC 2 / HIPAA Ready**: Compliance by design with defense-in-depth security
- **Real-Time Collaboration**: Live updates, document version control, and approval workflows

## Architecture

### Tech Stack

**Backend:**
- Python 3.11+ with FastAPI
- PostgreSQL 15+ with pgvector for semantic search
- Redis for caching and background task queues
- Celery for async processing
- OpenTelemetry for observability

**Frontend:**
- React 18 with TypeScript 5.3+
- Vite for build tooling
- TanStack Query for server state
- Tailwind CSS for styling

**Infrastructure:**
- Docker & Docker Compose for local development
- Kubernetes for production deployment
- AWS/Azure/GCP compatible

### Design Principles

- **Test-First Development**: 90%+ code coverage, 100% for security-critical code
- **SOLID Architecture**: Repository, Adapter, Factory, Strategy, Facade patterns
- **Zero-Trust Security**: 6-layer defense-in-depth with OIDC authentication
- **Compliance by Design**: ALCOA+, SOC 2 Type II, HIPAA BAA-ready, GDPR compliant

## Quick Start

### Prerequisites

- **Docker Desktop** (v20.10+) - [Install](https://www.docker.com/products/docker-desktop/)
- **Python 3.11+** - [Install](https://www.python.org/downloads/)
- **Node.js 20 LTS** - [Install](https://nodejs.org/)
- **Poetry** (Python package manager) - [Install](https://python-poetry.org/docs/#installation)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/rcst-v2.git
cd RCSTv2
```

### 2. Environment Setup

Create `.env` file in project root:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://rcst_user:rcst_password@localhost:5432/rcst_db

# Redis
REDIS_URL=redis://:rcst_redis_password@localhost:6379/0

# Security (CHANGE IN PRODUCTION!)
SECRET_KEY=dev-secret-key-minimum-32-characters-long!
JWT_SECRET_KEY=dev-jwt-secret-key-minimum-32-characters!

# LLM API Keys (optional for development)
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# OIDC (configure with your identity provider)
OIDC_DISCOVERY_URL=https://your-oidc-provider.com/.well-known/openid-configuration
OIDC_CLIENT_ID=your-client-id
OIDC_CLIENT_SECRET=your-client-secret

# CORS
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Observability
LOG_LEVEL=DEBUG
ENABLE_TRACING=true
```

### 3. Start Development Stack

```bash
# Start all services (PostgreSQL, Redis, Backend, Frontend, Jaeger, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f backend
```

Services will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Jaeger Tracing**: http://localhost:16686
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### 4. Run Database Migrations

```bash
# Enter backend container
docker-compose exec backend bash

# Run migrations
poetry run alembic upgrade head

# (Optional) Create initial seed data
poetry run python scripts/seed_data.py
```

### 5. Verify Installation

```bash
# Check backend health
curl http://localhost:8000/health

# Check API version
curl http://localhost:8000/api/v1/

# Run backend tests
docker-compose exec backend poetry run pytest

# Run frontend tests
docker-compose exec frontend npm test
```

## Development Workflow

### Backend Development

```bash
# Install dependencies
cd backend
poetry install

# Run tests with coverage
poetry run pytest --cov=src --cov-report=html

# Run type checking
poetry run mypy src

# Run linting
poetry run ruff check src
poetry run black --check src

# Format code
poetry run black src
poetry run ruff check --fix src

# Run security scan
poetry run bandit -r src
```

### Frontend Development

```bash
# Install dependencies
cd frontend
npm install

# Run dev server
npm run dev

# Run tests
npm test

# Run E2E tests
npm run test:e2e

# Run accessibility tests
npm run test:accessibility

# Type check
npm run type-check

# Lint
npm run lint
npm run format:check
```

### Creating Database Migrations

```bash
# Auto-generate migration from model changes
docker-compose exec backend poetry run alembic revision --autogenerate -m "Add user table"

# Apply migrations
docker-compose exec backend poetry run alembic upgrade head

# Rollback last migration
docker-compose exec backend poetry run alembic downgrade -1
```

## Project Structure

```
RCSTv2/
├── backend/              # Python FastAPI backend
│   ├── src/
│   │   ├── api/         # API endpoints (v1/)
│   │   ├── models/      # Domain, database, and schema models
│   │   ├── services/    # Business logic
│   │   ├── repositories/ # Data access layer
│   │   ├── adapters/    # External service integrations
│   │   ├── core/        # Configuration, database, security
│   │   ├── tasks/       # Celery background tasks
│   │   └── utils/       # Shared utilities
│   ├── tests/           # Pytest tests (unit, integration, E2E)
│   ├── alembic/         # Database migrations
│   └── pyproject.toml   # Poetry dependencies
├── frontend/            # React TypeScript frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API clients
│   │   ├── stores/      # Zustand state management
│   │   └── hooks/       # Custom React hooks
│   ├── tests/           # Vitest + Playwright tests
│   └── package.json     # NPM dependencies
├── infrastructure/      # Kubernetes, Terraform configs
├── .github/workflows/   # CI/CD pipelines
└── docker-compose.yml   # Local development stack
```

## Testing

### Backend Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
poetry run pytest tests/unit/services/test_gap_analyzer.py

# Run tests matching pattern
poetry run pytest -k "test_gap_analysis"

# Run property-based tests
poetry run pytest tests/property/

# Run performance tests
poetry run locust -f tests/performance/load/locustfile.py
```

### Frontend Testing

```bash
# Unit and component tests
npm test

# E2E tests
npm run test:e2e

# Accessibility tests
npm run test:accessibility

# Coverage report
npm run test:coverage
```

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Security

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities. Email security@your-org.com instead.

### Security Features

- OIDC authentication with PKCE flow
- JWT access tokens (15 min expiry)
- Role-based access control (6 roles: Viewer, Editor, Approver, Auditor, TenantAdmin, SystemAdmin)
- Row-level security for multi-tenant isolation
- Hash-chained audit logs in WORM storage
- Encrypted at rest (AES-256) and in transit (TLS 1.3)
- Rate limiting per authentication level
- Comprehensive security testing (SAST, DAST, penetration testing)

## Compliance

RCST v2 is designed for compliance with:

- **ALCOA+**: Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, Available
- **SOC 2 Type II**: Security, Availability, Confidentiality, Processing Integrity, Privacy
- **HIPAA**: BAA-ready with encryption, audit logs, access controls
- **GDPR**: Right to access, deletion (30 days), portability, granular consent

## Observability

### Metrics (Prometheus)

Available at http://localhost:9090

Key metrics:
- `http_requests_total`: Request count by endpoint, method, status
- `http_request_duration_seconds`: Request latency (p50, p95, p99)
- `db_pool_connections`: Database connection pool usage
- `cache_hit_rate`: Redis cache performance
- `llm_api_calls_total`: LLM provider usage and costs

### Tracing (Jaeger)

Available at http://localhost:16686

End-to-end distributed tracing for:
- API requests
- Database queries
- Cache operations
- LLM API calls
- Background tasks

### Dashboards (Grafana)

Available at http://localhost:3000 (admin/admin)

Pre-configured dashboards:
- API Performance (latency, throughput, error rates)
- Database Performance (query latency, connection pool)
- Cache Performance (hit/miss rates, evictions)
- Business Metrics (gap analyses, verifications, accuracy scores)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Support

- **Documentation**: [specs/001-rcst-v2-platform/](specs/001-rcst-v2-platform/)
- **Issues**: [GitHub Issues](https://github.com/your-org/rcst-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rcst-v2/discussions)
- **Email**: support@your-org.com
