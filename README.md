# ACE Playbook - Adaptive Code Evolution

Self-improving LLM system using the Generator-Reflector-Curator pattern for online learning from execution feedback.

## Architecture

**Generator-Reflector-Curator Pattern:**

- **Generator**: DSPy ReAct/CoT modules that execute tasks using playbook strategies
- **Reflector**: Analyzes outcomes and extracts labeled insights (Helpful/Harmful/Neutral)
- **Curator**: Pure Python semantic deduplication with FAISS (0.8 cosine similarity threshold)

## Key Features

- **Append-only playbook**: Never rewrite bullet content, only increment counters
- **Semantic deduplication**: 0.8 cosine similarity threshold prevents context collapse
- **Staged rollout**: shadow → staging → prod with automated promotion gates
- **Multi-domain isolation**: Per-tenant namespaces with separate FAISS indices
- **Rollback procedures**: <5 minute automated rollback on regression detection
- **Performance budgets**: ≤10ms P50 playbook retrieval, ≤+15% end-to-end overhead
- **Observability metrics**: Prometheus-format metrics for monitoring (T065)
- **Guardrail monitoring**: Automated rollback on performance regression (T066)
- **Docker support**: Full containerization with Docker Compose (T067)
- **E2E testing**: Comprehensive smoke tests for production readiness (T068)

## Quick Start

### Local Installation

```bash
# Install dependencies with uv (fast package manager)
uv pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)

# Initialize database
alembic upgrade head

# Run smoke tests
pytest tests/e2e/test_smoke.py -v

# Start with examples
python examples/arithmetic_learning.py
```

### Docker Compose (Recommended for Production)

```bash
# Create .env file with your API keys
echo "OPENAI_API_KEY=sk-..." > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f ace

# Stop services
docker-compose down
```

### Observability

```python
# Export Prometheus metrics
from ace.ops import get_metrics_collector

collector = get_metrics_collector()
print(collector.export_prometheus())
```

### Guardrail Monitoring

```python
# Check for performance regressions
from ace.ops import create_guardrail_monitor

monitor = create_guardrail_monitor(session)
trigger = monitor.check_guardrails("customer-acme")
if trigger:
    print(f"Rollback triggered: {trigger.reason}")
```

## Project Structure

```text
ace-playbook/
├── ace/                    # Core ACE framework
│   ├── generator/         # DSPy Generator modules
│   ├── reflector/         # Reflector analysis
│   ├── curator/           # Semantic deduplication
│   ├── models/            # Data models and schemas
│   ├── repositories/      # Database access layer
│   ├── utils/             # Embeddings, FAISS, logging
│   └── ops/               # Operations (metrics, guardrails, training)
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end smoke tests
├── examples/               # Usage examples
├── config/                 # Configuration files
├── alembic/                # Database migrations
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Local development stack
└── docs/                   # Additional documentation
```

## Development

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install
pre-commit install --hook-type commit-msg

# Run manually on all files
pre-commit run --all-files

# Skip hooks for a specific commit (use sparingly)
git commit --no-verify -m "WIP: temporary commit"
```

**Installed Hooks:**

- **Code Quality**: Black formatting, Ruff linting, isort import sorting
- **Type Safety**: mypy static type checking
- **Security**: Bandit vulnerability scanning, detect-secrets
- **Documentation**: Docstring coverage (interrogate), markdown linting
- **Standards**: Conventional commits validation, trailing whitespace, end-of-file fixes
- **Infrastructure**: YAML/JSON/TOML validation, Dockerfile linting, SQL linting

### Manual Testing

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy ace/

# Code formatting
black ace/ tests/
ruff check ace/ tests/

# Security scan
bandit -r ace/

# Docstring coverage
interrogate -vv ace/
```

## Documentation

- **Specification**: `/Users/speed/specs/004-implementing-the-ace/spec.md`
- **Implementation Plan**: `/Users/speed/specs/004-implementing-the-ace/plan.md`
- **Data Model**: `/Users/speed/specs/004-implementing-the-ace/data-model.md`
- **Quick Start Guide**: `/Users/speed/specs/004-implementing-the-ace/quickstart.md`

## License

MIT
