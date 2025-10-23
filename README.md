# ACE Playbook - Adaptive Code Evolution

Self-improving LLM system using the Generator-Reflector-Curator pattern for online learning from execution feedback.

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Guardrails as High-Precision Sensors](#guardrails-as-high-precision-sensors)
- [Quick Start](#quick-start)
    - [Agent Learning Live Loop](#agent-learning-live-loop)
- [Benchmarking & Runtime Adaptation](#benchmarking--runtime-adaptation)
- [Release Notes](#release-notes)
- [Project Structure](#project-structure)
- [Development](#development)
- [Documentation](#documentation)

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
- **Runtime adaptation**: Merge coordinator + runtime adapter enable in-flight learning with optional benchmark harness

### Guardrails as High-Precision Sensors

ACE turns tiny heuristic checks into reusable guardrails without manual babysitting:

- **Detect**: Domain heuristics (e.g., ±0.4% drift, missing "%") label a generator trajectory as a precise failure mode.
- **Distill**: The reflector converts that signal into a lesson (“round to whole percent and append %”).
- **Persist**: The curator records a typed delta with helpful/harmful counters and merges it into the playbook.
- **Reuse**: Runtime adapter + merge coordinator surface the tactic immediately so later tasks cannot repeat the mistake.

This loop mirrors the +8.6% improvements reported on FiNER/XBRL benchmarks—subtle finance errors become actionable context upgrades instead of one-off patches.

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

# Or run the single-domain validation
python examples/single_domain_arithmetic_validation.py

# Generate structured benchmark reports
python scripts/run_benchmark.py benchmarks/finance_subset.jsonl ace_full --output results/ace_full_finance_subset.json
```

### Agent Learning Live Loop

The Agent Learning (Early Experience) harness now lives in this repository under
`ace/agent_learning`.  It reuses the ACE runtime client, curator, and metrics
stack to run a live loop that streams experience back into the playbook.  See
[`docs/combined_quickstart.md`](docs/combined_quickstart.md) for a walkthrough
and run the demo script with:

```bash
python examples/live_loop_quickstart.py
# Or run with your configured DSPy backend
python examples/live_loop_quickstart.py --backend dspy --episodes 10
```

**Environment checklist**

- `OPENROUTER_API_KEY` (preferred), `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`
- `DATABASE_URL` (defaults to `sqlite:///ace_playbook.db`)
- Optional: `OPENROUTER_MODEL` if you want to experiment with different hosted LLMs

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

## Benchmarking & Runtime Adaptation

Use the benchmark harness to compare variants and capture guardrail activity. Detailed notes live in [docs/runtime_benchmarks.rst](docs/runtime_benchmarks.rst).

### Run Baseline vs ACE

```bash
# Baseline: Chain-of-Thought generator only
python scripts/run_benchmark.py benchmarks/finance_subset.jsonl baseline --output results/baseline_finance_subset.json

# Full ACE stack: ReAct generator + runtime adapter + merge coordinator + refinement scheduler
python scripts/run_benchmark.py benchmarks/finance_subset.jsonl ace_full --output results/ace_full_finance_subset.json

# ACE vs baseline live loop comparison (ACE + EE harness)
python benchmarks/run_live_loop_benchmark.py --backend dspy --episodes 10

# Trigger the CI workflow (optional)
gh workflow run ace-benchmark.yml

# Audit agent heuristics locally (sample 20 tasks)
python scripts/audit_agent_scoring.py benchmarks/agent_small.jsonl --sample 20

# Hard finance split (Table 2 replication)
ACE_BENCHMARK_TEMPERATURE=0.9 \
  python scripts/run_benchmark.py benchmarks/finance_hard.jsonl baseline \
  --output results/benchmark/baseline_finance_hard.json

python scripts/run_benchmark.py benchmarks/finance_hard.jsonl ace_full \
  --output results/benchmark/ace_finance_hard_gt.json

ACE_BENCHMARK_USE_GROUND_TRUTH=false \
  python scripts/run_benchmark.py benchmarks/finance_hard.jsonl ace_full \
  --output results/benchmark/ace_finance_hard_no_gt.json

# Agent/AppWorld hard split with conservative heuristics
ACE_BENCHMARK_TEMPERATURE=0.9 \
  python scripts/run_benchmark.py benchmarks/agent_hard.jsonl baseline \
  --output results/benchmark/baseline_agent_hard.json

python scripts/run_benchmark.py benchmarks/agent_hard.jsonl ace_full \
  --output results/benchmark/ace_agent_hard.json

# Quickly sanity-check heuristic thresholds on harder agent tasks
python scripts/audit_agent_scoring.py benchmarks/agent_hard.jsonl --sample 20
```

Key metrics in the JSON output:

- `correct` / `total` – benchmark score
- `promotions`, `new_bullets`, `increments` – curator activity
- `auto_corrections` – guardrail canonical replacements (e.g., finance rounding)
- `format_corrections` – post-process clamps that strip extra words but retain the raw answer for reflection
- `agent_feedback_log` – path to the per-task ledger (`*.feedback.jsonl`) emitted for every run

Populate `benchmarks/RESULTS.md` with the numbers emitted by these commands. The guardrails and heuristics default to a fail-closed posture: when they cannot certify an answer they mark it `unknown`, mirroring the safety constraint highlighted in the paper.

### Add a New Finance Guardrail

1. Edit `ace/utils/finance_guardrails.py` and add an entry to `FINANCE_GUARDRAILS` with `instructions`, `calculator`, and `decimals`.
2. Set `auto_correct=True` if the calculator should override the raw answer.
3. Re-run `scripts/run_benchmark.py` for the relevant dataset.
4. Inspect `results/*.json` to confirm the guardrail triggered and push the refreshed artifact.

Pro tip: keep regenerated results in source control so regressions surface in diffs.

### Add a New Domain in 5 Steps

1. **Scaffold stubs**

   ```bash
   python scripts/scaffold_domain.py claims-processing
   ```

   This creates:

   - `benchmarks/claims-processing.jsonl`
   - `ace/utils/claims-processing_guardrails.py`
   - `docs/domains/claims-processing.rst`

2. **Populate ground truth** – Fill the benchmark file with real tasks (one JSON per line).
3. **Implement guardrails** – Update the guardrail module with instructions, calculators, and `auto_correct` flags.
4. **Run the benchmark** – `python scripts/run_benchmark.py benchmarks/claims-processing.jsonl ace_full --output results/ace_full_claims-processing.json`
5. **Document & commit** – Summarize behavior in the docs stub, review `results/*.json`, and push the changes.

Tip: repeat the harness run periodically (or in CI) so regressions surface immediately.

## Release Notes

See [`docs/release_notes.md`](docs/release_notes.md) for the changelog and upgrade
instructions for the unified ACE + Agent Learning stack. Tag `v1.0.0`
corresponds to the integration referenced in the companion papers.

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

- **Code Quality**: Black formatting, Ruff linting, isort import sorting, autoflake (unused imports)
- **Type Safety**: mypy static type checking
- **Security**: Bandit vulnerability scanning, detect-secrets, Safety (dependency vulnerabilities)
- **Documentation**: Docstring coverage (interrogate), markdown linting
- **Standards**: Conventional commits validation, trailing whitespace, end-of-file fixes
- **Infrastructure**: YAML/JSON/TOML validation, Dockerfile linting, SQL linting
- **Testing**: pytest coverage ≥80% (on push)
- **Complexity**: Radon cyclomatic complexity and maintainability index (on push)
- **Dead Code**: Dead code detection

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

### Comprehensive Documentation (v1.14.0+)

Build and view the complete documentation:

```bash
# Build HTML documentation
make docs

# Serve documentation locally
make docs-serve  # http://localhost:8000
```

**Available Documentation:**

- 📚 **API Reference**: Auto-generated Sphinx docs for all modules
- 🏗️ **Architecture Guide**: System design with Mermaid diagrams ([docs/architecture.md](docs/architecture.md))
- 🎓 **Developer Onboarding**: Setup, workflows, and best practices ([docs/onboarding.md](docs/onboarding.md))
- ⚠️ **Edge Cases**: Error handling and recovery procedures ([docs/edge_cases.md](docs/edge_cases.md))
- 🚀 **Tutorials**: Step-by-step guides ([docs/tutorials/01-quick-start.rst](docs/tutorials/01-quick-start.rst))
- 📖 **Getting Started**: Quick installation guide ([docs/getting_started.rst](docs/getting_started.rst))

### Specification Documents

- **Specification**: `/Users/speed/specs/004-implementing-the-ace/spec.md`
- **Implementation Plan**: `/Users/speed/specs/004-implementing-the-ace/plan.md`
- **Data Model**: `/Users/speed/specs/004-implementing-the-ace/data-model.md`
- **Quick Start Guide**: `/Users/speed/specs/004-implementing-the-ace/quickstart.md`

## License

MIT
# Benchmark automation
python scripts/run_benchmark.py benchmarks/agent_small.jsonl baseline --output results/baseline_agent_small.json
