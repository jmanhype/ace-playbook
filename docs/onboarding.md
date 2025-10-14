# Developer Onboarding Guide

Welcome to ACE Playbook development! This guide will help you set up your development environment and understand how to contribute to the project.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Development Workflows](#development-workflows)
4. [Testing Guidelines](#testing-guidelines)
5. [Common Workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Development Environment Setup

### Prerequisites

- **Python 3.11+** (Check: `python --version`)
- **Git** (Check: `git --version`)
- **Docker** (Optional, for containerized development)
- **8GB RAM minimum** (16GB recommended)
- **OpenAI or Anthropic API key**

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/yourusername/ace-playbook.git
cd ace-playbook

# Install uv (fastest package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project with dev dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Step 2: Configure Environment

```bash
# Create .env file
cat > .env <<EOF
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
DATABASE_URL=sqlite:///./playbook.db
LOG_LEVEL=INFO
EOF

# Or export directly
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Step 3: Verify Installation

```bash
# Run tests
pytest tests/unit -v

# Check code quality
make lint

# View available commands
make help
```

### IDE Setup

#### VS Code (Recommended)

Install extensions:
- Python (Microsoft)
- Pylance
- Ruff
- Test Explorer UI

`.vscode/settings.json`:
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

1. Open project in PyCharm
2. Settings → Project → Python Interpreter → Add Interpreter
3. Select virtual environment from project
4. Enable: Settings → Tools → Python Integrated Tools → pytest
5. Enable: Settings → Editor → Code Style → Python → Black

## Project Structure

### Top-Level Layout

```
ace-playbook/
├── ace/                    # Main package
│   ├── generator/          # Task execution
│   ├── reflector/          # Feedback analysis
│   ├── curator/            # Playbook management
│   ├── models/             # Pydantic models
│   ├── repositories/       # Database access
│   ├── utils/              # Utilities
│   ├── ops/                # Observability
│   └── runner/             # Workflows
├── tests/                  # Test suites
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── e2e/                # End-to-end tests
│   └── benchmarks/         # Performance tests
├── docs/                   # Documentation
├── examples/               # Example scripts
├── scripts/                # Utility scripts
└── alembic/                # Database migrations
```

### Key Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| `ace.generator` | Task execution with CoT reasoning | `cot_generator.py`, `signatures.py` |
| `ace.reflector` | Insight extraction from feedback | `grounded_reflector.py` |
| `ace.curator` | Semantic deduplication | `semantic_curator.py`, `curator_service.py` |
| `ace.models` | Pydantic models | `playbook.py` |
| `ace.repositories` | Database access | `playbook_repository.py`, `journal_repository.py` |
| `ace.utils` | Core utilities | `embeddings.py`, `faiss_index.py` |
| `ace.ops` | Observability | `metrics.py`, `guardrails.py`, `tracing.py` |

### Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project config, dependencies |
| `Makefile` | Development commands |
| `.pre-commit-config.yaml` | Code quality hooks |
| `alembic.ini` | Database migration config |
| `pytest.ini` | Test configuration |

## Development Workflows

### Daily Development Cycle

```bash
# 1. Pull latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feature/add-new-reflector

# 3. Make changes and run tests frequently
pytest tests/unit/test_reflector.py -v

# 4. Format and lint before commit
make format
make lint

# 5. Run full test suite
make test

# 6. Commit with descriptive message
git add .
git commit -m "feat: Add custom ReflectorModule for code analysis"

# 7. Push and create PR
git push origin feature/add-new-reflector
```

### Running Tests

```bash
# Unit tests (fast, no external dependencies)
pytest tests/unit -v

# Integration tests (database, FAISS)
pytest tests/integration -v

# End-to-end tests (full pipeline)
pytest tests/e2e -v

# Specific test file
pytest tests/unit/test_curator.py -v

# Specific test function
pytest tests/unit/test_curator.py::test_deduplication -v

# With coverage
pytest --cov=ace --cov-report=html
open htmlcov/index.html

# Performance benchmarks
pytest tests/benchmarks/ --benchmark-only
```

### Code Quality Checks

```bash
# Format code (auto-fix)
black ace/ tests/

# Lint (ruff - fast)
ruff check ace/ tests/

# Type checking
mypy ace/

# Security scans
bandit -r ace/
safety check
pip-audit

# Complexity analysis
radon cc ace/ -a -nb

# Full CI pipeline locally
make ci
```

## Testing Guidelines

### Test Organization

```python
# tests/unit/test_curator.py
"""
Unit tests for SemanticCurator.

Tests focus on isolated logic without external dependencies.
Mock FAISS, database, and LLM calls.
"""

import pytest
from unittest.mock import Mock, patch

class TestSemanticCurator:
    """Test suite for SemanticCurator class."""

    @pytest.fixture
    def curator(self):
        """Create curator with mocked dependencies."""
        embedding_service = Mock()
        faiss_manager = Mock()
        return SemanticCurator(
            embedding_service=embedding_service,
            faiss_manager=faiss_manager,
            similarity_threshold=0.8
        )

    def test_deduplication_increments_counter(self, curator):
        """
        Test that similar insights increment counter instead of adding new bullet.

        Given: Existing bullet "Break into steps"
        When: Add similar insight "Decompose into parts" (cosine=0.85)
        Then: helpful_count increments to 2, no new bullet added
        """
        # Test implementation
        ...
```

### Test Types

**Unit Tests** (tests/unit/):
- Test single functions/methods in isolation
- Mock all external dependencies
- Fast (< 1s per test)
- Coverage target: 90%

**Integration Tests** (tests/integration/):
- Test component interactions (e.g., Curator + Repository + FAISS)
- Real SQLite database (test DB, cleaned after)
- Real embeddings and FAISS
- Medium speed (1-5s per test)

**End-to-End Tests** (tests/e2e/):
- Test full pipeline (Generator → Reflector → Curator)
- Real components, test database
- Slow (5-30s per test)
- Smoke tests for deployment verification

### Writing Good Tests

```python
# ✅ GOOD: Descriptive name, clear Given/When/Then
def test_curator_increments_helpful_when_similar_insight_matches():
    """
    Test counter increment for similar insights.

    Given: Playbook with 1 bullet (helpful=1)
    When: Add similar insight (cosine≥0.8)
    Then: helpful_count=2, bullet_count stays at 1
    """
    # Given
    existing_bullet = PlaybookBullet(content="Break into steps", helpful_count=1)
    playbook = [existing_bullet]

    # When
    output = curator.apply_delta(CuratorInput(
        insights=[InsightCandidate(content="Decompose into parts")],
        current_playbook=playbook
    ))

    # Then
    assert len(output.updated_playbook) == 1
    assert output.updated_playbook[0].helpful_count == 2


# ❌ BAD: Unclear name, no docstring, unclear assertions
def test_curator_1():
    curator = SemanticCurator()
    output = curator.apply_delta(input)
    assert output.stats["increments"] > 0
```

## Common Workflows

### 1. Adding a New Playbook Bullet Type

**Scenario**: Add support for "Warning" bullets (in addition to Helpful/Harmful/Neutral)

**Steps**:

```python
# 1. Update PlaybookSection enum
# ace/models/playbook.py
from enum import Enum

class PlaybookSection(str, Enum):
    HELPFUL = "Helpful"
    HARMFUL = "Harmful"
    NEUTRAL = "Neutral"
    WARNING = "Warning"  # NEW

# 2. Update PlaybookBullet model
# ace/models/playbook.py
class PlaybookBullet(BaseModel):
    section: PlaybookSection  # Auto-validates against enum

# 3. Update Reflector to generate Warning insights
# ace/reflector/grounded_reflector.py
def analyze_feedback(self, feedback: ExecutionFeedback) -> List[InsightCandidate]:
    insights = []

    # Check for warnings
    if feedback.performance_metrics.get("latency") > 2000:
        insights.append(InsightCandidate(
            content="This approach may be slow for large inputs",
            section="Warning",  # NEW
            confidence=0.8
        ))

    return insights

# 4. Add tests
# tests/unit/test_reflector.py
def test_reflector_generates_warning_for_slow_execution():
    feedback = ExecutionFeedback(
        performance_metrics={"latency": 3000}  # 3s - slow
    )

    reflection = reflector.forward(task_output, feedback)

    warning_insights = [i for i in reflection.insights if i.section == "Warning"]
    assert len(warning_insights) > 0

# 5. Run migration (if DB schema changed)
alembic revision -m "Add warning section support"
# Edit migration file to add enum value
alembic upgrade head
```

### 2. Implementing a Custom Reflector

**Scenario**: Create a CodeReviewReflector for analyzing code changes

**Steps**:

```python
# 1. Create new reflector module
# ace/reflector/code_review_reflector.py
from ace.reflector.signatures import ReflectorInput, ReflectorOutput
import dspy

class CodeReviewSignature(dspy.Signature):
    """Analyze code review feedback to extract insights."""
    code_diff: str = dspy.InputField()
    review_comments: list[str] = dspy.InputField()
    insights: list[dict] = dspy.OutputField()

class CodeReviewReflector(dspy.Module):
    """Reflector specialized for code review analysis."""

    def __init__(self, model: str = "gpt-4"):
        super().__init__()
        self.predictor = dspy.ChainOfThought(CodeReviewSignature)

    def forward(
        self,
        code_diff: str,
        review_comments: list[str]
    ) -> ReflectorOutput:
        """Extract insights from code review feedback."""
        # Call LLM with custom prompt
        result = self.predictor(
            code_diff=code_diff,
            review_comments=review_comments
        )

        # Parse insights
        insights = [
            InsightCandidate(
                content=i["content"],
                section=i["section"],
                confidence=i["confidence"],
                rationale=i["rationale"]
            )
            for i in result.insights
        ]

        return ReflectorOutput(
            task_id="code-review",
            insights=insights
        )

# 2. Add tests
# tests/unit/test_code_review_reflector.py
def test_code_review_reflector_extracts_insights():
    reflector = CodeReviewReflector()

    diff = "+def calculate(x): return x * 2"
    comments = ["Consider edge case: x=0", "Add type hints"]

    output = reflector.forward(diff, comments)

    assert len(output.insights) >= 2
    assert any("type hints" in i.content.lower() for i in output.insights)

# 3. Register in __init__.py
# ace/reflector/__init__.py
from ace.reflector.grounded_reflector import GroundedReflector
from ace.reflector.code_review_reflector import CodeReviewReflector

__all__ = ["GroundedReflector", "CodeReviewReflector"]
```

### 3. Adding a New Metric

**Scenario**: Track deduplication effectiveness

**Steps**:

```python
# 1. Add metric to MetricsCollector
# ace/ops/metrics.py
class MetricsCollector:
    def __init__(self):
        self.counters = {}
        # Add new metric
        self.dedup_rate_histogram = []

    def record_dedup_rate(self, rate: float, domain_id: str):
        """Record deduplication rate (0.0-1.0)."""
        self.dedup_rate_histogram.append(rate)
        # Export to Prometheus
        self.prometheus_metrics[f'dedup_rate{{domain="{domain_id}"}}'] = rate

# 2. Instrument Curator
# ace/curator/semantic_curator.py
def apply_delta(self, curator_input: CuratorInput) -> CuratorOutput:
    # ... existing logic ...

    # Calculate dedup rate
    dedup_rate = increments / (increments + new_bullets)

    # Record metric
    metrics.record_dedup_rate(dedup_rate, curator_input.domain_id)

    return output

# 3. Add test
# tests/unit/test_metrics.py
def test_metrics_records_dedup_rate():
    metrics = MetricsCollector()

    metrics.record_dedup_rate(0.75, "arithmetic")

    assert "dedup_rate" in metrics.prometheus_metrics
    assert metrics.prometheus_metrics['dedup_rate{domain="arithmetic"}'] == 0.75
```

### 4. Debugging FAISS Issues

**Common Issues**:

1. **Dimension Mismatch**

```python
# Problem: FAISS expects 384-dim, got 768-dim
# Solution: Check embedding model dimension

embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
# This model outputs 384-dim vectors

# For larger models, update FAISSIndexManager
faiss_manager = FAISSIndexManager(dimension=768)
```

2. **Empty Index Search**

```python
# Problem: Searching before adding vectors
# Solution: Check if index is populated

if faiss_manager.get_index_size(domain_id) == 0:
    # Build index first
    faiss_manager.add_vectors(domain_id, vectors, ids)
```

3. **Cross-Domain Contamination**

```python
# Problem: Mixed domain_id in FAISS index
# Solution: Validate domain isolation

def add_vectors(self, domain_id: str, vectors, ids):
    # Validate all vectors belong to same domain
    assert all(id.startswith(f"{domain_id}-") for id in ids)

    # Use separate index per domain
    if domain_id not in self.indices:
        self.indices[domain_id] = faiss.IndexFlatIP(self.dimension)

    self.indices[domain_id].add(vectors)
```

## Troubleshooting

### Common Errors

#### 1. ImportError: No module named 'ace'

**Cause**: Project not installed in editable mode

**Solution**:
```bash
pip install -e ".[dev]"
```

#### 2. OpenAI API errors

**Cause**: Missing or invalid API key

**Solution**:
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Or check .env file
cat .env | grep OPENAI_API_KEY

# Set manually
export OPENAI_API_KEY=your_key_here
```

#### 3. FAISS dimension mismatch

**Cause**: Embedding dimension doesn't match FAISS index

**Solution**:
```python
# Check embedding dimension
embedding = embedding_service.embed("test")
print(f"Dimension: {len(embedding)}")  # Should be 384

# Ensure FAISS manager matches
faiss_manager = FAISSIndexManager(dimension=384)
```

#### 4. SQLite database locked

**Cause**: Multiple processes accessing database

**Solution**:
```bash
# Enable WAL mode (done automatically in migrations)
sqlite3 playbook.db "PRAGMA journal_mode=WAL;"

# Or use separate test database
pytest --db-url=sqlite:///test.db
```

#### 5. Pre-commit hooks failing

**Cause**: Code doesn't meet quality standards

**Solution**:
```bash
# Run formatters manually
make format

# Fix lint issues
ruff check ace/ tests/ --fix

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

### Debugging Tips

**1. Use pytest -v -s for detailed output**:
```bash
pytest tests/unit/test_curator.py -v -s
# -v: verbose
# -s: show print statements
```

**2. Use pdb for interactive debugging**:
```python
def test_curator():
    curator = SemanticCurator()
    import pdb; pdb.set_trace()  # Breakpoint
    output = curator.apply_delta(input)
```

**3. Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**4. Use pytest fixtures for common setup**:
```python
@pytest.fixture
def curator_with_data():
    """Pre-populated curator for testing."""
    curator = SemanticCurator()
    # Add test bullets
    bullets = create_test_bullets(10)
    curator.repository.batch_create(bullets)
    return curator
```

## Best Practices

### Code Style

1. **Type Hints**: All public functions must have type hints
```python
def apply_delta(self, curator_input: CuratorInput) -> CuratorOutput:
    ...
```

2. **Docstrings**: Google-style docstrings for all public APIs
```python
def apply_delta(self, curator_input: CuratorInput) -> CuratorOutput:
    """Merge insights into playbook with semantic deduplication.

    Args:
        curator_input: Contains insights and current playbook state.

    Returns:
        Updated playbook with deduplication stats.

    Raises:
        ValueError: If domain_id invalid or insights list empty.
    """
```

3. **Error Handling**: Specific exceptions with context
```python
if not curator_input.insights:
    raise ValueError(
        f"Insights list is empty for domain {curator_input.domain_id}"
    )
```

4. **Line Length**: Max 100 characters (enforced by black)

5. **Imports**: Organized by isort (auto-sorted by pre-commit)

### Performance

1. **Profile Before Optimizing**: Use pytest-benchmark
```python
def test_curator_performance(benchmark):
    result = benchmark(curator.apply_delta, input)
    assert result.stats["dedup_rate"] > 0.5
```

2. **Batch Operations**: Use batch methods for multiple items
```python
# ✅ GOOD: Batch insert
repo.batch_create(bullets)

# ❌ BAD: Loop insert
for bullet in bullets:
    repo.create(bullet)
```

3. **Cache Expensive Operations**: Lazy-load models
```python
class EmbeddingService:
    def __init__(self):
        self._model = None  # Lazy-load

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model
```

### Security

1. **Input Validation**: Validate all user inputs
```python
def validate_domain_id(domain_id: str) -> None:
    if not re.match(r"^[a-z0-9-]{3,50}$", domain_id):
        raise ValueError(f"Invalid domain_id format: {domain_id}")
```

2. **SQL Injection Prevention**: Use SQLAlchemy ORM (automatic)
```python
# ✅ GOOD: Parameterized query (SQLAlchemy)
query = select(PlaybookBullet).where(PlaybookBullet.domain_id == domain_id)

# ❌ BAD: String formatting (vulnerable)
query = f"SELECT * FROM bullets WHERE domain_id = '{domain_id}'"
```

3. **API Key Protection**: Never commit .env files
```bash
# .gitignore already includes
.env
.env.local
```

### Testing

1. **Test Naming**: Descriptive names following pattern
```python
def test_<component>_<action>_<expected_result>():
    """Test that <component> <action> when <condition>."""
```

2. **Arrange-Act-Assert**: Clear test structure
```python
def test_curator_deduplication():
    # Arrange
    curator = SemanticCurator()
    input = create_test_input()

    # Act
    output = curator.apply_delta(input)

    # Assert
    assert output.stats["new_bullets"] == 0
    assert output.stats["increments"] == 1
```

3. **Fixtures Over Setup**: Use pytest fixtures
```python
@pytest.fixture
def test_db():
    """Create test database, clean up after test."""
    db = setup_test_database()
    yield db
    cleanup_test_database(db)
```

## Next Steps

- [Architecture Overview](architecture.md) - Deep dive into system design
- [API Reference](api/index.rst) - Complete API documentation
- [Tutorials](tutorials/index.rst) - Step-by-step guides
- [Edge Cases](edge_cases.md) - Error handling patterns
- [Runbook](runbook.md) - Operations and troubleshooting

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions, share ideas
- **Documentation**: Check docs first for common questions
- **Code Examples**: See `examples/` directory

Welcome to the team! 🎉
