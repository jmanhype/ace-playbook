# Contributing to ACE Playbook

Thank you for your interest in contributing to ACE Playbook! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Code Standards](#code-standards)
- [Commit Message Convention](#commit-message-convention)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/ace-playbook.git
   cd ace-playbook
   ```

3. Add the upstream repository:

   ```bash
   git remote add upstream https://github.com/jmanhype/ace-playbook.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (recommended) or `pip`
- Git

### Installation

```bash
# Install dependencies with development tools
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize database
alembic upgrade head

# Install pre-commit hooks (REQUIRED)
pre-commit install
pre-commit install --hook-type commit-msg
```

### Verify Setup

```bash
# Run tests to verify installation
pytest tests/unit/ -v

# Run pre-commit on all files
pre-commit run --all-files
```

## Pre-commit Hooks

Pre-commit hooks ensure code quality and consistency. They run automatically before each commit.

### Hook Categories

#### Code Quality

- **Black**: Automatic code formatting (line length: 100)
- **Ruff**: Fast Python linting (replaces flake8)
- **isort**: Import statement sorting

#### Type Safety

- **mypy**: Static type checking with strict mode

#### Security

- **Bandit**: Security vulnerability scanning
- **detect-secrets**: Prevents committing secrets (API keys, tokens)

#### Documentation

- **interrogate**: Docstring coverage checking (≥80% required)
- **markdownlint**: Markdown file linting

#### Standards

- **Conventional Commits**: Enforces commit message format
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline
- **check-yaml/toml/json**: Configuration file validation

#### Infrastructure

- **hadolint**: Dockerfile linting
- **sqlfluff**: SQL migration linting
- **yamllint**: YAML file linting

### Running Hooks

```bash
# Automatic: Hooks run on every commit
git commit -m "feat: add new feature"

# Manual: Run on all files
pre-commit run --all-files

# Manual: Run specific hook
pre-commit run black --all-files
pre-commit run mypy --all-files

# Skip hooks (use sparingly, only for WIP commits)
git commit --no-verify -m "WIP: work in progress"
```

### Handling Hook Failures

If a hook fails:

1. **Auto-fixable issues** (Black, Ruff, isort): Files are auto-fixed. Review changes and re-commit.
2. **Manual fixes required** (mypy, bandit): Fix the reported issues and re-commit.
3. **Commit message issues**: Rewrite commit message following conventions.

Example workflow:

```bash
git add myfile.py
git commit -m "feat: add new feature"
# Black reformats the file
# Review changes
git add myfile.py
git commit -m "feat: add new feature"
# Success!
```

## Code Standards

### Python Style

- **Line length**: 100 characters (enforced by Black)
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Required for all public modules, classes, and functions (Google style)
- **Import order**: Standard library → Third-party → Local (enforced by isort)

### Type Hints Example

```python
from typing import List, Optional
from ace.models.playbook import PlaybookBullet

def get_bullets(
    domain_id: str,
    section: Optional[str] = None,
    limit: int = 10
) -> List[PlaybookBullet]:
    """Retrieve playbook bullets for a domain.

    Args:
        domain_id: Domain identifier (namespace)
        section: Optional section filter (Helpful/Harmful/Neutral)
        limit: Maximum number of bullets to return

    Returns:
        List of PlaybookBullet objects matching criteria

    Raises:
        ValueError: If domain_id is invalid
    """
    # Implementation
```

### Docstring Coverage

All public APIs require docstrings. Minimum coverage: 80%

```bash
# Check docstring coverage
interrogate -vv ace/

# Generate coverage report
interrogate -vv --generate-badge docs/ ace/
```

### Security Guidelines

- **Never commit secrets**: Use environment variables
- **Input validation**: Validate all user inputs
- **SQL injection**: Use parameterized queries (SQLAlchemy ORM)
- **Path traversal**: Validate file paths

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```text
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring (no feature or bug fix)
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, tooling)
- **ci**: CI/CD changes

### Examples

```bash
# Simple feature
git commit -m "feat: add semantic deduplication threshold config"

# Bug fix with scope
git commit -m "fix(curator): handle empty embedding vectors"

# Breaking change
git commit -m "feat!: replace FAISS with custom similarity search

BREAKING CHANGE: FaissIndexManager API changed, embeddings now normalized"

# Multiple paragraphs
git commit -m "refactor(reflector): extract insight classification logic

Split InsightClassifier into separate module for better testability.
Updated tests to use new interface.

Closes #42"
```

### Validation

The `conventional-pre-commit` hook validates commit messages:

```bash
# ✅ Valid
git commit -m "feat: add metrics collector"

# ❌ Invalid - missing type
git commit -m "add metrics collector"

# ❌ Invalid - uppercase subject
git commit -m "feat: Add metrics collector"

# ❌ Invalid - period at end
git commit -m "feat: add metrics collector."
```

## Testing

### Test Structure

```text
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Multi-component integration tests
└── e2e/           # End-to-end smoke tests
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# With coverage
pytest tests/ --cov=ace --cov-report=html

# Specific test file
pytest tests/unit/test_semantic_curator.py -v

# Specific test
pytest tests/unit/test_semantic_curator.py::TestSemanticCurator::test_deduplication -v
```

### Writing Tests

```python
import pytest
from ace.curator import SemanticCurator

class TestSemanticCurator:
    """Tests for SemanticCurator."""

    @pytest.fixture
    def curator(self) -> SemanticCurator:
        """Create curator instance."""
        return SemanticCurator()

    def test_deduplication(self, curator: SemanticCurator) -> None:
        """Test semantic deduplication at 0.8 threshold."""
        # Arrange
        insights = [...]

        # Act
        result = curator.batch_merge(...)

        # Assert
        assert len(result["updated_playbook"]) == 1
```

### Test Requirements

- **Coverage**: Maintain ≥80% code coverage
- **Isolation**: Tests should not depend on external services (use mocks)
- **Speed**: Unit tests should run in <1s each
- **Clarity**: Use descriptive test names and clear arrange-act-assert structure

### Mutation Testing

Mutation testing verifies test suite quality by introducing small code changes (mutations) and checking if tests catch them. A high mutation kill rate (≥90%) indicates a robust test suite.

#### What is Mutation Testing?

Mutation testing works by:
1. Creating "mutants" - small modifications to source code (e.g., changing `>` to `>=`, `+` to `-`)
2. Running your test suite against each mutant
3. Checking if tests fail (killing the mutant) or pass (mutant survives)

A surviving mutant indicates:
- Missing test coverage
- Weak assertions
- Logic that doesn't affect behavior

#### Running Mutation Tests

```bash
# Using Make (recommended)
make mutation-test          # Test curator module
make mutation-results       # Show results summary
make mutation-show ID=1     # Show specific mutation

# Using script directly
./scripts/run_mutation_tests.sh

# Manual mutmut usage
mutmut run                  # Run on all code
mutmut results              # Show summary
mutmut show 5               # View mutation #5
mutmut html                 # Generate HTML report
```

#### Interpreting Results

Mutation Score Guide:
- **100% killed**: Excellent - all mutations detected
- **90-99% killed**: Good - minor gaps acceptable
- **80-89% killed**: Acceptable - needs improvement
- **<80% killed**: Poor - significant test gaps

Example output:
```
Survived: 5 mutants
Killed: 45 mutants
Mutation score: 90%
```

#### Fixing Surviving Mutants

If a mutant survives:

1. **View the mutation**:
   ```bash
   mutmut show 5
   ```

2. **Analyze why it survived**:
   - Missing test case?
   - Weak assertion (e.g., checking type but not value)?
   - Dead code that should be removed?

3. **Add/improve tests**:
   ```python
   # Before: Weak assertion
   def test_calculation():
       result = calculate(5, 3)
       assert isinstance(result, int)  # Mutant survives

   # After: Strong assertion
   def test_calculation():
       result = calculate(5, 3)
       assert result == 8  # Mutant killed
   ```

4. **Re-run mutation test** to verify the fix

#### Common Mutation Types

mutmut introduces various mutations:

- **Arithmetic**: `+` → `-`, `*` → `/`, `//` → `%`
- **Comparison**: `>` → `>=`, `==` → `!=`, `<` → `<=`
- **Boolean**: `and` → `or`, `True` → `False`
- **Numbers**: `0` → `1`, `1` → `0`, `n` → `n+1`
- **Strings**: `"text"` → `"XXtextXX"`

#### Best Practices

1. **Run mutation tests periodically** (not on every commit - they're slow)
2. **Focus on critical modules** first (curator, reflector, generator)
3. **Aim for 90%+ mutation score** on core business logic
4. **Use mutation testing to find test gaps**, not just coverage holes
5. **Document surviving mutants** if they're intentional (e.g., logging code)

#### CI Integration

Mutation testing is resource-intensive. Consider:

```bash
# Run on specific modules only
mutmut run ace/curator/semantic_curator.py

# Run in CI on PRs to core modules
if [ "$CHANGED_MODULE" = "ace/curator" ]; then
  make mutation-test
fi
```

#### Configuration

Mutation testing is configured in `pyproject.toml`:

```toml
[tool.mutmut]
# Configuration handled by .mutmut-config
# See scripts/run_mutation_tests.sh for usage
```

#### Further Reading

- [Mutation Testing Concepts](https://en.wikipedia.org/wiki/Mutation_testing)
- [mutmut Documentation](https://mutmut.readthedocs.io/)
- [Test Quality vs Coverage](https://martinfowler.com/bliki/TestCoverage.html)

### Property-Based Testing

Property-based testing automatically generates test inputs to verify invariants that should hold for ALL inputs, not just hand-picked examples. Uses the `hypothesis` library to find edge cases that break your code.

#### What is Property-Based Testing?

Instead of writing specific test cases:
```python
# Traditional example-based test
def test_addition():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0
```

Write properties that should always be true:
```python
# Property-based test
@given(x=st.integers(), y=st.integers())
def test_addition_commutative(x, y):
    assert add(x, y) == add(y, x)  # Tests MILLIONS of inputs
```

#### Why Use Property-Based Testing?

1. **Finds edge cases** humans miss (negative numbers, zero, MAX_INT)
2. **Acts as executable specification** (documents what code MUST do)
3. **Shrinks failures** to minimal reproducing example
4. **Tests more code paths** with less test code

#### Running Property Tests

```bash
# Run property tests
pytest tests/unit/test_curator_properties.py -v

# Run with more examples (slower but more thorough)
pytest tests/unit/test_curator_properties.py -v --hypothesis-max-examples=100

# Show generated examples
pytest tests/unit/test_curator_properties.py -v --hypothesis-show-most-frequent=10
```

#### Writing Property Tests

Located in `tests/unit/test_curator_properties.py` with examples:

**Property: Idempotence**
```python
@given(domain_id=domain_ids, section=insight_sections)
def test_deduplication_idempotent(curator, domain_id, section):
    """
    Applying the same insight twice should only add one bullet.
    curator(curator(playbook, insight)) = curator(playbook, insight)
    """
    playbook = []
    insight = {"content": "Test", "section": section}

    output1 = curator.apply_delta(playbook, [insight])
    output2 = curator.apply_delta(output1.playbook, [insight])

    assert len(output2.playbook) == 1  # Not 2!
```

**Property: Monotonicity**
```python
@given(domain_id=domain_ids)
def test_counters_monotonic(curator, domain_id):
    """
    Counters never decrease during operations.
    For all operations: counter_after ≥ counter_before
    """
    bullet = create_bullet(helpful_count=5, harmful_count=3)

    output = curator.apply_delta([bullet], new_insights)

    for updated in output.playbook:
        assert updated.helpful_count >= 5
        assert updated.harmful_count >= 3
```

**Property: Symmetry**
```python
@given(vec1=embeddings_384, vec2=embeddings_384)
def test_similarity_symmetric(vec1, vec2):
    """
    Cosine similarity is symmetric.
    sim(A, B) = sim(B, A)
    """
    assert compute_similarity(vec1, vec2) == compute_similarity(vec2, vec1)
```

#### Hypothesis Strategies

Hypothesis provides generators for test data:

```python
from hypothesis import strategies as st

# Built-in strategies
integers = st.integers(min_value=0, max_value=100)
floats = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
text = st.text(min_size=1, max_size=100, alphabet=st.characters())
lists = st.lists(st.integers(), min_size=0, max_size=10)

# Custom strategies (domain IDs)
domain_ids = st.from_regex(r"^[a-z0-9-]{3,20}$").filter(
    lambda x: x not in {"system", "admin", "test"}
)

# Composite strategies (complex objects)
@st.composite
def playbook_bullet(draw):
    domain_id = draw(domain_ids)
    content = draw(st.text(min_size=10, max_size=200))
    embedding = draw(st.lists(st.floats(), min_size=384, max_size=384))

    return PlaybookBullet(
        domain_id=domain_id,
        content=content,
        embedding=embedding,
        ...
    )
```

#### Common Properties to Test

**Invariants** (always true):
- Idempotence: `f(f(x)) = f(x)`
- Commutativity: `f(x, y) = f(y, x)`
- Associativity: `f(f(x, y), z) = f(x, f(y, z))`
- Identity: `f(x, identity) = x`

**Relations** (between operations):
- Inverse: `g(f(x)) = x`
- Monotonicity: `x < y → f(x) < f(y)`
- Symmetry: `f(x, y) = f(y, x)`

**Bounds**:
- Range limits: `0 ≤ f(x) ≤ 1`
- Size constraints: `len(result) ≤ len(input)`

#### Debugging Property Test Failures

When hypothesis finds a failure:

1. **Read the minimal example**:
   ```
   Falsifying example: test_counters_monotonic(
       curator=<SemanticCurator>,
       domain_id='a-b-1',
       section1='Helpful',
       section2='Harmful'
   )
   ```

2. **Reproduce with exact values**:
   ```python
   def test_regression_counters_monotonic():
       curator = SemanticCurator()
       domain_id = 'a-b-1'
       # ... use exact failing values
   ```

3. **Fix the bug** or **adjust the property** if it's too strict

4. **Add as regression test** to prevent future failures

#### Best Practices

1. **Start with simple properties** (symmetry, bounds, commutativity)
2. **Use `assume()` to filter invalid inputs**:
   ```python
   @given(x=st.integers(), y=st.integers())
   def test_division(x, y):
       assume(y != 0)  # Skip zero divisor
       assert x / y == x / y
   ```

3. **Combine with example-based tests** for clarity
4. **Run property tests in CI** (they're slower than unit tests)
5. **Document discovered properties** as they emerge
6. **Keep max_examples low** in development (20-50), high in CI (100-1000)

#### Configuration

Property testing is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "property: Property-based tests with hypothesis",
]
```

Run with custom settings:
```bash
# More examples (slower, more thorough)
pytest --hypothesis-max-examples=1000

# Print statistics
pytest --hypothesis-show-statistics

# Deterministic (for CI)
pytest --hypothesis-seed=12345
```

#### Further Reading

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Property-Based Testing Guide](https://hypothesis.works/)
- [Property Testing Patterns](https://fsharpforfunandprofit.com/posts/property-based-testing/)

### Complexity Monitoring

Code complexity monitoring uses `radon` to measure cyclomatic complexity and maintainability index. High complexity indicates code that's hard to understand, test, and maintain.

#### What is Cyclomatic Complexity?

Cyclomatic complexity measures the number of independent paths through code:
- **1-5** (A): Simple, easy to test
- **6-10** (B): Moderate, acceptable
- **11-20** (C): Complex, consider refactoring
- **21-50** (D): Very complex, should refactor
- **51+** (F): Unmaintainable, must refactor

#### Maintainability Index

Maintainability index combines complexity, lines of code, and Halstead volume:
- **100-20** (A): Very maintainable
- **19-10** (B): Maintainable
- **9-0** (C): Needs attention

#### Running Complexity Checks

```bash
# Check complexity (informational)
make complexity

# Check with strict thresholds (fail on C-grade or worse)
make complexity-strict

# Generate JSON reports
make complexity-report

# Manual radon commands
radon cc ace/ -a                    # Cyclomatic complexity with average
radon cc ace/ -n C -s               # Fail on C-grade or worse
radon mi ace/ -nb                   # Maintainability index (no Berkley)
radon mi ace/ -s                    # Show maintainability index
```

#### Pre-commit Integration

Complexity checks run automatically on `git push`:

```yaml
# .pre-commit-config.yaml
- id: radon-cc
  name: Check code complexity
  entry: radon cc ace/ -n C -s      # Fail on C-grade or worse
  stages: [push]

- id: radon-mi
  name: Check maintainability index
  entry: radon mi ace/ -n B -s      # Fail on B-grade or worse
  stages: [push]
```

To skip complexity checks (not recommended):
```bash
git push --no-verify
```

#### Interpreting Results

Example output:
```
ace/curator/semantic_curator.py
    M 435:0 SemanticCurator - B (19.2)
    M 81:4 SemanticCurator.apply_delta - B (6)
    M 237:4 SemanticCurator.batch_merge - C (12)

Average complexity: B (8.4)
```

Reading the output:
- **M**: Method
- **435:0**: Line number
- **B (19.2)**: Grade (B) and maintainability index (19.2)
- **B (6)**: Grade (B) and cyclomatic complexity (6)

#### Refactoring High Complexity Code

When you encounter C-grade or worse complexity:

**1. Extract Methods**:
```python
# Before: Complex method (C-grade)
def process_data(data):
    if condition1:
        if condition2:
            if condition3:
                # ... nested logic
                pass

# After: Extracted methods (A/B-grade)
def process_data(data):
    if should_process(data):
        return perform_processing(data)
    return default_result()

def should_process(data):
    return condition1 and condition2 and condition3

def perform_processing(data):
    # ... logic here
    pass
```

**2. Use Guard Clauses**:
```python
# Before: Nested conditions (high complexity)
def validate(data):
    if data:
        if data.is_valid():
            if data.has_required_fields():
                return True
    return False

# After: Guard clauses (lower complexity)
def validate(data):
    if not data:
        return False
    if not data.is_valid():
        return False
    if not data.has_required_fields():
        return False
    return True
```

**3. Replace Conditionals with Polymorphism**:
```python
# Before: Multiple isinstance checks (high complexity)
def process(obj):
    if isinstance(obj, TypeA):
        # ... complex logic
    elif isinstance(obj, TypeB):
        # ... complex logic
    elif isinstance(obj, TypeC):
        # ... complex logic

# After: Polymorphism (low complexity)
class BaseType:
    def process(self):
        raise NotImplementedError

class TypeA(BaseType):
    def process(self):
        # ... logic

def process(obj: BaseType):
    return obj.process()
```

**4. Use Data Structures**:
```python
# Before: Long if-elif chain (high complexity)
def get_status(code):
    if code == 200:
        return "OK"
    elif code == 404:
        return "Not Found"
    elif code == 500:
        return "Error"
    # ... many more

# After: Dictionary lookup (low complexity)
STATUS_MAP = {
    200: "OK",
    404: "Not Found",
    500: "Error",
    # ...
}

def get_status(code):
    return STATUS_MAP.get(code, "Unknown")
```

#### Complexity Targets

For ACE Playbook codebase:
- **Target**: All modules ≤ B-grade average
- **Maximum**: No functions > C-grade (complexity 20)
- **Pre-commit**: Blocks push if C-grade or worse detected

#### Best Practices

1. **Monitor complexity regularly** during development
2. **Refactor proactively** before complexity grows
3. **Break down large functions** into smaller, focused functions
4. **Use helper functions** to reduce nesting
5. **Document complex logic** if refactoring isn't possible
6. **Review complexity in PRs** before merging

#### CI Integration

Run complexity checks in CI pipeline:

```bash
# In CI script
make complexity-strict || exit 1
```

Add to PR checks:
- Fail if average complexity > B-grade
- Warn if any function > 10 complexity
- Block if any function > 20 complexity

#### Further Reading

- [Cyclomatic Complexity Explained](https://en.wikipedia.org/wiki/Cyclomatic_complexity)
- [Radon Documentation](https://radon.readthedocs.io/)
- [Code Complexity Thresholds](https://docs.sonarqube.org/latest/user-guide/metric-definitions/)
- [Refactoring Catalog](https://refactoring.com/catalog/)

## Pull Request Process

### Before Submitting

1. **Update your branch**:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:

   ```bash
   # Pre-commit hooks
   pre-commit run --all-files

   # Tests with coverage
   pytest tests/ --cov=ace --cov-report=term-missing

   # Type checking
   mypy ace/
   ```

3. **Update documentation**:
   - Update README.md if adding features
   - Add docstrings to new functions/classes
   - Update CHANGELOG.md following Keep a Changelog format

### Submitting PR

1. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - Clear title following conventional commits format
   - Description of changes
   - Link to related issues (e.g., "Closes #42")
   - Screenshots/examples if applicable

3. **PR Checklist**:
   - [ ] All pre-commit hooks pass
   - [ ] All tests pass
   - [ ] Code coverage ≥80%
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Conventional commit format
   - [ ] No merge conflicts

### PR Review Process

1. Maintainers will review your PR within 1-2 business days
2. Address review feedback with new commits
3. Once approved, maintainers will merge using squash-and-merge

### After Merge

1. **Delete your branch**:

   ```bash
   git checkout main
   git branch -D feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Update your fork**:

   ```bash
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

## Questions?

- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Security**: Email <security@example.com> for security vulnerabilities

Thank you for contributing to ACE Playbook! 🚀
