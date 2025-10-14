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
