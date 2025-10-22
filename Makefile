# ACE Playbook Makefile
# Development and testing utilities

.PHONY: help test test-unit test-integration test-cov mutation-test clean install format lint security

help:  ## Show this help message
	@echo "ACE Playbook - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run all tests
	pytest

test-unit:  ## Run unit tests only
	pytest tests/unit -v

test-integration:  ## Run integration tests only
	pytest tests/integration -v

test-cov:  ## Run tests with coverage report
	pytest --cov=ace --cov-report=term-missing --cov-report=html

mutation-test:  ## Run mutation tests on curator module
	@echo "Running mutation tests on semantic_curator..."
	./scripts/run_mutation_tests.sh ace/curator/semantic_curator.py

mutation-test-all:  ## Run mutation tests on all modules
	@echo "Running mutation tests on all modules..."
	mutmut run

mutation-results:  ## Show mutation test results
	mutmut results

mutation-show:  ## Show specific mutation (usage: make mutation-show ID=1)
	mutmut show $(ID)

format:  ## Format code with black
	black ace/ tests/

lint:  ## Run linters (ruff, mypy)
	ruff check ace/ tests/
	mypy ace/

security:  ## Run security checks
	bandit -r ace/
	safety check
	pip-audit

smoke:  ## Run quick smoke test with dummy backend
	python examples/live_loop_quickstart.py --backend dummy --episodes 2

complexity:  ## Check code complexity with radon
	@echo "Checking cyclomatic complexity..."
	radon cc ace/ -a -nb
	@echo ""
	@echo "Checking maintainability index..."
	radon mi ace/ -nb
	@echo ""
	@echo "Complexity grades: A=simple, B=moderate, C=complex, D=very complex, F=unmaintainable"
	@echo "Maintainability index: A=very maintainable, B=maintainable, C=needs attention"

complexity-strict:  ## Check complexity with strict thresholds (fail on C or worse)
	@echo "Checking cyclomatic complexity (fail on C-grade)..."
	radon cc ace/ -n C -s
	@echo ""
	@echo "Checking maintainability index (fail on B-grade)..."
	radon mi ace/ -n B -s
	@echo ""
	@echo "✓ All complexity checks passed!"

complexity-report:  ## Generate detailed complexity reports
	@echo "Generating complexity reports..."
	@mkdir -p reports
	radon cc ace/ -a -j > reports/complexity.json
	radon mi ace/ -j > reports/maintainability.json
	@echo "Reports saved to reports/"

clean:  ## Clean temporary files
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mutmut-cache
	rm -rf html/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

ci:  ## Run full CI pipeline locally
	@echo "Running full CI pipeline..."
	make format
	make lint
	make security
	make test-cov
	@echo "✓ CI pipeline complete!"

docs:  ## Build Sphinx documentation
	@echo "Building documentation..."
	cd docs && sphinx-build -b html . _build/html
	@echo "✓ Documentation built!"
	@echo "View at: docs/_build/html/index.html"

docs-clean:  ## Clean documentation build
	rm -rf docs/_build docs/api/*.rst !docs/api/index.rst

docs-serve:  ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000

docs-autobuild:  ## Auto-rebuild docs on changes
	@echo "Starting auto-build server..."
	sphinx-autobuild docs docs/_build/html --port 8000
