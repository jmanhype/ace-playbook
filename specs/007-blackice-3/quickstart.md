# Quickstart: BLACKICE 3.0

**Feature**: BLACKICE 3.0 Agentic Software Factory
**Date**: 2026-01-18

## Overview

BLACKICE 3.0 converts a single human vision into working software with tests, documentation, and a reproducible artifact trail. This guide covers installation, basic usage, and key concepts.

## Prerequisites

- Python 3.11+
- Docker (optional, for isolated execution)
- Model provider API key (Claude, OpenAI, or local Ollama)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/blackice.git
cd blackice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Verify installation
blackice doctor
```

## Configuration

Set up your model provider credentials:

```bash
# Option 1: Environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Option 2: Configuration file
cat > ~/.blackice/config.yaml << EOF
model_provider:
  primary: claude
  fallback: [openai, ollama]

execution:
  default: local  # or 'container' for isolation

memory:
  provider: letta
  fallback: local_jsonl

edition: lite  # or 'core' or 'enterprise'
EOF
```

## Basic Usage

### Vision to Working Software (Lite)

```bash
# Single command to convert vision to code
blackice build "Create a CLI tool that converts markdown files to HTML with syntax highlighting"

# Output: run workspace at ./runs/<run-id>/
# Contains:
#   - repo/           (generated code)
#   - tests/          (test suite)
#   - docs/           (documentation)
#   - decisions/      (artifact trail)
#   - logs/           (execution logs)
```

### Check Run Status

```bash
# List recent runs
blackice status

# Get details of a specific run
blackice status <run-id>

# Watch a running build
blackice watch <run-id>
```

### Resume After Crash (Core)

```bash
# Resume a crashed or paused run
blackice resume <run-id>

# The system:
# 1. Skips completed tasks
# 2. Retries in-flight tasks with new attempt IDs
# 3. Uses idempotency keys for external effects
```

### Spec-Validated Execution (Enterprise)

```bash
# Create a TaskSpec for validation
cat > my-taskspec.yaml << EOF
name: web-api
version: 1.0.0
strictness: strict
input_schema:
  type: object
  required: [framework]
  properties:
    framework:
      type: string
      enum: [fastapi, flask, django]
output_schema:
  type: object
  required: [has_tests, coverage_min]
  properties:
    has_tests: {type: boolean, const: true}
    coverage_min: {type: number, minimum: 90}
EOF

# Run with spec validation
blackice build --taskspec my-taskspec.yaml "Create a REST API for user management"

# Get cryptographic receipt
blackice receipt <run-id>
```

## Key Concepts

### The Ralph Loop

Every task goes through an iterative repair cycle:

```
TRY → FAIL → REFLECT → LEARN → RETRY
```

- **TRY**: Attempt the task
- **FAIL**: Detect failures via tests/verification
- **REFLECT**: Analyze what went wrong
- **LEARN**: Adjust approach based on failure
- **RETRY**: Attempt again (until success or budget exhausted)

### Multi-Agent Consensus

Complex tasks involve specialist agents:

| Agent | Role |
|-------|------|
| Planner | Breaks vision into tasks |
| Implementer | Writes code |
| Reviewer | Reviews for issues |
| Tester | Verifies correctness |
| Security | Audits for vulnerabilities |

Agents vote on decisions. Configurable policies: majority, supermajority, unanimous, quorum, weighted.

### Editions

| Feature | Lite | Core | Enterprise |
|---------|------|------|------------|
| Vision to software | Yes | Yes | Yes |
| Artifact trail | Yes | Yes | Yes |
| Safe execution | Yes | Yes | Yes |
| Crash resume | - | Yes | Yes |
| Event sourcing | - | Yes | Yes |
| TaskSpec validation | - | - | Yes |
| Cryptographic receipts | - | - | Yes |

## API Usage

Start the API server:

```bash
blackice serve --port 8000
```

Create a run via API:

```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{
    "vision": "Create a CLI calculator supporting +, -, *, /",
    "edition": "lite"
  }'
```

## Troubleshooting

### Health Check

```bash
blackice doctor
```

Checks:
- Model provider connectivity
- Execution environment availability
- Memory backend status
- Configuration validity

### Common Issues

**Model provider timeout**
- Check API key validity
- Verify network connectivity
- Try fallback provider: `blackice build --model ollama "..."`

**Container execution fails**
- Ensure Docker is running
- Try local execution: `blackice build --execution local "..."`

**Memory search returns empty**
- Letta may be unavailable; using JSONL fallback
- Semantic search not available in fallback mode

## Next Steps

- Read the [Architecture Guide](./plan.md) for system design details
- Review the [Data Model](./data-model.md) for entity schemas
- Check the [API Reference](./contracts/openapi.yaml) for endpoint details
- See [Research](./research.md) for technology decisions
