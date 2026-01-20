# BLACKICE 3.0

AI-powered autonomous software development pipeline.

## Overview

BLACKICE is an agentic software factory that transforms natural language visions into working code through a multi-phase flywheel:

1. **Research** - Analyze requirements and gather context
2. **Plan** - Decompose vision into actionable tasks
3. **Implement** - Generate code using LLM providers
4. **Test** - Run tests and validate functionality
5. **Verify** - Quality checks and security review
6. **Deliver** - Package and deliver artifacts

## Installation

```bash
# Clone the repository
git clone https://github.com/hextropian/blackice.git
cd blackice

# Install dependencies
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.11

## Quick Start

### CLI Usage

```bash
# Check system health and provider status
blackice doctor

# Build a project from a vision
blackice build "Create a REST API for user management with CRUD operations"

# Resume a previous run
blackice resume <run-id>

# Watch run progress
blackice watch <run-id>

# Start the API server
blackice serve --port 8000
```

### API Server

Start the API server using the factory pattern:

```bash
# Development (localhost only, no auth) - default and secure
blackice serve --port 8000

# Development with hot reload
blackice serve --reload

# Production (with auth, external access)
export BLACKICE_API_KEY=$(python -c "from blackice.api.auth import generate_api_key; print(generate_api_key())")
blackice serve --host 0.0.0.0 --port 8000

# Direct uvicorn (alternative)
uvicorn blackice.api:create_app --factory --host 127.0.0.1 --port 8000
```

**Security:** The server defaults to localhost (127.0.0.1) and will refuse to bind to external addresses unless `BLACKICE_API_KEY` is set or `--insecure` is passed.

## API Reference

### Authentication

Set `BLACKICE_API_KEY` environment variable to enable authentication:

```bash
export BLACKICE_API_KEY="your-secure-api-key"
# Or generate a secure key:
export BLACKICE_API_KEY=$(python -c "from blackice.api.auth import generate_api_key; print(generate_api_key())")
```

All `/runs` endpoints require authentication when enabled:

```bash
# HTTP endpoints - use X-API-Key header
curl -H "X-API-Key: your-secure-api-key" http://localhost:8000/api/v1/runs

# WebSocket (browser) - use query param (browsers can't set headers)
ws://localhost:8000/api/v1/runs/run-abc123/stream?api_key=your-secure-api-key

# WebSocket (header-capable clients like websocat)
websocat -H "X-API-Key: your-api-key" ws://localhost:8000/api/v1/runs/run-abc123/stream
```

### Endpoints

#### Health Check
```bash
# Liveness probe (no auth required)
curl http://localhost:8000/api/v1/health/live

# Readiness probe with provider status
curl http://localhost:8000/api/v1/health/ready
```

#### Create a Run
```bash
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "vision": "Create a Python CLI tool that converts markdown to HTML",
    "edition": "core",
    "provider": "ollama"
  }'
```

Response:
```json
{
  "run_id": "run-abc123",
  "status": "created",
  "message": "Run created and started",
  "workspace": "/tmp/blackice-workspaces/run-abc123"
}
```

#### List Runs
```bash
curl http://localhost:8000/api/v1/runs \
  -H "X-API-Key: your-api-key"
```

#### Get Run Status
```bash
curl http://localhost:8000/api/v1/runs/run-abc123 \
  -H "X-API-Key: your-api-key"
```

#### Get Run Result
```bash
curl http://localhost:8000/api/v1/runs/run-abc123/result \
  -H "X-API-Key: your-api-key"
```

#### Cancel a Run
```bash
curl -X POST http://localhost:8000/api/v1/runs/run-abc123/cancel \
  -H "X-API-Key: your-api-key"
```

#### Delete a Run
```bash
# Delete completed run (purges workspace)
curl -X DELETE http://localhost:8000/api/v1/runs/run-abc123 \
  -H "X-API-Key: your-api-key"

# Force delete running run
curl -X DELETE "http://localhost:8000/api/v1/runs/run-abc123?force=true" \
  -H "X-API-Key: your-api-key"
```

#### WebSocket Streaming

When auth is enabled, browser WebSocket connections must use query param auth (since browsers can't set custom headers):

```javascript
// Browser (auth enabled) - use query param
const ws = new WebSocket('ws://localhost:8000/api/v1/runs/run-abc123/stream?api_key=your-api-key');

// Browser (no auth) or header-capable clients
const ws = new WebSocket('ws://localhost:8000/api/v1/runs/run-abc123/stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.event, data);
};
```

Header-capable clients (Python, websocat) can use `X-API-Key` header instead:
```bash
websocat -H "X-API-Key: your-api-key" ws://localhost:8000/api/v1/runs/run-abc123/stream
```

Events: `connected`, `phase_started`, `phase_completed`, `run_finished`, `run_deleted`, `keepalive`

### Providers

List available LLM providers:
```bash
curl http://localhost:8000/api/v1/providers
```

Supported providers:
- `ollama` - Local Ollama instance (default)
- `claude` - Anthropic Claude API
- `claude-max` - Claude with extended context
- `openai` - OpenAI API
- `zhipu` - Zhipu AI (free tier)
- `z.ai` - Z.AI (free tier)

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BLACKICE_API_KEY` | API key for authentication | None (auth disabled) |
| `BLACKICE_CORS_ORIGINS` | Comma-separated CORS origins | localhost only |
| `OLLAMA_BASE_URL` | Ollama API URL | http://localhost:11434 |
| `ANTHROPIC_API_KEY` | Claude API key | None |
| `ANTHROPIC_MAX_API_KEY` | Claude Max API key (extended context) | None |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ZHIPU_API_KEY` | Zhipu AI API key (optional, has free tier) | None |
| `ZAI_API_KEY` | Z.AI API key (optional, has free tier) | None |
| `LETTA_BASE_URL` | Letta MAS API URL | http://localhost:8283/v1 |
| `LETTA_API_TOKEN` | Letta API token | None |

### AI Factory (3090 GPU)

Configure connection to your AI Factory infrastructure:

```bash
export OLLAMA_BASE_URL="http://192.168.1.143:11434"
export LETTA_BASE_URL="http://192.168.1.143:8283/v1"
export LETTA_API_TOKEN="your-letta-token"
```

## Security

BLACKICE implements multiple security layers:

- **API Key Authentication** - Header-based auth with constant-time comparison
- **Fail-Closed Design** - Auth dependency returns 500 if misconfigured
- **CORS Protection** - Restricted origins, no credentials by default
- **Server-Controlled Workspaces** - Clients cannot specify filesystem paths
- **Symlink Attack Prevention** - Workspace paths validated before deletion
- **WebSocket Safety** - Single-sender queue pattern, disconnect detection

## Troubleshooting

### "401 Unauthorized" with auth enabled
- Ensure `BLACKICE_API_KEY` is set before starting the server
- For HTTP: Use `X-API-Key` header
- For WebSocket in browsers: Use `?api_key=...` query param

### "CORS blocked" errors
- Check `BLACKICE_CORS_ORIGINS` includes your frontend origin
- Default allows only localhost:3000 and localhost:8000

### "Ollama not reachable"
- Verify `OLLAMA_BASE_URL` is correct (default: http://localhost:11434)
- Run `blackice doctor` to check provider connectivity

### "Cannot bind to external address"
- Set `BLACKICE_API_KEY` to enable auth, or
- Use `--insecure` flag (not recommended for production)

### Workspace location
- Run workspaces are stored in `/tmp/blackice-workspaces/run-{id}/`
- Workspaces are automatically purged on `DELETE /api/v1/runs/{id}`

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=blackice

# Type checking
mypy blackice

# Linting
ruff check blackice
```

## License

MIT
