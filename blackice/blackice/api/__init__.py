"""BLACKICE 3.0 API.

FastAPI-based HTTP API for programmatic access to BLACKICE.

P0 Security Fixes (Phase 8.1):
- API key authentication (set BLACKICE_API_KEY env var to enable)
- Tightened CORS (no wildcard with credentials)
- Secure defaults for production

P0 Security Fixes (Phase 8.2):
- Auth reads env var at runtime (not import time) to prevent fail-open
- require_api_key fails CLOSED when attached as dependency

P0 Security Fixes (Phase 8.3):
- Removed import-time app creation to prevent auth fail-open
- Use create_app() factory explicitly in ASGI entrypoint
- Example: uvicorn blackice.api:create_app --factory
"""

from __future__ import annotations

import os
from typing import Callable

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from blackice.api.auth import RequireAPIKey, is_auth_configured, require_api_key
from blackice.api.routes import health_router, providers_router, runs_router


def create_app(
    *,
    require_auth: bool | None = None,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        require_auth: If True, require API key for all endpoints except health.
                     If None (default), auto-detect from BLACKICE_API_KEY env var.
        allowed_origins: List of allowed CORS origins. If None, defaults to
                        localhost only (secure default).
    """
    # Determine if auth is enabled
    # P0 Fix: Use is_auth_configured() to read env var at runtime, not import time
    auth_enabled = require_auth
    if auth_enabled is None:
        auth_enabled = is_auth_configured()

    # Determine CORS origins
    if allowed_origins is None:
        # Secure default: localhost only
        # Set BLACKICE_CORS_ORIGINS env var for production (comma-separated)
        cors_env = os.environ.get("BLACKICE_CORS_ORIGINS", "")
        if cors_env:
            allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        else:
            allowed_origins = [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
            ]

    app = FastAPI(
        title="BLACKICE API",
        description="AI-powered autonomous software development pipeline",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS (secure: no wildcard with credentials)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,  # Secure: don't allow credentials with CORS
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
        max_age=600,  # Cache preflight for 10 minutes
    )

    # Global exception handler for auth errors
    @app.exception_handler(401)
    async def auth_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "detail": str(exc)},
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Include health router (no auth required for health checks)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(providers_router, prefix="/api/v1")

    # Include runs router with optional auth dependency
    if auth_enabled:
        # Add auth dependency to runs router
        app.include_router(
            runs_router,
            prefix="/api/v1",
            dependencies=[Depends(require_api_key)],
        )
    else:
        app.include_router(runs_router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint with API info."""
        return {
            "name": "BLACKICE API",
            "version": "3.0.0",
            "docs": "/docs",
            "auth_enabled": str(auth_enabled).lower(),
        }

    return app


# P0 Fix (Phase 8.3): Do NOT create app at import time.
# This prevents auth fail-open when env vars are set after import.
# Use the factory pattern:
#   uvicorn blackice.api:create_app --factory
# Or in code:
#   from blackice.api import create_app
#   app = create_app()

__all__ = ["create_app"]
