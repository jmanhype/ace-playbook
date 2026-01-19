"""BLACKICE 3.0 API.

FastAPI-based HTTP API for programmatic access to BLACKICE.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from blackice.api.routes import health_router, providers_router, runs_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="BLACKICE API",
        description="AI-powered autonomous software development pipeline",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(providers_router, prefix="/api/v1")
    app.include_router(runs_router, prefix="/api/v1")

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint with API info."""
        return {
            "name": "BLACKICE API",
            "version": "3.0.0",
            "docs": "/docs",
        }

    return app


# Create default app instance
app = create_app()

__all__ = ["app", "create_app"]
