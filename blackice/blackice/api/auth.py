"""Authentication for BLACKICE API.

P0 Security: Simple API key authentication.
In production, replace with proper OAuth2/JWT.

P0 Fix (Phase 8.2): Removed import-time env caching to prevent fail-open bug.
Auth now reads env var at runtime and fails closed when dependency is attached.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

# API key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_configured_api_key() -> str | None:
    """Read the configured API key at runtime.

    P0 Fix: Do NOT cache at import time. This ensures env vars set after
    import (e.g., via python-dotenv, runtime config) are properly detected.
    """
    key = os.environ.get("BLACKICE_API_KEY")
    return key if key and len(key) > 0 else None


def _verify_api_key(api_key: str | None, *, configured_key: str) -> bool:
    """Verify the API key using constant-time comparison.

    Args:
        api_key: The key provided by the client
        configured_key: The expected key from configuration

    Returns:
        True if keys match, False otherwise
    """
    if not api_key:
        return False
    # Constant-time comparison to prevent timing attacks
    return secrets.compare_digest(api_key, configured_key)


async def require_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> None:
    """Dependency that requires a valid API key.

    P0 Fix: Fails CLOSED when this dependency is attached.
    - If BLACKICE_API_KEY is not configured -> 500 Server Error
    - If key is missing/invalid -> 401 Unauthorized

    This ensures that if you attach this dependency, auth is ALWAYS enforced.
    Use this for protected routes.
    """
    configured = _get_configured_api_key()

    if not configured:
        # Fail closed: If this dependency is attached, auth MUST be configured
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: BLACKICE_API_KEY is not set",
        )

    if not _verify_api_key(api_key, configured_key=configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def optional_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> bool:
    """Dependency that optionally validates API key.

    Returns True if a valid API key was provided, False otherwise.
    Does not raise. Useful for endpoints with different behavior for auth users.
    """
    configured = _get_configured_api_key()
    if not configured:
        return False
    return _verify_api_key(api_key, configured_key=configured)


# Type aliases for dependency injection
RequireAPIKey = Annotated[None, Depends(require_api_key)]
OptionalAPIKey = Annotated[bool, Depends(optional_api_key)]


def is_auth_configured() -> bool:
    """Check if authentication is configured (for use in create_app).

    This reads the env var at call time, not import time.
    """
    return _get_configured_api_key() is not None


def generate_api_key() -> str:
    """Generate a secure random API key.

    Use this to generate keys for clients:
        python -c "from blackice.api.auth import generate_api_key; print(generate_api_key())"
    """
    return secrets.token_urlsafe(32)
