"""Authentication for BLACKICE API.

P0 Security: Simple API key authentication.
In production, replace with proper OAuth2/JWT.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

# API key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Environment variable for API key (set BLACKICE_API_KEY to enable auth)
# If not set, authentication is disabled (dev mode)
_API_KEY = os.environ.get("BLACKICE_API_KEY")


def _is_auth_enabled() -> bool:
    """Check if authentication is enabled."""
    return _API_KEY is not None and len(_API_KEY) > 0


def _verify_api_key(api_key: str | None) -> bool:
    """Verify the API key using constant-time comparison."""
    if not _is_auth_enabled():
        return True  # Auth disabled in dev mode

    if not api_key:
        return False

    # Constant-time comparison to prevent timing attacks
    return secrets.compare_digest(api_key, _API_KEY)


async def require_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """Dependency that requires a valid API key.

    If BLACKICE_API_KEY is not set, authentication is bypassed (dev mode).
    In production, always set BLACKICE_API_KEY.
    """
    if not _is_auth_enabled():
        return None  # Auth disabled

    if not _verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


async def optional_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """Dependency that optionally validates API key.

    Returns the key if valid, None otherwise. Does not raise.
    Useful for endpoints that have different behavior for authenticated users.
    """
    if _verify_api_key(api_key):
        return api_key
    return None


# Type alias for dependency injection
RequireAPIKey = Annotated[str | None, Depends(require_api_key)]
OptionalAPIKey = Annotated[str | None, Depends(optional_api_key)]


def generate_api_key() -> str:
    """Generate a secure random API key.

    Use this to generate keys for clients:
        python -c "from blackice.api.auth import generate_api_key; print(generate_api_key())"
    """
    return secrets.token_urlsafe(32)
