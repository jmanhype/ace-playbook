# UMES Quick

Start Guide

**Target Audience**: Service developers integrating with UMES for authentication and authorization
**Time to Complete**: ~1 hour
**Prerequisites**: Hextropian service account, access to UMES endpoints

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication Setup](#authentication-setup)
3. [Authorization Integration](#authorization-integration)
4. [API Key Management](#api-key-management)
5. [Error Handling](#error-handling)
6. [Production Checklist](#production-checklist)

---

## Overview

UMES (Unified Management of Entitlements and Identity Subsystem) provides:
- **Authentication**: Local, OIDC, and SAML identity providers
- **Authorization**: Capability-based + role-based access control
- **Multi-tenancy**: Tenant-aware token validation and permission checks
- **Audit logging**: Comprehensive identity event tracking

### Architecture

```
┌──────────────┐         ┌─────────┐         ┌──────────────┐
│ Your Service │────────►│  UMES   │────────►│ IdP (Okta,   │
│              │  Verify  │         │  Auth   │  Auth0, etc) │
└──────────────┘  Token   └─────────┘         └──────────────┘
       │                       │
       │ Check Permission      │ Sign JWT
       ▼                       ▼
┌──────────────┐         ┌─────────┐
│ User Action  │         │   KMS   │
│ (requires    │         │ (GCP,   │
│  permission) │         │  AWS,   │
└──────────────┘         │  Azure) │
                         └─────────┘
```

### Key Endpoints

| Endpoint | Purpose | When to Use |
|----------|---------|-------------|
| `POST /auth/login` | User login | User-facing authentication |
| `POST /tokens/validate` | Validate JWT | Every protected request |
| `POST /authz/check` | Check permission | Before allowing action |
| `POST /api-keys` | Create API key | Programmatic access |

---

## Authentication Setup

### Step 1: Install UMES Client Library

```bash
# Python
pip install umes-client

# Node.js
npm install @hextropian/umes-client

# Go
go get github.com/hextropian/umes-go
```

### Step 2: Configure UMES Client

```python
# Python FastAPI example
from fastapi import FastAPI, Depends, HTTPException
from umes_client import UMESClient, UMESAuth

app = FastAPI()

# Initialize UMES client
umes = UMESClient(
    base_url="https://umes.hextropian.com",
    api_key="umes_live_YourServiceAPIKey...",  # Service account API key
    timeout=5.0  # 5-second timeout
)

# Dependency to validate JWT token
async def get_current_user(
    authorization: str = Header(...),
    tenant_id: str = Header(..., alias="X-Tenant-Id")
):
    """Validate JWT and return user context."""
    try:
        # Extract token from "Bearer <token>"
        token = authorization.replace("Bearer ", "")

        # Validate with UMES
        user_context = await umes.validate_token(token, tenant_id)

        return user_context
    except umes.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except umes.ExpiredTokenError:
        raise HTTPException(status_code=401, detail="Token expired")
```

### Step 3: Protect Your Endpoints

```python
@app.get("/api/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """List documents for authenticated user."""

    # user = {
    #     "user_id": "550e8400-e29b-41d4-a716-446655440000",
    #     "tenant_id": "tenant-123",
    #     "email": "user@example.com",
    #     "permissions": ["read:documents", "write:documents"]
    # }

    documents = await fetch_documents(user["tenant_id"])
    return {"documents": documents}
```

### Step 4: User Login Flow

```python
@app.post("/auth/login")
async def login(username: str, password: str):
    """Login endpoint (delegates to UMES)."""

    try:
        # Authenticate with UMES
        tokens = await umes.authenticate(
            username=username,
            password=password,
            tenant_slug="your-org"  # Tenant identifier
        )

        return {
            "access_token": tokens["access_token"],   # 15-minute expiration
            "refresh_token": tokens["refresh_token"], # 7-day expiration
            "expires_in": 900  # seconds
        }
    except umes.AuthenticationError as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")
```

### Step 5: Token Refresh

```python
@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token."""

    try:
        new_tokens = await umes.refresh_token(refresh_token)

        return {
            "access_token": new_tokens["access_token"],
            "expires_in": 900
        }
    except umes.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
```

---

## Authorization Integration

### Step 1: Define Required Permissions

Map your service actions to UMES entitlements:

```python
# permissions.py
PERMISSIONS = {
    "list_documents": "read:documents",
    "create_document": "write:documents",
    "delete_document": "delete:documents",
    "manage_users": "manage:users",
    "view_audit_logs": "read:audit_logs"
}
```

### Step 2: Check Permissions

```python
from umes_client import UMESClient

async def require_permission(permission: str):
    """Dependency to check if user has required permission."""

    async def _check(user: dict = Depends(get_current_user)):
        # Check if permission is in user's entitlements
        if permission not in user.get("permissions", []):
            # Double-check with UMES (in case of cache mismatch)
            has_permission = await umes.check_permission(
                user_id=user["user_id"],
                tenant_id=user["tenant_id"],
                permission=permission
            )

            if not has_permission:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permission: {permission}"
                )

        return user

    return _check

# Usage
@app.delete("/api/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user: dict = Depends(require_permission("delete:documents"))
):
    """Delete document (requires delete:documents permission)."""
    await delete_doc(doc_id, user["tenant_id"])
    return {"status": "deleted"}
```

### Step 3: Batch Permission Checks

For efficiency, check multiple permissions in one call:

```python
@app.post("/api/documents/bulk-delete")
async def bulk_delete_documents(
    doc_ids: list[str],
    user: dict = Depends(get_current_user)
):
    """Delete multiple documents."""

    # Check permissions in batch
    permissions_required = [
        {"permission": "delete:documents", "resource_id": doc_id}
        for doc_id in doc_ids
    ]

    results = await umes.batch_check_permissions(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        checks=permissions_required
    )

    # Process only allowed deletions
    allowed_docs = [
        doc_id for doc_id, result in zip(doc_ids, results)
        if result["granted"]
    ]

    await delete_docs(allowed_docs, user["tenant_id"])

    return {
        "deleted": len(allowed_docs),
        "denied": len(doc_ids) - len(allowed_docs)
    }
```

---

## API Key Management

For programmatic access (CI/CD, integrations, background jobs):

### Step 1: Create API Key

```python
# Admin creates API key for service account
api_key_response = await umes.create_api_key(
    name="Document Processor Service",
    tenant_id="tenant-123",
    user_id="service-account-uuid",
    permissions=["read:documents", "write:documents"],
    expires_in_days=365
)

# Response:
# {
#     "key_id": "umes_live_AbCd1234",
#     "secret": "umes_live_AbCd1234.YourSecretKey32Characters...",  # SHOW ONCE
#     "expires_at": "2026-12-02T00:00:00Z"
# }

# CRITICAL: Store secret securely (environment variable, secret manager)
# Secret is shown ONLY ONCE and cannot be retrieved later
```

### Step 2: Use API Key

```python
import httpx

# Use API key in X-API-Key header
async def call_umes_with_api_key():
    headers = {
        "X-API-Key": "umes_live_AbCd1234.YourSecretKey32Characters...",
        "X-Tenant-Id": "tenant-123"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://umes.hextropian.com/api/v1/authz/check",
            headers=headers,
            json={
                "user_id": "user-uuid",
                "permission": "read:documents"
            }
        )
        return response.json()
```

### Step 3: Rotate API Key

```python
# Rotate before expiration
new_key = await umes.rotate_api_key(
    key_id="umes_live_AbCd1234",
    grace_period_days=7  # Old key valid for 7 more days
)

# Update your service with new key within grace period
```

### Step 4: Revoke API Key

```python
# Immediate revocation (security incident)
await umes.revoke_api_key(key_id="umes_live_AbCd1234")

# Audit: All requests with revoked key will fail immediately
```

---

## Error Handling

### Error Response Format

All UMES errors follow consistent format:

```json
{
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Invalid credentials",
    "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
    "details": {
      "reason": "password_mismatch"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Meaning | Action |
|------|-------------|---------|--------|
| `AUTHENTICATION_FAILED` | 401 | Invalid credentials | Prompt user to re-login |
| `AUTHORIZATION_DENIED` | 403 | Missing permission | Show "Access Denied" UI |
| `INVALID_TOKEN` | 401 | Token expired/malformed | Refresh token or re-authenticate |
| `TENANT_ISOLATION_VIOLATION` | 403 | Cross-tenant access attempt | Log security event, block request |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Wait (retry-after header) |
| `KMS_UNAVAILABLE` | 503 | KMS service down | Retry with exponential backoff |
| `IDP_UNAVAILABLE` | 503 | IdP service down | Show maintenance message |

### Error Handling Best Practices

```python
from umes_client import (
    UMESClient,
    AuthenticationError,
    AuthorizationError,
    InvalidTokenError,
    RateLimitError,
    ServiceUnavailableError
)

async def handle_umes_errors():
    try:
        result = await umes.check_permission(...)
        return result

    except InvalidTokenError:
        # Token expired or invalid - redirect to login
        return redirect("/login")

    except AuthorizationError as e:
        # User lacks permission - show access denied
        return JSONResponse(
            status_code=403,
            content={"error": "Access denied", "required_permission": e.permission}
        )

    except RateLimitError as e:
        # Rate limit hit - wait and retry
        retry_after = e.retry_after_seconds
        await asyncio.sleep(retry_after)
        return await handle_umes_errors()

    except ServiceUnavailableError as e:
        # UMES down - use cached permissions or fail open/closed
        logger.error(f"UMES unavailable: {e.correlation_id}")

        # Option 1: Fail closed (deny all)
        raise HTTPException(status_code=503, detail="Service unavailable")

        # Option 2: Use cached permissions (if available)
        # cached_perms = await get_cached_permissions(user_id)
        # return cached_perms
```

---

## Production Checklist

Before deploying UMES integration to production:

### Security
- [ ] API keys stored in environment variables or secret manager (NOT in code)
- [ ] Token validation on EVERY protected endpoint
- [ ] Permission checks before allowing sensitive operations
- [ ] Tenant ID validated and passed in `X-Tenant-Id` header
- [ ] HTTPS enforced for all UMES communication
- [ ] Correlation IDs logged for debugging (error.correlation_id)

### Performance
- [ ] Token validation cached (in-memory, 60-second TTL)
- [ ] Permission results cached (2-minute TTL with event-driven invalidation)
- [ ] Batch permission checks used for bulk operations
- [ ] Connection pooling enabled for UMES client
- [ ] Timeout configured (5-10 seconds max)
- [ ] Circuit breaker configured for UMES calls

### Reliability
- [ ] Graceful degradation strategy defined (fail open vs fail closed)
- [ ] Retry logic with exponential backoff for transient errors
- [ ] Health check endpoint verifies UMES connectivity
- [ ] Alerts configured for UMES errors (>5% error rate)
- [ ] Fallback to cached permissions if UMES unavailable

### Monitoring
- [ ] UMES response times tracked (p50, p95, p99)
- [ ] Authentication success/failure rates monitored
- [ ] Authorization denial rates tracked by permission
- [ ] Rate limit hits logged and alerted
- [ ] Token expiration errors tracked (may indicate client clock skew)

### Audit & Compliance
- [ ] All identity operations logged with user_id, tenant_id, action
- [ ] Audit log queries implemented for compliance reports
- [ ] User deletion triggers token revocation
- [ ] Tenant deletion triggers bulk token revocation
- [ ] API key rotation policy documented (12-month max lifetime)

---

## Example Integration: Complete Service

```python
# main.py - Complete FastAPI service with UMES integration
from fastapi import FastAPI, Depends, HTTPException, Header
from umes_client import UMESClient, InvalidTokenError, AuthorizationError
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Document Service")

# Initialize UMES client
umes = UMESClient(
    base_url="https://umes.hextropian.com",
    api_key=os.getenv("UMES_API_KEY"),
    timeout=5.0
)

# Authentication dependency
async def get_current_user(
    authorization: str = Header(...),
    tenant_id: str = Header(..., alias="X-Tenant-Id")
):
    """Validate JWT and return user context."""
    try:
        token = authorization.replace("Bearer ", "")
        user = await umes.validate_token(token, tenant_id)
        return user
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Authorization dependency
def require_permission(permission: str):
    async def _check(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            has_perm = await umes.check_permission(
                user_id=user["user_id"],
                tenant_id=user["tenant_id"],
                permission=permission
            )
            if not has_perm:
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing permission: {permission}"
                )
        return user
    return _check

# Public endpoint (no auth)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Protected endpoint (authentication only)
@app.get("/api/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """List documents for user's tenant."""
    documents = await fetch_documents(user["tenant_id"])
    return {"documents": documents, "count": len(documents)}

# Protected endpoint (authentication + authorization)
@app.post("/api/documents")
async def create_document(
    title: str,
    content: str,
    user: dict = Depends(require_permission("write:documents"))
):
    """Create new document (requires write:documents permission)."""
    doc = await create_doc(
        title=title,
        content=content,
        tenant_id=user["tenant_id"],
        created_by=user["user_id"]
    )
    return {"document_id": doc.id, "status": "created"}

# Admin endpoint (requires admin permission)
@app.get("/api/admin/users")
async def list_all_users(user: dict = Depends(require_permission("manage:users"))):
    """List all users in tenant (admin only)."""
    users = await fetch_all_users(user["tenant_id"])
    return {"users": users, "count": len(users)}
```

---

## Next Steps

1. **Read the API Reference**: Full OpenAPI spec at `contracts/openapi.yaml`
2. **Explore Examples**: See `examples/` directory for language-specific samples
3. **Join the Community**: #umes channel in Hextropian Slack
4. **Report Issues**: GitHub issues or api-support@hextropian.com

---

## Support

- **Documentation**: https://docs.hextropian.com/umes
- **API Reference**: https://umes.hextropian.com/docs
- **Support Email**: api-support@hextropian.com
- **Slack**: #umes channel

---

**Last Updated**: 2025-12-02
**Version**: 1.0.0
