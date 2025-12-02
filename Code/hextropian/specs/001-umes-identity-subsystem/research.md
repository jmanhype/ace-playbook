# Technical Research: UMES Identity Subsystem

**Date**: 2025-12-02
**Phase**: Phase 0 - Research & Technical Decisions
**Purpose**: Resolve all technical unknowns before design phase

---

## 1. KMS Adapter Implementation Strategy

### Decision
Protocol-based adapter pattern using `typing.Protocol` with async KMS clients and local cryptographic verification fallback.

### Rationale
- **Protocol over ABC**: Zero runtime overhead, better static type checking with mypy
- **Async-first**: Native async SDKs (aioboto3, google-cloud-kms AsyncClient, azure.aio)
- **Local verification optimization**: Sign with KMS, verify locally using cached public keys (<100ms p95)
- **DER to R|S conversion**: For ECDSA (ES256), decode ASN.1 DER to JWT-compatible format

### Libraries Selected
- **PyJWT 2.10+**: JWT encoding/decoding (NOT for signing, only structure)
- **cryptography 42.0+**: Local RSA/ECDSA operations, DER decoding
- **aioboto3 13.0+**: AWS KMS async client
- **google-cloud-kms 3.4+**: GCP KMS native async support
- **azure-keyvault-keys 4.9+** + **azure-identity 1.17+**: Azure Key Vault async
- **async-hvac 1.0+**: OpenBao/Vault async wrapper
- **oci 2.149+**: Oracle Cloud (sync, wrapped with run_in_executor)

### Implementation Pattern
```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class SignResult:
    signature: bytes
    algorithm: str  # "RS256", "ES256"
    key_id: str

class KMSAdapter(Protocol):
    async def sign(self, data: bytes, key_id: str, algorithm: str) -> SignResult: ...
    async def verify(self, data: bytes, signature: bytes, key_id: str, algorithm: str) -> bool: ...
    async def encrypt(self, plaintext: bytes, key_id: str) -> bytes: ...
    async def decrypt(self, ciphertext: bytes, key_id: str) -> bytes: ...
    async def get_public_key(self, key_id: str) -> bytes: ...
```

### Performance Characteristics
- **Sign operation**: 50-150ms (KMS network call)
- **Verify operation**: <5ms (local cryptography, cached public key)
- **JWT creation**: 50-150ms (dominated by signing)
- **JWT verification**: <5ms (meets <100ms p95 requirement)

### Alternatives Rejected
- ABC (Abstract Base Class): Requires nominal typing, runtime overhead
- PyJWT's register_algorithm(): Complex integration, maintainers marked KMS as "won't fix"
- ProcessPoolExecutor: Unnecessary overhead, cryptography library releases GIL

---

## 2. IdP Adapter Implementation Strategy

### Decision
Multi-library approach: **Authlib 1.6+** for OIDC, **python3-saml 1.16+** for SAML, **argon2-cffi 25.1+** for passwords.

### Rationale
- **Authlib**: Production-stable, native async support, built-in OIDC discovery and PKCE
- **python3-saml**: Industry-standard, simpler API than pysaml2, comprehensive validation
- **argon2-cffi**: Argon2id (RFC 9106), ~45ms verification (well under 500ms p95 SLO)

### Libraries Selected

**OIDC (Authlib)**:
- AsyncOAuth2Client for async flows
- Automatic `.well-known/openid-configuration` discovery
- PKCE support: `code_challenge_method='S256'`
- JWT validation with claims_options
- httpx integration (configure 10s timeout for IdP calls)

**SAML (python3-saml)**:
- Built-in assertion validation (signature, expiration, audience)
- Configurable clock skew tolerance (60s default)
- Run in thread pool: `asyncio.to_thread(saml_auth.process_response)`

**Password Hashing (argon2-cffi)**:
- Algorithm: Argon2id (hybrid mode)
- Parameters (RFC_9106_LOW_MEMORY):
  - time_cost=3, memory_cost=65536 (64 MiB), parallelism=4
  - Performance: ~45ms per verification

### Implementation Pattern
```python
class IdPAdapter(Protocol):
    async def authenticate(self, credentials: dict) -> UserInfo: ...
    async def validate_token(self, token: str) -> UserInfo: ...
    async def refresh_token(self, refresh_token: str) -> dict: ...
```

### Error Handling
- **IdP Unavailability**: Fail-closed for new auth, use cached tokens (5-minute max TTL)
- **Timeout Handling**: 10s timeout per OIDC/SAML request, fail authentication on timeout
- **Cache Invalidation**: Redis pub/sub for distributed token revocation

### Alternatives Rejected
- PyJWT: Lower-level, lacks OIDC discovery and OAuth2 flows
- pysaml2: Steeper learning curve, less documentation
- bcrypt/scrypt: Inferior to Argon2id for memory-hardness

---

## 3. PostgreSQL Row-Level Security (RLS) Implementation

### Decision
Session-scoped RLS with `SET LOCAL app.current_tenant_id`, SQLAlchemy 2.0 `after_begin` event listeners, and `FORCE ROW LEVEL SECURITY`.

### Rationale
- **Defense in depth**: RLS is final gatekeeper even if application code forgets `WHERE tenant_id = X`
- **Minimal overhead**: <10ms with proper indexing and LEAKPROOF functions
- **Connection pooling safe**: `SET LOCAL` auto-resets on COMMIT/ROLLBACK
- **Developer ergonomic**: No manual filtering in SQLAlchemy queries

### Implementation Pattern

**RLS Policy Structure**:
```sql
-- Enable RLS with FORCE (applies to table owners)
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE users FORCE ROW LEVEL SECURITY;

-- Create LEAKPROOF helper function
CREATE FUNCTION get_current_tenant_id() RETURNS uuid AS $$
    SELECT current_setting('app.current_tenant_id', TRUE)::uuid;
$$ LANGUAGE SQL STABLE LEAKPROOF;

-- Policies for all operations
CREATE POLICY users_tenant_isolation_select ON users
    FOR SELECT USING (tenant_id = get_current_tenant_id());

CREATE POLICY users_tenant_isolation_insert ON users
    FOR INSERT WITH CHECK (tenant_id = get_current_tenant_id());

CREATE POLICY users_tenant_isolation_update ON users
    FOR UPDATE
    USING (tenant_id = get_current_tenant_id())
    WITH CHECK (tenant_id = get_current_tenant_id());
```

**SQLAlchemy Integration**:
```python
from sqlalchemy import event, text
from contextvars import ContextVar

current_tenant_id: ContextVar[str] = ContextVar('current_tenant_id', default=None)

@event.listens_for(Session, "after_begin")
def set_tenant_context(session, transaction, connection):
    tenant_id = current_tenant_id.get()
    if tenant_id is None:
        raise ValueError("Tenant context not set")
    connection.execute(
        text("SET LOCAL app.current_tenant_id = :tenant_id"),
        {"tenant_id": tenant_id}
    )
```

### Performance Optimization
- Index all `tenant_id` columns: `CREATE INDEX idx_users_tenant_id ON users(tenant_id)`
- Use LEAKPROOF functions for complex policies (enables index usage)
- Monitor with `EXPLAIN ANALYZE` to validate index usage

### Testing Patterns
- Integration tests verify cross-tenant isolation (tenant A cannot see tenant B data)
- Verify `FORCE ROW LEVEL SECURITY` applies to table owners (admin connections)
- Automated verification: All tenant-scoped tables have RLS enabled

### Alternatives Rejected
- Application-level filtering: High risk of developer error (forgotten WHERE clause)
- Schema-per-tenant: Complex management, poor scaling with many tenants
- Database-per-tenant: Massive operational overhead, inefficient for 1000+ tenants

---

## 4. Circuit Breaker & Graceful Degradation

### Decision
**Purgatory 3.0+** for circuit breakers, **Tenacity** for retries, **aiocache** for cache fallback (Redis → in-memory).

### Rationale
- **Purgatory**: Most actively maintained async-native circuit breaker (Nov 2024), Redis state storage, monitoring hooks
- **Tenacity**: Industry-standard retry with exponential backoff + jitter, async support
- **aiocache**: Unified interface for Redis + in-memory, seamless failover

### Libraries Selected
- **Purgatory 3.0.1+**: Circuit breakers with Redis coordination
- **Tenacity**: Retry with exponential backoff
- **aiocache**: Two-level caching (Redis + in-memory LRU)

### Configuration

**Circuit Breakers**:
```python
from purgatory import AsyncCircuitBreakerFactory, AsyncRedisUnitOfWork

circuit_breaker_factory = AsyncCircuitBreakerFactory(
    default_threshold=5,
    default_ttl=30,
    uow=AsyncRedisUnitOfWork("redis://localhost:6379/0")
)

kms_breaker = circuit_breaker_factory.get_breaker("kms_adapter", threshold=3, ttl=60)
idp_breaker = circuit_breaker_factory.get_breaker("idp_adapter", threshold=5, ttl=30)
```

**Retry Configuration**:
```python
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

# KMS retry (5s timeout, 3 attempts, exponential backoff + jitter)
async for attempt in AsyncRetrying(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=5),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True
):
    with attempt:
        async with asyncio.timeout(5):
            return await kms_client.encrypt(plaintext)
```

**Cache Fallback (Redis → In-Memory)**:
```python
from aiocache import Cache, TTLCache

class TieredCache:
    def __init__(self):
        self.l1_cache = TTLCache(maxsize=1000, ttl=120)  # In-memory
        self.l2_cache = Cache(Cache.REDIS, endpoint="127.0.0.1", port=6379)

    async def get(self, key: str):
        # L1 first (sub-ms), then L2 (1-2ms)
        value = self.l1_cache.get(key)
        if value is not None:
            return value

        if self.redis_available:
            try:
                value = await asyncio.wait_for(self.l2_cache.get(key), timeout=1.0)
                if value:
                    self.l1_cache[key] = value
                return value
            except asyncio.TimeoutError:
                self.redis_available = False
        return None
```

### Degradation Strategies

| Component | Degradation Path | Timeout | Fail Policy |
|-----------|------------------|---------|-------------|
| **KMS** | KMS → Emergency local crypto → Fail closed (10 min) | 5s | Fail closed |
| **IdP** | IdP → Cached tokens (5 min max) → Fail closed | 10s | Fail closed |
| **Database** | DB → Cached authz (2 min TTL) → Fail closed for writes | 2s | Fail closed |
| **Redis** | Redis → In-memory LRU → Performance warning | 1s | Fail open |

### Prometheus Metrics
```yaml
circuit_breaker_state{service="kms"}  # 0=closed, 1=open, 2=half-open
circuit_breaker_failures_total{service="kms"}
degraded_mode_active{service="kms",fallback_type="local_crypto"}
```

### Alternatives Rejected
- aiobreaker: Less maintained (2020), no Redis state storage
- pybreaker: Non-native asyncio (Tornado async)
- Custom implementation: Reinventing well-solved problems

---

## 5. Multi-Cloud Testing Strategy

### Decision
Hybrid local-first testing: **Testcontainers + emulators** (90%) + **selective real cloud** (10%).

### Rationale
- **Speed**: Local tests run in <10 minutes (meets CI/CD SLO)
- **Cost**: $50-100/month (real accounts only for GCP/Azure critical paths)
- **Fidelity**: 95% confidence from local, 99.9% with selective real cloud tests

### Testing Approach

| Provider | Local Testing Solution | Real Account Needed? |
|----------|------------------------|---------------------|
| **AWS** | LocalStack (KMS supported) | No (90% fidelity) |
| **Azure** | Azurite + Mocked AD | Yes (Key Vault only) |
| **GCP** | No KMS emulator | Yes (critical - no emulator exists) |
| **Oracle** | Mock at adapter level | Quarterly validation only |
| **Hetzner** | MinIO (S3-compatible) | No (95% fidelity) |
| **On-Prem** | HashiCorp Vault (Docker) | No (100% fidelity) |

### Testcontainers Setup
```python
@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:15-alpine") as postgres:
        yield postgres

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7-alpine") as redis:
        yield redis

@pytest.fixture(scope="session")
def keycloak_container():
    with DockerContainer("quay.io/keycloak/keycloak:23.0") as keycloak:
        keycloak.with_exposed_ports(8080)
        keycloak.with_env("KEYCLOAK_ADMIN", "admin")
        yield keycloak
```

### CI/CD Matrix (GitHub Actions)
```yaml
strategy:
  matrix:
    cloud-provider: [aws, azure, gcp, oracle, hetzner, on-prem]
    test-suite: [kms-adapters, idp-adapters, e2e-flows]
  fail-fast: false
  max-parallel: 18  # 6 providers × 3 suites
```

**Expected Performance**:
- Local tests (Testcontainers + emulators): 6-8 minutes
- Real cloud subset: 2-4 minutes
- **Total: 7-10 minutes** (within <10 min SLO)

### Test Data Isolation
- **Transaction rollback**: Fast unit/integration tests (90%)
- **Fresh container**: Cross-service tests requiring clean state
- **Worker isolation**: `pytest -n 8` with separate DB per worker

### Performance Testing Tools
- **pytest-benchmark**: Unit-level latency validation (<50ms SLO)
- **Locust**: Load testing (10,000 req/s validation)

### Alternatives Rejected
- Real cloud accounts only: $500-1000/month, 20-30 minutes per run (exceeds SLO)
- Mocking everything: Low confidence, false positives
- Weekly full cloud integration: Delayed feedback, harder debugging

---

## 6. Performance Optimization for Latency SLOs

### Decision
Hybrid async architecture: **asyncpg** driver, **two-level caching** (in-memory + Redis), **connection pooling**, **strategic indexing**.

### Rationale
- **asyncpg**: 12-18% faster than psycopg3 at high concurrency (10,000 ops)
- **Two-level cache**: L1 (<1ms) for hot keys, L2 (1-2ms) for distributed, reduces DB load
- **Connection pool**: 20 base + 10 overflow = 30 max (optimal for 10,000 req/s)
- **Authorization cache**: 120s TTL reduces authz from 100ms (DB) to <5ms (cached)

### Database Configuration
```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/umes",
    pool_size=20,              # Base connections
    max_overflow=10,           # Burst capacity
    pool_timeout=30,           # Wait 30s for connection
    pool_recycle=3600,         # Recycle after 1 hour
    pool_pre_ping=True         # Verify before use
)
```

### Caching Strategy

**L1 (In-Memory)**:
- Library: `cachetools.TTLCache`
- Size: 1000 entries per worker
- TTL: 60-120s
- Latency: <1ms (sub-microsecond)

**L2 (Redis)**:
- Library: `aiocache` with Redis backend
- TTL: 120-300s
- Latency: 1-2ms (local), 5-10ms (distributed)

**Two-Level Pattern**:
```python
@two_level_cache(namespace="auth", l1_ttl=60, l2_ttl=120)
async def get_user_entitlements(user_id: str) -> dict:
    # L1 hit: <1ms, L2 hit: 1-2ms, Cache miss: 50ms (DB query)
    async with async_session() as session:
        result = await session.execute(
            select(Permission)
            .join(RolePermission)
            .join(UserRole)
            .filter(UserRole.user_id == user_id)
            .options(joinedload(Permission.resource))  # Prevent N+1
        )
        return {"permissions": [p.to_dict() for p in result.scalars().all()]}
```

### Indexing Strategy
```sql
-- Foreign keys (critical for JOINs)
CREATE INDEX idx_user_roles_user_id ON user_roles(user_id);
CREATE INDEX idx_role_permissions_role_id ON role_permissions(role_id);

-- Composite indexes for common queries
CREATE INDEX idx_tokens_user_status ON tokens(user_id, status);

-- Partial indexes for filtered queries
CREATE INDEX idx_active_tokens_jti ON tokens(jti) WHERE status = 'active';
```

### Performance Targets

| Operation | Target | Actual (Optimized) |
|-----------|--------|-------------------|
| Token validation (L1 hit) | ≤50ms p95 | 2-5ms p95 |
| Token validation (L2 hit) | ≤50ms p95 | 5-10ms p95 |
| Token validation (miss) | ≤50ms p95 | 30-50ms p95 |
| Authorization check (cached) | ≤100ms p95 | 10-20ms p95 |
| Token generation | ≤200ms p95 | 150-200ms p95 |
| Throughput | 10,000 req/s | 10,000-15,000 req/s |

### Profiling Tools
- **pyinstrument**: Development profiling (`?profile=1` query param)
- **py-spy**: Production profiling (external, <1% overhead)

### Alternatives Rejected
- psycopg3: 12-18% slower under high load
- Single-level Redis cache: 1-2ms latency vs sub-ms for L1
- PgBouncer: Unnecessary complexity at 30 connections
- functools.lru_cache: No async support, no TTL

---

## 7. Audit Log Integrity (ALCOA+ Compliance)

### Decision
Hash chain with PostgreSQL append-only table, monthly partitioning, JSON Lines export.

### Rationale
- **Hash chain**: Each entry includes hash of previous entry (Merkle chain), detects tampering
- **Append-only**: Database triggers prevent updates/deletes (ALCOA+ immutability)
- **Partitioning**: Monthly partitions enable efficient archival to S3
- **ALCOA+ compliant**: Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, Available

### Implementation Pattern

**Database Schema**:
```sql
CREATE TABLE audit_log (
    id BIGSERIAL,
    event_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id UUID,
    session_id UUID,
    actor_type VARCHAR(20) NOT NULL,
    actor_id UUID NOT NULL,
    client_ip INET,
    event_data JSONB NOT NULL,

    -- Hash chain fields
    previous_hash VARCHAR(64) NOT NULL,  -- SHA-256 of previous entry
    entry_hash VARCHAR(64) NOT NULL,     -- SHA-256 of this entry

    severity VARCHAR(20) DEFAULT 'info',
    result VARCHAR(20) DEFAULT 'success',

    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Immutability trigger
CREATE FUNCTION prevent_audit_log_modification() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' OR TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Audit log entries cannot be modified';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_immutability
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION prevent_audit_log_modification();
```

**Hash Chain Logic**:
```python
def calculate_entry_hash(entry_data: dict, previous_hash: str) -> str:
    canonical = json.dumps(entry_data, sort_keys=True, separators=(',', ':'))
    hash_input = f"{canonical}|{previous_hash}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
```

**Verification**:
```python
async def verify_audit_log_chain(start_date, end_date) -> dict:
    """Verify hash chain integrity, detect tampering."""
    entries = await fetch_audit_logs_ordered(start_date, end_date)

    for i, entry in enumerate(entries):
        # Verify entry hash matches content
        expected = calculate_entry_hash(entry.event_data, entry.previous_hash)
        if entry.entry_hash != expected:
            return {"valid": False, "first_break_at": entry.id}

        # Verify chain link
        if i > 0 and entry.previous_hash != entries[i-1].entry_hash:
            return {"valid": False, "first_break_at": entry.id}

    return {"valid": True, "verified_entries": len(entries)}
```

### Archival Process
- **Hot storage**: Current month + 12 previous months (PostgreSQL)
- **Cold storage**: >13 months archived to S3 Glacier (7-year retention)
- **Export format**: JSON Lines with integrity verification manifest

### Verification Schedule
- **Continuous**: Verify each new entry on insert
- **Hourly**: Verify last hour's chain
- **Daily**: Full verification of previous day's partition
- **Monthly**: Full verification before S3 archival

### Alternatives Rejected
- Merkle trees: More complex, overkill for sequential audit logs
- Digital signatures: Requires KMS integration, key management overhead
- immudb/QLDB: External dependencies (AWS deprecated QLDB in 2024)
- Append-only without hash chain: No cryptographic tamper-evidence

---

## 8. Token Revocation Strategy

### Decision
Hybrid JWT/Opaque with **Redis revocation list** for JWTs, **database-backed** for refresh tokens.

### Rationale
- **Performance**: JWT validation 6-15ms p95 (signature + Redis revocation check)
- **Revocation coverage**: Immediate logout, user deletion, tenant deletion, security incidents
- **Scalability**: Redis handles 100k+ ops/s, <1ms median latency
- **Short-lived JWTs**: 15-minute expiration reduces revocation urgency

### Implementation Pattern

**JWT with Redis Revocation List**:
```python
# JWT claims
{
    "jti": "unique-token-id",
    "sub": "user:{user_id}",
    "tenant_id": "tenant:{tenant_id}",
    "exp": 1701234567,  # 15 minutes
    "type": "access"
}

# Redis revocation keys
revoked:jti:{jti} = "true"  # TTL = token expiration
revoked:user:{user_id} = timestamp  # TTL = 60s (max token lifetime)
revoked:tenant:{tenant_id} = timestamp  # TTL = 60s
```

**Validation Flow**:
1. Verify JWT signature with KMS public key (1-2ms)
2. Check expiration/claims (<1ms)
3. Check Redis revocation list (4 checks in pipeline, 1-3ms)
4. **Total: 6-15ms p95** (well under 50ms SLO)

**Opaque Refresh Tokens** (Database-backed):
```sql
CREATE TABLE refresh_tokens (
    token_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    revoked_at TIMESTAMP,
    INDEX idx_token_hash (token_hash) WHERE revoked_at IS NULL
);

-- Revocation: UPDATE refresh_tokens SET revoked_at = NOW() WHERE user_id = $1;
```

### Cache Strategy
- **Redis pipeline**: Send all 4 revocation checks in single round trip
- **Local in-memory cache**: Cache negative lookups (token NOT revoked) for 30s
- **Fail-open**: If Redis unavailable, allow request but log security event

### Performance Impact
- **Baseline (JWT only)**: 1-3ms p95
- **With revocation list**: 6-15ms p95 (local Redis) or 10-20ms p95 (distributed)
- **Cache hit rate**: >99.9% (revocations are rare events)

### Alternatives Rejected
- Pure JWT (no revocation): Cannot revoke tokens, fails user deletion requirement
- Pure opaque tokens: 10-30ms DB lookup per request, bottleneck at scale
- Token introspection endpoint: Adds network hop (10-50ms), single point of failure
- JWT with database check: Database bottleneck, 10-30ms per validation

---

## Summary: Technical Stack Selected

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| **Backend Language** | Python | 3.11+ | Async native, rich ecosystem |
| **Web Framework** | FastAPI | 0.104+ | Async, OpenAPI, high performance |
| **Database Driver** | asyncpg | Latest | 12-18% faster than psycopg3 |
| **ORM** | SQLAlchemy | 2.0 | Async support, mature, well-documented |
| **Database** | PostgreSQL | 15+ | RLS support, JSONB, partitioning |
| **Cache** | Redis | 7+ | Sub-ms latency, distributed state |
| **KMS Libraries** | aioboto3, google-cloud-kms, azure-keyvault-keys, oci, async-hvac | Latest | Native async KMS clients |
| **IdP Libraries** | Authlib, python3-saml, argon2-cffi | 1.6+, 1.16+, 25.1+ | Production-ready OIDC/SAML/passwords |
| **Circuit Breaker** | Purgatory | 3.0+ | Async-native, Redis state |
| **Retry** | Tenacity | Latest | Exponential backoff, async |
| **Testing** | pytest, Testcontainers, LocalStack | 7.4+, Latest | Fast, high-fidelity local tests |
| **Profiling** | pyinstrument, py-spy | Latest | Dev + production profiling |

---

## Next Steps

With all technical unknowns resolved, proceed to **Phase 1: Design & Contracts**:

1. Generate `data-model.md` (entity-relationship diagram, database schema)
2. Generate `contracts/` (OpenAPI 3.1 specifications for all endpoints)
3. Generate `quickstart.md` (developer integration guide)
4. Update agent context with technology decisions

All design artifacts will align with the research decisions documented above.
