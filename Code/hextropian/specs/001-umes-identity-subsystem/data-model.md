# Data Model: UMES Identity Subsystem

**Date**: 2025-12-02
**Phase**: Phase 1 - Design & Contracts
**Status**: Draft

---

## Overview

UMES data model supports multi-tenant identity, authentication, authorization, and audit logging with strict tenant isolation via PostgreSQL Row-Level Security (RLS).

**Key Design Principles**:
- **Tenant isolation**: All user-scoped data partitioned by `tenant_id` with RLS enforcement
- **Multi-tenancy**: Users can belong to multiple tenants with different roles per tenant
- **B2B2B hierarchy**: Support partner organizations with parent-child relationships
- **Audit trail**: Immutable append-only audit log with hash chain integrity
- **Soft deletes**: Critical entities use `deleted_at` timestamp for recovery and audit

---

## Entity-Relationship Diagram

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Tenant    │◄───────►│  Membership  │────────►│    User     │
│             │ 1     * │              │ *     1 │             │
│ - id        │         │ - tenant_id  │         │ - id        │
│ - name      │         │ - user_id    │         │ - email     │
│ - tier      │         │ - role_ids[] │         │ - idp_type  │
│ - parent_id │         │              │         │ - idp_subj  │
└─────────────┘         └──────────────┘         └─────────────┘
      │                        │
      │                        │
      │                        ▼
      │                 ┌─────────────┐
      │                 │    Role     │
      │                 │             │
      │                 │ - id        │
      │                 │ - name      │
      │                 │ - tenant_id │
      │                 └─────────────┘
      │                        │
      │                        │ *
      │                        ▼ *
      │                 ┌─────────────┐
      │                 │ Entitlement │
      │                 │             │
      │                 │ - id        │
      │                 │ - action    │
      │                 │ - resource  │
      │                 └─────────────┘
      │
      ├──────────────► ┌─────────────┐
      │         1    * │   APIKey    │
      │                │             │
      │                │ - id        │
      │                │ - user_id   │
      │                │ - tenant_id │
      │                │ - hash      │
      │                │ - expires   │
      │                └─────────────┘
      │
      └──────────────► ┌──────────────┐
               1    * │  AuditEvent  │
                      │              │
                      │ - id         │
                      │ - tenant_id  │
                      │ - event_type │
                      │ - prev_hash  │
                      │ - entry_hash │
                      └──────────────┘
```

---

## Core Entities

### 1. User

**Purpose**: Individual person with identity credentials, can be member of multiple tenants.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `email` (VARCHAR(255), UNIQUE): Email address (indexed)
- `email_verified` (BOOLEAN): Email verification status
- `name` (VARCHAR(255)): Full name
- `idp_type` (ENUM): Identity provider type ('local', 'oidc', 'saml')
- `idp_subject` (VARCHAR(512), UNIQUE): IdP-specific identifier (OIDC sub, SAML nameID, or local username)
- `password_hash` (VARCHAR(255), NULLABLE): Argon2id hash (only for local IdP)
- `mfa_enabled` (BOOLEAN): Multi-factor authentication status
- `mfa_secret` (TEXT, ENCRYPTED, NULLABLE): TOTP secret (KMS-encrypted)
- `created_at` (TIMESTAMPTZ): Creation timestamp
- `updated_at` (TIMESTAMPTZ): Last modification timestamp
- `deleted_at` (TIMESTAMPTZ, NULLABLE): Soft delete timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX idx_users_idp_subject ON users(idp_type, idp_subject) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_created_at ON users(created_at DESC);
```

**Constraints**:
- Email must be valid format (CHECK constraint)
- If `idp_type = 'local'`, `password_hash` MUST NOT be NULL
- If `idp_type IN ('oidc', 'saml')`, `password_hash` MUST be NULL

**RLS**: No tenant scoping (users are global, memberships provide tenant access)

---

### 2. Tenant

**Purpose**: Organization or customer instance with isolated data boundary.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `name` (VARCHAR(255), UNIQUE): Tenant display name
- `slug` (VARCHAR(100), UNIQUE): URL-safe identifier
- `tier` (ENUM): Subscription tier ('free', 'pro', 'enterprise')
- `parent_id` (UUID, FK → tenants.id, NULLABLE): Parent tenant for B2B2B hierarchy
- `status` (ENUM): Tenant status ('active', 'suspended', 'deleted')
- `settings` (JSONB): Tenant-specific configuration (SSO config, branding, etc.)
- `created_at` (TIMESTAMPTZ): Creation timestamp
- `updated_at` (TIMESTAMPTZ): Last modification timestamp
- `deleted_at` (TIMESTAMPTZ, NULLABLE): Soft delete timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_tenants_name ON tenants(name) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX idx_tenants_slug ON tenants(slug) WHERE deleted_at IS NULL;
CREATE INDEX idx_tenants_parent_id ON tenants(parent_id);
CREATE INDEX idx_tenants_status ON tenants(status);
```

**Constraints**:
- `slug` must match regex `^[a-z0-9-]+$`
- If `parent_id` is set, parent must have tier ≥ child's tier

**RLS**: No tenant scoping (tenants are global metadata)

---

### 3. Membership

**Purpose**: Association between user and tenant with assigned roles (enables multi-tenancy).

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `user_id` (UUID, FK → users.id): User reference
- `tenant_id` (UUID, FK → tenants.id): Tenant reference
- `status` (ENUM): Membership status ('active', 'invited', 'suspended')
- `invited_by` (UUID, FK → users.id, NULLABLE): User who sent invitation
- `invited_at` (TIMESTAMPTZ, NULLABLE): Invitation timestamp
- `accepted_at` (TIMESTAMPTZ, NULLABLE): Invitation acceptance timestamp
- `created_at` (TIMESTAMPTZ): Creation timestamp
- `updated_at` (TIMESTAMPTZ): Last modification timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_memberships_user_tenant ON memberships(user_id, tenant_id);
CREATE INDEX idx_memberships_tenant_id ON memberships(tenant_id);
CREATE INDEX idx_memberships_status ON memberships(status);
```

**Constraints**:
- Unique constraint on (user_id, tenant_id) - one membership per user per tenant
- If `status = 'invited'`, `invited_by` and `invited_at` MUST NOT be NULL

**RLS**:
```sql
CREATE POLICY memberships_tenant_isolation ON memberships
    FOR SELECT USING (tenant_id = get_current_tenant_id());
```

---

### 4. Role

**Purpose**: Named collection of entitlements, scoped to tenant or global.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `name` (VARCHAR(100)): Role name (e.g., "Document Reviewer", "Admin")
- `description` (TEXT, NULLABLE): Role description
- `tenant_id` (UUID, FK → tenants.id, NULLABLE): Tenant scope (NULL = global role)
- `is_system` (BOOLEAN): System-managed role (cannot be deleted by users)
- `created_at` (TIMESTAMPTZ): Creation timestamp
- `updated_at` (TIMESTAMPTZ): Last modification timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_roles_name_tenant ON roles(name, tenant_id);
CREATE INDEX idx_roles_tenant_id ON roles(tenant_id);
CREATE INDEX idx_roles_is_system ON roles(is_system);
```

**Constraints**:
- If `is_system = true`, role cannot be deleted or modified

**RLS**:
```sql
CREATE POLICY roles_tenant_isolation ON roles
    FOR SELECT USING (
        tenant_id = get_current_tenant_id() OR
        tenant_id IS NULL  -- Global roles visible to all
    );
```

**Predefined System Roles**:
- `system:super_admin` (tenant_id=NULL): Full system access
- `system:tenant_admin` (tenant_id=NULL): Full tenant access
- `system:user` (tenant_id=NULL): Basic user access

---

### 5. Entitlement

**Purpose**: Granular permission, format `action:resource` (e.g., "read:documents").

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `action` (VARCHAR(50)): Action verb (read, write, delete, manage)
- `resource` (VARCHAR(100)): Resource type (documents, users, settings)
- `scope` (JSONB, NULLABLE): Optional scope restrictions (e.g., `{"project_id": "123"}`)
- `description` (TEXT, NULLABLE): Human-readable description
- `created_at` (TIMESTAMPTZ): Creation timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_entitlements_action_resource ON entitlements(action, resource);
CREATE INDEX idx_entitlements_resource ON entitlements(resource);
```

**RLS**: No tenant scoping (entitlements are global definitions)

**Examples**:
```
read:documents
write:documents
delete:documents
manage:users
manage:tenants
read:audit_logs
manage:api_keys
```

---

### 6. MembershipRole (Junction Table)

**Purpose**: Many-to-many relationship between memberships and roles.

**Attributes**:
- `membership_id` (UUID, FK → memberships.id)
- `role_id` (UUID, FK → roles.id)
- `granted_at` (TIMESTAMPTZ): When role was granted
- `granted_by` (UUID, FK → users.id, NULLABLE): User who granted role

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_membership_roles_pk ON membership_roles(membership_id, role_id);
CREATE INDEX idx_membership_roles_role_id ON membership_roles(role_id);
```

**RLS**:
```sql
CREATE POLICY membership_roles_tenant_isolation ON membership_roles
    FOR SELECT USING (
        membership_id IN (
            SELECT id FROM memberships WHERE tenant_id = get_current_tenant_id()
        )
    );
```

---

### 7. RoleEntitlement (Junction Table)

**Purpose**: Many-to-many relationship between roles and entitlements.

**Attributes**:
- `role_id` (UUID, FK → roles.id)
- `entitlement_id` (UUID, FK → entitlements.id)
- `created_at` (TIMESTAMPTZ): When entitlement was added to role

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_role_entitlements_pk ON role_entitlements(role_id, entitlement_id);
CREATE INDEX idx_role_entitlements_entitlement_id ON role_entitlements(entitlement_id);
```

**RLS**: Follows role RLS policies

---

### 8. APIKey

**Purpose**: Programmatic access credential, scoped to tenant and user.

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `key_id` (VARCHAR(50), UNIQUE): Public key identifier (e.g., `umes_live_AbCd1234`)
- `key_hash` (VARCHAR(64)): SHA-256 hash of secret key
- `user_id` (UUID, FK → users.id): Owner user
- `tenant_id` (UUID, FK → tenants.id): Tenant scope
- `name` (VARCHAR(100)): User-provided key name
- `last_used_at` (TIMESTAMPTZ, NULLABLE): Last usage timestamp
- `expires_at` (TIMESTAMPTZ): Expiration timestamp
- `revoked_at` (TIMESTAMPTZ, NULLABLE): Revocation timestamp
- `created_at` (TIMESTAMPTZ): Creation timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_api_keys_key_id ON api_keys(key_id) WHERE revoked_at IS NULL;
CREATE INDEX idx_api_keys_user_tenant ON api_keys(user_id, tenant_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash) WHERE revoked_at IS NULL;
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at) WHERE revoked_at IS NULL;
```

**Constraints**:
- `key_id` format: `umes_{env}_{random}` where env is 'test' or 'live'
- `expires_at` must be > `created_at`
- Once revoked, `revoked_at` cannot be set back to NULL

**RLS**:
```sql
CREATE POLICY api_keys_tenant_isolation ON api_keys
    FOR SELECT USING (tenant_id = get_current_tenant_id());
```

**Key Format**:
- Prefix: `umes_live_` (production) or `umes_test_` (development)
- Public part (key_id): 8-character random (shown in UI)
- Secret part: 32-character random (shown once at creation, then hashed)
- Full key: `umes_live_AbCd1234.YourSecretKey32Characters...`

---

### 9. RefreshToken

**Purpose**: Long-lived opaque tokens for session refresh (database-backed for revocation).

**Attributes**:
- `id` (UUID, PK): Unique identifier
- `token_hash` (VARCHAR(64), UNIQUE): SHA-256 hash of opaque token
- `user_id` (UUID, FK → users.id): Owner user
- `tenant_id` (UUID, FK → tenants.id): Tenant context
- `session_id` (UUID): Session identifier (for grouped revocation)
- `device_fingerprint` (JSONB, NULLABLE): Device metadata (user agent, IP)
- `expires_at` (TIMESTAMPTZ): Expiration timestamp (7 days default)
- `revoked_at` (TIMESTAMPTZ, NULLABLE): Revocation timestamp
- `last_used_at` (TIMESTAMPTZ, NULLABLE): Last usage timestamp
- `created_at` (TIMESTAMPTZ): Creation timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash) WHERE revoked_at IS NULL;
CREATE INDEX idx_refresh_tokens_user_tenant ON refresh_tokens(user_id, tenant_id);
CREATE INDEX idx_refresh_tokens_session_id ON refresh_tokens(session_id);
CREATE INDEX idx_refresh_tokens_expires_at ON refresh_tokens(expires_at) WHERE revoked_at IS NULL;
```

**RLS**:
```sql
CREATE POLICY refresh_tokens_tenant_isolation ON refresh_tokens
    FOR SELECT USING (tenant_id = get_current_tenant_id());
```

---

### 10. AuditEvent

**Purpose**: Immutable append-only audit log with hash chain integrity (ALCOA+ compliance).

**Attributes**:
- `id` (BIGSERIAL, PK): Sequential identifier (partition key)
- `event_id` (UUID, UNIQUE): Globally unique event identifier
- `created_at` (TIMESTAMPTZ, NOT NULL, DEFAULT NOW()): Event timestamp
- `event_type` (VARCHAR(50)): Event type (e.g., 'auth.login', 'authz.permission_granted')
- `user_id` (UUID, FK → users.id, NULLABLE): Subject user (if applicable)
- `tenant_id` (UUID, FK → tenants.id, NULLABLE): Tenant context (if applicable)
- `session_id` (UUID, NULLABLE): Session identifier
- `api_key_id` (UUID, FK → api_keys.id, NULLABLE): API key used (if applicable)
- `actor_type` (ENUM): Actor type ('user', 'service', 'admin', 'system')
- `actor_id` (UUID): Actor identifier
- `client_ip` (INET, NULLABLE): Client IP address
- `user_agent` (TEXT, NULLABLE): Client user agent
- `request_id` (UUID, NULLABLE): Request correlation ID
- `event_data` (JSONB): Event-specific payload
- `previous_hash` (VARCHAR(64)): SHA-256 hash of previous entry (hash chain)
- `entry_hash` (VARCHAR(64)): SHA-256 hash of this entry
- `severity` (ENUM): Severity level ('debug', 'info', 'warn', 'error', 'critical')
- `result` (ENUM): Event result ('success', 'failure', 'denied')

**Partitioning Strategy**:
```sql
-- Monthly partitions for efficient archival
CREATE TABLE audit_events (
    -- ... columns above ...
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Example partitions (auto-created by pg_partman)
CREATE TABLE audit_events_2025_12 PARTITION OF audit_events
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
```

**Indexes**:
```sql
CREATE INDEX idx_audit_events_user_id ON audit_events(user_id, created_at DESC);
CREATE INDEX idx_audit_events_tenant_id ON audit_events(tenant_id, created_at DESC);
CREATE INDEX idx_audit_events_event_type ON audit_events(event_type, created_at DESC);
CREATE INDEX idx_audit_events_session_id ON audit_events(session_id);
CREATE INDEX idx_audit_events_event_id ON audit_events(event_id);
CREATE INDEX idx_audit_events_entry_hash ON audit_events(entry_hash);
CREATE INDEX idx_audit_events_event_data ON audit_events USING GIN(event_data);
```

**Immutability Trigger**:
```sql
CREATE FUNCTION prevent_audit_modification() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'Audit log entries cannot be updated (ALCOA+ immutability)';
    ELSIF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Audit log entries cannot be deleted (ALCOA+ immutability)';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_audit_immutability
BEFORE UPDATE OR DELETE ON audit_events
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();
```

**RLS**: No RLS (audit events are globally accessible, filtered by application logic)

---

## Database Schema (SQL)

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ENUMS
CREATE TYPE idp_type AS ENUM ('local', 'oidc', 'saml');
CREATE TYPE tenant_tier AS ENUM ('free', 'pro', 'enterprise');
CREATE TYPE tenant_status AS ENUM ('active', 'suspended', 'deleted');
CREATE TYPE membership_status AS ENUM ('active', 'invited', 'suspended');
CREATE TYPE actor_type AS ENUM ('user', 'service', 'admin', 'system');
CREATE TYPE event_severity AS ENUM ('debug', 'info', 'warn', 'error', 'critical');
CREATE TYPE event_result AS ENUM ('success', 'failure', 'denied');

-- Users (global, no RLS)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) NOT NULL,
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    name VARCHAR(255) NOT NULL,
    idp_type idp_type NOT NULL,
    idp_subject VARCHAR(512) NOT NULL,
    password_hash VARCHAR(255),
    mfa_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    mfa_secret TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    CONSTRAINT users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT users_local_password CHECK (
        (idp_type = 'local' AND password_hash IS NOT NULL) OR
        (idp_type != 'local' AND password_hash IS NULL)
    )
);

CREATE UNIQUE INDEX idx_users_email ON users(email) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX idx_users_idp_subject ON users(idp_type, idp_subject) WHERE deleted_at IS NULL;
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Tenants (global, no RLS)
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    tier tenant_tier NOT NULL DEFAULT 'free',
    parent_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    status tenant_status NOT NULL DEFAULT 'active',
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ,

    CONSTRAINT tenants_slug_format CHECK (slug ~* '^[a-z0-9-]+$')
);

CREATE UNIQUE INDEX idx_tenants_name ON tenants(name) WHERE deleted_at IS NULL;
CREATE UNIQUE INDEX idx_tenants_slug ON tenants(slug) WHERE deleted_at IS NULL;
CREATE INDEX idx_tenants_parent_id ON tenants(parent_id);
CREATE INDEX idx_tenants_status ON tenants(status);

-- Memberships (tenant-scoped via RLS)
CREATE TABLE memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    status membership_status NOT NULL DEFAULT 'active',
    invited_by UUID REFERENCES users(id) ON DELETE SET NULL,
    invited_at TIMESTAMPTZ,
    accepted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT memberships_invited_check CHECK (
        (status = 'invited' AND invited_by IS NOT NULL AND invited_at IS NOT NULL) OR
        (status != 'invited')
    )
);

CREATE UNIQUE INDEX idx_memberships_user_tenant ON memberships(user_id, tenant_id);
CREATE INDEX idx_memberships_tenant_id ON memberships(tenant_id);
CREATE INDEX idx_memberships_status ON memberships(status);

-- Enable RLS on memberships
ALTER TABLE memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE memberships FORCE ROW LEVEL SECURITY;

CREATE POLICY memberships_tenant_isolation_select ON memberships
    FOR SELECT USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::uuid);

-- Roles (global or tenant-scoped)
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    is_system BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_roles_name_tenant ON roles(name, COALESCE(tenant_id, '00000000-0000-0000-0000-000000000000'::uuid));
CREATE INDEX idx_roles_tenant_id ON roles(tenant_id);
CREATE INDEX idx_roles_is_system ON roles(is_system);

ALTER TABLE roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE roles FORCE ROW LEVEL SECURITY;

CREATE POLICY roles_tenant_isolation_select ON roles
    FOR SELECT USING (
        tenant_id = current_setting('app.current_tenant_id', TRUE)::uuid OR
        tenant_id IS NULL
    );

-- Entitlements (global, no RLS)
CREATE TABLE entitlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    action VARCHAR(50) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    scope JSONB,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_entitlements_action_resource ON entitlements(action, resource);
CREATE INDEX idx_entitlements_resource ON entitlements(resource);

-- MembershipRole junction table
CREATE TABLE membership_roles (
    membership_id UUID NOT NULL REFERENCES memberships(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL,

    PRIMARY KEY (membership_id, role_id)
);

CREATE INDEX idx_membership_roles_role_id ON membership_roles(role_id);

ALTER TABLE membership_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE membership_roles FORCE ROW LEVEL SECURITY;

CREATE POLICY membership_roles_tenant_isolation ON membership_roles
    FOR SELECT USING (
        membership_id IN (
            SELECT id FROM memberships WHERE tenant_id = current_setting('app.current_tenant_id', TRUE)::uuid
        )
    );

-- RoleEntitlement junction table
CREATE TABLE role_entitlements (
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    entitlement_id UUID NOT NULL REFERENCES entitlements(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (role_id, entitlement_id)
);

CREATE INDEX idx_role_entitlements_entitlement_id ON role_entitlements(entitlement_id);

-- API Keys (tenant-scoped via RLS)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_id VARCHAR(50) NOT NULL,
    key_hash VARCHAR(64) NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT api_keys_expires CHECK (expires_at > created_at),
    CONSTRAINT api_keys_key_id_format CHECK (key_id ~* '^umes_(test|live)_[A-Za-z0-9]{8}$')
);

CREATE UNIQUE INDEX idx_api_keys_key_id ON api_keys(key_id) WHERE revoked_at IS NULL;
CREATE INDEX idx_api_keys_user_tenant ON api_keys(user_id, tenant_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash) WHERE revoked_at IS NULL;
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at) WHERE revoked_at IS NULL;

ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys FORCE ROW LEVEL SECURITY;

CREATE POLICY api_keys_tenant_isolation ON api_keys
    FOR SELECT USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::uuid);

-- Refresh Tokens (tenant-scoped via RLS)
CREATE TABLE refresh_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_hash VARCHAR(64) NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    device_fingerprint JSONB,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_refresh_tokens_hash ON refresh_tokens(token_hash) WHERE revoked_at IS NULL;
CREATE INDEX idx_refresh_tokens_user_tenant ON refresh_tokens(user_id, tenant_id);
CREATE INDEX idx_refresh_tokens_session_id ON refresh_tokens(session_id);
CREATE INDEX idx_refresh_tokens_expires_at ON refresh_tokens(expires_at) WHERE revoked_at IS NULL;

ALTER TABLE refresh_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE refresh_tokens FORCE ROW LEVEL SECURITY;

CREATE POLICY refresh_tokens_tenant_isolation ON refresh_tokens
    FOR SELECT USING (tenant_id = current_setting('app.current_tenant_id', TRUE)::uuid);

-- Audit Events (partitioned, no RLS)
CREATE TABLE audit_events (
    id BIGSERIAL,
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    session_id UUID,
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    actor_type actor_type NOT NULL,
    actor_id UUID NOT NULL,
    client_ip INET,
    user_agent TEXT,
    request_id UUID,
    event_data JSONB NOT NULL,
    previous_hash VARCHAR(64) NOT NULL,
    entry_hash VARCHAR(64) NOT NULL,
    severity event_severity DEFAULT 'info',
    result event_result DEFAULT 'success',

    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Indexes (applied to all partitions)
CREATE INDEX idx_audit_events_user_id ON audit_events(user_id, created_at DESC);
CREATE INDEX idx_audit_events_tenant_id ON audit_events(tenant_id, created_at DESC);
CREATE INDEX idx_audit_events_event_type ON audit_events(event_type, created_at DESC);
CREATE INDEX idx_audit_events_session_id ON audit_events(session_id);
CREATE INDEX idx_audit_events_event_id ON audit_events(event_id);
CREATE INDEX idx_audit_events_entry_hash ON audit_events(entry_hash);
CREATE INDEX idx_audit_events_event_data ON audit_events USING GIN(event_data);

-- Immutability trigger
CREATE FUNCTION prevent_audit_modification() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'Audit log entries cannot be updated (ALCOA+ immutability)';
    ELSIF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'Audit log entries cannot be deleted (ALCOA+ immutability)';
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_audit_immutability
BEFORE UPDATE OR DELETE ON audit_events
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modification();

-- Initial partition (December 2025)
CREATE TABLE audit_events_2025_12 PARTITION OF audit_events
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
```

---

## State Transitions

### User Lifecycle
```
[Created] → [Email Verified] → [Active] → [Deleted]
                                    ↓
                              [MFA Enabled]
```

### Membership Lifecycle
```
[Invited] → [Accepted/Active] → [Suspended] → [Deleted]
```

### API Key Lifecycle
```
[Created/Active] → [Expired] (automatic)
                → [Revoked] (manual)
```

### Refresh Token Lifecycle
```
[Created/Active] → [Used] (last_used_at updated) → [Expired/Revoked]
```

---

## Relationships Summary

| Parent | Child | Cardinality | Cascade Behavior |
|--------|-------|-------------|------------------|
| User | Membership | 1:N | CASCADE (delete user → delete memberships) |
| Tenant | Membership | 1:N | CASCADE (delete tenant → delete memberships) |
| Membership | MembershipRole | 1:N | CASCADE |
| Role | MembershipRole | 1:N | CASCADE |
| Role | RoleEntitlement | 1:N | CASCADE |
| Entitlement | RoleEntitlement | 1:N | CASCADE |
| User | APIKey | 1:N | CASCADE |
| Tenant | APIKey | 1:N | CASCADE |
| User | RefreshToken | 1:N | CASCADE |
| Tenant | RefreshToken | 1:N | CASCADE |
| Tenant | Tenant (self) | 1:N | SET NULL (parent deletion) |

---

## Next Steps

With data model complete, proceed to:
1. Generate OpenAPI 3.1 contracts for all endpoints (contracts/ directory)
2. Generate quickstart.md for developer integration
3. Update agent context with technology stack
