# Feature Specification: Unified Management of Entitlements and Identity Subsystem (UMES)

**Feature Branch**: `001-umes-identity-subsystem`
**Created**: 2025-12-02
**Status**: Draft
**Input**: User description: "Design and implement a fully cloud-agnostic, multi-tenant Unified Management of Entitlements and Identity Subsystem (UMES) that acts as the canonical authority for user identity, authentication strategy, authorization decisions, API key lifecycle, and entitlement management across all Hextropian products."

## Problem Statement

**Current State:**
Hextropian products currently lack a unified system for managing user identity, authentication, and authorization. Each service potentially implements its own authentication logic, leading to:

- Duplicated identity and authorization code across services
- Inconsistent security models and practices
- Difficulty deploying in customer-controlled environments (on-prem, private clouds)
- Hard dependencies on specific cloud providers (AWS, GCP, Azure) for identity services
- Complex integration requirements for new services
- Security vulnerabilities from implementation inconsistencies
- Audit and compliance challenges from fragmented identity management

**Why This Matters:**
Enterprise customers require deployment flexibility, security consistency, and compliance guarantees. Without a unified identity system, Hextropian cannot:
- Support customer VPC deployments
- Provide consistent authorization across all products
- Meet enterprise security and audit requirements
- Scale to multi-tenant SaaS while supporting dedicated deployments
- Support B2B2B partnership models

## Business Value

- **Unified Identity & Authorization**: Eliminates duplicated logic across services, reduces engineering overhead, prevents security model divergence, accelerates new service development
- **Cloud Neutrality**: Prevents vendor lock-in by abstracting KMS and IdP dependencies, enables deployment on any cloud (GCP, AWS, Azure, Oracle, Hetzner, on-prem)
- **Portable Appliance**: Enables frictionless deployment into customer-controlled VPCs, satisfies data residency and sovereignty requirements, supports disconnected/offline environments
- **Future-Proofing for SaaS**: Provides foundation for multi-tenant SaaS, partner ecosystems (B2B2B), customer-facing flows (B2B2C)
- **Consistent Authorization**: Ensures identical authorization behavior regardless of product, region, or deployment model
- **Reduced Integration Time**: New services integrate once via stable APIs instead of reimplementing authentication/authorization
- **Operational Simplicity**: Standardizes secrets, API keys, tokens, and audit logging across all environments
- **Compliance & Auditability**: Centralized authority for identity actions simplifies audits and regulatory reviews
- **Customer Trust**: Security model isolated, tenant-aware, deployable within customer perimeter without cloud-vendor dependencies

## User Scenarios & Testing

### User Story 1 - Service Administrator Configures UMES for Customer Deployment (Priority: P1)

A service administrator needs to deploy UMES into a customer's Oracle Cloud environment and configure it to use the customer's existing Okta identity provider and Oracle KMS for key management, without modifying any code.

**Why this priority**: Core value proposition - cloud-agnostic deployment. If UMES cannot deploy identically across all clouds, the feature fails its primary objective.

**Independent Test**: Deploy UMES to Oracle Cloud, configure Okta IdP and Oracle KMS via environment variables, verify authentication and token signing work identically to GCP/AWS deployments.

**Acceptance Scenarios**:

1. **Given** UMES container image, **When** administrator deploys to Oracle Cloud and sets environment variables for Okta IdP and Oracle KMS, **Then** UMES starts successfully and uses configured adapters
2. **Given** configured UMES in Oracle Cloud, **When** user authenticates via Okta, **Then** UMES issues JWT signed by Oracle KMS
3. **Given** UMES deployed in Oracle vs GCP, **When** same user authenticates in both, **Then** authorization decisions are identical
4. **Given** UMES running in on-prem environment, **When** administrator configures local IdP and OpenBao, **Then** UMES operates without any cloud dependencies

---

### User Story 2 - End User Authenticates Across Multiple Hextropian Products (Priority: P1)

An end user needs to authenticate once and access multiple Hextropian products (RKC, NLF, LCE, HexGraph) without re-authenticating, with consistent permissions across all products.

**Why this priority**: Core functionality - users cannot use the system without authentication. Single sign-on is essential for user experience.

**Independent Test**: User logs in once, receives JWT token, uses token to access multiple Hextropian services, permissions evaluated consistently by each service.

**Acceptance Scenarios**:

1. **Given** user with valid credentials, **When** user authenticates to UMES, **Then** UMES issues JWT with user identity, tenant context, and entitlements
2. **Given** JWT from UMES, **When** user accesses RKC, NLF, LCE, or HexGraph, **Then** each service validates token via UMES introspection API
3. **Given** user with multiple tenant memberships, **When** user switches tenant context, **Then** UMES issues new token with updated tenant context and role-specific entitlements
4. **Given** user session expires, **When** user attempts to access service, **Then** service rejects request and redirects to UMES for re-authentication

---

### User Story 3 - System Administrator Manages User Roles and Entitlements (Priority: P1)

A system administrator needs to define roles (e.g., "Document Reviewer", "Compliance Officer", "System Admin"), assign capabilities to roles (e.g., "read:documents", "write:documents", "manage:users"), and assign users to roles within specific tenants.

**Why this priority**: Authorization is core UMES functionality. Without role/entitlement management, UMES cannot enforce access control.

**Independent Test**: Administrator creates roles, assigns entitlements to roles, assigns users to roles, verifies permissions are enforced correctly by services calling UMES authorization API.

**Acceptance Scenarios**:

1. **Given** administrator authenticated to UMES, **When** administrator creates role "Document Reviewer" with entitlements ["read:documents", "export:documents"], **Then** role is persisted and available for assignment
2. **Given** role "Document Reviewer", **When** administrator assigns user to role in Tenant A, **Then** user receives entitlements only within Tenant A (tenant-scoped)
3. **Given** user with role "Document Reviewer" in Tenant A, **When** service calls UMES authorization API for "write:documents" permission, **Then** UMES denies authorization
4. **Given** user with role "Document Reviewer" in Tenant A, **When** user attempts to access Tenant B resources, **Then** UMES denies authorization (tenant isolation enforced)

---

### User Story 4 - Service Developer Integrates New Hextropian Service with UMES (Priority: P2)

A developer building a new Hextropian service needs to integrate authentication and authorization with UMES using a simple SDK/library, without implementing custom security logic.

**Why this priority**: Enables UMES adoption across all services. If integration is complex, services won't use UMES, defeating the unified identity objective.

**Independent Test**: Developer adds UMES client library to new service, configures UMES endpoint, protects endpoints with authentication middleware, verifies tokens and permissions via UMES APIs.

**Acceptance Scenarios**:

1. **Given** new Hextropian service, **When** developer adds UMES client library and configures UMES endpoint, **Then** library validates JWT tokens on every request
2. **Given** protected endpoint requiring "write:analytics" permission, **When** request arrives with valid JWT, **Then** UMES client library calls authorization API, grants/denies access based on user entitlements
3. **Given** service calling UMES APIs, **When** UMES is unavailable, **Then** service fails closed (denies access) and logs error with correlation ID
4. **Given** service needs to propagate user context, **When** service makes downstream calls to other services, **Then** service includes JWT token in Authorization header for consistent identity propagation

---

### User Story 5 - System Generates and Manages API Keys for Programmatic Access (Priority: P2)

A user or service needs to generate API keys for programmatic access to Hextropian services, with keys scoped to specific tenants and entitlements, supporting rotation and revocation.

**Why this priority**: API keys enable automation and integration. Essential for enterprise use cases but not blocking for initial user-facing authentication.

**Independent Test**: User generates API key via UMES, uses key to access services, rotates key, revokes key, verifies old key no longer works.

**Acceptance Scenarios**:

1. **Given** authenticated user, **When** user requests API key creation for Tenant A with entitlements ["read:documents"], **Then** UMES generates key, returns plaintext once, stores hashed version
2. **Given** API key for Tenant A, **When** service receives request with API key, **Then** UMES validates key, returns user context and entitlements for authorization
3. **Given** API key approaching expiration, **When** user rotates key, **Then** UMES generates new key, marks old key for expiration after grace period (e.g., 7 days)
4. **Given** compromised API key, **When** administrator revokes key, **Then** UMES immediately rejects all requests with that key
5. **Given** revoked API key, **When** audit query runs, **Then** revocation event is logged with timestamp, user ID, tenant ID, and reason

---

### User Story 6 - Compliance Officer Audits Identity and Access Events (Priority: P2)

A compliance officer needs to query all identity-related events (logins, permission changes, API key usage, token issuance) for a specific tenant or user, with complete audit trail including timestamps, actors, and outcomes.

**Why this priority**: Compliance and auditability are key business value. Essential for enterprise customers but not blocking for initial development.

**Independent Test**: Compliance officer queries audit logs for specific tenant/user, verifies all identity events are captured with complete metadata, exports logs for regulatory review.

**Acceptance Scenarios**:

1. **Given** user authentication event, **When** UMES authenticates user, **Then** event is logged with timestamp, user ID, tenant ID, authentication method, success/failure, IP address
2. **Given** permission change event, **When** administrator modifies user roles, **Then** event is logged with timestamp, administrator ID, affected user ID, old roles, new roles
3. **Given** API key creation event, **When** user generates API key, **Then** event is logged with timestamp, user ID, tenant ID, key ID (not plaintext), assigned entitlements
4. **Given** 90 days of audit logs, **When** compliance officer queries for Tenant A events, **Then** UMES returns all identity events for that tenant in chronological order
5. **Given** audit log export request, **When** compliance officer exports logs, **Then** UMES provides logs in machine-readable format (JSON) with integrity verification (hash chain)

---

### User Story 7 - Multi-Tenant User Switches Between Organizations (Priority: P3)

A user who is a member of multiple tenants (e.g., employee of Partner Org A and contractor for Partner Org B) needs to switch between tenant contexts and receive appropriate permissions for each context.

**Why this priority**: Supports B2B2B partnership models. Important for future growth but not blocking for initial deployment.

**Independent Test**: User with memberships in Tenant A (role: Admin) and Tenant B (role: Viewer) switches contexts, verifies different permissions in each tenant.

**Acceptance Scenarios**:

1. **Given** user with memberships in Tenant A and Tenant B, **When** user authenticates, **Then** UMES returns list of available tenants
2. **Given** user selects Tenant A, **When** UMES issues token, **Then** token includes Tenant A context and user's Tenant A roles/entitlements
3. **Given** user with Tenant A token, **When** user switches to Tenant B, **Then** UMES issues new token with Tenant B context, invalidates old token
4. **Given** user accessing service with Tenant A token, **When** user attempts to access Tenant B resources, **Then** service denies access (tenant isolation enforced)

---

### Edge Cases

- **What happens when KMS provider is unavailable?** UMES falls back to emergency local crypto mode (alerts triggered), or fails closed if emergency mode disabled. All token operations logged with degraded mode indicator.
- **What happens when IdP provider is unavailable?** UMES uses cached tokens (max 5 minutes), then fails closed. Users cannot authenticate but existing valid tokens continue to work within cache window.
- **How does system handle token revocation for distributed services?** UMES maintains revocation list, services check revocation on token introspection. Services with cached tokens check revocation within 2-minute cache TTL.
- **What happens when user is deleted mid-session?** Next token validation fails, service terminates session. Background job revokes all tokens for deleted user within 60 seconds.
- **How does system handle tenant deletion?** All users in tenant are disabled, tokens revoked, access denied within 60 seconds. Audit trail preserved for compliance.
- **What happens during UMES upgrades?** Rolling deployment with zero downtime. Old and new versions coexist temporarily, token format remains backward-compatible.

## Requirements

### Functional Requirements

- **FR-001**: System MUST support authentication via Local (username/password), OIDC (Auth0/Okta/Keycloak/AAD/Cloudflare), and SAML identity providers
- **FR-002**: System MUST support key management via Local (dev), GCP KMS, AWS KMS, Azure Key Vault, Oracle KMS, and OpenBao (on-prem) adapters
- **FR-003**: System MUST issue JWT access tokens signed by configured KMS provider
- **FR-004**: System MUST issue ID tokens following OIDC specification
- **FR-005**: System MUST support opaque tokens stored in database for revocable sessions
- **FR-006**: System MUST provide token introspection API returning user identity, tenant context, roles, and entitlements
- **FR-007**: System MUST enforce tenant isolation - users can only access resources within authorized tenants
- **FR-008**: System MUST support multi-tenant users with membership in multiple organizations
- **FR-009**: System MUST support B2B2B hierarchy (parent org → partner org → user)
- **FR-010**: System MUST provide capability-based authorization (e.g., "read:documents", "write:compliance_reports")
- **FR-011**: System MUST provide role-based authorization with roles assigned per tenant
- **FR-012**: System MUST support entitlement sets (collections of capabilities assigned to roles)
- **FR-013**: System MUST provide API key management (create, rotate, revoke, list)
- **FR-014**: System MUST hash API keys for storage (plaintext shown only once at creation)
- **FR-015**: System MUST encrypt API keys at rest using configured KMS provider
- **FR-016**: System MUST log all authentication events (success, failure, method, IP)
- **FR-017**: System MUST log all authorization decisions (granted, denied, resource, action)
- **FR-018**: System MUST log all entitlement changes (role assignments, permission modifications)
- **FR-019**: System MUST log all API key lifecycle events (create, rotate, revoke, usage)
- **FR-020**: System MUST provide audit log export in machine-readable format (JSON)
- **FR-021**: System MUST support adapter configuration via environment variables (no code changes)
- **FR-022**: System MUST validate that KMS and IdP adapters are swappable with only configuration changes
- **FR-023**: System MUST deploy identically on GCP, AWS, Azure, Oracle, Hetzner, and on-prem
- **FR-024**: System MUST provide health check endpoints (`/health`, `/ready`, `/live`)
- **FR-025**: System MUST rate limit authentication attempts (10 req/min per IP for unauthenticated, 1000 req/min for authenticated)
- **FR-026**: System MUST enforce token expiration (15 minutes for access tokens, 7 days for refresh tokens)
- **FR-027**: System MUST support token refresh without re-authentication
- **FR-028**: System MUST validate all inputs using schema validation
- **FR-029**: System MUST use parameterized queries for all database operations (prevent SQL injection)
- **FR-030**: System MUST enforce Row-Level Security in database for tenant isolation

### Key Entities

- **User**: Individual person with credentials, can be member of multiple tenants, has attributes (email, OIDC subject, SAML name ID)
- **Tenant**: Organization/customer instance, isolated data boundary, has tier level for access control
- **Membership**: Association between user and tenant with specific role(s)
- **Role**: Named collection of entitlements, scoped to tenant or global, defines what user can do
- **Entitlement**: Granular permission, format: `action:resource` (e.g., "read:documents"), assigned to roles
- **Capability Set**: Named, versioned collection of entitlements, enables bulk permission management
- **API Key**: Programmatic access credential, scoped to tenant and user, has expiration and revocation status
- **Partner Organization**: Entity in B2B2B hierarchy, can have parent organization and child users
- **Workspace**: Optional sub-division within tenant for finer-grained access control
- **Audit Event**: Immutable record of identity/authorization action, includes timestamp, actor, resource, outcome, metadata

## Success Criteria

### Measurable Outcomes

- **SC-001**: Services can validate tokens in under 50ms (p95 latency) for 10,000 concurrent requests
- **SC-002**: Users can authenticate via any supported IdP provider (Local, OIDC, SAML) and receive valid token
- **SC-003**: Administrator can swap KMS provider (e.g., AWS KMS → Azure Key Vault) by changing configuration only, with zero code changes
- **SC-004**: Administrator can deploy UMES to any target environment (GCP, AWS, Azure, Oracle, Hetzner, on-prem) and integration tests pass identically
- **SC-005**: Service developer can integrate authentication in under 1 hour using provided SDK/library
- **SC-006**: All identity events are captured in audit log with 100% coverage (no missed events)
- **SC-007**: Authorization decisions are consistent across all deployment environments (verified by integration tests)
- **SC-008**: Token generation completes in under 200ms (p95 latency) including KMS signing operation
- **SC-009**: System maintains 99.9% uptime with graceful degradation when dependencies fail
- **SC-010**: Tenant isolation is verified with integration tests (no cross-tenant data leakage)
- **SC-011**: API key rotation completes without service disruption (zero downtime)
- **SC-012**: Audit logs are exportable in under 5 seconds for 90 days of tenant events

## Integration Tests

- **IT-001**: End-to-end authentication flow → User provides credentials to any IdP adapter → UMES authenticates → KMS signs JWT → Token validated by service → User accesses resource
- **IT-002**: Multi-cloud adapter isolation → Deploy UMES with GCP KMS → Authenticate user → Sign token → Deploy UMES with AWS KMS → Authenticate same user → Sign token → Verify tokens functionally equivalent
- **IT-003**: Tenant isolation enforcement → User in Tenant A attempts to access Tenant B resource → UMES authorization API denies → Service rejects request
- **IT-004**: Authorization decision consistency → Same user, same role, same resource across GCP/AWS/Azure deployments → Authorization API returns identical decision
- **IT-005**: API key lifecycle → User creates API key → Service validates key → User rotates key → Old key expires after grace period → Revoked key fails validation immediately
- **IT-006**: Token revocation propagation → Administrator revokes user token → Within 60 seconds, all services reject token → User must re-authenticate
- **IT-007**: Adapter swapping validation → Start with LocalIdP + LocalKMS → Authenticate user → Swap to OIDCIdP + GCP KMS → Authenticate user → Verify consistent behavior
- **IT-008**: Audit trail completeness → Perform authentication, authorization, API key creation → Query audit logs → Verify all events captured with correct metadata
- **IT-009**: Graceful degradation → Simulate KMS unavailability → UMES falls back to emergency mode → Logs degraded operation → Services continue with cached tokens
- **IT-010**: Multi-tenant user context switching → User with memberships in Tenant A and B → User switches context → Token reflects new tenant → Access to previous tenant denied

## Acceptance Criteria

1. UMES authenticates and authorizes users correctly using any configured IdP adapter (Local, OIDC, SAML)
2. Token issuance and verification succeed via any configured KMS adapter (Local, GCP, AWS, Azure, Oracle, OpenBao)
3. Swapping KMS or IdP provider requires ONLY configuration changes (no code modifications)
4. Services integrate once via stable API; no service requires cloud-specific IAM or cryptography logic
5. Identity context and entitlement resolution are consistent across all deployment environments
6. Tokens carry tenant context; entitlement evaluation matches defined policy
7. Users with multiple memberships can switch contexts without ambiguity
8. All API key operations are cryptographically sound, tenant-scoped, and audited
9. API key rotation and revocation are enforced globally and immediately
10. All identity and entitlement changes are logged with complete metadata
11. Audit logs are consistent and exportable regardless of deployment environment
12. No plaintext secrets stored anywhere in system
13. Strong hashing (Argon2/bcrypt) applied to all passwords
14. KMS keys rotate according to policy
15. Token revocation works for both JWT and opaque tokens
16. UMES deploys cleanly as container bundle, Helm chart, Docker Compose, and VM package
17. All deployments pass end-to-end integration test suite
18. Integration tests validate identical behavior across GCP, AWS, Azure, Oracle, Hetzner, and on-prem deployments
