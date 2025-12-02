# Implementation Tasks: UMES Identity Subsystem

**Feature Branch**: `001-umes-identity-subsystem`
**Created**: 2025-12-02
**Status**: Ready for Implementation
**Test Coverage Requirement**: 100% (ALL code is security-critical per constitution v1.0.1)

---

## Task Format

```
- [ ] [TaskID] [P?] [Story?] Description with file path
```

- **TaskID**: Unique identifier (T001, T002, etc.)
- **P?**: Priority (P1=MVP, P2=Important, P3=Nice-to-have)
- **[P]**: Parallel execution opportunity (can run simultaneously with previous task)
- **Story?**: User story reference (US1-US7) or FOUNDATION/SETUP/POLISH

---

## Progress Tracking

- **Started**: 2025-12-02
- **Last Updated**: 2025-12-02
- **Completed Tasks**: 0 / 288
- **Current Phase**: Not started
- **Estimated Velocity**: ~3-5 tasks/hour (varies by complexity)

---

## Phase 1: Project Setup & Infrastructure (12 tasks)

**Goal**: Initialize project structure, configure dependencies, establish testing framework.

**Independent Test**: Project builds successfully, all dependencies installed, test framework runs.

### Setup Tasks

- [ ] [T001] [P1] [SETUP] Initialize backend project structure - Create directory layout per plan.md: `backend/src/umes/{adapters,models,schemas,services,api,middleware,utils}`, `backend/tests/{unit,integration,smoke,contract,security,performance}`
- [ ] [T002] [P1] [SETUP] [P] Create pyproject.toml with all dependencies - Include FastAPI 0.104+, SQLAlchemy 2.0, asyncpg, aioboto3, google-cloud-kms, azure-keyvault-keys, oci, async-hvac, Authlib 1.6+, python3-saml 1.16+, argon2-cffi 25.1+, Purgatory 3.0+, Tenacity, aiocache, pytest 7.4+, Testcontainers per research.md
- [ ] [T003] [P1] [SETUP] Configure pytest with asyncio support - Create `pytest.ini` with asyncio_mode=auto, configure coverage reporting for 100% target in `backend/tests/pytest.ini`
- [ ] [T004] [P1] [SETUP] Set up Testcontainers fixtures for PostgreSQL, Redis, Keycloak - Create `backend/tests/conftest.py` with session-scoped containers per research.md section 5
- [ ] [T005] [P1] [SETUP] [P] Create Docker Compose for local development - PostgreSQL 15, Redis 7, Keycloak 23, LocalStack (for AWS KMS emulation) in `backend/docker-compose.yml`
- [ ] [T006] [P1] [SETUP] Configure SQLAlchemy async engine with connection pooling - Create `backend/src/umes/database.py` with asyncpg driver, pool_size=20, max_overflow=10 per research.md section 6
- [ ] [T007] [P1] [SETUP] Implement RLS context manager with ContextVar - Create `backend/src/umes/middleware/tenant_context.py` with `current_tenant_id` ContextVar and SQLAlchemy `after_begin` event listener per research.md section 3
- [ ] [T008] [P1] [SETUP] Create base SQLAlchemy models with RLS support - Create `backend/src/umes/models/base.py` with Base declarative class, updated_at trigger, soft delete mixin
- [ ] [T009] [P1] [SETUP] Configure two-level caching (L1 in-memory + L2 Redis) - Create `backend/src/umes/utils/cache.py` with `TieredCache` class per research.md section 6
- [ ] [T010] [P1] [SETUP] Implement circuit breaker factory with Purgatory - Create `backend/src/umes/utils/circuit_breaker.py` with `AsyncCircuitBreakerFactory` for KMS and IdP adapters per research.md section 4
- [ ] [T011] [P1] [SETUP] Create configuration management with Pydantic settings - Create `backend/src/umes/config.py` with environment-based config (KMS_PROVIDER, IDP_TYPE, DATABASE_URL, REDIS_URL)
- [ ] [T012] [P1] [SETUP] Write smoke test for project setup - Verify imports work, database connects, Redis connects, config loads in `backend/tests/smoke/test_setup.py`

---

## Phase 2: Foundational Components (48 tasks)

**Goal**: Implement core adapters (KMS, IdP), database models, audit logging, and token infrastructure.

**Independent Test**: All adapters pass contract tests, database schema deployed with RLS verified, audit logging operational.

### Database Models (10 tasks)

- [ ] [T013] [P1] [FOUNDATION] Write unit tests for User model - Test constraints (email format, idp_type/password_hash validation), soft deletes, indexes in `backend/tests/unit/models/test_user.py`
- [ ] [T014] [P1] [FOUNDATION] Implement User model - Complete SQLAlchemy model with all attributes from data-model.md in `backend/src/umes/models/user.py`
- [ ] [T015] [P1] [FOUNDATION] [P] Write unit tests for Tenant model - Test slug validation, B2B2B hierarchy, status transitions in `backend/tests/unit/models/test_tenant.py`
- [ ] [T016] [P1] [FOUNDATION] [P] Implement Tenant model - Complete model with parent_id foreign key, settings JSONB in `backend/src/umes/models/tenant.py`
- [ ] [T017] [P1] [FOUNDATION] Write unit tests for Membership model - Test unique constraint (user_id, tenant_id), status transitions, RLS policies in `backend/tests/unit/models/test_membership.py`
- [ ] [T018] [P1] [FOUNDATION] Implement Membership model - Model with user/tenant foreign keys, status enum in `backend/src/umes/models/membership.py`
- [ ] [T019] [P1] [FOUNDATION] [P] Write unit tests for Role, Entitlement, junction tables - Test role/entitlement relationships, RLS policies in `backend/tests/unit/models/test_authorization.py`
- [ ] [T020] [P1] [FOUNDATION] [P] Implement Role, Entitlement, MembershipRole, RoleEntitlement models - Complete authorization models per data-model.md in `backend/src/umes/models/authorization.py`
- [ ] [T021] [P1] [FOUNDATION] Write unit tests for APIKey, RefreshToken models - Test key format validation, expiration, revocation in `backend/tests/unit/models/test_tokens.py`
- [ ] [T022] [P1] [FOUNDATION] Implement APIKey and RefreshToken models - Models with hash storage, expiration, RLS in `backend/src/umes/models/tokens.py`

### Audit Logging (6 tasks)

- [ ] [T023] [P1] [FOUNDATION] Write unit tests for AuditEvent model - Test hash chain calculation, immutability trigger, partitioning in `backend/tests/unit/models/test_audit.py`
- [ ] [T024] [P1] [FOUNDATION] Implement AuditEvent model with hash chain - Model with partitioning, immutability triggers per research.md section 7 in `backend/src/umes/models/audit.py`
- [ ] [T025] [P1] [FOUNDATION] Write unit tests for audit logger service - Test event creation, hash chain integrity, concurrent writes in `backend/tests/unit/services/test_audit_logger.py`
- [ ] [T026] [P1] [FOUNDATION] Implement audit logger service - Async service with hash chain calculation, event insertion in `backend/src/umes/services/audit_logger.py`
- [ ] [T027] [P1] [FOUNDATION] Write integration tests for audit log verification - Test chain verification, tamper detection in `backend/tests/integration/test_audit_verification.py`
- [ ] [T028] [P1] [FOUNDATION] Implement audit log verification functions - Hash chain verification, integrity checks in `backend/src/umes/services/audit_verifier.py`

### Database Migrations (3 tasks)

- [ ] [T029] [P1] [FOUNDATION] Set up Alembic for migrations - Initialize Alembic, configure async support in `backend/alembic/`
- [ ] [T030] [P1] [FOUNDATION] Create initial migration for all models - Generate migration with all tables, indexes, RLS policies, triggers from data-model.md in `backend/alembic/versions/001_initial_schema.py`
- [ ] [T031] [P1] [FOUNDATION] Write integration test for RLS enforcement - Verify tenant isolation works, cross-tenant queries fail in `backend/tests/integration/test_rls_enforcement.py`

### KMS Adapters (12 tasks)

- [ ] [T032] [P1] [FOUNDATION] Write contract tests for KMS adapter protocol - Test sign/verify/encrypt/decrypt/get_public_key interface in `backend/tests/contract/test_kms_adapter.py`
- [ ] [T033] [P1] [FOUNDATION] Define KMS adapter protocol - Create Protocol class per research.md section 1 in `backend/src/umes/adapters/kms/protocol.py`
- [ ] [T034] [P1] [FOUNDATION] [P] Write unit tests for LocalKMS adapter - Test in-memory crypto operations (RS256, ES256) in `backend/tests/unit/adapters/kms/test_local_kms.py`
- [ ] [T035] [P1] [FOUNDATION] [P] Implement LocalKMS adapter - Development-only adapter with cryptography library in `backend/src/umes/adapters/kms/local.py`
- [ ] [T036] [P1] [FOUNDATION] Write contract tests for AWS KMS adapter - Test with LocalStack emulator in `backend/tests/contract/test_aws_kms.py`
- [ ] [T037] [P1] [FOUNDATION] Implement AWS KMS adapter - Use aioboto3 per research.md in `backend/src/umes/adapters/kms/aws.py`
- [ ] [T038] [P1] [FOUNDATION] [P] Write contract tests for GCP KMS adapter - Test with real GCP account (critical path) in `backend/tests/contract/test_gcp_kms.py`
- [ ] [T039] [P1] [FOUNDATION] [P] Implement GCP KMS adapter - Use google-cloud-kms async client in `backend/src/umes/adapters/kms/gcp.py`
- [ ] [T040] [P1] [FOUNDATION] Write contract tests for Azure Key Vault adapter - Test with Azurite + real Key Vault in `backend/tests/contract/test_azure_kms.py`
- [ ] [T041] [P1] [FOUNDATION] Implement Azure Key Vault adapter - Use azure-keyvault-keys async in `backend/src/umes/adapters/kms/azure.py`
- [ ] [T042] [P1] [FOUNDATION] [P] Write contract tests for Oracle KMS and OpenBao adapters - Mock Oracle, test OpenBao with Docker Vault in `backend/tests/contract/test_oracle_openbao_kms.py`
- [ ] [T043] [P1] [FOUNDATION] [P] Implement Oracle KMS and OpenBao adapters - Oracle with oci library, OpenBao with async-hvac in `backend/src/umes/adapters/kms/oracle.py` and `backend/src/umes/adapters/kms/openbao.py`
- [ ] [T044] [P1] [FOUNDATION] Create KMS adapter factory - Factory to instantiate adapter based on config in `backend/src/umes/adapters/kms/factory.py`

### IdP Adapters (12 tasks)

- [ ] [T045] [P1] [FOUNDATION] Write contract tests for IdP adapter protocol - Test authenticate/validate_token/refresh_token interface in `backend/tests/contract/test_idp_adapter.py`
- [ ] [T046] [P1] [FOUNDATION] Define IdP adapter protocol - Create Protocol class per research.md section 2 in `backend/src/umes/adapters/idp/protocol.py`
- [ ] [T047] [P1] [FOUNDATION] Write unit tests for LocalIdP adapter - Test password hashing (Argon2id), verification, timing in `backend/tests/unit/adapters/idp/test_local_idp.py`
- [ ] [T048] [P1] [FOUNDATION] Implement LocalIdP adapter - Argon2id password hashing with RFC_9106_LOW_MEMORY params in `backend/src/umes/adapters/idp/local.py`
- [ ] [T049] [P1] [FOUNDATION] [P] Write contract tests for OIDC adapter - Test with Keycloak Testcontainer, verify discovery, PKCE in `backend/tests/contract/test_oidc_adapter.py`
- [ ] [T050] [P1] [FOUNDATION] [P] Implement OIDC adapter - Use Authlib AsyncOAuth2Client per research.md in `backend/src/umes/adapters/idp/oidc.py`
- [ ] [T051] [P1] [FOUNDATION] Write contract tests for SAML adapter - Test assertion validation, signature verification in `backend/tests/contract/test_saml_adapter.py`
- [ ] [T052] [P1] [FOUNDATION] Implement SAML adapter - Use python3-saml with asyncio.to_thread in `backend/src/umes/adapters/idp/saml.py`
- [ ] [T053] [P1] [FOUNDATION] Create IdP adapter factory - Factory to instantiate based on config in `backend/src/umes/adapters/idp/factory.py`

### Token Services (5 tasks)

- [ ] [T054] [P1] [FOUNDATION] Write unit tests for JWT service - Test token creation, signing (mocked KMS), claims validation in `backend/tests/unit/services/test_jwt_service.py`
- [ ] [T055] [P1] [FOUNDATION] Implement JWT service - Token creation with KMS adapter, PyJWT for structure in `backend/src/umes/services/jwt_service.py`
- [ ] [T056] [P1] [FOUNDATION] Write unit tests for token validation service - Test signature verification, expiration, revocation checks in `backend/tests/unit/services/test_token_validator.py`
- [ ] [T057] [P1] [FOUNDATION] Implement token validation service - Verify with KMS, check Redis revocation list per research.md section 8 in `backend/src/umes/services/token_validator.py`
- [ ] [T058] [P1] [FOUNDATION] Write integration tests for token lifecycle - Test create → validate → revoke → validate fails in `backend/tests/integration/test_token_lifecycle.py`
- [ ] [T059] [P1] [FOUNDATION] Implement token revocation service - Redis revocation list, database revocation for refresh tokens in `backend/src/umes/services/token_revoker.py`
- [ ] [T060] [P1] [FOUNDATION] Write performance tests for token operations - Verify <50ms validation, <200ms generation per success criteria SC-001, SC-008 in `backend/tests/performance/test_token_performance.py`

---

## Phase 3: User Story 1 - Service Administrator Configures UMES (Priority P1) (28 tasks)

**User Story**: Service administrator deploys UMES to Oracle Cloud, configures Okta IdP and Oracle KMS via environment variables.

**Acceptance Criteria**:
1. UMES container deploys to Oracle Cloud with env vars
2. Authenticates via Okta, issues JWT signed by Oracle KMS
3. Authorization decisions identical across Oracle/GCP deployments
4. On-prem deployment works with local IdP and OpenBao

**Independent Test**: Deploy UMES with different KMS/IdP configs, verify identical behavior via integration tests.

### Configuration & Adapter Selection (8 tasks)

- [ ] [T061] [P1] [US1] Write unit tests for configuration validator - Test env var parsing, adapter selection logic in `backend/tests/unit/test_config_validator.py`
- [ ] [T062] [P1] [US1] Implement configuration validator - Validate KMS_PROVIDER, IDP_TYPE, required secrets in `backend/src/umes/config_validator.py`
- [ ] [T063] [P1] [US1] Write integration tests for adapter factory with different configs - Test LocalKMS+LocalIdP, GCPKMS+OIDC, etc. in `backend/tests/integration/test_adapter_factory.py`
- [ ] [T064] [P1] [US1] Implement adapter initialization service - Service to create KMS and IdP adapters from config in `backend/src/umes/services/adapter_initializer.py`
- [ ] [T065] [P1] [US1] Write unit tests for health check endpoints - Test /health, /ready, /live with adapter status in `backend/tests/unit/api/test_health.py`
- [ ] [T066] [P1] [US1] Implement health check endpoints - FastAPI routes checking DB, Redis, adapter connectivity in `backend/src/umes/api/health.py`
- [ ] [T067] [P1] [US1] Write smoke tests for multi-cloud deployment scenarios - Test AWS+OIDC, GCP+SAML, Oracle+Okta, on-prem+OpenBao in `backend/tests/smoke/test_multi_cloud_deployment.py`
- [ ] [T068] [P1] [US1] Create Docker entrypoint script - Script to validate config, initialize adapters, run migrations in `backend/docker-entrypoint.sh`

### Container Build & Deployment (6 tasks)

- [ ] [T069] [P1] [US1] Write Dockerfile for UMES backend - Multi-stage build, Python 3.11+, all dependencies in `backend/Dockerfile`
- [ ] [T070] [P1] [US1] Create Kubernetes manifests - Deployment, Service, ConfigMap, Secret templates in `backend/k8s/`
- [ ] [T071] [P1] [US1] [P] Create Helm chart - Chart with values for KMS/IdP config, resource limits in `backend/helm/umes/`
- [ ] [T072] [P1] [US1] [P] Write Docker Compose for production-like local testing - PostgreSQL, Redis, UMES with env vars in `backend/docker-compose.prod.yml`
- [ ] [T073] [P1] [US1] Write integration tests for container startup - Test container starts, migrations run, health checks pass in `backend/tests/integration/test_container_startup.py`
- [ ] [T074] [P1] [US1] Create deployment documentation - Document env vars, secrets, cloud-specific setup in `backend/docs/deployment.md`

### Cross-Cloud Consistency Validation (8 tasks)

- [ ] [T075] [P1] [US1] Write integration tests for GCP deployment - Deploy with GCP KMS + OIDC, verify auth flow in `backend/tests/integration/cloud/test_gcp_deployment.py`
- [ ] [T076] [P1] [US1] Write integration tests for AWS deployment - Deploy with AWS KMS + SAML, verify auth flow in `backend/tests/integration/cloud/test_aws_deployment.py`
- [ ] [T077] [P1] [US1] Write integration tests for Azure deployment - Deploy with Azure Key Vault + AAD OIDC, verify auth flow in `backend/tests/integration/cloud/test_azure_deployment.py`
- [ ] [T078] [P1] [US1] Write integration tests for Oracle deployment - Deploy with Oracle KMS + Okta, verify auth flow in `backend/tests/integration/cloud/test_oracle_deployment.py`
- [ ] [T079] [P1] [US1] Write integration tests for on-prem deployment - Deploy with OpenBao + LocalIdP, verify auth flow in `backend/tests/integration/cloud/test_onprem_deployment.py`
- [ ] [T080] [P1] [US1] Write cross-cloud consistency test - Same user, same action across all clouds, verify identical authz decisions per AC-3 in `backend/tests/integration/cloud/test_cross_cloud_consistency.py`
- [ ] [T081] [P1] [US1] Create CI/CD matrix for multi-cloud testing - GitHub Actions matrix testing all 6 cloud providers per research.md section 5 in `.github/workflows/multi-cloud-test.yml`
- [ ] [T082] [P1] [US1] Write smoke test for US1 acceptance - Deploy to Oracle+Okta, authenticate, verify JWT signature, compare to GCP in `backend/tests/smoke/test_us1_acceptance.py`

### Circuit Breakers & Graceful Degradation (6 tasks)

- [ ] [T083] [P1] [US1] Write unit tests for KMS circuit breaker - Test threshold, recovery, degraded mode in `backend/tests/unit/utils/test_kms_circuit_breaker.py`
- [ ] [T084] [P1] [US1] Implement KMS circuit breaker wrapper - Wrap all KMS calls with Purgatory breaker per research.md section 4 in `backend/src/umes/utils/kms_circuit_breaker.py`
- [ ] [T085] [P1] [US1] Write unit tests for IdP circuit breaker - Test threshold, cached token fallback in `backend/tests/unit/utils/test_idp_circuit_breaker.py`
- [ ] [T086] [P1] [US1] Implement IdP circuit breaker wrapper - Wrap IdP calls, fall back to cached tokens (5 min max) in `backend/src/umes/utils/idp_circuit_breaker.py`
- [ ] [T087] [P1] [US1] Write integration tests for graceful degradation - Simulate KMS/IdP failures, verify degraded operation in `backend/tests/integration/test_graceful_degradation.py`
- [ ] [T088] [P1] [US1] Add Prometheus metrics for circuit breaker state - Export circuit_breaker_state, failures_total, degraded_mode_active in `backend/src/umes/utils/metrics.py`

---

## Phase 4: User Story 2 - End User Authentication & SSO (Priority P1) (32 tasks)

**User Story**: End user authenticates once, accesses multiple Hextropian products with consistent permissions.

**Acceptance Criteria**:
1. User authenticates, receives JWT with identity, tenant, entitlements
2. Services validate token via UMES introspection API
3. User switches tenant context, receives new token
4. Expired session redirects to UMES for re-authentication

**Independent Test**: User logs in, token validated by mock services, tenant switch verified, session expiration tested.

### Authentication API Endpoints (12 tasks)

- [ ] [T089] [P1] [US2] Write unit tests for login endpoint - Test credential validation, token generation, audit logging in `backend/tests/unit/api/test_auth_login.py`
- [ ] [T090] [P1] [US2] Implement POST /auth/login endpoint - Accept username/password, authenticate via IdP, generate tokens per openapi.yaml in `backend/src/umes/api/auth.py::login()`
- [ ] [T091] [P1] [US2] Write unit tests for logout endpoint - Test token revocation, session cleanup in `backend/tests/unit/api/test_auth_logout.py`
- [ ] [T092] [P1] [US2] Implement POST /auth/logout endpoint - Revoke access and refresh tokens, audit log in `backend/src/umes/api/auth.py::logout()`
- [ ] [T093] [P1] [US2] Write unit tests for refresh endpoint - Test refresh token validation, new access token generation in `backend/tests/unit/api/test_auth_refresh.py`
- [ ] [T094] [P1] [US2] Implement POST /auth/refresh endpoint - Validate refresh token, issue new access token per openapi.yaml in `backend/src/umes/api/auth.py::refresh()`
- [ ] [T095] [P1] [US2] Write integration tests for complete auth flow - Login → validate → refresh → logout in `backend/tests/integration/test_auth_flow.py`
- [ ] [T096] [P1] [US2] Write unit tests for MFA enable endpoint - Test TOTP secret generation, QR code, verification in `backend/tests/unit/api/test_auth_mfa.py`
- [ ] [T097] [P1] [US2] Implement POST /auth/mfa/enable endpoint - Generate TOTP secret, encrypt with KMS, return QR code in `backend/src/umes/api/auth.py::mfa_enable()`
- [ ] [T098] [P1] [US2] Write unit tests for MFA verify endpoint - Test TOTP validation, backup codes in `backend/tests/unit/api/test_auth_mfa_verify.py`
- [ ] [T099] [P1] [US2] Implement POST /auth/mfa/verify endpoint - Validate TOTP code, update user.mfa_enabled in `backend/src/umes/api/auth.py::mfa_verify()`
- [ ] [T100] [P1] [US2] Write smoke test for authentication flow - End-to-end login with all IdP types in `backend/tests/smoke/test_authentication_flow.py`

### Token Validation & Introspection (8 tasks)

- [ ] [T101] [P1] [US2] Write unit tests for token validate endpoint - Test JWT validation, revocation check, claims extraction in `backend/tests/unit/api/test_tokens_validate.py`
- [ ] [T102] [P1] [US2] Implement POST /tokens/validate endpoint - Validate JWT, check revocation, return user context per openapi.yaml in `backend/src/umes/api/tokens.py::validate()`
- [ ] [T103] [P1] [US2] Write unit tests for token introspect endpoint - Test detailed token inspection, metadata return in `backend/tests/unit/api/test_tokens_introspect.py`
- [ ] [T104] [P1] [US2] Implement POST /tokens/introspect endpoint - Return full token metadata (user, tenant, roles, entitlements) in `backend/src/umes/api/tokens.py::introspect()`
- [ ] [T105] [P1] [US2] Write unit tests for token revoke endpoint - Test immediate revocation, Redis update in `backend/tests/unit/api/test_tokens_revoke.py`
- [ ] [T106] [P1] [US2] Implement POST /tokens/revoke endpoint - Revoke token, update Redis, audit log in `backend/src/umes/api/tokens.py::revoke()`
- [ ] [T107] [P1] [US2] Write integration tests for token introspection by services - Mock service calls /tokens/validate, receives user context in `backend/tests/integration/test_service_token_validation.py`
- [ ] [T108] [P1] [US2] Write performance tests for token validation - Verify <50ms p95 latency per SC-001 in `backend/tests/performance/test_token_validation_latency.py`

### Tenant Context Switching (6 tasks)

- [ ] [T109] [P1] [US2] Write unit tests for tenant switch logic - Test user with multi-tenant membership, token regeneration in `backend/tests/unit/services/test_tenant_switcher.py`
- [ ] [T110] [P1] [US2] Implement tenant switch service - Validate membership, revoke old token, issue new token with updated tenant_id in `backend/src/umes/services/tenant_switcher.py`
- [ ] [T111] [P1] [US2] Write unit tests for GET /users/{user_id}/memberships - Test multi-tenant user, return tenant list in `backend/tests/unit/api/test_users_memberships.py`
- [ ] [T112] [P1] [US2] Implement GET /users/{user_id}/memberships endpoint - Return user's tenant memberships with roles per openapi.yaml in `backend/src/umes/api/users.py::get_memberships()`
- [ ] [T113] [P1] [US2] Write integration tests for tenant switching - User switches from Tenant A to Tenant B, verify token changes, old token revoked in `backend/tests/integration/test_tenant_switching.py`
- [ ] [T114] [P1] [US2] Write smoke test for multi-tenant user flow - User with 3 tenants, switch between them, verify isolation in `backend/tests/smoke/test_multi_tenant_flow.py`

### Session Management (6 tasks)

- [ ] [T115] [P1] [US2] Write unit tests for session expiration - Test access token 15min TTL, refresh token 7day TTL in `backend/tests/unit/services/test_session_manager.py`
- [ ] [T116] [P1] [US2] Implement session expiration background job - Cleanup expired refresh tokens, audit log in `backend/src/umes/jobs/session_cleanup.py`
- [ ] [T117] [P1] [US2] Write unit tests for session revocation on user deletion - Test cascade revocation in `backend/tests/unit/services/test_user_deletion.py`
- [ ] [T118] [P1] [US2] Implement user deletion handler - Revoke all tokens, update Redis, soft delete user in `backend/src/umes/services/user_manager.py::delete_user()`
- [ ] [T119] [P1] [US2] Write integration tests for session timeout - Simulate expired token, verify service rejects, redirects to auth in `backend/tests/integration/test_session_timeout.py`
- [ ] [T120] [P1] [US2] Write smoke test for US2 acceptance - Complete SSO flow across mock services, tenant switch, expiration in `backend/tests/smoke/test_us2_acceptance.py`

---

## Phase 5: User Story 3 - Role & Entitlement Management (Priority P1) (30 tasks)

**User Story**: System administrator defines roles, assigns capabilities to roles, assigns users to roles within tenants.

**Acceptance Criteria**:
1. Admin creates role "Document Reviewer" with entitlements
2. Admin assigns user to role in Tenant A (tenant-scoped)
3. Service calls authz API for "write:documents", UMES denies
4. User attempts Tenant B access, UMES denies (tenant isolation)

**Independent Test**: Admin creates roles, assigns to users, authorization API enforces permissions correctly.

### Authorization Service (10 tasks)

- [ ] [T121] [P1] [US3] Write unit tests for entitlement evaluator - Test capability-based checks, role resolution in `backend/tests/unit/services/test_entitlement_evaluator.py`
- [ ] [T122] [P1] [US3] Implement entitlement evaluator service - Resolve user entitlements from roles, check permission in `backend/src/umes/services/entitlement_evaluator.py`
- [ ] [T123] [P1] [US3] Write unit tests for authorization check service - Test permission granted/denied logic, tenant scoping in `backend/tests/unit/services/test_authz_checker.py`
- [ ] [T124] [P1] [US3] Implement authorization check service - Check user has required entitlement for tenant/resource in `backend/src/umes/services/authz_checker.py`
- [ ] [T125] [P1] [US3] Write unit tests for POST /authz/check endpoint - Test single permission check, audit logging in `backend/tests/unit/api/test_authz_check.py`
- [ ] [T126] [P1] [US3] Implement POST /authz/check endpoint - Accept permission request, return granted/denied per openapi.yaml in `backend/src/umes/api/authz.py::check()`
- [ ] [T127] [P1] [US3] Write unit tests for POST /authz/batch endpoint - Test multiple permission checks in single call in `backend/tests/unit/api/test_authz_batch.py`
- [ ] [T128] [P1] [US3] Implement POST /authz/batch endpoint - Batch authorization checks, return array of results in `backend/src/umes/api/authz.py::batch_check()`
- [ ] [T129] [P1] [US3] Write integration tests for authorization decisions - Test tenant isolation, permission denial in `backend/tests/integration/test_authorization_decisions.py`
- [ ] [T130] [P1] [US3] Write performance tests for authorization checks - Verify <100ms p95 latency per SC-001 in `backend/tests/performance/test_authz_performance.py`

### Role Management API (10 tasks)

- [ ] [T131] [P1] [US3] Write unit tests for POST /roles endpoint - Test role creation, tenant scoping, validation in `backend/tests/unit/api/test_roles_create.py`
- [ ] [T132] [P1] [US3] Implement POST /roles endpoint - Create role with name, description, tenant_id per openapi.yaml in `backend/src/umes/api/roles.py::create_role()`
- [ ] [T133] [P1] [US3] Write unit tests for GET /roles endpoint - Test list roles, filter by tenant, pagination in `backend/tests/unit/api/test_roles_list.py`
- [ ] [T134] [P1] [US3] Implement GET /roles endpoint - List roles with RLS filtering, pagination in `backend/src/umes/api/roles.py::list_roles()`
- [ ] [T135] [P1] [US3] Write unit tests for GET /roles/{role_id} endpoint - Test role retrieval, RLS enforcement in `backend/tests/unit/api/test_roles_get.py`
- [ ] [T136] [P1] [US3] Implement GET /roles/{role_id} endpoint - Retrieve single role with entitlements in `backend/src/umes/api/roles.py::get_role()`
- [ ] [T137] [P1] [US3] Write unit tests for POST /roles/{role_id}/entitlements - Test add entitlements to role in `backend/tests/unit/api/test_roles_add_entitlements.py`
- [ ] [T138] [P1] [US3] Implement POST /roles/{role_id}/entitlements endpoint - Add entitlements to role, audit log in `backend/src/umes/api/roles.py::add_entitlements()`
- [ ] [T139] [P1] [US3] Write unit tests for DELETE /roles/{role_id}/entitlements - Test remove entitlements from role in `backend/tests/unit/api/test_roles_remove_entitlements.py`
- [ ] [T140] [P1] [US3] Implement DELETE /roles/{role_id}/entitlements endpoint - Remove entitlements, audit log in `backend/src/umes/api/roles.py::remove_entitlements()`

### User Role Assignment (10 tasks)

- [ ] [T141] [P1] [US3] Write unit tests for assign user to role - Test membership role assignment, tenant scoping in `backend/tests/unit/services/test_role_assigner.py`
- [ ] [T142] [P1] [US3] Implement role assignment service - Assign role to user membership, audit log in `backend/src/umes/services/role_assigner.py`
- [ ] [T143] [P1] [US3] Write unit tests for POST /tenants/{tenant_id}/members endpoint - Test invite user to tenant, assign default role in `backend/tests/unit/api/test_tenants_members.py`
- [ ] [T144] [P1] [US3] Implement POST /tenants/{tenant_id}/members endpoint - Create membership, assign roles per openapi.yaml in `backend/src/umes/api/tenants.py::add_member()`
- [ ] [T145] [P1] [US3] Write unit tests for GET /users/{user_id}/entitlements - Test aggregate entitlements from all roles in `backend/tests/unit/api/test_users_entitlements.py`
- [ ] [T146] [P1] [US3] Implement GET /users/{user_id}/entitlements endpoint - Return user's effective entitlements for tenant in `backend/src/umes/api/users.py::get_entitlements()`
- [ ] [T147] [P1] [US3] Write integration tests for role-based authorization - Create role, assign to user, verify authz check respects roles in `backend/tests/integration/test_role_based_authz.py`
- [ ] [T148] [P1] [US3] Write integration tests for tenant isolation in authz - User in Tenant A cannot access Tenant B resources per AC-4 in `backend/tests/integration/test_tenant_isolation_authz.py`
- [ ] [T149] [P1] [US3] Write security tests for privilege escalation - Verify users cannot grant themselves admin roles in `backend/tests/security/test_privilege_escalation.py`
- [ ] [T150] [P1] [US3] Write smoke test for US3 acceptance - Admin creates role, assigns to user, authz enforced correctly in `backend/tests/smoke/test_us3_acceptance.py`

---

## Phase 6: User Story 4 - Service Developer Integration (Priority P2) (22 tasks)

**User Story**: Developer integrates new service with UMES using simple SDK/library.

**Acceptance Criteria**:
1. Developer adds UMES client library, configures endpoint
2. Protected endpoint requires "write:analytics", client calls authz API
3. Service fails closed when UMES unavailable
4. Service propagates JWT in downstream calls

**Independent Test**: Mock service integrates UMES client, validates tokens, enforces permissions.

### UMES Client Library (Python SDK) (12 tasks)

- [ ] [T151] [P2] [US4] Write unit tests for UMESClient initialization - Test config validation, timeout settings in `client-sdk/tests/unit/test_client_init.py`
- [ ] [T152] [P2] [US4] Implement UMESClient class - Client with base_url, api_key, timeout config in `client-sdk/src/umes_client/client.py`
- [ ] [T153] [P2] [US4] Write unit tests for client.validate_token() - Test JWT validation via /tokens/validate in `client-sdk/tests/unit/test_validate_token.py`
- [ ] [T154] [P2] [US4] Implement client.validate_token() method - Call POST /tokens/validate, return user context in `client-sdk/src/umes_client/client.py::validate_token()`
- [ ] [T155] [P2] [US4] Write unit tests for client.check_permission() - Test authz check via /authz/check in `client-sdk/tests/unit/test_check_permission.py`
- [ ] [T156] [P2] [US4] Implement client.check_permission() method - Call POST /authz/check, return granted/denied in `client-sdk/src/umes_client/client.py::check_permission()`
- [ ] [T157] [P2] [US4] Write unit tests for client.batch_check_permissions() - Test batch authz via /authz/batch in `client-sdk/tests/unit/test_batch_check.py`
- [ ] [T158] [P2] [US4] Implement client.batch_check_permissions() method - Call POST /authz/batch with array of checks in `client-sdk/src/umes_client/client.py::batch_check_permissions()`
- [ ] [T159] [P2] [US4] Write unit tests for client.authenticate() - Test login via /auth/login in `client-sdk/tests/unit/test_authenticate.py`
- [ ] [T160] [P2] [US4] Implement client.authenticate() method - Call POST /auth/login, return tokens in `client-sdk/src/umes_client/client.py::authenticate()`
- [ ] [T161] [P2] [US4] Write unit tests for client error handling - Test timeout, connection errors, fail-closed behavior in `client-sdk/tests/unit/test_client_errors.py`
- [ ] [T162] [P2] [US4] Implement client exception hierarchy - UMESError, AuthenticationError, AuthorizationError, ServiceUnavailableError, RateLimitError in `client-sdk/src/umes_client/exceptions.py`

### FastAPI Integration Middleware (6 tasks)

- [ ] [T163] [P2] [US4] Write unit tests for UMESAuth FastAPI dependency - Test token extraction from header, validation in `client-sdk/tests/unit/test_fastapi_dependency.py`
- [ ] [T164] [P2] [US4] Implement UMESAuth FastAPI dependency - Dependency to extract Bearer token, validate via client in `client-sdk/src/umes_client/fastapi.py::UMESAuth()`
- [ ] [T165] [P2] [US4] Write unit tests for require_permission() dependency factory - Test permission check decorator in `client-sdk/tests/unit/test_require_permission.py`
- [ ] [T166] [P2] [US4] Implement require_permission() dependency factory - Factory creating dependency that checks permission in `client-sdk/src/umes_client/fastapi.py::require_permission()`
- [ ] [T167] [P2] [US4] Write integration tests for FastAPI middleware - Mock FastAPI app with protected endpoints, verify authz in `client-sdk/tests/integration/test_fastapi_integration.py`
- [ ] [T168] [P2] [US4] Create example FastAPI service using UMES client - Complete working example per quickstart.md in `examples/fastapi-service/main.py`

### Documentation & Developer Experience (4 tasks)

- [ ] [T169] [P2] [US4] Update quickstart.md with SDK installation - Add pip install umes-client instructions in `specs/001-umes-identity-subsystem/quickstart.md`
- [ ] [T170] [P2] [US4] Create SDK API reference documentation - Auto-generate from docstrings with Sphinx in `client-sdk/docs/api.md`
- [ ] [T171] [P2] [US4] Write SDK integration guide - Step-by-step guide for FastAPI, Flask, Django in `client-sdk/docs/integration-guide.md`
- [ ] [T172] [P2] [US4] Write smoke test for US4 acceptance - Developer integrates SDK in <1 hour, protected endpoint works per AC in `client-sdk/tests/smoke/test_us4_acceptance.py`

---

## Phase 7: User Story 5 - API Key Management (Priority P2) (24 tasks)

**User Story**: User generates API keys for programmatic access, with rotation and revocation support.

**Acceptance Criteria**:
1. User creates API key, receives plaintext once, hash stored
2. Service validates key, receives user context
3. User rotates key, old key expires after grace period
4. Admin revokes key, all requests fail immediately
5. Revocation logged in audit

**Independent Test**: Create API key, use for service access, rotate, verify old key expires, revoke, verify fails.

### API Key Service (10 tasks)

- [ ] [T173] [P2] [US5] Write unit tests for API key generation - Test key format (umes_live_*), SHA-256 hashing, expiration in `backend/tests/unit/services/test_api_key_generator.py`
- [ ] [T174] [P2] [US5] Implement API key generation service - Generate key_id and secret, hash secret, store in DB in `backend/src/umes/services/api_key_generator.py`
- [ ] [T175] [P2] [US5] Write unit tests for API key validation - Test hash comparison, expiration check, revocation check in `backend/tests/unit/services/test_api_key_validator.py`
- [ ] [T176] [P2] [US5] Implement API key validation service - Validate key hash, check expiration/revocation, return user context in `backend/src/umes/services/api_key_validator.py`
- [ ] [T177] [P2] [US5] Write unit tests for API key rotation - Test new key generation, grace period, old key expiration in `backend/tests/unit/services/test_api_key_rotator.py`
- [ ] [T178] [P2] [US5] Implement API key rotation service - Generate new key, mark old key for expiration after grace period in `backend/src/umes/services/api_key_rotator.py`
- [ ] [T179] [P2] [US5] Write unit tests for API key revocation - Test immediate revocation, audit logging in `backend/tests/unit/services/test_api_key_revoker.py`
- [ ] [T180] [P2] [US5] Implement API key revocation service - Set revoked_at timestamp, audit log in `backend/src/umes/services/api_key_revoker.py`
- [ ] [T181] [P2] [US5] Write integration tests for API key lifecycle - Create → use → rotate → revoke → verify fails in `backend/tests/integration/test_api_key_lifecycle.py`
- [ ] [T181b] [P2] [US5] Write security test for refresh token rotation attack - Test old refresh token cannot be reused after rotation, verify rotation invalidates previous token in `backend/tests/security/test_refresh_token_rotation.py`
- [ ] [T182] [P2] [US5] Write security tests for API key storage - Verify plaintext never stored, hash irreversible in `backend/tests/security/test_api_key_security.py`

### API Key Management Endpoints (10 tasks)

- [ ] [T183] [P2] [US5] Write unit tests for POST /api-keys endpoint - Test key creation, plaintext return once in `backend/tests/unit/api/test_api_keys_create.py`
- [ ] [T184] [P2] [US5] Implement POST /api-keys endpoint - Create key, return plaintext once, audit log per openapi.yaml in `backend/src/umes/api/api_keys.py::create_key()`
- [ ] [T185] [P2] [US5] Write unit tests for GET /api-keys endpoint - Test list user's keys, pagination in `backend/tests/unit/api/test_api_keys_list.py`
- [ ] [T186] [P2] [US5] Implement GET /api-keys endpoint - List keys for user/tenant, RLS enforced in `backend/src/umes/api/api_keys.py::list_keys()`
- [ ] [T187] [P2] [US5] Write unit tests for GET /api-keys/{key_id} endpoint - Test retrieve key metadata (not secret) in `backend/tests/unit/api/test_api_keys_get.py`
- [ ] [T188] [P2] [US5] Implement GET /api-keys/{key_id} endpoint - Return key metadata without secret in `backend/src/umes/api/api_keys.py::get_key()`
- [ ] [T189] [P2] [US5] Write unit tests for POST /api-keys/{key_id}/rotate - Test rotation, grace period in `backend/tests/unit/api/test_api_keys_rotate.py`
- [ ] [T190] [P2] [US5] Implement POST /api-keys/{key_id}/rotate endpoint - Rotate key, return new plaintext, set grace period in `backend/src/umes/api/api_keys.py::rotate_key()`
- [ ] [T191] [P2] [US5] Write unit tests for POST /api-keys/{key_id}/revoke - Test immediate revocation in `backend/tests/unit/api/test_api_keys_revoke.py`
- [ ] [T192] [P2] [US5] Implement POST /api-keys/{key_id}/revoke endpoint - Revoke key immediately, audit log in `backend/src/umes/api/api_keys.py::revoke_key()`

### API Key Authentication Middleware (4 tasks)

- [ ] [T193] [P2] [US5] Write unit tests for X-API-Key authentication middleware - Test header extraction, validation in `backend/tests/unit/middleware/test_api_key_auth.py`
- [ ] [T194] [P2] [US5] Implement X-API-Key authentication middleware - Extract key from header, validate, set user context in `backend/src/umes/middleware/api_key_auth.py`
- [ ] [T195] [P2] [US5] Write integration tests for API key authentication - Service calls UMES with X-API-Key header, receives user context in `backend/tests/integration/test_api_key_authentication.py`
- [ ] [T196] [P2] [US5] Write smoke test for US5 acceptance - Complete API key lifecycle per AC in `backend/tests/smoke/test_us5_acceptance.py`

---

## Phase 8: User Story 6 - Audit & Compliance (Priority P2) (20 tasks)

**User Story**: Compliance officer queries identity events for tenant/user, exports logs for regulatory review.

**Acceptance Criteria**:
1. User authentication event logged with metadata
2. Permission change event logged with old/new roles
3. API key creation event logged (key_id, not secret)
4. Compliance officer queries 90 days of Tenant A events
5. Export provides JSON with hash chain verification

**Independent Test**: Perform identity operations, query audit logs, verify completeness, export and verify integrity.

### Audit Query Service (8 tasks)

- [ ] [T197] [P2] [US6] Write unit tests for audit query service - Test filters (user_id, tenant_id, event_type, date range) in `backend/tests/unit/services/test_audit_query.py`
- [ ] [T198] [P2] [US6] Implement audit query service - Query audit_events with filters, pagination in `backend/src/umes/services/audit_query.py`
- [ ] [T199] [P2] [US6] Write unit tests for GET /audit/events endpoint - Test query filters, pagination, RLS in `backend/tests/unit/api/test_audit_events.py`
- [ ] [T200] [P2] [US6] Implement GET /audit/events endpoint - Query audit logs with filters per openapi.yaml in `backend/src/umes/api/audit.py::query_events()`
- [ ] [T201] [P2] [US6] Write unit tests for audit event enrichment - Test join with user/tenant/role metadata in `backend/tests/unit/services/test_audit_enricher.py`
- [ ] [T202] [P2] [US6] Implement audit event enrichment service - Join audit events with related entities for readable output in `backend/src/umes/services/audit_enricher.py`
- [ ] [T203] [P2] [US6] Write integration tests for audit query - Perform operations, query logs, verify all events captured in `backend/tests/integration/test_audit_query.py`
- [ ] [T204] [P2] [US6] Write performance tests for audit queries - Verify <5s for 90 days per SC-012 in `backend/tests/performance/test_audit_query_performance.py`

### Audit Export & Verification (8 tasks)

- [ ] [T205] [P2] [US6] Write unit tests for audit export service - Test JSON Lines format, manifest generation in `backend/tests/unit/services/test_audit_exporter.py`
- [ ] [T206] [P2] [US6] Implement audit export service - Export to JSON Lines with hash chain manifest in `backend/src/umes/services/audit_exporter.py`
- [ ] [T207] [P2] [US6] Write unit tests for POST /audit/export endpoint - Test export request, streaming response in `backend/tests/unit/api/test_audit_export.py`
- [ ] [T208] [P2] [US6] Implement POST /audit/export endpoint - Stream audit logs as JSON Lines per openapi.yaml in `backend/src/umes/api/audit.py::export_events()`
- [ ] [T209] [P2] [US6] Write unit tests for hash chain verification - Test verify integrity, detect tampering in `backend/tests/unit/services/test_audit_verifier_advanced.py`
- [ ] [T210] [P2] [US6] Implement POST /audit/verify endpoint - Verify hash chain for date range, return integrity status in `backend/src/umes/api/audit.py::verify_chain()`
- [ ] [T211] [P2] [US6] Write integration tests for audit export - Export logs, re-import, verify hash chain in `backend/tests/integration/test_audit_export_import.py`
- [ ] [T212] [P2] [US6] Write security tests for audit immutability - Verify UPDATE/DELETE triggers prevent modification in `backend/tests/security/test_audit_immutability.py`

### Audit Event Coverage (4 tasks)

- [ ] [T213] [P2] [US6] Write integration tests for authentication event coverage - Login/logout/mfa events logged per AC-1 in `backend/tests/integration/test_audit_auth_events.py`
- [ ] [T214] [P2] [US6] Write integration tests for authorization event coverage - Permission granted/denied logged per FR-017 in `backend/tests/integration/test_audit_authz_events.py`
- [ ] [T215] [P2] [US6] Write integration tests for entitlement change event coverage - Role assignments logged per AC-2 in `backend/tests/integration/test_audit_entitlement_events.py`
- [ ] [T216] [P2] [US6] Write smoke test for US6 acceptance - Complete audit query and export per AC in `backend/tests/smoke/test_us6_acceptance.py`

---

## Phase 9: User Story 7 - Multi-Tenant User Context Switching (Priority P3) (16 tasks)

**User Story**: User with memberships in multiple tenants switches contexts, receives appropriate permissions.

**Acceptance Criteria**:
1. User authenticates, receives list of available tenants
2. User selects Tenant A, token includes Tenant A context/roles
3. User switches to Tenant B, new token issued, old token invalidated
4. User with Tenant A token cannot access Tenant B resources

**Independent Test**: User with 3 tenants switches contexts, verify token changes, test isolation.

### Multi-Tenant Context Service (6 tasks)

- [ ] [T217] [P3] [US7] Write unit tests for multi-tenant context resolver - Test resolve user's tenants, roles per tenant in `backend/tests/unit/services/test_multi_tenant_resolver.py`
- [ ] [T218] [P3] [US7] Implement multi-tenant context resolver - Query user memberships, aggregate roles per tenant in `backend/src/umes/services/multi_tenant_resolver.py`
- [ ] [T219] [P3] [US7] Write unit tests for context switch validation - Test user has membership in target tenant in `backend/tests/unit/services/test_context_switch_validator.py`
- [ ] [T220] [P3] [US7] Implement context switch validation service - Verify user can switch to target tenant in `backend/src/umes/services/context_switch_validator.py`
- [ ] [T221] [P3] [US7] Write integration tests for multi-tenant context resolution - User with 3 tenants, verify role aggregation in `backend/tests/integration/test_multi_tenant_resolution.py`
- [ ] [T222] [P3] [US7] Write integration tests for context switch - Switch tenant, verify old token revoked, new token issued in `backend/tests/integration/test_context_switch.py`

### Multi-Tenant User Management (6 tasks)

- [ ] [T223] [P3] [US7] Write unit tests for POST /users endpoint - Test user creation, email uniqueness in `backend/tests/unit/api/test_users_create.py`
- [ ] [T224] [P3] [US7] Implement POST /users endpoint - Create user, audit log per openapi.yaml in `backend/src/umes/api/users.py::create_user()`
- [ ] [T225] [P3] [US7] Write unit tests for GET /users/{user_id} endpoint - Test retrieve user, RLS in `backend/tests/unit/api/test_users_get.py`
- [ ] [T226] [P3] [US7] Implement GET /users/{user_id} endpoint - Retrieve user with memberships in `backend/src/umes/api/users.py::get_user()`
- [ ] [T227] [P3] [US7] Write unit tests for tenant invitation flow - Test invite user, accept invitation, status transitions in `backend/tests/unit/services/test_tenant_invitation.py`
- [ ] [T228] [P3] [US7] Implement tenant invitation service - Send invitation, create pending membership in `backend/src/umes/services/tenant_invitation.py`

### Tenant Management Endpoints (4 tasks)

- [ ] [T229] [P3] [US7] Write unit tests for POST /tenants endpoint - Test tenant creation, slug validation in `backend/tests/unit/api/test_tenants_create.py`
- [ ] [T230] [P3] [US7] Implement POST /tenants endpoint - Create tenant, create admin membership for creator per openapi.yaml in `backend/src/umes/api/tenants.py::create_tenant()`
- [ ] [T231] [P3] [US7] Write unit tests for GET /tenants endpoint - Test list user's tenants in `backend/tests/unit/api/test_tenants_list.py`
- [ ] [T232] [P3] [US7] Implement GET /tenants endpoint - List tenants where user is member in `backend/src/umes/api/tenants.py::list_tenants()`
- [ ] [T233] [P3] [US7] Write smoke test for US7 acceptance - Complete multi-tenant user flow per AC in `backend/tests/smoke/test_us7_acceptance.py`

---

## Phase 10: Polish & Cross-Cutting Concerns (27 tasks)

**Goal**: Rate limiting, monitoring, security hardening, deployment readiness, final integration tests.

**Independent Test**: System meets all performance SLOs, security requirements, deployment validated.

### Rate Limiting (5 tasks)

- [ ] [T234] [P2] [POLISH] Write unit tests for rate limiter - Test 10 req/min unauthenticated, 1000 req/min authenticated per FR-025 in `backend/tests/unit/middleware/test_rate_limiter.py`
- [ ] [T235] [P2] [POLISH] Implement rate limiting middleware - Redis-backed rate limiter with different limits per auth status in `backend/src/umes/middleware/rate_limiter.py`
- [ ] [T236] [P2] [POLISH] Write integration tests for rate limiting - Exceed limit, verify 429 response with retry-after header in `backend/tests/integration/test_rate_limiting.py`
- [ ] [T237] [P2] [POLISH] Write performance tests for rate limiter overhead - Verify <5ms p95 latency impact in `backend/tests/performance/test_rate_limiter_overhead.py`
- [ ] [T238] [P2] [POLISH] Add Prometheus metrics for rate limiting - Export rate_limit_hits_total, rate_limit_blocks_total in `backend/src/umes/middleware/rate_limiter.py`

### Monitoring & Observability (6 tasks)

- [ ] [T239] [P2] [POLISH] Implement Prometheus metrics exporter - Expose /metrics endpoint with request latency, error rates, circuit breaker state in `backend/src/umes/utils/metrics.py`
- [ ] [T240] [P2] [POLISH] Add structured logging with correlation IDs - Use structlog, include correlation_id in all log entries in `backend/src/umes/utils/logging.py`
- [ ] [T241] [P2] [POLISH] Implement request tracing middleware - OpenTelemetry integration, trace token validation, authz checks in `backend/src/umes/middleware/tracing.py`
- [ ] [T242] [P2] [POLISH] Create Grafana dashboards - Dashboard for auth metrics, authz metrics, performance in `backend/monitoring/grafana/umes-dashboard.json`
- [ ] [T243] [P2] [POLISH] Write performance tests for throughput - Verify 10,000 req/s sustained per SC-001 in `backend/tests/performance/test_throughput.py`
- [ ] [T244] [P2] [POLISH] Write performance tests for concurrency - Verify 100,000 concurrent users per requirements in `backend/tests/performance/test_concurrency.py`

### Security Hardening (8 tasks)

- [ ] [T245] [P1] [POLISH] Write security tests for SQL injection - Verify parameterized queries per FR-029 in `backend/tests/security/test_sql_injection.py`
- [ ] [T246] [P1] [POLISH] Write security tests for input validation - Test all API endpoints reject malformed input in `backend/tests/security/test_input_validation.py`
- [ ] [T247] [P1] [POLISH] Implement input sanitization middleware - Validate all inputs with Pydantic schemas in `backend/src/umes/middleware/input_sanitizer.py`
- [ ] [T248] [P1] [POLISH] Write security tests for HTTPS enforcement - Verify HTTP requests redirected to HTTPS in `backend/tests/security/test_https_enforcement.py`
- [ ] [T249] [P1] [POLISH] Implement HTTPS redirect middleware - Redirect HTTP to HTTPS in production in `backend/src/umes/middleware/https_redirect.py`
- [ ] [T250] [P1] [POLISH] Write security tests for CORS policy - Verify restricted origins in `backend/tests/security/test_cors.py`
- [ ] [T251] [P1] [POLISH] Configure CORS middleware - Restrict origins, allow credentials in `backend/src/umes/main.py`
- [ ] [T252] [P1] [POLISH] Run security audit with Bandit - Scan for common vulnerabilities, fix issues in CI/CD

### Final Integration Tests (8 tasks)

- [ ] [T253] [P1] [POLISH] Write end-to-end integration test - Complete flow: deploy → admin setup → user auth → service authz → audit query in `backend/tests/integration/test_e2e_flow.py`
- [ ] [T254] [P1] [POLISH] Write integration test for all user stories - Verify all 7 user stories acceptance criteria in `backend/tests/integration/test_all_user_stories.py`
- [ ] [T255] [P1] [POLISH] Write integration test for edge cases - KMS unavailable, IdP unavailable, token revocation propagation per spec.md edge cases in `backend/tests/integration/test_edge_cases.py`
- [ ] [T256] [P1] [POLISH] Verify 100% test coverage - Run coverage report, ensure 100% per constitution in CI/CD
- [ ] [T257] [P1] [POLISH] Write load test scenario - Simulate 10,000 req/s for 10 minutes, verify stability in `backend/tests/performance/test_load.py`
- [ ] [T258] [P1] [POLISH] Write soak test scenario - Run at 5,000 req/s for 4 hours, verify no memory leaks in `backend/tests/performance/test_soak.py`
- [ ] [T259] [P1] [POLISH] Verify success criteria SC-001 through SC-012 - Automated test validating all measurable outcomes in `backend/tests/integration/test_success_criteria.py`
- [ ] [T260] [P1] [POLISH] Run full multi-cloud integration test suite - Deploy to all 6 clouds, verify IT-001 through IT-010 in CI/CD

---

## Phase 11: Documentation & Deployment (27 tasks)

**Goal**: Complete documentation, deployment automation, production readiness.

### API Documentation (5 tasks)

- [ ] [T261] [P2] [POLISH] Generate OpenAPI spec from FastAPI - Auto-generate openapi.json from code in `backend/src/umes/main.py`
- [ ] [T262] [P2] [POLISH] Validate generated OpenAPI matches contracts - Compare generated vs hand-written, ensure consistency in CI/CD
- [ ] [T263] [P2] [POLISH] Set up Swagger UI endpoint - Serve interactive API docs at /docs in `backend/src/umes/main.py`
- [ ] [T264] [P2] [POLISH] Set up ReDoc endpoint - Alternative API docs at /redoc in `backend/src/umes/main.py`
- [ ] [T265] [P2] [POLISH] Create Postman collection - Export collection for manual testing in `backend/docs/umes.postman_collection.json`

### Deployment Automation (10 tasks)

- [ ] [T266] [P1] [POLISH] Create GitHub Actions CI workflow - Run tests, linting, security scan on every push in `.github/workflows/ci.yml`
- [ ] [T267] [P1] [POLISH] Create GitHub Actions multi-cloud test workflow - Matrix testing all 6 cloud providers in `.github/workflows/multi-cloud.yml`
- [ ] [T268] [P1] [POLISH] Create Docker build workflow - Build and push image to registry in `.github/workflows/docker-build.yml`
- [ ] [T269] [P1] [POLISH] Create Helm chart values for production - Production-ready values with resource limits, secrets in `backend/helm/umes/values.prod.yaml`
- [ ] [T270] [P1] [POLISH] Create Terraform modules for cloud deployment - Modules for GCP, AWS, Azure, Oracle in `backend/terraform/`
- [ ] [T271] [P1] [POLISH] Write deployment runbook - Step-by-step deployment guide per cloud in `backend/docs/runbook.md`
- [ ] [T272] [P1] [POLISH] Create database backup and restore procedures - Backup scripts, restoration docs in `backend/scripts/backup/`
- [ ] [T273] [P1] [POLISH] Create disaster recovery plan - RTO/RPO targets, failover procedures in `backend/docs/disaster-recovery.md`
- [ ] [T274] [P1] [POLISH] Set up log aggregation - Configure shipping to centralized logging (ELK/Splunk) in `backend/helm/umes/templates/filebeat.yaml`
- [ ] [T275] [P1] [POLISH] Set up alerting rules - Prometheus alerting for error rates, latency, circuit breakers in `backend/monitoring/prometheus/alerts.yml`

### Operational Documentation (7 tasks)

- [ ] [T276] [P2] [POLISH] Write operations manual - Monitoring, troubleshooting, common issues in `backend/docs/operations.md`
- [ ] [T277] [P2] [POLISH] Write security incident response plan - Procedures for key compromise, breach detection in `backend/docs/incident-response.md`
- [ ] [T278] [P2] [POLISH] Document API key rotation policy - 12-month max lifetime per FR, rotation procedures in `backend/docs/api-key-policy.md`
- [ ] [T279] [P2] [POLISH] Create migration guide for existing services - How to migrate from custom auth to UMES in `backend/docs/migration-guide.md`
- [ ] [T280] [P2] [POLISH] Write performance tuning guide - Database indexes, caching, connection pooling in `backend/docs/performance-tuning.md`
- [ ] [T281] [P2] [POLISH] Document compliance requirements - ALCOA+, GDPR, SOC2 mappings in `backend/docs/compliance.md`
- [ ] [T282] [P2] [POLISH] Create architecture decision records - ADRs for KMS, IdP, RLS, audit decisions in `backend/docs/adr/`

### Production Readiness Checklist (5 tasks)

- [ ] [T283] [P1] [POLISH] Verify production checklist from quickstart.md - All security, performance, reliability, monitoring, audit items checked
- [ ] [T284] [P1] [POLISH] Run penetration testing - Engage security team for pen test, fix findings
- [ ] [T285] [P1] [POLISH] Conduct load testing in staging - Simulate production load, verify SLOs met
- [ ] [T286] [P1] [POLISH] Review and sign off on security audit - Security team approval for production deployment
- [ ] [T287] [P1] [POLISH] Final acceptance test - All user stories, success criteria, integration tests pass - Ready for production

---

## Dependency Graph

```
Phase 1 (Setup) → Phase 2 (Foundation) → Phase 3-9 (User Stories in priority order) → Phase 10 (Polish) → Phase 11 (Deployment)

Phase 3 (US1: Cloud-agnostic deployment) is foundational for all other stories
Phase 4 (US2: Authentication) and Phase 5 (US3: Authorization) are co-dependent
Phase 6 (US4: SDK) depends on Phases 4 and 5
Phase 7 (US5: API Keys) depends on Phase 4
Phase 8 (US6: Audit) runs in parallel with other phases
Phase 9 (US7: Multi-tenant) depends on Phases 4 and 5
```

### Parallel Execution Opportunities

Tasks marked with `[P]` can run in parallel with the previous task:
- T002, T005 (dependency installation can parallelize)
- T015-T016, T019-T020 (independent models)
- T034-T035, T038-T039, T042-T043 (independent KMS adapters)
- T049-T050 (OIDC adapter while LocalIdP in progress)
- T071-T072 (Helm chart and Docker Compose can be built in parallel)

---

## Notes

- **Test-Driven Development**: ALL test tasks precede implementation tasks. Red-green-refactor cycle enforced.
- **100% Coverage Requirement**: Constitution v1.0.1 mandates 100% test coverage for UMES (all code is security-critical).
- **Independent Test Criteria**: Each phase has independent test criteria verifying that phase's functionality works in isolation.
- **Beads Integration**: Track task progress in Beads, not TodoWrite tool. Update Beads status as tasks complete.
- **Commit Frequency**: Commit after completing each logical group (e.g., after T014 User model + tests).
- **Session Continuity**: Update this file's Progress Tracking section at end of each session.

---

**Last Updated**: 2025-12-02
**Total Tasks**: 288
**Estimated Duration**: 96-144 hours (at 3-5 tasks/hour velocity)
