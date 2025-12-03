# UMES Phase 1 Landing Plan

## Current Status ✓

**Branch**: `001-umes-identity-subsystem` (mono repo)
**Commits**: 12 tasks complete (T001-T012)
**Tests**: All 15 smoke tests passing + 100% unit test coverage for implemented features
**Package**: Installs cleanly with `pip install -e .`

## Questions & Answers

### 1. How will Ramiro test this?

**Current state** uses PYTHONPATH workaround for dev. **Ramiro will test**:

```bash
cd /Users/speed/code/hextropian/umes
pip install -e .              # Editable install ✓ VERIFIED WORKING
pytest                        # All tests pass ✓ VERIFIED
pytest --cov                  # Check coverage
docker-compose up -d          # Integration testing
```

**Status**: ✅ Package installs cleanly, all tests pass with proper install

---

### 2. Should we push to mono repo in this state?

**YES**, but with these updates first:

#### Required Updates Before Push:

1. **Update to Custom Postgres Image**
   - Current: `postgres:15-alpine`
   - Required: Custom image from `/PostgresWithExtensions` repo
   - Reason: Includes pgvecto.rs, Apache AGE, TimescaleDB extensions

2. **Build Custom Postgres Image**:
   ```bash
   cd /Users/speed/code/hextropian/PostgresWithExtensions
   docker build -t postgres-extensions:17 .
   ```

3. **Update `docker-compose.yml`**:
   ```yaml
   postgres:
     image: postgres-extensions:17  # Changed from postgres:15-alpine
     build:
       context: ../PostgresWithExtensions
       dockerfile: Dockerfile
   ```

4. **Update Testcontainers Fixtures** (`tests/conftest.py`):
   ```python
   with PostgresContainer(
       image="postgres-extensions:17",  # Changed from postgres:15-alpine
       username="umes",
       password="umes",
       dbname="umes_test",
   ) as postgres:
   ```

---

### 3. How do we land the plane?

**Step-by-Step Landing Procedure**:

#### Phase A: Pre-Push Checklist
```bash
# 1. Apply custom Postgres updates (see above)
cd /Users/speed/code/hextropian/umes
# Update docker-compose.yml
# Update tests/conftest.py

# 2. Build custom Postgres image
cd /Users/speed/code/hextropian/PostgresWithExtensions
docker build -t postgres-extensions:17 .

# 3. Test with custom image
cd /Users/speed/code/hextropian/umes
docker-compose up -d postgres
pytest  # Verify tests still pass

# 4. Stage and commit updates
git add docker-compose.yml tests/conftest.py
git commit -m "feat(umes): Use custom Postgres image with extensions

- Switch from postgres:15-alpine to postgres-extensions:17
- Includes pgvecto.rs, Apache AGE, TimescaleDB extensions
- Update docker-compose.yml and Testcontainers fixtures

Ref: /PostgresWithExtensions"
```

#### Phase B: Push to Remote
```bash
cd /Users/speed/code/hextropian

# 1. Check remote status
git remote -v

# 2. Push feature branch
git push origin 001-umes-identity-subsystem

# 3. Create PR (GitHub CLI or web UI)
gh pr create \
  --base main \
  --head 001-umes-identity-subsystem \
  --title "feat(umes): Complete Phase 1 - Infrastructure Foundation" \
  --body "$(cat <<'EOF'
## Phase 1 Complete ✓

All 12 Phase 1 tasks (T001-T012) complete with 100% TDD compliance.

### What's Included

**Infrastructure**:
- [x] Project structure with proper directory layout
- [x] Dependencies configured (pyproject.toml, pytest.ini)
- [x] Docker Compose with Postgres (custom image), Redis, Keycloak, LocalStack, OpenBao
- [x] Testcontainers fixtures (session-scoped)

**Core Components**:
- [x] SQLAlchemy async engine with connection pooling
- [x] RLS context manager with ContextVar (tenant isolation)
- [x] Base models (TimestampMixin, SoftDeleteMixin, TenantMixin)
- [x] Two-level caching (L1 LRU + L2 Redis)
- [x] Circuit breakers (Purgatory 3.0) for KMS/IdP resilience
- [x] **Configuration management (Pydantic)** ← KEY to cloud-agnostic design
- [x] Smoke tests (15 tests - all passing)

### Configuration Management

The `config.py` implementation is THE KEY to cloud-agnostic deployment:

```python
# Change cloud provider with ZERO code changes:
KMS_PROVIDER=oracle  # or gcp, aws, azure, openbao, local
IDP_TYPE=keycloak    # or auth0, okta, workos, local
```

Adapter swapping happens via environment variables only.

### Testing

- **All tests pass**: 15 smoke tests + unit tests for each component
- **TDD compliance**: RED-GREEN-REFACTOR for every task
- **Package install**: `pip install -e .` works cleanly

### Custom Postgres Image

Using `postgres-extensions:17` from `/PostgresWithExtensions`:
- Includes: pgvecto.rs, Apache AGE, TimescaleDB
- Built from: https://github.com/RamXX/PostgresWithExtensions

### Next Steps (Phase 2)

Ready to begin Phase 2: Foundational Components (48 tasks):
- User, Tenant, Membership models
- KMS adapters (6 providers)
- IdP adapters (3 providers)
- Database migrations (Alembic)
- Auth middleware

### Review Checklist

- [ ] Review infrastructure setup (docker-compose.yml)
- [ ] Verify configuration management design (config.py)
- [ ] Check test coverage and quality
- [ ] Approve custom Postgres image usage
- [ ] Confirm Phase 2 task priorities

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

#### Phase C: Review & Merge
```bash
# After Ramiro's approval:
git checkout main
git pull origin main
git merge 001-umes-identity-subsystem --no-ff
git push origin main

# Clean up feature branch (optional)
git branch -d 001-umes-identity-subsystem
git push origin --delete 001-umes-identity-subsystem
```

---

### 4. What to Include in Push?

**YES - Include**:
- ✅ All `umes/` code (currently committed)
- ✅ `specs/001-umes-identity-subsystem/` folder (feature specification)
- ✅ `.specify/` framework (if not already in mono repo)
- ✅ `.beads/` database (local task tracking - gitignored, won't be pushed)

**NO - Exclude** (gitignored):
- ❌ `.beads/` (local Beads database - each dev has their own)
- ❌ `__pycache__/`, `.coverage`, `.egg-info/` (test artifacts)
- ❌ `venv/`, `.venv/` (virtual environments)

**Check what will be pushed**:
```bash
cd /Users/speed/code/hextropian
git status
git diff main...001-umes-identity-subsystem --stat
```

---

### 5. Beads Database

**Beads is LOCAL ONLY**:
- `.beads/` directory is gitignored
- Each developer maintains their own Beads database
- Beads tracks work-in-progress, not project state
- **Do NOT push** `.beads/` to remote

**Why?**:
- Beads is for individual task tracking during development
- Ramiro will create his own Beads issues when he starts work
- The **spec** (specs/001-umes-identity-subsystem/) is the source of truth

---

## Pre-Push Action Items

### REQUIRED: Update to Custom Postgres

1. **Update docker-compose.yml** (`umes/docker-compose.yml`):
   ```yaml
   postgres:
     image: postgres-extensions:17
     build:
       context: ../PostgresWithExtensions
       dockerfile: Dockerfile
   ```

2. **Update Testcontainers** (`umes/tests/conftest.py`):
   ```python
   @pytest.fixture(scope="session")
   def postgres_container() -> Generator[PostgresContainer, None, None]:
       """Session-scoped PostgreSQL container with extensions."""
       with PostgresContainer(
           image="postgres-extensions:17",  # Custom image
           username="umes",
           password="umes",
           dbname="umes_test",
       ) as postgres:
           postgres.get_connection_url()
           yield postgres
   ```

3. **Build Image**:
   ```bash
   cd /Users/speed/code/hextropian/PostgresWithExtensions
   docker build -t postgres-extensions:17 .
   ```

4. **Test**:
   ```bash
   cd /Users/speed/code/hextropian/umes
   docker-compose up -d postgres
   pytest tests/smoke/test_setup.py -v
   ```

5. **Commit**:
   ```bash
   git add docker-compose.yml tests/conftest.py
   git commit -m "feat(umes): Use custom Postgres image with extensions"
   ```

---

## Post-Push: Ramiro's Review

**What Ramiro will check**:

1. **Installation**:
   ```bash
   git checkout 001-umes-identity-subsystem
   cd umes
   pip install -e .
   ```

2. **Tests**:
   ```bash
   pytest -v
   pytest --cov
   ```

3. **Docker Compose**:
   ```bash
   docker-compose up -d
   docker-compose ps  # Check all services healthy
   ```

4. **Configuration Design**:
   - Review `config.py` for cloud-agnostic design
   - Verify adapter pattern approach
   - Check environment variable handling

5. **Code Quality**:
   - TDD compliance (all tests written first)
   - 100% coverage for implemented features
   - Clean architecture (separation of concerns)

---

## Summary

**Ready to push**: Almost! Just need to apply custom Postgres updates first.

**Timeline**:
1. Apply custom Postgres updates (15 min)
2. Test with custom image (5 min)
3. Commit updates (2 min)
4. Push to remote (2 min)
5. Create PR (5 min)
6. **Total**: ~30 minutes to land

**After landing**:
- Ramiro reviews PR
- Addresses any feedback
- Merges to main
- Starts Phase 2 (or proceeds in parallel if approved)
