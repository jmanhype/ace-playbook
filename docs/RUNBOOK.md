# ACE Playbook - Operations Runbook

**Version**: 1.0.0
**Last Updated**: 2025-10-14
**Owner**: ACE Platform Team

## Table of Contents

1. [Emergency Procedures](#emergency-procedures)
2. [Rollback Procedures](#rollback-procedures)
3. [Common Incidents](#common-incidents)
4. [Health Checks](#health-checks)
5. [Performance Troubleshooting](#performance-troubleshooting)
6. [Database Operations](#database-operations)
7. [Monitoring & Alerts](#monitoring--alerts)
8. [Circuit Breaker Management](#circuit-breaker-management)

---

## Emergency Procedures

### Emergency Contacts

- **On-Call Engineer**: Check PagerDuty rotation
- **Platform Lead**: [Contact in team docs]
- **Database Admin**: [Contact in team docs]

### Severity Definitions

- **SEV1 (Critical)**: System down, data loss, security breach
- **SEV2 (High)**: Major functionality impaired, performance degraded >50%
- **SEV3 (Medium)**: Minor functionality issue, workarounds available
- **SEV4 (Low)**: Cosmetic issues, feature requests

### SEV1 Response Steps

1. **Immediate Actions** (< 5 minutes):
   ```bash
   # Check system health
   curl http://localhost:8000/health

   # Check database connectivity
   docker exec ace-db psql -U ace -c "SELECT 1;"

   # Check recent logs
   docker logs ace-app --tail 100
   ```

2. **Escalation**:
   - Page on-call engineer immediately
   - Create incident in PagerDuty
   - Start incident Slack channel: `#incident-YYYYMMDD-NNN`

3. **Communication**:
   - Post status update every 15 minutes
   - Update status page if customer-facing

---

## Rollback Procedures

### Emergency Rollback (< 5 minutes)

**When to Use**: Critical bug in production, data corruption, cascading failures

**Prerequisites**:
- Git access to production repository
- Docker Compose or Kubernetes access
- Database backup verification

**Steps**:

1. **Identify Target Version**:
   ```bash
   # List recent deployments
   git log --oneline -10

   # Identify last known good commit
   GOOD_COMMIT="109663d"  # Example from git log
   ```

2. **Rollback Application Code**:
   ```bash
   # Navigate to production directory
   cd /opt/ace-playbook

   # Create rollback branch
   git checkout -b rollback-emergency-$(date +%Y%m%d-%H%M)

   # Revert to last known good commit
   git revert --no-commit HEAD~3..HEAD  # Revert last 3 commits
   git commit -m "Emergency rollback to $GOOD_COMMIT"

   # Alternative: Hard reset (use with caution)
   # git reset --hard $GOOD_COMMIT
   ```

3. **Rollback Database Schema** (if needed):
   ```bash
   # Check current migration version
   docker exec ace-app alembic current

   # Rollback 1 migration
   docker exec ace-app alembic downgrade -1

   # Rollback to specific version
   docker exec ace-app alembic downgrade <revision_id>

   # Verify migration status
   docker exec ace-app alembic current
   ```

4. **Restart Services**:
   ```bash
   # Docker Compose
   docker-compose down
   docker-compose up -d

   # Kubernetes
   kubectl rollout undo deployment/ace-app
   kubectl rollout status deployment/ace-app

   # Verify pods are running
   kubectl get pods -l app=ace-app
   ```

5. **Verify Rollback**:
   ```bash
   # Check application version
   curl http://localhost:8000/version

   # Run smoke tests
   pytest tests/e2e/test_smoke.py -v

   # Check circuit breaker metrics
   curl http://localhost:8000/metrics | grep circuit_breaker
   ```

6. **Post-Rollback Tasks**:
   - Notify team in incident channel
   - Update status page: "Issue resolved via rollback"
   - Schedule post-mortem within 24 hours
   - Create JIRA ticket for root cause analysis

**Rollback Time Target**: < 5 minutes from decision to production

---

### Staged Rollback (Recommended)

**When to Use**: Non-critical issues, planned rollbacks, testing rollback procedures

**Steps**:

1. **Rollback Shadow Environment**:
   ```bash
   cd /opt/ace-playbook
   git checkout shadow
   git revert <commit_hash>
   git push origin shadow

   # Deploy to shadow
   ./scripts/deploy.sh shadow
   ```

2. **Run Regression Tests**:
   ```bash
   pytest tests/ -v --stage=shadow

   # Verify playbook retrieval performance
   pytest tests/benchmarks/test_retrieval_perf.py
   ```

3. **Rollback Staging**:
   ```bash
   git checkout staging
   git merge shadow
   git push origin staging

   ./scripts/deploy.sh staging
   ```

4. **Monitor for 30 Minutes**:
   ```bash
   # Watch key metrics
   watch -n 10 "curl -s http://staging:8000/metrics | grep -E '(latency|error_rate)'"
   ```

5. **Rollback Production** (if staging is healthy):
   ```bash
   git checkout production
   git merge staging
   git push origin production

   ./scripts/deploy.sh production
   ```

---

## Common Incidents

### High Retrieval Latency

**Symptoms**:
- P95 latency > 15ms
- Alert: `HighRetrievalLatency` firing
- User complaints about slow responses

**Diagnosis**:
```bash
# Check FAISS index size
docker exec ace-app python -c "
from ace.utils.faiss_index import get_faiss_manager
fm = get_faiss_manager()
for domain in ['customer-a', 'customer-b']:
    size = fm.get_index_size(domain)
    print(f'{domain}: {size} vectors')
"

# Check database query performance
docker exec ace-db psql -U ace -c "
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
WHERE query LIKE '%playbook_bullets%'
ORDER BY mean_exec_time DESC
LIMIT 10;
"

# Check circuit breaker state
curl http://localhost:8000/metrics | grep circuit_breaker_state
```

**Resolution**:
1. If FAISS index > 100K vectors per domain:
   ```bash
   # Clear unused indices
   docker exec ace-app python scripts/cleanup_faiss.py --domain <domain_id>
   ```

2. If circuit breaker is OPEN:
   ```bash
   # Check LLM API status
   curl https://status.openai.com/api/v2/status.json

   # Reset circuit breaker if API is healthy
   docker exec ace-app python scripts/reset_circuit_breaker.py --name generator
   ```

3. If database queries are slow:
   ```bash
   # Check missing indexes
   docker exec ace-db psql -U ace -f scripts/check_indexes.sql

   # Rebuild indexes if needed
   docker exec ace-db psql -U ace -c "REINDEX DATABASE ace;"
   ```

**Prevention**:
- Monitor FAISS index growth
- Set up automated index cleanup
- Add database query monitoring

---

### Out of Memory (OOM) Errors

**Symptoms**:
- `MemoryError` or `OOMKilled` in logs
- Container restarts frequently
- Alert: `HighMemoryUsage` firing

**Diagnosis**:
```bash
# Check memory usage
docker stats ace-app

# Check FAISS memory footprint
docker exec ace-app python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Check for memory leaks
docker exec ace-app python scripts/memory_profiler.py
```

**Resolution**:
1. If FAISS indices are not being cleaned up:
   ```bash
   # Verify T072 fix is deployed
   git log --oneline | grep "T072"

   # Restart service to clear memory
   docker-compose restart ace-app
   ```

2. If batch size is too large:
   ```bash
   # Check batch_merge calls in logs
   docker logs ace-app | grep batch_merge_start

   # Reduce MAX_BATCH_SIZE if needed
   # Edit ace/curator/semantic_curator.py
   # MAX_BATCH_SIZE = 100 -> 50
   ```

3. If embedding service is leaking memory:
   ```bash
   # Restart embedding service
   docker-compose restart ace-embeddings

   # Check for stuck processes
   docker exec ace-embeddings ps aux | grep transformers
   ```

**Prevention**:
- Enforce MAX_BATCH_SIZE limits
- Monitor memory usage per domain
- Add memory profiling to CI

---

### Failed Promotions (Shadow → Staging → Prod)

**Symptoms**:
- Bullets stuck in SHADOW stage
- Alert: `PromotionGateFailure` firing
- Playbook not updating in production

**Diagnosis**:
```bash
# Check promotion gate thresholds
docker exec ace-app python -c "
from ace.models.playbook import PlaybookStage
from ace.repositories.playbook_repository import PlaybookRepository
repo = PlaybookRepository()
bullets = repo.get_all('customer-a', stage=PlaybookStage.SHADOW)
for b in bullets[:10]:
    print(f'{b.id}: helpful={b.helpful_count}, harmful={b.harmful_count}, ratio={b.helpful_count/(b.harmful_count or 1):.2f}')
"

# Check stage manager logs
docker logs ace-app | grep stage_manager
```

**Resolution**:
1. If thresholds are too strict:
   ```python
   # Temporarily lower thresholds (requires code change)
   # ace/curator/semantic_curator.py
   promotion_helpful_min = 3  # Default
   promotion_ratio_min = 3.0  # Default
   ```

2. If bullets lack feedback signals:
   ```bash
   # Check reflection job status
   docker logs ace-reflector | grep reflection_complete

   # Manually trigger reflection for stuck bullets
   docker exec ace-app python scripts/manual_reflection.py --bullet-id <id>
   ```

3. If stage manager is not running:
   ```bash
   # Check stage manager cron job
   docker exec ace-app crontab -l | grep stage_manager

   # Manually trigger promotion
   docker exec ace-app python -m ace.ops.stage_manager --domain customer-a
   ```

**Prevention**:
- Monitor promotion metrics
- Add promotion alerts
- Document threshold rationale

---

### Circuit Breaker Stuck OPEN

**Symptoms**:
- All LLM calls failing immediately
- Error: `Circuit breaker 'generator' is OPEN`
- Alert: `CircuitBreakerOpen` firing for > 5 minutes

**Diagnosis**:
```bash
# Check circuit breaker state
curl http://localhost:8000/metrics | grep circuit_breaker_state

# Check circuit breaker metrics
docker exec ace-app python -c "
from ace.utils.llm_circuit_breaker import get_llm_breaker_metrics
metrics = get_llm_breaker_metrics()
for m in metrics:
    print(f'{m[\"name\"]}: {m[\"state\"]} (failures: {m[\"total_failures\"]})')
"

# Check LLM API status
curl https://status.openai.com/api/v2/status.json
curl https://status.anthropic.com/api/v2/status.json
```

**Resolution**:
1. If LLM API is healthy but circuit is stuck:
   ```bash
   # Reset circuit breaker
   docker exec ace-app python -c "
   from ace.utils.llm_circuit_breaker import reset_llm_breakers
   reset_llm_breakers()
   print('Circuit breakers reset')
   "

   # Verify reset
   curl http://localhost:8000/metrics | grep circuit_breaker_state
   ```

2. If LLM API is down:
   ```bash
   # Wait for recovery timeout (60 seconds by default)
   # Circuit will auto-transition to HALF_OPEN

   # Monitor for recovery
   watch -n 5 "curl -s http://localhost:8000/metrics | grep circuit_breaker_state"
   ```

3. If recovery timeout is too short:
   ```python
   # Increase recovery_timeout (requires code change)
   # ace/generator/cot_generator.py and ace/reflector/grounded_reflector.py
   recovery_timeout=60  # -> 120
   ```

**Prevention**:
- Monitor LLM API status proactively
- Implement retry with exponential backoff
- Add circuit breaker dashboard

---

## Health Checks

### Manual Health Check

```bash
# Application health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "v1.9.0",
#   "database": "connected",
#   "faiss": "ready",
#   "circuit_breakers": {
#     "generator": "closed",
#     "reflector": "closed"
#   }
# }

# Database health
docker exec ace-db pg_isready -U ace

# FAISS index health
docker exec ace-app python -c "
from ace.utils.faiss_index import get_faiss_manager
fm = get_faiss_manager()
print('FAISS ready:', fm is not None)
"
```

### Automated Health Monitoring

```yaml
# prometheus/alerts.yml
groups:
  - name: ace-playbook
    rules:
      - alert: ServiceDown
        expr: up{job="ace-app"} == 0
        for: 1m
        annotations:
          summary: "ACE service is down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Error rate > 5% for 5 minutes"

      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state{state="open"} == 1
        for: 5m
        annotations:
          summary: "Circuit breaker stuck open"
```

---

## Performance Troubleshooting

### Playbook Retrieval Slow (> 10ms P50)

**Target**: P50 ≤ 10ms, P95 ≤ 15ms

**Diagnosis**:
```bash
# Run performance benchmark
pytest tests/benchmarks/test_retrieval_perf.py -v

# Check FAISS search performance
docker exec ace-app python -c "
import time
from ace.utils.faiss_index import get_faiss_manager
import numpy as np

fm = get_faiss_manager()
query = np.random.rand(384).astype('float32')

start = time.perf_counter()
results = fm.search('customer-a', query, k=10)
elapsed_ms = (time.perf_counter() - start) * 1000
print(f'FAISS search: {elapsed_ms:.2f}ms')
"

# Profile database queries
docker exec ace-db psql -U ace -c "
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
WHERE query LIKE '%playbook_bullets%'
ORDER BY mean_exec_time DESC;
"
```

**Resolution**:
1. **Optimize FAISS**:
   - Reduce index size (clear old domains)
   - Use IndexIVFFlat for large indices
   - Add index sharding

2. **Optimize Database**:
   ```sql
   -- Add missing indexes
   CREATE INDEX IF NOT EXISTS idx_playbook_bullets_domain_stage
   ON playbook_bullets(domain_id, stage);

   CREATE INDEX IF NOT EXISTS idx_playbook_bullets_section
   ON playbook_bullets(section) WHERE section IS NOT NULL;
   ```

3. **Add Caching**:
   ```python
   # Add Redis caching layer
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def get_playbook(domain_id: str, stage: str):
       # Cache playbook retrieval
       pass
   ```

---

### Batch Operations Timing Out

**Symptoms**:
- `batch_merge()` exceeds 30 second timeout
- Large insight batches failing
- Memory usage spikes during batch

**Diagnosis**:
```bash
# Check batch sizes in logs
docker logs ace-app | grep batch_merge_start | tail -20

# Monitor memory during batch
docker stats ace-app --no-stream
```

**Resolution**:
1. **Reduce Batch Size**:
   ```python
   # ace/curator/semantic_curator.py
   MAX_BATCH_SIZE = 100  # Reduce if needed
   ```

2. **Process in Chunks**:
   ```python
   # Split large batches
   def batch_merge_chunked(self, task_insights, chunk_size=50):
       for i in range(0, len(task_insights), chunk_size):
           chunk = task_insights[i:i+chunk_size]
           self.batch_merge(chunk, ...)
   ```

3. **Increase Timeout**:
   ```python
   # Increase timeout if batches are legitimately large
   @timeout(60)  # 60 seconds instead of 30
   def batch_merge(...):
       pass
   ```

---

## Database Operations

### Manual Backup

```bash
# Create backup
docker exec ace-db pg_dump -U ace ace > backup-$(date +%Y%m%d-%H%M%S).sql

# Verify backup
head -20 backup-*.sql

# Compress backup
gzip backup-*.sql

# Copy to S3
aws s3 cp backup-*.sql.gz s3://ace-backups/$(date +%Y/%m/%d)/
```

### Restore from Backup

```bash
# Stop application
docker-compose stop ace-app

# Drop and recreate database
docker exec ace-db psql -U postgres -c "DROP DATABASE ace;"
docker exec ace-db psql -U postgres -c "CREATE DATABASE ace OWNER ace;"

# Restore backup
gunzip < backup-20251014-120000.sql.gz | docker exec -i ace-db psql -U ace

# Run migrations
docker exec ace-app alembic upgrade head

# Restart application
docker-compose start ace-app
```

### Migration Rollback

```bash
# List migrations
docker exec ace-app alembic history

# Show current version
docker exec ace-app alembic current

# Rollback one migration
docker exec ace-app alembic downgrade -1

# Rollback to specific version
docker exec ace-app alembic downgrade <revision_id>

# Rollback all migrations (DANGEROUS)
docker exec ace-app alembic downgrade base
```

---

## Monitoring & Alerts

### Key Metrics to Monitor

1. **Performance**:
   - `playbook_retrieval_latency_ms` (P50, P95, P99)
   - `batch_merge_duration_seconds`
   - `faiss_search_latency_ms`

2. **Errors**:
   - `http_requests_total{status="5xx"}`
   - `circuit_breaker_failures_total`
   - `llm_api_errors_total`

3. **Resource Usage**:
   - `process_resident_memory_bytes`
   - `faiss_index_size_vectors`
   - `database_connections_active`

4. **Business Metrics**:
   - `bullets_added_total`
   - `bullets_promoted_total`
   - `insights_processed_total`

### Prometheus Queries

```promql
# Retrieval latency P95
histogram_quantile(0.95, rate(playbook_retrieval_latency_ms_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Circuit breaker failures
rate(circuit_breaker_failures_total[5m])
```

---

## Circuit Breaker Management

### Check Circuit Breaker Status

```bash
# Get all circuit breaker metrics
docker exec ace-app python -c "
from ace.utils.llm_circuit_breaker import get_llm_breaker_metrics
metrics = get_llm_breaker_metrics()
for m in metrics:
    print(f'{m[\"name\"]}:')
    print(f'  State: {m[\"state\"]}')
    print(f'  Failures: {m[\"total_failures\"]}')
    print(f'  Successes: {m[\"total_successes\"]}')
    print(f'  Open count: {m[\"open_count\"]}')
"
```

### Reset Circuit Breakers

```bash
# Reset all circuit breakers
docker exec ace-app python -c "
from ace.utils.llm_circuit_breaker import reset_llm_breakers
reset_llm_breakers()
print('All circuit breakers reset')
"

# Reset specific circuit breaker
docker exec ace-app python -c "
from ace.utils.llm_circuit_breaker import get_or_create_llm_breaker
breaker = get_or_create_llm_breaker('generator')
breaker.reset()
print('Generator circuit breaker reset')
"
```

### Adjust Circuit Breaker Thresholds

```python
# ace/generator/cot_generator.py (line 260-268)
prediction = protected_predict(
    self.predictor,
    circuit_name="generator",
    failure_threshold=5,      # Increase to tolerate more failures
    recovery_timeout=60,      # Increase to wait longer before retry
    ...
)
```

**Recommendations**:
- `failure_threshold=5`: Increase to 10 for flaky APIs
- `recovery_timeout=60`: Increase to 120 for slow API recovery

---

## Post-Incident Procedures

### Post-Mortem Template

```markdown
# Post-Mortem: [Incident Title]

**Date**: YYYY-MM-DD
**Duration**: HH:MM
**Severity**: SEV1/SEV2/SEV3/SEV4
**Responders**: [Names]

## Summary
[Brief description of what happened]

## Timeline
- HH:MM - Incident detected
- HH:MM - Team notified
- HH:MM - Root cause identified
- HH:MM - Fix deployed
- HH:MM - Incident resolved

## Root Cause
[Technical explanation]

## Impact
- Users affected: [Number/percentage]
- Downtime: [Minutes]
- Data loss: [Yes/No + details]

## Resolution
[What was done to fix it]

## Lessons Learned
- What went well:
- What went poorly:
- Where we got lucky:

## Action Items
- [ ] Fix root cause (Owner: [Name], Due: [Date])
- [ ] Add monitoring (Owner: [Name], Due: [Date])
- [ ] Update runbook (Owner: [Name], Due: [Date])
```

---

## Contact Information

- **On-Call Rotation**: [PagerDuty link]
- **Incident Channel**: #incident-response
- **Status Page**: [URL]
- **Documentation**: [Confluence/GitHub Wiki]

---

**Last Reviewed**: 2025-10-14
**Next Review**: 2025-11-14
