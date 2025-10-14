# ADR 0003: Circuit Breaker for LLM API Calls

**Date**: 2025-10-14
**Status**: Accepted
**Deciders**: ACE Team
**Tags**: reliability, fault-tolerance, llm

## Context

ACE Playbook makes frequent LLM API calls to OpenAI, Anthropic, or local models via DSPy. These external services can fail due to:

1. **Service Outages**: Provider downtime (e.g., OpenAI 502 errors)
2. **Rate Limiting**: Exceeding API quotas (429 Too Many Requests)
3. **Timeout**: Slow responses exceeding client timeout
4. **Invalid Credentials**: Expired or misconfigured API keys

### Problem Statement

Without fault tolerance, LLM API failures cause:
- **Cascading Failures**: Failed requests block processing, queue backlog grows
- **Resource Exhaustion**: Threads wait on timeouts, connection pool depleted
- **Wasted Costs**: Retrying failed requests burns API quota without progress
- **Poor UX**: Users see cryptic errors, no graceful degradation

### Desired Behavior

- **Fail-Fast**: Detect repeated failures early, stop sending requests
- **Self-Healing**: Automatically test recovery after cooldown period
- **Isolation**: Per-component breakers (Generator, Reflector) avoid global blocking
- **Observability**: Metrics and logs for monitoring state transitions

## Alternatives Considered

1. **No Circuit Breaker (Status Quo)**
   - Pros: Simple, no additional code
   - Cons: Cascading failures, resource exhaustion, poor UX

2. **Retry with Exponential Backoff**
   ```python
   for attempt in range(5):
       try:
           return llm.predict(task)
       except Exception:
           sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, 16s
   ```
   - Pros: Simple, eventually succeeds if transient
   - Cons: Wastes time/quota on persistent failures, no fail-fast

3. **Simple Error Counter**
   ```python
   if error_count > 5:
       raise CircuitOpen("Too many failures")
   ```
   - Pros: Fast fail after threshold
   - Cons: No recovery testing, stuck in OPEN state forever

4. **Circuit Breaker Pattern (CHOSEN)**
   - Pros: Fail-fast + self-healing, industry-standard, observable
   - Cons: More complex, requires state management

5. **External Circuit Breaker (e.g., Hystrix, Resilience4j)**
   - Pros: Battle-tested, feature-rich
   - Cons: JVM dependency (Python project), operational overhead

## Decision

We will implement a **custom Circuit Breaker** pattern in Python with three states: CLOSED, OPEN, HALF_OPEN.

### State Machine

```
CLOSED (normal) ──[5 consecutive failures]──> OPEN (fail-fast)
                                                    │
                                                    │ [60s timeout]
                                                    ▼
                                              HALF_OPEN (testing)
                                                    │
                                      ┌─────────────┴──────────────┐
                                      │                            │
                                [failure]                    [success]
                                      │                            │
                                      ▼                            ▼
                                    OPEN                        CLOSED
```

### Configuration

- **Failure Threshold**: 5 consecutive failures → OPEN
- **Recovery Timeout**: 60 seconds in OPEN before testing HALF_OPEN
- **Half-Open Test**: 1 request succeeds → CLOSED, fails → OPEN
- **Per-Component**: Separate breakers for Generator, Reflector

### Implementation

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerOpen("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### Integration with DSPy

```python
# ace/utils/llm_circuit_breaker.py
def protected_predict(
    predictor: dspy.Predict,
    component_name: str,
    **inputs: Any
) -> dspy.Prediction:
    breaker = get_circuit_breaker(component_name)

    def predict_fn():
        return predictor(**inputs)

    return breaker.call(predict_fn)

# Usage in Generator
class CoTGenerator:
    def generate(self, task: str) -> str:
        prediction = protected_predict(
            self.predictor,
            component_name="generator",
            task=task,
            playbook_context=context
        )
        return prediction.response
```

## Consequences

### Positive

- **Fail-Fast**: 5 consecutive failures → stop sending requests (save costs)
- **Self-Healing**: Automatic recovery testing after 60s cooldown
- **Resource Protection**: Prevents connection pool exhaustion, thread blocking
- **Isolation**: Per-component breakers (Generator OPEN, Reflector still CLOSED)
- **Observable**: Metrics track state transitions, failure counts
- **User-Friendly**: Clear error messages ("LLM service unavailable, retrying in 60s")

### Negative

- **False Positives**: Transient errors (1-2 requests) still succeed
  - Mitigation: Threshold = 5 (not 1), only persistent failures trigger OPEN

- **Recovery Delay**: 60s wait before testing HALF_OPEN
  - Mitigation: Tunable timeout, can reduce to 30s for faster recovery
  - Trade-off: Lower timeout → more frequent testing → quota waste

- **State Management Overhead**: Lock contention on high concurrency
  - Mitigation: threading.Lock (fast), per-component breakers reduce contention

### Risks

- **Stuck OPEN**: Bug in recovery logic could permanently block LLM calls
  - Mitigation: Manual reset endpoint (`/admin/circuit-breaker/reset`)
  - Monitoring: Alert on "OPEN > 5 minutes"

- **Cascading OPEN**: All components OPEN → no progress
  - Mitigation: Staggered recovery timeouts (Generator 60s, Reflector 90s)
  - Fallback: Offline mode (use cached playbooks, no new generations)

## Validation

Tests (T071):
- `test_circuit_breaker_opens_after_threshold`: 5 failures → OPEN
- `test_circuit_breaker_half_open_on_success`: Recovery path
- `test_circuit_breaker_reopens_on_half_open_failure`: Persistent failure handling
- `test_circuit_breaker_thread_safety`: 20 concurrent threads

Metrics:
```
# Prometheus format
circuit_breaker_state{component="generator"} 0  # 0=CLOSED, 1=OPEN, 2=HALF_OPEN
circuit_breaker_failure_count{component="generator"} 3
circuit_breaker_success_count{component="generator"} 147
```

## Integration with Rate Limiting

Circuit breaker works in conjunction with rate limiting (T074):

```
Request Flow:
  Rate Limiter (throttle before circuit breaker)
      │
      ▼
  Circuit Breaker (fail-fast if OPEN)
      │
      ▼
  LLM API Call
```

Multi-layered protection:
1. **Rate Limiter**: Prevents quota exhaustion (60 calls/min)
2. **Circuit Breaker**: Prevents cascading failures (5 consecutive errors)

## References

- Implementation: `ace/utils/circuit_breaker.py`
- Integration: `ace/utils/llm_circuit_breaker.py`
- Tests: `tests/unit/test_circuit_breaker.py` (18 tests)
- RUNBOOK: `docs/RUNBOOK.md` (Circuit Breaker Stuck OPEN)
- Metrics: `ace/ops/metrics.py`

## Further Reading

- [Release It! Circuit Breaker Pattern](https://pragprog.com/titles/mnee2/release-it-second-edition/)
- [Martin Fowler: Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html)
- [AWS: Circuit Breaker Pattern](https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/rel_withstand_component_failures_avoid_overload.html)
