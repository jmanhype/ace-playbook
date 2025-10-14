"""
LLM-specific circuit breaker wrappers for DSPy predictors (T071, T074).

Provides circuit breaker protection and rate limiting for external LLM API calls
through DSPy, preventing:
- Cascading failures when OpenAI, Anthropic, or other LLM services are down (T071)
- API quota exhaustion and cost overruns (T074)
- DoS attacks (T074)

Usage:
    from ace.utils.llm_circuit_breaker import protected_predict

    # In generator/reflector __call__ method:
    prediction = protected_predict(
        self.predictor,
        circuit_name="generator",
        **kwargs
    )
"""

from typing import Any, Callable, Optional
import dspy

from ace.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    register_circuit_breaker,
    get_circuit_breaker
)
from ace.utils.rate_limiter import get_llm_rate_limiter, RateLimitExceeded
from ace.utils.logging_config import get_logger

logger = get_logger(__name__, component="llm_circuit_breaker")

# Global circuit breakers for LLM services
_llm_breakers: dict[str, CircuitBreaker] = {}


def get_or_create_llm_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
) -> CircuitBreaker:
    """
    Get or create a circuit breaker for LLM calls.

    Args:
        name: Circuit breaker name (e.g., "generator", "reflector")
        failure_threshold: Consecutive failures before opening (default: 5)
        recovery_timeout: Seconds before testing recovery (default: 60)

    Returns:
        CircuitBreaker instance
    """
    global _llm_breakers

    if name not in _llm_breakers:
        # Create new circuit breaker for this LLM component
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=Exception,  # Catch all LLM API exceptions
            name=f"llm-{name}"
        )

        _llm_breakers[name] = breaker
        register_circuit_breaker(breaker)

        logger.info(
            "llm_circuit_breaker_created",
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )

    return _llm_breakers[name]


def protected_predict(
    predictor: dspy.Predict,
    circuit_name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    rate_limit: bool = True,
    domain_id: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """
    Execute DSPy predictor with circuit breaker protection and rate limiting.

    T071: Circuit breaker protection - fails fast when LLM APIs are down
    T074: Rate limiting - prevents API quota exhaustion and DoS attacks

    This wrapper provides two layers of protection:
    1. Rate limiting (token bucket) - limits calls per minute globally
    2. Circuit breaker - fails fast during outages

    Args:
        predictor: DSPy predictor (ChainOfThought, etc.)
        circuit_name: Circuit breaker identifier (e.g., "generator", "reflector")
        failure_threshold: Consecutive failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        rate_limit: Enable rate limiting (default: True)
        domain_id: Domain ID for logging (optional)
        **kwargs: Arguments to pass to predictor

    Returns:
        Predictor output

    Raises:
        RateLimitExceeded: If rate limit exceeded (with retry_after)
        CircuitBreakerError: If circuit is open (service unavailable)
        Exception: If predictor call fails (and circuit breaker records failure)

    Example:
        prediction = protected_predict(
            self.predictor,
            circuit_name="generator",
            rate_limit=True,
            task_description=task.description,
            playbook_context=context,
            domain=task.domain
        )
    """
    # T074: Apply rate limiting first (before circuit breaker)
    if rate_limit:
        rate_limiter = get_llm_rate_limiter()
        try:
            rate_limiter.acquire(domain_id=None, tokens=1, block=True, timeout=30.0)
            logger.debug(
                "llm_rate_limit_acquired",
                circuit_name=circuit_name,
                domain_id=domain_id
            )
        except RateLimitExceeded as e:
            logger.warning(
                "llm_rate_limit_exceeded",
                circuit_name=circuit_name,
                domain_id=domain_id,
                retry_after=e.retry_after
            )
            raise

    # T071: Apply circuit breaker protection
    breaker = get_or_create_llm_breaker(
        name=circuit_name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )

    try:
        # Execute predictor with circuit breaker protection
        result = breaker.call(predictor, **kwargs)

        logger.debug(
            "llm_predict_success",
            circuit_name=circuit_name,
            state=breaker.state.value,
            domain_id=domain_id
        )

        return result

    except CircuitBreakerError as e:
        # Circuit is open - fail fast
        logger.error(
            "llm_circuit_open",
            circuit_name=circuit_name,
            failure_count=breaker.failure_count,
            domain_id=domain_id,
            error=str(e)
        )
        raise

    except Exception as e:
        # LLM API error - circuit breaker recorded failure
        logger.error(
            "llm_predict_error",
            circuit_name=circuit_name,
            state=breaker.state.value,
            failure_count=breaker.failure_count,
            domain_id=domain_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise


def get_llm_breaker_metrics() -> list[dict]:
    """
    Get metrics from all LLM circuit breakers.

    Returns:
        List of metric dicts, one per LLM circuit breaker

    Example:
        metrics = get_llm_breaker_metrics()
        for m in metrics:
            print(f"{m['name']}: {m['state']}, {m['total_failures']} failures")
    """
    return [breaker.get_metrics() for breaker in _llm_breakers.values()]


def get_llm_rate_limiter_metrics() -> dict:
    """
    Get metrics from global LLM rate limiter.

    Returns:
        Dictionary with rate limiter metrics

    Example:
        metrics = get_llm_rate_limiter_metrics()
        print(f"Throttled: {metrics['total_throttled']}/{metrics['total_calls']}")
    """
    rate_limiter = get_llm_rate_limiter()
    return rate_limiter.get_metrics()


def reset_llm_breakers() -> None:
    """
    Reset all LLM circuit breakers (for testing).

    This clears the global _llm_breakers dict, allowing tests to start fresh.
    Should only be used in test fixtures.
    """
    global _llm_breakers
    _llm_breakers = {}
    logger.debug("llm_circuit_breakers_reset")


def reset_llm_rate_limiter() -> None:
    """
    Reset global LLM rate limiter (for testing).

    This resets the rate limiter state, allowing tests to start fresh.
    Should only be used in test fixtures.
    """
    rate_limiter = get_llm_rate_limiter()
    rate_limiter.reset()
    logger.debug("llm_rate_limiter_reset")
