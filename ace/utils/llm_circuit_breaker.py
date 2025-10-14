"""
LLM-specific circuit breaker wrappers for DSPy predictors (T071).

Provides circuit breaker protection for external LLM API calls through DSPy,
preventing cascading failures when OpenAI, Anthropic, or other LLM services
are experiencing issues.

Usage:
    from ace.utils.llm_circuit_breaker import protected_predict

    # In generator/reflector __call__ method:
    prediction = protected_predict(
        self.predictor,
        circuit_name="generator",
        **kwargs
    )
"""

from typing import Any, Callable
import dspy

from ace.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    register_circuit_breaker,
    get_circuit_breaker
)
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
    **kwargs: Any
) -> Any:
    """
    Execute DSPy predictor with circuit breaker protection.

    This wrapper protects against cascading failures when LLM APIs are down.
    If the circuit is open, it fails fast with CircuitBreakerError instead of
    making slow, failing API calls.

    Args:
        predictor: DSPy predictor (ChainOfThought, etc.)
        circuit_name: Circuit breaker identifier (e.g., "generator", "reflector")
        failure_threshold: Consecutive failures before opening circuit
        recovery_timeout: Seconds before attempting recovery
        **kwargs: Arguments to pass to predictor

    Returns:
        Predictor output

    Raises:
        CircuitBreakerError: If circuit is open (service unavailable)
        Exception: If predictor call fails (and circuit breaker records failure)

    Example:
        prediction = protected_predict(
            self.predictor,
            circuit_name="generator",
            task_description=task.description,
            playbook_context=context,
            domain=task.domain
        )
    """
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
            state=breaker.state.value
        )

        return result

    except CircuitBreakerError as e:
        # Circuit is open - fail fast
        logger.error(
            "llm_circuit_open",
            circuit_name=circuit_name,
            failure_count=breaker.failure_count,
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


def reset_llm_breakers() -> None:
    """
    Reset all LLM circuit breakers (for testing).

    This clears the global _llm_breakers dict, allowing tests to start fresh.
    Should only be used in test fixtures.
    """
    global _llm_breakers
    _llm_breakers = {}
    logger.debug("llm_circuit_breakers_reset")
