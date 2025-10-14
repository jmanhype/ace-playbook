"""
Unit tests for OpenTelemetry distributed tracing (T076).

Tests verify:
- Tracing setup and configuration
- Tracer acquisition
- Automatic span creation via decorators
- Span attributes and events
- Error handling and exception recording
- Tracing enable/disable functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from ace.utils.tracing import (
    setup_tracing,
    get_tracer,
    trace_operation,
    add_span_attributes,
    add_span_event,
    shutdown_tracing,
    _tracing_enabled
)


@pytest.fixture(autouse=True)
def reset_tracing():
    """Reset tracing state before each test."""
    import ace.utils.tracing as tracing_module
    tracing_module._tracer_provider = None
    tracing_module._tracing_enabled = True
    yield
    # Cleanup after test
    if tracing_module._tracer_provider:
        tracing_module._tracer_provider.shutdown()
    tracing_module._tracer_provider = None


def test_setup_tracing_default_config():
    """T076: setup_tracing should use default configuration."""
    with patch("ace.utils.tracing.TracerProvider") as mock_provider_class, \
         patch("ace.utils.tracing.OTLPSpanExporter") as mock_exporter_class, \
         patch("ace.utils.tracing.BatchSpanProcessor") as mock_processor_class, \
         patch("ace.utils.tracing.trace.set_tracer_provider"), \
         patch("ace.utils.tracing.SQLAlchemyInstrumentor"), \
         patch("ace.utils.tracing.RequestsInstrumentor"):

        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider

        setup_tracing()

        # Should create tracer provider
        assert mock_provider_class.called
        # Should configure OTLP exporter
        assert mock_exporter_class.called
        # Should add span processor
        assert mock_provider.add_span_processor.called


def test_setup_tracing_custom_config():
    """T076: setup_tracing should accept custom configuration."""
    with patch("ace.utils.tracing.TracerProvider") as mock_provider_class, \
         patch("ace.utils.tracing.OTLPSpanExporter") as mock_exporter_class, \
         patch("ace.utils.tracing.BatchSpanProcessor"), \
         patch("ace.utils.tracing.trace.set_tracer_provider"), \
         patch("ace.utils.tracing.SQLAlchemyInstrumentor"), \
         patch("ace.utils.tracing.RequestsInstrumentor"):

        mock_provider_class.return_value = Mock()

        setup_tracing(
            service_name="test-service",
            environment="staging",
            otlp_endpoint="https://test-collector.example.com"
        )

        # Should create OTLP exporter with custom endpoint
        mock_exporter_class.assert_called_once()
        call_kwargs = mock_exporter_class.call_args[1]
        assert "https://test-collector.example.com/v1/traces" in call_kwargs["endpoint"]


def test_setup_tracing_disabled():
    """T076: setup_tracing should respect enabled=False."""
    with patch("ace.utils.tracing.TracerProvider") as mock_provider_class:

        setup_tracing(enabled=False)

        # Should not create tracer provider
        assert not mock_provider_class.called


def test_get_tracer():
    """T076: get_tracer should return tracer instance."""
    with patch("ace.utils.tracing.trace.get_tracer") as mock_get_tracer:
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer

        tracer = get_tracer("test_module")

        assert tracer == mock_tracer
        mock_get_tracer.assert_called_once_with("test_module")


def test_trace_operation_decorator():
    """T076: trace_operation decorator should create spans."""
    with patch("ace.utils.tracing.get_tracer") as mock_get_tracer:
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span = Mock(return_value=mock_context)
        mock_get_tracer.return_value = mock_tracer

        @trace_operation("test_operation", {"key": "value"})
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"
        mock_tracer.start_as_current_span.assert_called_once_with("test_operation")
        assert mock_span.set_attribute.called


def test_trace_operation_decorator_with_error():
    """T076: trace_operation should record exceptions."""
    with patch("ace.utils.tracing.get_tracer") as mock_get_tracer:
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span = Mock(return_value=mock_context)
        mock_get_tracer.return_value = mock_tracer

        @trace_operation("test_operation")
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()

        # Should record exception
        assert mock_span.record_exception.called
        # Should set error attributes
        calls = [call[0] for call in mock_span.set_attribute.call_args_list]
        assert any("error.type" in str(call) for call in calls)


def test_trace_operation_when_disabled():
    """T076: trace_operation should skip tracing when disabled."""
    import ace.utils.tracing as tracing_module
    tracing_module._tracing_enabled = False

    with patch("ace.utils.tracing.get_tracer") as mock_get_tracer:
        @trace_operation("test_operation")
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"
        # Should not call tracer when disabled
        assert not mock_get_tracer.called

    # Reset for other tests
    tracing_module._tracing_enabled = True


def test_add_span_attributes():
    """T076: add_span_attributes should add attributes to current span."""
    with patch("ace.utils.tracing.trace.get_current_span") as mock_get_span:
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        add_span_attributes(
            domain_id="test-domain",
            task_count=5,
            bullet_count=10
        )

        # Should set all attributes
        assert mock_span.set_attribute.call_count == 3


def test_add_span_attributes_when_no_span():
    """T076: add_span_attributes should handle no active span."""
    with patch("ace.utils.tracing.trace.get_current_span") as mock_get_span:
        mock_get_span.return_value = None

        # Should not raise error
        add_span_attributes(key="value")


def test_add_span_event():
    """T076: add_span_event should add event to current span."""
    with patch("ace.utils.tracing.trace.get_current_span") as mock_get_span:
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        add_span_event("bullet_promoted", {"bullet_id": "abc123", "stage": "PROD"})

        mock_span.add_event.assert_called_once()
        call_args = mock_span.add_event.call_args
        assert call_args[0][0] == "bullet_promoted"
        assert "bullet_id" in call_args[1]["attributes"]


def test_add_span_event_when_disabled():
    """T076: add_span_event should skip when tracing disabled."""
    import ace.utils.tracing as tracing_module
    tracing_module._tracing_enabled = False

    with patch("ace.utils.tracing.trace.get_current_span") as mock_get_span:
        add_span_event("test_event")

        # Should not call get_current_span when disabled
        assert not mock_get_span.called

    # Reset for other tests
    tracing_module._tracing_enabled = True


def test_shutdown_tracing():
    """T076: shutdown_tracing should flush all pending spans."""
    with patch("ace.utils.tracing.TracerProvider") as mock_provider_class, \
         patch("ace.utils.tracing.OTLPSpanExporter"), \
         patch("ace.utils.tracing.BatchSpanProcessor"), \
         patch("ace.utils.tracing.trace.set_tracer_provider"), \
         patch("ace.utils.tracing.SQLAlchemyInstrumentor"), \
         patch("ace.utils.tracing.RequestsInstrumentor"):

        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider

        setup_tracing()
        shutdown_tracing()

        # Should call shutdown on provider
        mock_provider.shutdown.assert_called_once()


def test_trace_operation_preserves_function_metadata():
    """T076: trace_operation should preserve function name and docstring."""
    @trace_operation("test_op")
    def test_function():
        """Test docstring."""
        return "result"

    assert test_function.__name__ == "test_function"
    assert test_function.__doc__ == "Test docstring."


def test_trace_operation_with_args_and_kwargs():
    """T076: trace_operation should work with function args/kwargs."""
    with patch("ace.utils.tracing.get_tracer") as mock_get_tracer:
        mock_tracer = Mock()
        mock_span = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = Mock(return_value=mock_span)
        mock_context.__exit__ = Mock(return_value=None)
        mock_tracer.start_as_current_span = Mock(return_value=mock_context)
        mock_get_tracer.return_value = mock_tracer

        @trace_operation("test_operation")
        def test_function(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = test_function("x", "y", c="z")

        assert result == "x-y-z"
        assert mock_tracer.start_as_current_span.called


def test_setup_tracing_auto_instruments_sqlalchemy():
    """T076: setup_tracing should auto-instrument SQLAlchemy."""
    with patch("ace.utils.tracing.TracerProvider"), \
         patch("ace.utils.tracing.OTLPSpanExporter"), \
         patch("ace.utils.tracing.BatchSpanProcessor"), \
         patch("ace.utils.tracing.trace.set_tracer_provider"), \
         patch("ace.utils.tracing.SQLAlchemyInstrumentor") as mock_sql_instrumentor, \
         patch("ace.utils.tracing.RequestsInstrumentor"):

        mock_sql_instance = Mock()
        mock_sql_instrumentor.return_value = mock_sql_instance

        setup_tracing()

        # Should instrument SQLAlchemy
        mock_sql_instance.instrument.assert_called_once()


def test_setup_tracing_auto_instruments_requests():
    """T076: setup_tracing should auto-instrument HTTP requests."""
    with patch("ace.utils.tracing.TracerProvider"), \
         patch("ace.utils.tracing.OTLPSpanExporter"), \
         patch("ace.utils.tracing.BatchSpanProcessor"), \
         patch("ace.utils.tracing.trace.set_tracer_provider"), \
         patch("ace.utils.tracing.SQLAlchemyInstrumentor"), \
         patch("ace.utils.tracing.RequestsInstrumentor") as mock_requests_instrumentor:

        mock_requests_instance = Mock()
        mock_requests_instrumentor.return_value = mock_requests_instance

        setup_tracing()

        # Should instrument HTTP requests
        mock_requests_instance.instrument.assert_called_once()
