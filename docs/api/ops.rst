Operations Module
=================

Observability, metrics, tracing, and guardrail monitoring for production operations.

.. automodule:: ace.ops
   :members:
   :undoc-members:
   :show-inheritance:

MetricsCollector
----------------

Prometheus-style metrics collection.

.. autoclass:: ace.ops.metrics.MetricsCollector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

GuardrailMonitor
----------------

Automated rollback on performance regression.

.. autoclass:: ace.ops.guardrails.GuardrailMonitor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

TracingService
--------------

OpenTelemetry distributed tracing.

.. autoclass:: ace.ops.tracing.TracingService
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Metrics Collection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.ops import MetricsCollector

   # Initialize collector
   metrics = MetricsCollector()

   # Track operations
   metrics.increment("playbook.bullets.added", labels={"domain": "arithmetic"})
   metrics.increment("curator.dedup.hits")

   # Record latency
   with metrics.timer("curator.apply_delta"):
       curator.apply_delta(input)

   # Export for Prometheus
   prometheus_text = metrics.export_prometheus()
   print(prometheus_text)

Guardrail Monitoring
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.ops import GuardrailMonitor
   from ace.ops.metrics import PerformanceSnapshot

   # Initialize monitor
   monitor = GuardrailMonitor(
       success_delta_threshold=-0.08,  # -8%
       latency_delta_threshold=0.30,    # +30%
       error_rate_threshold=0.15        # 15%
   )

   # Take baseline snapshot
   baseline = PerformanceSnapshot(
       success_rate=0.95,
       p50_latency=700,
       p95_latency=1500,
       error_rate=0.02,
       timestamp=datetime.utcnow()
   )

   monitor.set_baseline(baseline)

   # Check current performance
   current = PerformanceSnapshot(
       success_rate=0.85,  # Dropped 10% - triggers alert!
       p50_latency=720,
       p95_latency=1600,
       error_rate=0.03,
       timestamp=datetime.utcnow()
   )

   trigger = monitor.check(current)

   if trigger:
       print(f"ALERT: {trigger.reason}")
       print(f"Metric: {trigger.metric_name}")
       print(f"Threshold: {trigger.threshold_value}")

       # Rollback bullets from last 24h
       rollback_bullets = repo.get_recent(hours=24)
       for bullet in rollback_bullets:
           service.quarantine_bullet(bullet.id)

Distributed Tracing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.ops import TracingService

   # Initialize tracing
   tracing = TracingService(service_name="ace-playbook")

   # Trace operation
   with tracing.start_span("curator.apply_delta") as span:
       span.set_attribute("domain_id", "arithmetic")
       span.set_attribute("insights_count", len(insights))

       output = curator.apply_delta(input)

       span.set_attribute("new_bullets", output.stats["new_bullets"])
       span.set_attribute("dedup_rate", output.stats["dedup_rate"])

   # Nested spans
   with tracing.start_span("end_to_end") as parent:
       with tracing.start_span("generate") as gen_span:
           output = generator.forward(input)

       with tracing.start_span("reflect") as ref_span:
           reflection = reflector.forward(output)

       with tracing.start_span("curate") as cur_span:
           curator_output = curator.apply_delta(reflection)

Alert Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set up alert callback
   def send_alert(trigger):
       print(f"CRITICAL: {trigger.reason}")
       # Send to PagerDuty, Slack, etc.

   monitor.on_trigger(send_alert)

   # Monitor continuously
   while True:
       current_snapshot = collect_metrics()
       trigger = monitor.check(current_snapshot)

       if trigger:
           # Alert fires automatically via callback
           pass

       time.sleep(60)  # Check every minute

Notes
-----

* **MetricsCollector**:

  * Prometheus text format export
  * Thread-safe counters with Lock
  * Labels for multi-dimensional metrics
  * Histograms for latency percentiles

* **GuardrailMonitor**:

  * Thresholds:

    * Success rate delta < -8%
    * Latency delta > +30%
    * Error rate > 15%

  * Rollback: quarantine bullets from last 24h
  * Alert callbacks for notifications
  * Audit trail in RollbackTrigger

* **TracingService**:

  * OpenTelemetry integration
  * Jaeger export for visualization
  * Automatic context propagation
  * Performance overhead: <5ms per span
