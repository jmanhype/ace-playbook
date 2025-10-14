Utilities Module
================

Core utilities for embeddings, FAISS indexing, circuit breakers, and rate limiting.

.. automodule:: ace.utils
   :members:
   :undoc-members:
   :show-inheritance:

EmbeddingService
----------------

Sentence-transformers wrapper for semantic embeddings.

.. autoclass:: ace.utils.embeddings.EmbeddingService
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

FAISSIndexManager
-----------------

Fast similarity search with per-domain isolation.

.. autoclass:: ace.utils.faiss_index.FAISSIndexManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

CircuitBreaker
--------------

Fault tolerance for external service calls.

.. autoclass:: ace.utils.circuit_breaker.CircuitBreaker
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

RateLimiter
-----------

Token-bucket rate limiting for API calls.

.. autoclass:: ace.utils.rate_limiter.RateLimiter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Embeddings
~~~~~~~~~~

.. code-block:: python

   from ace.utils import EmbeddingService

   # Initialize service (lazy-loads model)
   service = EmbeddingService(model_name="all-MiniLM-L6-v2")

   # Embed single text
   embedding = service.embed("Break problems into steps")
   print(f"Dimensions: {len(embedding)}")  # 384

   # Batch embedding
   texts = ["Strategy 1", "Strategy 2", "Strategy 3"]
   embeddings = service.embed_batch(texts)
   print(f"Batch size: {len(embeddings)}")  # 3

FAISS Indexing
~~~~~~~~~~~~~~

.. code-block:: python

   from ace.utils import FAISSIndexManager

   # Initialize manager
   faiss_mgr = FAISSIndexManager(dimension=384)

   # Add vectors for domain
   vectors = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
   bullet_ids = ["b1", "b2", "b3"]
   faiss_mgr.add_vectors("arithmetic", vectors, bullet_ids)

   # Search similar
   query = [0.15] * 384
   results = faiss_mgr.search("arithmetic", query, k=2)

   for bullet_id, score in results:
       print(f"{bullet_id}: similarity={score:.3f}")

Circuit Breaker
~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.utils import CircuitBreaker
   import openai

   # Initialize breaker
   breaker = CircuitBreaker(
       failure_threshold=5,
       recovery_timeout=60
   )

   # Wrap API call
   @breaker.call
   def call_openai(prompt):
       return openai.Completion.create(
           model="gpt-4",
           prompt=prompt
       )

   try:
       result = call_openai("Hello")
   except CircuitBreakerOpen:
       print("Circuit open - service unavailable")
       # Use fallback

Rate Limiting
~~~~~~~~~~~~~

.. code-block:: python

   from ace.utils import RateLimiter

   # 10 calls per minute
   limiter = RateLimiter(calls=10, period=60)

   @limiter.limit
   def api_call(data):
       return expensive_operation(data)

   # Calls are rate-limited
   for i in range(20):
       try:
           result = api_call(data)
       except RateLimitExceeded:
           print("Rate limit hit - backing off")
           time.sleep(5)

Notes
-----

* **EmbeddingService**:

  * Lazy-loads model on first use
  * Caches model in memory
  * Batch size: ≤100 for optimal performance
  * Latency: ~50ms per batch

* **FAISSIndexManager**:

  * Per-domain indices for isolation
  * L2-normalized vectors (IndexFlatIP)
  * Search latency: <10ms P50
  * Memory: ~4KB per 1000 vectors

* **CircuitBreaker**:

  * States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
  * Opens after N consecutive failures
  * Half-opens after timeout to test recovery
  * Thread-safe

* **RateLimiter**:

  * Token-bucket algorithm
  * Per-provider and per-domain limits
  * Configurable backpressure (queue vs reject)
  * Metrics exported
