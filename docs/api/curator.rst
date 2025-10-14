Curator Module
==============

The Curator module manages semantic deduplication and playbook updates with append-only guarantees.

.. automodule:: ace.curator
   :members:
   :undoc-members:
   :show-inheritance:

SemanticCurator
---------------

Core curation engine with FAISS-powered similarity search.

.. autoclass:: ace.curator.semantic_curator.SemanticCurator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

CuratorService
--------------

High-level service layer for curation operations.

.. autoclass:: ace.curator.curator_service.CuratorService
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

DomainValidator
---------------

Domain isolation and validation logic.

.. autoclass:: ace.curator.domain_validator.DomainValidator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PromotionPolicy
---------------

Staged rollout and promotion gate logic.

.. autoclass:: ace.curator.promotion_policy.PromotionPolicy
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Curator Models
--------------

.. automodule:: ace.curator.curator_models
   :members:
   :undoc-members:
   :show-inheritance:

Curator Utilities
-----------------

.. automodule:: ace.curator.curator_utils
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Curation
~~~~~~~~~~~~~~

.. code-block:: python

   from ace.curator import SemanticCurator, CuratorInput
   from ace.models.playbook import InsightCandidate

   # Initialize curator
   curator = SemanticCurator(similarity_threshold=0.8)

   # Create insights from Reflector
   insights = [
       InsightCandidate(
           content="Break problems into smaller steps",
           section="Helpful",
           confidence=0.9,
           rationale="Led to correct answer"
       ),
       InsightCandidate(
           content="Always validate input bounds",
           section="Helpful",
           confidence=0.85,
           rationale="Prevented edge case errors"
       )
   ]

   # Create curator input
   curator_input = CuratorInput(
       task_id="task-001",
       domain_id="arithmetic",
       insights=insights,
       current_playbook=current_bullets,
       similarity_threshold=0.8
   )

   # Apply delta
   output = curator.apply_delta(curator_input)

   print(f"New bullets: {output.stats['new_bullets']}")
   print(f"Increments: {output.stats['increments']}")
   print(f"Dedup rate: {output.stats['dedup_rate']}")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.curator import CuratorService

   # Initialize service with batch support
   service = CuratorService()

   # Batch merge multiple tasks
   task_insights = [
       ("task-001", domain, insights1),
       ("task-002", domain, insights2),
       ("task-003", domain, insights3)
   ]

   output = service.batch_merge(task_insights)

   print(f"Processed {len(task_insights)} tasks")
   print(f"Total bullets added: {output.total_new_bullets}")

Shadow Learning
~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.models.playbook import PlaybookStage

   # Insights start in SHADOW stage by default
   output = curator.apply_delta(curator_input)

   # Retrieve only SHADOW bullets for monitoring
   shadow_bullets = repo.get_by_stage(
       domain_id="arithmetic",
       stage=PlaybookStage.SHADOW
   )

   # Check promotion eligibility
   for bullet in shadow_bullets:
       if bullet.helpful_count >= 3 and \
          bullet.helpful_count / max(bullet.harmful_count, 1) >= 3:
           # Promote to STAGING
           service.promote_bullet(bullet.id, PlaybookStage.STAGING)

Quarantine Management
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check for harmful bullets
   bullets = repo.get_all(domain_id="arithmetic")

   for bullet in bullets:
       if bullet.harmful_count >= bullet.helpful_count and \
          bullet.helpful_count > 0:
           # Quarantine harmful strategy
           service.quarantine_bullet(bullet.id)
           print(f"Quarantined: {bullet.content}")

Notes
-----

* **Semantic Deduplication**: Cosine similarity ≥0.8 → increment counter (not add new bullet)
* **Append-Only**: Never rewrite content, only increment helpful/harmful counters
* **Domain Isolation**: Strict per-tenant boundaries, cross-domain access raises ValueError
* **Staged Rollout**: SHADOW → STAGING → PROD with promotion gates
* **Thread Safety**: All operations use atomic transactions
* **Performance**:

  * apply_delta: ~50ms P50
  * batch_merge: ~200ms for 10 tasks
  * FAISS search: <10ms P50

* **Audit Trail**: All changes logged to diff_journal with SHA-256 hashes
