Models Module
=============

Pydantic models for all ACE Playbook data structures with validation and serialization.

.. automodule:: ace.models
   :members:
   :undoc-members:
   :show-inheritance:

Playbook Models
---------------

.. automodule:: ace.models.playbook
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
~~~~~~~~~~~

**PlaybookBullet**: Individual strategy with counters and metadata

* ``id``: Unique identifier
* ``domain_id``: Tenant namespace
* ``content``: Strategy text (10-500 chars)
* ``section``: Helpful/Harmful/Neutral
* ``helpful_count``: Success counter
* ``harmful_count``: Failure counter
* ``embedding``: 384-dim semantic vector
* ``stage``: SHADOW/STAGING/PROD
* ``created_at``, ``last_used_at``: Timestamps

**Playbook**: Collection of bullets with validation

* ``bullets``: List[PlaybookBullet] (max 300)
* ``bullet_count``: Computed property
* Methods: ``add_bullet()``, ``get_bullet()``, ``get_by_section()``

**ExecutionFeedback**: Signals for Reflector labeling

* ``ground_truth``: Expected answer (optional)
* ``test_results``: Dict[str, bool] (optional)
* ``errors``: List[str] (optional)
* ``performance_metrics``: Dict (optional)

Usage Examples
--------------

Creating Bullets
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.models.playbook import PlaybookBullet, PlaybookStage
   from datetime import datetime

   bullet = PlaybookBullet(
       id="bullet-001",
       domain_id="arithmetic",
       content="Break complex problems into smaller steps",
       section="Helpful",
       helpful_count=5,
       harmful_count=0,
       embedding=[0.1] * 384,  # 384-dim vector
       stage=PlaybookStage.SHADOW,
       created_at=datetime.utcnow(),
       last_used_at=datetime.utcnow()
   )

   # Serialize to dict
   bullet_dict = bullet.model_dump()

   # Compute SHA-256 hash
   content_hash = bullet.compute_hash()

Validation
~~~~~~~~~~

.. code-block:: python

   # Content length validation
   try:
       bullet = PlaybookBullet(
           content="too short",  # < 10 chars
           ...
       )
   except ValueError as e:
       print(f"Validation error: {e}")

   # Section enum validation
   try:
       bullet = PlaybookBullet(
           section="Invalid",  # Not in enum
           ...
       )
   except ValueError as e:
       print(f"Invalid section: {e}")

Notes
-----

* All models use Pydantic v2 for validation
* Type hints enforced for all fields
* Serialization via ``model_dump()`` and ``model_dump_json()``
* Immutability enforced with ``frozen=True`` for critical fields
* Custom validators for domain-specific logic
