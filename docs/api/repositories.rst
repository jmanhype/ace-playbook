Repositories Module
===================

Data access layer for playbooks and diff journal with SQLite/SQLAlchemy.

.. automodule:: ace.repositories
   :members:
   :undoc-members:
   :show-inheritance:

PlaybookRepository
------------------

CRUD operations for playbook bullets with domain isolation.

.. autoclass:: ace.repositories.playbook_repository.PlaybookRepository
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

JournalRepository
-----------------

Append-only audit trail for all playbook changes.

.. autoclass:: ace.repositories.journal_repository.JournalRepository
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Playbook Operations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.repositories import PlaybookRepository
   from ace.models.playbook import PlaybookBullet, PlaybookStage

   # Initialize repository
   repo = PlaybookRepository(db_url="sqlite:///playbook.db")

   # Create bullet
   bullet = PlaybookBullet(...)
   repo.create(bullet)

   # Retrieve top-K by recency/relevance
   bullets = repo.get_top_k(
       domain_id="arithmetic",
       k=40,
       stage=PlaybookStage.PROD  # Only production bullets
   )

   # Update counters (atomic transaction)
   repo.increment_helpful(bullet_id="bullet-001")
   repo.increment_harmful(bullet_id="bullet-002")

   # Get by section
   helpful = repo.get_by_section(
       domain_id="arithmetic",
       section="Helpful"
   )

Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Batch insert with transaction
   bullets = [bullet1, bullet2, bullet3]
   repo.batch_create(bullets)

   # Batch update counters
   increments = [
       ("bullet-001", "helpful"),
       ("bullet-002", "helpful"),
       ("bullet-003", "harmful")
   ]
   repo.batch_increment(increments)

Journal Operations
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.repositories import JournalRepository
   from ace.models.playbook import DiffJournalEntry

   # Initialize journal
   journal = JournalRepository(db_url="sqlite:///playbook.db")

   # Create entry
   entry = DiffJournalEntry(
       bullet_id="bullet-001",
       operation="increment_helpful",
       before_hash="abc123...",
       after_hash="def456...",
       task_id="task-001"
   )
   journal.append(entry)

   # Get history for bullet
   history = journal.get_history(bullet_id="bullet-001")

   for entry in history:
       print(f"{entry.timestamp}: {entry.operation}")
       print(f"  Before: {entry.before_hash}")
       print(f"  After: {entry.after_hash}")

   # Get all changes for task
   task_changes = journal.get_by_task(task_id="task-001")

Notes
-----

* **Thread Safety**: All methods use SQLAlchemy sessions with atomic transactions
* **Domain Isolation**: All queries filtered by domain_id, cross-domain raises ValueError
* **Indexes**: Optimized for queries on (domain_id, helpful_count, last_used_at)
* **WAL Mode**: SQLite configured with WAL for concurrent reads
* **Connection Pooling**: Configurable pool size for high-concurrency scenarios
* **Performance**:

  * get_top_k: <5ms P50
  * increment_*: <10ms P50
  * batch operations: ~2ms per item
