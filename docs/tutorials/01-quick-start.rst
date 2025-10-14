Tutorial 1: Quick Start
=======================

**Time**: 10 minutes

**Goal**: Execute your first task through the complete Generator-Reflector-Curator pipeline.

What You'll Learn
-----------------

* How to execute a task with the Generator
* How to reflect on execution feedback
* How to curate insights into your playbook
* How to verify playbook growth

Prerequisites
-------------

* ACE Playbook installed
* API keys configured
* Empty SQLite database

Step 1: Initialize Components
------------------------------

Create a Python script ``quick_start.py``:

.. code-block:: python

   from ace.generator import CoTGenerator
   from ace.reflector import GroundedReflector
   from ace.curator import SemanticCurator
   from ace.repositories import PlaybookRepository
   from ace.utils import EmbeddingService, FAISSIndexManager
   from ace.models.playbook import (
       TaskInput, ExecutionFeedback, InsightCandidate
   )

   # Initialize services
   embedding_service = EmbeddingService()
   faiss_manager = FAISSIndexManager(dimension=384)
   repo = PlaybookRepository(db_url="sqlite:///playbook.db")

   # Initialize pipeline components
   generator = CoTGenerator(model="gpt-4")
   reflector = GroundedReflector(model="gpt-4")
   curator = SemanticCurator(
       embedding_service=embedding_service,
       faiss_manager=faiss_manager
   )

Step 2: Execute Task
--------------------

.. code-block:: python

   # Create task
   task_input = TaskInput(
       task_id="task-001",
       description="What is 15 + 27?",
       domain="arithmetic",
       playbook_bullets=[]  # Empty initially
   )

   # Execute
   print("Executing task...")
   task_output = generator.forward(task_input)

   print(f"Answer: {task_output.answer}")
   print(f"Confidence: {task_output.confidence}")
   print(f"Reasoning trace:")
   for step in task_output.reasoning_trace:
       print(f"  - {step}")

**Expected Output**::

   Executing task...
   Answer: 42
   Confidence: 0.95
   Reasoning trace:
     - Step 1: Identify operation (addition)
     - Step 2: Add numbers: 15 + 27
     - Step 3: Calculate result: 42

Step 3: Reflect on Feedback
----------------------------

.. code-block:: python

   # Create feedback with ground truth
   feedback = ExecutionFeedback(
       task_id="task-001",
       ground_truth="42",  # Correct answer
       test_results={"basic_arithmetic": True}
   )

   # Reflect
   print("\nReflecting on execution...")
   reflection = reflector.forward(task_output, feedback)

   print(f"Insights extracted: {len(reflection.insights)}")
   for insight in reflection.insights:
       print(f"  [{insight.section}] {insight.content}")
       print(f"    Confidence: {insight.confidence:.2f}")
       print(f"    Rationale: {insight.rationale}")

**Expected Output**::

   Reflecting on execution...
   Insights extracted: 2
     [Helpful] Break complex problems into clear steps
       Confidence: 0.90
       Rationale: Led to correct answer
     [Helpful] Show intermediate calculations explicitly
       Confidence: 0.85
       Rationale: Made reasoning transparent

Step 4: Curate Insights
------------------------

.. code-block:: python

   from ace.curator.curator_models import CuratorInput

   # Get current playbook (empty first time)
   current_playbook = repo.get_all(domain_id="arithmetic")

   # Curate
   print("\nCurating insights...")
   curator_output = curator.apply_delta(CuratorInput(
       task_id="task-001",
       domain_id="arithmetic",
       insights=reflection.insights,
       current_playbook=current_playbook,
       similarity_threshold=0.8
   ))

   print(f"New bullets added: {curator_output.stats['new_bullets']}")
   print(f"Counters incremented: {curator_output.stats['increments']}")
   print(f"Dedup rate: {curator_output.stats['dedup_rate']:.2%}")

**Expected Output**::

   Curating insights...
   New bullets added: 2
   Counters incremented: 0
   Dedup rate: 0.00%

Step 5: Persist to Database
----------------------------

.. code-block:: python

   # Save bullets
   print("\nPersisting to database...")
   for bullet in curator_output.updated_playbook:
       repo.create(bullet)

   # Verify
   all_bullets = repo.get_all(domain_id="arithmetic")
   print(f"Total bullets in playbook: {len(all_bullets)}")

   for bullet in all_bullets:
       print(f"  - {bullet.content}")
       print(f"    Section: {bullet.section}")
       print(f"    Helpful: {bullet.helpful_count}")
       print(f"    Stage: {bullet.stage}")

**Expected Output**::

   Persisting to database...
   Total bullets in playbook: 2
     - Break complex problems into clear steps
       Section: Helpful
       Helpful: 1
       Stage: SHADOW
     - Show intermediate calculations explicitly
       Section: Helpful
       Helpful: 1
       Stage: SHADOW

Step 6: Run Second Task (With Playbook Context)
------------------------------------------------

.. code-block:: python

   # Retrieve playbook for context injection
   bullets = repo.get_top_k(domain_id="arithmetic", k=40)

   # New task
   task_input_2 = TaskInput(
       task_id="task-002",
       description="What is 35 * 8?",
       domain="arithmetic",
       playbook_bullets=[b.content for b in bullets]  # Inject context
   )

   # Execute with playbook
   print("\nExecuting second task with playbook...")
   task_output_2 = generator.forward(task_input_2)

   print(f"Answer: {task_output_2.answer}")
   print(f"Bullets referenced: {task_output_2.bullets_referenced}")

**Expected Output**::

   Executing second task with playbook...
   Answer: 280
   Bullets referenced: ['bullet-abc123', 'bullet-def456']

Complete Script
---------------

Download the complete script: `examples/quick_start.py <../../examples/quick_start.py>`_

Run it:

.. code-block:: bash

   python examples/quick_start.py

Next Steps
----------

* :doc:`02-offline-training` - Pre-train on datasets
* :doc:`03-domain-isolation` - Multi-tenant setup
* :doc:`../api/index` - Explore the full API

What You Learned
----------------

 How to execute tasks with the Generator

 How to extract insights with the Reflector

 How to deduplicate and curate with the Curator

 How playbook context improves task execution

 How insights start in SHADOW stage for safety
