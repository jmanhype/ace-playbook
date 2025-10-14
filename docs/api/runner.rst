Runner Module
=============

Workflow orchestration for offline training and online learning loops.

.. automodule:: ace.runner
   :members:
   :undoc-members:
   :show-inheritance:

OfflineTrainer
--------------

Batch processing for dataset pre-training.

.. autoclass:: ace.runner.offline_trainer.OfflineTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

OnlineLearner
-------------

Continuous adaptation from production traffic.

.. autoclass:: ace.runner.online_learner.OnlineLearner
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Examples
--------------

Offline Training
~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.runner import OfflineTrainer

   # Initialize trainer
   trainer = OfflineTrainer(
       generator=generator,
       reflector=reflector,
       curator=curator
   )

   # Load dataset
   from datasets import load_dataset
   gsm8k = load_dataset("gsm8k", "main", split="train[:100]")

   # Train on dataset
   stats = trainer.train(
       dataset=gsm8k,
       domain_id="arithmetic",
       num_examples=100
   )

   print(f"Processed: {stats['examples_processed']}")
   print(f"Bullets added: {stats['bullets_added']}")
   print(f"Final playbook size: {stats['playbook_size']}")

Online Learning
~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.runner import OnlineLearner

   # Initialize learner
   learner = OnlineLearner(
       generator=generator,
       reflector=reflector,
       curator=curator,
       check_interval=100  # Promote every 100 tasks
   )

   # Start daemon (runs forever)
   learner.start(task_queue=queue)

   # Or process single task
   feedback = learner.process_task(
       task_input=task,
       ground_truth="42"
   )

   print(f"Insights: {len(feedback.insights)}")
   print(f"New bullets (shadow): {feedback.bullets_added}")

With Progress Tracking
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tqdm import tqdm

   # Train with progress bar
   for example in tqdm(dataset, desc="Training"):
       stats = trainer.process_example(
           task_id=example["id"],
           description=example["question"],
           ground_truth=example["answer"],
           domain_id="arithmetic"
       )

       if stats["new_bullets"] > 0:
           print(f"  → Added {stats['new_bullets']} bullets")

Notes
-----

* **OfflineTrainer**:

  * Batch processing with progress tracking
  * Automatic FAISS index rebuilds
  * Stats: examples_processed, bullets_added, dedup_rate
  * Performance: ~5s per example (including LLM calls)

* **OnlineLearner**:

  * Daemon process for continuous learning
  * Shadow mode by default (SHADOW stage)
  * Periodic promotion checks
  * Graceful shutdown support

* Both support:

  * Domain isolation
  * Error handling with retry logic
  * Metrics export
  * Checkpointing for resumability
