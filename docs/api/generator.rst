Generator Module
================

The Generator module executes tasks using Chain-of-Thought reasoning with playbook context injection.

.. automodule:: ace.generator
   :members:
   :undoc-members:
   :show-inheritance:

CoTGenerator
------------

.. autoclass:: ace.generator.cot_generator.CoTGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

ReActGenerator
--------------

.. autoclass:: ace.generator.react_generator.ReActGenerator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Generator Signatures
--------------------

.. automodule:: ace.generator.signatures
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Task Execution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.generator import CoTGenerator
   from ace.generator.signatures import TaskInput

   # Initialize generator
   generator = CoTGenerator(model="gpt-4")

   # Create task input
   task_input = TaskInput(
       task_id="task-001",
       description="Calculate the sum of 15 and 27",
       domain="arithmetic",
       playbook_bullets=[
           "Break complex problems into steps",
           "Show your work explicitly"
       ]
   )

   # Execute task
   output = generator.forward(task_input)

   print(f"Answer: {output.answer}")
   print(f"Reasoning trace: {output.reasoning_trace}")
   print(f"Confidence: {output.confidence}")

With Playbook Context
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.repositories import PlaybookRepository

   # Retrieve top-K strategies from playbook
   repo = PlaybookRepository()
   bullets = repo.get_top_k(domain_id="arithmetic", k=40)

   # Inject into task context
   task_input = TaskInput(
       task_id="task-002",
       description="Solve: What is 15% of 240?",
       domain="arithmetic",
       playbook_bullets=[b.content for b in bullets]
   )

   output = generator.forward(task_input)

Notes
-----

* Generator maintains reasoning trace history for Reflector analysis
* All strategies referenced in reasoning are tracked in ``bullets_referenced``
* Context budget limited to 300 tokens for playbook bullets
* Performance: ~700ms P50 latency for typical tasks
