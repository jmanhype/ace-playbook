Reflector Module
================

The Reflector module analyzes execution feedback to extract labeled insights without manual annotation.

.. automodule:: ace.reflector
   :members:
   :undoc-members:
   :show-inheritance:

GroundedReflector
-----------------

.. autoclass:: ace.reflector.grounded_reflector.GroundedReflector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Reflector Signatures
---------------------

.. automodule:: ace.reflector.signatures
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Reflection with Ground Truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ace.reflector import GroundedReflector
   from ace.reflector.signatures import ReflectorInput
   from ace.models.playbook import ExecutionFeedback

   # Initialize reflector
   reflector = GroundedReflector(model="gpt-4")

   # Create feedback with ground truth
   feedback = ExecutionFeedback(
       task_id="task-001",
       answer="42",
       ground_truth="42",  # Correct answer
       test_results={"test_arithmetic": True}
   )

   # Create reflector input
   reflector_input = ReflectorInput(
       task_id="task-001",
       reasoning_trace=[
           "Step 1: Break problem into parts",
           "Step 2: Calculate each part",
           "Step 3: Sum the results"
       ],
       answer="42",
       confidence=0.95,
       bullets_referenced=["bullet-123"],
       feedback=feedback
   )

   # Extract insights
   output = reflector.forward(reflector_input)

   for insight in output.insights:
       print(f"Section: {insight.section}")  # Helpful/Harmful/Neutral
       print(f"Content: {insight.content}")
       print(f"Confidence: {insight.confidence}")
       print(f"Rationale: {insight.rationale}")

Reflection from Test Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Feedback with test failures
   feedback = ExecutionFeedback(
       task_id="task-002",
       test_results={
           "test_basic": True,
           "test_edge_case": False,  # Failed
           "test_boundary": False    # Failed
       }
   )

   reflector_input = ReflectorInput(
       task_id="task-002",
       reasoning_trace=trace,
       answer=answer,
       feedback=feedback
   )

   output = reflector.forward(reflector_input)

   # Insights will be labeled Harmful for failed test strategies
   harmful_insights = [i for i in output.insights if i.section == "Harmful"]
   helpful_insights = [i for i in output.insights if i.section == "Helpful"]

Reflection from Error Messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Feedback with execution errors
   feedback = ExecutionFeedback(
       task_id="task-003",
       errors=[
           "ValueError: Division by zero",
           "Input validation failed"
       ]
   )

   reflector_input = ReflectorInput(
       task_id="task-003",
       reasoning_trace=trace,
       answer=None,
       feedback=feedback
   )

   output = reflector.forward(reflector_input)

   # Insights will identify problematic strategies
   for insight in output.insights:
       if insight.section == "Harmful":
           print(f"Problematic strategy: {insight.content}")
           print(f"Reason: {insight.rationale}")

Notes
-----

* Reflector automatically labels insights based on feedback signals:

  * Correct answer → Helpful
  * Wrong answer → Harmful
  * Test pass → Helpful
  * Test fail → Harmful
  * Errors → Harmful with diagnostic rationale

* No manual annotation required
* Confidence scores based on feedback quality
* Analysis summary provides high-level reflection context
* Performance: ~500ms P50 latency
