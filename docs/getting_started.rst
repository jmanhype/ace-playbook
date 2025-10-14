Getting Started
===============

This guide will help you set up ACE Playbook and run your first self-improving LLM workflow.

Prerequisites
-------------

* Python 3.11 or higher
* OpenAI or Anthropic API key
* 8GB RAM minimum (16GB recommended)
* Linux, macOS, or Windows with WSL2

Installation
------------

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

`uv <https://github.com/astral-sh/uv>`_ is the fastest Python package installer:

.. code-block:: bash

   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Clone repository
   git clone https://github.com/yourusername/ace-playbook.git
   cd ace-playbook

   # Install with dev dependencies
   uv pip install -e ".[dev]"

   # Set up pre-commit hooks
   pre-commit install

Using pip
~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/ace-playbook.git
   cd ace-playbook

   # Create virtual environment
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install
   pip install -e ".[dev]"

Configuration
-------------

Set up environment variables for LLM access:

.. code-block:: bash

   # Create .env file
   cat > .env <<EOF
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   EOF

Or export directly:

.. code-block:: bash

   export OPENAI_API_KEY=your_openai_api_key
   export ANTHROPIC_API_KEY=your_anthropic_api_key

Quick Start Example
-------------------

Here's a complete example of the Generator-Reflector-Curator pattern:

.. code-block:: python

   from ace.generator import CoTGenerator
   from ace.reflector import GroundedReflector
   from ace.curator import SemanticCurator, CuratorInput
   from ace.repositories import PlaybookRepository
   from ace.utils import EmbeddingService, FAISSIndexManager
   from ace.models.playbook import (
       TaskInput, ExecutionFeedback, InsightCandidate, PlaybookStage
   )

   # Initialize components
   embedding_service = EmbeddingService()
   faiss_manager = FAISSIndexManager(dimension=384)
   repo = PlaybookRepository(db_url="sqlite:///playbook.db")

   generator = CoTGenerator(model="gpt-4")
   reflector = GroundedReflector(model="gpt-4")
   curator = SemanticCurator(
       embedding_service=embedding_service,
       faiss_manager=faiss_manager
   )

   # Execute task
   task_input = TaskInput(
       task_id="task-001",
       description="Calculate 15 + 27",
       domain="arithmetic",
       playbook_bullets=[]  # Empty initially
   )

   task_output = generator.forward(task_input)

   print(f"Answer: {task_output.answer}")
   print(f"Reasoning trace: {task_output.reasoning_trace}")

   # Reflect with feedback
   feedback = ExecutionFeedback(
       task_id="task-001",
       ground_truth="42",  # Correct answer
       test_results={"basic_arithmetic": True}
   )

   reflection = reflector.forward(task_output, feedback)

   print(f"Insights generated: {len(reflection.insights)}")
   for insight in reflection.insights:
       print(f"  - [{insight.section}] {insight.content}")

   # Curate insights into playbook
   curator_input = CuratorInput(
       task_id="task-001",
       domain_id="arithmetic",
       insights=reflection.insights,
       current_playbook=repo.get_all("arithmetic"),
       similarity_threshold=0.8
   )

   curator_output = curator.apply_delta(curator_input)

   print(f"Bullets added: {curator_output.stats['new_bullets']}")
   print(f"Dedup rate: {curator_output.stats['dedup_rate']:.2%}")

   # Persist to database
   for bullet in curator_output.updated_playbook:
       repo.create(bullet)

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Unit tests only
   pytest tests/unit -v

   # Integration tests only
   pytest tests/integration -v

   # With coverage report
   pytest --cov=ace --cov-report=html

   # View coverage
   open htmlcov/index.html  # macOS
   xdg-open htmlcov/index.html  # Linux

Code Quality
------------

.. code-block:: bash

   # Format code
   black ace/ tests/

   # Lint
   ruff check ace/ tests/

   # Type checking
   mypy ace/

   # Complexity analysis
   radon cc ace/ -a -nb

   # Run all checks (what CI runs)
   make ci

Using Make
----------

The project includes a Makefile with convenient targets:

.. code-block:: bash

   make help              # Show all available commands
   make test              # Run all tests
   make test-cov          # Run tests with coverage
   make format            # Format code with black
   make lint              # Run linters
   make security          # Security scans
   make complexity        # Check code complexity
   make mutation-test     # Run mutation tests
   make clean             # Clean temporary files
   make ci                # Run full CI pipeline locally

Next Steps
----------

* :doc:`architecture` - Understand the system design
* :doc:`tutorials/01-quick-start` - Complete beginner tutorial
* :doc:`tutorials/02-offline-training` - Pre-train on datasets
* :doc:`tutorials/03-domain-isolation` - Multi-tenant setup
* :doc:`api/index` - Explore the full API reference
* :doc:`onboarding` - Contribute to development

Common Issues
-------------

**ModuleNotFoundError: No module named 'ace'**

Make sure you installed in editable mode:

.. code-block:: bash

   pip install -e ".[dev]"

**OpenAI API errors**

Verify your API key is set:

.. code-block:: bash

   echo $OPENAI_API_KEY

**FAISS dimension mismatch**

Embeddings must be 384-dimensional for the default model. If using a different embedding model, update the dimension parameter:

.. code-block:: python

   faiss_manager = FAISSIndexManager(dimension=768)  # For larger models

Support
-------

* GitHub Issues: https://github.com/yourusername/ace-playbook/issues
* Documentation: https://ace-playbook.readthedocs.io/
* Examples: ``examples/`` directory in repository
