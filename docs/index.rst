ACE Playbook Documentation
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   architecture
   onboarding
   api/index
   tutorials/index
   edge_cases
   runbook
   changelog

Welcome to ACE Playbook
-----------------------

ACE Playbook is an Adaptive Code Evolution system that implements self-improving LLM capabilities through the Generator-Reflector-Curator pattern. It enables domain-specific knowledge accumulation without fine-tuning the base language model.

Key Features
------------

* **Generator-Reflector-Curator Pattern**: Modular architecture for task execution, reflection, and knowledge curation
* **Semantic Deduplication**: FAISS-powered similarity search prevents duplicate strategies
* **Append-Only Playbook**: Never rewrite content - only increment counters for stability
* **Domain Isolation**: Per-tenant namespaces with strict security boundaries
* **Staged Rollout**: SHADOW → STAGING → PROD progression with promotion gates
* **Observability**: Comprehensive metrics, tracing, and guardrail monitoring
* **Production-Ready**: Circuit breakers, rate limiting, health checks, and rollback procedures

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/ace-playbook.git
   cd ace-playbook

   # Install with uv (recommended)
   pip install uv
   uv pip install -e ".[dev]"

   # Or with pip
   pip install -e ".[dev]"

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from ace.curator import SemanticCurator, CuratorInput
   from ace.generator import CoTGenerator
   from ace.reflector import GroundedReflector

   # Initialize components
   curator = SemanticCurator()
   generator = CoTGenerator()
   reflector = GroundedReflector()

   # Execute task
   task_output = generator.forward(task_input)

   # Reflect on execution
   reflection = reflector.forward(reflector_input)

   # Curate insights into playbook
   curator_output = curator.apply_delta(curator_input)

Architecture Overview
---------------------

ACE Playbook follows a three-stage pipeline:

1. **Generator**: Executes tasks using Chain-of-Thought reasoning with playbook context
2. **Reflector**: Analyzes execution feedback to extract labeled insights (Helpful/Harmful/Neutral)
3. **Curator**: Merges insights into playbook with semantic deduplication and counter updates

All changes are append-only with SHA-256 diff journal for full auditability.

Links
-----

* :doc:`getting_started` - Quick start guide
* :doc:`architecture` - System architecture and design decisions
* :doc:`onboarding` - Developer onboarding guide
* :doc:`api/index` - Complete API reference
* :doc:`tutorials/index` - Step-by-step tutorials
* :doc:`edge_cases` - Edge cases and error handling
* :doc:`runbook` - Operations and troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
