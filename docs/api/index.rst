API Reference
=============

This section contains the complete API reference for all ACE Playbook modules.

.. toctree::
   :maxdepth: 2
   :caption: API Modules:

   generator
   reflector
   curator
   models
   repositories
   utils
   ops
   runner

Overview
--------

The ACE Playbook API is organized into the following major components:

Core Pipeline Components
~~~~~~~~~~~~~~~~~~~~~~~~

* :doc:`generator` - Task execution with Chain-of-Thought reasoning
* :doc:`reflector` - Execution feedback analysis and insight extraction
* :doc:`curator` - Semantic deduplication and playbook management

Data Models
~~~~~~~~~~~

* :doc:`models` - Pydantic models for all data structures

Data Access
~~~~~~~~~~~

* :doc:`repositories` - Database access layer for playbooks and journals

Utilities
~~~~~~~~~

* :doc:`utils` - Embeddings, FAISS indexing, circuit breakers, rate limiting
* :doc:`ops` - Observability, metrics, tracing, and guardrails

Workflow Runners
~~~~~~~~~~~~~~~~

* :doc:`runner` - Offline training and online learning loops

Module Conventions
------------------

All public APIs follow these conventions:

**Type Hints**: All public functions and methods include type hints

**Docstrings**: Google-style docstrings with:

* Purpose and context
* Args with types and descriptions
* Returns with type and description
* Raises with exception types and conditions
* Example usage
* Notes about thread-safety, performance, or architectural context

**Error Handling**:

* ``ValueError`` for invalid inputs
* ``RuntimeError`` for system failures
* Custom exceptions (``DomainIsolationError``, ``DeduplicationError``) for domain logic

**Thread Safety**: All repository methods and curator operations are thread-safe with atomic transactions.
