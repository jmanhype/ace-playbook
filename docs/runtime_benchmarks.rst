Runtime Adaptation & Benchmarks
================================

Overview
--------

ACE’s runtime learning loop now combines the **RuntimeAdapter**, **MergeCoordinator**, and
**RefinementScheduler** to absorb insights while benchmarks are running:

* **RuntimeAdapter** caches fresh insights in-memory so subsequent tasks can reuse them before the curator persists changes.
* **MergeCoordinator** batches those insights per domain and hands them to the curator in a single `merge_batch` call.
* **RefinementScheduler** periodically flushes outstanding merges, applies promotion gates, and quarantines duplicates via `prune_redundant`.

This flow powers the `scripts/run_benchmark.py` harness, allowing you to compare the baseline Generator against the full ACE stack without manual orchestration.

Prerequisites
-------------

1. **API access** – set one of the following in your environment (or `.env`):

   * ``OPENROUTER_API_KEY`` (preferred, supports custom ``OPENROUTER_MODEL``)
   * ``OPENAI_API_KEY``
   * ``ANTHROPIC_API_KEY``

2. **Database** – ensure ``ace_playbook.db`` (or the URL specified by ``DATABASE_URL``) is reachable; migrations should be up-to-date.
3. **Python environment** – install project dependencies with ``uv``/``pip`` and activate the same environment when running the scripts.
4. **Optional smoke tests** – run ``pytest tests/unit/test_runtime_adapter.py`` and ``pytest tests/test_reflector.py`` after changes that touch runtime or reflector code.

Benchmark Variants
------------------

``scripts/run_benchmark.py`` supports multiple execution profiles:

=================  ===============================
Variant            Description
=================  ===============================
``baseline``       CoT generator only (no runtime adapter, no merge batches)
``ace_full``       ReAct generator + runtime adapter + merge coordinator + refinement scheduler
=================  ===============================

Add new variants by extending ``VARIANTS`` inside ``scripts/run_benchmark.py``.

Running the Harness
-------------------

.. code-block:: bash

   # Compare the full ACE stack on the finance benchmark subset
   python scripts/run_benchmark.py benchmarks/finance_subset.jsonl ace_full \
       --output results/ace_full_finance_subset.json

Positional arguments:

* ``TASKS`` – path to a newline-delimited JSON benchmark file.
* ``VARIANT`` – execution profile (see table above).

Key options:

* ``--output`` – location to write the JSON metrics summary.
* ``--max-tasks`` – (optional) limit the number of benchmark tasks for quick smoke runs.

Understanding the Output
------------------------

Each run emits a JSON document with fields such as:

* ``correct`` / ``total`` – aggregate score.
* ``promotions`` / ``quarantines`` / ``new_bullets`` / ``increments`` – curator activity.
* ``failures`` – list of remaining incorrect tasks with model answer vs. ground truth.
* ``auto_corrections`` – guardrail-driven canonical replacements (e.g., finance percent formatting).
* ``format_corrections`` – post-processor substitutions when the raw answer contained the correct token but extra phrasing.

Finance Guardrails & Auto-Correction
------------------------------------

Finance tasks leverage ``ace.utils.finance_guardrails`` to canonicalize answers. When ``auto_correct``
is enabled for a guardrail, the benchmark runner overrides the model answer with the calculator’s
output so scoring remains deterministic. Format corrections kick in afterward to strip trailing
explanations while keeping the raw answer available to the reflector for auditability.

Recommended Workflow
--------------------

1. Implement feature or guardrail updates.
2. Run targeted unit tests:

   .. code-block:: bash

      pytest tests/unit/test_runtime_adapter.py tests/test_reflector.py -q

3. Execute a focused benchmark (e.g., finance subset) to validate auto-corrections and promotions.
4. Review ``results/*.json`` before committing; delete or rename if you prefer not to track regenerated outputs.

Continuous Integration Ideas
----------------------------

* Add a ``workflow_dispatch`` GitHub Actions job that runs ``python scripts/run_benchmark.py`` with
  ``--max-tasks`` to produce a lightweight smoke signal.
* Cache embeddings or mock the LLM through environment toggles if CI cannot access external APIs.
* Surface benchmark metrics as artifacts for quick comparison between branches.

Adding a New Domain
-------------------

Use this checklist to bootstrap a fresh benchmark + guardrail setup:

1. **Harvest exemplar tasks**
   * Collect reliable input/output pairs with unambiguous ground truth.
   * Normalize them into newline-delimited JSON (``benchmarks/<domain>.jsonl``).

2. **Map failure modes**
   * Identify the mistakes you want to prevent: numeric drift, missing fields, tone violations, etc.
   * Decide whether each can be auto-corrected or only flagged.

3. **Implement guardrails**
   * Create ``ace/utils/<domain>_guardrails.py`` mirroring the finance module.
   * Define instructions, calculators/validators, format specifiers, and set ``auto_correct=True`` where appropriate.
   * Expose a ``get_guardrail(task_id)`` helper.

4. **Wire into the benchmark runner**
   * Import your guardrail getter in ``scripts/run_benchmark.py``.
   * Update ``augment_description`` and ``evaluate_answer`` to use it.
   * Add any domain-specific CLI variants if needed.

5. **Run & document**
   * Execute the harness (``ace_full`` variant) and inspect ``results/<domain>.json``.
   * Commit the benchmark, guardrails, results, and README/docs updates.
   * Optionally add a CI job or docs entry so others can rerun it.

Tip: If you expect to repeat this often, create a cookiecutter or script that scaffolds the benchmark file, guardrail module, and docs stub.
