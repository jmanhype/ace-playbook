# ACE + Agent Learning Release Notes

## v1.0.0 – Unified ACE + Early Experience Stack

**Date:** 2025-10-22

This release introduces the first cohesive distribution that bundles the
Agentic Context Engineering (ACE) runtime with the Agent Learning “Early
Experience” (EE) live loop inside a single repository. Highlights:

- **DSPy-backed JSON workflow** – `ace.llm_client.DSPyLLMClient` wraps any
  configured `dspy.LM`, normalises responses (including fenced JSON), and
  enforces schema validation across the generator, reflector, and curator.
- **Configurable live loop** – `examples/live_loop_quickstart.py` now toggles
  between deterministic dummy runs and real DSPy backends via `--backend`.
- **Benchmark harness** – `benchmarks/run_live_loop_benchmark.py` compares
  baseline and ACE-enabled runs, saving reproducible summaries to
  `results/benchmark/live_loop_benchmark.json`.
- **Documentation refresh** – `docs/combined_quickstart.md` shows how to
  configure DSPy with OpenRouter (or any provider), run the quick-start, and
  scale to production datasets.
- **Regression coverage** – Expanded unit tests guard the JSON normaliser and
  dummy client safety rails.

### Upgrade guide

1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```
2. Configure DSPy once per session:
   ```python
   import dspy, os
   dspy.configure(
       lm=dspy.LM(
           "openrouter/openai/gpt-4.1-mini",
           api_key=os.environ["OPENROUTER_API_KEY"],
           api_base="https://openrouter.ai/api/v1",
       )
   )
   ```
3. Run the live loop:
   ```bash
   python examples/live_loop_quickstart.py --backend dspy --episodes 10
   ```
4. Capture lift numbers:
   ```bash
   python benchmarks/run_live_loop_benchmark.py --backend dspy --episodes 10
   ```

Tag `v1.0.0` marks the integration point referenced in the ACE and Agent
Learning manuscripts. Subsequent releases will extend the benchmark suite and
add dataset adapters for larger SWE workloads.
