# ACE + Agent Learning Quick Start

This guide walks through running the unified Adaptive Code Evolution (ACE) stack
alongside the Agent Learning “Early Experience” (EE) loop that now lives inside
this repository.

The workflow looks like this:

```
┌──────────┐     ┌────────────┐     ┌────────────┐     ┌─────────────┐
│ Tasks /  │ --> │ WorldModel │ --> │ Reflector  │ --> │ Curator      │
│ Dataset  │     │ (Generator │     │ (Insights) │     │ (Playbook Δ) │
└──────────┘     │  + Policy) │     └────────────┘     └─────────────┘
                 └────────────┘             │
                         │                  │ metrics
                         └──────────────────┘
```

The live loop draws tasks from a dataset, generates structured predictions using
the JSON-safe LLM client, evaluates metrics, then streams curated insights back
into the ACE playbook via the runtime client.

## Installation

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

The optional `dev` extra installs pytest, black, ruff, and other tooling that
covers both the original ACE modules and the new Agent Learning components.

## Running the quick-start example

A short example script lives in `examples/live_loop_quickstart.py`. It can run
with the dummy backend (deterministic canned responses) or your configured
`dspy.LM` by switching the `--backend` flag.

```bash
# Dummy backend (no network calls)
python examples/live_loop_quickstart.py

# Real backend once dspy.configure(...) has run
python examples/live_loop_quickstart.py --backend dspy --episodes 10
```

Expected output:

```
Episode math-1: accuracy=1.00 operations=0
Episode math-2: accuracy=0.00 operations=1
```

The dummy client purposely makes a mistake on the second episode so the policy
routes feedback back into the curator.  Replace the dummy client with a real
`JSONSafeLLMClient` subclass to connect OpenAI/Anthropic backends.

### Using a real LLM with DSPy

1. Configure DSPy once per process (for example, in your entrypoint) using the
   provided `.env` secrets:

   ```python
   import dspy
   import os

   dspy.configure(
       lm=dspy.LM(
           "openrouter/openai/gpt-4.1-mini",
           api_key=os.environ["OPENROUTER_API_KEY"],
           api_base="https://openrouter.ai/api/v1",
       )
   )
   ```

2. Instantiate `ace.llm_client.DSPyLLMClient()` instead of the dummy stub. This
   client delegates to the configured `dspy.LM`, and when you set
   `ACE_JSON_MODE=on` it automatically wraps calls with DSPy’s JSON adapter.

3. Run the quick-start again (or your own driver script). Any schema violations
   will surface immediately as `ace.llm_client.LLMError`, making it safe to
   promote the run to larger task batches.

4. When the real backend is active you can pass `--episodes N --temperature T`
   or a custom dataset to capture adoption metrics against your baseline runs.

## Integrating in your own project

1. **Create the runtime bridge.** Use `ace.agent_learning.create_in_memory_runtime`
   for local tests or instantiate a full `CuratorService` when running against a
   database.
2. **Instantiate the world model.** Pass a concrete LLM client (for example,
   a class that subclasses `JSONSafeLLMClient`) into
   `ace.agent_learning.WorldModel`.
3. **Pick a policy.** `ace.agent_learning.EpsilonGreedyPolicy` provides a simple
   threshold/epsilon controller; plug in your own by subclassing
   `BasePolicy`.
4. **Prepare metrics.** The helper `prepare_default_metrics()` registers the
   built-in accuracy metric.  Custom metrics can be attached via
   `MetricRegistry.register()`.
5. **Run the loop.** Construct `LiveLoop` with the pieces above and call
   `run(episodes=...)`.  The return value is a list of
   `ace.agent_learning.types.EpisodeResult` instances which include metrics and
   curator operations.

## Next steps

* Adapt `StaticTaskDataset` into a streaming dataset for production tasks.
* Swap in your real LLM backends and register additional metrics to capture
  latency or cost.
* Feed the `ExperienceBuffer` into training dashboards or analytics systems to
  monitor adoption of new playbook bullets.
