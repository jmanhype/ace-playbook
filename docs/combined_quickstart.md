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

## CI Benchmarks

The repository includes a GitHub Actions workflow
(`.github/workflows/ace-benchmark.yml`) that runs the finance and agent
benchmarks under several configurations:

- Finance baseline vs ACE (with ground-truth feedback),
- Finance ACE with ground-truth disabled (reflector relies solely on execution
  cues),
- Agent baseline vs ACE on `benchmarks/agent_small.jsonl` at a higher generator
  temperature.

Each matrix entry runs in an isolated job, initialises a fresh SQLite schema,
and uploads its metrics JSON as an artifact (for example,
`ace-benchmark-finance-ace-no-gt`).

### Triggering the workflow

1. Store `OPENROUTER_API_KEY` (or another provider key) as a repository secret.
2. From the **Actions** tab, choose **ACE Benchmark → Run workflow** (manual) or
   rely on the automatic trigger for pushes to `main`.
3. After the run completes, download the artifacts. You’ll find:
   - `baseline_finance.json`, `ace_finance_gt.json`,
     `ace_finance_no_gt.json`,
   - `baseline_agent.json`, `ace_agent.json`.

### Environment knobs

`scripts/run_benchmark.py` respects the following environment variables (also
used by the workflow matrix):

- `ACE_BENCHMARK_TEMPERATURE` – overrides the generator temperature for both CoT
  and ReAct variants.
- `ACE_BENCHMARK_USE_GROUND_TRUTH` – set to `false`/`0`/`off` to withhold
  ground-truth answers from the reflector (accuracy is still evaluated against
  ground truth).

Example local invocation:

```bash
ACE_BENCHMARK_TEMPERATURE=0.5 \
ACE_BENCHMARK_USE_GROUND_TRUTH=false \
python scripts/run_benchmark.py benchmarks/finance_subset.jsonl ace_full \
  --output results/benchmark/ace_finance_no_gt.json
```

The resulting JSON files provide the raw evidence (accuracy, promotions,
increments, auto-format corrections) that mirrors the tables in the ACE paper.
