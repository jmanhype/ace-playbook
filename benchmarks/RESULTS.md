# Benchmark Results Ledger

This ledger captures the benchmark configurations referenced in the ACE/EE paper alignment notes. Populate each table after executing the commands in the [workflow](../.github/workflows/ace-benchmark.yml) or by running the harness locally.

> **Note:** Metrics remain blank until you execute the runs. Fill the tables by copying the reported values from the JSON metrics output and the accompanying `.feedback.jsonl` logs in `results/benchmark/`.

## Finance Benchmarks

| Variant | Dataset | Temperature | Accuracy (`correct/total`) | Promotions | New Bullets | Increments | Auto Corrections | Format Corrections | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `benchmarks/finance_subset.jsonl` | 0.5 | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | Guardrail normalization active |
| ACE (GT) | `benchmarks/finance_subset.jsonl` | 0.5 | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | Reflector sees labels |
| ACE (No GT) | `benchmarks/finance_subset.jsonl` | 0.5 | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | Guardrail-only evaluation |
| Baseline | `benchmarks/finance_hard.jsonl` | 0.9 | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | Expect sharp accuracy drop |
| ACE (GT) | `benchmarks/finance_hard.jsonl` | 0.5 | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | Target paper lift |
| ACE (No GT) | `benchmarks/finance_hard.jsonl` | 0.5 | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | _pending run_ | Guardrail-only evaluator |

The per-task guardrail verdicts are stored alongside each run as `<output>.feedback.jsonl`. These logs make it easy to confirm how auto-corrections or format clamps were applied.

## Agent / AppWorld Benchmarks

| Variant | Dataset | Temperature | Success | Fail | Unknown | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `benchmarks/agent_hard.jsonl` | 0.9 | _pending run_ | _pending run_ | _pending run_ | Heuristics expected to fail closed |
| ACE | `benchmarks/agent_hard.jsonl` | 0.5 | _pending run_ | _pending run_ | _pending run_ | Should promote more structured bullets |

Each agent run writes a feedback ledger (e.g., `results/benchmark/baseline_agent_hard.feedback.jsonl`) that records heuristic decisions per task. Retain these artifacts for reviewer inspection, mirroring the AppWorld appendix in the paper.

## Ablation Study

Run the finance and agent hard splits with each component disabled and track the regression relative to the full ACE stack.

| Configuration | Accuracy Delta | Promotions Delta | Notes |
| --- | --- | --- | --- |
| `ACE_ENABLE_REFLECTOR=off` | _pending run_ | _pending run_ | Reflector disabled |
| `ACE_MULTI_EPOCH=off` | _pending run_ | _pending run_ | No multi-epoch refinement |
| `ACE_OFFLINE_WARMUP=off` | _pending run_ | _pending run_ | Warmup disabled |

Record the exact commands and timestamps next to the table once results are available.

## Guardrail & Heuristic Policy

Finance guardrails enforce exact-match scoring and automatically fall back to `unknown` when the answer cannot be normalized. Agent heuristics adopt the same fail-closed strategy: if required keywords, bullet structure, or time markers are missing, the classifier reports `unknown` rather than optimistic success. Update `benchmarks/data/agent_feedback_config.json` cautiously and rerun `scripts/audit_agent_scoring.py` after every change to confirm conservative thresholds.
