# Benchmark Results Ledger

This ledger captures the benchmark configurations referenced in the ACE/EE paper alignment notes. Populate each table after executing the commands in the [workflow](../.github/workflows/ace-benchmark.yml) or by running the harness locally.

> **Note:** Metrics remain blank until you execute the runs. Fill the tables by copying the reported values from the JSON metrics output and the accompanying `.feedback.jsonl` logs in `results/benchmark/`.

## Finance Benchmarks

| Variant | Dataset | Temperature | Accuracy (`correct/total`) | Promotions | New Bullets | Increments | Auto Corrections | Format Corrections | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `benchmarks/finance_subset.jsonl` | 0.5 | 24/26 | 0 | 58 | 25 | 4 | 3 | Guardrail normalization active - artifacts: `results/actions/18735223220/ace-benchmark-finance-baseline/` |
| ACE (GT) | `benchmarks/finance_subset.jsonl` | 0.5 | 25/26 | 3 | 6 | 1 | 5 | 3 | Reflector sees labels - artifacts: `results/actions/18735223220/ace-benchmark-finance-ace-gt/` |
| ACE (No GT) | `benchmarks/finance_subset.jsonl` | 0.5 | 26/26 | 4 | 24 | 9 | 7 | 3 | Guardrail-only evaluation - artifacts: `results/actions/18735223220/ace-benchmark-finance-ace-no-gt/` |
| Baseline | `benchmarks/finance_hard.jsonl` | 0.9 | 21/26 | 0 | 57 | 12 | 8 | 0 | Expect sharp accuracy drop - artifacts: `results/actions/18735223220/ace-benchmark-finance-hard-baseline/` |
| ACE (GT) | `benchmarks/finance_hard.jsonl` | 0.5 | 20/26 | 3 | 22 | 11 | 6 | 0 | Target paper lift - artifacts: `results/actions/18735223220/ace-benchmark-finance-hard-ace-gt/` |
| ACE (No GT) | `benchmarks/finance_hard.jsonl` | 0.5 | 20/26 | 7 | 6 | 14 | 8 | 1 | Guardrail-only evaluator - artifacts: `results/actions/18735223220/ace-benchmark-finance-hard-ace-no-gt/` |

The per-task guardrail verdicts are stored alongside each run as `<output>.feedback.jsonl`. These logs make it easy to confirm how auto-corrections or format clamps were applied.

## Agent / AppWorld Benchmarks

| Variant | Dataset | Temperature | Success | Fail | Unknown | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `benchmarks/agent_hard.jsonl` | 0.9 | 5 | 1 | 6 | Heuristics fail closed - artifacts: `results/actions/18735223220/ace-benchmark-agent-hard-baseline/` |
| ACE | `benchmarks/agent_hard.jsonl` | 0.5 | 5 | 1 | 6 | Playbook reduces failures, keeps conservative unknowns - artifacts: `results/actions/18735223220/ace-benchmark-agent-hard-ace/` |

Each agent run writes a feedback ledger (e.g., `results/benchmark/baseline_agent_hard.feedback.jsonl`) that records heuristic decisions per task. Retain these artifacts for reviewer inspection, mirroring the AppWorld appendix in the paper.

## Ablation Study

Run the finance and agent hard splits with each component disabled and track the regression relative to the full ACE stack.

| Configuration | Accuracy Delta | Promotions Delta | Notes |
| --- | --- | --- | --- |
| `ACE_ENABLE_REFLECTOR=off` | 19/26 (-1 vs ACE GT) | 4 (+1) | Reflector disabled (`results/actions/18735223220/ace-benchmark-finance-hard-ace-no-reflector/`) |
| `ACE_MULTI_EPOCH=off` | 18/26 (-2 vs ACE GT) | 4 (+1) | No multi-epoch refinement (`results/actions/18735223220/ace-benchmark-finance-hard-ace-no-multiepoch/`) |
| `ACE_OFFLINE_WARMUP=off` | 18/26 (-2 vs ACE GT) | 2 (-1) | Warmup disabled (`results/actions/18735223220/ace-benchmark-finance-hard-ace-no-warmup/`) |

Record the exact commands and timestamps next to the table once results are available.

## Guardrail & Heuristic Policy

Finance guardrails enforce exact-match scoring and automatically fall back to `unknown` when the answer cannot be normalized. Agent heuristics adopt the same fail-closed strategy: if required keywords, bullet structure, or time markers are missing, the classifier reports `unknown` rather than optimistic success. Update `benchmarks/data/agent_feedback_config.json` cautiously and rerun `scripts/audit_agent_scoring.py` after every change to confirm conservative thresholds.
