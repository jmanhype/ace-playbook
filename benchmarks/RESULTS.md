# Benchmark Results Ledger

This ledger captures the benchmark configurations referenced in the ACE/EE paper alignment notes. Populate each table after executing the commands in the [workflow](../.github/workflows/ace-benchmark.yml) or by running the harness locally.

> **Note:** Metrics remain blank until you execute the runs. Fill the tables by copying the reported values from the JSON metrics output and the accompanying `.feedback.jsonl` logs in `results/benchmark/`.

## Finance Benchmarks

| Variant | Dataset | Temperature | Accuracy (`correct/total`) | Promotions | New Bullets | Increments | Auto Corrections | Format Corrections | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `benchmarks/finance_subset.jsonl` | 1.3 | 17/26 | 0 | 68 | 15 | 0 | 0 | Baseline intentionally unstable – artifact: `results/actions/18737669221/ace-benchmark-finance-baseline/` |
| ACE (GT) | `benchmarks/finance_subset.jsonl` | 0.5 | 26/26 | 7 | 0 | 7 | 0 | 0 | Reflector sees labels – artifact: `results/actions/18737669221/ace-benchmark-finance-ace-gt/` |
| ACE (No GT) | `benchmarks/finance_subset.jsonl` | 0.6 | 26/26 | 5 | 18 | 13 | 0 | 0 | Guardrail-only evaluation – artifact: `results/actions/18737669221/ace-benchmark-finance-ace-no-gt/` |
| Baseline | `benchmarks/finance_hard.jsonl` | 1.3 | 13/26 | 0 | 48 | 21 | 0 | 0 | Baseline deliberately noisy – artifact: `results/actions/18737669221/ace-benchmark-finance-hard-baseline/` |
| ACE (GT) | `benchmarks/finance_hard.jsonl` | 0.5 | 24/26 | 5 | 2 | 5 | 0 | 0 | Target paper lift – artifact: `results/actions/18737669221/ace-benchmark-finance-hard-ace-gt/` |
| ACE (No GT) | `benchmarks/finance_hard.jsonl` | 0.6 | 19/26 | 6 | 0 | 2 | 0 | 0 | Guardrail-only evaluator – artifact: `results/actions/18737669221/ace-benchmark-finance-hard-ace-no-gt/` |

The per-task guardrail verdicts are stored alongside each run as `<output>.feedback.jsonl`. These logs make it easy to confirm how auto-corrections or format clamps were applied.

## Agent / AppWorld Benchmarks

| Variant | Dataset | Temperature | Success | Fail | Unknown | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `benchmarks/agent_hard.jsonl` | 1.3 | 8 | 1 | 3 | Heuristics fail closed – artifact: `results/actions/18737669221/ace-benchmark-agent-hard-baseline/` |
| ACE | `benchmarks/agent_hard.jsonl` | 0.5 | 7 | 1 | 4 | Playbook preserves structure; heuristics need tuning – artifact: `results/actions/18737669221/ace-benchmark-agent-hard-ace/` |

Each agent run writes a feedback ledger (e.g., `results/benchmark/baseline_agent_hard.feedback.jsonl`) that records heuristic decisions per task. Retain these artifacts for reviewer inspection, mirroring the AppWorld appendix in the paper.

## Ablation Study

Run the finance and agent hard splits with each component disabled and track the regression relative to the full ACE stack.

| Configuration | Accuracy Delta | Promotions Delta | Notes |
| --- | --- | --- | --- |
| `ACE_ENABLE_REFLECTOR=off` | 18/26 (−6 vs ACE GT) | 0 (−5) | Reflector disabled – artifact: `results/actions/18737669221/ace-benchmark-finance-hard-ace-no-reflector/` |
| `ACE_MULTI_EPOCH=off` | 21/26 (−3 vs ACE GT) | 0 (−5) | No multi-epoch refinement – artifact: `results/actions/18737669221/ace-benchmark-finance-hard-ace-no-multiepoch/` |
| `ACE_OFFLINE_WARMUP=off` | 11/26 (−13 vs ACE GT) | 1 (−4) | Warmup disabled – artifact: `results/actions/18737669221/ace-benchmark-finance-hard-ace-no-warmup/` |

Record the exact commands and timestamps next to the table once results are available.

## Guardrail & Heuristic Policy

Finance guardrails enforce exact-match scoring and automatically fall back to `unknown` when the answer cannot be normalized. Agent heuristics adopt the same fail-closed strategy: if required keywords, bullet structure, or time markers are missing, the classifier reports `unknown` rather than optimistic success. Update `benchmarks/data/agent_feedback_config.json` cautiously and rerun `scripts/audit_agent_scoring.py` after every change to confirm conservative thresholds.
