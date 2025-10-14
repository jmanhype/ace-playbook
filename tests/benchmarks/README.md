# ACE Playbook Performance Benchmarks

This directory contains pytest-benchmark performance tests that verify ACE Playbook meets its strict performance SLAs.

## Performance SLAs

- **Retrieval P50 ≤10ms** for 100 bullets
- **Retrieval P95 ≤25ms** for 100 bullets
- **Curator apply_delta P50 ≤50ms** for 10 insights

## Running Benchmarks

### Basic Run
```bash
# Run all benchmarks
pytest tests/benchmarks/ --benchmark-only --no-cov

# Run specific benchmark
pytest tests/benchmarks/test_retrieval_perf.py::test_benchmark_retrieval_100_bullets --benchmark-only --no-cov
```

### Save Baseline
```bash
# Save current performance as baseline
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline --no-cov
```

### Compare Against Baseline
```bash
# Run benchmarks and compare with saved baseline
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline --no-cov
```

### Generate HTML Reports
```bash
# Auto-save with timestamp and generate reports
pytest tests/benchmarks/ --benchmark-only --benchmark-autosave --no-cov

# View reports in .benchmarks/Darwin-CPython-3.10-64bit/
```

## Benchmark Tests

### test_benchmark_retrieval_100_bullets
Measures playbook retrieval latency with 100 bullets.
- **Current P50**: ~1.3ms (well below 10ms SLA)
- **Current P95**: ~1.6ms (well below 25ms SLA)

### test_benchmark_retrieval_stage_filter
Measures stage-filtered retrieval (PROD only).
- **Current P50**: ~0.7ms (well below 10ms SLA)

### test_benchmark_curator_apply_delta
Measures curator merge operation for 10 new insights.
- **Current P50**: ~13ms (below 50ms SLA)

### test_benchmark_multi_domain_retrieval
Measures round-robin retrieval across multiple domains (multi-tenant simulation).
- **Current P50**: ~1.2ms (well below 10ms SLA)

## Performance Results

Latest benchmark results (2025-10-13):

| Test | Min | P50 | Mean | P95 | Max | SLA | Status |
|------|-----|-----|------|-----|-----|-----|--------|
| Stage Filter Retrieval | 0.69ms | 0.72ms | 0.75ms | 0.85ms | 1.10ms | ≤10ms | ✅ Pass |
| Multi-Domain Retrieval | 1.14ms | 1.23ms | 1.26ms | 1.45ms | 5.58ms | ≤10ms | ✅ Pass |
| 100 Bullets Retrieval | 1.20ms | 1.25ms | 1.29ms | 1.45ms | 1.82ms | ≤10ms | ✅ Pass |
| Curator Apply Delta | 9.88ms | 13.26ms | 13.30ms | 16.89ms | 39.20ms | ≤50ms | ✅ Pass |

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'ace'`, set PYTHONPATH:
```bash
PYTHONPATH=/Users/speed/ace-playbook:$PYTHONPATH pytest tests/benchmarks/
```

### Python Version
The package requires Python 3.11+. If you see version errors, upgrade Python or use pyenv/conda.

### Database Conflicts
Benchmarks create temporary test databases. If you see database errors, clean up:
```bash
rm test_benchmark.db test_*.db
```

## CI/CD Integration

Add to GitHub Actions workflow:
```yaml
- name: Run performance benchmarks
  run: |
    pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-fail-at=median:10%
```

This will fail the build if performance degrades by more than 10%.
