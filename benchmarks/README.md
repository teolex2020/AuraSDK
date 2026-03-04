# AuraSDK Benchmarks

Reproducible performance benchmarks for AuraSDK.

## Quick Run

```bash
# Full summary (1,000 records, ~10 seconds)
python benchmarks/bench_all.py

# Custom scale
python benchmarks/bench_all.py 10000
```

## Individual Benchmarks

```bash
# Store latency at 1K, 10K, 100K records
python benchmarks/bench_store.py

# Recall latency: cold / warm / cached
python benchmarks/bench_recall.py

# Maintenance cycle performance
python benchmarks/bench_maintenance.py
```

## Output

`bench_all.py` saves results to `benchmarks/results.json` for CI tracking.

## Reference Numbers

Benchmarked on Windows 10 / Ryzen 7 with 1,000 records:

| Operation | Latency |
|-----------|---------|
| Store | ~0.09 ms/op |
| Recall (structured) | ~0.74 ms/op |
| Recall (cached) | ~0.48 us/op |
| Maintenance cycle | ~1.1 ms/cycle |
