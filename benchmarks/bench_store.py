"""Benchmark: store() latency at different scales."""

import os
import sys
import time
import tempfile
import statistics

from aura import Aura, Level

SCALES = [1_000, 10_000, 100_000]
TAGS_POOL = [
    ["python", "api"], ["rust", "core"], ["deploy", "ci"],
    ["user", "preference"], ["bug", "fix"], ["design", "ui"],
    ["database", "query"], ["auth", "security"], ["test", "qa"],
    ["docs", "readme"],
]
LEVELS = [Level.Working, Level.Decisions, Level.Domain, Level.Identity]


def make_content(i: int) -> str:
    return f"Memory record number {i}: the quick brown fox jumps over the lazy dog with context about topic {i % 50}"


def bench_store(n: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        brain = Aura(os.path.join(tmp, "bench.db"))
        latencies = []

        for i in range(n):
            content = make_content(i)
            tags = TAGS_POOL[i % len(TAGS_POOL)]
            level = LEVELS[i % len(LEVELS)]

            t0 = time.perf_counter()
            brain.store(content, level=level, tags=tags, deduplicate=False)
            elapsed = (time.perf_counter() - t0) * 1000  # ms

            latencies.append(elapsed)

        brain.close()

    return {
        "n": n,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(n * 0.95)],
        "p99_ms": sorted(latencies)[int(n * 0.99)],
        "total_s": sum(latencies) / 1000,
    }


def main():
    scales = SCALES
    if len(sys.argv) > 1:
        scales = [int(x) for x in sys.argv[1:]]

    print("=" * 65)
    print("AuraSDK Benchmark: store()")
    print("=" * 65)

    for n in scales:
        print(f"\n--- {n:,} records ---")
        result = bench_store(n)
        print(f"  Mean:   {result['mean_ms']:.3f} ms/op")
        print(f"  Median: {result['median_ms']:.3f} ms/op")
        print(f"  P95:    {result['p95_ms']:.3f} ms/op")
        print(f"  P99:    {result['p99_ms']:.3f} ms/op")
        print(f"  Total:  {result['total_s']:.2f} s")


if __name__ == "__main__":
    main()
