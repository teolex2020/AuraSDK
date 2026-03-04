"""Benchmark: run_maintenance() cycle at different scales."""

import os
import sys
import time
import tempfile
import statistics

from aura import Aura, Level

TAGS_POOL = [
    ["python", "api"], ["rust", "core"], ["deploy", "ci"],
    ["user", "preference"], ["bug", "fix"], ["design", "ui"],
    ["database", "query"], ["auth", "security"], ["test", "qa"],
    ["docs", "readme"],
]
LEVELS = [Level.Working, Level.Decisions, Level.Domain, Level.Identity]


def populate(brain: Aura, n: int):
    for i in range(n):
        content = f"Memory record {i}: context about topic {i % 50} with enough text for meaningful processing"
        tags = TAGS_POOL[i % len(TAGS_POOL)]
        level = LEVELS[i % len(LEVELS)]
        brain.store(content, level=level, tags=tags, deduplicate=False)


def bench_maintenance(n: int, cycles: int = 10) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        brain = Aura(os.path.join(tmp, "bench.db"))
        populate(brain, n)

        latencies = []
        for _ in range(cycles):
            t0 = time.perf_counter()
            report = brain.run_maintenance()
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        brain.close()

    return {
        "n": n,
        "cycles": cycles,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }


def main():
    scales = [1_000, 10_000]
    if len(sys.argv) > 1:
        scales = [int(x) for x in sys.argv[1:]]

    print("=" * 65)
    print("AuraSDK Benchmark: run_maintenance()")
    print("=" * 65)

    for n in scales:
        print(f"\n--- {n:,} records, 10 cycles ---")
        print(f"  Running...", end=" ", flush=True)

        result = bench_maintenance(n)
        print("done")

        print(f"  Mean:   {result['mean_ms']:.2f} ms/cycle")
        print(f"  Median: {result['median_ms']:.2f} ms/cycle")
        print(f"  Min:    {result['min_ms']:.2f} ms/cycle")
        print(f"  Max:    {result['max_ms']:.2f} ms/cycle")


if __name__ == "__main__":
    main()
