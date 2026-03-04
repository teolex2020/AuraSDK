"""Benchmark: recall() and recall_structured() latency — cold, warm, cached."""

import os
import sys
import time
import tempfile
import statistics

from aura import Aura, Level

QUERIES = [
    "user preferences and settings",
    "deployment workflow and CI",
    "authentication security issues",
    "python programming language features",
    "database query optimization",
    "design patterns for UI components",
    "bug fix in production system",
    "rust memory safety guarantees",
    "testing and quality assurance",
    "documentation and readme updates",
]

TAGS_POOL = [
    ["python", "api"], ["rust", "core"], ["deploy", "ci"],
    ["user", "preference"], ["bug", "fix"], ["design", "ui"],
    ["database", "query"], ["auth", "security"], ["test", "qa"],
    ["docs", "readme"],
]
LEVELS = [Level.Working, Level.Decisions, Level.Domain, Level.Identity]


def populate(brain: Aura, n: int):
    for i in range(n):
        content = f"Memory record {i}: topic {i % 50} with detailed context about various subjects including programming, deployment, and user preferences"
        tags = TAGS_POOL[i % len(TAGS_POOL)]
        level = LEVELS[i % len(LEVELS)]
        brain.store(content, level=level, tags=tags, deduplicate=False)


def bench_recall(brain: Aura, n_records: int, iterations: int = 100) -> dict:
    # Cold recall (first run per query)
    cold_latencies = []
    for q in QUERIES:
        t0 = time.perf_counter()
        brain.recall_structured(q, top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000
        cold_latencies.append(elapsed)

    # Warm recall (repeated queries)
    warm_latencies = []
    for _ in range(iterations):
        q = QUERIES[_ % len(QUERIES)]
        t0 = time.perf_counter()
        brain.recall_structured(q, top_k=10)
        elapsed = (time.perf_counter() - t0) * 1000
        warm_latencies.append(elapsed)

    # Cached recall (same query repeated)
    cached_latencies = []
    q = QUERIES[0]
    brain.recall_structured(q, top_k=10)  # prime cache
    for _ in range(iterations):
        t0 = time.perf_counter()
        brain.recall(q)
        elapsed = (time.perf_counter() - t0) * 1000
        cached_latencies.append(elapsed)

    return {
        "n_records": n_records,
        "cold": {
            "mean_ms": statistics.mean(cold_latencies),
            "median_ms": statistics.median(cold_latencies),
            "max_ms": max(cold_latencies),
        },
        "warm": {
            "mean_ms": statistics.mean(warm_latencies),
            "median_ms": statistics.median(warm_latencies),
            "p95_ms": sorted(warm_latencies)[int(len(warm_latencies) * 0.95)],
        },
        "cached": {
            "mean_ms": statistics.mean(cached_latencies),
            "median_ms": statistics.median(cached_latencies),
            "min_us": min(cached_latencies) * 1000,  # microseconds
        },
    }


def main():
    scales = [1_000, 10_000]
    if len(sys.argv) > 1:
        scales = [int(x) for x in sys.argv[1:]]

    print("=" * 65)
    print("AuraSDK Benchmark: recall()")
    print("=" * 65)

    for n in scales:
        print(f"\n--- {n:,} records ---")
        print(f"  Populating...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmp:
            brain = Aura(os.path.join(tmp, "bench.db"))
            t0 = time.perf_counter()
            populate(brain, n)
            pop_time = time.perf_counter() - t0
            print(f"done ({pop_time:.1f}s)")

            result = bench_recall(brain, n)
            brain.close()

        print(f"\n  Cold recall (first hit per query):")
        print(f"    Mean:   {result['cold']['mean_ms']:.3f} ms")
        print(f"    Median: {result['cold']['median_ms']:.3f} ms")
        print(f"    Max:    {result['cold']['max_ms']:.3f} ms")

        print(f"\n  Warm recall (repeated queries):")
        print(f"    Mean:   {result['warm']['mean_ms']:.3f} ms")
        print(f"    Median: {result['warm']['median_ms']:.3f} ms")
        print(f"    P95:    {result['warm']['p95_ms']:.3f} ms")

        print(f"\n  Cached recall (same query):")
        print(f"    Mean:   {result['cached']['mean_ms']:.4f} ms")
        print(f"    Median: {result['cached']['median_ms']:.4f} ms")
        print(f"    Min:    {result['cached']['min_us']:.2f} us")


if __name__ == "__main__":
    main()
