"""Run all AuraSDK benchmarks and produce a summary report."""

import os
import sys
import time
import json
import tempfile
import platform
import statistics

from aura import Aura, Level

TAGS_POOL = [
    ["python", "api"], ["rust", "core"], ["deploy", "ci"],
    ["user", "preference"], ["bug", "fix"], ["design", "ui"],
    ["database", "query"], ["auth", "security"], ["test", "qa"],
    ["docs", "readme"],
]
LEVELS = [Level.Working, Level.Decisions, Level.Domain, Level.Identity]
QUERIES = [
    "user preferences and settings",
    "deployment workflow and CI",
    "authentication security issues",
    "python programming language features",
    "database query optimization",
]
N = 1_000  # default scale


def populate(brain, n):
    for i in range(n):
        content = f"Memory record {i}: topic {i % 50} with detailed context"
        brain.store(
            content,
            level=LEVELS[i % len(LEVELS)],
            tags=TAGS_POOL[i % len(TAGS_POOL)],
            deduplicate=False,
        )


def run_all(n=N):
    results = {}

    with tempfile.TemporaryDirectory() as tmp:
        brain = Aura(os.path.join(tmp, "bench.db"))

        # --- Store ---
        store_times = []
        for i in range(n):
            content = f"Bench record {i}: topic {i % 50} with context"
            t0 = time.perf_counter()
            brain.store(
                content,
                level=LEVELS[i % len(LEVELS)],
                tags=TAGS_POOL[i % len(TAGS_POOL)],
                deduplicate=False,
            )
            store_times.append((time.perf_counter() - t0) * 1000)

        results["store"] = {
            "mean_ms": round(statistics.mean(store_times), 4),
            "median_ms": round(statistics.median(store_times), 4),
            "p95_ms": round(sorted(store_times)[int(n * 0.95)], 4),
        }

        # --- Recall structured (warm) ---
        # Prime with one pass
        for q in QUERIES:
            brain.recall_structured(q, top_k=10)

        recall_times = []
        for i in range(100):
            q = QUERIES[i % len(QUERIES)]
            t0 = time.perf_counter()
            brain.recall_structured(q, top_k=10)
            recall_times.append((time.perf_counter() - t0) * 1000)

        results["recall_structured"] = {
            "mean_ms": round(statistics.mean(recall_times), 4),
            "median_ms": round(statistics.median(recall_times), 4),
            "p95_ms": round(sorted(recall_times)[95], 4),
        }

        # --- Recall cached ---
        q = QUERIES[0]
        brain.recall(q)  # prime
        cached_times = []
        for _ in range(1000):
            t0 = time.perf_counter()
            brain.recall(q)
            cached_times.append((time.perf_counter() - t0) * 1_000_000)  # us

        results["recall_cached"] = {
            "mean_us": round(statistics.mean(cached_times), 2),
            "median_us": round(statistics.median(cached_times), 2),
            "min_us": round(min(cached_times), 2),
        }

        # --- Maintenance ---
        maint_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            brain.run_maintenance()
            maint_times.append((time.perf_counter() - t0) * 1000)

        results["maintenance"] = {
            "mean_ms": round(statistics.mean(maint_times), 4),
            "median_ms": round(statistics.median(maint_times), 4),
        }

        brain.close()

    return results


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N

    print("=" * 65)
    print(f"AuraSDK Benchmark Suite  |  {n:,} records")
    print(f"Platform: {platform.system()} {platform.release()} / {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print("=" * 65)

    results = run_all(n)

    print(f"\n  Store:              {results['store']['mean_ms']:.3f} ms/op  (median {results['store']['median_ms']:.3f}, p95 {results['store']['p95_ms']:.3f})")
    print(f"  Recall (structured): {results['recall_structured']['mean_ms']:.3f} ms/op  (median {results['recall_structured']['median_ms']:.3f}, p95 {results['recall_structured']['p95_ms']:.3f})")
    print(f"  Recall (cached):    {results['recall_cached']['mean_us']:.1f} us/op   (median {results['recall_cached']['median_us']:.1f}, min {results['recall_cached']['min_us']:.1f})")
    print(f"  Maintenance:        {results['maintenance']['mean_ms']:.2f} ms/cycle (median {results['maintenance']['median_ms']:.2f})")

    # Save JSON for CI
    output_path = os.path.join(os.path.dirname(__file__), "results.json")
    report = {
        "records": n,
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
