"""Stress test: 1M records — store, recall, and memory profiling.

WARNING: This test takes significant time and memory.
         Expected: 10-30 minutes, 500MB-2GB RAM depending on system.

Usage:
    python benchmarks/stress_1m.py

Measures:
    - Store throughput for 1M records
    - Recall latency at 1M scale
    - Memory (RSS) at 100K / 250K / 500K / 750K / 1M checkpoints
    - Maintenance cycle time at 1M
"""

import os
import sys
import time
import json
import shutil
import tempfile
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura import Aura, Level

RECORD_COUNT = 1_000_000
CHECKPOINTS = [100_000, 250_000, 500_000, 750_000, 1_000_000]

WORDS = [
    "agent", "memory", "recall", "context", "decision", "workflow", "deploy",
    "testing", "performance", "latency", "rust", "python", "model", "neural",
    "embedding", "vector", "search", "index", "cache", "database", "query",
    "response", "request", "server", "client", "token", "prompt", "system",
    "user", "assistant", "function", "parameter", "output", "input", "layer",
    "architecture", "module", "interface", "protocol", "network", "compute",
    "storage", "throughput", "bandwidth", "optimization", "algorithm", "data",
    "structure", "pipeline", "stream", "buffer", "scheduler", "process",
    "gradient", "backprop", "inference", "training", "weights", "bias",
    "activation", "transformer", "attention", "encoder", "decoder", "batch",
]

LEVELS = [Level.Working, Level.Decisions, Level.Domain, Level.Identity]
TAGS_POOL = [
    "preference", "workflow", "bug", "feature", "meeting", "research",
    "code", "design", "review", "deploy", "config", "security", "test",
    "api", "frontend", "backend", "database", "infra", "monitoring",
]


def random_content(min_words=5, max_words=20):
    n = random.randint(min_words, max_words)
    return " ".join(random.choices(WORDS, k=n))


def random_tags(max_tags=3):
    n = random.randint(0, max_tags)
    return random.sample(TAGS_POOL, k=min(n, len(TAGS_POOL)))


def get_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_stress_test():
    results = {
        "record_count": RECORD_COUNT,
        "checkpoints": {},
        "store": {},
        "recall": {},
        "maintenance": {},
    }

    tmpdir = tempfile.mkdtemp(prefix="aura_stress_1m_")
    db_path = os.path.join(tmpdir, "stress.db")

    print(f"=== Aura Stress Test: {RECORD_COUNT:,} records ===")
    print(f"DB path: {db_path}")
    print(f"WARNING: This will take a while. Grab a coffee.")
    print()

    brain = Aura(db_path)

    # ── Store phase ──
    print("Storing records...")
    store_start = time.perf_counter()

    for i in range(1, RECORD_COUNT + 1):
        content = random_content()
        level = random.choice(LEVELS)
        tags = random_tags()

        brain.store(content, level=level, tags=tags, deduplicate=False)

        if i in CHECKPOINTS:
            elapsed = time.perf_counter() - store_start
            rss = get_rss_mb()
            rate = i / elapsed
            results["checkpoints"][str(i)] = {
                "elapsed_s": round(elapsed, 2),
                "rss_mb": round(rss, 1),
                "store_rate_per_s": round(rate, 0),
            }
            print(f"  [{i:>10,}] {elapsed:>8.1f}s | {rate:>8,.0f} rec/s | RSS: {rss:.0f} MB")

    total_store_time = time.perf_counter() - store_start
    results["store"] = {
        "total_s": round(total_store_time, 2),
        "avg_us": round(total_store_time / RECORD_COUNT * 1_000_000, 1),
        "rate_per_s": round(RECORD_COUNT / total_store_time, 0),
    }
    print(f"\n  Total store: {total_store_time:.1f}s "
          f"({total_store_time / RECORD_COUNT * 1000:.3f} ms/record)")

    # ── Recall phase ──
    print("\nRecall (cold, 50 queries)...")
    queries = [random_content(3, 6) for _ in range(50)]

    cold_start = time.perf_counter()
    for q in queries:
        brain.recall(q)
    cold_time = time.perf_counter() - cold_start
    cold_avg_ms = cold_time / len(queries) * 1000

    results["recall"]["cold_50_total_s"] = round(cold_time, 3)
    results["recall"]["cold_avg_ms"] = round(cold_avg_ms, 3)
    print(f"  50 cold recalls: {cold_time:.3f}s (avg {cold_avg_ms:.3f} ms)")

    print("Recall (warm, cached)...")
    warm_start = time.perf_counter()
    for q in queries:
        brain.recall(q)
    warm_time = time.perf_counter() - warm_start
    warm_avg_ms = warm_time / len(queries) * 1000

    results["recall"]["warm_50_total_s"] = round(warm_time, 3)
    results["recall"]["warm_avg_ms"] = round(warm_avg_ms, 3)
    print(f"  50 warm recalls: {warm_time:.3f}s (avg {warm_avg_ms:.3f} ms)")

    # ── Maintenance ──
    print("\nMaintenance cycle...")
    maint_start = time.perf_counter()
    report = brain.run_maintenance()
    maint_time = time.perf_counter() - maint_start

    results["maintenance"] = {
        "duration_s": round(maint_time, 3),
        "report": report,
    }
    print(f"  Maintenance: {maint_time:.3f}s")

    # ── Final stats ──
    final_count = brain.count()
    final_rss = get_rss_mb()
    results["final"] = {
        "record_count": final_count,
        "rss_mb": round(final_rss, 1),
    }

    print(f"\n  Final count: {final_count:,}")
    print(f"  Final RSS:   {final_rss:.0f} MB")

    # ── Save results ──
    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "stress_1m_results.json",
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    brain.close()
    shutil.rmtree(tmpdir, ignore_errors=True)

    return results


if __name__ == "__main__":
    run_stress_test()
