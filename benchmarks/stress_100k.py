"""Stress test: 100K records — store, recall, and memory profiling.

Usage:
    python benchmarks/stress_100k.py

Measures:
    - Store throughput for 100K records
    - Recall latency at 100K scale (cold + warm)
    - Memory (RSS) at 10K / 50K / 100K checkpoints
    - Maintenance cycle time at 100K
"""

import os
import sys
import time
import json
import shutil
import tempfile
import random
import string

# Add parent to path for aura import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aura import Aura, Level

RECORD_COUNT = 100_000
CHECKPOINTS = [10_000, 50_000, 100_000]

# Word pool for generating realistic-ish content
WORDS = [
    "agent", "memory", "recall", "context", "decision", "workflow", "deploy",
    "testing", "performance", "latency", "rust", "python", "model", "neural",
    "embedding", "vector", "search", "index", "cache", "database", "query",
    "response", "request", "server", "client", "token", "prompt", "system",
    "user", "assistant", "function", "parameter", "output", "input", "layer",
    "architecture", "module", "interface", "protocol", "network", "compute",
    "storage", "throughput", "bandwidth", "optimization", "algorithm", "data",
    "structure", "pipeline", "stream", "buffer", "scheduler", "process",
]

LEVELS = [Level.Working, Level.Decisions, Level.Domain, Level.Identity]
TAGS_POOL = [
    "preference", "workflow", "bug", "feature", "meeting", "research",
    "code", "design", "review", "deploy", "config", "security", "test",
]


def random_content(min_words=5, max_words=20):
    """Generate a random sentence from the word pool."""
    n = random.randint(min_words, max_words)
    return " ".join(random.choices(WORDS, k=n))


def random_tags(max_tags=3):
    """Pick 0-3 random tags."""
    n = random.randint(0, max_tags)
    return random.sample(TAGS_POOL, k=min(n, len(TAGS_POOL)))


def get_rss_mb():
    """Get current process RSS in MB (cross-platform)."""
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

    tmpdir = tempfile.mkdtemp(prefix="aura_stress_100k_")
    db_path = os.path.join(tmpdir, "stress.db")

    print(f"=== Aura Stress Test: {RECORD_COUNT:,} records ===")
    print(f"DB path: {db_path}")
    print()

    brain = Aura(db_path)

    # ── Store phase ──
    print("Storing records...")
    store_start = time.perf_counter()
    checkpoint_times = {}

    for i in range(1, RECORD_COUNT + 1):
        content = random_content()
        level = random.choice(LEVELS)
        tags = random_tags()

        brain.store(content, level=level, tags=tags)

        if i in CHECKPOINTS:
            elapsed = time.perf_counter() - store_start
            rss = get_rss_mb()
            rate = i / elapsed
            checkpoint_times[i] = elapsed
            results["checkpoints"][str(i)] = {
                "elapsed_s": round(elapsed, 2),
                "rss_mb": round(rss, 1),
                "store_rate_per_s": round(rate, 0),
            }
            print(f"  [{i:>7,}] {elapsed:>7.1f}s | {rate:>8,.0f} rec/s | RSS: {rss:.0f} MB")

    total_store_time = time.perf_counter() - store_start
    results["store"] = {
        "total_s": round(total_store_time, 2),
        "avg_us": round(total_store_time / RECORD_COUNT * 1_000_000, 1),
        "rate_per_s": round(RECORD_COUNT / total_store_time, 0),
    }
    print(f"\n  Total store: {total_store_time:.1f}s "
          f"({total_store_time / RECORD_COUNT * 1000:.3f} ms/record)")

    # ── Recall phase (cold) ──
    print("\nRecall (cold, no cache)...")
    queries = [random_content(3, 6) for _ in range(100)]

    cold_start = time.perf_counter()
    for q in queries:
        brain.recall(q)
    cold_time = time.perf_counter() - cold_start
    cold_avg_ms = cold_time / len(queries) * 1000

    results["recall"]["cold_100_total_s"] = round(cold_time, 3)
    results["recall"]["cold_avg_ms"] = round(cold_avg_ms, 3)
    print(f"  100 cold recalls: {cold_time:.3f}s (avg {cold_avg_ms:.3f} ms)")

    # ── Recall phase (warm — same queries, cache populated) ──
    print("Recall (warm, cached)...")
    warm_start = time.perf_counter()
    for q in queries:
        brain.recall(q)
    warm_time = time.perf_counter() - warm_start
    warm_avg_ms = warm_time / len(queries) * 1000

    results["recall"]["warm_100_total_s"] = round(warm_time, 3)
    results["recall"]["warm_avg_ms"] = round(warm_avg_ms, 3)
    print(f"  100 warm recalls: {warm_time:.3f}s (avg {warm_avg_ms:.3f} ms)")

    # ── Recall structured ──
    print("Recall structured (top 20)...")
    struct_start = time.perf_counter()
    for q in queries[:20]:
        brain.recall_structured(q, top_k=20)
    struct_time = time.perf_counter() - struct_start
    struct_avg_ms = struct_time / 20 * 1000

    results["recall"]["structured_20_total_s"] = round(struct_time, 3)
    results["recall"]["structured_avg_ms"] = round(struct_avg_ms, 3)
    print(f"  20 structured recalls: {struct_time:.3f}s (avg {struct_avg_ms:.3f} ms)")

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
        "stress_100k_results.json",
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
