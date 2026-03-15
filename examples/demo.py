#!/usr/bin/env python3
"""
AuraSDK -- 60-second demo.

Shows how the cognitive layer learns from interactions
without any fine-tuning, embeddings, or LLM calls.

Usage:
    pip install aura-memory
    python demo.py
"""

import tempfile
import time
import shutil

try:
    from aura import Aura, Level
except ImportError:
    print("Install first: pip install aura-memory")
    raise SystemExit(1)


def heading(text):
    print("\n" + "-" * 56)
    print("  " + text)
    print("-" * 56)


def wait(label):
    print("    " + label, end="", flush=True)
    for _ in range(3):
        time.sleep(0.3)
        print(".", end="", flush=True)
    print(" done")


def run_demo():
    tmp = tempfile.mkdtemp(prefix="aura_demo_")

    print()
    print("=" * 56)
    print("         AuraSDK -- Cognitive Layer Demo")
    print("   Your AI model, learning from every interaction")
    print("=" * 56)

    brain = Aura(tmp)
    brain.enable_full_cognitive_stack()

    # ── Phase 1: simulate interactions ───────────────────────────────────────
    heading("Phase 1 - Simulating 30 interactions")
    print("  (this is what happens as your agent talks to users)")

    print("\n[1] User preferences accumulating...")
    preferences = [
        ("User always reviews code before merging to main branch", ["workflow", "code-review"]),
        ("User prefers detailed error messages over silent failures", ["workflow", "errors"]),
        ("User wants notifications only for critical issues", ["notifications", "workflow"]),
        ("User likes concise answers, not long explanations", ["communication"]),
        ("User prefers dark mode in all tools", ["ui", "preferences"]),
        ("User schedules deep work in morning, meetings in afternoon", ["schedule", "productivity"]),
        ("User prefers async communication over meetings when possible", ["communication", "workflow"]),
    ]
    for content, tags in preferences:
        brain.store(content, level=Level.Domain, tags=tags)
    wait("Stored 7 user preferences")

    print("\n[2] Decisions and outcomes accumulating...")
    events = [
        ("Deployed directly to production -- caused 2h outage", ["deploy", "incident"], "deploy-1"),
        ("Added staging environment -- caught 3 bugs before prod", ["deploy", "staging"], "deploy-2"),
        ("Staging deploy prevented database migration failure", ["deploy", "staging", "database"], "deploy-3"),
        ("Direct prod deploy skipped staging -- caused data loss", ["deploy", "incident"], "deploy-4"),
        ("Staging review caught API breaking change in time", ["deploy", "staging", "api"], "deploy-5"),
        ("Skipped code review -- introduced security vulnerability", ["code-review", "security"], "review-1"),
        ("Code review caught SQL injection before merge", ["code-review", "security"], "review-2"),
        ("Code review found performance regression early", ["code-review", "performance"], "review-3"),
        ("Merged without review -- broke production auth", ["code-review", "incident"], "review-4"),
        ("Critical alert ignored -- escalated to incident", ["notifications", "incident"], "notif-1"),
        ("Non-critical noise alerts caused alert fatigue", ["notifications", "workflow"], "notif-2"),
        ("Filtered notifications to critical-only -- response time improved", ["notifications", "workflow"], "notif-3"),
    ]
    ids = {}
    for content, tags, key in events:
        result = brain.store(content, level=Level.Decisions, tags=tags)
        ids[key] = result if isinstance(result, str) else result.id
    wait("Stored 12 decisions with outcomes")

    print("\n[3] Linking causes to effects...")
    causal_pairs = [
        ("deploy-4", "deploy-3"),
        ("review-4", "review-2"),
        ("notif-1", "notif-3"),
    ]
    event_index = {e[2]: e[0] for e in events}
    for effect_key, cause_key in causal_pairs:
        if effect_key in ids and cause_key in ids:
            brain.store(
                "Pattern: " + event_index[effect_key][:60],
                level=Level.Domain,
                tags=["pattern", "causal"],
                caused_by_id=ids[cause_key],
            )
    wait("Linked causal relationships")

    # ── Phase 2: learning cycle ───────────────────────────────────────────────
    heading("Phase 2 - Running learning cycle")
    print("  (no LLM calls, no cloud, no fine-tuning)")

    t0 = time.perf_counter()
    cycles = 5
    for i in range(cycles):
        brain.run_maintenance()
        print(f"    cycle {i+1}/{cycles}", end="\r", flush=True)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n  [ok] {cycles} learning cycles complete in {elapsed:.1f}ms")

    # ── Phase 3: what the system learned ─────────────────────────────────────
    heading("Phase 3 - What the system learned on its own")

    concepts = brain.get_surfaced_concepts(5)
    if concepts:
        print("\n  Discovered topic clusters:")
        for c in concepts:
            label = getattr(c, "label", None) or getattr(c, "key", str(c))
            print(f"    * {label}  ({len(c.record_ids)} observations, score={c.score:.2f})")
    else:
        print("\n  Topic clusters: forming (need more cycles on real data)")

    try:
        patterns = brain.get_surfaced_causal_patterns(5)
        if patterns:
            print("\n  Learned cause -> effect patterns:")
            for p in patterns:
                print(f"    * {p.cause_label} -> {p.effect_label}")
                print(f"      confidence={p.score:.2f}  support={p.support}")
    except AttributeError:
        pass

    hints = brain.get_surfaced_policy_hints(5)
    if hints:
        print("\n  Derived behavioral policies (no one wrote these rules):")
        for h in hints:
            action = getattr(h, "action_kind", getattr(h, "action", "Recommend"))
            domain = getattr(h, "domain", "")
            desc = getattr(h, "description", str(h))
            strength = getattr(h, "strength", 0.0)
            print(f"\n    [{action}] {desc[:70]}")
            print(f"    domain={domain}  confidence={strength:.2f}")
    else:
        print("\n  Behavioral policies: accumulating")
        print("  (policies emerge after repeated patterns -- run demo multiple times)")

    # ── Phase 4: recall demo ──────────────────────────────────────────────────
    heading("Phase 4 - Recall in action")
    print("  (what gets injected into your LLM prompt)")

    queries = ["deployment decision", "code review", "user preferences"]
    for q in queries:
        t0 = time.perf_counter()
        results = brain.recall_structured(q, top_k=3, min_strength=0.0)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"\n  Query: \"{q}\"  [{elapsed:.2f}ms]")
        for i, r in enumerate(results[:2], 1):
            content = r["content"] if isinstance(r, dict) else r.content
            print(f"    {i}. {content[:70]}")

    # ── summary ──────────────────────────────────────────────────────────────
    heading("Summary")
    print("""
  30 interactions -> cognitive layer extracted:
    * User preference patterns
    * Causal relationships between decisions and outcomes
    * Behavioral policies your agent can act on

  Zero LLM calls during learning.
  Zero cloud. Zero fine-tuning.
  Works fully offline.

  The model stays the same.
  Your agent gets smarter.

  ---------------------------------------------
  pip install aura-memory
  github.com/teolex2020/AuraSDK
  aurasdk.dev
  ---------------------------------------------
    """)

    brain.close()
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass


if __name__ == "__main__":
    run_demo()
