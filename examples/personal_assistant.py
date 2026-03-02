"""Personal assistant that remembers your preferences after a week offline.

This example demonstrates Aura's core value proposition:
- Store user preferences as Identity (permanent) memories
- Store tasks as Working (ephemeral) memories
- Simulate time passing with maintenance cycles
- Show that identity persists while tasks naturally decay

No LLM. No cloud. No API keys. Just persistent, biologically-inspired memory.

Run:
    pip install aura-memory
    python examples/personal_assistant.py
"""

import sys
import os
import shutil

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aura import Aura, Level

DATA_DIR = "./personal_assistant_data"


def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_memories(brain, query, label=""):
    """Show what the agent remembers about a topic."""
    results = brain.recall_structured(query, top_k=10)
    if label:
        print(f"\n  {label}:")
    if not results:
        print("    (nothing)")
        return
    for r in results:
        strength = r.get("strength", r.get("score", 0))
        print(f"    [{r['level']:>9}] strength={strength:.3f} | {r['content'][:70]}")


def main():
    # Start fresh for demo
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    brain = Aura(DATA_DIR)

    # ================================================================
    # DAY 1: User tells the assistant about themselves
    # ================================================================
    print_header("DAY 1: Learning about the user")

    # Identity memories -- these define WHO the user is (decay rate: 0.99)
    brain.store("User is vegan and avoids all animal products",
                level=Level.Identity, tags=["diet", "preference"])
    brain.store("User loves jazz music, especially Miles Davis",
                level=Level.Identity, tags=["music", "preference"])
    brain.store("User works from 10am to 6pm, hates early meetings",
                level=Level.Identity, tags=["schedule", "preference"])
    brain.store("User prefers dark mode in all applications",
                level=Level.Identity, tags=["ui", "preference"])
    brain.store("User speaks Ukrainian and English",
                level=Level.Identity, tags=["language", "preference"])

    # Domain knowledge -- things the agent learned (decay rate: 0.95)
    brain.store("User's company uses PostgreSQL and Redis for backend",
                level=Level.Domain, tags=["tech", "work"])
    brain.store("User's team follows trunk-based development workflow",
                level=Level.Domain, tags=["workflow", "work"])

    # Decisions -- recent choices made (decay rate: 0.90)
    brain.store("Decided to migrate auth service to Rust",
                level=Level.Decisions, tags=["decision", "work"])
    brain.store("Chose to use Tailwind CSS for the new dashboard",
                level=Level.Decisions, tags=["decision", "work"])

    # Working memory -- current tasks (decay rate: 0.80)
    brain.store("Need to finish quarterly report by Friday",
                level=Level.Working, tags=["task", "urgent"])
    brain.store("Review PR #142 from Alex before lunch",
                level=Level.Working, tags=["task", "code-review"])
    brain.store("Book restaurant for team dinner tonight",
                level=Level.Working, tags=["task", "social"])

    print(f"  Stored {brain.count()} memories across all levels")
    print_memories(brain, "user preferences and personality",
                   "What we know about the user")
    print_memories(brain, "current tasks and deadlines",
                   "Current tasks")

    # ================================================================
    # SIMULATE A WEEK PASSING
    # ================================================================
    print_header("SIMULATING 7 DAYS (7 maintenance cycles)")

    for day in range(1, 8):
        report = brain.run_maintenance()
        decayed = report.decay.decayed
        archived = report.records_archived
        if decayed > 0 or archived > 0:
            print(f"  Day {day}: {decayed} memories decayed, {archived} archived")
        else:
            print(f"  Day {day}: maintenance complete (no changes)")

    # ================================================================
    # DAY 8: What does the agent remember?
    # ================================================================
    print_header("DAY 8: What does the agent still remember?")

    remaining = brain.count()
    print(f"\n  Memories remaining: {remaining} (started with 12)")

    print_memories(brain, "user preferences personality diet music",
                   "Identity (who the user is)")

    print_memories(brain, "technology stack workflow PostgreSQL",
                   "Domain knowledge (learned facts)")

    print_memories(brain, "decisions migration Rust Tailwind",
                   "Decisions (recent choices)")

    print_memories(brain, "quarterly report PR review restaurant task",
                   "Working memory (tasks)")

    # ================================================================
    # DAY 8: Using remembered context with any LLM
    # ================================================================
    print_header("DAY 8: Generating LLM context")

    context = brain.recall("The user wants restaurant recommendations",
                           token_budget=2000)
    print("\n  Context injected into LLM system prompt:")
    print("  " + "-" * 50)
    for line in context.strip().split("\n"):
        print(f"  {line}")
    print("  " + "-" * 50)
    print("\n  The LLM now knows the user is vegan -- without re-asking.")

    # ================================================================
    # TIER STATS
    # ================================================================
    print_header("MEMORY TIER STATISTICS")

    tier = brain.tier_stats()
    print(f"\n  Core tier (Identity + Domain):     {tier['core_total']} memories")
    print(f"  Cognitive tier (Decisions + Working): {tier['cognitive_total']} memories")

    stats = brain.stats()
    for key in ["total_records", "total_tags", "mean_strength"]:
        if key in stats:
            print(f"  {key}: {stats[key]}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print_header("SUMMARY")
    print("""
  What happened:
    - Identity memories (diet, music, schedule) stayed strong (~0.93)
    - Domain knowledge (tech stack, workflow) held well (~0.70)
    - Working tasks decayed significantly or got promoted to Decisions
    - Maintenance also ran reflect (auto-promotion of important records)

  The agent remembers WHO you are. Tasks fade or get promoted.
  Like a real brain.

  Zero LLM calls. Zero API cost. <1ms recall. ~3 MB binary.
""")

    brain.close()

    # Clean up demo data
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    print("  Demo data cleaned up. Try it with your own agent!")


if __name__ == "__main__":
    main()
