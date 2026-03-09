"""Personal assistant with persistent local memory.

This example shows a simple user-facing flow:
- stable user preferences stay easy to recall,
- temporary tasks become less prominent over time,
- memory maintenance keeps the store tidy.

Run:
    pip install aura-memory
    python examples/personal_assistant.py
"""

import os
import shutil
import sys

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
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    brain = Aura(DATA_DIR)

    print_header("DAY 1: Learning about the user")

    brain.store("User is vegan and avoids all animal products", level=Level.Identity, tags=["diet", "preference"])
    brain.store("User loves jazz music, especially Miles Davis", level=Level.Identity, tags=["music", "preference"])
    brain.store("User works from 10am to 6pm, hates early meetings", level=Level.Identity, tags=["schedule", "preference"])
    brain.store("User prefers dark mode in all applications", level=Level.Identity, tags=["ui", "preference"])
    brain.store("User speaks Ukrainian and English", level=Level.Identity, tags=["language", "preference"])

    brain.store("User's company uses PostgreSQL and Redis for backend", level=Level.Domain, tags=["tech", "work"])
    brain.store("User's team follows trunk-based development workflow", level=Level.Domain, tags=["workflow", "work"])

    brain.store("Decided to migrate auth service to Rust", level=Level.Decisions, tags=["decision", "work"])
    brain.store("Chose to use Tailwind CSS for the new dashboard", level=Level.Decisions, tags=["decision", "work"])

    brain.store("Need to finish quarterly report by Friday", level=Level.Working, tags=["task", "urgent"])
    brain.store("Review PR #142 from Alex before lunch", level=Level.Working, tags=["task", "code-review"])
    brain.store("Book restaurant for team dinner tonight", level=Level.Working, tags=["task", "social"])

    print(f"  Stored {brain.count()} memories across all levels")
    print_memories(brain, "user preferences and personality", "What we know about the user")
    print_memories(brain, "current tasks and deadlines", "Current tasks")

    print_header("SIMULATING A WEEK")
    for day in range(1, 8):
        report = brain.run_maintenance()
        decayed = report.decay.decayed
        archived = report.records_archived
        if decayed > 0 or archived > 0:
            print(f"  Day {day}: {decayed} memories changed, {archived} archived")
        else:
            print(f"  Day {day}: maintenance complete")

    print_header("DAY 8: What does the agent still remember?")
    print(f"\n  Memories remaining: {brain.count()} (started with 12)")
    print_memories(brain, "user preferences personality diet music", "Identity")
    print_memories(brain, "technology stack workflow PostgreSQL", "Domain knowledge")
    print_memories(brain, "decisions migration Rust Tailwind", "Decisions")
    print_memories(brain, "quarterly report PR review restaurant task", "Working memory")

    print_header("DAY 8: Generating LLM context")
    context = brain.recall("The user wants restaurant recommendations", token_budget=2000)
    print("\n  Context injected into an assistant prompt:")
    print("  " + "-" * 50)
    for line in context.strip().split("\n"):
        print(f"  {line}")
    print("  " + "-" * 50)
    print("\n  The assistant still remembers that the user is vegan.")

    print_header("MEMORY TIER STATISTICS")
    tier = brain.tier_stats()
    print(f"\n  Core tier:      {tier['core_total']} memories")
    print(f"  Cognitive tier: {tier['cognitive_total']} memories")

    print_header("SUMMARY")
    print(
        """
  What happened:
    - Stable user context stayed available
    - Learned project context remained useful
    - Short-lived tasks became less prominent over time
    - Maintenance kept the memory store tidy

  The agent remembers who you are while temporary tasks fade.
  Local memory, no external dependency required.
"""
    )

    brain.close()
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    print("  Demo data cleaned up.")


if __name__ == "__main__":
    main()
