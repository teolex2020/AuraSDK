"""Aura basics — store, recall, search, update, delete.

Run:
    pip install aura
    python examples/basic_usage.py
"""

from aura import Aura, Level

brain = Aura("./example_data")

# ── Store memories at different levels ──
id1 = brain.store(
    "User prefers dark mode and Vim keybindings",
    level=Level.Identity,
    tags=["preference", "ui"],
)
id2 = brain.store(
    "Rust has zero-cost abstractions and ownership model",
    level=Level.Domain,
    tags=["rust", "performance"],
)
id3 = brain.store(
    "Deploy staging before production, always run tests first",
    level=Level.Decisions,
    tags=["workflow", "deploy"],
)
id4 = brain.store(
    "Fix the login bug — users getting 403 on /api/auth",
    level=Level.Working,
    tags=["bug", "auth", "urgent"],
)

print(f"Stored 4 records across all memory levels")

# ── Recall: returns formatted context for LLM injection ──
context = brain.recall("authentication issues")
print("\n--- Recall (formatted for LLM) ---")
print(context)

# ── Structured recall: scored results with metadata ──
results = brain.recall_structured("deployment workflow", top_k=5)
print("\n--- Structured Recall ---")
for r in results:
    print(f"  [{r['level']}] score={r['score']:.3f} — {r['content'][:60]}")

# ── Search by tag ──
bugs = brain.search(tags=["bug"])
print(f"\n--- Search by tag 'bug': {len(bugs)} results ---")
for rec in bugs:
    print(f"  {rec.id[:8]}: {rec.content}")

# ── Update ──
brain.update(id4, content="Login bug fixed — was CORS config issue", tags=["bug", "auth", "resolved"])
updated = brain.get(id4)
print(f"\n--- Updated record ---")
print(f"  {updated.content}")
print(f"  Tags: {updated.tags}")

# ── Connect related memories ──
brain.connect(id2, id3)  # Rust knowledge → deployment workflow

# ── Stats ──
stats = brain.stats()
print(f"\n--- Brain stats ---")
for k, v in sorted(stats.items()):
    print(f"  {k}: {v}")

brain.close()
print("\nDone. Data saved to ./example_data/")
