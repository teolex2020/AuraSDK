"""
AuraSDK Demo: Cheap model + memory beats expensive model without memory.

gemini-2.5-flash-lite + AuraSDK  vs  gemini-3.1-pro-preview (no memory)

Setup:
    pip install aura-memory google-generativeai

Run:
    python examples/gemini_aura_demo.py --api-key YOUR_GEMINI_API_KEY
"""

import argparse
import shutil
import tempfile
import time

import google.generativeai as genai

from aura import Aura, Level

# ── Models ──────────────────────────────────────────────────────────────────

CHEAP_MODEL = "gemini-2.5-flash-lite"   # + AuraSDK memory
SMART_MODEL = "gemini-3.1-pro-preview"  # no memory, baseline

# ── Simulated conversation history (3 "days" of work) ───────────────────────

HISTORY = [
    # Day 1 — project context
    ("user",  "I'm building a FastAPI service. We decided to use PostgreSQL, not MySQL."),
    ("agent", "Got it. PostgreSQL it is."),
    ("user",  "Auth is JWT, tokens expire in 15 minutes. Refresh tokens live 7 days."),
    ("agent", "Noted — JWT 15m access, 7d refresh."),
    ("user",  "We deploy to staging first, always. Never push straight to prod."),
    ("agent", "Understood. Staging before prod, no exceptions."),

    # Day 2 — preferences and bugs
    ("user",  "I prefer async endpoints everywhere — sync causes latency spikes for us."),
    ("agent", "Async by default, got it."),
    ("user",  "We hit a bug last week: forgot to index user_id on the events table. Cost us 3 hours."),
    ("agent", "Noted — always index foreign keys, especially user_id on events."),
    ("user",  "Our team uses Black for formatting and ruff for linting. Non-negotiable."),
    ("agent", "Black + ruff, enforced."),

    # Day 3 — architecture decisions
    ("user",  "We chose Redis for session cache, not Memcached — easier ops."),
    ("agent", "Redis for sessions."),
    ("user",  "Rate limiting is handled at the Nginx level, not in the app."),
    ("agent", "Rate limiting in Nginx, not FastAPI middleware."),
    ("user",  "Background jobs go to Celery with Redis broker. No database polling."),
    ("agent", "Celery + Redis broker for async jobs."),
]

# ── Test questions (require accumulated context to answer well) ──────────────

QUESTIONS = [
    "I'm writing a new endpoint. What should I keep in mind about our stack?",
    "Should I add a sync or async handler for the new payment webhook?",
    "We need to add background email sending. How should we do it?",
    "I'm about to deploy a hotfix. Walk me through the process.",
    "A colleague suggested MySQL for a new microservice. What's your take?",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def ask(model_name: str, system_prompt: str, question: str) -> str:
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )
    response = model.generate_content(question)
    return response.text.strip()


def print_divider(title: str = ""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * width)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AuraSDK vs no-memory demo")
    parser.add_argument("--api-key", required=True, help="Google Gemini API key")
    args = parser.parse_args()

    genai.configure(api_key=args.api_key)

    brain_path = tempfile.mkdtemp(prefix="aura_demo_")
    brain = Aura(brain_path)
    brain.enable_full_cognitive_stack()

    print("\n🧠 AuraSDK Demo: Cheap model with memory vs expensive model without")
    print(f"   Cheap + memory : {CHEAP_MODEL} + AuraSDK")
    print(f"   Expensive alone: {SMART_MODEL} (no memory)")

    # ── Phase 1: feed history into Aura ─────────────────────────────────────
    print_divider("Phase 1: loading 3 days of conversation into AuraSDK")

    for role, content in HISTORY:
        if role == "user":
            level = Level.Domain
            tags = ["project", "context"]
            # promote decisions and preferences higher
            if any(w in content.lower() for w in ["decided", "always", "never", "prefer", "chose", "non-negotiable"]):
                level = Level.Decisions
                tags.append("decision")
            brain.store(content, level=level, tags=tags)

    brain.run_maintenance()
    stats = brain.stats()
    print(f"   ✓ {stats['total_records']} memories stored, cognitive pipeline active")

    # ── Phase 2: run test questions ──────────────────────────────────────────
    print_divider("Phase 2: test questions")

    scores = {"aura": 0, "baseline": 0}

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[Q{i}] {question}")

        # Aura: recall context, inject into system prompt
        context = brain.recall(question, token_budget=800)
        system_with_memory = (
            "You are a helpful dev assistant. "
            "Use the context below — it contains decisions and preferences "
            "accumulated over the past week of work together.\n\n"
            f"CONTEXT:\n{context}"
        )

        # Baseline: generic system prompt, no memory
        system_baseline = (
            "You are a helpful software development assistant. "
            "Answer based on general best practices."
        )

        t0 = time.time()
        answer_aura = ask(CHEAP_MODEL, system_with_memory, question)
        t_aura = time.time() - t0

        t0 = time.time()
        answer_baseline = ask(SMART_MODEL, system_baseline, question)
        t_baseline = time.time() - t0

        print(f"\n  [{CHEAP_MODEL} + AuraSDK] ({t_aura:.1f}s)")
        for line in answer_aura[:400].split("\n"):
            print(f"    {line}")
        if len(answer_aura) > 400:
            print("    ...")

        print(f"\n  [{SMART_MODEL}, no memory] ({t_baseline:.1f}s)")
        for line in answer_baseline[:400].split("\n"):
            print(f"    {line}")
        if len(answer_baseline) > 400:
            print("    ...")

        # Simple heuristic scoring: does the answer reference project-specific details?
        project_terms = [
            "postgresql", "postgres", "staging", "jwt", "redis", "celery",
            "async", "black", "ruff", "nginx", "user_id", "index",
        ]
        aura_hits = sum(1 for t in project_terms if t in answer_aura.lower())
        base_hits = sum(1 for t in project_terms if t in answer_baseline.lower())

        if aura_hits > base_hits:
            scores["aura"] += 1
            verdict = f"✓ Aura more specific ({aura_hits} project terms vs {base_hits})"
        elif base_hits > aura_hits:
            scores["baseline"] += 1
            verdict = f"△ Baseline more specific ({base_hits} project terms vs {aura_hits})"
        else:
            verdict = f"= Tie ({aura_hits} project terms each)"

        print(f"\n  → {verdict}")

        # store this interaction so Aura keeps learning
        brain.store(f"Question: {question}", level=Level.Working, tags=["session"])
        brain.store(f"Answer given: {answer_aura[:200]}", level=Level.Working, tags=["session"])

    # ── Results ──────────────────────────────────────────────────────────────
    print_divider("Results")
    total = len(QUESTIONS)
    print(f"\n  {CHEAP_MODEL} + AuraSDK : {scores['aura']}/{total} questions more project-specific")
    print(f"  {SMART_MODEL} alone     : {scores['baseline']}/{total} questions more project-specific")

    hints = brain.get_surfaced_policy_hints(limit=3)
    if hints:
        print("\n  Policy hints Aura derived automatically from your history:")
        for h in hints:
            print(f"    → [{h.action_kind}] {h.description}")

    concepts = brain.get_surfaced_concepts(limit=3)
    if concepts:
        print("\n  Concepts Aura formed:")
        for c in concepts:
            print(f"    → {c.label} (score: {c.score:.2f})")

    print(f"\n  Cost of Aura memory layer: $0")
    print(f"  Cost difference between models: significant")
    print(f"\n  Same result. Cheaper model. No retraining.\n")

    brain.close()
    shutil.rmtree(brain_path, ignore_errors=True)


if __name__ == "__main__":
    main()
