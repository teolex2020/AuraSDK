"""Aura + Ollama: enhanced demo chat for video recording.

Fully local AI assistant with persistent, trust-scored memory.
Designed for screen recording -- colored output, recall timing,
memory stats bar, and a 'restart' command for persistence demo.

Requirements:
    pip install aura-memory requests
    ollama pull llama3.2  # or any model you prefer

Run:
    python examples/demo_ollama_chat.py

Commands during chat:
    restart   -- close and reopen brain (proves persistence)
    stats     -- show memory tier breakdown
    memories  -- show all recalled memories with scores
    quit      -- exit
"""

import json
import sys
import os
import re
import time
import requests

# Windows ANSI support + encoding fix
if sys.platform == "win32":
    os.system("")  # enable VT100 processing
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aura import Aura, Level, AgentPersona, TrustConfig

# -- ANSI Colors --
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"  # change to your preferred model
DATA_DIR = "./demo_chat_data"


def setup_brain(brain):
    """Configure persona and trust on a fresh or reopened brain."""
    persona = AgentPersona()
    persona.name = "Atlas"
    persona.role = "Research Assistant"
    brain.set_persona(persona)

    tc = TrustConfig()
    tc.source_trust = {"user": 1.0, "agent": 0.7, "web_scrape": 0.5}
    brain.set_trust_config(tc)


def chat(brain, messages, user_input):
    """Send a message to Ollama with Aura memory context."""

    # 1. Recall with timing
    t0 = time.perf_counter()
    context = brain.recall(user_input, token_budget=2000)
    recall_ms = (time.perf_counter() - t0) * 1000

    # Show injected context in gray
    print(f"\n{GRAY}--- Context injected ({recall_ms:.2f}ms) ---{RESET}")
    for line in context.strip().split("\n"):
        if line.strip():
            print(f"{GRAY}  {line}{RESET}")
    print(f"{GRAY}---{RESET}")

    # 2. Build system prompt
    system_prompt = f"""You are Atlas, a helpful research assistant with persistent memory.
You remember things about the user across conversations.

{context}

Use the memories above to personalize your responses.
If the user shares new information worth remembering (their name, preferences,
decisions, facts), say [REMEMBER: <brief summary>] at the end of your response."""

    # 3. Send to Ollama
    messages.append({"role": "user", "content": user_input})
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        reply = resp.json()["message"]["content"]
    except requests.ConnectionError:
        return (
            f"{RED}[ERROR] Cannot connect to Ollama. Is it running? (ollama serve){RESET}",
            recall_ms,
            0,
        )
    except Exception as e:
        return f"{RED}[ERROR] {e}{RESET}", recall_ms, 0

    messages.append({"role": "assistant", "content": reply})

    # 4. Extract [REMEMBER: ...] and store
    stored = 0
    for match in re.findall(r"\[REMEMBER:\s*(.+?)\]", reply):
        brain.store(
            match.strip(),
            level=Level.Decisions,
            tags=["learned", "from-conversation"],
            channel="agent",
        )
        stored += 1
        print(f"  {YELLOW}>> Stored memory: {match.strip()[:60]}{RESET}")

    return reply, recall_ms, stored


def status_bar(brain, recall_ms, session_stores):
    """Print memory stats bar after each exchange."""
    count = brain.count()
    print(
        f"{GRAY}[Memories: {count} | "
        f"Last recall: {recall_ms:.2f}ms | "
        f"Stored this session: {session_stores}]{RESET}"
    )


def main():
    brain = Aura(DATA_DIR)
    setup_brain(brain)

    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"{CYAN}  Aura + Ollama -- Fully Local AI with Persistent Memory{RESET}")
    print(f"{CYAN}  No cloud, no API keys, no embeddings{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")
    print(f"  Model:   {YELLOW}{MODEL}{RESET}")
    print(f"  Records: {GREEN}{brain.count()}{RESET}")
    print(f"  Commands: {DIM}quit | stats | memories | restart{RESET}")
    print(f"{CYAN}{'=' * 60}{RESET}")

    messages = []
    session_stores = 0

    while True:
        try:
            user_input = input(f"\n{CYAN}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "restart":
            print(f"\n{YELLOW}Running maintenance before close...{RESET}")
            report = brain.run_maintenance()
            print(
                f"  {DIM}Decayed: {report.decay.decayed} | "
                f"Archived: {report.records_archived}{RESET}"
            )
            count_before = brain.count()
            brain.close()
            print(f"{YELLOW}Brain closed.{RESET}")

            time.sleep(0.5)
            print(f"{YELLOW}Reopening...{RESET}")
            brain = Aura(DATA_DIR)
            setup_brain(brain)
            count_after = brain.count()

            print(f"{GREEN}Brain reopened. {count_after} memories persisted.{RESET}")
            print(f"{DIM}Chat history cleared -- but Aura remembers everything.{RESET}")
            messages = []
            session_stores = 0
            continue

        if user_input.lower() == "stats":
            stats = brain.stats()
            ts = brain.tier_stats()
            print(f"\n{YELLOW}--- Memory Stats ---{RESET}")
            print(f"  Total records:  {GREEN}{stats['total_records']}{RESET}")
            print(
                f"  Cognitive tier:  Working={ts.get('cognitive_working', 0)}"
                f"  Decisions={ts.get('cognitive_decisions', 0)}"
            )
            print(
                f"  Core tier:       Domain={ts.get('core_domain', 0)}"
                f"  Identity={ts.get('core_identity', 0)}"
            )
            print(f"  Connections:    {stats.get('total_connections', 0)}")
            print(f"  Tags:           {stats.get('total_tags', 0)}")
            continue

        if user_input.lower() == "memories":
            t0 = time.perf_counter()
            results = brain.recall_structured("*", top_k=10)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"\n{YELLOW}--- All Memories ({elapsed:.2f}ms) ---{RESET}")
            for r in results:
                print(
                    f"  {DIM}[{r['level']:10}]{RESET} "
                    f"score={GREEN}{r['score']:.3f}{RESET} | "
                    f"{r['content'][:60]}"
                )
            if not results:
                print(f"  {DIM}(empty){RESET}")
            continue

        # Store user message as Working memory
        brain.store(
            f"User said: {user_input}",
            level=Level.Working,
            tags=["conversation"],
            channel="user",
        )
        session_stores += 1

        # Chat with Ollama
        reply, recall_ms, stored = chat(brain, messages, user_input)
        session_stores += stored

        print(f"\n{GREEN}Atlas: {RESET}{reply}")
        status_bar(brain, recall_ms, session_stores)

    # Graceful shutdown
    brain.run_maintenance()
    brain.close()
    print(f"\n{GREEN}Memories saved. Goodbye!{RESET}")


if __name__ == "__main__":
    main()
