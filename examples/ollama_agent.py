"""Aura + Ollama: fully local AI assistant with persistent memory.

No cloud. No API keys. No embeddings. Everything runs on your machine.

Requirements:
    pip install aura requests
    ollama pull llama3.2  # or any model you prefer

Run:
    python examples/ollama_agent.py
"""

import json
import requests
from aura import Aura, Level, AgentPersona

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"  # change to your preferred model


def chat(brain: Aura, messages: list, user_input: str) -> str:
    """Send a message to Ollama with Aura memory context."""

    # 1. Recall relevant memories (RRF Fusion: SDR + MinHash + Tag Jaccard)
    context = brain.recall(user_input, token_budget=2000)

    # 2. Build system prompt with memory
    system_prompt = f"""You are a helpful assistant with persistent memory.
You remember things about the user across conversations.

{context}

Use the memories above to personalize your responses.
If the user shares new information worth remembering, say [REMEMBER: ...] at the end."""

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
        return "[ERROR] Cannot connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"[ERROR] {e}"

    messages.append({"role": "assistant", "content": reply})

    # 4. Extract [REMEMBER: ...] tags and store as memories
    import re
    for match in re.findall(r"\[REMEMBER:\s*(.+?)\]", reply):
        brain.store(match.strip(), level=Level.Decisions, tags=["learned", "from-conversation"])
        print(f"  💾 Stored memory: {match.strip()[:60]}")

    return reply


def main():
    brain = Aura("./ollama_agent_data")

    # Set agent persona
    persona = AgentPersona()
    persona.name = "Nova"
    persona.role = "Personal AI Assistant"
    brain.set_persona(persona)

    print("=" * 60)
    print("  Aura + Ollama — Fully Local AI with Memory")
    print("  No cloud, no API keys, no embeddings")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print(f"  Records in memory: {brain.count()}")
    print(f"  Type 'quit' to exit, 'stats' for memory stats")
    print("=" * 60)

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            stats = brain.stats()
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v}")
            continue
        if user_input.lower() == "memories":
            results = brain.recall_structured(user_input, top_k=10)
            for r in results:
                print(f"  [{r['level']}] {r['content'][:80]}")
            continue

        # Store user messages as Working memory (short-term)
        brain.store(
            f"User said: {user_input}",
            level=Level.Working,
            tags=["conversation"],
        )

        reply = chat(brain, messages, user_input)
        print(f"\nNova: {reply}")

    # Run maintenance before closing (decay old Working memories)
    brain.run_maintenance()
    brain.close()
    print("\nMemories saved. Goodbye!")


if __name__ == "__main__":
    main()
