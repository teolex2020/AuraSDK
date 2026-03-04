"""Aura + LlamaIndex: chat engine with persistent memory.

Aura acts as a ChatMemoryBuffer replacement — recalled context is injected
into the system prompt before each LLM call, and new messages are stored
automatically.

Requirements:
    pip install aura-memory llama-index-core llama-index-llms-openai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/llamaindex_agent.py
"""

import os

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from aura import Aura, Level

BRAIN_PATH = "./llamaindex_data"

SYSTEM_PROMPT = """You are a helpful assistant with persistent memory.
Use the memory context below to personalize your responses.
If the user shares new information about themselves, acknowledge it.

--- Recalled memories ---
{memory}
--- End memories ---"""


def chat_with_memory(brain: Aura, llm: OpenAI, user_input: str) -> str:
    """Single turn: recall -> generate -> store."""
    # 1. Recall relevant context
    context = brain.recall(user_input, token_budget=1500)

    # 2. Build messages with memory-augmented system prompt
    system_msg = ChatMessage(
        role=MessageRole.SYSTEM,
        content=SYSTEM_PROMPT.format(memory=context if context.strip() else "No relevant memories yet."),
    )
    user_msg = ChatMessage(role=MessageRole.USER, content=user_input)

    # 3. Generate response
    response = llm.chat([system_msg, user_msg])
    reply = response.message.content

    # 4. Store the exchange
    brain.store(f"User: {user_input}", level=Level.Working, tags=["conversation"])
    brain.store(f"Assistant: {reply[:200]}", level=Level.Working, tags=["conversation"])

    return reply


def main():
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var"

    brain = Aura(BRAIN_PATH)
    llm = OpenAI(model="gpt-4o-mini", temperature=0.7)

    # Pre-load some knowledge
    brain.store("User is a backend developer who uses FastAPI", level=Level.Identity, tags=["user", "tech"])
    brain.store("User prefers PostgreSQL over MySQL", level=Level.Identity, tags=["user", "database"])

    print("=" * 60)
    print("  Aura + LlamaIndex -- Persistent Memory Chat")
    print(f"  Records in memory: {brain.count()}")
    print("=" * 60)

    questions = [
        "What database should I use for my new project?",
        "I'm also interested in learning Rust. Remember that.",
        "What do you know about me so far?",
    ]

    for user_input in questions:
        reply = chat_with_memory(brain, llm, user_input)
        print(f"\n  User: {user_input}")
        print(f"  Assistant: {reply}")

    # Run maintenance to decay working memories
    brain.run_maintenance()

    print(f"\n  Total memories: {brain.count()}")
    brain.close()
    print(f"  Brain saved to {BRAIN_PATH}/")


if __name__ == "__main__":
    main()
