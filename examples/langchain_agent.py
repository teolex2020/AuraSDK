"""Aura + LangChain: conversational agent with persistent memory.

Aura injects recalled context into the system prompt before each LLM call.
No custom memory class needed — just a template variable.

Requirements:
    pip install aura-memory langchain langchain-openai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/langchain_agent.py
"""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from aura import Aura, Level

BRAIN_PATH = "./langchain_data"

SYSTEM_TEMPLATE = """You are a helpful assistant with persistent memory.
Use the memory context below to personalize your responses.
If the user shares new information, acknowledge that you'll remember it.

{memory_context}"""


def main():
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var"

    brain = Aura(BRAIN_PATH)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        ("human", "{input}"),
    ])
    chain = prompt | llm

    print("=" * 60)
    print("  Aura + LangChain — Persistent Memory Agent")
    print(f"  Records in memory: {brain.count()}")
    print("=" * 60)

    # ── Demo conversation ──
    questions = [
        "My name is Alex and I'm building an AI startup in Kyiv.",
        "What tech stack would you recommend for my startup?",
        "What's my name and where am I based?",
    ]

    for user_input in questions:
        # 1. Recall relevant context
        context = brain.recall(user_input, token_budget=1500)

        # 2. Run chain with memory context injected
        response = chain.invoke({
            "memory_context": context,
            "input": user_input,
        })
        reply = response.content

        # 3. Store the exchange
        brain.store(f"User: {user_input}", level=Level.Working, tags=["conversation"])
        brain.store(f"Assistant: {reply[:200]}", level=Level.Working, tags=["conversation"])

        print(f"\n  User: {user_input}")
        print(f"  Assistant: {reply}")

    print(f"\n  Total memories: {brain.count()}")
    brain.close()
    print(f"  Brain saved to {BRAIN_PATH}/")


if __name__ == "__main__":
    main()
