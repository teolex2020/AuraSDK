"""Aura + Anthropic Claude SDK: agent with persistent memory.

Demonstrates two patterns:
  1. System prompt injection — recall context before each message
  2. Tool use — Claude decides when to store/recall via tools

Requirements:
    pip install aura-memory anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/claude_sdk_agent.py
"""

import os

import anthropic

from aura import Aura, Level

BRAIN_PATH = "./claude_sdk_data"

MEMORY_TOOLS = [
    {
        "name": "remember",
        "description": (
            "Store an important fact about the user or conversation in "
            "long-term memory. Use this when the user shares preferences, "
            "personal details, or decisions worth remembering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization (e.g. ['preference', 'work']).",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Search long-term memory for relevant information. "
            "Use this when you need context about the user or past conversations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
            },
            "required": ["query"],
        },
    },
]


def handle_tool_call(brain: Aura, tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result."""
    if tool_name == "remember":
        tags = tool_input.get("tags", [])
        rid = brain.store(
            tool_input["content"], level=Level.Decisions, tags=tags
        )
        return f"Stored in memory (id={rid})"
    elif tool_name == "recall":
        results = brain.recall_structured(tool_input["query"], top_k=5)
        if not results:
            return "No relevant memories found."
        lines = [f"- {r['content']} (score={r['score']:.2f})" for r in results]
        return "Recalled memories:\n" + "\n".join(lines)
    return "Unknown tool"


def chat_with_tools(
    client: anthropic.Anthropic,
    brain: Aura,
    user_message: str,
    system_prompt: str,
) -> str:
    """Send a message to Claude with tool use, handling tool calls in a loop."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            tools=MEMORY_TOOLS,
            messages=messages,
        )

        # If no tool use, return the text
        if response.stop_reason == "end_turn":
            text_parts = [b.text for b in response.content if b.type == "text"]
            return " ".join(text_parts)

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = handle_tool_call(brain, block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
        messages.append({"role": "user", "content": tool_results})


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "Set ANTHROPIC_API_KEY env var"

    client = anthropic.Anthropic(api_key=api_key)
    brain = Aura(BRAIN_PATH)

    # Pre-load some knowledge
    brain.store(
        "User is a Python developer interested in AI",
        level=Level.Identity,
        tags=["user", "tech"],
    )

    print("=" * 60)
    print("  Aura + Claude SDK -- Persistent Memory Agent")
    print(f"  Records in memory: {brain.count()}")
    print("=" * 60)

    # ── Pattern 1: System prompt injection ──
    print("\n--- Pattern 1: System Prompt Injection ---")
    context = brain.recall("user background", token_budget=1500)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=(
            "You are a helpful assistant with persistent memory.\n\n"
            f"Recalled context:\n{context}"
        ),
        messages=[
            {"role": "user", "content": "What do you know about me?"}
        ],
    )
    reply = response.content[0].text
    print(f"  User: What do you know about me?")
    print(f"  Claude: {reply}")

    # ── Pattern 2: Tool use ──
    print("\n--- Pattern 2: Tool Use (Claude decides when to store/recall) ---")
    system = (
        "You are a helpful assistant with persistent memory tools.\n"
        "Use the 'remember' tool to store important facts about the user.\n"
        "Use the 'recall' tool when you need context from past conversations.\n"
        "Always try to recall relevant context before answering."
    )

    questions = [
        "I just moved to Tokyo and I'm learning Japanese. Remember that!",
        "What programming languages and cities are relevant to me?",
    ]

    for user_input in questions:
        reply = chat_with_tools(client, brain, user_input, system)
        brain.store(
            f"User: {user_input}", level=Level.Working, tags=["conversation"]
        )
        print(f"\n  User: {user_input}")
        print(f"  Claude: {reply}")

    # Maintenance
    brain.run_maintenance()
    print(f"\n  Total memories: {brain.count()}")
    brain.close()
    print(f"  Brain saved to {BRAIN_PATH}/")


if __name__ == "__main__":
    main()
