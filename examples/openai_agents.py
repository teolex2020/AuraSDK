"""Aura + OpenAI Agents SDK: agent with persistent memory via dynamic instructions.

The agent's system prompt is generated dynamically on each turn,
injecting Aura's recalled context. A store tool lets the agent
explicitly save important information.

Requirements:
    pip install aura-memory openai-agents
    export OPENAI_API_KEY=sk-...

Run:
    python examples/openai_agents.py
"""

import asyncio
import os
from dataclasses import dataclass

from agents import Agent, Runner, RunContextWrapper, function_tool

from aura import Aura, Level

BRAIN_PATH = "./openai_agents_data"


@dataclass
class Ctx:
    """Shared context passed to every agent run."""
    brain: Aura


def make_instructions(ctx: RunContextWrapper[Ctx], agent: Agent[Ctx]) -> str:
    """Dynamic instructions — called before each LLM turn."""
    context = ctx.context.brain.recall("current conversation context", token_budget=1500)
    return (
        "You are a helpful assistant with persistent memory.\n"
        "Use the recalled context below to personalize your answers.\n\n"
        f"{context}"
    )


@function_tool
def store_memory(ctx: RunContextWrapper[Ctx], content: str, tags: str = "") -> str:
    """Store an important fact in long-term memory.

    Args:
        content: The information to remember.
        tags: Comma-separated tags for categorization.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    rid = ctx.context.brain.store(content, level=Level.Decisions, tags=tag_list)
    return f"Stored (id={rid})"


async def main():
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var"

    brain = Aura(BRAIN_PATH)

    agent = Agent[Ctx](
        name="Nova",
        instructions=make_instructions,
        model="gpt-4o-mini",
        tools=[store_memory],
    )

    print("=" * 60)
    print("  Aura + OpenAI Agents SDK — Dynamic Memory Instructions")
    print(f"  Records in memory: {brain.count()}")
    print("=" * 60)

    ctx = Ctx(brain=brain)

    # ── Demo conversation ──
    questions = [
        "Remember that my favorite language is Rust and I live in Berlin.",
        "What do you know about me?",
        "Suggest a side project idea based on what you know about me.",
    ]

    for user_input in questions:
        brain.store(f"User: {user_input}", level=Level.Working, tags=["conversation"])

        result = await Runner.run(agent, input=user_input, context=ctx)

        brain.store(f"Assistant: {result.final_output[:200]}", level=Level.Working, tags=["conversation"])

        print(f"\n  User: {user_input}")
        print(f"  Nova: {result.final_output}")

    print(f"\n  Total memories: {brain.count()}")
    brain.close()
    print(f"  Brain saved to {BRAIN_PATH}/")


if __name__ == "__main__":
    asyncio.run(main())
