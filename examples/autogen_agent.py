"""Aura + AutoGen: agent with Aura as a Memory backend.

Implements AutoGen's Memory protocol so the agent automatically
recalls context before each LLM call.

Requirements:
    pip install aura-memory autogen-agentchat autogen-ext[openai]
    export OPENAI_API_KEY=sk-...

Run:
    python examples/autogen_agent.py
"""

import asyncio
import os
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import Memory, MemoryContent, MemoryQueryResult, UpdateContextResult
from autogen_core.model_context import ChatCompletionContext
from autogen_ext.models.openai import OpenAIChatCompletionClient

from aura import Aura, Level

BRAIN_PATH = "./autogen_data"


class AuraMemory(Memory):
    """AutoGen Memory backed by Aura SDK."""

    def __init__(self, path: str):
        self.brain = Aura(path)
        self.name = "aura"

    async def add(self, content: MemoryContent, cancellation_token: Any = None) -> None:
        text = str(content.content) if not isinstance(content.content, str) else content.content
        self.brain.store(text, level=Level.Working, tags=["autogen"])

    async def query(
        self, query: MemoryContent, cancellation_token: Any = None, **kwargs: Any
    ) -> MemoryQueryResult:
        text = str(query.content) if not isinstance(query.content, str) else query.content
        results = self.brain.recall_structured(text, top_k=10)
        memory_results = [
            MemoryContent(content=r["content"], mime_type="text/plain")
            for r in results
        ]
        return MemoryQueryResult(results=memory_results)

    async def update_context(
        self, model_context: ChatCompletionContext, cancellation_token: Any = None
    ) -> UpdateContextResult:
        """Called before each LLM call — inject Aura context."""
        messages = await model_context.get_messages()
        last_user = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                last_user = msg.content
                break

        if last_user:
            context = self.brain.recall(last_user, token_budget=1500)
            if context.strip():
                from autogen_core.models import SystemMessage
                await model_context.add_message(
                    SystemMessage(content=f"Recalled memories:\n{context}")
                )
                return UpdateContextResult(
                    memories=MemoryQueryResult(results=[
                        MemoryContent(content=context, mime_type="text/plain")
                    ])
                )

        return UpdateContextResult(memories=MemoryQueryResult(results=[]))

    async def clear(self) -> None:
        pass

    async def close(self) -> None:
        self.brain.close()


async def main():
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var"

    memory = AuraMemory(BRAIN_PATH)

    # Pre-load some knowledge
    memory.brain.store(
        "User is a Python developer who prefers async code",
        level=Level.Identity,
        tags=["preference"],
    )

    model = OpenAIChatCompletionClient(model="gpt-4o-mini")
    agent = AssistantAgent(
        name="assistant",
        model_client=model,
        memory=[memory],
        system_message="You are a helpful assistant with persistent memory.",
    )

    print("=" * 60)
    print("  Aura + AutoGen -- Memory Protocol Integration")
    print(f"  Records in memory: {memory.brain.count()}")
    print("=" * 60)

    # ── Demo: single task ──
    from autogen_agentchat.base import TaskResult
    stream = agent.run_stream(task="What do you know about my coding preferences? Suggest a project.")
    await Console(stream)

    print(f"\n  Total memories: {memory.brain.count()}")
    await memory.close()
    print(f"  Brain saved to {BRAIN_PATH}/")


if __name__ == "__main__":
    asyncio.run(main())
