"""Aura + CrewAI: crew with persistent memory tools.

Agents use Aura tools to recall and store knowledge across tasks.
Memory persists between crew runs — no embeddings needed.

Requirements:
    pip install aura-memory crewai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/crewai_agent.py
"""

import os

from crewai import Agent, Task, Crew
from crewai.tools import tool

from aura import Aura, Level

BRAIN_PATH = "./crewai_data"

brain = Aura(BRAIN_PATH)


@tool("recall_memory")
def recall_memory(query: str) -> str:
    """Search long-term memory for relevant context about a topic."""
    return brain.recall(query, token_budget=2000)


@tool("store_memory")
def store_memory(content: str) -> str:
    """Store an important fact or decision in long-term memory."""
    rid = brain.store(content, level=Level.Decisions, tags=["crew-learned"])
    return f"Stored memory (id={rid})"


def main():
    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY env var"

    # ── Pre-load some knowledge ──
    brain.store(
        "Company tech stack: Python backend, React frontend, PostgreSQL database",
        level=Level.Domain,
        tags=["tech", "stack"],
    )
    brain.store(
        "Team preference: prefer simple solutions over complex architectures",
        level=Level.Decisions,
        tags=["team", "preference"],
    )

    print("=" * 60)
    print("  Aura + CrewAI -- Crew with Persistent Memory")
    print(f"  Records in memory: {brain.count()}")
    print("=" * 60)

    # ── Define agent ──
    analyst = Agent(
        role="Technical Analyst",
        goal="Provide recommendations based on team knowledge and preferences",
        backstory="You have access to the team's long-term memory via tools.",
        tools=[recall_memory, store_memory],
        verbose=True,
    )

    # ── Define task ──
    task = Task(
        description=(
            "First, recall what you know about our tech stack and team preferences. "
            "Then recommend a caching solution for our Python backend. "
            "Store your recommendation in memory for future reference."
        ),
        expected_output="A caching recommendation with reasoning based on team context.",
        agent=analyst,
    )

    # ── Run crew ──
    crew = Crew(agents=[analyst], tasks=[task], verbose=True)
    result = crew.kickoff()

    print(f"\n{'=' * 60}")
    print(f"  Result: {result.raw[:300]}")
    print(f"  Total memories: {brain.count()}")
    brain.close()
    print(f"  Brain saved to {BRAIN_PATH}/")


if __name__ == "__main__":
    main()
