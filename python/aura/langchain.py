"""LangChain integration for Aura — drop-in memory classes.

Usage with LCEL (new LangChain):
    from aura import Aura
    from aura.langchain import AuraChatMessageHistory

    brain = Aura("./data")
    history = AuraChatMessageHistory(brain, session_id="user_123")

Usage with legacy chains:
    from aura.langchain import AuraMemory

    brain = Aura("./data")
    chain = ConversationChain(llm=llm, memory=AuraMemory(brain))

Requirements:
    pip install aura-memory langchain-core
"""

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with: pip install langchain-core"
    ) from e

from typing import Any, Sequence

from aura._core import Aura as _Aura
from aura._core import Level


class AuraChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by Aura.

    Each message is stored as a record in Aura with cognitive decay.
    The session_id maps to an Aura namespace for per-user isolation.

    Unlike a simple buffer, Aura automatically:
    - Decays old messages over time (Working level)
    - Promotes frequently accessed messages (to Domain/Identity)
    - Deduplicates near-identical content

    Example:
        brain = Aura("./data")
        history = AuraChatMessageHistory(brain, session_id="user_123")
        history.add_user_message("I prefer dark mode")
        history.add_ai_message("Noted! I'll remember that.")
        print(history.messages)  # returns list of BaseMessage
    """

    def __init__(
        self,
        brain: _Aura,
        session_id: str = "default",
        level: Level = Level.Working,
    ):
        self.brain = brain
        self.session_id = session_id
        self.level = level

    @property
    def messages(self) -> list[BaseMessage]:
        """Return all messages for this session, ordered by creation time."""
        records = self.brain.search(
            tags=["langchain_message"],
            limit=1000,
            namespace=self.session_id,
        )
        msgs = []
        for rec in records:
            role = rec.metadata.get("role", "human")
            if role == "human":
                msgs.append(HumanMessage(content=rec.content))
            elif role == "ai":
                msgs.append(AIMessage(content=rec.content))
            elif role == "system":
                msgs.append(SystemMessage(content=rec.content))
            else:
                msgs.append(HumanMessage(content=rec.content))
        return msgs

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Bulk add messages to Aura."""
        for msg in messages:
            role = _message_role(msg)
            self.brain.store(
                msg.content,
                level=self.level,
                tags=["langchain_message", f"role:{role}"],
                metadata={"role": role, "session": self.session_id},
                namespace=self.session_id,
                deduplicate=False,
            )

    def add_message(self, message: BaseMessage) -> None:
        """Add a single message to Aura."""
        self.add_messages([message])

    def clear(self) -> None:
        """Remove all messages for this session.

        Note: Aura doesn't have a bulk-delete-by-namespace API,
        so we delete messages one by one. For production use with
        large histories, consider using a separate brain per session.
        """
        records = self.brain.search(
            tags=["langchain_message"],
            limit=10000,
            namespace=self.session_id,
        )
        for rec in records:
            self.brain.delete(rec.id)


class AuraMemory:
    """LangChain-compatible Memory backed by Aura.

    Provides two memory variables:
    - chat_history: formatted conversation history
    - memory_context: semantically recalled context from all sessions

    The memory_context is what makes Aura different from BufferMemory:
    it recalls relevant facts from ALL past conversations, not just
    the current chat buffer.

    Compatible with LangChain's memory protocol via duck typing.
    Works as a drop-in for any component that calls
    load_memory_variables() and save_context().

    Example:
        brain = Aura("./data")
        memory = AuraMemory(brain, session_id="user_123")
        memory.save_context({"input": "I like Rust"}, {"output": "Noted!"})
        memory.load_memory_variables({"input": "what do I like?"})
    """

    def __init__(
        self,
        brain: _Aura,
        session_id: str = "default",
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        memory_key: str = "chat_history",
        context_key: str = "memory_context",
        input_key: str = "input",
        output_key: str = "output",
        token_budget: int = 2000,
        return_context: bool = True,
    ):
        self.brain = brain
        self.session_id = session_id
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.memory_key = memory_key
        self.context_key = context_key
        self.input_key = input_key
        self.output_key = output_key
        self.token_budget = token_budget
        self.return_context = return_context

    @property
    def memory_variables(self) -> list[str]:
        """Keys this memory provides to the chain."""
        keys = [self.memory_key]
        if self.return_context:
            keys.append(self.context_key)
        return keys

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load memory for the chain.

        Returns chat_history (recent messages) and memory_context
        (semantically recalled facts from ALL sessions).
        """
        result = {}

        # Chat history from this session
        records = self.brain.search(
            tags=["langchain_message"],
            limit=20,
            namespace=self.session_id,
        )
        lines = []
        for rec in records:
            role = rec.metadata.get("role", "human")
            prefix = self.human_prefix if role == "human" else self.ai_prefix
            lines.append(f"{prefix}: {rec.content}")
        result[self.memory_key] = "\n".join(lines)

        # Semantic recall across all memory (the Aura advantage)
        if self.return_context:
            query = ""
            if isinstance(inputs, dict):
                query = inputs.get(self.input_key, "")
            if query:
                context = self.brain.recall(
                    query,
                    token_budget=self.token_budget,
                    namespace=self.session_id,
                )
            else:
                context = ""
            result[self.context_key] = context

        return result

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Store the conversation turn in Aura."""
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        if user_input:
            self.brain.store(
                user_input,
                level=Level.Working,
                tags=["langchain_message", "role:human"],
                metadata={"role": "human", "session": self.session_id},
                namespace=self.session_id,
                deduplicate=False,
            )
        if ai_output:
            self.brain.store(
                ai_output,
                level=Level.Working,
                tags=["langchain_message", "role:ai"],
                metadata={"role": "ai", "session": self.session_id},
                namespace=self.session_id,
                deduplicate=False,
            )

    def clear(self) -> None:
        """Clear all messages for this session."""
        records = self.brain.search(
            tags=["langchain_message"],
            limit=10000,
            namespace=self.session_id,
        )
        for rec in records:
            self.brain.delete(rec.id)


def _message_role(msg: BaseMessage) -> str:
    """Extract the role string from a LangChain message."""
    if isinstance(msg, HumanMessage):
        return "human"
    elif isinstance(msg, AIMessage):
        return "ai"
    elif isinstance(msg, SystemMessage):
        return "system"
    return "human"
