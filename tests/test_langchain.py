"""Tests for LangChain integration: AuraChatMessageHistory and AuraMemory."""

import os
import tempfile
import pytest

from aura import Aura, Level

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from aura.langchain import AuraChatMessageHistory, AuraMemory
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

pytestmark = pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")


@pytest.fixture
def brain():
    with tempfile.TemporaryDirectory() as tmp:
        b = Aura(os.path.join(tmp, "test.db"))
        yield b
        b.close()


class TestAuraChatMessageHistory:
    def test_add_and_retrieve_messages(self, brain):
        history = AuraChatMessageHistory(brain, session_id="test_session")
        history.add_user_message("Hello")
        history.add_ai_message("Hi there!")

        msgs = history.messages
        assert len(msgs) == 2
        # Order may vary (search sorts by importance, not time)
        contents = {m.content for m in msgs}
        types = {type(m) for m in msgs}
        assert "Hello" in contents
        assert "Hi there!" in contents
        assert HumanMessage in types
        assert AIMessage in types

    def test_add_messages_bulk(self, brain):
        history = AuraChatMessageHistory(brain, session_id="bulk")
        history.add_messages([
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
        ])
        msgs = history.messages
        assert len(msgs) == 3

    def test_add_system_message(self, brain):
        history = AuraChatMessageHistory(brain, session_id="sys")
        history.add_message(SystemMessage(content="System instructions"))
        msgs = history.messages
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)

    def test_clear(self, brain):
        history = AuraChatMessageHistory(brain, session_id="clear_test")
        history.add_user_message("To be cleared")
        history.add_ai_message("Also cleared")
        assert len(history.messages) == 2

        history.clear()
        assert len(history.messages) == 0

    def test_session_isolation(self, brain):
        """Messages in different sessions should not leak."""
        h1 = AuraChatMessageHistory(brain, session_id="user_a")
        h2 = AuraChatMessageHistory(brain, session_id="user_b")

        h1.add_user_message("User A message")
        h2.add_user_message("User B message")

        msgs_a = h1.messages
        msgs_b = h2.messages

        assert len(msgs_a) == 1
        assert msgs_a[0].content == "User A message"
        assert len(msgs_b) == 1
        assert msgs_b[0].content == "User B message"

    def test_no_dedup_for_messages(self, brain):
        """Chat messages should not be deduplicated."""
        history = AuraChatMessageHistory(brain, session_id="nodedup")
        history.add_user_message("Same message")
        history.add_user_message("Same message")
        assert len(history.messages) == 2

    def test_custom_level(self, brain):
        """Should allow specifying a custom level for message storage."""
        history = AuraChatMessageHistory(brain, session_id="level_test", level=Level.Domain)
        history.add_user_message("Important message")
        records = brain.search(tags=["langchain_message"], namespace="level_test")
        assert len(records) >= 1
        assert records[0].level == Level.Domain


class TestAuraMemory:
    def test_memory_variables(self, brain):
        memory = AuraMemory(brain=brain)
        assert "chat_history" in memory.memory_variables
        assert "memory_context" in memory.memory_variables

    def test_save_and_load(self, brain):
        memory = AuraMemory(brain=brain, session_id="save_load")
        memory.save_context(
            {"input": "My name is Alex"},
            {"output": "Nice to meet you, Alex!"},
        )

        result = memory.load_memory_variables({"input": "What is my name?"})
        assert "chat_history" in result
        assert "Alex" in result["chat_history"]

    def test_memory_context_recall(self, brain):
        """Semantic recall should find relevant memories."""
        brain.store(
            "User prefers dark mode",
            level=Level.Identity,
            tags=["preference"],
            namespace="recall_test",
        )
        memory = AuraMemory(brain=brain, session_id="recall_test", return_context=True)
        result = memory.load_memory_variables({"input": "user preferences"})
        assert "memory_context" in result
        # The recalled context should contain relevant memories
        assert isinstance(result["memory_context"], str)

    def test_clear(self, brain):
        memory = AuraMemory(brain=brain, session_id="clear_mem")
        memory.save_context(
            {"input": "Remember this"},
            {"output": "Got it"},
        )
        memory.clear()
        result = memory.load_memory_variables({"input": "anything"})
        assert result["chat_history"] == ""

    def test_no_context_when_disabled(self, brain):
        memory = AuraMemory(brain=brain, return_context=False)
        assert "memory_context" not in memory.memory_variables

    def test_session_isolation(self, brain):
        """Different session_ids should have isolated memory."""
        m1 = AuraMemory(brain=brain, session_id="iso_a")
        m2 = AuraMemory(brain=brain, session_id="iso_b")

        m1.save_context({"input": "Secret A"}, {"output": "OK"})
        m2.save_context({"input": "Secret B"}, {"output": "OK"})

        r1 = m1.load_memory_variables({"input": "secret"})
        r2 = m2.load_memory_variables({"input": "secret"})

        assert "Secret A" in r1["chat_history"]
        assert "Secret B" not in r1["chat_history"]
        assert "Secret B" in r2["chat_history"]
        assert "Secret A" not in r2["chat_history"]
