"""Tests for event callbacks: on_store, on_recall, on_maintenance."""

import os
import tempfile
import pytest
from aura import Level
from aura.events import AuraEvents


@pytest.fixture
def brain():
    with tempfile.TemporaryDirectory() as tmp:
        b = AuraEvents(os.path.join(tmp, "test.db"))
        yield b
        b.close()


class TestOnStore:
    def test_on_store_fires(self, brain):
        """on_store callback should fire after store()."""
        fired = []
        brain.on_store(lambda rid, content, level, tags: fired.append(content))
        brain.store("Test event", level=Level.Working)
        assert len(fired) == 1
        assert fired[0] == "Test event"

    def test_on_store_receives_all_args(self, brain):
        """Callback receives record_id, content, level, tags."""
        captured = {}
        def cb(rid, content, level, tags):
            captured["rid"] = rid
            captured["content"] = content
            captured["level"] = level
            captured["tags"] = tags
        brain.on_store(cb)
        brain.store("Event data", level=Level.Domain, tags=["a", "b"])
        assert isinstance(captured["rid"], str)
        assert captured["content"] == "Event data"
        assert captured["level"] == Level.Domain
        assert captured["tags"] == ["a", "b"]

    def test_on_store_tag_filter(self, brain):
        """on_store with tags filter should only fire for matching tags."""
        fired = []
        brain.on_store(lambda rid, c, l, t: fired.append(c), tags=["important"])
        brain.store("Not important", level=Level.Working, tags=["misc"])
        brain.store("Very important", level=Level.Working, tags=["important"])
        assert len(fired) == 1
        assert "Very important" in fired[0]

    def test_on_store_level_filter(self, brain):
        """on_store with level filter should only fire for matching level."""
        fired = []
        brain.on_store(lambda rid, c, l, t: fired.append(c), level=Level.Identity)
        brain.store("Working memory", level=Level.Working)
        brain.store("Identity memory", level=Level.Identity)
        assert len(fired) == 1
        assert "Identity" in fired[0]

    def test_multiple_listeners(self, brain):
        """Multiple on_store listeners should all fire."""
        fired_a = []
        fired_b = []
        brain.on_store(lambda rid, c, l, t: fired_a.append(c))
        brain.on_store(lambda rid, c, l, t: fired_b.append(c))
        brain.store("Multi listener", level=Level.Working)
        assert len(fired_a) == 1
        assert len(fired_b) == 1

    def test_callback_error_does_not_break_store(self, brain):
        """A failing callback should not prevent store from succeeding."""
        def bad_callback(rid, c, l, t):
            raise ValueError("callback error")
        brain.on_store(bad_callback)
        rid = brain.store("Should still work", level=Level.Working)
        assert isinstance(rid, str)
        assert brain.count() >= 1


class TestOnRecall:
    def test_on_recall_fires(self, brain):
        """on_recall callback should fire after recall_structured()."""
        brain.store("Recall target", level=Level.Domain, tags=["test"])
        fired = []
        brain.on_recall(lambda q, r: fired.append((q, len(r) if isinstance(r, list) else 0)))
        brain.recall_structured("target", top_k=5)
        assert len(fired) == 1
        assert fired[0][0] == "target"

    def test_on_recall_text_fires(self, brain):
        """on_recall should also fire for recall() (text version)."""
        brain.store("Recall text target", level=Level.Domain)
        fired = []
        brain.on_recall(lambda q, r: fired.append(q))
        brain.recall("text target")
        assert len(fired) == 1

    def test_on_recall_receives_results(self, brain):
        """Callback should receive the actual recall results."""
        brain.store("Specific content for recall", level=Level.Domain, tags=["specific"])
        captured = {}
        brain.on_recall(lambda q, r: captured.update({"results": r}))
        brain.recall_structured("specific content", top_k=5)
        assert "results" in captured
        assert isinstance(captured["results"], list)


class TestOnMaintenance:
    def test_on_maintenance_fires(self, brain):
        """on_maintenance callback should fire after run_maintenance()."""
        brain.store("Some data", level=Level.Working)
        fired = []
        brain.on_maintenance(lambda report: fired.append(report))
        brain.run_maintenance()
        assert len(fired) == 1

    def test_on_maintenance_receives_report(self, brain):
        """Callback should receive the maintenance report."""
        brain.store("Data for maintenance", level=Level.Working)
        captured = {}
        brain.on_maintenance(lambda report: captured.update({"report": report}))
        brain.run_maintenance()
        assert "report" in captured


class TestOff:
    def test_off_unsubscribes(self, brain):
        """off() should prevent future callbacks."""
        fired = []
        handle = brain.on_store(lambda rid, c, l, t: fired.append(c))
        brain.store("Before off", level=Level.Working)
        assert len(fired) == 1
        brain.off(handle)
        brain.store("After off", level=Level.Working)
        assert len(fired) == 1  # still 1, not 2

    def test_off_returns_true_for_valid_handle(self, brain):
        handle = brain.on_store(lambda rid, c, l, t: None)
        assert brain.off(handle) is True

    def test_off_returns_false_for_invalid_handle(self, brain):
        assert brain.off(9999) is False

    def test_off_only_removes_target(self, brain):
        """off() should only remove the specified listener."""
        fired_a = []
        fired_b = []
        handle_a = brain.on_store(lambda rid, c, l, t: fired_a.append(c))
        brain.on_store(lambda rid, c, l, t: fired_b.append(c))
        brain.off(handle_a)
        brain.store("After partial off", level=Level.Working)
        assert len(fired_a) == 0
        assert len(fired_b) == 1


class TestProxy:
    def test_proxied_methods_work(self, brain):
        """Methods not intercepted should proxy to Rust Aura."""
        rid = brain.store("Proxy test", level=Level.Domain, tags=["proxy"])
        rec = brain.get(rid)
        assert rec is not None
        assert "Proxy test" in rec.content

    def test_count_works(self, brain):
        brain.store("Count test", level=Level.Working)
        assert brain.count() >= 1

    def test_search_works(self, brain):
        brain.store("Searchable content", level=Level.Domain, tags=["search"])
        results = brain.search(tags=["search"])
        assert len(results) >= 1
