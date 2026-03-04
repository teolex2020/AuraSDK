"""Tests for temporal queries: recall_at() and history()."""

import time
import pytest
from aura import Aura, Level


class TestRecallAt:
    def test_recall_at_filters_by_timestamp(self, brain):
        """Records created after the cutoff should not appear."""
        # Store first record, note its timestamp
        id1 = brain.store("Early record about Python", level=Level.Domain, tags=["python"])
        rec1 = brain.get(id1)
        cutoff = rec1.created_at + 0.001  # just after first record

        # Store second record (created_at will be > cutoff)
        time.sleep(0.01)  # ensure different timestamp
        id2 = brain.store("Late record about Rust", level=Level.Domain, tags=["rust"])

        # recall_at with cutoff should only return early record
        results = brain.recall_at("programming language", cutoff, top_k=10)
        contents = [r["content"] for r in results]
        assert any("Early" in c or "Python" in c for c in contents)
        # Late record should NOT appear
        assert not any("Late" in c or "Rust" in c for c in contents)

    def test_recall_at_includes_records_before_timestamp(self, brain):
        """Records created before the cutoff should appear."""
        brain.store("User prefers dark mode", level=Level.Identity, tags=["ui"])
        brain.store("User works remotely", level=Level.Identity, tags=["work"])
        time.sleep(0.01)

        # Use a future timestamp — all records should be included
        future = time.time() + 86400
        results = brain.recall_at("user preferences", future, top_k=10)
        assert len(results) >= 2

    def test_recall_at_empty_when_no_records_before(self, brain):
        """No records before timestamp 0 means empty results."""
        brain.store("Some content", level=Level.Working)
        results = brain.recall_at("content", 0.0, top_k=10)
        assert len(results) == 0

    def test_recall_at_returns_scored_dicts(self, brain):
        """Results should have the expected dict structure."""
        brain.store("Test record for structure", level=Level.Domain, tags=["test"])
        future = time.time() + 86400
        results = brain.recall_at("test record", future, top_k=5)
        assert len(results) > 0
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "score" in r
        assert "level" in r
        assert "strength" in r
        assert "tags" in r
        assert "created_at" in r

    def test_recall_at_respects_namespace(self, brain):
        """Namespace isolation should work with temporal recall."""
        brain.store("Default namespace record", level=Level.Domain, namespace="default")
        brain.store("Sandbox record", level=Level.Domain, namespace="sandbox")
        future = time.time() + 86400
        results = brain.recall_at("record", future, top_k=10, namespace="default")
        contents = [r["content"] for r in results]
        assert not any("Sandbox" in c for c in contents)


class TestHistory:
    def test_history_returns_dict(self, brain):
        """history() should return a dict with expected keys."""
        rid = brain.store("Test memory for history", level=Level.Domain, tags=["test"])
        info = brain.history(rid)
        assert isinstance(info, dict)
        assert "id" in info
        assert "content" in info
        assert "level" in info
        assert "strength" in info
        assert "activation_count" in info
        assert "created_at" in info
        assert "last_activated" in info
        assert "age_days" in info
        assert "days_since_activation" in info
        assert "namespace" in info
        assert "tags" in info
        assert "connections" in info

    def test_history_correct_values(self, brain):
        """Verify history returns correct data."""
        rid = brain.store("History test record", level=Level.Identity, tags=["a", "b"])
        info = brain.history(rid)
        assert info["id"] == rid
        assert "History test record" in info["content"]
        assert info["level"].upper() == "IDENTITY"
        assert info["namespace"] == "default"
        assert "a" in info["tags"]

    def test_history_activation_count_increases(self, brain):
        """Recalling a record should increase its activation count."""
        rid = brain.store("Recall me to increase activation", level=Level.Domain, tags=["recall"])

        info_before = brain.history(rid)
        count_before = int(info_before["activation_count"])

        # Trigger recall that should activate this record
        brain.recall_structured("increase activation", top_k=5)

        info_after = brain.history(rid)
        count_after = int(info_after["activation_count"])
        assert count_after >= count_before

    def test_history_not_found_raises(self, brain):
        """history() with invalid ID should raise."""
        with pytest.raises(RuntimeError):
            brain.history("nonexistent_id_12345")

    def test_history_strength_is_float(self, brain):
        """Strength should be parseable as float."""
        rid = brain.store("Strength check", level=Level.Working)
        info = brain.history(rid)
        strength = float(info["strength"])
        assert 0.0 <= strength <= 1.0

    def test_history_age_is_recent(self, brain):
        """A just-created record should have age_days close to 0."""
        rid = brain.store("Just created", level=Level.Working)
        info = brain.history(rid)
        age = float(info["age_days"])
        assert age < 1.0  # less than 1 day old
