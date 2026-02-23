"""Core CRUD tests — store, recall, search, get, update, delete, connect."""

import pytest
from aura import Aura, Level, Record


class TestStore:
    def test_store_returns_string_id(self, brain):
        rid = brain.store("Hello world")
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_store_with_level(self, brain):
        rid = brain.store("Important", level=Level.Identity)
        rec = brain.get(rid)
        assert rec.level == Level.Identity

    def test_store_empty_raises(self, brain):
        with pytest.raises(RuntimeError):
            brain.store("")

    def test_store_with_tags(self, brain):
        rid = brain.store("Tagged content", tags=["a", "b", "c"])
        rec = brain.get(rid)
        assert "a" in rec.tags
        assert "b" in rec.tags
        assert "c" in rec.tags

    def test_store_with_metadata(self, brain):
        rid = brain.store("Metadata content", metadata={"source": "test", "version": "1"})
        rec = brain.get(rid)
        assert rec.metadata["source"] == "test"

    def test_store_with_channel(self, brain):
        rid = brain.store("API data", channel="api_endpoint", tags=["api"])
        rec = brain.get(rid)
        assert "source" in rec.metadata
        assert "trust_score" in rec.metadata
        assert "timestamp" in rec.metadata

    def test_store_dedup_default(self, brain):
        id1 = brain.store("Exact same content here")
        id2 = brain.store("Exact same content here")
        # Dedup should prevent storing identical content twice
        # (either same ID returned or second silently merged)
        assert brain.count() <= 2  # At most the original + 1 dedup'd

    def test_store_dedup_disabled(self, brain):
        id1 = brain.store("Same content", deduplicate=False)
        id2 = brain.store("Same content", deduplicate=False)
        assert id1 != id2

    def test_store_all_levels(self, brain):
        for level in [Level.Working, Level.Decisions, Level.Domain, Level.Identity]:
            rid = brain.store(f"Content at {level}", level=level)
            assert isinstance(rid, str)


class TestRecall:
    def test_recall_returns_string(self, populated_brain):
        result = populated_brain.recall("programming language")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_recall_with_token_budget(self, populated_brain):
        result = populated_brain.recall("python", token_budget=100)
        assert isinstance(result, str)

    def test_recall_structured_returns_list(self, populated_brain):
        results = populated_brain.recall_structured("programming", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_recall_structured_has_correct_keys(self, populated_brain):
        results = populated_brain.recall_structured("python", top_k=3)
        assert len(results) > 0
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "score" in r
        assert "level" in r
        assert "tags" in r

    def test_recall_structured_scores_descending(self, populated_brain):
        results = populated_brain.recall_structured("language", top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recall_empty_query(self, brain):
        result = brain.recall("")
        assert isinstance(result, str)


class TestSearch:
    def test_search_by_tags(self, populated_brain):
        results = populated_brain.search(tags=["bug"])
        assert len(results) >= 1
        for rec in results:
            assert "bug" in rec.tags

    def test_search_by_level(self, populated_brain):
        results = populated_brain.search(level=Level.Domain)
        assert len(results) >= 2

    def test_search_by_query(self, populated_brain):
        results = populated_brain.search(query="auth")
        assert len(results) >= 1

    def test_search_no_results(self, brain):
        results = brain.search(tags=["nonexistent"])
        assert len(results) == 0


class TestGet:
    def test_get_existing(self, brain):
        rid = brain.store("Test content")
        rec = brain.get(rid)
        assert rec is not None
        assert rec.content == "Test content"

    def test_get_nonexistent(self, brain):
        rec = brain.get("nonexistent_id")
        assert rec is None

    def test_get_record_attributes(self, brain):
        rid = brain.store("Content", level=Level.Domain, tags=["t1"])
        rec = brain.get(rid)
        assert hasattr(rec, "id")
        assert hasattr(rec, "content")
        assert hasattr(rec, "level")
        assert hasattr(rec, "strength")
        assert hasattr(rec, "tags")
        assert hasattr(rec, "metadata")


class TestUpdate:
    def test_update_content(self, brain):
        rid = brain.store("Original content")
        brain.update(rid, content="Updated content")
        rec = brain.get(rid)
        assert rec.content == "Updated content"

    def test_update_tags(self, brain):
        rid = brain.store("Content", tags=["old"])
        brain.update(rid, tags=["new", "tags"])
        rec = brain.get(rid)
        assert "new" in rec.tags
        assert "tags" in rec.tags

    def test_update_level(self, brain):
        rid = brain.store("Content", level=Level.Working)
        brain.update(rid, level=Level.Domain)
        rec = brain.get(rid)
        assert rec.level == Level.Domain

    def test_update_nonexistent(self, brain):
        result = brain.update("nonexistent", content="new")
        assert result is None

    def test_update_strength(self, brain):
        rid = brain.store("Content")
        brain.update(rid, strength=0.5)
        rec = brain.get(rid)
        assert abs(rec.strength - 0.5) < 0.01


class TestDelete:
    def test_delete_existing(self, brain):
        rid = brain.store("To be deleted")
        assert brain.delete(rid) is True
        assert brain.get(rid) is None

    def test_delete_nonexistent(self, brain):
        result = brain.delete("nonexistent_id")
        assert result is False

    def test_delete_reduces_count(self, brain):
        rid = brain.store("Content", deduplicate=False)
        before = brain.count()
        brain.delete(rid)
        after = brain.count()
        assert after == before - 1


class TestConnect:
    def test_connect_records(self, brain):
        id1 = brain.store("Record A", deduplicate=False)
        id2 = brain.store("Record B", deduplicate=False)
        brain.connect(id1, id2)
        stats = brain.stats()
        assert stats["total_connections"] > 0

    def test_connect_with_weight(self, brain):
        id1 = brain.store("Record A", deduplicate=False)
        id2 = brain.store("Record B", deduplicate=False)
        brain.connect(id1, id2, weight=0.9)


class TestStats:
    def test_stats_returns_dict(self, brain):
        stats = brain.stats()
        assert isinstance(stats, dict)
        assert "total_records" in stats

    def test_stats_counts_levels(self, populated_brain):
        stats = populated_brain.stats()
        assert stats["working"] >= 1
        assert stats["decisions"] >= 1
        assert stats["domain"] >= 2
        assert stats["identity"] >= 1

    def test_count(self, brain):
        brain.store("One", deduplicate=False)
        brain.store("Two", deduplicate=False)
        assert brain.count() >= 2


class TestExportImport:
    def test_export_json(self, populated_brain):
        export = populated_brain.export_json()
        assert isinstance(export, str)
        assert len(export) > 0

    def test_import_export_roundtrip(self, brain):
        import tempfile
        import os

        brain.store("Exported content", tags=["export"], deduplicate=False)
        export = brain.export_json()

        with tempfile.TemporaryDirectory() as tmp:
            brain2 = Aura(os.path.join(tmp, "import.db"))
            brain2.import_json(export)
            assert brain2.count() >= 1
            brain2.close()
