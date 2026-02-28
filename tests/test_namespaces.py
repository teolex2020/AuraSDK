"""Namespace isolation tests — store, recall, search, list, move, stats."""

import pytest
from aura import Aura, Level


class TestStoreWithNamespace:
    def test_store_default_namespace(self, brain):
        rid = brain.store("Default ns content")
        rec = brain.get(rid)
        assert rec.namespace == "default"

    def test_store_with_namespace(self, brain):
        rid = brain.store("Sandbox content", namespace="sandbox")
        rec = brain.get(rid)
        assert rec.namespace == "sandbox"

    def test_store_custom_namespace(self, brain):
        rid = brain.store("Project data", namespace="project-x")
        rec = brain.get(rid)
        assert rec.namespace == "project-x"


class TestRecallIsolation:
    def test_recall_only_sees_own_namespace(self, brain):
        brain.store("Python is great for scripting", namespace="default")
        brain.store("Python test case for sandbox", namespace="sandbox")

        results = brain.recall_structured("Python", namespace="default", top_k=10)
        for r in results:
            rec = brain.get(r["id"])
            assert rec.namespace == "default"

    def test_recall_sandbox_only(self, brain):
        brain.store("Real user data about health", namespace="default")
        brain.store("Test case health scenario", namespace="sandbox")

        results = brain.recall_structured("health", namespace="sandbox", top_k=10)
        for r in results:
            rec = brain.get(r["id"])
            assert rec.namespace == "sandbox"

    def test_recall_none_defaults_to_default(self, brain):
        brain.store("Content in default", namespace="default")
        brain.store("Content in sandbox", namespace="sandbox")

        # namespace=None should behave as "default"
        results_none = brain.recall_structured("Content", top_k=10)
        results_default = brain.recall_structured("Content", namespace="default", top_k=10)

        # Both should return the same records
        ids_none = {r["id"] for r in results_none}
        ids_default = {r["id"] for r in results_default}
        assert ids_none == ids_default


class TestSearchIsolation:
    def test_search_respects_namespace(self, brain):
        brain.store("Record A", tags=["test"], namespace="default")
        brain.store("Record B", tags=["test"], namespace="sandbox")

        results = brain.search(tags=["test"], namespace="default")
        assert len(results) == 1
        assert results[0].namespace == "default"

    def test_search_sandbox_namespace(self, brain):
        brain.store("Record A", tags=["demo"], namespace="default")
        brain.store("Record B", tags=["demo"], namespace="sandbox")
        brain.store("Record C", tags=["demo"], namespace="sandbox")

        results = brain.search(tags=["demo"], namespace="sandbox")
        assert len(results) == 2
        for rec in results:
            assert rec.namespace == "sandbox"


class TestListNamespaces:
    def test_list_namespaces_fresh_brain(self, brain):
        ns = brain.list_namespaces()
        assert isinstance(ns, list)
        # Fresh brain may have "default" or be empty depending on implementation
        assert len(ns) <= 1

    def test_list_namespaces(self, brain):
        brain.store("A", namespace="default")
        brain.store("B", namespace="sandbox")
        brain.store("C", namespace="project-x")

        ns = brain.list_namespaces()
        assert set(ns) == {"default", "sandbox", "project-x"}

    def test_list_namespaces_no_duplicates(self, brain):
        brain.store("A", namespace="sandbox", deduplicate=False)
        brain.store("B", namespace="sandbox", deduplicate=False)

        ns = brain.list_namespaces()
        assert ns.count("sandbox") == 1


class TestMoveRecord:
    def test_move_record(self, brain):
        rid = brain.store("Movable content", namespace="default")
        result = brain.move_record(rid, "sandbox")
        assert result is not None

        rec = brain.get(rid)
        assert rec.namespace == "sandbox"

    def test_move_record_nonexistent(self, brain):
        result = brain.move_record("nonexistent_id", "sandbox")
        assert result is None

    def test_move_changes_search_visibility(self, brain):
        rid = brain.store("Moving record", tags=["movable"], namespace="default")

        # Visible in default
        results = brain.search(tags=["movable"], namespace="default")
        assert len(results) == 1

        # Move to sandbox
        brain.move_record(rid, "sandbox")

        # No longer visible in default
        results = brain.search(tags=["movable"], namespace="default")
        assert len(results) == 0

        # Visible in sandbox
        results = brain.search(tags=["movable"], namespace="sandbox")
        assert len(results) == 1


class TestNamespaceStats:
    def test_namespace_stats(self, brain):
        brain.store("A", namespace="default", deduplicate=False)
        brain.store("B", namespace="default", deduplicate=False)
        brain.store("C", namespace="sandbox", deduplicate=False)

        stats = brain.namespace_stats()
        assert isinstance(stats, dict)
        assert stats["default"] == 2
        assert stats["sandbox"] == 1

    def test_namespace_stats_empty(self, brain):
        stats = brain.namespace_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0


class TestExportImportNamespace:
    def test_export_import_preserves_namespace(self, brain):
        import tempfile
        import os

        brain.store("Default content", namespace="default", deduplicate=False)
        brain.store("Sandbox content", namespace="sandbox", deduplicate=False)

        export = brain.export_json()

        with tempfile.TemporaryDirectory() as tmp:
            brain2 = Aura(os.path.join(tmp, "import.db"))
            brain2.import_json(export)

            stats = brain2.namespace_stats()
            assert stats.get("default", 0) >= 1
            assert stats.get("sandbox", 0) >= 1
            brain2.close()


class TestDedupWithinNamespace:
    def test_same_content_different_namespaces(self, brain):
        """Identical content in different namespaces should NOT be deduped."""
        id1 = brain.store("Identical content", namespace="default")
        id2 = brain.store("Identical content", namespace="sandbox")

        # Both should exist as separate records
        rec1 = brain.get(id1)
        rec2 = brain.get(id2)
        assert rec1 is not None
        assert rec2 is not None
        assert rec1.namespace == "default"
        assert rec2.namespace == "sandbox"


# ── Multi-namespace recall/search (v1.2.0) ──

class TestMultiNamespaceRecall:
    def test_recall_multi_namespace(self, brain):
        """namespace=["default", "sandbox"] returns results from both."""
        brain.store("User health data about vitamins", namespace="default")
        brain.store("Test health scenario about vitamins", namespace="sandbox")
        brain.store("Project health dashboard about vitamins", namespace="project-x")

        results = brain.recall_structured("health vitamins", namespace=["default", "sandbox"], top_k=10)
        found_ns = {brain.get(r["id"]).namespace for r in results}
        # Should find from default and/or sandbox but NOT project-x
        assert not any(brain.get(r["id"]).namespace == "project-x" for r in results)

    def test_search_multi_namespace(self, brain):
        """namespace=["default", "sandbox"] returns records from both."""
        brain.store("Record A", tags=["multi-test"], namespace="default")
        brain.store("Record B", tags=["multi-test"], namespace="sandbox")
        brain.store("Record C", tags=["multi-test"], namespace="project-x")

        results = brain.search(tags=["multi-test"], namespace=["default", "sandbox"])
        assert len(results) == 2
        found_ns = {r.namespace for r in results}
        assert found_ns == {"default", "sandbox"}

    def test_search_all_namespaces(self, brain):
        """namespace=["default", "sandbox", "project-x"] returns all."""
        brain.store("A", tags=["all-test"], namespace="default")
        brain.store("B", tags=["all-test"], namespace="sandbox")
        brain.store("C", tags=["all-test"], namespace="project-x")

        results = brain.search(tags=["all-test"], namespace=["default", "sandbox", "project-x"])
        assert len(results) == 3


class TestNamespaceBackwardCompat:
    def test_namespace_string_still_works(self, brain):
        """namespace="default" (str) should still work as before."""
        brain.store("Backward compat test content", namespace="default")

        results = brain.search(query="Backward compat", namespace="default")
        assert len(results) == 1

    def test_namespace_none_still_works(self, brain):
        """namespace=None should default to ["default"]."""
        brain.store("None namespace test content data")

        # None = default namespace
        results = brain.search(query="None namespace test")
        assert len(results) == 1
        assert results[0].namespace == "default"
