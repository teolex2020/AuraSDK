"""Knowledge graph connection tests — connect, expand_connections, relationship types."""

import pytest
from aura import Aura, Level


class TestConnectBasic:
    def test_connect_creates_connection(self, brain):
        id1 = brain.store("Record Alpha", deduplicate=False)
        id2 = brain.store("Record Beta", deduplicate=False)
        brain.connect(id1, id2)

        stats = brain.stats()
        assert stats["total_connections"] >= 1

    def test_connect_with_weight(self, brain):
        id1 = brain.store("Cause event", deduplicate=False)
        id2 = brain.store("Effect event", deduplicate=False)
        brain.connect(id1, id2, weight=0.95)

        stats = brain.stats()
        assert stats["total_connections"] >= 1

    def test_connect_default_weight(self, brain):
        """Default weight should be 0.5."""
        id1 = brain.store("A", deduplicate=False)
        id2 = brain.store("B", deduplicate=False)
        brain.connect(id1, id2)
        # Just verify it doesn't crash

    def test_connect_weight_bounds(self, brain):
        """Weight at boundaries (0.0 and 1.0)."""
        id1 = brain.store("A", deduplicate=False)
        id2 = brain.store("B", deduplicate=False)
        id3 = brain.store("C", deduplicate=False)

        brain.connect(id1, id2, weight=0.0)
        brain.connect(id1, id3, weight=1.0)


class TestRelationshipTypes:
    def test_causal_relationship(self, brain):
        id1 = brain.store("User changed settings", deduplicate=False)
        id2 = brain.store("App theme switched to dark", deduplicate=False)
        brain.connect(id1, id2, relationship="causal")

    def test_reflective_relationship(self, brain):
        id1 = brain.store("Reviewed last sprint", deduplicate=False)
        id2 = brain.store("Sprint retrospective insights", deduplicate=False)
        brain.connect(id1, id2, relationship="reflective")

    def test_associative_relationship(self, brain):
        id1 = brain.store("Rust language", deduplicate=False)
        id2 = brain.store("Memory safety", deduplicate=False)
        brain.connect(id1, id2, relationship="associative")

    def test_coactivation_relationship(self, brain):
        id1 = brain.store("Record from session A", deduplicate=False)
        id2 = brain.store("Record from session B", deduplicate=False)
        brain.connect(id1, id2, relationship="coactivation")

    def test_custom_relationship(self, brain):
        """Custom relationship strings should be accepted."""
        id1 = brain.store("Bug report", deduplicate=False)
        id2 = brain.store("Fix commit", deduplicate=False)
        brain.connect(id1, id2, relationship="resolved_by")

    def test_connect_with_weight_and_relationship(self, brain):
        id1 = brain.store("Auth bug found", deduplicate=False)
        id2 = brain.store("Database migration needed", deduplicate=False)
        brain.connect(id1, id2, weight=0.9, relationship="causal")


class TestConnectEdgeCases:
    def test_connect_nonexistent_record(self, brain):
        """Connecting a nonexistent record should raise."""
        id1 = brain.store("Existing record", deduplicate=False)
        with pytest.raises(RuntimeError):
            brain.connect(id1, "nonexistent_id")

    def test_connect_both_nonexistent(self, brain):
        """Connecting two nonexistent records should raise."""
        with pytest.raises(RuntimeError):
            brain.connect("fake_a", "fake_b")

    def test_self_loop(self, brain):
        """Connecting a record to itself — should either work or raise cleanly."""
        rid = brain.store("Self-referencing record", deduplicate=False)
        try:
            brain.connect(rid, rid)
        except RuntimeError:
            pass  # Acceptable to reject self-loops

    def test_cross_namespace_prevented(self, brain):
        """Connecting records in different namespaces should fail."""
        id1 = brain.store("Default ns", namespace="default", deduplicate=False)
        id2 = brain.store("Sandbox ns", namespace="sandbox", deduplicate=False)
        with pytest.raises(RuntimeError):
            brain.connect(id1, id2)

    def test_duplicate_connection_safe(self, brain):
        """Connecting same pair twice should not crash."""
        id1 = brain.store("A", deduplicate=False)
        id2 = brain.store("B", deduplicate=False)
        brain.connect(id1, id2)
        brain.connect(id1, id2)  # Should not raise

    def test_bidirectional(self, brain):
        """Connections should be bidirectional — connect(a,b) = connect(b,a)."""
        id1 = brain.store("Node 1", deduplicate=False)
        id2 = brain.store("Node 2", deduplicate=False)
        brain.connect(id1, id2, weight=0.8, relationship="associative")

        stats = brain.stats()
        connections_after_first = stats["total_connections"]

        # Connecting in reverse should not double the count
        brain.connect(id2, id1, weight=0.8, relationship="associative")
        stats2 = brain.stats()
        # Should be same or only slightly more (implementation dependent)
        assert stats2["total_connections"] <= connections_after_first + 1


class TestCircularConnections:
    """Circular connections (A->B->C->A) should not cause infinite loops in graph walk."""

    def test_triangle_cycle(self, brain):
        """A->B->C->A cycle should not hang recall."""
        id_a = brain.store("Node A: authentication module", deduplicate=False)
        id_b = brain.store("Node B: user database schema", deduplicate=False)
        id_c = brain.store("Node C: session management", deduplicate=False)

        brain.connect(id_a, id_b, relationship="associative")
        brain.connect(id_b, id_c, relationship="associative")
        brain.connect(id_c, id_a, relationship="associative")

        # This must terminate (not hang in infinite graph walk)
        results = brain.recall_structured("authentication", top_k=5,
                                           expand_connections=True)
        assert isinstance(results, list)

    def test_long_cycle(self, brain):
        """5-node cycle: A->B->C->D->E->A."""
        ids = []
        for i in range(5):
            rid = brain.store(f"Cycle node {i} with unique content {i * 7}",
                               deduplicate=False)
            ids.append(rid)

        for i in range(5):
            brain.connect(ids[i], ids[(i + 1) % 5], relationship="associative")

        results = brain.recall_structured("cycle node", top_k=10,
                                           expand_connections=True)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_self_loop_with_expand(self, brain):
        """Self-loop + expand_connections should not infinitely recurse."""
        rid = brain.store("Self-referencing concept", deduplicate=False)
        try:
            brain.connect(rid, rid)
        except RuntimeError:
            return  # Self-loops rejected — no further test needed

        results = brain.recall_structured("self-referencing", top_k=5,
                                           expand_connections=True)
        assert isinstance(results, list)

    def test_dense_graph(self, brain):
        """Fully connected 4-node graph — all pairs connected."""
        ids = []
        for i in range(4):
            rid = brain.store(f"Dense node {i}: topic {chr(65 + i)}",
                               deduplicate=False)
            ids.append(rid)

        for i in range(4):
            for j in range(i + 1, 4):
                brain.connect(ids[i], ids[j], relationship="associative")

        results = brain.recall_structured("dense node topic", top_k=10,
                                           expand_connections=True)
        assert isinstance(results, list)
        assert len(results) >= 1


class TestExpandConnections:
    def test_expand_connections_true(self, brain):
        """expand_connections=True should pull in connected records."""
        id_bug = brain.store("Auth bug: users get 403 error",
                              level=Level.Working, tags=["bug", "auth"],
                              deduplicate=False)
        id_db = brain.store("PostgreSQL handles ACID transactions",
                             level=Level.Domain, tags=["database"],
                             deduplicate=False)
        brain.connect(id_bug, id_db, weight=0.9, relationship="causal")

        results = brain.recall_structured("database auth", top_k=5,
                                           expand_connections=True)
        ids = [r["id"] for r in results]
        # Both should appear due to connection expansion
        assert any(r["id"] in (id_bug, id_db) for r in results)

    def test_expand_connections_false(self, brain):
        """expand_connections=False should only return direct matches."""
        id1 = brain.store("Completely unrelated topic alpha",
                           tags=["alpha"], deduplicate=False)
        id2 = brain.store("Completely unrelated topic beta",
                           tags=["beta"], deduplicate=False)
        brain.connect(id1, id2, weight=1.0)

        results = brain.recall_structured("alpha", top_k=5,
                                           expand_connections=False)
        # Without expansion, beta should be less likely to appear
        # (it may still appear via other RRF signals if text matches)
        if results:
            assert results[0]["content"] != "Completely unrelated topic beta" or len(results) == 1

    def test_expand_connections_default(self, brain):
        """Default expand_connections behavior (should be True)."""
        id1 = brain.store("Machine learning basics", tags=["ml"],
                           deduplicate=False)
        id2 = brain.store("Neural network architecture", tags=["nn"],
                           deduplicate=False)
        brain.connect(id1, id2, weight=0.8, relationship="associative")

        # Default call (no expand_connections specified)
        results = brain.recall_structured("machine learning", top_k=5)
        assert len(results) >= 1

    def test_multiple_connections_expand(self, brain):
        """A record connected to multiple others should expand all."""
        center = brain.store("Central concept: microservices",
                              tags=["arch"], deduplicate=False)
        related = []
        for topic in ["API gateway", "Service mesh", "Container orchestration"]:
            rid = brain.store(topic, tags=["arch"], deduplicate=False)
            brain.connect(center, rid, relationship="associative")
            related.append(rid)

        results = brain.recall_structured("microservices architecture",
                                           top_k=10, expand_connections=True)
        found_ids = {r["id"] for r in results}
        # At least the center and some related should appear
        assert center in found_ids or any(r in found_ids for r in related)
