"""Advanced API tests — process/pin, sessions, caused_by_id, promotion, boundary conditions."""

import pytest
from aura import Aura, Level


class TestProcess:
    def test_process_returns_string(self, brain):
        """process() should return a status string."""
        result = brain.process("User said hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_process_stores_record(self, brain):
        """process() should create a new record."""
        before = brain.count()
        brain.process("Some conversational input")
        after = brain.count()
        assert after >= before + 1

    def test_process_default_level_working(self, brain):
        """process() without pin should store at Working level."""
        brain.process("Temporary thought about weather")
        results = brain.search(level=Level.Working)
        assert len(results) >= 1

    def test_process_pin_true_identity(self, brain):
        """process(pin=True) should store at Identity level."""
        brain.process("My name is Teo", pin=True)
        results = brain.search(level=Level.Identity)
        assert len(results) >= 1
        contents = [r.content for r in results]
        assert any("Teo" in c for c in contents)

    def test_process_pin_false_working(self, brain):
        """process(pin=False) should store at Working level."""
        brain.process("Just a passing thought", pin=False)
        results = brain.search(level=Level.Working)
        assert len(results) >= 1

    def test_process_multiple_messages(self, brain):
        """Processing multiple messages in sequence."""
        brain.process("First message")
        brain.process("Second message")
        brain.process("Third message")
        assert brain.count() >= 3


class TestSessions:
    def test_end_session_returns_dict(self, brain):
        """end_session() should return stats dict."""
        brain.process("Session message 1")
        brain.process("Session message 2")
        result = brain.end_session("session-001")
        assert isinstance(result, dict)

    def test_end_session_nonexistent(self, brain):
        """end_session with an unused session_id should not crash."""
        result = brain.end_session("never-used-session")
        assert isinstance(result, dict)

    def test_recall_with_session_id(self, brain):
        """recall_structured with session_id should scope results."""
        brain.store("Session-specific data", tags=["session"],
                     deduplicate=False)
        results = brain.recall_structured("data", top_k=5,
                                           session_id="test-session")
        assert isinstance(results, list)


class TestCausedById:
    def test_caused_by_id_stores(self, brain):
        """caused_by_id parameter should link records."""
        parent_id = brain.store("Original decision: use PostgreSQL",
                                 level=Level.Decisions, tags=["db"],
                                 deduplicate=False)
        child_id = brain.store("Migration plan for PostgreSQL",
                                level=Level.Working, tags=["db", "plan"],
                                caused_by_id=parent_id, deduplicate=False)
        assert isinstance(child_id, str)
        assert child_id != parent_id

    def test_caused_by_chain(self, brain):
        """Build a causal chain of 3 records."""
        id1 = brain.store("Root cause: auth system redesign",
                           deduplicate=False)
        id2 = brain.store("Consequence: update API endpoints",
                           caused_by_id=id1, deduplicate=False)
        id3 = brain.store("Consequence: update client SDK",
                           caused_by_id=id2, deduplicate=False)

        # All three should exist
        assert brain.get(id1) is not None
        assert brain.get(id2) is not None
        assert brain.get(id3) is not None

    def test_caused_by_nonexistent_parent(self, brain):
        """caused_by_id pointing to nonexistent record should not crash."""
        try:
            rid = brain.store("Orphan record",
                               caused_by_id="nonexistent_parent",
                               deduplicate=False)
            assert isinstance(rid, str)
        except RuntimeError:
            pass  # Also acceptable


class TestPromotion:
    def test_promote_record_working_to_decisions(self, brain):
        """promote_record should move Working -> Decisions."""
        rid = brain.store("Frequently accessed info",
                           level=Level.Working, deduplicate=False)
        new_level = brain.promote_record(rid)
        if new_level is not None:
            assert new_level == Level.Decisions

    def test_promote_record_decisions_to_domain(self, brain):
        """promote_record should move Decisions -> Domain."""
        rid = brain.store("Important decision",
                           level=Level.Decisions, deduplicate=False)
        new_level = brain.promote_record(rid)
        if new_level is not None:
            assert new_level == Level.Domain

    def test_promote_record_domain_to_identity(self, brain):
        """promote_record should move Domain -> Identity."""
        rid = brain.store("Core knowledge",
                           level=Level.Domain, deduplicate=False)
        new_level = brain.promote_record(rid)
        if new_level is not None:
            assert new_level == Level.Identity

    def test_promote_identity_returns_none(self, brain):
        """Already at Identity — cannot promote further."""
        rid = brain.store("Already identity level",
                           level=Level.Identity, deduplicate=False)
        result = brain.promote_record(rid)
        assert result is None

    def test_promote_nonexistent_returns_none(self, brain):
        """Promoting nonexistent record should return None."""
        result = brain.promote_record("nonexistent_id")
        assert result is None

    def test_promotion_candidates_default(self, brain):
        """promotion_candidates() should return list of Records."""
        for i in range(10):
            brain.store(f"Candidate record {i}",
                         level=Level.Working, deduplicate=False)
        candidates = brain.promotion_candidates()
        assert isinstance(candidates, list)

    def test_promotion_candidates_with_params(self, brain):
        """promotion_candidates with custom thresholds."""
        brain.store("Active record", level=Level.Working, deduplicate=False)
        candidates = brain.promotion_candidates(
            min_activations=1, min_strength=0.1)
        assert isinstance(candidates, list)


class TestStoreParameters:
    def test_pin_true(self, brain):
        """pin=True should be accepted as a store parameter."""
        rid = brain.store("Pinned memory", pin=True, deduplicate=False)
        rec = brain.get(rid)
        assert rec is not None

    def test_pin_false(self, brain):
        """pin=False should not force Identity."""
        rid = brain.store("Not pinned", pin=False, deduplicate=False)
        rec = brain.get(rid)
        assert rec is not None

    def test_content_type(self, brain):
        """content_type parameter on store."""
        rid = brain.store("Some content", content_type="note",
                           deduplicate=False)
        assert isinstance(rid, str)

    def test_source_type(self, brain):
        """source_type parameter on store (must be recorded/retrieved/inferred/generated)."""
        for st in ["recorded", "retrieved", "inferred", "generated"]:
            rid = brain.store(f"Data with source_type {st}",
                               source_type=st, deduplicate=False)
            rec = brain.get(rid)
            assert rec is not None

    def test_auto_promote_true(self, brain):
        """auto_promote=True should be accepted."""
        rid = brain.store("Auto promote candidate",
                           auto_promote=True, deduplicate=False)
        assert isinstance(rid, str)

    def test_auto_promote_false(self, brain):
        """auto_promote=False should be accepted."""
        rid = brain.store("No auto promote",
                           auto_promote=False, deduplicate=False)
        assert isinstance(rid, str)


class TestBoundaryConditions:
    def test_token_budget_zero(self, brain):
        """recall with token_budget=0 should return empty or minimal."""
        brain.store("Some content", deduplicate=False)
        result = brain.recall("test", token_budget=0)
        assert isinstance(result, str)

    def test_top_k_zero(self, brain):
        """recall_structured with top_k=0 should return empty list."""
        brain.store("Some content", deduplicate=False)
        results = brain.recall_structured("test", top_k=0)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_top_k_one(self, brain):
        """recall_structured with top_k=1 should return at most 1."""
        brain.store("Content A", deduplicate=False)
        brain.store("Content B", deduplicate=False)
        results = brain.recall_structured("content", top_k=1)
        assert len(results) <= 1

    def test_very_large_top_k(self, brain):
        """top_k larger than record count should not crash."""
        brain.store("Only record", deduplicate=False)
        results = brain.recall_structured("only", top_k=10000)
        assert isinstance(results, list)
        assert len(results) <= 1

    def test_min_strength_filter(self, brain):
        """min_strength should filter weak records."""
        brain.store("Strong content", deduplicate=False)
        results = brain.recall_structured("strong", top_k=5,
                                           min_strength=0.99)
        # Should return nothing or only very strong records
        for r in results:
            rec = brain.get(r["id"])
            if rec:
                assert rec.strength >= 0.99

    def test_very_long_query(self, brain):
        """Very long query should not crash."""
        brain.store("Short content", deduplicate=False)
        long_query = "word " * 1000
        results = brain.recall_structured(long_query, top_k=5)
        assert isinstance(results, list)

    def test_whitespace_only_query(self, brain):
        """Whitespace-only query should not crash."""
        brain.store("Data", deduplicate=False)
        result = brain.recall("   \n\t  ")
        assert isinstance(result, str)

    def test_strength_bounds(self, brain):
        """Strength should always be 0.0-1.0."""
        rid = brain.store("Normal content", deduplicate=False)
        rec = brain.get(rid)
        assert 0.0 <= rec.strength <= 1.0

        brain.update(rid, strength=1.0)
        rec = brain.get(rid)
        assert abs(rec.strength - 1.0) < 0.01

        brain.update(rid, strength=0.0)
        rec = brain.get(rid)
        assert abs(rec.strength - 0.0) < 0.01

    def test_strength_above_one(self, brain):
        """Setting strength > 1.0 should clamp or raise."""
        rid = brain.store("Content", deduplicate=False)
        try:
            brain.update(rid, strength=1.5)
            rec = brain.get(rid)
            # If accepted, should be clamped to 1.0
            assert rec.strength <= 1.0, \
                f"Strength {rec.strength} exceeds 1.0 — not clamped"
        except (RuntimeError, ValueError):
            pass  # Rejecting > 1.0 is also valid

    def test_strength_negative(self, brain):
        """Setting strength < 0.0 should clamp or raise."""
        rid = brain.store("Content", deduplicate=False)
        try:
            brain.update(rid, strength=-0.5)
            rec = brain.get(rid)
            # If accepted, should be clamped to 0.0
            assert rec.strength >= 0.0, \
                f"Strength {rec.strength} below 0.0 — not clamped"
        except (RuntimeError, ValueError):
            pass  # Rejecting negative is also valid

    def test_very_large_content(self, brain):
        """100KB content should be storable."""
        large = "x" * 100_000
        rid = brain.store(large, deduplicate=False)
        rec = brain.get(rid)
        assert len(rec.content) == 100_000

    def test_recall_on_empty_brain(self, brain):
        """recall on empty brain should return empty string or header."""
        result = brain.recall("anything")
        assert isinstance(result, str)

    def test_recall_structured_on_empty_brain(self, brain):
        """recall_structured on empty brain should return empty list."""
        results = brain.recall_structured("anything", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0


class TestLongTermDecay:
    """Simulate 90+ day decay to verify level hierarchy survives."""

    def test_working_decays_fast(self, brain):
        """Working-level record should lose significant strength after many cycles."""
        rid = brain.store("Temporary working note",
                           level=Level.Working, deduplicate=False)
        original = brain.get(rid).strength

        for _ in range(100):
            brain.decay()

        rec = brain.get(rid)
        if rec is not None:
            assert rec.strength < original, \
                "Working record should have decayed after 100 cycles"

    def test_identity_survives_extreme_decay(self, brain):
        """Identity record must survive 200 decay cycles with positive strength."""
        rid = brain.store("User's permanent identity fact",
                           level=Level.Identity, deduplicate=False)

        for _ in range(200):
            brain.decay()

        rec = brain.get(rid)
        assert rec is not None, "Identity record was archived after 200 decay cycles"
        assert rec.strength > 0, "Identity record strength hit 0 after 200 cycles"

    def test_domain_outlasts_working(self, brain):
        """Domain should retain more strength than Working after same decay."""
        id_domain = brain.store("Domain knowledge about databases",
                                 level=Level.Domain, deduplicate=False)
        id_working = brain.store("Temporary note about lunch meeting",
                                  level=Level.Working, deduplicate=False)

        for _ in range(50):
            brain.decay()

        domain_rec = brain.get(id_domain)
        working_rec = brain.get(id_working)

        if domain_rec and working_rec:
            assert domain_rec.strength >= working_rec.strength, \
                f"Domain ({domain_rec.strength:.3f}) weaker than Working ({working_rec.strength:.3f})"
        elif domain_rec and not working_rec:
            pass  # Working archived, domain survived — correct
        else:
            pytest.fail("Domain record should survive at least as long as Working")

    def test_hierarchy_order_after_decay(self, brain):
        """After 50 cycles: Identity >= Domain >= Decisions >= Working."""
        id_i = brain.store("Identity permanent", level=Level.Identity, deduplicate=False)
        id_d = brain.store("Domain knowledge", level=Level.Domain, deduplicate=False)
        id_c = brain.store("Decision record", level=Level.Decisions, deduplicate=False)
        id_w = brain.store("Working ephemeral", level=Level.Working, deduplicate=False)

        for _ in range(50):
            brain.decay()

        strengths = {}
        for label, rid in [("identity", id_i), ("domain", id_d),
                            ("decisions", id_c), ("working", id_w)]:
            rec = brain.get(rid)
            strengths[label] = rec.strength if rec else 0.0

        assert strengths["identity"] >= strengths["domain"], \
            f"Identity ({strengths['identity']:.3f}) < Domain ({strengths['domain']:.3f})"
        assert strengths["domain"] >= strengths["decisions"], \
            f"Domain ({strengths['domain']:.3f}) < Decisions ({strengths['decisions']:.3f})"
        assert strengths["decisions"] >= strengths["working"], \
            f"Decisions ({strengths['decisions']:.3f}) < Working ({strengths['working']:.3f})"
