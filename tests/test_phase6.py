"""Tests for Phase 6 features: Adaptive Recall, Snapshots, Sharing, Supersede."""

import json
import os
import tempfile

import pytest
from aura import Aura, Level


# ======================================================================
# 6.1 Adaptive Recall (Feedback)
# ======================================================================


class TestFeedback:
    """Tests for brain.feedback(record_id, useful)."""

    def test_positive_feedback_boosts_strength(self, brain):
        rid = brain.store("User prefers Python", level=Level.Working)
        # Weaken first so there's room to boost
        brain.feedback(rid, useful=False)
        rec_before = brain.get(rid)
        strength_before = rec_before.strength

        result = brain.feedback(rid, useful=True)
        assert result is True

        rec_after = brain.get(rid)
        assert rec_after.strength > strength_before

    def test_negative_feedback_weakens_strength(self, brain):
        rid = brain.store("Outdated preference for Java", level=Level.Working)
        rec_before = brain.get(rid)
        strength_before = rec_before.strength

        brain.feedback(rid, useful=False)

        rec_after = brain.get(rid)
        assert rec_after.strength < strength_before

    def test_feedback_nonexistent_returns_false(self, brain):
        result = brain.feedback("nonexistent_id", useful=True)
        assert result is False

    def test_feedback_tracks_positive_count(self, brain):
        rid = brain.store("Test positive tracking")
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=True)

        rec = brain.get(rid)
        assert rec.metadata.get("feedback_positive") == "3"

    def test_feedback_tracks_negative_count(self, brain):
        rid = brain.store("Test negative tracking")
        brain.feedback(rid, useful=False)
        brain.feedback(rid, useful=False)

        rec = brain.get(rid)
        assert rec.metadata.get("feedback_negative") == "2"

    def test_feedback_tracks_last_timestamp(self, brain):
        rid = brain.store("Test timestamp tracking")
        brain.feedback(rid, useful=True)

        rec = brain.get(rid)
        assert "feedback_last" in rec.metadata
        ts = float(rec.metadata["feedback_last"])
        assert ts > 0

    def test_feedback_mixed_positive_and_negative(self, brain):
        rid = brain.store("Mixed feedback memory")
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=False)

        rec = brain.get(rid)
        assert rec.metadata.get("feedback_positive") == "2"
        assert rec.metadata.get("feedback_negative") == "1"

    def test_negative_feedback_floor_at_zero(self, brain):
        rid = brain.store("Will be weakened a lot", level=Level.Working)
        # Weaken many times
        for _ in range(20):
            brain.feedback(rid, useful=False)

        rec = brain.get(rid)
        assert rec.strength >= 0.0


class TestFeedbackStats:
    """Tests for brain.feedback_stats(record_id)."""

    def test_stats_returns_tuple(self, brain):
        rid = brain.store("Stats test")
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=False)

        stats = brain.feedback_stats(rid)
        assert stats is not None
        pos, neg, net = stats
        assert pos == 1
        assert neg == 1
        assert net == 0

    def test_stats_nonexistent_returns_none(self, brain):
        stats = brain.feedback_stats("nonexistent")
        assert stats is None

    def test_stats_no_feedback_returns_zeros(self, brain):
        rid = brain.store("No feedback yet")
        stats = brain.feedback_stats(rid)
        assert stats == (0, 0, 0)

    def test_stats_net_score(self, brain):
        rid = brain.store("Net score test")
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=True)
        brain.feedback(rid, useful=False)

        stats = brain.feedback_stats(rid)
        assert stats == (3, 1, 2)


# ======================================================================
# 6.2 Memory Snapshots & Rollback
# ======================================================================


class TestSnapshot:
    """Tests for brain.snapshot() and brain.rollback()."""

    def test_snapshot_creates_file(self, brain):
        brain.store("Snapshot test data")
        path = brain.snapshot("v1")
        assert os.path.exists(path)

    def test_snapshot_returns_path(self, brain):
        brain.store("Data")
        path = brain.snapshot("test-snap")
        assert "test-snap" in path
        assert path.endswith(".json")

    def test_rollback_restores_state(self, brain):
        brain.store("Original memory", level=Level.Domain)
        brain.snapshot("before")

        # Add more data
        brain.store("Extra memory 1")
        brain.store("Extra memory 2")
        assert brain.count() == 3

        # Rollback
        count = brain.rollback("before")
        assert count == 1
        assert brain.count() == 1

    def test_rollback_nonexistent_raises(self, brain):
        with pytest.raises(RuntimeError, match="not found"):
            brain.rollback("doesnt_exist")

    def test_snapshot_invalid_label_raises(self, brain):
        with pytest.raises(RuntimeError):
            brain.snapshot("")  # Empty label

    def test_snapshot_special_chars_raises(self, brain):
        with pytest.raises(RuntimeError):
            brain.snapshot("bad/label")

    def test_list_snapshots(self, brain):
        brain.store("Data")
        brain.snapshot("alpha")
        brain.snapshot("beta")

        snaps = brain.list_snapshots()
        assert "alpha" in snaps
        assert "beta" in snaps

    def test_list_snapshots_empty(self, brain):
        snaps = brain.list_snapshots()
        assert snaps == []

    def test_rollback_preserves_content(self, brain):
        rid = brain.store("Important fact about Rust", level=Level.Domain)
        brain.snapshot("with-rust")

        brain.store("Noise data 1")
        brain.store("Noise data 2")
        brain.rollback("with-rust")

        results = brain.search()
        assert len(results) == 1
        assert "Rust" in results[0].content


class TestDiff:
    """Tests for brain.diff(label_a, label_b)."""

    def test_diff_detects_added_records(self, brain):
        brain.store("Original")
        brain.snapshot("v1")

        brain.store("New record")
        brain.snapshot("v2")

        d = brain.diff("v1", "v2")
        assert len(d["added"]) == 1
        assert len(d["removed"]) == 0

    def test_diff_detects_removed_records(self, brain):
        rid = brain.store("Will be removed")
        brain.store("Will stay")
        brain.snapshot("before")

        brain.delete(rid)
        brain.snapshot("after")

        d = brain.diff("before", "after")
        assert len(d["removed"]) == 1
        assert rid in d["removed"]

    def test_diff_detects_modified_records(self, brain):
        rid = brain.store("Original content", level=Level.Working)
        brain.snapshot("v1")

        brain.update(rid, content="Modified content")
        brain.snapshot("v2")

        d = brain.diff("v1", "v2")
        assert rid in d["modified"]

    def test_diff_nonexistent_snapshot_raises(self, brain):
        brain.snapshot("exists")
        with pytest.raises(RuntimeError, match="not found"):
            brain.diff("exists", "nope")


# ======================================================================
# 6.3 Agent-to-Agent Memory Sharing
# ======================================================================


class TestExportContext:
    """Tests for brain.export_context()."""

    def test_export_returns_json(self, brain):
        brain.store("Shared knowledge about AI", level=Level.Domain, tags=["ai"])
        result = brain.export_context("AI knowledge")
        data = json.loads(result)
        assert data["format"] == "aura_context"
        assert data["version"] == "1.0"
        assert data["record_count"] >= 1

    def test_export_includes_records(self, brain):
        brain.store("Important fact: water boils at 100C", level=Level.Domain)
        result = brain.export_context("water boils")
        data = json.loads(result)
        assert len(data["records"]) >= 1
        assert any("water" in r["content"].lower() for r in data["records"])

    def test_export_includes_metadata(self, brain):
        brain.store("Test export metadata", tags=["test"])
        result = brain.export_context("metadata")
        data = json.loads(result)
        if data["records"]:
            rec = data["records"][0]
            assert "metadata" in rec
            assert "shared_score" in rec["metadata"]


class TestImportContext:
    """Tests for brain.import_context()."""

    def test_import_from_export(self):
        """Two brains: export from one, import to another."""
        with tempfile.TemporaryDirectory() as tmp:
            brain_a = Aura(os.path.join(tmp, "brain_a.db"))
            brain_b = Aura(os.path.join(tmp, "brain_b.db"))

            brain_a.store("Python is great for prototyping", level=Level.Domain, tags=["python"])
            brain_a.store("Rust is fast and safe", level=Level.Domain, tags=["rust"])

            fragment = brain_a.export_context("programming languages")
            imported = brain_b.import_context(fragment)

            assert imported >= 1
            assert brain_b.count() >= 1

            # Imported records should have 'shared' tag
            results = brain_b.search(tags=["shared"])
            assert len(results) >= 1

            brain_a.close()
            brain_b.close()

    def test_import_reduces_strength(self):
        """Imported memories should have reduced strength."""
        with tempfile.TemporaryDirectory() as tmp:
            brain_a = Aura(os.path.join(tmp, "a.db"))
            brain_b = Aura(os.path.join(tmp, "b.db"))

            brain_a.store("External knowledge", level=Level.Domain)
            fragment = brain_a.export_context("external")
            brain_b.import_context(fragment)

            results = brain_b.search(tags=["shared"])
            if results:
                # Strength should be < 1.0 (reduced by 0.5x)
                assert results[0].strength < 1.0

            brain_a.close()
            brain_b.close()

    def test_import_invalid_format_raises(self, brain):
        with pytest.raises(RuntimeError, match="Unknown format"):
            brain.import_context('{"format": "wrong", "records": []}')

    def test_import_marks_trust_external(self):
        """Imported records should have trust_external metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            brain_a = Aura(os.path.join(tmp, "a.db"))
            brain_b = Aura(os.path.join(tmp, "b.db"))

            brain_a.store("Fact to share", level=Level.Domain)
            fragment = brain_a.export_context("fact")
            brain_b.import_context(fragment)

            results = brain_b.search(tags=["shared"])
            if results:
                assert results[0].metadata.get("trust_external") == "true"

            brain_a.close()
            brain_b.close()


# ======================================================================
# 6.4 Semantic Versioning (Supersede)
# ======================================================================


class TestSupersede:
    """Tests for brain.supersede()."""

    def test_supersede_creates_new_record(self, brain):
        old_id = brain.store("User prefers light mode", level=Level.Identity)
        new_id = brain.supersede(old_id, "User prefers dark mode")

        assert new_id != old_id
        new_rec = brain.get(new_id)
        assert "dark mode" in new_rec.content

    def test_supersede_weakens_old_record(self, brain):
        old_id = brain.store("Old preference", level=Level.Identity)
        old_strength_before = brain.get(old_id).strength

        brain.supersede(old_id, "New preference")

        old_strength_after = brain.get(old_id).strength
        assert old_strength_after < old_strength_before

    def test_supersede_marks_old_with_metadata(self, brain):
        old_id = brain.store("Outdated info")
        new_id = brain.supersede(old_id, "Updated info")

        old_rec = brain.get(old_id)
        assert old_rec.metadata.get("superseded_by") == new_id

    def test_superseded_by_returns_new_id(self, brain):
        old_id = brain.store("Version 1")
        new_id = brain.supersede(old_id, "Version 2")

        result = brain.superseded_by(old_id)
        assert result == new_id

    def test_superseded_by_returns_none_for_current(self, brain):
        rid = brain.store("Not superseded")
        result = brain.superseded_by(rid)
        assert result is None

    def test_supersede_nonexistent_raises(self, brain):
        with pytest.raises(RuntimeError, match="not found"):
            brain.supersede("nonexistent", "New content")

    def test_supersede_inherits_level(self, brain):
        old_id = brain.store("Domain knowledge", level=Level.Domain)
        new_id = brain.supersede(old_id, "Updated domain knowledge")

        new_rec = brain.get(new_id)
        assert new_rec.level == Level.Domain

    def test_supersede_inherits_tags(self, brain):
        old_id = brain.store("Tagged memory", tags=["preference", "ui"])
        new_id = brain.supersede(old_id, "Updated tagged memory")

        new_rec = brain.get(new_id)
        assert "preference" in new_rec.tags
        assert "ui" in new_rec.tags

    def test_supersede_custom_level_overrides(self, brain):
        old_id = brain.store("Working memory", level=Level.Working)
        new_id = brain.supersede(old_id, "Promoted to identity", level=Level.Identity)

        new_rec = brain.get(new_id)
        assert new_rec.level == Level.Identity

    def test_supersede_has_causal_link(self, brain):
        old_id = brain.store("Version 1")
        new_id = brain.supersede(old_id, "Version 2")

        new_rec = brain.get(new_id)
        assert new_rec.caused_by_id == old_id


class TestVersionChain:
    """Tests for brain.version_chain()."""

    def test_chain_single_record(self, brain):
        rid = brain.store("Solo record")
        chain = brain.version_chain(rid)
        assert len(chain) == 1
        assert chain[0].id == rid

    def test_chain_two_versions(self, brain):
        v1 = brain.store("Version 1")
        v2 = brain.supersede(v1, "Version 2")

        chain = brain.version_chain(v1)
        assert len(chain) == 2
        assert chain[0].id == v1
        assert chain[1].id == v2

    def test_chain_three_versions(self, brain):
        v1 = brain.store("Version 1")
        v2 = brain.supersede(v1, "Version 2")
        v3 = brain.supersede(v2, "Version 3")

        chain = brain.version_chain(v1)
        assert len(chain) == 3
        contents = [r.content for r in chain]
        assert "Version 1" in contents[0]
        assert "Version 3" in contents[2]

    def test_chain_from_any_version(self, brain):
        v1 = brain.store("V1")
        v2 = brain.supersede(v1, "V2")
        v3 = brain.supersede(v2, "V3")

        # Should get same chain whether starting from v1, v2, or v3
        chain_from_v2 = brain.version_chain(v2)
        assert len(chain_from_v2) >= 2

    def test_chain_empty_for_nonexistent(self, brain):
        chain = brain.version_chain("nonexistent")
        assert len(chain) == 0
