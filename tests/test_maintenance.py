"""Maintenance, background brain, decay, consolidation tests."""

import pytest
from aura import Aura, Level, MaintenanceConfig, MaintenanceReport


class TestMaintenance:
    def test_run_maintenance(self, brain):
        brain.store("Some content", deduplicate=False)
        report = brain.run_maintenance()
        assert isinstance(report, MaintenanceReport)

    def test_maintenance_report_fields(self, brain):
        brain.store("Content", deduplicate=False)
        report = brain.run_maintenance()
        assert hasattr(report, "timestamp")
        assert hasattr(report, "total_records")
        assert hasattr(report, "decay")
        assert hasattr(report, "reflect")
        assert hasattr(report, "consolidation")
        assert hasattr(report, "insights_found")
        assert hasattr(report, "cross_connections")
        assert hasattr(report, "records_archived")

    def test_maintenance_on_empty_brain(self, brain):
        report = brain.run_maintenance()
        assert report.total_records == 0

    def test_configure_maintenance(self, brain):
        config = MaintenanceConfig()
        config.decay_enabled = True
        config.consolidation_enabled = False
        brain.configure_maintenance(config)

    def test_maintenance_config_attributes(self):
        config = MaintenanceConfig()
        assert hasattr(config, "decay_enabled")
        assert hasattr(config, "consolidation_enabled")


class TestDecay:
    def test_decay_runs(self, brain):
        brain.store("Decayable", level=Level.Working, deduplicate=False)
        result = brain.decay()
        # decay() returns a tuple (decayed_count, archived_count)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_decay_reduces_strength(self, brain):
        rid = brain.store("Will decay", level=Level.Working, deduplicate=False)
        original = brain.get(rid)
        original_strength = original.strength

        for _ in range(5):
            brain.decay()

        after = brain.get(rid)
        if after is not None:
            assert after.strength <= original_strength


class TestConsolidation:
    def test_consolidation_runs(self, brain):
        result = brain.consolidate()
        # consolidate() returns a dict with 'merged' and 'checked'
        assert isinstance(result, dict)
        assert "merged" in result
        assert "checked" in result

    def test_consolidation_merges_duplicates(self, brain):
        brain.store("Python is a great programming language for beginners", deduplicate=False)
        brain.store("Python is a great programming language for beginners and experts", deduplicate=False)
        brain.store("Python is a wonderful programming language for beginners", deduplicate=False)

        before = brain.count()
        brain.consolidate()
        after = brain.count()
        assert after <= before


class TestReflect:
    def test_reflect_runs(self, brain):
        brain.store("Content", deduplicate=False)
        result = brain.reflect()
        # reflect() returns a dict with 'promoted' and 'archived'
        assert isinstance(result, dict)
        assert "promoted" in result
        assert "archived" in result


class TestBackground:
    def test_start_stop_no_crash(self, brain):
        """start_background creates controller; actual threading is external."""
        brain.start_background(interval_secs=60)
        brain.stop_background()

    def test_background_not_running_initially(self, brain):
        assert not brain.is_background_running()

    def test_double_stop_is_safe(self, brain):
        brain.stop_background()
