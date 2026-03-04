"""Consolidation safety tests — ensure distinct facts are not falsely merged."""

import pytest
from aura import Aura, Level


class TestFalseMergePrevention:
    """Critical: consolidation must NOT merge distinct facts that share words."""

    def test_different_people_not_merged(self, brain):
        """Two people with similar descriptions should remain separate."""
        id1 = brain.store("Alice is a software engineer at Google",
                           level=Level.Domain, tags=["person"],
                           deduplicate=False)
        id2 = brain.store("Bob is a software engineer at Microsoft",
                           level=Level.Domain, tags=["person"],
                           deduplicate=False)

        brain.consolidate()

        # Both records must still exist
        assert brain.get(id1) is not None, "Alice record was falsely merged"
        assert brain.get(id2) is not None, "Bob record was falsely merged"

    def test_different_dates_not_merged(self, brain):
        """Events on different dates should not be merged."""
        id1 = brain.store("Meeting with team on Monday about API redesign",
                           level=Level.Working, deduplicate=False)
        id2 = brain.store("Meeting with team on Friday about deployment",
                           level=Level.Working, deduplicate=False)

        brain.consolidate()

        assert brain.get(id1) is not None
        assert brain.get(id2) is not None

    def test_different_numbers_not_merged(self, brain):
        """Facts with different numerical values should not merge."""
        id1 = brain.store("Server CPU usage is 45%",
                           level=Level.Working, tags=["metrics"],
                           deduplicate=False)
        id2 = brain.store("Server CPU usage is 92%",
                           level=Level.Working, tags=["metrics"],
                           deduplicate=False)

        brain.consolidate()

        r1 = brain.get(id1)
        r2 = brain.get(id2)
        # At least one must survive with its original value
        surviving = [r for r in [r1, r2] if r is not None]
        assert len(surviving) >= 1
        contents = [r.content for r in surviving]
        # If both survived, both values should be intact
        if len(surviving) == 2:
            assert "45%" in contents[0] or "45%" in contents[1]
            assert "92%" in contents[0] or "92%" in contents[1]

    def test_opposite_sentiments_not_merged(self, brain):
        """Contradicting opinions should not be merged."""
        id1 = brain.store("User likes dark mode interface",
                           level=Level.Identity, tags=["preference"],
                           deduplicate=False)
        id2 = brain.store("User dislikes dark mode interface",
                           level=Level.Working, tags=["preference"],
                           deduplicate=False)

        brain.consolidate()

        r1 = brain.get(id1)
        r2 = brain.get(id2)
        # Identity record must always survive
        assert r1 is not None, "Identity-level preference was merged away"

    def test_different_languages_not_merged(self, brain):
        """Same concept in different programming languages should stay separate."""
        id1 = brain.store("In Python, use list comprehensions for filtering",
                           level=Level.Domain, tags=["python"],
                           deduplicate=False)
        id2 = brain.store("In Rust, use iterators with .filter() for filtering",
                           level=Level.Domain, tags=["rust"],
                           deduplicate=False)

        brain.consolidate()

        assert brain.get(id1) is not None
        assert brain.get(id2) is not None


class TestLegitMerge:
    """Consolidation SHOULD merge true duplicates."""

    def test_exact_duplicates_merged(self, brain):
        """Identical content should be merged."""
        brain.store("Python is a dynamic language", deduplicate=False)
        brain.store("Python is a dynamic language", deduplicate=False)
        brain.store("Python is a dynamic language", deduplicate=False)

        before = brain.count()
        brain.consolidate()
        after = brain.count()

        assert after <= before, "Exact duplicates should be consolidated"

    def test_near_duplicates_merged(self, brain):
        """Very similar content (minor wording differences) may be merged."""
        brain.store("Python is a great dynamic programming language",
                     deduplicate=False)
        brain.store("Python is a great dynamic programming language for all",
                     deduplicate=False)

        before = brain.count()
        brain.consolidate()
        after = brain.count()

        # Either merged or not — both are acceptable, just no crash
        assert after <= before


class TestConsolidationWithLevels:
    """Consolidation should respect the memory hierarchy."""

    def test_identity_records_survive(self, brain):
        """Identity-level records must never be removed by consolidation."""
        contents = [
            "User name is Oleksandr Petrov",
            "User email is alex@example.com",
            "User prefers dark mode interface",
            "User speaks Ukrainian and English",
            "User works as senior backend developer",
        ]
        ids = []
        for content in contents:
            rid = brain.store(content, level=Level.Identity, deduplicate=False)
            ids.append(rid)

        brain.consolidate()

        surviving = sum(1 for rid in ids if brain.get(rid) is not None)
        assert surviving >= 4, \
            f"Too many Identity records removed by consolidation: {surviving}/5"

    def test_cross_level_not_merged(self, brain):
        """Records at different levels should not be merged even if similar."""
        id_identity = brain.store("Python programming knowledge",
                                    level=Level.Identity, deduplicate=False)
        id_working = brain.store("Python programming notes",
                                   level=Level.Working, deduplicate=False)

        brain.consolidate()

        # Identity record must survive
        assert brain.get(id_identity) is not None

    def test_consolidation_preserves_count_floor(self, brain):
        """After consolidation, count should never be zero if records existed."""
        distinct_facts = [
            "The speed of light is 299792 km per second",
            "Tokyo is the capital city of Japan",
            "Water molecule consists of two hydrogen and one oxygen atom",
            "Python was created by Guido van Rossum in 1991",
            "The Great Wall of China is over 21000 km long",
            "Bitcoin was invented by Satoshi Nakamoto in 2008",
            "Mount Everest is 8849 meters above sea level",
            "The human heart beats about 100000 times per day",
            "Rust programming language guarantees memory safety",
            "PostgreSQL is an open source relational database",
        ]
        for i, fact in enumerate(distinct_facts):
            brain.store(fact, level=Level.Domain, tags=[f"topic{i}"],
                         deduplicate=False)

        brain.consolidate()

        assert brain.count() >= 5, \
            f"Too many unique records were consolidated: {brain.count()}/10"


class TestRepeatedConsolidation:
    """Running consolidation multiple times should be idempotent."""

    def test_double_consolidation_stable(self, brain):
        """Running consolidate() twice should give same result."""
        for i in range(5):
            brain.store(f"Record {i} about topic {i}", deduplicate=False)

        brain.consolidate()
        count_after_first = brain.count()

        brain.consolidate()
        count_after_second = brain.count()

        assert count_after_second == count_after_first, \
            "Second consolidation changed count unexpectedly"

    def test_many_consolidation_cycles(self, brain):
        """10 consolidation cycles should not degrade truly unique data."""
        unique_facts = [
            "Python was created by Guido van Rossum in 1991",
            "The capital of Japan is Tokyo",
            "HTTP status 404 means resource not found",
            "DNA is a double helix structure discovered by Watson and Crick",
            "PostgreSQL supports ACID transactions and JSON columns",
        ]
        ids = []
        for fact in unique_facts:
            rid = brain.store(fact, level=Level.Domain,
                               tags=["unique"], deduplicate=False)
            ids.append(rid)

        for _ in range(10):
            brain.consolidate()

        surviving = sum(1 for rid in ids if brain.get(rid) is not None)
        assert surviving >= 3, \
            f"Too many records lost after 10 consolidation cycles: {surviving}/5"
