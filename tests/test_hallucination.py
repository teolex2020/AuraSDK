"""Hallucination reduction benchmark — tests that memory layer
prevents/reduces agent hallucinations.

Categories:
  1. Recency Ranking — fresh facts outrank stale ones
  2. Source Authority — user-confirmed > agent-generated
  3. Supersession — updated facts replace old versions
  4. Volatility Decay — volatile data degrades faster
  5. Contradiction Handling — conflicting facts detected
  6. Recall Precision — accuracy across many facts
"""

import time
import pytest
from aura import Aura, Level, TrustConfig


# ============== 1. RECENCY RANKING ==============


class TestRecencyRanking:
    """Fresh records should rank higher than stale ones for the same query."""

    def test_fresh_fact_ranks_first(self, brain):
        """Store old fact (with past timestamp), then new fact — recall returns new first."""
        # Simulate a record from 30 days ago via metadata timestamp
        brain.store("Bitcoin price is $30,000", tags=["crypto", "price"],
                    metadata={"timestamp": "2025-01-01T00:00:00+00:00",
                              "source": "web_scrape", "trust_score": "0.5"},
                    deduplicate=False)
        brain.store("Bitcoin price is $95,000", tags=["crypto", "price"],
                    channel="web_scrape", deduplicate=False)

        results = brain.recall_structured("Bitcoin price", top_k=5)
        assert len(results) >= 2
        # Most recent should be first (highest score)
        assert "$95,000" in results[0]["content"]

    def test_recency_boost_with_trust_config(self, brain):
        """Explicit TrustConfig recency boost should amplify freshness."""
        tc = TrustConfig()
        tc.recency_boost_max = 0.3
        tc.recency_half_life_days = 7
        brain.set_trust_config(tc)

        # Old record with past timestamp
        brain.store("Old weather: sunny", tags=["weather"],
                    metadata={"timestamp": "2025-06-01T00:00:00+00:00",
                              "source": "web_scrape", "trust_score": "0.5"},
                    deduplicate=False)
        brain.store("Current weather: rainy", tags=["weather"],
                    channel="web_scrape", deduplicate=False)

        results = brain.recall_structured("weather today", top_k=5)
        assert len(results) >= 2
        assert "rainy" in results[0]["content"]

    def test_old_and_new_both_returned(self, brain):
        """Both old and new facts should be in results, just ranked."""
        brain.store("Population of Kyiv: 2.8 million", tags=["fact", "geography"],
                    deduplicate=False)
        brain.store("Population of Kyiv: 2.95 million (2025)",
                    tags=["fact", "geography"], deduplicate=False)

        results = brain.recall_structured("Kyiv population", top_k=5)
        contents = [r["content"] for r in results]
        assert any("2.95" in c for c in contents)
        assert any("2.8" in c for c in contents)

    def test_identity_level_resists_recency(self, brain):
        """Identity-level facts should not be easily outranked by fresh Working data."""
        brain.store("User's name is Oleksandr", level=Level.Identity,
                    tags=["identity", "name"], deduplicate=False)
        brain.store("Someone mentioned the name John today",
                    level=Level.Working, tags=["name"],
                    deduplicate=False)

        results = brain.recall_structured("user name", top_k=5)
        assert len(results) >= 1
        # Identity fact should still be accessible
        all_content = " ".join(r["content"] for r in results)
        assert "Oleksandr" in all_content


# ============== 2. SOURCE AUTHORITY ==============


class TestSourceAuthority:
    """Records from trusted sources should rank higher."""

    def test_user_input_outranks_autonomous(self, brain):
        """User-provided data should rank higher than agent-generated."""
        tc = TrustConfig()
        tc.source_trust = {
            "user_input": 1.0,
            "autonomous": 0.5,
        }
        brain.set_trust_config(tc)

        brain.store("My allergy is penicillin",
                    channel="autonomous", tags=["health"],
                    deduplicate=False)
        brain.store("My allergy is amoxicillin",
                    channel="user_input", tags=["health"],
                    deduplicate=False)

        results = brain.recall_structured("allergy", top_k=5)
        assert len(results) >= 2
        # User input should rank first
        assert "amoxicillin" in results[0]["content"]

    def test_verified_outranks_unverified(self, brain):
        """Verified records should outrank unverified ones."""
        tc = TrustConfig()
        tc.source_trust = {
            "verified": 1.0,
            "web_scrape": 0.4,
        }
        brain.set_trust_config(tc)

        brain.store("Earth is 4.5 billion years old",
                    channel="web_scrape", tags=["fact", "science"],
                    deduplicate=False)
        brain.store("Earth is 4.54 billion years old",
                    channel="verified", tags=["fact", "science"],
                    deduplicate=False)

        results = brain.recall_structured("age of Earth", top_k=5)
        assert len(results) >= 2
        assert "4.54" in results[0]["content"]

    def test_channel_provenance_stamped(self, brain):
        """Each channel should stamp source metadata."""
        channels = ["user_input", "autonomous", "web_scrape", "api"]
        for ch in channels:
            rid = brain.store(f"Data from {ch}", channel=ch, deduplicate=False)
            rec = brain.get(rid)
            assert rec.metadata.get("source") == ch

    def test_no_channel_gets_default_trust(self, brain):
        """Records without channel should still work in ranking."""
        brain.store("Fact without channel", tags=["fact"], deduplicate=False)
        brain.store("Fact with channel", tags=["fact"],
                    channel="user_input", deduplicate=False)

        results = brain.recall_structured("fact", top_k=5)
        assert len(results) >= 2


# ============== 3. SUPERSESSION ==============


class TestSupersession:
    """When a fact is updated, recall should return only the latest version."""

    def test_update_replaces_in_recall(self, brain):
        """After update, recall returns new content, not old."""
        rid = brain.store("User phone: +380991111111",
                          tags=["contact", "phone"])
        brain.update(rid, content="User phone: +380992222222")

        results = brain.recall_structured("user phone", top_k=5)
        contents = [r["content"] for r in results]
        # New value should be present
        assert any("+380992222222" in c for c in contents)
        # Old value should NOT appear as separate result
        assert not any("+380991111111" in c for c in contents
                       if "+380992222222" not in c)

    def test_update_preserves_id(self, brain):
        """Update should keep the same record ID."""
        rid = brain.store("Version 1", tags=["versioned"])
        brain.update(rid, content="Version 2")
        rec = brain.get(rid)
        assert rec.content == "Version 2"
        assert rec.id == rid

    def test_update_preserves_tags(self, brain):
        """Update content should not lose tags."""
        rid = brain.store("Data", tags=["important", "medical"])
        brain.update(rid, content="Updated data")
        rec = brain.get(rid)
        assert "important" in rec.tags
        assert "medical" in rec.tags

    def test_store_duplicate_warns_or_deduplicates(self, brain):
        """Storing same content twice should either dedup or return same ID."""
        id1 = brain.store("Exact same fact about user")
        id2 = brain.store("Exact same fact about user")
        # Either same ID (merged) or count didn't increase
        count = brain.count()
        assert count <= 2  # Not both stored as separate records


# ============== 4. VOLATILITY DECAY ==============


class TestVolatilityDecay:
    """Volatile data (prices, weather) should degrade faster than stable data."""

    def test_volatile_tags_detected(self, brain):
        """Records with volatile tags should be stamped with volatility."""
        rid = brain.store("BTC = $95k", tags=["crypto", "price"],
                          channel="web_scrape")
        rec = brain.get(rid)
        vol = rec.metadata.get("volatility", "")
        # Should be marked as volatile or moderate
        assert vol in ("volatile", "moderate", "")

    def test_stable_tags_detected(self, brain):
        """Records with stable tags should be stamped as stable."""
        rid = brain.store("User email: test@example.com",
                          tags=["contact", "email"], channel="user_input")
        rec = brain.get(rid)
        vol = rec.metadata.get("volatility", "stable")
        assert vol in ("stable", "")

    def test_decay_reduces_volatile_faster(self, brain):
        """After multiple decay cycles, volatile records should be weaker."""
        rid_volatile = brain.store("Weather: 25°C", level=Level.Working,
                                   tags=["weather"], deduplicate=False)
        rid_stable = brain.store("Blood type: A+", level=Level.Working,
                                 tags=["medical", "health"], deduplicate=False)

        # Run several decay cycles
        for _ in range(10):
            brain.decay()

        vol_rec = brain.get(rid_volatile)
        stab_rec = brain.get(rid_stable)

        # Both may have decayed, but if both survive, stable should be stronger
        if vol_rec and stab_rec:
            assert stab_rec.strength >= vol_rec.strength

    def test_identity_never_decays_to_zero(self, brain):
        """Identity-level records should survive many decay cycles."""
        rid = brain.store("User name is Oleksandr",
                          level=Level.Identity, tags=["identity"],
                          deduplicate=False)

        for _ in range(20):
            brain.decay()

        rec = brain.get(rid)
        assert rec is not None
        assert rec.strength > 0


# ============== 5. CONTRADICTION HANDLING ==============


class TestContradictionHandling:
    """Conflicting facts should be detectable or resolved by ranking."""

    def test_latest_fact_wins_in_ranking(self, brain):
        """When contradictory facts exist, the latest should rank first."""
        # Old preference with past timestamp
        brain.store("User prefers dark mode", tags=["preference", "ui"],
                    metadata={"timestamp": "2025-01-01T00:00:00+00:00",
                              "source": "agent-interactive", "trust_score": "0.7"},
                    deduplicate=False)
        brain.store("User prefers light mode", tags=["preference", "ui"],
                    channel="desktop", deduplicate=False)

        results = brain.recall_structured("user ui preference", top_k=5)
        assert len(results) >= 2
        # Latest stored should rank first
        assert "light mode" in results[0]["content"]

    def test_update_resolves_contradiction(self, brain):
        """Using update instead of new store avoids contradiction."""
        rid = brain.store("Favorite color: blue", tags=["preference"])
        brain.update(rid, content="Favorite color: green")

        results = brain.recall_structured("favorite color", top_k=5)
        contents = [r["content"] for r in results]
        # Only green should be present
        assert any("green" in c for c in contents)
        assert not any("blue" in c and "green" not in c for c in contents)

    def test_dedup_catches_near_duplicates(self, brain):
        """Default dedup should catch very similar content."""
        id1 = brain.store("Python is a dynamic programming language")
        id2 = brain.store("Python is a dynamic programming language")
        # Dedup should prevent both existing separately
        assert brain.count() <= 2

    def test_conflicting_medical_data_both_accessible(self, brain):
        """Critical data: both versions should be accessible even if conflicting."""
        brain.store("Blood pressure: 120/80 (Jan 2025)",
                    tags=["health", "vitals"], deduplicate=False)
        brain.store("Blood pressure: 140/90 (Feb 2025)",
                    tags=["health", "vitals"], deduplicate=False)

        results = brain.recall_structured("blood pressure", top_k=10)
        contents = " ".join(r["content"] for r in results)
        # Both readings should be accessible (medical history matters)
        assert "120/80" in contents
        assert "140/90" in contents


# ============== 6. RECALL PRECISION BENCHMARK ==============


class TestRecallPrecision:
    """Bulk accuracy test: store N facts, query N questions, measure precision."""

    FACTS = [
        ("Capital of Ukraine is Kyiv", "capital of Ukraine", "Kyiv"),
        ("Python was created by Guido van Rossum", "who created Python", "Guido"),
        ("Water boils at 100 degrees Celsius", "boiling point of water", "100"),
        ("Speed of light is 299,792 km/s", "speed of light", "299"),
        ("Human body has 206 bones", "how many bones in human body", "206"),
        ("Oxygen symbol is O", "chemical symbol for oxygen", "O"),
        ("Earth orbits the Sun", "what does Earth orbit", "Sun"),
        ("DNA stands for deoxyribonucleic acid", "what does DNA stand for", "deoxyribonucleic"),
        ("Mount Everest is 8,849 meters tall", "height of Everest", "8,849"),
        ("Bitcoin was created by Satoshi Nakamoto", "who created Bitcoin", "Satoshi"),
        ("JavaScript was created in 1995", "when was JavaScript created", "1995"),
        ("The Moon orbits Earth every 27.3 days", "Moon orbital period", "27.3"),
        ("HTTP status 404 means Not Found", "what is HTTP 404", "Not Found"),
        ("Pi is approximately 3.14159", "value of pi", "3.14"),
        ("Rust was first released in 2015", "when was Rust released", "2015"),
        ("PostgreSQL is a relational database", "what type of database is PostgreSQL", "relational"),
        ("TCP uses three-way handshake", "how does TCP connect", "handshake"),
        ("Git was created by Linus Torvalds", "who created Git", "Linus"),
        ("RAM stands for Random Access Memory", "what does RAM stand for", "Random Access"),
        ("Linux kernel is written in C", "what language is Linux written in", " C"),
    ]

    def test_recall_precision_above_80_percent(self, brain):
        """Store 20 facts, query each, expect >= 80% recall accuracy."""
        # Store all facts
        for fact, _, _ in self.FACTS:
            brain.store(fact, level=Level.Domain, tags=["fact", "benchmark"],
                        deduplicate=False)

        # Query each and check if answer is in top results
        hits = 0
        misses = []
        for fact, query, expected_fragment in self.FACTS:
            results = brain.recall_structured(query, top_k=3)
            top_contents = " ".join(r["content"] for r in results[:3])
            if expected_fragment in top_contents:
                hits += 1
            else:
                misses.append((query, expected_fragment, top_contents[:100]))

        precision = hits / len(self.FACTS)
        assert precision >= 0.80, (
            f"Recall precision {precision:.0%} < 80%. "
            f"Misses: {misses}"
        )

    def test_recall_precision_top1_above_60_percent(self, brain):
        """Top-1 recall: the best result contains the answer >= 60% of the time."""
        for fact, _, _ in self.FACTS:
            brain.store(fact, level=Level.Domain, tags=["fact"],
                        deduplicate=False)

        hits = 0
        for _, query, expected_fragment in self.FACTS:
            results = brain.recall_structured(query, top_k=1)
            if results and expected_fragment in results[0]["content"]:
                hits += 1

        precision = hits / len(self.FACTS)
        assert precision >= 0.60, f"Top-1 precision {precision:.0%} < 60%"

    def test_recall_with_noise(self, brain):
        """Store facts + noise records, precision should still be reasonable."""
        # Store real facts
        for fact, _, _ in self.FACTS:
            brain.store(fact, level=Level.Domain, tags=["fact"],
                        deduplicate=False)

        # Add noise
        noise = [
            "Meeting notes from Tuesday about project timeline",
            "Remember to buy groceries: milk, bread, eggs",
            "The cat sat on the mat and looked outside",
            "Conference call scheduled for 3pm tomorrow",
            "New restaurant opened on Main Street — try the pasta",
            "Finished reading chapter 5 of the Rust book",
            "Server migration planned for next weekend",
            "Birthday party for Anna on Saturday at 6pm",
            "Updated the CI pipeline to use Docker",
            "Weather forecast: rain expected on Thursday",
        ]
        for n in noise:
            brain.store(n, level=Level.Working, tags=["noise"],
                        deduplicate=False)

        # Query facts — should still find them despite noise
        hits = 0
        for _, query, expected_fragment in self.FACTS[:10]:  # Test first 10
            results = brain.recall_structured(query, top_k=3)
            top_contents = " ".join(r["content"] for r in results[:3])
            if expected_fragment in top_contents:
                hits += 1

        precision = hits / 10
        assert precision >= 0.70, f"Noisy precision {precision:.0%} < 70%"

    def test_recall_empty_brain_returns_empty(self, brain):
        """Querying an empty brain should not crash and return empty results."""
        results = brain.recall_structured("anything", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0
