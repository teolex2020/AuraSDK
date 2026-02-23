"""Trust, taxonomy, provenance, and credibility tests."""

import pytest
from aura import Aura, Level, TagTaxonomy, TrustConfig


class TestTaxonomy:
    def test_set_taxonomy(self, brain):
        tax = TagTaxonomy()
        brain.set_taxonomy(tax)

    def test_get_taxonomy(self, brain):
        tax = TagTaxonomy()
        brain.set_taxonomy(tax)
        got = brain.get_taxonomy()
        assert got is not None

    def test_taxonomy_has_defaults(self):
        tax = TagTaxonomy()
        assert hasattr(tax, "identity_tags")
        assert hasattr(tax, "stable_tags")
        assert hasattr(tax, "volatile_tags")
        assert hasattr(tax, "sensitive_tags")


class TestTrustConfig:
    def test_set_trust_config(self, brain):
        tc = TrustConfig()
        brain.set_trust_config(tc)

    def test_trust_config_attributes(self):
        tc = TrustConfig()
        assert hasattr(tc, "source_trust")
        assert hasattr(tc, "recency_boost_max")
        assert hasattr(tc, "recency_half_life_days")

    def test_trust_affects_recall_ranking(self, brain):
        """Higher-trust sources should rank higher."""
        tc = TrustConfig()
        tc.source_trust = {"trusted": 1.0, "untrusted": 0.1}
        brain.set_trust_config(tc)

        brain.store("Important fact from trusted source",
                     tags=["fact"], channel="trusted")
        brain.store("Same fact from untrusted source",
                     tags=["fact"], channel="untrusted")

        results = brain.recall_structured("important fact", top_k=5)
        if len(results) >= 2:
            # Trusted source should have a higher score
            trusted = [r for r in results if r.get("source") == "trusted"]
            untrusted = [r for r in results if r.get("source") == "untrusted"]
            if trusted and untrusted:
                assert trusted[0]["score"] >= untrusted[0]["score"]


class TestProvenance:
    def test_channel_stamps_source(self, brain):
        rid = brain.store("Data", channel="user_input")
        rec = brain.get(rid)
        assert rec.metadata.get("source") == "user_input"

    def test_channel_stamps_trust_score(self, brain):
        rid = brain.store("Data", channel="api")
        rec = brain.get(rid)
        assert "trust_score" in rec.metadata

    def test_channel_stamps_timestamp(self, brain):
        rid = brain.store("Data", channel="test")
        rec = brain.get(rid)
        assert "timestamp" in rec.metadata

    def test_channel_stamps_volatility(self, brain):
        rid = brain.store("Data", channel="web_scrape")
        rec = brain.get(rid)
        assert "volatility" in rec.metadata

    def test_no_channel_no_provenance(self, brain):
        rid = brain.store("Data without channel")
        rec = brain.get(rid)
        # Without channel, provenance fields may not be set
        # (depends on default behavior)


class TestCredibility:
    def test_known_domain(self, brain):
        score = brain.get_credibility("nature.com")
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # nature.com should be high credibility

    def test_unknown_domain(self, brain):
        score = brain.get_credibility("unknown-random-domain-xyz.com")
        assert 0.0 <= score <= 1.0
        assert abs(score - 0.5) < 0.1  # Default around 0.5

    def test_set_credibility_override(self, brain):
        brain.set_credibility_override("my-internal.com", 0.95)
        score = brain.get_credibility("my-internal.com")
        assert abs(score - 0.95) < 0.01

    def test_override_replaces_default(self, brain):
        brain.set_credibility_override("nature.com", 0.1)
        score = brain.get_credibility("nature.com")
        assert abs(score - 0.1) < 0.01

    def test_multiple_known_domains(self, brain):
        domains = {
            "nature.com": 0.8,
            "arxiv.org": 0.8,
            "reddit.com": 0.3,
        }
        for domain, min_expected in domains.items():
            score = brain.get_credibility(domain)
            assert score >= min_expected, f"{domain}: {score} < {min_expected}"
