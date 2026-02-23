"""Guards (auto-protect), encryption, and edge case tests."""

import os
import tempfile
import pytest
from aura import Aura, Level


class TestAutoProtect:
    def test_phone_number_tagged(self, brain):
        rid = brain.store("Call me at +380991234567")
        rec = brain.get(rid)
        assert "contact" in rec.tags

    def test_email_tagged(self, brain):
        rid = brain.store("Email: admin@example.com for support")
        rec = brain.get(rid)
        assert "contact" in rec.tags

    def test_wallet_tagged(self, brain):
        rid = brain.store("Send to 0x1234567890abcdef1234567890abcdef12345678")
        rec = brain.get(rid)
        assert "financial" in rec.tags

    def test_api_key_tagged(self, brain):
        rid = brain.store("api_key: sk-proj-abc123def456ghi789jkl012mno")
        rec = brain.get(rid)
        assert "credential" in rec.tags

    def test_normal_content_not_tagged(self, brain):
        rid = brain.store("Just a normal sentence with no sensitive data")
        rec = brain.get(rid)
        assert "contact" not in rec.tags
        assert "financial" not in rec.tags
        assert "credential" not in rec.tags


class TestEncryption:
    def test_create_encrypted_brain(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = Aura(os.path.join(tmp, "enc.db"), password="test-password")
            assert brain.is_encrypted()
            brain.store("Secret data")
            brain.close()

    def test_encrypted_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "enc.db")
            brain = Aura(path, password="my-pass")
            rid = brain.store("Secret content", deduplicate=False)
            brain.close()

            brain2 = Aura(path, password="my-pass")
            rec = brain2.get(rid)
            assert rec is not None
            assert rec.content == "Secret content"
            brain2.close()

    def test_unencrypted_brain(self, brain):
        assert not brain.is_encrypted()


class TestEdgeCases:
    def test_empty_content_raises(self, brain):
        with pytest.raises(RuntimeError):
            brain.store("")

    def test_unicode_content(self, brain):
        rid = brain.store("Привіт світ! 你好世界 🌍")
        rec = brain.get(rid)
        assert "Привіт" in rec.content

    def test_large_content(self, brain):
        large = "x" * 50000
        rid = brain.store(large)
        rec = brain.get(rid)
        assert len(rec.content) == 50000

    def test_many_tags(self, brain):
        tags = [f"tag{i}" for i in range(40)]
        rid = brain.store("Many tags", tags=tags)
        rec = brain.get(rid)
        assert len(rec.tags) >= 40

    def test_special_chars_in_tags(self, brain):
        rid = brain.store("Content", tags=["tag-with-dash", "tag_underscore", "tag.dot"])
        rec = brain.get(rid)
        assert "tag-with-dash" in rec.tags

    def test_concurrent_operations(self, brain):
        """Basic test that operations don't deadlock."""
        for i in range(50):
            brain.store(f"Concurrent record {i}", deduplicate=False)
        assert brain.count() >= 50

        results = brain.recall_structured("concurrent", top_k=10)
        assert len(results) > 0

    def test_multiple_brains_different_paths(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            b1 = Aura(os.path.join(tmp1, "a.db"))
            b2 = Aura(os.path.join(tmp2, "b.db"))

            b1.store("Brain 1 data")
            b2.store("Brain 2 data")

            r1 = b1.recall_structured("data", top_k=5)
            r2 = b2.recall_structured("data", top_k=5)

            assert len(r1) > 0
            assert len(r2) > 0
            # Content should be different
            assert r1[0]["content"] != r2[0]["content"]

            b1.close()
            b2.close()
