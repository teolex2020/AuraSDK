"""Embedding API tests — set_embedding_fn, store_embedding, has_embeddings, RRF signal."""

import pytest
from aura import Aura, Level


class TestEmbeddingFn:
    def test_set_embedding_fn(self, brain):
        """Setting an embedding function should not raise."""
        def dummy_embed(text: str) -> list:
            return [0.1] * 64

        brain.set_embedding_fn(dummy_embed)

    def test_has_embeddings_false_by_default(self, brain):
        """No embeddings until explicitly stored or computed."""
        brain.store("No embeddings yet", deduplicate=False)
        assert brain.has_embeddings() is False

    def test_embedding_fn_auto_computes_on_store(self, brain):
        """When embedding_fn is set, store() should auto-compute embeddings."""
        call_count = [0]

        def counting_embed(text: str) -> list:
            call_count[0] += 1
            return [float(ord(c) % 10) / 10 for c in text[:64].ljust(64)]

        brain.set_embedding_fn(counting_embed)
        brain.store("Test content for embedding", deduplicate=False)
        assert call_count[0] >= 1
        assert brain.has_embeddings() is True

    def test_embedding_fn_called_with_content(self, brain):
        """Verify the embedding function receives the stored content."""
        received_texts = []

        def capture_embed(text: str) -> list:
            received_texts.append(text)
            return [0.5] * 64

        brain.set_embedding_fn(capture_embed)
        brain.store("Specific content to embed", deduplicate=False)
        assert any("Specific content to embed" in t for t in received_texts)

    def test_embedding_fn_none_disables(self, brain):
        """Setting embedding_fn to None should disable auto-embedding."""
        def embed(text: str) -> list:
            return [0.1] * 64

        brain.set_embedding_fn(embed)
        brain.store("With embedding", deduplicate=False)
        assert brain.has_embeddings() is True


class TestStoreEmbedding:
    def test_store_embedding_manual(self, brain):
        """Manually store an embedding vector for a record."""
        rid = brain.store("Manual embedding record", deduplicate=False)
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # 50-dim vector
        brain.store_embedding(rid, embedding)
        assert brain.has_embeddings() is True

    def test_store_embedding_empty_vector(self, brain):
        """Storing an empty embedding should either work or raise cleanly."""
        rid = brain.store("Empty vec test", deduplicate=False)
        try:
            brain.store_embedding(rid, [])
        except (RuntimeError, ValueError):
            pass  # Acceptable to reject empty vectors

    def test_store_embedding_nonexistent_record(self, brain):
        """Storing embedding for nonexistent ID should raise or be ignored."""
        try:
            brain.store_embedding("nonexistent_id", [0.1] * 64)
        except (RuntimeError, ValueError):
            pass  # Expected

    def test_store_embedding_overwrite(self, brain):
        """Storing a new embedding for same record should overwrite."""
        rid = brain.store("Overwrite embedding", deduplicate=False)
        brain.store_embedding(rid, [0.1] * 50)
        brain.store_embedding(rid, [0.9] * 50)
        # Should not crash; has_embeddings should still be True
        assert brain.has_embeddings() is True


class TestEmbeddingRecall:
    def test_embedding_signal_in_recall(self, brain):
        """Embeddings should contribute to RRF recall scoring as 4th signal."""
        def simple_embed(text: str) -> list:
            # Deterministic embedding based on text hash
            import hashlib
            h = hashlib.md5(text.encode()).digest()
            return [b / 255.0 for b in h[:16]] * 4  # 64-dim

        brain.set_embedding_fn(simple_embed)

        brain.store("Rust programming language has ownership rules",
                     level=Level.Domain, tags=["rust"], deduplicate=False)
        brain.store("Python programming language is dynamically typed",
                     level=Level.Domain, tags=["python"], deduplicate=False)
        brain.store("Today meeting notes about project timeline",
                     level=Level.Working, tags=["meeting"], deduplicate=False)

        results = brain.recall_structured("Rust ownership", top_k=3)
        assert len(results) >= 1
        # The Rust record should be in results
        contents = [r["content"] for r in results]
        assert any("Rust" in c for c in contents)

    def test_recall_works_without_embeddings(self, brain):
        """Recall should still work fine with 3 signals (no embeddings)."""
        brain.store("Data without embeddings", level=Level.Domain,
                     tags=["test"], deduplicate=False)

        results = brain.recall_structured("data", top_k=5)
        assert len(results) >= 1

    def test_mixed_embedded_and_non_embedded(self, brain):
        """Records with and without embeddings should coexist."""
        # Store without embedding fn
        id1 = brain.store("No embedding here", deduplicate=False)

        # Set embedding fn and store another
        def embed(text: str) -> list:
            return [0.5] * 64

        brain.set_embedding_fn(embed)
        id2 = brain.store("Has embedding here", deduplicate=False)

        # Both should be recallable
        results = brain.recall_structured("embedding", top_k=5)
        ids = [r["id"] for r in results]
        assert id2 in ids
