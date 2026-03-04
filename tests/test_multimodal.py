"""Tests for multimodal memory stubs (store_image, store_audio_transcript)."""

from aura import Level  # noqa: F401 (brain fixture from conftest.py)


class TestStoreImage:
    """Tests for store_image()."""

    def test_store_image_returns_id(self, brain):
        rid = brain.store_image("/photos/cat.jpg", "A fluffy orange cat on a sofa")
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_store_image_content_searchable(self, brain):
        brain.store_image("/photos/sunset.png", "Beautiful sunset over the ocean")
        results = brain.search()
        contents = [r.content for r in results]
        assert any("sunset" in c for c in contents)

    def test_store_image_has_image_tag(self, brain):
        brain.store_image("/photos/dog.jpg", "A golden retriever playing fetch")
        results = brain.search(tags=["image"])
        assert len(results) >= 1
        assert "image" in results[0].tags

    def test_store_image_metadata_has_source_path(self, brain):
        brain.store_image("/photos/logo.svg", "Company logo in blue")
        results = brain.search(tags=["image"])
        rec = results[0]
        assert rec.metadata.get("source_path") == "/photos/logo.svg"
        assert rec.metadata.get("media_type") == "image"

    def test_store_image_custom_tags(self, brain):
        brain.store_image(
            "/photos/chart.png",
            "Q4 revenue chart",
            tags=["finance", "chart"],
        )
        results = brain.search(tags=["finance"])
        assert len(results) >= 1
        tags = results[0].tags
        assert "finance" in tags
        assert "image" in tags  # auto-added

    def test_store_image_with_level(self, brain):
        brain.store_image(
            "/photos/id.jpg",
            "User profile photo",
            level=Level.Identity,
        )
        results = brain.search(level=Level.Identity)
        assert len(results) >= 1

    def test_store_image_with_namespace(self, brain):
        brain.store_image(
            "/photos/team.jpg",
            "Team photo from offsite",
            namespace="work",
        )
        results = brain.search(namespace="work")
        assert len(results) >= 1

    def test_store_image_source_type_is_multimodal(self, brain):
        brain.store_image("/photos/x.jpg", "Test image")
        results = brain.search(tags=["image"])
        assert results[0].source_type == "recorded"

    def test_store_image_content_type_is_image(self, brain):
        brain.store_image("/img/y.png", "Another test image")
        results = brain.search(tags=["image"])
        assert results[0].content_type == "image"


class TestStoreAudioTranscript:
    """Tests for store_audio_transcript()."""

    def test_store_audio_returns_id(self, brain):
        rid = brain.store_audio_transcript(
            "Hello, welcome to the meeting.",
            "/recordings/meeting_01.wav",
        )
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_store_audio_content_searchable(self, brain):
        brain.store_audio_transcript(
            "The quarterly revenue increased by 15 percent.",
            "/recordings/earnings.mp3",
        )
        results = brain.search()
        contents = [r.content for r in results]
        assert any("revenue" in c for c in contents)

    def test_store_audio_has_audio_tag(self, brain):
        brain.store_audio_transcript(
            "Testing one two three",
            "/recordings/test.wav",
        )
        results = brain.search(tags=["audio"])
        assert len(results) >= 1
        assert "audio" in results[0].tags

    def test_store_audio_metadata_has_source_path(self, brain):
        brain.store_audio_transcript(
            "User said they prefer dark mode.",
            "/recordings/call_42.ogg",
        )
        results = brain.search(tags=["audio"])
        rec = results[0]
        assert rec.metadata.get("source_path") == "/recordings/call_42.ogg"
        assert rec.metadata.get("media_type") == "audio"

    def test_store_audio_custom_tags(self, brain):
        brain.store_audio_transcript(
            "Patient reports headache and nausea.",
            "/recordings/patient_001.wav",
            tags=["medical", "patient"],
        )
        results = brain.search(tags=["medical"])
        assert len(results) >= 1
        tags = results[0].tags
        assert "medical" in tags
        assert "audio" in tags  # auto-added

    def test_store_audio_with_level(self, brain):
        brain.store_audio_transcript(
            "Key decision: switch to Rust for performance.",
            "/recordings/standup.wav",
            level=Level.Decisions,
        )
        results = brain.search(level=Level.Decisions)
        assert len(results) >= 1

    def test_store_audio_with_namespace(self, brain):
        brain.store_audio_transcript(
            "Remember to buy groceries.",
            "/recordings/personal.wav",
            namespace="personal",
        )
        results = brain.search(namespace="personal")
        assert len(results) >= 1

    def test_store_audio_source_type_is_multimodal(self, brain):
        brain.store_audio_transcript("Test audio", "/test.wav")
        results = brain.search(tags=["audio"])
        assert results[0].source_type == "recorded"

    def test_store_audio_content_type_is_audio(self, brain):
        brain.store_audio_transcript("Another test", "/test2.wav")
        results = brain.search(tags=["audio"])
        assert results[0].content_type == "audio_transcript"


class TestMultimodalRecall:
    """Test that multimodal records participate in recall."""

    def test_image_appears_in_recall(self, brain):
        brain.store_image("/photos/cat.jpg", "A cute cat playing with yarn")
        result = brain.recall("cat")
        assert "cat" in result.lower()

    def test_audio_appears_in_recall(self, brain):
        brain.store_audio_transcript(
            "The project deadline is next Friday.",
            "/recordings/meeting.wav",
        )
        result = brain.recall("project deadline")
        assert "deadline" in result.lower() or "friday" in result.lower()

    def test_mixed_modalities_in_recall(self, brain):
        brain.store("Regular text memory about dogs", level=Level.Working)
        brain.store_image("/img/puppy.jpg", "Golden retriever puppy")
        brain.store_audio_transcript("The vet said the dog is healthy", "/vet.wav")
        results = brain.recall_structured("dog")
        assert len(results) >= 2  # at least 2 of 3 should match
