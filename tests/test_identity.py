"""Identity, persona, and research tests."""

import pytest
from aura import Aura, Level, AgentPersona


class TestUserProfile:
    def test_store_user_profile(self, brain):
        pid = brain.store_user_profile({"name": "Teo", "language": "Ukrainian"})
        assert isinstance(pid, str)

    def test_get_user_profile(self, brain):
        brain.store_user_profile({"name": "Teo", "role": "developer"})
        profile = brain.get_user_profile()
        assert profile is not None
        assert "Teo" in str(profile)

    def test_get_user_profile_empty(self, brain):
        profile = brain.get_user_profile()
        # No profile stored yet — may return None or empty
        # Just verify it doesn't crash


class TestPersona:
    def test_set_persona(self, brain):
        persona = AgentPersona()
        persona.name = "Atlas"
        persona.role = "Research Assistant"
        pid = brain.set_persona(persona)
        assert isinstance(pid, str)

    def test_get_persona(self, brain):
        persona = AgentPersona()
        persona.name = "Atlas"
        brain.set_persona(persona)
        got = brain.get_persona()
        assert got is not None

    def test_get_persona_empty(self, brain):
        result = brain.get_persona()
        # No persona set — may return None

    def test_persona_attributes(self):
        persona = AgentPersona()
        assert hasattr(persona, "name")
        assert hasattr(persona, "role")


class TestResearch:
    def test_start_research(self, brain):
        project = brain.start_research("AI Safety")
        assert isinstance(project, dict)
        assert "id" in project
        assert "topic" in project

    def test_add_finding(self, brain):
        project = brain.start_research("Quantum Computing")
        pid = project["id"]
        brain.add_research_finding(pid, "What is a qubit?",
                                    "A qubit is a quantum bit")

    def test_add_finding_with_url(self, brain):
        project = brain.start_research("ML")
        pid = project["id"]
        brain.add_research_finding(pid, "What is backprop?",
                                    "Backpropagation computes gradients",
                                    "https://arxiv.org/paper")

    def test_complete_research(self, brain):
        project = brain.start_research("Testing")
        pid = project["id"]
        brain.add_research_finding(pid, "q1", "result1")
        brain.add_research_finding(pid, "q2", "result2")
        synthesis_id = brain.complete_research(pid)
        assert isinstance(synthesis_id, str)

    def test_complete_research_creates_record(self, brain):
        project = brain.start_research("Test topic")
        pid = project["id"]
        brain.add_research_finding(pid, "q", "r")
        synthesis_id = brain.complete_research(pid)
        rec = brain.get(synthesis_id)
        assert rec is not None
        assert "Test topic" in rec.content or len(rec.content) > 0

    def test_complete_research_with_synthesis(self, brain):
        project = brain.start_research("Custom")
        pid = project["id"]
        brain.add_research_finding(pid, "q", "r")
        synthesis_id = brain.complete_research(pid, synthesis="My custom summary")
        rec = brain.get(synthesis_id)
        assert rec is not None

    def test_finding_for_nonexistent_project(self, brain):
        with pytest.raises(RuntimeError):
            brain.add_research_finding("nonexistent", "q", "r")

    def test_complete_nonexistent_project(self, brain):
        with pytest.raises(RuntimeError):
            brain.complete_research("nonexistent")


class TestCircuitBreaker:
    def test_tool_available_by_default(self, brain):
        assert brain.is_tool_available("any_tool") is True

    def test_failures_open_circuit(self, brain):
        for _ in range(5):
            brain.record_tool_failure("bad_tool")
        assert brain.is_tool_available("bad_tool") is False

    def test_success_resets(self, brain):
        brain.record_tool_failure("tool")
        brain.record_tool_success("tool")
        assert brain.is_tool_available("tool") is True

    def test_tool_health_returns_dict(self, brain):
        health = brain.tool_health()
        assert isinstance(health, dict)

    def test_tool_health_shows_broken(self, brain):
        for _ in range(5):
            brain.record_tool_failure("broken")
        health = brain.tool_health()
        assert "broken" in health
        assert "unavailable" in health["broken"].lower()
