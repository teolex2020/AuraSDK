"""Shared fixtures for Aura Python integration tests."""

import tempfile
import os
import pytest
from aura import Aura


@pytest.fixture
def brain():
    """Create a temporary Aura brain for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        b = Aura(os.path.join(tmp, "test.db"))
        yield b
        b.close()


@pytest.fixture
def populated_brain():
    """Create a brain pre-populated with test data."""
    with tempfile.TemporaryDirectory() as tmp:
        b = Aura(os.path.join(tmp, "test.db"))
        from aura import Level

        b.store("Python is a dynamic language", level=Level.Domain, tags=["python", "lang"])
        b.store("Rust has zero-cost abstractions", level=Level.Domain, tags=["rust", "lang"])
        b.store("Fix the auth bug", level=Level.Working, tags=["bug", "auth"])
        b.store("Always deploy to staging first", level=Level.Decisions, tags=["workflow", "deploy"])
        b.store("User prefers dark mode", level=Level.Identity, tags=["preference", "ui"])
        yield b
        b.close()
