"""
Pytest configuration and fixtures for ReUnity tests.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_distribution():
    """Provide a sample probability distribution."""
    import numpy as np
    return np.array([0.25, 0.25, 0.25, 0.25])


@pytest.fixture
def sample_interactions():
    """Provide sample interaction data."""
    import time
    return [
        {"text": "Hello, how are you?", "timestamp": time.time()},
        {"text": "I'm feeling good today", "timestamp": time.time()},
        {"text": "Thanks for asking", "timestamp": time.time()},
    ]


@pytest.fixture
def sample_memory_data():
    """Provide sample memory data."""
    return {
        "identity": "primary",
        "content": "Test memory content",
        "memory_type": "episodic",
        "tags": ["test"],
        "consent_scope": "private",
        "emotional_valence": 0.5,
        "importance": 0.7,
    }
