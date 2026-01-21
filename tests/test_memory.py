"""
Tests for ReUnity Memory and Pattern Recognition Modules.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.
"""

import time
import pytest

from reunity.memory.continuity_store import (
    RecursiveIdentityMemoryEngine,
    ConsentScope,
    MemoryType,
    MemoryEntry,
)
from reunity.protective.pattern_recognizer import (
    ProtectivePatternRecognizer,
    PatternType,
    PatternSeverity,
)


class TestRecursiveIdentityMemoryEngine:
    """Tests for RIME memory engine."""

    @pytest.fixture
    def engine(self):
        """Create a fresh memory engine for each test."""
        return RecursiveIdentityMemoryEngine()

    def test_add_memory(self, engine):
        """Test adding a memory."""
        memory = engine.add_memory(
            identity="primary",
            content="Test memory content",
            memory_type=MemoryType.EPISODIC,
            tags=["test"],
        )
        assert memory.id is not None
        assert memory.content == "Test memory content"
        assert memory.identity_state == "primary"

    def test_add_memory_with_consent_scope(self, engine):
        """Test adding memory with consent scope."""
        memory = engine.add_memory(
            identity="primary",
            content="Private memory",
            consent_scope=ConsentScope.PRIVATE,
        )
        assert memory.consent_scope == ConsentScope.PRIVATE

    def test_retrieve_grounding(self, engine):
        """Test grounding memory retrieval."""
        # Add some memories
        engine.add_memory(
            identity="primary",
            content="Safe place memory",
            memory_type=MemoryType.GROUNDING,
            tags=["safe", "grounding"],
        )
        engine.add_memory(
            identity="primary",
            content="Regular memory",
            memory_type=MemoryType.EPISODIC,
        )

        result = engine.retrieve_grounding(
            current_identity="primary",
            query="safe",
            crisis_level=0.8,
        )

        assert len(result.memories) > 0

    def test_consent_filtering(self, engine):
        """Test that consent scopes are respected."""
        engine.add_memory(
            identity="primary",
            content="Private memory",
            consent_scope=ConsentScope.PRIVATE,
        )
        engine.add_memory(
            identity="primary",
            content="Therapist memory",
            consent_scope=ConsentScope.THERAPIST,
        )

        # Self-only access should see private
        result = engine.retrieve_grounding(
            current_identity="primary",
            query="memory",
            accessor_scope=ConsentScope.SELF_ONLY,
        )
        assert result.filtered_by_consent >= 0

    def test_journal_entry(self, engine):
        """Test adding journal entries."""
        entry = engine.add_journal_entry(
            title="Test Journal",
            content="Journal content",
            identity="primary",
            mood="calm",
            energy_level=0.7,
            entropy_level=0.3,
        )
        assert entry.id is not None
        assert entry.title == "Test Journal"

    def test_get_journals(self, engine):
        """Test retrieving journal entries."""
        engine.add_journal_entry(
            title="Entry 1",
            content="Content 1",
            identity="primary",
            mood="happy",
            energy_level=0.8,
            entropy_level=0.2,
        )
        engine.add_journal_entry(
            title="Entry 2",
            content="Content 2",
            identity="alter",
            mood="sad",
            energy_level=0.3,
            entropy_level=0.6,
        )

        # Get all journals
        all_journals = engine.get_journals()
        assert len(all_journals) == 2

        # Filter by identity
        primary_journals = engine.get_journals(identity="primary")
        assert len(primary_journals) == 1

    def test_link_memories(self, engine):
        """Test linking memories together."""
        mem1 = engine.add_memory(
            identity="primary",
            content="Memory 1",
        )
        mem2 = engine.add_memory(
            identity="primary",
            content="Memory 2",
        )

        success = engine.link_memories(mem1.id, mem2.id)
        assert success

        # Check links
        retrieved = engine._find_memory_by_id(mem1.id)
        assert mem2.id in retrieved.linked_memories

    def test_export_memories(self, engine):
        """Test exporting memories."""
        engine.add_memory(
            identity="primary",
            content="Export test",
        )

        export_data = engine.export_memories()
        assert "memories" in export_data
        assert len(export_data["memories"]) > 0

    def test_get_statistics(self, engine):
        """Test getting memory statistics."""
        engine.add_memory(identity="primary", content="Test 1")
        engine.add_memory(identity="primary", content="Test 2")
        engine.add_memory(identity="alter", content="Test 3")

        stats = engine.get_statistics()
        assert stats["total_memories"] == 3
        assert stats["identity_count"] == 2


class TestProtectivePatternRecognizer:
    """Tests for protective pattern recognition."""

    @pytest.fixture
    def recognizer(self):
        """Create a fresh recognizer for each test."""
        return ProtectivePatternRecognizer()

    def test_analyze_empty_interactions(self, recognizer):
        """Test analyzing empty interactions."""
        result = recognizer.analyze_interactions([])
        assert result.overall_risk == 0.0
        assert result.stability_assessment == "insufficient_data"

    def test_detect_gaslighting(self, recognizer):
        """Test gaslighting detection."""
        interactions = [
            {"text": "You're crazy, that never happened", "timestamp": time.time()},
            {"text": "You're imagining things", "timestamp": time.time()},
        ]
        result = recognizer.analyze_interactions(interactions)

        gaslighting_patterns = [
            p for p in result.patterns_detected
            if p.pattern_type == PatternType.GASLIGHTING
        ]
        assert len(gaslighting_patterns) > 0

    def test_detect_hot_cold_cycle(self, recognizer):
        """Test hot-cold cycle detection."""
        # Create oscillating sentiment pattern
        interactions = [
            {"text": "I love you so much, you're perfect", "timestamp": 1},
            {"text": "I hate everything about you", "timestamp": 2},
            {"text": "You're the best thing in my life", "timestamp": 3},
            {"text": "I can't stand being around you", "timestamp": 4},
            {"text": "I love you more than anything", "timestamp": 5},
            {"text": "You make me miserable", "timestamp": 6},
        ]
        result = recognizer.analyze_interactions(interactions)
        assert result.sentiment_variance > 0

    def test_detect_isolation_attempt(self, recognizer):
        """Test isolation attempt detection."""
        interactions = [
            {"text": "Your friends don't understand you like I do", "timestamp": time.time()},
            {"text": "You don't need anyone else", "timestamp": time.time()},
        ]
        result = recognizer.analyze_interactions(interactions)

        isolation_patterns = [
            p for p in result.patterns_detected
            if p.pattern_type == PatternType.ISOLATION_ATTEMPT
        ]
        assert len(isolation_patterns) > 0

    def test_overall_risk_calculation(self, recognizer):
        """Test overall risk is calculated correctly."""
        interactions = [
            {"text": "Normal conversation", "timestamp": time.time()},
            {"text": "How are you today?", "timestamp": time.time()},
        ]
        result = recognizer.analyze_interactions(interactions)
        assert 0 <= result.overall_risk <= 1

    def test_recommendations_generated(self, recognizer):
        """Test that recommendations are generated."""
        interactions = [
            {"text": "You're crazy", "timestamp": time.time()},
        ]
        result = recognizer.analyze_interactions(interactions)
        assert len(result.recommendations) > 0

    def test_mirror_link_reflection(self, recognizer):
        """Test MirrorLink reflection generation."""
        reflection = recognizer.mirror_link_reflection(
            current_emotion="betrayed and hurt",
            past_context="you called them your anchor last week",
        )
        assert "betrayed" in reflection.lower() or "both" in reflection.lower()

    def test_pattern_severity_levels(self, recognizer):
        """Test that severity levels are assigned."""
        interactions = [
            {"text": "You're crazy and no one will believe you", "timestamp": time.time()},
        ]
        result = recognizer.analyze_interactions(interactions)

        if result.patterns_detected:
            for pattern in result.patterns_detected:
                assert pattern.severity in PatternSeverity


class TestConsentScopes:
    """Tests for consent scope functionality."""

    def test_all_scopes_defined(self):
        """Test all consent scopes are defined."""
        scopes = [
            ConsentScope.PRIVATE,
            ConsentScope.SELF_ONLY,
            ConsentScope.THERAPIST,
            ConsentScope.CAREGIVER,
            ConsentScope.EMERGENCY,
            ConsentScope.RESEARCH,
        ]
        for scope in scopes:
            assert scope.value is not None

    def test_scope_values(self):
        """Test scope values are strings."""
        assert ConsentScope.PRIVATE.value == "private"
        assert ConsentScope.SELF_ONLY.value == "self_only"


class TestMemoryTypes:
    """Tests for memory type functionality."""

    def test_all_types_defined(self):
        """Test all memory types are defined."""
        types = [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.EMOTIONAL,
            MemoryType.RELATIONAL,
            MemoryType.GROUNDING,
            MemoryType.IDENTITY,
            MemoryType.JOURNAL,
        ]
        for mem_type in types:
            assert mem_type.value is not None
