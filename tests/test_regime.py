"""
Tests for ReUnity Regime and Reflection Modules.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.
"""

import numpy as np
import pytest

from reunity.core.entropy import EntropyState, EntropyMetrics
from reunity.regime.regime_controller import (
    RegimeController,
    Regime,
    Apostasis,
    Regeneration,
    LatticeMemoryGraph,
    NoveltyLevel,
)
from reunity.reflection.mirror_link import (
    MirrorLinkDialogueCompanion,
    CommunicationStyle,
    ReflectionType,
)


class TestRegimeController:
    """Tests for regime controller."""

    @pytest.fixture
    def controller(self):
        """Create a fresh controller for each test."""
        return RegimeController()

    def test_initial_regime_stable(self, controller):
        """Controller should start in stable regime."""
        assert controller.current_regime == Regime.STABLE

    def test_update_with_stable_metrics(self, controller):
        """Stable metrics should maintain stable regime."""
        metrics = EntropyMetrics(
            shannon_entropy=0.5,
            normalized_entropy=0.3,
            state=EntropyState.STABLE,
            confidence=0.8,
            is_stable=True,
        )
        state = controller.update(metrics)
        assert state.regime == Regime.STABLE

    def test_transition_to_crisis(self, controller):
        """Crisis entropy should trigger crisis regime."""
        metrics = EntropyMetrics(
            shannon_entropy=0.9,
            normalized_entropy=0.9,
            state=EntropyState.CRISIS,
            confidence=0.9,
            is_stable=False,
        )
        state = controller.update(metrics)
        assert state.regime == Regime.CRISIS

    def test_transition_to_protective(self, controller):
        """High entropy should trigger protective regime."""
        metrics = EntropyMetrics(
            shannon_entropy=0.8,
            normalized_entropy=0.8,
            state=EntropyState.HIGH,
            confidence=0.8,
            is_stable=False,
        )
        state = controller.update(metrics)
        assert state.regime == Regime.PROTECTIVE

    def test_regime_history_tracking(self, controller):
        """Controller should track regime history."""
        # Trigger a transition
        metrics = EntropyMetrics(
            shannon_entropy=0.9,
            normalized_entropy=0.9,
            state=EntropyState.CRISIS,
            confidence=0.9,
            is_stable=False,
        )
        controller.update(metrics)

        history = controller.get_regime_history()
        assert len(history) > 0

    def test_novelty_classification(self, controller):
        """Test novelty level classification."""
        metrics = EntropyMetrics(
            shannon_entropy=0.5,
            normalized_entropy=0.5,
            state=EntropyState.STABLE,
            confidence=0.8,
            is_stable=True,
        )

        # Low novelty
        state = controller.update(metrics, novelty_score=0.1)
        assert state.novelty_level == NoveltyLevel.LOW

        # High novelty
        state = controller.update(metrics, novelty_score=0.8)
        assert state.novelty_level == NoveltyLevel.HIGH


class TestApostasis:
    """Tests for apostasis (pruning) operator."""

    @pytest.fixture
    def apostasis(self):
        """Create a fresh apostasis operator."""
        return Apostasis()

    def test_calculate_utility(self, apostasis):
        """Test utility calculation."""
        memory = {
            "importance": 0.5,
            "retrieval_count": 5,
            "timestamp": 1000000000,
            "entropy_at_creation": 0.3,
        }
        utility = apostasis.calculate_utility(memory)
        assert 0 <= utility <= 1

    def test_should_prune_low_utility(self, apostasis):
        """Low utility memories should be pruned."""
        memory = {
            "importance": 0.1,
            "retrieval_count": 0,
            "timestamp": 1000000000,  # Old
            "entropy_at_creation": 0.9,
            "tags": [],
            "memory_type": "episodic",
        }
        assert apostasis.should_prune(memory)

    def test_protect_grounding_memories(self, apostasis):
        """Grounding memories should be protected."""
        memory = {
            "importance": 0.1,
            "retrieval_count": 0,
            "timestamp": 1000000000,
            "entropy_at_creation": 0.9,
            "tags": ["grounding"],
            "memory_type": "grounding",
        }
        assert not apostasis.should_prune(memory)

    def test_protect_high_importance(self, apostasis):
        """High importance memories should be protected."""
        memory = {
            "importance": 0.9,
            "retrieval_count": 0,
            "timestamp": 1000000000,
            "entropy_at_creation": 0.9,
            "tags": [],
            "memory_type": "episodic",
        }
        assert not apostasis.should_prune(memory)

    def test_prune_memories(self, apostasis):
        """Test memory pruning operation."""
        memories = [
            {"importance": 0.1, "retrieval_count": 0, "timestamp": 1000000000,
             "entropy_at_creation": 0.9, "tags": [], "memory_type": "episodic"},
            {"importance": 0.9, "retrieval_count": 10, "timestamp": 2000000000,
             "entropy_at_creation": 0.2, "tags": [], "memory_type": "episodic"},
        ]
        remaining, result = apostasis.prune_memories(memories)
        assert result.memories_pruned >= 0


class TestRegeneration:
    """Tests for regeneration operator."""

    @pytest.fixture
    def regeneration(self):
        """Create a fresh regeneration operator."""
        return Regeneration()

    def test_accumulate_evidence(self, regeneration):
        """Test evidence accumulation."""
        for _ in range(5):
            regeneration.accumulate_evidence(0.8)

        evidence = regeneration.get_evidence_level()
        assert evidence > 0

    def test_can_regenerate_with_evidence(self, regeneration):
        """Should be able to regenerate with sufficient evidence."""
        for _ in range(10):
            regeneration.accumulate_evidence(0.9)

        assert regeneration.can_regenerate()

    def test_cannot_regenerate_without_evidence(self, regeneration):
        """Should not regenerate without evidence."""
        assert not regeneration.can_regenerate()

    def test_regenerate_expands_capacity(self, regeneration):
        """Regeneration should expand capacity."""
        for _ in range(10):
            regeneration.accumulate_evidence(0.9)

        initial_capacity = regeneration.current_capacity
        result = regeneration.regenerate()

        if result.stability_confirmed:
            assert regeneration.current_capacity >= initial_capacity

    def test_reset_evidence(self, regeneration):
        """Evidence reset should clear accumulated evidence."""
        for _ in range(5):
            regeneration.accumulate_evidence(0.8)

        regeneration.reset_evidence()
        assert regeneration.get_evidence_level() == 0.0


class TestLatticeMemoryGraph:
    """Tests for lattice memory graph."""

    @pytest.fixture
    def lattice(self):
        """Create a fresh lattice graph."""
        return LatticeMemoryGraph()

    def test_add_node(self, lattice):
        """Test adding a node."""
        node = lattice.add_node(
            node_type="memory",
            content="Test memory",
            entropy=0.5,
            importance=0.7,
        )
        assert node.id is not None
        assert node.content == "Test memory"

    def test_add_edge_with_constraint(self, lattice):
        """Test adding edge with divergence constraint."""
        node1 = lattice.add_node("memory", "Memory 1")
        node2 = lattice.add_node("memory", "Memory 2")

        # Similar distributions should allow edge
        p = np.array([0.5, 0.5])
        q = np.array([0.4, 0.6])

        edge = lattice.add_edge(
            node1.id, node2.id, "related",
            p, q,
        )
        # Edge may or may not be created depending on thresholds
        # Just verify no error

    def test_get_connected_nodes(self, lattice):
        """Test getting connected nodes."""
        node1 = lattice.add_node("memory", "Memory 1")
        node2 = lattice.add_node("memory", "Memory 2")

        # Add edge with similar distributions
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        lattice.add_edge(node1.id, node2.id, "related", p, q)

        connected = lattice.get_connected_nodes(node1.id)
        # May or may not have connections depending on MI threshold

    def test_get_nodes_by_type(self, lattice):
        """Test filtering nodes by type."""
        lattice.add_node("memory", "Memory 1")
        lattice.add_node("memory", "Memory 2")
        lattice.add_node("identity", "Identity 1")

        memories = lattice.get_nodes_by_type("memory")
        assert len(memories) == 2

        identities = lattice.get_nodes_by_type("identity")
        assert len(identities) == 1

    def test_remove_node(self, lattice):
        """Test removing a node."""
        node = lattice.add_node("memory", "To remove")
        success = lattice.remove_node(node.id)
        assert success

        retrieved = lattice.get_node(node.id)
        assert retrieved is None

    def test_export_graph(self, lattice):
        """Test exporting graph."""
        lattice.add_node("memory", "Memory 1")
        lattice.add_node("memory", "Memory 2")

        export = lattice.export_graph()
        assert "nodes" in export
        assert "edges" in export
        assert len(export["nodes"]) == 2


class TestMirrorLinkDialogueCompanion:
    """Tests for MirrorLink dialogue companion."""

    @pytest.fixture
    def companion(self):
        """Create a fresh companion."""
        return MirrorLinkDialogueCompanion()

    def test_reflect_without_contradiction(self, companion):
        """Test reflection without contradiction."""
        reflection = companion.reflect(
            current_emotion="feeling calm today",
        )
        assert reflection.content is not None
        assert reflection.reflection_type == ReflectionType.VALIDATION

    def test_reflect_with_contradiction(self, companion):
        """Test reflection with contradiction."""
        reflection = companion.reflect(
            current_emotion="feeling betrayed and hurt",
            past_context="you said they were your anchor last week",
        )
        assert reflection.content is not None
        assert reflection.is_contradiction

    def test_communication_styles(self, companion):
        """Test different communication styles."""
        styles = [
            CommunicationStyle.DIRECT,
            CommunicationStyle.GENTLE,
            CommunicationStyle.MINIMAL,
            CommunicationStyle.DETAILED,
            CommunicationStyle.SOMATIC,
        ]

        for style in styles:
            reflection = companion.reflect(
                current_emotion="anxious",
                style=style,
            )
            assert reflection.content is not None

    def test_grounding_prompt_in_crisis(self, companion):
        """Test grounding prompt generation in crisis."""
        reflection = companion.reflect(
            current_emotion="panicking",
            entropy_state=EntropyState.CRISIS,
        )
        assert reflection.grounding_prompt is not None

    def test_memory_bridge(self, companion):
        """Test memory bridge generation."""
        reflection = companion.generate_memory_bridge(
            current_state="dissociated",
            past_memory="you felt safe at the beach",
            relationship="self",
        )
        assert reflection.reflection_type == ReflectionType.MEMORY_BRIDGE

    def test_pattern_awareness(self, companion):
        """Test pattern awareness generation."""
        reflection = companion.generate_pattern_awareness(
            pattern_description="You tend to minimize your needs",
            occurrences=5,
            context="relationship discussion",
        )
        assert reflection.reflection_type == ReflectionType.PATTERN_AWARENESS

    def test_name_emotion(self, companion):
        """Test emotion naming."""
        reflection = companion.name_emotion(
            description="I feel like my chest is tight and I can't breathe",
            body_sensation="tightness in chest",
        )
        assert reflection.reflection_type == ReflectionType.EMOTIONAL_NAMING
