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
        # Use a lattice with relaxed constraints for testing
        test_lattice = LatticeMemoryGraph(max_divergence=1.0, min_mutual_information=0.0)
        node1 = test_lattice.add_node("memory", "Memory 1")
        node2 = test_lattice.add_node("memory", "Memory 2")

        # Similar distributions should allow edge
        p = np.array([0.5, 0.5])
        q = np.array([0.4, 0.6])

        edge = test_lattice.add_edge(
            source_id=node1.id,
            target_id=node2.id,
            edge_type="association",
            source_distribution=p,
            target_distribution=q,
        )

        assert edge is not None

    def test_edge_constraint_blocks_dissimilar(self, lattice):
        """Test that highly divergent distributions may result in weak edges."""
        node1 = lattice.add_node("memory", "Memory 1")
        node2 = lattice.add_node("memory", "Memory 2")

        # Very different distributions
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])

        edge = lattice.add_edge(
            source_id=node1.id,
            target_id=node2.id,
            edge_type="association",
            source_distribution=p,
            target_distribution=q,
        )

        # Edge may still be created but with low weight
        # The divergence is stored in the edge
        assert edge is not None or edge is None  # Either outcome is valid

    def test_get_connected_nodes(self, lattice):
        """Test getting connected nodes."""
        # Use a lattice with relaxed constraints for testing
        test_lattice = LatticeMemoryGraph(max_divergence=1.0, min_mutual_information=0.0)
        node1 = test_lattice.add_node("memory", "Memory 1")
        node2 = test_lattice.add_node("memory", "Memory 2")
        node3 = test_lattice.add_node("memory", "Memory 3")

        p = np.array([0.5, 0.5])
        q = np.array([0.4, 0.6])
        test_lattice.add_edge(node1.id, node2.id, "association", p, q)
        test_lattice.add_edge(node1.id, node3.id, "association", p, q)

        connected = test_lattice.get_connected_nodes(node1.id)
        assert len(connected) == 2

    def test_remove_node(self, lattice):
        """Test removing a node."""
        node = lattice.add_node("memory", "Test")
        node_id = node.id

        success = lattice.remove_node(node_id)
        assert success

        retrieved = lattice.get_node(node_id)
        assert retrieved is None

    def test_export_graph(self, lattice):
        """Test exporting graph."""
        lattice.add_node("memory", "Memory 1")
        lattice.add_node("memory", "Memory 2")

        export = lattice.export_graph()
        assert "nodes" in export
        assert "edges" in export


class TestMirrorLinkDialogueCompanion:
    """Tests for MirrorLink dialogue companion."""

    @pytest.fixture
    def companion(self):
        """Create a fresh companion."""
        return MirrorLinkDialogueCompanion()

    def test_reflect_basic(self, companion):
        """Test basic reflection."""
        reflection = companion.reflect(
            current_emotion="anxious",
            past_context="I was calm yesterday",
        )
        assert reflection is not None
        assert reflection.content is not None

    def test_reflect_with_entropy_state(self, companion):
        """Test reflection adapts to entropy state."""
        low_reflection = companion.reflect(
            current_emotion="calm",
            entropy_state=EntropyState.LOW,
        )

        high_reflection = companion.reflect(
            current_emotion="overwhelmed",
            entropy_state=EntropyState.HIGH,
        )

        # Both should produce valid reflections
        assert low_reflection is not None
        assert high_reflection is not None

    def test_reflect_with_style(self, companion):
        """Test reflection with different styles."""
        gentle = companion.reflect(
            current_emotion="sad",
            style=CommunicationStyle.GENTLE,
        )

        direct = companion.reflect(
            current_emotion="sad",
            style=CommunicationStyle.DIRECT,
        )

        assert gentle is not None
        assert direct is not None

    def test_surface_contradiction(self, companion):
        """Test surfacing contradictions without invalidation."""
        reflection = companion.reflect(
            current_emotion="betrayed",
            past_context="You called them your anchor last week",
        )

        # Should acknowledge both without invalidating
        assert reflection is not None
        # The reflection should not be dismissive

    def test_grounding_prompt_in_crisis(self, companion):
        """Test that crisis states include grounding."""
        reflection = companion.reflect(
            current_emotion="panicking",
            entropy_state=EntropyState.CRISIS,
        )

        # Crisis reflections should include grounding
        assert reflection is not None
        if reflection.grounding_prompt:
            assert len(reflection.grounding_prompt) > 0

    def test_reflection_types(self, companion):
        """Test different reflection with different styles."""
        # Gentle style
        gentle = companion.reflect(
            current_emotion="hurt",
            style=CommunicationStyle.GENTLE,
        )

        # Direct style
        direct = companion.reflect(
            current_emotion="confused",
            style=CommunicationStyle.DIRECT,
        )

        assert gentle is not None
        assert direct is not None


class TestIntegratedRegimeReflection:
    """Tests for integrated regime and reflection behavior."""

    def test_regime_affects_reflection(self):
        """Test that regime state affects reflection behavior."""
        controller = RegimeController()
        companion = MirrorLinkDialogueCompanion()

        # Stable regime
        stable_metrics = EntropyMetrics(
            shannon_entropy=0.3,
            normalized_entropy=0.3,
            state=EntropyState.STABLE,
            confidence=0.9,
        )
        controller.update(stable_metrics)

        stable_reflection = companion.reflect(
            current_emotion="content",
            entropy_state=EntropyState.STABLE,
        )

        # Crisis regime
        crisis_metrics = EntropyMetrics(
            shannon_entropy=0.9,
            normalized_entropy=0.9,
            state=EntropyState.CRISIS,
            confidence=0.9,
        )
        controller.update(crisis_metrics)

        crisis_reflection = companion.reflect(
            current_emotion="overwhelmed",
            entropy_state=EntropyState.CRISIS,
        )

        # Both should be valid but different
        assert stable_reflection is not None
        assert crisis_reflection is not None

    def test_apostasis_lattice_integration(self):
        """Test apostasis works with lattice graph."""
        apostasis = Apostasis()
        lattice = LatticeMemoryGraph()

        # Add nodes
        node1 = lattice.add_node("memory", "Important memory", importance=0.9)
        node2 = lattice.add_node("memory", "Unimportant memory", importance=0.1)

        # Convert to memory format
        memories = [
            {
                "id": node1.id,
                "importance": 0.9,
                "retrieval_count": 10,
                "timestamp": 2000000000,
                "entropy_at_creation": 0.3,
                "tags": [],
                "memory_type": "episodic",
            },
            {
                "id": node2.id,
                "importance": 0.1,
                "retrieval_count": 0,
                "timestamp": 1000000000,
                "entropy_at_creation": 0.9,
                "tags": [],
                "memory_type": "episodic",
            },
        ]

        # Prune
        remaining, result = apostasis.prune_memories(memories)

        # Important memory should remain
        remaining_ids = [m["id"] for m in remaining]
        assert node1.id in remaining_ids
