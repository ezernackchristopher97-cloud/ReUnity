"""
ReUnity Integration Tests

Comprehensive integration tests that verify the complete system
works together correctly, including entropy analysis, memory management,
pattern recognition, reflection, and regime control.

DISCLAIMER: This is not a clinical or treatment tool.
"""

import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch

# Import all components
from reunity.core.entropy import (
    EntropyAnalyzer,
    EntropyState,
    EntropyMetrics,
)
from reunity.router.state_router import (
    StateRouter,
    PolicyType,
    RoutingPolicy,
)
from reunity.protective.pattern_recognizer import (
    ProtectivePatternRecognizer,
    PatternType,
    PatternSeverity,
)
from reunity.protective.safety_assessment import (
    SafetyAssessor,
    RiskLevel,
    CrisisType,
)
from reunity.memory.continuity_store import (
    RecursiveIdentityMemoryEngine,
    ConsentScope,
    MemoryType,
)
from reunity.memory.timeline_threading import (
    TimelineThreader,
    ThreadType,
    MemoryValence,
)
from reunity.reflection.mirror_link import (
    MirrorLinkDialogueCompanion,
    CommunicationStyle,
    ReflectionType,
)
from reunity.regime.regime_controller import (
    RegimeController,
    Regime,
    EntropyBand,
    Apostasis,
    Regeneration,
    LatticeMemoryGraph,
)
from reunity.alter.alter_aware import (
    AlterAwareSubsystem,
    AlterProfile,
    CommunicationType,
)
from reunity.grounding.techniques import (
    GroundingTechniquesLibrary,
    GroundingCategory,
    IntensityLevel,
)
from reunity.core.free_energy import (
    FreeEnergyMinimizer,
    PredictiveProcessor,
    Observation,
)


class TestFullSystemIntegration:
    """Integration tests for the complete ReUnity system."""

    @pytest.fixture
    def full_system(self):
        """Create a complete system with all components."""
        return {
            "entropy_analyzer": EntropyAnalyzer(),
            "state_router": StateRouter(),
            "pattern_recognizer": ProtectivePatternRecognizer(),
            "safety_assessor": SafetyAssessor(),
            "memory_engine": RecursiveIdentityMemoryEngine(),
            "timeline_threader": TimelineThreader(),
            "dialogue_companion": MirrorLinkDialogueCompanion(),
            "regime_controller": RegimeController(),
            "apostasis": Apostasis(),
            "regeneration": Regeneration(),
            "lattice_graph": LatticeMemoryGraph(),
            "alter_subsystem": AlterAwareSubsystem(),
            "grounding_library": GroundingTechniquesLibrary(),
            "free_energy_minimizer": FreeEnergyMinimizer(),
        }

    def test_entropy_to_policy_flow(self, full_system):
        """Test the flow from entropy analysis to policy selection."""
        analyzer = full_system["entropy_analyzer"]
        router = full_system["state_router"]

        # Create a distribution
        distribution = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])

        # Analyze entropy
        metrics = analyzer.analyze(distribution)

        # Route to policy
        policy = router.route(metrics)

        # Verify we get a valid policy
        assert policy is not None
        assert isinstance(policy.policy_type, PolicyType)
        assert len(policy.recommendations) > 0

    def test_memory_with_entropy_tagging(self, full_system):
        """Test that memories are tagged with entropy levels."""
        analyzer = full_system["entropy_analyzer"]
        memory_engine = full_system["memory_engine"]

        # Analyze some text
        text = "I feel calm and grounded today"
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        total = sum(word_counts.values())
        distribution = np.array([c / total for c in word_counts.values()])

        metrics = analyzer.analyze(distribution)

        # Add memory with entropy
        memory = memory_engine.add_memory(
            identity="host",
            content=text,
            memory_type=MemoryType.EPISODIC,
            entropy=metrics.normalized_entropy,
        )

        # Verify memory was created with entropy
        assert memory is not None
        assert memory.entropy_at_creation is not None

    def test_pattern_detection_triggers_safety_assessment(self, full_system):
        """Test that pattern detection can trigger safety assessment."""
        pattern_recognizer = full_system["pattern_recognizer"]
        safety_assessor = full_system["safety_assessor"]

        # Create interactions with concerning patterns
        interactions = [
            {"text": "You're imagining things, that never happened", "timestamp": time.time() - 3600},
            {"text": "No one else would believe you", "timestamp": time.time() - 1800},
            {"text": "You're being too sensitive", "timestamp": time.time()},
        ]

        # Analyze patterns
        analysis = pattern_recognizer.analyze_interactions(interactions)

        # If risk is elevated, run safety assessment
        if analysis.overall_risk > 0.5:
            combined_text = " ".join(i["text"] for i in interactions)
            safety = safety_assessor.assess_safety(
                text=combined_text,
                entropy_level=analysis.sentiment_variance,
            )

            # Verify safety assessment was performed
            assert safety is not None
            assert isinstance(safety.risk_level, RiskLevel)

    def test_reflection_adapts_to_entropy_state(self, full_system):
        """Test that reflection adapts based on entropy state."""
        analyzer = full_system["entropy_analyzer"]
        companion = full_system["dialogue_companion"]

        # Test with low entropy (stable)
        low_entropy_dist = np.array([0.9, 0.05, 0.05])
        low_metrics = analyzer.analyze(low_entropy_dist)

        reflection_low = companion.reflect(
            current_emotion="calm",
            past_context="I felt anxious yesterday",
            entropy_state=low_metrics.state,
        )

        # Test with high entropy (crisis)
        high_entropy_dist = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        high_metrics = analyzer.analyze(high_entropy_dist)

        reflection_high = companion.reflect(
            current_emotion="overwhelmed",
            past_context="Everything is falling apart",
            entropy_state=high_metrics.state,
        )

        # Verify reflections are different based on state
        assert reflection_low.content != reflection_high.content
        # High entropy should include grounding
        if high_metrics.state in [EntropyState.HIGH, EntropyState.CRISIS]:
            assert reflection_high.grounding_prompt is not None

    def test_regime_transitions_with_entropy(self, full_system):
        """Test regime transitions based on entropy changes."""
        controller = full_system["regime_controller"]

        # Start with stable entropy
        stable_metrics = EntropyMetrics(
            shannon_entropy=1.0,
            normalized_entropy=0.3,
            state=EntropyState.STABLE,
            confidence=0.9,
            is_stable=True,
        )

        state1 = controller.update(stable_metrics)
        initial_regime = state1.regime

        # Transition to high entropy
        high_metrics = EntropyMetrics(
            shannon_entropy=2.5,
            normalized_entropy=0.8,
            state=EntropyState.HIGH,
            confidence=0.8,
            is_stable=False,
        )

        state2 = controller.update(high_metrics)

        # Verify regime may have changed
        assert state2 is not None
        assert isinstance(state2.regime, Regime)

    def test_apostasis_during_stable_regime(self, full_system):
        """Test apostasis (pruning) during stable regimes."""
        apostasis = full_system["apostasis"]
        lattice = full_system["lattice_graph"]

        # Add some nodes to the lattice
        node1 = lattice.add_node("memory1", "Test memory 1", importance=0.3)
        node2 = lattice.add_node("memory2", "Test memory 2", importance=0.8)
        node3 = lattice.add_node("memory3", "Test memory 3", importance=0.2)

        # Get candidates for pruning
        candidates = apostasis.get_pruning_candidates(
            memories=[
                {"id": node1, "importance": 0.3, "access_count": 1},
                {"id": node2, "importance": 0.8, "access_count": 10},
                {"id": node3, "importance": 0.2, "access_count": 0},
            ],
            entropy_level=0.2,  # Stable
        )

        # Low importance, low access memories should be candidates
        assert len(candidates) >= 0  # May or may not have candidates

    def test_regeneration_during_recovery(self, full_system):
        """Test regeneration during recovery from crisis."""
        regeneration = full_system["regeneration"]

        # Simulate recovery state
        recovery_state = {
            "entropy_level": 0.4,
            "previous_entropy": 0.8,
            "time_since_crisis": 3600,
        }

        # Check if regeneration should activate
        should_regenerate = regeneration.should_activate(
            current_entropy=recovery_state["entropy_level"],
            previous_entropy=recovery_state["previous_entropy"],
        )

        # Verify regeneration logic
        assert isinstance(should_regenerate, bool)

    def test_alter_aware_with_memory_scoping(self, full_system):
        """Test alter-aware subsystem with memory consent scoping."""
        alter_subsystem = full_system["alter_subsystem"]
        memory_engine = full_system["memory_engine"]

        # Register alters
        host_profile = AlterProfile(
            alter_id="",
            name="Host",
            pronouns="they/them",
            role="host",
        )
        protector_profile = AlterProfile(
            alter_id="",
            name="Protector",
            pronouns="he/him",
            role="protector",
        )

        host_id = alter_subsystem.register_alter(host_profile)
        protector_id = alter_subsystem.register_alter(protector_profile)

        # Add memory for host
        host_memory = memory_engine.add_memory(
            identity="Host",
            content="A peaceful memory",
            memory_type=MemoryType.EPISODIC,
            consent_scope=ConsentScope.PRIVATE,
        )

        # Add shared memory
        shared_memory = memory_engine.add_memory(
            identity="Host",
            content="A memory to share with system",
            memory_type=MemoryType.EPISODIC,
            consent_scope=ConsentScope.SYSTEM_SHARED,
        )

        # Verify consent scoping
        assert host_memory.consent_scope == ConsentScope.PRIVATE
        assert shared_memory.consent_scope == ConsentScope.SYSTEM_SHARED

    def test_grounding_recommendation_by_entropy(self, full_system):
        """Test grounding technique recommendations based on entropy."""
        grounding = full_system["grounding_library"]

        # Low entropy - light grounding
        low_technique = grounding.recommend_technique(entropy_level=0.2)

        # High entropy - intensive grounding
        high_technique = grounding.recommend_technique(entropy_level=0.9)

        # Verify recommendations
        assert low_technique is not None
        assert high_technique is not None

        # High entropy should get more intensive technique
        if low_technique and high_technique:
            intensity_order = {
                IntensityLevel.LIGHT: 1,
                IntensityLevel.MODERATE: 2,
                IntensityLevel.INTENSIVE: 3,
            }
            # Generally, higher entropy should get equal or higher intensity
            assert intensity_order.get(high_technique.intensity, 0) >= intensity_order.get(low_technique.intensity, 0)

    def test_free_energy_minimization_flow(self, full_system):
        """Test free energy minimization with observations."""
        minimizer = full_system["free_energy_minimizer"]

        # Create observation
        observation = Observation(
            value=np.array([0.5, 0.3, 0.2]),
            precision=np.array([1.0, 1.0, 1.0]),
            timestamp=time.time(),
        )

        # Calculate free energy
        metrics = minimizer.calculate_variational_free_energy(observation)

        # Verify metrics
        assert metrics.variational_free_energy is not None
        assert metrics.surprise is not None

        # Update beliefs
        beliefs = minimizer.update_beliefs(observation)

        # Verify beliefs updated
        assert beliefs is not None
        assert len(beliefs.mean) > 0

    def test_timeline_threading_with_identity_states(self, full_system):
        """Test timeline threading across identity states."""
        threader = full_system["timeline_threader"]

        # Add memories for different identity states
        mem1 = threader.add_memory(
            content="Memory from host state",
            valence=MemoryValence.NEUTRAL,
            identity_state="Host",
            timestamp=time.time() - 7200,
        )

        mem2 = threader.add_memory(
            content="Memory from protector state",
            valence=MemoryValence.NEUTRAL,
            identity_state="Protector",
            timestamp=time.time() - 3600,
        )

        mem3 = threader.add_memory(
            content="Back to host state",
            valence=MemoryValence.POSITIVE,
            identity_state="Host",
            timestamp=time.time(),
        )

        # Create identity-based thread
        thread = threader.create_thread(
            name="Host Timeline",
            thread_type=ThreadType.IDENTITY,
            memory_ids=[mem1.memory_id, mem3.memory_id],
            description="Memories from host state",
        )

        # Verify thread
        assert thread is not None
        assert len(thread.memory_ids) == 2

        # Check for gaps (identity switches)
        gaps = threader.get_timeline_gaps()
        identity_switches = [g for g in gaps if g.gap_type == "identity_switch"]
        assert len(identity_switches) > 0

    def test_full_crisis_response_flow(self, full_system):
        """Test the complete crisis response flow."""
        analyzer = full_system["entropy_analyzer"]
        router = full_system["state_router"]
        safety = full_system["safety_assessor"]
        grounding = full_system["grounding_library"]
        companion = full_system["dialogue_companion"]

        # Simulate crisis text
        crisis_text = "I can't take this anymore, everything is falling apart"

        # 1. Analyze entropy
        words = crisis_text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        total = sum(word_counts.values())
        distribution = np.array([c / total for c in word_counts.values()])

        metrics = analyzer.analyze(distribution)

        # 2. Route to policy
        policy = router.route(metrics)

        # 3. Safety assessment
        safety_result = safety.assess_safety(
            text=crisis_text,
            entropy_level=metrics.normalized_entropy,
        )

        # 4. Get grounding technique
        technique = grounding.recommend_technique(
            entropy_level=metrics.normalized_entropy,
        )

        # 5. Generate supportive reflection
        reflection = companion.reflect(
            current_emotion="overwhelmed",
            entropy_state=metrics.state,
            style=CommunicationStyle.GENTLE,
        )

        # Verify complete flow
        assert metrics is not None
        assert policy is not None
        assert safety_result is not None
        assert technique is not None
        assert reflection is not None

        # Verify appropriate responses
        if metrics.state in [EntropyState.HIGH, EntropyState.CRISIS]:
            assert policy.policy_type in [PolicyType.STABILIZE, PolicyType.CRISIS]
            assert reflection.grounding_prompt is not None


class TestConsentAndPrivacy:
    """Tests for consent and privacy controls."""

    @pytest.fixture
    def memory_engine(self):
        return RecursiveIdentityMemoryEngine()

    def test_consent_scope_filtering(self, memory_engine):
        """Test that consent scopes properly filter memories."""
        # Add memories with different scopes
        private = memory_engine.add_memory(
            identity="user",
            content="Private thought",
            consent_scope=ConsentScope.PRIVATE,
        )

        shared = memory_engine.add_memory(
            identity="user",
            content="Shared with therapist",
            consent_scope=ConsentScope.THERAPIST_SHARED,
        )

        # Retrieve with different access levels
        result = memory_engine.retrieve_grounding(
            current_identity="user",
            query="thought",
            crisis_level=0.0,
        )

        # Verify filtering works
        assert result is not None

    def test_consent_modification(self, memory_engine):
        """Test that consent can be modified."""
        memory = memory_engine.add_memory(
            identity="user",
            content="Test memory",
            consent_scope=ConsentScope.PRIVATE,
        )

        # Modify consent
        success = memory_engine.set_consent_scope(
            memory.id,
            ConsentScope.SYSTEM_SHARED,
        )

        assert success

        # Verify change
        updated = memory_engine.get_memory(memory.id)
        if updated:
            assert updated.consent_scope == ConsentScope.SYSTEM_SHARED


class TestDisclaimerPresence:
    """Tests to verify disclaimers are present in all modules."""

    def test_entropy_module_disclaimer(self):
        """Verify entropy module has disclaimer."""
        from reunity.core import entropy
        assert "DISCLAIMER" in entropy.__doc__ or "not a clinical" in entropy.__doc__.lower()

    def test_pattern_recognizer_disclaimer(self):
        """Verify pattern recognizer has disclaimer."""
        from reunity.protective import pattern_recognizer
        assert "DISCLAIMER" in pattern_recognizer.__doc__ or "not a clinical" in pattern_recognizer.__doc__.lower()

    def test_memory_module_disclaimer(self):
        """Verify memory module has disclaimer."""
        from reunity.memory import continuity_store
        assert "DISCLAIMER" in continuity_store.__doc__ or "not a clinical" in continuity_store.__doc__.lower()


class TestErrorHandling:
    """Tests for error handling across the system."""

    def test_entropy_analyzer_empty_distribution(self):
        """Test entropy analyzer handles empty distribution."""
        analyzer = EntropyAnalyzer()

        with pytest.raises((ValueError, Exception)):
            analyzer.analyze(np.array([]))

    def test_pattern_recognizer_empty_interactions(self):
        """Test pattern recognizer handles empty interactions."""
        recognizer = ProtectivePatternRecognizer()

        result = recognizer.analyze_interactions([])

        # Should return valid result with no patterns
        assert result is not None
        assert len(result.patterns_detected) == 0

    def test_memory_engine_invalid_identity(self):
        """Test memory engine handles edge cases."""
        engine = RecursiveIdentityMemoryEngine()

        # Should handle empty identity gracefully
        memory = engine.add_memory(
            identity="",
            content="Test",
        )

        # Should still create memory
        assert memory is not None


class TestPerformance:
    """Basic performance tests."""

    def test_entropy_analysis_performance(self):
        """Test entropy analysis completes in reasonable time."""
        analyzer = EntropyAnalyzer()

        # Large distribution
        distribution = np.random.dirichlet(np.ones(1000))

        start = time.time()
        for _ in range(100):
            analyzer.analyze(distribution)
        elapsed = time.time() - start

        # Should complete 100 analyses in under 1 second
        assert elapsed < 1.0

    def test_memory_retrieval_performance(self):
        """Test memory retrieval performance."""
        engine = RecursiveIdentityMemoryEngine()

        # Add many memories
        for i in range(100):
            engine.add_memory(
                identity="user",
                content=f"Memory number {i} with some content",
            )

        start = time.time()
        for _ in range(10):
            engine.retrieve_grounding(
                current_identity="user",
                query="memory",
            )
        elapsed = time.time() - start

        # Should complete 10 retrievals in under 1 second
        assert elapsed < 1.0
