"""
ReUnity Regime Logic Module

This module implements the regime controller that manages system behavior
based on entropy bands, confidence levels, and novelty detection. It includes:

1. Regime Logic - Controller switching behavior based on entropy, confidence, novelty
2. Apostasis - Pruning/forgetting operator for low-utility memories during stable regimes
3. Regeneration - Controlled restoration when stability returns
4. Lattice Function - Discrete state graph with divergence-constrained edges

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from reunity.core.entropy import (
    EntropyState,
    EntropyMetrics,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
)


class Regime(Enum):
    """Operating regimes for the system."""

    STABLE = "stable"  # Normal operation, apostasis active
    ELEVATED = "elevated"  # Increased monitoring
    PROTECTIVE = "protective"  # Protective measures active
    CRISIS = "crisis"  # Crisis mode, regeneration paused
    RECOVERY = "recovery"  # Post-crisis recovery, regeneration active


class NoveltyLevel(Enum):
    """Levels of novelty in input."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class RegimeState:
    """Current state of the regime controller."""

    regime: Regime
    entropy_band: EntropyState
    confidence: float
    novelty_level: NoveltyLevel
    time_in_regime: float
    transitions_count: int
    apostasis_active: bool
    regeneration_active: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApostasisResult:
    """Result of an apostasis (pruning) operation."""

    memories_pruned: int
    memories_downweighted: int
    total_utility_removed: float
    pruning_criteria: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RegenerationResult:
    """Result of a regeneration operation."""

    memories_restored: int
    capacity_expanded: float
    evidence_threshold_met: bool
    stability_confirmed: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatticeNode:
    """A node in the lattice memory graph."""

    id: str
    node_type: str  # "identity", "memory", "relationship", "emotion"
    content: str
    entropy_at_creation: float
    importance: float
    connections: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LatticeEdge:
    """An edge in the lattice memory graph."""

    id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    divergence: float  # JS divergence constraint
    mutual_information: float
    metadata: dict[str, Any] = field(default_factory=dict)


class RegimeController:
    """
    Regime Controller for managing system behavior.

    The controller switches behavior based on:
    - Entropy bands (LOW, STABLE, ELEVATED, HIGH, CRISIS)
    - Confidence levels in state detection
    - Novelty detection in inputs

    It coordinates apostasis (pruning) and regeneration operations
    to maintain system health and memory efficiency.

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        stability_threshold: float = 0.3,
        confidence_threshold: float = 0.6,
        novelty_threshold: float = 0.5,
        min_stable_time: float = 300.0,  # 5 minutes
        apostasis_interval: float = 3600.0,  # 1 hour
    ) -> None:
        """
        Initialize the regime controller.

        Args:
            stability_threshold: Threshold for stable regime.
            confidence_threshold: Minimum confidence for regime changes.
            novelty_threshold: Threshold for high novelty detection.
            min_stable_time: Minimum time in stable regime for apostasis.
            apostasis_interval: Interval between apostasis operations.
        """
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.novelty_threshold = novelty_threshold
        self.min_stable_time = min_stable_time
        self.apostasis_interval = apostasis_interval

        # Current state
        self._current_regime = Regime.STABLE
        self._regime_start_time = time.time()
        self._transitions_count = 0
        self._last_apostasis = 0.0
        self._last_regeneration = 0.0

        # History tracking
        self._regime_history: list[tuple[Regime, float]] = []
        self._novelty_history: list[float] = []

        # Callbacks
        self._regime_change_callbacks: list[Callable[[Regime, Regime], None]] = []

    def update(
        self,
        entropy_metrics: EntropyMetrics,
        novelty_score: float = 0.0,
    ) -> RegimeState:
        """
        Update regime based on current metrics.

        Args:
            entropy_metrics: Current entropy analysis.
            novelty_score: Novelty score of recent input (0-1).

        Returns:
            Current RegimeState.
        """
        # Track novelty
        self._novelty_history.append(novelty_score)
        if len(self._novelty_history) > 100:
            self._novelty_history = self._novelty_history[-100:]

        # Determine novelty level
        novelty_level = self._classify_novelty(novelty_score)

        # Determine target regime
        target_regime = self._determine_regime(
            entropy_metrics.state,
            entropy_metrics.confidence,
            novelty_level,
        )

        # Check if regime change is warranted
        if target_regime != self._current_regime:
            if entropy_metrics.confidence >= self.confidence_threshold:
                self._transition_regime(target_regime)

        # Calculate time in current regime
        time_in_regime = time.time() - self._regime_start_time

        # Determine if apostasis/regeneration should be active
        apostasis_active = self._should_apostasis_be_active(time_in_regime)
        regeneration_active = self._should_regeneration_be_active()

        return RegimeState(
            regime=self._current_regime,
            entropy_band=entropy_metrics.state,
            confidence=entropy_metrics.confidence,
            novelty_level=novelty_level,
            time_in_regime=time_in_regime,
            transitions_count=self._transitions_count,
            apostasis_active=apostasis_active,
            regeneration_active=regeneration_active,
            metadata={
                "novelty_score": novelty_score,
                "entropy_value": entropy_metrics.normalized_entropy,
            },
        )

    def _classify_novelty(self, score: float) -> NoveltyLevel:
        """Classify novelty score into level."""
        if score < 0.3:
            return NoveltyLevel.LOW
        elif score < self.novelty_threshold:
            return NoveltyLevel.MODERATE
        else:
            return NoveltyLevel.HIGH

    def _determine_regime(
        self,
        entropy_state: EntropyState,
        confidence: float,
        novelty: NoveltyLevel,
    ) -> Regime:
        """Determine target regime based on inputs."""
        # Crisis state always triggers crisis regime
        if entropy_state == EntropyState.CRISIS:
            return Regime.CRISIS

        # High entropy triggers protective regime
        if entropy_state == EntropyState.HIGH:
            return Regime.PROTECTIVE

        # Elevated entropy with high novelty triggers elevated regime
        if entropy_state == EntropyState.ELEVATED:
            if novelty == NoveltyLevel.HIGH:
                return Regime.PROTECTIVE
            return Regime.ELEVATED

        # Coming from crisis/protective with improving metrics
        if self._current_regime in [Regime.CRISIS, Regime.PROTECTIVE]:
            if entropy_state in [EntropyState.STABLE, EntropyState.LOW]:
                return Regime.RECOVERY

        # Recovery transitions to stable after sufficient time
        if self._current_regime == Regime.RECOVERY:
            time_in_regime = time.time() - self._regime_start_time
            if time_in_regime > self.min_stable_time:
                return Regime.STABLE

        # Default to stable for low/stable entropy
        if entropy_state in [EntropyState.LOW, EntropyState.STABLE]:
            return Regime.STABLE

        return self._current_regime

    def _transition_regime(self, new_regime: Regime) -> None:
        """Transition to a new regime."""
        old_regime = self._current_regime

        # Record history
        self._regime_history.append((old_regime, time.time()))

        # Update state
        self._current_regime = new_regime
        self._regime_start_time = time.time()
        self._transitions_count += 1

        # Execute callbacks
        for callback in self._regime_change_callbacks:
            try:
                callback(old_regime, new_regime)
            except Exception:
                pass

    def _should_apostasis_be_active(self, time_in_regime: float) -> bool:
        """Determine if apostasis should be active."""
        # Only active in stable regime
        if self._current_regime != Regime.STABLE:
            return False

        # Need minimum time in stable regime
        if time_in_regime < self.min_stable_time:
            return False

        return True

    def _should_regeneration_be_active(self) -> bool:
        """Determine if regeneration should be active."""
        # Active in recovery and stable regimes
        return self._current_regime in [Regime.RECOVERY, Regime.STABLE]

    def register_regime_callback(
        self,
        callback: Callable[[Regime, Regime], None],
    ) -> None:
        """Register a callback for regime changes."""
        self._regime_change_callbacks.append(callback)

    def get_regime_history(self) -> list[tuple[Regime, float]]:
        """Get regime transition history."""
        return self._regime_history.copy()

    @property
    def current_regime(self) -> Regime:
        """Get current regime."""
        return self._current_regime


class Apostasis:
    """
    Apostasis - Pruning/Forgetting Operator.

    Implements controlled pruning of low-utility memories during stable
    regimes. This helps maintain system efficiency while preserving
    important memories.

    Apostasis is only active during stable regimes and uses multiple
    criteria to determine what can be safely pruned:
    - Low retrieval frequency
    - Low importance scores
    - High entropy at creation (unstable memories)
    - Age-based decay

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        utility_threshold: float = 0.2,
        age_decay_factor: float = 0.01,
        min_retrievals: int = 2,
        protect_grounding: bool = True,
        protect_high_importance: float = 0.8,
    ) -> None:
        """
        Initialize apostasis operator.

        Args:
            utility_threshold: Threshold below which memories may be pruned.
            age_decay_factor: Factor for age-based utility decay.
            min_retrievals: Minimum retrievals to protect from pruning.
            protect_grounding: Whether to protect grounding memories.
            protect_high_importance: Importance threshold for protection.
        """
        self.utility_threshold = utility_threshold
        self.age_decay_factor = age_decay_factor
        self.min_retrievals = min_retrievals
        self.protect_grounding = protect_grounding
        self.protect_high_importance = protect_high_importance

        # Tracking
        self._pruning_history: list[ApostasisResult] = []

    def calculate_utility(
        self,
        memory: dict[str, Any],
        current_time: float | None = None,
    ) -> float:
        """
        Calculate utility score for a memory.

        Utility is based on:
        - Importance score
        - Retrieval frequency
        - Recency
        - Entropy at creation (lower is better)

        Args:
            memory: Memory dictionary with required fields.
            current_time: Current timestamp (defaults to now).

        Returns:
            Utility score (0-1).
        """
        current_time = current_time or time.time()

        importance = memory.get("importance", 0.5)
        retrieval_count = memory.get("retrieval_count", 0)
        timestamp = memory.get("timestamp", current_time)
        entropy_at_creation = memory.get("entropy_at_creation", 0.5)

        # Age factor (decays over time)
        age_days = (current_time - timestamp) / 86400
        age_factor = np.exp(-self.age_decay_factor * age_days)

        # Retrieval factor (more retrievals = higher utility)
        retrieval_factor = min(1.0, retrieval_count / 10)

        # Entropy factor (lower entropy at creation = higher utility)
        entropy_factor = 1.0 - entropy_at_creation

        # Combined utility
        utility = (
            0.3 * importance +
            0.25 * retrieval_factor +
            0.25 * age_factor +
            0.2 * entropy_factor
        )

        return float(utility)

    def should_prune(self, memory: dict[str, Any]) -> bool:
        """
        Determine if a memory should be pruned.

        Args:
            memory: Memory dictionary.

        Returns:
            True if memory should be pruned.
        """
        # Protect grounding memories
        if self.protect_grounding:
            tags = memory.get("tags", [])
            memory_type = memory.get("memory_type", "")
            if "grounding" in tags or "safe" in tags or memory_type == "grounding":
                return False

        # Protect high importance memories
        if memory.get("importance", 0) >= self.protect_high_importance:
            return False

        # Protect frequently retrieved memories
        if memory.get("retrieval_count", 0) >= self.min_retrievals:
            return False

        # Check utility threshold
        utility = self.calculate_utility(memory)
        return utility < self.utility_threshold

    def prune_memories(
        self,
        memories: list[dict[str, Any]],
        max_prune: int | None = None,
    ) -> tuple[list[dict[str, Any]], ApostasisResult]:
        """
        Prune low-utility memories from a list.

        Args:
            memories: List of memory dictionaries.
            max_prune: Maximum number to prune (None = no limit).

        Returns:
            Tuple of (remaining memories, pruning result).
        """
        to_prune = []
        to_keep = []

        for memory in memories:
            if self.should_prune(memory):
                to_prune.append(memory)
            else:
                to_keep.append(memory)

        # Apply max_prune limit
        if max_prune and len(to_prune) > max_prune:
            # Sort by utility and prune lowest
            to_prune.sort(key=lambda m: self.calculate_utility(m))
            to_prune = to_prune[:max_prune]
            to_keep.extend(to_prune[max_prune:])

        # Calculate total utility removed
        total_utility = sum(self.calculate_utility(m) for m in to_prune)

        result = ApostasisResult(
            memories_pruned=len(to_prune),
            memories_downweighted=0,
            total_utility_removed=total_utility,
            pruning_criteria=f"utility < {self.utility_threshold}",
        )

        self._pruning_history.append(result)

        return to_keep, result

    def downweight_unstable(
        self,
        memories: list[dict[str, Any]],
        entropy_threshold: float = 0.7,
        weight_factor: float = 0.5,
    ) -> int:
        """
        Downweight memories created during unstable states.

        Instead of pruning, reduces the importance of memories
        created during high-entropy states.

        Args:
            memories: List of memory dictionaries.
            entropy_threshold: Entropy threshold for downweighting.
            weight_factor: Factor to multiply importance by.

        Returns:
            Number of memories downweighted.
        """
        count = 0

        for memory in memories:
            entropy = memory.get("entropy_at_creation", 0.5)
            if entropy > entropy_threshold:
                current_importance = memory.get("importance", 0.5)
                memory["importance"] = current_importance * weight_factor
                count += 1

        return count

    def get_pruning_history(self) -> list[ApostasisResult]:
        """Get history of pruning operations."""
        return self._pruning_history.copy()


class Regeneration:
    """
    Regeneration - Controlled Restoration Operator.

    Implements controlled restoration of system capacity when stability
    returns. This includes:
    - Re-expanding memory capacity
    - Restoring access to previously restricted features
    - Rebuilding confidence in state detection

    Regeneration is gated by evidence of sustained stability.

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        stability_window: int = 10,
        evidence_threshold: float = 0.7,
        expansion_rate: float = 0.1,
        max_capacity: float = 1.0,
    ) -> None:
        """
        Initialize regeneration operator.

        Args:
            stability_window: Number of observations for stability check.
            evidence_threshold: Threshold for evidence accumulation.
            expansion_rate: Rate of capacity expansion per cycle.
            max_capacity: Maximum capacity (1.0 = full).
        """
        self.stability_window = stability_window
        self.evidence_threshold = evidence_threshold
        self.expansion_rate = expansion_rate
        self.max_capacity = max_capacity

        # State
        self._current_capacity = 0.5
        self._stability_evidence: list[float] = []
        self._regeneration_history: list[RegenerationResult] = []

    def accumulate_evidence(self, stability_score: float) -> float:
        """
        Accumulate evidence of stability.

        Args:
            stability_score: Current stability score (0-1).

        Returns:
            Current accumulated evidence level.
        """
        self._stability_evidence.append(stability_score)

        # Keep window size
        if len(self._stability_evidence) > self.stability_window:
            self._stability_evidence = self._stability_evidence[-self.stability_window:]

        return self.get_evidence_level()

    def get_evidence_level(self) -> float:
        """Get current evidence level."""
        if not self._stability_evidence:
            return 0.0

        return float(np.mean(self._stability_evidence))

    def can_regenerate(self) -> bool:
        """Check if regeneration is warranted."""
        evidence = self.get_evidence_level()
        return (
            evidence >= self.evidence_threshold and
            self._current_capacity < self.max_capacity
        )

    def regenerate(self) -> RegenerationResult:
        """
        Perform regeneration cycle.

        Returns:
            RegenerationResult with details.
        """
        evidence = self.get_evidence_level()
        stability_confirmed = evidence >= self.evidence_threshold

        if not stability_confirmed:
            return RegenerationResult(
                memories_restored=0,
                capacity_expanded=0.0,
                evidence_threshold_met=False,
                stability_confirmed=False,
            )

        # Expand capacity
        old_capacity = self._current_capacity
        self._current_capacity = min(
            self.max_capacity,
            self._current_capacity + self.expansion_rate,
        )
        expansion = self._current_capacity - old_capacity

        result = RegenerationResult(
            memories_restored=0,  # Actual restoration handled by memory store
            capacity_expanded=expansion,
            evidence_threshold_met=True,
            stability_confirmed=True,
        )

        self._regeneration_history.append(result)

        return result

    def reset_evidence(self) -> None:
        """Reset evidence accumulation (e.g., after destabilization)."""
        self._stability_evidence.clear()

    def reduce_capacity(self, factor: float = 0.5) -> None:
        """Reduce capacity (e.g., during crisis)."""
        self._current_capacity *= factor

    @property
    def current_capacity(self) -> float:
        """Get current capacity."""
        return self._current_capacity

    def get_regeneration_history(self) -> list[RegenerationResult]:
        """Get regeneration history."""
        return self._regeneration_history.copy()


class LatticeMemoryGraph:
    """
    Lattice Memory Graph - Discrete State Graph with Divergence Constraints.

    Implements a graph structure for organizing memories, identities, and
    relationships with edges constrained by Jensen-Shannon divergence and
    scored by mutual information.

    The lattice provides:
    - Identity nodes (self-states, alters)
    - Memory nodes (experiences, events)
    - Relationship nodes (people, connections)
    - Emotion nodes (emotional states)
    - Edges constrained by divergence (similar states connect)
    - Edge weights based on mutual information

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        max_divergence: float = 0.5,
        min_mutual_information: float = 0.1,
    ) -> None:
        """
        Initialize the lattice graph.

        Args:
            max_divergence: Maximum JS divergence for edge creation.
            min_mutual_information: Minimum MI for edge creation.
        """
        self.max_divergence = max_divergence
        self.min_mutual_information = min_mutual_information

        # Graph storage
        self._nodes: dict[str, LatticeNode] = {}
        self._edges: dict[str, LatticeEdge] = {}

        # Indexes
        self._nodes_by_type: dict[str, list[str]] = {}
        self._edges_by_source: dict[str, list[str]] = {}
        self._edges_by_target: dict[str, list[str]] = {}

    def add_node(
        self,
        node_type: str,
        content: str,
        entropy: float = 0.5,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> LatticeNode:
        """
        Add a node to the lattice.

        Args:
            node_type: Type of node (identity, memory, relationship, emotion).
            content: Node content.
            entropy: Entropy at creation.
            importance: Importance score.
            metadata: Additional metadata.

        Returns:
            The created node.
        """
        node_id = str(uuid.uuid4())

        node = LatticeNode(
            id=node_id,
            node_type=node_type,
            content=content,
            entropy_at_creation=entropy,
            importance=importance,
            metadata=metadata or {},
        )

        self._nodes[node_id] = node

        # Update type index
        if node_type not in self._nodes_by_type:
            self._nodes_by_type[node_type] = []
        self._nodes_by_type[node_type].append(node_id)

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        source_distribution: NDArray[np.floating],
        target_distribution: NDArray[np.floating],
        metadata: dict[str, Any] | None = None,
    ) -> LatticeEdge | None:
        """
        Add an edge between nodes if divergence constraint is met.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of edge.
            source_distribution: Probability distribution for source.
            target_distribution: Probability distribution for target.
            metadata: Additional metadata.

        Returns:
            The created edge, or None if constraint not met.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        # Calculate JS divergence
        divergence = calculate_jensen_shannon_divergence(
            source_distribution,
            target_distribution,
        )

        # Check divergence constraint
        if divergence > self.max_divergence:
            return None

        # Calculate mutual information
        # Create joint distribution (simplified)
        joint = np.outer(source_distribution, target_distribution)
        mi_metrics = calculate_mutual_information(joint)

        # Check MI constraint
        if mi_metrics.mutual_information < self.min_mutual_information:
            return None

        # Create edge
        edge_id = str(uuid.uuid4())
        weight = 1.0 - divergence  # Higher weight for lower divergence

        edge = LatticeEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            divergence=divergence,
            mutual_information=mi_metrics.mutual_information,
            metadata=metadata or {},
        )

        self._edges[edge_id] = edge

        # Update node connections
        self._nodes[source_id].connections.append(target_id)
        self._nodes[target_id].connections.append(source_id)

        # Update indexes
        if source_id not in self._edges_by_source:
            self._edges_by_source[source_id] = []
        self._edges_by_source[source_id].append(edge_id)

        if target_id not in self._edges_by_target:
            self._edges_by_target[target_id] = []
        self._edges_by_target[target_id].append(edge_id)

        return edge

    def get_connected_nodes(
        self,
        node_id: str,
        node_type: str | None = None,
        max_divergence: float | None = None,
    ) -> list[LatticeNode]:
        """
        Get nodes connected to a given node.

        Args:
            node_id: Source node ID.
            node_type: Filter by node type.
            max_divergence: Filter by maximum divergence.

        Returns:
            List of connected nodes.
        """
        if node_id not in self._nodes:
            return []

        node = self._nodes[node_id]
        connected = []

        for connected_id in node.connections:
            connected_node = self._nodes.get(connected_id)
            if not connected_node:
                continue

            # Filter by type
            if node_type and connected_node.node_type != node_type:
                continue

            # Filter by divergence
            if max_divergence:
                edge = self._find_edge(node_id, connected_id)
                if edge and edge.divergence > max_divergence:
                    continue

            connected.append(connected_node)

        return connected

    def _find_edge(
        self,
        source_id: str,
        target_id: str,
    ) -> LatticeEdge | None:
        """Find edge between two nodes."""
        for edge_id in self._edges_by_source.get(source_id, []):
            edge = self._edges[edge_id]
            if edge.target_id == target_id:
                return edge

        for edge_id in self._edges_by_target.get(source_id, []):
            edge = self._edges[edge_id]
            if edge.source_id == target_id:
                return edge

        return None

    def get_nodes_by_type(self, node_type: str) -> list[LatticeNode]:
        """Get all nodes of a specific type."""
        node_ids = self._nodes_by_type.get(node_type, [])
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def get_node(self, node_id: str) -> LatticeNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> LatticeEdge | None:
        """Get an edge by ID."""
        return self._edges.get(edge_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its edges."""
        if node_id not in self._nodes:
            return False

        node = self._nodes[node_id]

        # Remove edges
        edges_to_remove = (
            self._edges_by_source.get(node_id, []) +
            self._edges_by_target.get(node_id, [])
        )

        for edge_id in edges_to_remove:
            if edge_id in self._edges:
                del self._edges[edge_id]

        # Remove from indexes
        if node_id in self._edges_by_source:
            del self._edges_by_source[node_id]
        if node_id in self._edges_by_target:
            del self._edges_by_target[node_id]

        # Remove from type index
        if node.node_type in self._nodes_by_type:
            self._nodes_by_type[node.node_type] = [
                nid for nid in self._nodes_by_type[node.node_type]
                if nid != node_id
            ]

        # Remove node
        del self._nodes[node_id]

        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_type": {
                k: len(v) for k, v in self._nodes_by_type.items()
            },
            "avg_connections": (
                np.mean([len(n.connections) for n in self._nodes.values()])
                if self._nodes else 0
            ),
            "avg_divergence": (
                np.mean([e.divergence for e in self._edges.values()])
                if self._edges else 0
            ),
            "avg_mutual_information": (
                np.mean([e.mutual_information for e in self._edges.values()])
                if self._edges else 0
            ),
        }

    def export_graph(self) -> dict[str, Any]:
        """Export graph for serialization."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "node_type": n.node_type,
                    "content": n.content,
                    "entropy_at_creation": n.entropy_at_creation,
                    "importance": n.importance,
                    "connections": n.connections,
                    "metadata": n.metadata,
                }
                for n in self._nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type,
                    "weight": e.weight,
                    "divergence": e.divergence,
                    "mutual_information": e.mutual_information,
                    "metadata": e.metadata,
                }
                for e in self._edges.values()
            ],
        }
