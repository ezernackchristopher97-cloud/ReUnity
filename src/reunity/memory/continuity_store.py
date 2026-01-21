"""
ReUnity Continuity Memory Store

This module implements the Recursive Identity Memory Engine (RIME) and
supporting memory infrastructure for maintaining continuity across fragmented
identity states. It provides journaling, timeline threading, semantic retrieval,
and consent-scoped access controls.

The memory store serves as an external "memory mirror," offering emotional
pattern recognition, past-self reminders, and relationship context during
moments of split, distress, or derealization.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class ConsentScope(Enum):
    """Consent scopes for memory access."""

    PRIVATE = "private"  # Only user can access
    SELF_ONLY = "self_only"  # User and their alters
    THERAPIST = "therapist"  # Shared with designated therapist
    CAREGIVER = "caregiver"  # Shared with designated caregiver
    EMERGENCY = "emergency"  # Accessible in crisis situations
    RESEARCH = "research"  # Anonymized for research (with consent)


class MemoryType(Enum):
    """Types of memories stored in the system."""

    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # General knowledge and patterns
    EMOTIONAL = "emotional"  # Emotional states and responses
    RELATIONAL = "relational"  # Relationship-related memories
    GROUNDING = "grounding"  # Safe/grounding memories
    IDENTITY = "identity"  # Identity-related memories
    JOURNAL = "journal"  # Journal entries


class IdentityState(Enum):
    """Identity states for DID/plural support."""

    PRIMARY = "primary"
    ALTER = "alter"
    BLENDED = "blended"
    UNKNOWN = "unknown"


@dataclass
class MemoryEntry:
    """A single memory entry in the store."""

    id: str
    content: str
    memory_type: MemoryType
    identity_state: str  # Which identity created this
    timestamp: float
    tags: list[str]
    entropy_at_creation: float
    consent_scope: ConsentScope
    emotional_valence: float  # -1 to 1
    importance: float  # 0 to 1
    retrieval_count: int = 0
    last_retrieved: float | None = None
    linked_memories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "identity_state": self.identity_state,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "entropy_at_creation": self.entropy_at_creation,
            "consent_scope": self.consent_scope.value,
            "emotional_valence": self.emotional_valence,
            "importance": self.importance,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved,
            "linked_memories": self.linked_memories,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            identity_state=data["identity_state"],
            timestamp=data["timestamp"],
            tags=data["tags"],
            entropy_at_creation=data["entropy_at_creation"],
            consent_scope=ConsentScope(data["consent_scope"]),
            emotional_valence=data["emotional_valence"],
            importance=data["importance"],
            retrieval_count=data.get("retrieval_count", 0),
            last_retrieved=data.get("last_retrieved"),
            linked_memories=data.get("linked_memories", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class JournalEntry:
    """A journal entry with enhanced metadata."""

    id: str
    title: str
    content: str
    identity_state: str
    timestamp: float
    mood: str
    energy_level: float  # 0 to 1
    entropy_level: float
    tags: list[str]
    consent_scope: ConsentScope
    linked_memories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEvent:
    """An event in the identity timeline."""

    id: str
    event_type: str
    description: str
    timestamp: float
    identity_state: str
    entropy_level: float
    related_memories: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""

    memories: list[MemoryEntry]
    query: str
    retrieval_method: str
    total_found: int
    filtered_by_consent: int
    relevance_scores: list[float]
    source_identity: str


class RecursiveIdentityMemoryEngine:
    """
    RIME - Recursive Identity Memory Engine.

    The core component for grounding fragmentation in DID, PTSD, and other
    conditions. Provides external memory support during dissociation and
    emotional amnesia, maintaining continuous identity threads across
    dissociative episodes.

    Formula: RIME(t) = α · M_episodic(t) + β · M_semantic(t) + γ · C_context(t)

    Where:
    - M_episodic = episodic memory activation
    - M_semantic = semantic memory patterns
    - C_context = current contextual factors
    - α, β, γ = dynamically adjusted weights

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        max_memories: int = 10000,
        grounding_priority: float = 0.8,
    ) -> None:
        """
        Initialize the memory engine.

        Args:
            alpha: Weight for episodic memory activation.
            beta: Weight for semantic memory patterns.
            gamma: Weight for contextual factors.
            max_memories: Maximum memories to store.
            grounding_priority: Priority boost for grounding memories.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_memories = max_memories
        self.grounding_priority = grounding_priority

        # Memory storage by identity
        self._memory_threads: dict[str, list[MemoryEntry]] = {}

        # Emotional state tracking per identity
        self._emotional_states: dict[str, float] = {}

        # Relationship threads
        self._relationship_threads: dict[str, list[MemoryEntry]] = {}

        # Detected protective patterns
        self._protective_patterns: list[dict[str, Any]] = []

        # Journal entries
        self._journals: list[JournalEntry] = []

        # Timeline events
        self._timeline: list[TimelineEvent] = []

        # Consent configurations
        self._consent_config: dict[str, set[ConsentScope]] = {}

        # Retrieval hooks
        self._retrieval_hooks: list[Callable] = []

    def add_memory(
        self,
        identity: str,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        tags: list[str] | None = None,
        entropy: float = 0.5,
        consent_scope: ConsentScope = ConsentScope.PRIVATE,
        emotional_valence: float = 0.0,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """
        Add a tagged memory for a specific identity state.

        Args:
            identity: Identity state identifier.
            content: Memory content.
            memory_type: Type of memory.
            tags: Optional tags for categorization.
            entropy: Entropy level at time of creation.
            consent_scope: Access consent scope.
            emotional_valence: Emotional valence (-1 to 1).
            importance: Importance score (0 to 1).
            metadata: Additional metadata.

        Returns:
            The created MemoryEntry.
        """
        if identity not in self._memory_threads:
            self._memory_threads[identity] = []

        memory_id = str(uuid.uuid4())
        timestamp = time.time()

        memory = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            identity_state=identity,
            timestamp=timestamp,
            tags=tags or [],
            entropy_at_creation=entropy,
            consent_scope=consent_scope,
            emotional_valence=emotional_valence,
            importance=importance,
            metadata=metadata or {},
        )

        self._memory_threads[identity].append(memory)
        self._update_emotional_state(identity, memory)

        # Add to timeline
        self._add_timeline_event(
            "memory_added",
            f"Memory added: {content[:50]}...",
            identity,
            entropy,
            [memory_id],
        )

        # Enforce memory limit
        self._enforce_memory_limit()

        return memory

    def retrieve_grounding(
        self,
        current_identity: str,
        query: str,
        crisis_level: float = 0.0,
        max_results: int = 5,
        accessor_scope: ConsentScope = ConsentScope.SELF_ONLY,
    ) -> RetrievalResult:
        """
        Retrieve grounding memories during fragmentation or crisis.

        Used for DID alter switching, PTSD dissociation, BPD splitting.
        Prioritizes safe, grounding memories during high crisis levels.

        Args:
            current_identity: Current identity state.
            query: Search query.
            crisis_level: Current crisis level (0-1).
            max_results: Maximum memories to return.
            accessor_scope: Consent scope of the accessor.

        Returns:
            RetrievalResult with relevant memories.
        """
        relevant_memories = []
        relevance_scores = []
        filtered_count = 0

        # Search across all identity threads
        for identity, memories in self._memory_threads.items():
            for memory in memories:
                # Check consent
                if not self._check_consent(memory.consent_scope, accessor_scope):
                    filtered_count += 1
                    continue

                # Calculate relevance
                relevance = self._calculate_relevance(memory, query, current_identity)

                # Boost grounding memories during crisis
                if crisis_level > 0.7 and memory.memory_type == MemoryType.GROUNDING:
                    relevance *= (1 + self.grounding_priority)

                # Boost safe-tagged memories during crisis
                if crisis_level > 0.7 and "safe" in memory.tags:
                    relevance *= 1.5

                if relevance > 0:
                    memory.source_identity = identity
                    relevant_memories.append((memory, relevance))

        # Sort by relevance
        relevant_memories.sort(key=lambda x: x[1], reverse=True)

        # Filter during high crisis - prioritize safe memories
        if crisis_level > 0.7:
            safe_memories = [
                (m, r) for m, r in relevant_memories
                if "safe" in m.tags or "grounding" in m.tags or m.memory_type == MemoryType.GROUNDING
            ]
            if safe_memories:
                relevant_memories = safe_memories

        # Take top results
        top_memories = relevant_memories[:max_results]

        # Update retrieval counts
        result_memories = []
        result_scores = []
        for memory, score in top_memories:
            memory.retrieval_count += 1
            memory.last_retrieved = time.time()
            result_memories.append(memory)
            result_scores.append(score)

        # Execute retrieval hooks
        for hook in self._retrieval_hooks:
            try:
                hook(query, result_memories, crisis_level)
            except Exception:
                pass

        return RetrievalResult(
            memories=result_memories,
            query=query,
            retrieval_method="semantic_grounding",
            total_found=len(relevant_memories),
            filtered_by_consent=filtered_count,
            relevance_scores=result_scores,
            source_identity=current_identity,
        )

    def _calculate_relevance(
        self,
        memory: MemoryEntry,
        query: str,
        current_identity: str,
    ) -> float:
        """
        Calculate relevance score for a memory.

        Uses the RIME formula:
        RIME(t) = α · M_episodic(t) + β · M_semantic(t) + γ · C_context(t)
        """
        # Episodic component - recency and retrieval frequency
        time_decay = np.exp(-(time.time() - memory.timestamp) / (86400 * 30))  # 30-day decay
        episodic_score = time_decay * (1 + 0.1 * memory.retrieval_count)

        # Semantic component - content matching
        semantic_score = self._semantic_match(query, memory.content)

        # Context component - identity match and importance
        identity_match = 1.0 if memory.identity_state == current_identity else 0.5
        context_score = identity_match * memory.importance

        # RIME formula
        relevance = (
            self.alpha * episodic_score +
            self.beta * semantic_score +
            self.gamma * context_score
        )

        return float(relevance)

    def _semantic_match(self, query: str, content: str) -> float:
        """
        Calculate semantic similarity between query and content.

        Simplified word overlap implementation.
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words or not content_words:
            return 0.0

        intersection = len(query_words.intersection(content_words))
        union = len(query_words.union(content_words))

        return intersection / union if union > 0 else 0.0

    def _check_consent(
        self,
        memory_scope: ConsentScope,
        accessor_scope: ConsentScope,
    ) -> bool:
        """Check if accessor has consent to access memory."""
        # Define consent hierarchy
        consent_levels = {
            ConsentScope.PRIVATE: 0,
            ConsentScope.SELF_ONLY: 1,
            ConsentScope.THERAPIST: 2,
            ConsentScope.CAREGIVER: 2,
            ConsentScope.EMERGENCY: 3,
            ConsentScope.RESEARCH: 4,
        }

        # Private memories only accessible by self
        if memory_scope == ConsentScope.PRIVATE:
            return accessor_scope in [ConsentScope.PRIVATE, ConsentScope.SELF_ONLY]

        # Self-only accessible by self and emergency
        if memory_scope == ConsentScope.SELF_ONLY:
            return accessor_scope in [
                ConsentScope.PRIVATE,
                ConsentScope.SELF_ONLY,
                ConsentScope.EMERGENCY,
            ]

        # Therapist/caregiver scoped memories
        if memory_scope in [ConsentScope.THERAPIST, ConsentScope.CAREGIVER]:
            return accessor_scope == memory_scope or accessor_scope == ConsentScope.EMERGENCY

        # Emergency accessible in crisis
        if memory_scope == ConsentScope.EMERGENCY:
            return True

        return False

    def _update_emotional_state(
        self,
        identity: str,
        memory: MemoryEntry,
    ) -> None:
        """Update emotional state tracking for identity."""
        current = self._emotional_states.get(identity, 0.0)

        # Weighted average with new memory's valence
        weight = 0.3
        self._emotional_states[identity] = (
            (1 - weight) * current + weight * memory.emotional_valence
        )

    def _add_timeline_event(
        self,
        event_type: str,
        description: str,
        identity: str,
        entropy: float,
        related_memories: list[str],
    ) -> None:
        """Add an event to the timeline."""
        event = TimelineEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            description=description,
            timestamp=time.time(),
            identity_state=identity,
            entropy_level=entropy,
            related_memories=related_memories,
        )
        self._timeline.append(event)

    def _enforce_memory_limit(self) -> None:
        """Enforce maximum memory limit using importance-based pruning."""
        total_memories = sum(
            len(memories) for memories in self._memory_threads.values()
        )

        if total_memories <= self.max_memories:
            return

        # Collect all memories with their identities
        all_memories = []
        for identity, memories in self._memory_threads.items():
            for memory in memories:
                all_memories.append((identity, memory))

        # Sort by importance and recency (keep most important and recent)
        all_memories.sort(
            key=lambda x: (x[1].importance, x[1].timestamp),
            reverse=True,
        )

        # Keep top memories
        keep_memories = all_memories[:self.max_memories]
        keep_ids = {m.id for _, m in keep_memories}

        # Rebuild memory threads
        for identity in self._memory_threads:
            self._memory_threads[identity] = [
                m for m in self._memory_threads[identity]
                if m.id in keep_ids
            ]

    def add_journal_entry(
        self,
        title: str,
        content: str,
        identity: str,
        mood: str,
        energy_level: float,
        entropy_level: float,
        tags: list[str] | None = None,
        consent_scope: ConsentScope = ConsentScope.PRIVATE,
    ) -> JournalEntry:
        """
        Add a journal entry.

        Args:
            title: Entry title.
            content: Entry content.
            identity: Identity state.
            mood: Current mood description.
            energy_level: Energy level (0-1).
            entropy_level: Current entropy level.
            tags: Optional tags.
            consent_scope: Access consent scope.

        Returns:
            The created JournalEntry.
        """
        entry = JournalEntry(
            id=str(uuid.uuid4()),
            title=title,
            content=content,
            identity_state=identity,
            timestamp=time.time(),
            mood=mood,
            energy_level=energy_level,
            entropy_level=entropy_level,
            tags=tags or [],
            consent_scope=consent_scope,
        )

        self._journals.append(entry)

        # Also create a memory entry for the journal
        self.add_memory(
            identity=identity,
            content=f"Journal: {title} - {content[:200]}",
            memory_type=MemoryType.JOURNAL,
            tags=["journal"] + (tags or []),
            entropy=entropy_level,
            consent_scope=consent_scope,
            importance=0.7,
        )

        return entry

    def get_journals(
        self,
        identity: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        accessor_scope: ConsentScope = ConsentScope.SELF_ONLY,
    ) -> list[JournalEntry]:
        """
        Retrieve journal entries with filtering.

        Args:
            identity: Filter by identity state.
            start_time: Start of time range.
            end_time: End of time range.
            accessor_scope: Consent scope of accessor.

        Returns:
            List of matching journal entries.
        """
        results = []

        for entry in self._journals:
            # Check consent
            if not self._check_consent(entry.consent_scope, accessor_scope):
                continue

            # Filter by identity
            if identity and entry.identity_state != identity:
                continue

            # Filter by time range
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            results.append(entry)

        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def get_timeline(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        identity: str | None = None,
    ) -> list[TimelineEvent]:
        """
        Get timeline events with filtering.

        Args:
            start_time: Start of time range.
            end_time: End of time range.
            identity: Filter by identity state.

        Returns:
            List of timeline events.
        """
        results = []

        for event in self._timeline:
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if identity and event.identity_state != identity:
                continue

            results.append(event)

        return sorted(results, key=lambda x: x.timestamp)

    def link_memories(
        self,
        memory_id_1: str,
        memory_id_2: str,
    ) -> bool:
        """
        Create a bidirectional link between two memories.

        Args:
            memory_id_1: First memory ID.
            memory_id_2: Second memory ID.

        Returns:
            True if link was created successfully.
        """
        memory_1 = self._find_memory_by_id(memory_id_1)
        memory_2 = self._find_memory_by_id(memory_id_2)

        if not memory_1 or not memory_2:
            return False

        if memory_id_2 not in memory_1.linked_memories:
            memory_1.linked_memories.append(memory_id_2)

        if memory_id_1 not in memory_2.linked_memories:
            memory_2.linked_memories.append(memory_id_1)

        return True

    def _find_memory_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Find a memory by its ID."""
        for memories in self._memory_threads.values():
            for memory in memories:
                if memory.id == memory_id:
                    return memory
        return None

    def get_identity_memories(
        self,
        identity: str,
        accessor_scope: ConsentScope = ConsentScope.SELF_ONLY,
    ) -> list[MemoryEntry]:
        """
        Get all memories for a specific identity.

        Args:
            identity: Identity state.
            accessor_scope: Consent scope of accessor.

        Returns:
            List of memories for the identity.
        """
        memories = self._memory_threads.get(identity, [])

        return [
            m for m in memories
            if self._check_consent(m.consent_scope, accessor_scope)
        ]

    def get_emotional_state(self, identity: str) -> float:
        """Get current emotional state for an identity."""
        return self._emotional_states.get(identity, 0.0)

    def set_consent_scope(
        self,
        memory_id: str,
        new_scope: ConsentScope,
    ) -> bool:
        """
        Update consent scope for a memory.

        Args:
            memory_id: Memory ID to update.
            new_scope: New consent scope.

        Returns:
            True if update was successful.
        """
        memory = self._find_memory_by_id(memory_id)
        if memory:
            memory.consent_scope = new_scope
            return True
        return False

    def register_retrieval_hook(
        self,
        hook: Callable[[str, list[MemoryEntry], float], None],
    ) -> None:
        """
        Register a hook to be called on memory retrieval.

        Args:
            hook: Callback function(query, memories, crisis_level).
        """
        self._retrieval_hooks.append(hook)

    def export_memories(
        self,
        identity: str | None = None,
        accessor_scope: ConsentScope = ConsentScope.SELF_ONLY,
    ) -> dict[str, Any]:
        """
        Export memories for backup or transfer.

        Args:
            identity: Optional identity filter.
            accessor_scope: Consent scope of accessor.

        Returns:
            Dictionary of exportable memory data.
        """
        export_data = {
            "version": "1.0",
            "exported_at": time.time(),
            "memories": [],
            "journals": [],
            "timeline": [],
        }

        # Export memories
        for id_state, memories in self._memory_threads.items():
            if identity and id_state != identity:
                continue

            for memory in memories:
                if self._check_consent(memory.consent_scope, accessor_scope):
                    export_data["memories"].append(memory.to_dict())

        # Export journals
        for journal in self._journals:
            if identity and journal.identity_state != identity:
                continue
            if self._check_consent(journal.consent_scope, accessor_scope):
                export_data["journals"].append({
                    "id": journal.id,
                    "title": journal.title,
                    "content": journal.content,
                    "identity_state": journal.identity_state,
                    "timestamp": journal.timestamp,
                    "mood": journal.mood,
                    "energy_level": journal.energy_level,
                    "entropy_level": journal.entropy_level,
                    "tags": journal.tags,
                    "consent_scope": journal.consent_scope.value,
                })

        # Export timeline
        for event in self._timeline:
            if identity and event.identity_state != identity:
                continue
            export_data["timeline"].append({
                "id": event.id,
                "event_type": event.event_type,
                "description": event.description,
                "timestamp": event.timestamp,
                "identity_state": event.identity_state,
                "entropy_level": event.entropy_level,
            })

        return export_data

    def import_memories(
        self,
        data: dict[str, Any],
        merge: bool = True,
    ) -> int:
        """
        Import memories from exported data.

        Args:
            data: Exported memory data.
            merge: If True, merge with existing; if False, replace.

        Returns:
            Number of memories imported.
        """
        if not merge:
            self._memory_threads.clear()
            self._journals.clear()
            self._timeline.clear()

        imported = 0

        # Import memories
        for mem_data in data.get("memories", []):
            memory = MemoryEntry.from_dict(mem_data)
            identity = memory.identity_state

            if identity not in self._memory_threads:
                self._memory_threads[identity] = []

            # Check for duplicates
            existing_ids = {m.id for m in self._memory_threads[identity]}
            if memory.id not in existing_ids:
                self._memory_threads[identity].append(memory)
                imported += 1

        return imported

    def get_statistics(self) -> dict[str, Any]:
        """Get memory store statistics."""
        total_memories = sum(
            len(m) for m in self._memory_threads.values()
        )

        return {
            "total_memories": total_memories,
            "identity_count": len(self._memory_threads),
            "journal_count": len(self._journals),
            "timeline_events": len(self._timeline),
            "memories_by_identity": {
                k: len(v) for k, v in self._memory_threads.items()
            },
            "memories_by_type": self._count_by_type(),
        }

    def _count_by_type(self) -> dict[str, int]:
        """Count memories by type."""
        counts: dict[str, int] = {}
        for memories in self._memory_threads.values():
            for memory in memories:
                type_name = memory.memory_type.value
                counts[type_name] = counts.get(type_name, 0) + 1
        return counts
