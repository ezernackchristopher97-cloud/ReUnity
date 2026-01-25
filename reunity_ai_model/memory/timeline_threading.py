"""
ReUnity Timeline Threading Module

This module implements timeline threading for maintaining narrative
continuity across fragmented memory states. It provides mechanisms
for linking related memories, tracking temporal relationships, and
preserving coherent life narratives even during dissociative episodes.

Key Features:
- Temporal memory linking
- Narrative thread maintenance
- Cross-state memory bridging
- Anchor memory identification
- Timeline visualization support

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ThreadType(Enum):
    """Types of timeline threads."""

    CHRONOLOGICAL = "chronological"  # Time-based sequence
    THEMATIC = "thematic"  # Theme-based grouping
    RELATIONAL = "relational"  # Relationship-focused
    EMOTIONAL = "emotional"  # Emotion-linked
    IDENTITY = "identity"  # Identity state-linked
    ANCHOR = "anchor"  # Anchor memory chains


class MemoryValence(Enum):
    """Emotional valence of memories."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    ANCHOR = "anchor"  # Stabilizing memories


@dataclass
class TimelineMemory:
    """A memory node in the timeline."""

    memory_id: str
    content: str
    timestamp: float
    valence: MemoryValence
    emotional_intensity: float  # 0-1
    entropy_at_creation: float
    tags: list[str] = field(default_factory=list)
    linked_memories: list[str] = field(default_factory=list)
    thread_ids: list[str] = field(default_factory=list)
    identity_state: str | None = None  # For DID support
    is_anchor: bool = False
    verification_status: str = "unverified"  # unverified, self_verified, externally_verified
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineThread:
    """A thread connecting related memories."""

    thread_id: str
    thread_type: ThreadType
    name: str
    description: str
    memory_ids: list[str]
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineGap:
    """A detected gap in the timeline."""

    gap_id: str
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    preceding_memory_id: str | None
    following_memory_id: str | None
    gap_type: str  # "temporal", "thematic", "identity_switch"
    notes: str = ""


@dataclass
class NarrativeSegment:
    """A coherent narrative segment."""

    segment_id: str
    title: str
    memory_ids: list[str]
    start_timestamp: float
    end_timestamp: float
    summary: str
    themes: list[str]
    emotional_arc: str  # "ascending", "descending", "stable", "volatile"


class TimelineThreader:
    """
    Timeline threading system for memory continuity.

    This system maintains narrative continuity by linking memories
    across time and emotional states, identifying anchor memories,
    and bridging gaps in the timeline.

    DISCLAIMER: This is not a clinical or treatment tool.
    """

    def __init__(
        self,
        max_gap_hours: float = 24.0,
        similarity_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the timeline threader.

        Args:
            max_gap_hours: Maximum gap before flagging discontinuity.
            similarity_threshold: Threshold for automatic linking.
        """
        self.max_gap_hours = max_gap_hours
        self.max_gap_seconds = max_gap_hours * 3600
        self.similarity_threshold = similarity_threshold

        # Storage
        self._memories: dict[str, TimelineMemory] = {}
        self._threads: dict[str, TimelineThread] = {}
        self._gaps: list[TimelineGap] = []
        self._segments: dict[str, NarrativeSegment] = {}

    def add_memory(
        self,
        content: str,
        valence: MemoryValence = MemoryValence.NEUTRAL,
        emotional_intensity: float = 0.5,
        entropy_at_creation: float = 0.5,
        tags: list[str] | None = None,
        identity_state: str | None = None,
        is_anchor: bool = False,
        timestamp: float | None = None,
    ) -> TimelineMemory:
        """
        Add a new memory to the timeline.

        Args:
            content: Memory content.
            valence: Emotional valence.
            emotional_intensity: Intensity of emotion (0-1).
            entropy_at_creation: Entropy level when memory was created.
            tags: Tags for categorization.
            identity_state: Identity state (for DID support).
            is_anchor: Whether this is an anchor memory.
            timestamp: Optional timestamp (defaults to now).

        Returns:
            The created TimelineMemory.
        """
        memory_id = hashlib.sha256(
            f"{content}{time.time()}{uuid.uuid4()}".encode()
        ).hexdigest()[:16]

        memory = TimelineMemory(
            memory_id=memory_id,
            content=content,
            timestamp=timestamp or time.time(),
            valence=valence,
            emotional_intensity=emotional_intensity,
            entropy_at_creation=entropy_at_creation,
            tags=tags or [],
            identity_state=identity_state,
            is_anchor=is_anchor,
        )

        self._memories[memory_id] = memory

        # Auto-link to nearby memories
        self._auto_link_memory(memory)

        # Check for gaps
        self._detect_gaps()

        return memory

    def _auto_link_memory(self, memory: TimelineMemory) -> None:
        """Automatically link memory to related memories."""
        for other_id, other in self._memories.items():
            if other_id == memory.memory_id:
                continue

            # Link temporally close memories
            time_diff = abs(memory.timestamp - other.timestamp)
            if time_diff < 3600:  # Within 1 hour
                if other_id not in memory.linked_memories:
                    memory.linked_memories.append(other_id)
                if memory.memory_id not in other.linked_memories:
                    other.linked_memories.append(memory.memory_id)

            # Link memories with same tags
            common_tags = set(memory.tags) & set(other.tags)
            if common_tags:
                if other_id not in memory.linked_memories:
                    memory.linked_memories.append(other_id)

            # Link memories with same identity state
            if memory.identity_state and memory.identity_state == other.identity_state:
                if other_id not in memory.linked_memories:
                    memory.linked_memories.append(other_id)

    def _detect_gaps(self) -> None:
        """Detect gaps in the timeline."""
        if len(self._memories) < 2:
            return

        # Sort memories by timestamp
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.timestamp,
        )

        self._gaps = []

        for i in range(len(sorted_memories) - 1):
            current = sorted_memories[i]
            next_mem = sorted_memories[i + 1]

            gap_duration = next_mem.timestamp - current.timestamp

            if gap_duration > self.max_gap_seconds:
                gap = TimelineGap(
                    gap_id=f"gap_{i}_{int(current.timestamp)}",
                    start_timestamp=current.timestamp,
                    end_timestamp=next_mem.timestamp,
                    duration_seconds=gap_duration,
                    preceding_memory_id=current.memory_id,
                    following_memory_id=next_mem.memory_id,
                    gap_type="temporal",
                )
                self._gaps.append(gap)

            # Check for identity switches
            if (current.identity_state and next_mem.identity_state and
                current.identity_state != next_mem.identity_state):
                gap = TimelineGap(
                    gap_id=f"switch_{i}_{int(current.timestamp)}",
                    start_timestamp=current.timestamp,
                    end_timestamp=next_mem.timestamp,
                    duration_seconds=gap_duration,
                    preceding_memory_id=current.memory_id,
                    following_memory_id=next_mem.memory_id,
                    gap_type="identity_switch",
                    notes=f"Switch from {current.identity_state} to {next_mem.identity_state}",
                )
                self._gaps.append(gap)

    def create_thread(
        self,
        name: str,
        thread_type: ThreadType,
        memory_ids: list[str],
        description: str = "",
    ) -> TimelineThread:
        """
        Create a new timeline thread.

        Args:
            name: Thread name.
            thread_type: Type of thread.
            memory_ids: IDs of memories to include.
            description: Thread description.

        Returns:
            The created TimelineThread.
        """
        thread_id = f"thread_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        thread = TimelineThread(
            thread_id=thread_id,
            thread_type=thread_type,
            name=name,
            description=description,
            memory_ids=memory_ids,
        )

        self._threads[thread_id] = thread

        # Update memories with thread reference
        for memory_id in memory_ids:
            if memory_id in self._memories:
                self._memories[memory_id].thread_ids.append(thread_id)

        return thread

    def add_to_thread(self, thread_id: str, memory_id: str) -> bool:
        """
        Add a memory to an existing thread.

        Args:
            thread_id: ID of the thread.
            memory_id: ID of the memory to add.

        Returns:
            True if successful.
        """
        if thread_id not in self._threads:
            return False

        if memory_id not in self._memories:
            return False

        thread = self._threads[thread_id]
        if memory_id not in thread.memory_ids:
            thread.memory_ids.append(memory_id)
            thread.last_updated = time.time()

        memory = self._memories[memory_id]
        if thread_id not in memory.thread_ids:
            memory.thread_ids.append(thread_id)

        return True

    def get_anchor_memories(self) -> list[TimelineMemory]:
        """Get all anchor memories."""
        return [m for m in self._memories.values() if m.is_anchor]

    def get_memories_by_identity(
        self,
        identity_state: str,
    ) -> list[TimelineMemory]:
        """Get memories for a specific identity state."""
        return [
            m for m in self._memories.values()
            if m.identity_state == identity_state
        ]

    def get_memories_by_tag(self, tag: str) -> list[TimelineMemory]:
        """Get memories with a specific tag."""
        return [m for m in self._memories.values() if tag in m.tags]

    def get_memories_in_range(
        self,
        start_timestamp: float,
        end_timestamp: float,
    ) -> list[TimelineMemory]:
        """Get memories within a time range."""
        return [
            m for m in self._memories.values()
            if start_timestamp <= m.timestamp <= end_timestamp
        ]

    def get_timeline_gaps(self) -> list[TimelineGap]:
        """Get all detected timeline gaps."""
        return self._gaps.copy()

    def bridge_gap(
        self,
        gap_id: str,
        bridging_content: str,
        bridging_type: str = "reconstruction",
    ) -> TimelineMemory | None:
        """
        Bridge a timeline gap with new content.

        Args:
            gap_id: ID of the gap to bridge.
            bridging_content: Content to bridge the gap.
            bridging_type: Type of bridging (reconstruction, inference, etc.)

        Returns:
            The bridging memory or None if gap not found.
        """
        gap = next((g for g in self._gaps if g.gap_id == gap_id), None)
        if not gap:
            return None

        # Create bridging memory at midpoint
        midpoint = (gap.start_timestamp + gap.end_timestamp) / 2

        memory = self.add_memory(
            content=bridging_content,
            valence=MemoryValence.NEUTRAL,
            emotional_intensity=0.3,
            entropy_at_creation=0.5,
            tags=["bridging", bridging_type],
            timestamp=midpoint,
        )

        memory.metadata["bridging_type"] = bridging_type
        memory.metadata["original_gap_id"] = gap_id
        memory.verification_status = "reconstructed"

        # Link to adjacent memories
        if gap.preceding_memory_id:
            memory.linked_memories.append(gap.preceding_memory_id)
        if gap.following_memory_id:
            memory.linked_memories.append(gap.following_memory_id)

        # Re-detect gaps
        self._detect_gaps()

        return memory

    def create_narrative_segment(
        self,
        title: str,
        memory_ids: list[str],
        summary: str,
        themes: list[str],
    ) -> NarrativeSegment | None:
        """
        Create a narrative segment from memories.

        Args:
            title: Segment title.
            memory_ids: IDs of memories in segment.
            summary: Summary of the segment.
            themes: Themes present in segment.

        Returns:
            The created NarrativeSegment or None if no valid memories.
        """
        valid_memories = [
            self._memories[mid] for mid in memory_ids
            if mid in self._memories
        ]

        if not valid_memories:
            return None

        # Calculate time range
        timestamps = [m.timestamp for m in valid_memories]
        start_timestamp = min(timestamps)
        end_timestamp = max(timestamps)

        # Analyze emotional arc
        emotional_arc = self._analyze_emotional_arc(valid_memories)

        segment = NarrativeSegment(
            segment_id=f"seg_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            title=title,
            memory_ids=memory_ids,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            summary=summary,
            themes=themes,
            emotional_arc=emotional_arc,
        )

        self._segments[segment.segment_id] = segment
        return segment

    def _analyze_emotional_arc(
        self,
        memories: list[TimelineMemory],
    ) -> str:
        """Analyze the emotional arc of a set of memories."""
        if len(memories) < 2:
            return "stable"

        # Sort by timestamp
        sorted_mems = sorted(memories, key=lambda m: m.timestamp)

        # Calculate valence scores
        valence_scores = []
        for m in sorted_mems:
            if m.valence == MemoryValence.POSITIVE:
                valence_scores.append(m.emotional_intensity)
            elif m.valence == MemoryValence.NEGATIVE:
                valence_scores.append(-m.emotional_intensity)
            else:
                valence_scores.append(0)

        # Analyze trend
        if len(valence_scores) >= 2:
            trend = valence_scores[-1] - valence_scores[0]
            variance = np.std(valence_scores)

            if variance > 0.5:
                return "volatile"
            elif trend > 0.3:
                return "ascending"
            elif trend < -0.3:
                return "descending"

        return "stable"

    def get_memory(self, memory_id: str) -> TimelineMemory | None:
        """Get a memory by ID."""
        return self._memories.get(memory_id)

    def get_thread(self, thread_id: str) -> TimelineThread | None:
        """Get a thread by ID."""
        return self._threads.get(thread_id)

    def get_all_threads(self) -> list[TimelineThread]:
        """Get all threads."""
        return list(self._threads.values())

    def get_chronological_timeline(
        self,
        limit: int = 100,
    ) -> list[TimelineMemory]:
        """Get memories in chronological order."""
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.timestamp,
        )
        return sorted_memories[-limit:]

    def mark_as_anchor(self, memory_id: str) -> bool:
        """Mark a memory as an anchor memory."""
        if memory_id not in self._memories:
            return False

        self._memories[memory_id].is_anchor = True
        self._memories[memory_id].valence = MemoryValence.ANCHOR
        return True

    def verify_memory(
        self,
        memory_id: str,
        verification_type: str = "self_verified",
    ) -> bool:
        """
        Update verification status of a memory.

        Args:
            memory_id: ID of the memory.
            verification_type: Type of verification.

        Returns:
            True if successful.
        """
        if memory_id not in self._memories:
            return False

        self._memories[memory_id].verification_status = verification_type
        return True

    def export_timeline(self) -> dict[str, Any]:
        """
        Export the complete timeline.

        Returns:
            Dictionary containing all timeline data.
        """
        return {
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "valence": m.valence.value,
                    "emotional_intensity": m.emotional_intensity,
                    "tags": m.tags,
                    "is_anchor": m.is_anchor,
                    "identity_state": m.identity_state,
                    "verification_status": m.verification_status,
                }
                for m in sorted(self._memories.values(), key=lambda x: x.timestamp)
            ],
            "threads": [
                {
                    "thread_id": t.thread_id,
                    "name": t.name,
                    "type": t.thread_type.value,
                    "memory_count": len(t.memory_ids),
                }
                for t in self._threads.values()
            ],
            "gaps": [
                {
                    "gap_id": g.gap_id,
                    "duration_hours": g.duration_seconds / 3600,
                    "gap_type": g.gap_type,
                }
                for g in self._gaps
            ],
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "title": s.title,
                    "emotional_arc": s.emotional_arc,
                }
                for s in self._segments.values()
            ],
            "statistics": {
                "total_memories": len(self._memories),
                "anchor_memories": len(self.get_anchor_memories()),
                "total_threads": len(self._threads),
                "total_gaps": len(self._gaps),
            },
        }
