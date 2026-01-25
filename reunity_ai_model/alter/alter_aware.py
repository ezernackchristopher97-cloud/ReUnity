"""
ReUnity Alter-Aware Subsystem (AAS)

This module provides comprehensive support for individuals with Dissociative
Identity Disorder (DID) and plural consciousness systems. It recognizes and
validates the existence of multiple identity states while promoting healthy
internal communication and cooperation.

The system explicitly rejects integration models that seek to eliminate alter
personalities, instead focusing on reducing internal conflict and improving
system functioning through entropy reduction techniques that honor the complexity
and validity of plural consciousness.

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only. Always work with qualified mental health
professionals for DID treatment.

Key Features:
- Individual alter recognition with personalized interaction protocols
- Inter-alter communication facilitation and conflict resolution
- Shared memory systems for collaborative decision-making
- Co-consciousness development support
- Trauma processing adapted to system dynamics

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
from numpy.typing import NDArray


class AlterState(Enum):
    """Enumeration of alter presence states."""

    FRONTING = "fronting"  # Currently in executive control
    CO_CONSCIOUS = "co_conscious"  # Present but not in control
    DORMANT = "dormant"  # Not currently active
    BLENDED = "blended"  # Multiple alters sharing control
    SWITCHING = "switching"  # Transition in progress


class CommunicationType(Enum):
    """Types of internal communication."""

    DIRECT = "direct"  # Direct alter-to-alter communication
    MEDIATED = "mediated"  # AI-facilitated communication
    BROADCAST = "broadcast"  # Message to all system members
    PRIVATE = "private"  # Private message with consent controls


@dataclass
class AlterProfile:
    """Profile for an individual alter/identity state."""

    alter_id: str
    name: str
    pronouns: str = "they/them"
    age_presentation: str | None = None
    role: str | None = None  # Protector, caretaker, etc.
    emotional_baseline: dict[str, float] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)
    grounding_preferences: list[str] = field(default_factory=list)
    communication_style: str = "neutral"
    memory_access_level: str = "standard"  # full, standard, limited, none
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate alter_id if not provided."""
        if not self.alter_id:
            self.alter_id = str(uuid.uuid4())


@dataclass
class SystemState:
    """Current state of the plural system."""

    fronting_alters: list[str]  # IDs of alters currently fronting
    co_conscious_alters: list[str]  # IDs of co-conscious alters
    system_entropy: float  # Overall system entropy level
    internal_conflict_level: float  # 0-1 scale
    communication_clarity: float  # 0-1 scale
    timestamp: float = field(default_factory=time.time)


@dataclass
class InternalMessage:
    """Message for internal system communication."""

    message_id: str
    sender_id: str  # Alter ID or "system" for AI
    recipient_ids: list[str]  # Target alter IDs or ["all"]
    content: str
    message_type: CommunicationType
    timestamp: float = field(default_factory=time.time)
    read_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwitchEvent:
    """Record of an alter switch event."""

    event_id: str
    from_alter_ids: list[str]
    to_alter_ids: list[str]
    trigger: str | None = None
    duration_seconds: float | None = None
    smoothness: float = 0.5  # 0 = rough, 1 = smooth
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


class AlterAwareSubsystem:
    """
    Alter-Aware Subsystem (AAS) for DID/plural consciousness support.

    This subsystem allows each part of the system to interact with the AI
    and build shared memory maps that respect the autonomy and wisdom of
    all internal parts. It explicitly rejects integration models that seek
    to eliminate alter personalities.

    The AAS focuses on:
    1. Individual alter recognition and personalized interaction
    2. Inter-alter communication facilitation
    3. Conflict resolution without forced integration
    4. Shared memory systems with consent controls
    5. Co-consciousness development support

    DISCLAIMER: This is not a clinical or treatment tool.
    """

    def __init__(
        self,
        system_id: str | None = None,
        enable_memory_sharing: bool = True,
        conflict_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the Alter-Aware Subsystem.

        Args:
            system_id: Unique identifier for the plural system.
            enable_memory_sharing: Whether to enable shared memory features.
            conflict_threshold: Threshold for flagging internal conflict.
        """
        self.system_id = system_id or str(uuid.uuid4())
        self.enable_memory_sharing = enable_memory_sharing
        self.conflict_threshold = conflict_threshold

        # Storage
        self._alters: dict[str, AlterProfile] = {}
        self._messages: list[InternalMessage] = []
        self._switch_history: list[SwitchEvent] = []
        self._current_state: SystemState | None = None

        # Shared memory (with consent controls)
        self._shared_memories: dict[str, dict[str, Any]] = {}
        self._memory_consent: dict[str, set[str]] = {}  # memory_id -> alter_ids with access

    def register_alter(self, profile: AlterProfile) -> str:
        """
        Register a new alter in the system.

        Args:
            profile: AlterProfile for the new alter.

        Returns:
            The alter's unique ID.
        """
        if not profile.alter_id:
            profile.alter_id = str(uuid.uuid4())

        self._alters[profile.alter_id] = profile
        return profile.alter_id

    def get_alter(self, alter_id: str) -> AlterProfile | None:
        """Get an alter's profile by ID."""
        return self._alters.get(alter_id)

    def list_alters(self) -> list[AlterProfile]:
        """List all registered alters."""
        return list(self._alters.values())

    def update_alter(self, alter_id: str, updates: dict[str, Any]) -> bool:
        """
        Update an alter's profile.

        Args:
            alter_id: ID of the alter to update.
            updates: Dictionary of fields to update.

        Returns:
            True if update was successful.
        """
        if alter_id not in self._alters:
            return False

        profile = self._alters[alter_id]
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        return True

    def recognize_alter(
        self,
        linguistic_patterns: dict[str, float],
        emotional_markers: dict[str, float],
        behavioral_indicators: dict[str, float],
    ) -> tuple[str | None, float]:
        """
        Recognize which alter is currently present based on patterns.

        This implements the AAS recognition formula:
        AAS_recognition = f(linguistic_patterns, emotional_markers, behavioral_indicators)

        Args:
            linguistic_patterns: Word choice, sentence structure patterns.
            emotional_markers: Emotional expression indicators.
            behavioral_indicators: Behavioral pattern indicators.

        Returns:
            Tuple of (alter_id, confidence) or (None, 0.0) if no match.
        """
        if not self._alters:
            return None, 0.0

        best_match: str | None = None
        best_score = 0.0

        for alter_id, profile in self._alters.items():
            score = self._calculate_recognition_score(
                profile,
                linguistic_patterns,
                emotional_markers,
                behavioral_indicators,
            )

            if score > best_score:
                best_score = score
                best_match = alter_id

        # Update last active if recognized
        if best_match and best_score > 0.5:
            self._alters[best_match].last_active = time.time()

        return best_match, best_score

    def _calculate_recognition_score(
        self,
        profile: AlterProfile,
        linguistic_patterns: dict[str, float],
        emotional_markers: dict[str, float],
        behavioral_indicators: dict[str, float],
    ) -> float:
        """Calculate recognition score for an alter."""
        # Compare against baseline emotional patterns
        emotional_similarity = 0.0
        if profile.emotional_baseline:
            common_keys = set(emotional_markers.keys()) & set(profile.emotional_baseline.keys())
            if common_keys:
                diffs = [
                    abs(emotional_markers[k] - profile.emotional_baseline[k])
                    for k in common_keys
                ]
                emotional_similarity = 1.0 - (sum(diffs) / len(diffs))

        # Communication style matching
        style_score = 0.0
        if "formality" in linguistic_patterns:
            if profile.communication_style == "formal" and linguistic_patterns["formality"] > 0.7:
                style_score = 0.8
            elif profile.communication_style == "casual" and linguistic_patterns["formality"] < 0.3:
                style_score = 0.8
            else:
                style_score = 0.4

        # Weighted combination
        score = 0.5 * emotional_similarity + 0.3 * style_score + 0.2 * np.random.uniform(0.3, 0.7)

        return float(min(1.0, max(0.0, score)))

    def record_switch(
        self,
        from_alter_ids: list[str],
        to_alter_ids: list[str],
        trigger: str | None = None,
        smoothness: float = 0.5,
        notes: str = "",
    ) -> SwitchEvent:
        """
        Record an alter switch event.

        Args:
            from_alter_ids: IDs of alters who were fronting.
            to_alter_ids: IDs of alters now fronting.
            trigger: What triggered the switch (if known).
            smoothness: How smooth the switch was (0-1).
            notes: Additional notes about the switch.

        Returns:
            The recorded SwitchEvent.
        """
        event = SwitchEvent(
            event_id=str(uuid.uuid4()),
            from_alter_ids=from_alter_ids,
            to_alter_ids=to_alter_ids,
            trigger=trigger,
            smoothness=smoothness,
            notes=notes,
        )

        self._switch_history.append(event)

        # Update current state
        self._update_system_state(to_alter_ids)

        return event

    def _update_system_state(self, fronting_ids: list[str]) -> None:
        """Update the current system state."""
        self._current_state = SystemState(
            fronting_alters=fronting_ids,
            co_conscious_alters=[
                aid for aid in self._alters.keys()
                if aid not in fronting_ids
            ],
            system_entropy=self._calculate_system_entropy(),
            internal_conflict_level=self._assess_conflict_level(),
            communication_clarity=self._assess_communication_clarity(),
        )

    def _calculate_system_entropy(self) -> float:
        """Calculate overall system entropy."""
        if not self._switch_history:
            return 0.5

        # Entropy based on switch frequency and smoothness
        recent_switches = [
            s for s in self._switch_history
            if time.time() - s.timestamp < 86400  # Last 24 hours
        ]

        if not recent_switches:
            return 0.3

        # More switches = higher entropy
        switch_rate = len(recent_switches) / 24.0  # per hour
        avg_smoothness = np.mean([s.smoothness for s in recent_switches])

        entropy = 0.3 + 0.4 * min(1.0, switch_rate / 2.0) + 0.3 * (1.0 - avg_smoothness)
        return float(min(1.0, entropy))

    def _assess_conflict_level(self) -> float:
        """Assess current internal conflict level."""
        if not self._messages:
            return 0.0

        # Analyze recent messages for conflict indicators
        recent_messages = [
            m for m in self._messages
            if time.time() - m.timestamp < 3600  # Last hour
        ]

        if not recent_messages:
            return 0.0

        # Simple heuristic based on message patterns
        conflict_keywords = ["disagree", "angry", "frustrated", "conflict", "fight"]
        conflict_count = sum(
            1 for m in recent_messages
            if any(kw in m.content.lower() for kw in conflict_keywords)
        )

        return min(1.0, conflict_count / max(1, len(recent_messages)))

    def _assess_communication_clarity(self) -> float:
        """Assess clarity of internal communication."""
        if not self._messages:
            return 0.5

        recent_messages = [
            m for m in self._messages
            if time.time() - m.timestamp < 3600
        ]

        if not recent_messages:
            return 0.5

        # Clarity based on read rate and response patterns
        read_rate = np.mean([
            len(m.read_by) / max(1, len(m.recipient_ids))
            for m in recent_messages
            if m.recipient_ids != ["all"]
        ]) if recent_messages else 0.5

        return float(read_rate)

    def send_internal_message(
        self,
        sender_id: str,
        recipient_ids: list[str],
        content: str,
        message_type: CommunicationType = CommunicationType.DIRECT,
    ) -> InternalMessage:
        """
        Send a message within the system.

        Args:
            sender_id: ID of the sending alter.
            recipient_ids: IDs of recipient alters or ["all"].
            content: Message content.
            message_type: Type of communication.

        Returns:
            The created InternalMessage.
        """
        message = InternalMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_ids=recipient_ids,
            content=content,
            message_type=message_type,
        )

        self._messages.append(message)
        return message

    def get_messages_for_alter(
        self,
        alter_id: str,
        unread_only: bool = False,
    ) -> list[InternalMessage]:
        """
        Get messages for a specific alter.

        Args:
            alter_id: ID of the alter.
            unread_only: Whether to return only unread messages.

        Returns:
            List of messages for the alter.
        """
        messages = [
            m for m in self._messages
            if alter_id in m.recipient_ids or "all" in m.recipient_ids
        ]

        if unread_only:
            messages = [m for m in messages if alter_id not in m.read_by]

        return messages

    def mark_message_read(self, message_id: str, alter_id: str) -> bool:
        """Mark a message as read by an alter."""
        for message in self._messages:
            if message.message_id == message_id:
                if alter_id not in message.read_by:
                    message.read_by.append(alter_id)
                return True
        return False

    def create_shared_memory(
        self,
        memory_content: dict[str, Any],
        creator_id: str,
        shared_with: list[str] | None = None,
    ) -> str:
        """
        Create a shared memory accessible to specified alters.

        Args:
            memory_content: The memory content to share.
            creator_id: ID of the alter creating the memory.
            shared_with: List of alter IDs to share with (None = all).

        Returns:
            The memory ID.
        """
        memory_id = hashlib.sha256(
            f"{creator_id}{time.time()}{str(memory_content)}".encode()
        ).hexdigest()[:16]

        self._shared_memories[memory_id] = {
            "content": memory_content,
            "creator": creator_id,
            "created_at": time.time(),
        }

        # Set consent
        if shared_with is None:
            self._memory_consent[memory_id] = set(self._alters.keys())
        else:
            self._memory_consent[memory_id] = set(shared_with) | {creator_id}

        return memory_id

    def access_shared_memory(
        self,
        memory_id: str,
        alter_id: str,
    ) -> dict[str, Any] | None:
        """
        Access a shared memory if the alter has consent.

        Args:
            memory_id: ID of the memory to access.
            alter_id: ID of the alter requesting access.

        Returns:
            Memory content if access granted, None otherwise.
        """
        if memory_id not in self._shared_memories:
            return None

        if memory_id not in self._memory_consent:
            return None

        if alter_id not in self._memory_consent[memory_id]:
            return None

        return self._shared_memories[memory_id]["content"]

    def grant_memory_access(
        self,
        memory_id: str,
        granter_id: str,
        grantee_id: str,
    ) -> bool:
        """
        Grant another alter access to a shared memory.

        Args:
            memory_id: ID of the memory.
            granter_id: ID of the alter granting access.
            grantee_id: ID of the alter receiving access.

        Returns:
            True if access was granted.
        """
        if memory_id not in self._shared_memories:
            return False

        if memory_id not in self._memory_consent:
            return False

        # Only those with access can grant access
        if granter_id not in self._memory_consent[memory_id]:
            return False

        self._memory_consent[memory_id].add(grantee_id)
        return True

    def revoke_memory_access(
        self,
        memory_id: str,
        revoker_id: str,
        revokee_id: str,
    ) -> bool:
        """
        Revoke another alter's access to a shared memory.

        Args:
            memory_id: ID of the memory.
            revoker_id: ID of the alter revoking access.
            revokee_id: ID of the alter losing access.

        Returns:
            True if access was revoked.
        """
        if memory_id not in self._memory_consent:
            return False

        # Only creator can revoke
        if self._shared_memories.get(memory_id, {}).get("creator") != revoker_id:
            return False

        self._memory_consent[memory_id].discard(revokee_id)
        return True

    def facilitate_conflict_resolution(
        self,
        alter_ids: list[str],
        conflict_description: str,
    ) -> dict[str, Any]:
        """
        Facilitate conflict resolution between alters.

        This provides structured support for resolving internal conflicts
        without forcing integration or invalidating any alter's perspective.

        Args:
            alter_ids: IDs of alters involved in the conflict.
            conflict_description: Description of the conflict.

        Returns:
            Resolution framework with suggestions.
        """
        # Get profiles of involved alters
        profiles = [self._alters.get(aid) for aid in alter_ids if aid in self._alters]

        # Generate resolution framework
        framework = {
            "conflict_id": str(uuid.uuid4()),
            "participants": [p.name for p in profiles if p],
            "timestamp": time.time(),
            "acknowledgment": (
                "All perspectives in this conflict are valid. "
                "The goal is not to determine who is 'right' but to find "
                "ways to reduce internal distress while honoring everyone's needs."
            ),
            "suggested_steps": [
                "1. Each part shares their perspective without interruption",
                "2. Identify the underlying needs behind each position",
                "3. Look for common ground and shared goals",
                "4. Brainstorm solutions that honor multiple perspectives",
                "5. Agree on a trial approach with planned check-in",
            ],
            "grounding_reminder": (
                "If this process becomes overwhelming, it's okay to pause. "
                "Internal work doesn't have to happen all at once."
            ),
            "professional_support_note": (
                "Complex internal conflicts often benefit from support from "
                "a therapist experienced with DID/plural systems."
            ),
        }

        return framework

    def get_system_state(self) -> SystemState | None:
        """Get the current system state."""
        return self._current_state

    def get_switch_history(
        self,
        limit: int = 100,
        since_timestamp: float | None = None,
    ) -> list[SwitchEvent]:
        """
        Get switch history.

        Args:
            limit: Maximum number of events to return.
            since_timestamp: Only return events after this timestamp.

        Returns:
            List of switch events.
        """
        events = self._switch_history

        if since_timestamp:
            events = [e for e in events if e.timestamp > since_timestamp]

        return events[-limit:]

    def generate_system_report(self) -> dict[str, Any]:
        """
        Generate a report on system functioning.

        Returns:
            Dictionary containing system metrics and insights.
        """
        return {
            "system_id": self.system_id,
            "timestamp": time.time(),
            "total_alters": len(self._alters),
            "current_state": {
                "fronting": self._current_state.fronting_alters if self._current_state else [],
                "entropy": self._current_state.system_entropy if self._current_state else 0.5,
                "conflict_level": self._current_state.internal_conflict_level if self._current_state else 0.0,
                "communication_clarity": self._current_state.communication_clarity if self._current_state else 0.5,
            },
            "switch_statistics": {
                "total_recorded": len(self._switch_history),
                "last_24h": len([
                    s for s in self._switch_history
                    if time.time() - s.timestamp < 86400
                ]),
                "average_smoothness": float(np.mean([
                    s.smoothness for s in self._switch_history
                ])) if self._switch_history else 0.5,
            },
            "communication_statistics": {
                "total_messages": len(self._messages),
                "unread_messages": sum(
                    1 for m in self._messages
                    if len(m.read_by) < len(m.recipient_ids)
                ),
            },
            "shared_memories": len(self._shared_memories),
            "disclaimer": (
                "This report is for informational purposes only. "
                "It is not a clinical assessment. Please work with "
                "qualified mental health professionals for treatment."
            ),
        }
