"""Memory systems including RIME, continuity store, and timeline threading."""

from reunity.memory.continuity_store import (
    ConsentScope,
    IdentityState,
    JournalEntry,
    MemoryEntry,
    MemoryType,
    RecursiveIdentityMemoryEngine,
    RetrievalResult,
    TimelineEvent,
)
from reunity.memory.timeline_threading import (
    MemoryValence,
    NarrativeSegment,
    ThreadType,
    TimelineGap,
    TimelineMemory,
    TimelineThread,
    TimelineThreader,
)

# Backwards compatibility aliases
ContinuityMemoryStore = RecursiveIdentityMemoryEngine

__all__ = [
    # continuity_store classes
    "ConsentScope",
    "ContinuityMemoryStore",  # alias for backwards compatibility
    "IdentityState",
    "JournalEntry",
    "MemoryEntry",
    "MemoryType",
    "RecursiveIdentityMemoryEngine",
    "RetrievalResult",
    "TimelineEvent",
    # timeline_threading classes
    "MemoryValence",
    "NarrativeSegment",
    "ThreadType",
    "TimelineGap",
    "TimelineMemory",
    "TimelineThread",
    "TimelineThreader",
]
