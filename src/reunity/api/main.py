"""
ReUnity FastAPI Backend

Production-ready REST API for the ReUnity trauma-aware AI system.
Provides endpoints for:
- Entropy analysis and state detection
- Memory management with consent controls
- Protective pattern recognition
- Reflection and dialogue support
- Export and portability bundles

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only. ReUnity is not intended to diagnose, treat, cure,
or prevent any medical or psychological condition.

Author: Christopher Ezernack
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from reunity.core.entropy import (
    EntropyAnalyzer,
    EntropyState,
    EntropyMetrics,
)
from reunity.router.state_router import StateRouter, PolicyType
from reunity.protective.pattern_recognizer import (
    ProtectivePatternRecognizer,
    InteractionAnalysis,
)
from reunity.memory.continuity_store import (
    RecursiveIdentityMemoryEngine,
    ConsentScope,
    MemoryType,
    MemoryEntry,
)
from reunity.reflection.mirror_link import (
    MirrorLinkDialogueCompanion,
    CommunicationStyle,
)
from reunity.regime.regime_controller import (
    RegimeController,
    Regime,
    Apostasis,
    Regeneration,
    LatticeMemoryGraph,
)
from reunity.storage.encrypted_store import EncryptedStorage, StorageConfig

# Import extended endpoints
try:
    from reunity.api.endpoints_extended import router as extended_router
    HAS_EXTENDED_ENDPOINTS = True
except ImportError:
    HAS_EXTENDED_ENDPOINTS = False


# ============================================================================
# DISCLAIMER
# ============================================================================
DISCLAIMER = """
IMPORTANT DISCLAIMER:

ReUnity is NOT a clinical or treatment tool. It is a theoretical and support
framework only. ReUnity is not intended to diagnose, treat, cure, or prevent
any medical or psychological condition.

This system is designed to provide supportive tools for individuals working
with mental health professionals. It should be used as a complement to, not
a replacement for, professional mental health care.

If you are experiencing a mental health crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

Always consult with qualified mental health professionals for diagnosis
and treatment of mental health conditions.
"""


# ============================================================================
# Pydantic Models
# ============================================================================

class EntropyRequest(BaseModel):
    """Request for entropy analysis."""

    text: str = Field(..., description="Text to analyze for entropy")
    include_stability: bool = Field(
        default=True,
        description="Include stability analysis",
    )


class EntropyResponse(BaseModel):
    """Response from entropy analysis."""

    state: str
    normalized_entropy: float
    confidence: float
    is_stable: bool
    lyapunov_exponent: float | None
    recommendations: list[str]


class MemoryRequest(BaseModel):
    """Request to add a memory."""

    identity: str = Field(..., description="Identity state identifier")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(default="episodic", description="Type of memory")
    tags: list[str] = Field(default_factory=list, description="Tags")
    consent_scope: str = Field(default="private", description="Consent scope")
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryResponse(BaseModel):
    """Response for memory operations."""

    id: str
    content: str
    memory_type: str
    identity_state: str
    timestamp: float
    tags: list[str]
    consent_scope: str


class RetrievalRequest(BaseModel):
    """Request for memory retrieval."""

    identity: str = Field(..., description="Current identity state")
    query: str = Field(..., description="Search query")
    crisis_level: float = Field(default=0.0, ge=0.0, le=1.0)
    max_results: int = Field(default=5, ge=1, le=20)


class RetrievalResponse(BaseModel):
    """Response from memory retrieval."""

    memories: list[MemoryResponse]
    total_found: int
    filtered_by_consent: int
    retrieval_method: str


class PatternRequest(BaseModel):
    """Request for pattern analysis."""

    interactions: list[dict[str, Any]] = Field(
        ...,
        description="List of interactions with 'text' and 'timestamp'",
    )
    person_id: str | None = Field(default=None, description="Person identifier")


class PatternResponse(BaseModel):
    """Response from pattern analysis."""

    patterns_detected: list[dict[str, Any]]
    overall_risk: float
    sentiment_variance: float
    stability_assessment: str
    recommendations: list[str]


class ReflectionRequest(BaseModel):
    """Request for reflection generation."""

    current_emotion: str = Field(..., description="Current emotional state")
    past_context: str | None = Field(default=None, description="Past context")
    style: str = Field(default="gentle", description="Communication style")


class ReflectionResponse(BaseModel):
    """Response from reflection generation."""

    content: str
    reflection_type: str
    is_contradiction: bool
    follow_up_question: str | None
    grounding_prompt: str | None


class JournalRequest(BaseModel):
    """Request to add a journal entry."""

    title: str = Field(..., description="Entry title")
    content: str = Field(..., description="Entry content")
    identity: str = Field(..., description="Identity state")
    mood: str = Field(..., description="Current mood")
    energy_level: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    consent_scope: str = Field(default="private")


class JournalResponse(BaseModel):
    """Response for journal operations."""

    id: str
    title: str
    identity_state: str
    timestamp: float
    mood: str


class ExportRequest(BaseModel):
    """Request for data export."""

    identity: str | None = Field(default=None, description="Filter by identity")
    include_memories: bool = Field(default=True)
    include_journals: bool = Field(default=True)
    include_timeline: bool = Field(default=True)


class RegimeResponse(BaseModel):
    """Response for regime status."""

    regime: str
    entropy_band: str
    confidence: float
    time_in_regime: float
    apostasis_active: bool
    regeneration_active: bool


class ConsentUpdateRequest(BaseModel):
    """Request to update consent scope."""

    memory_id: str = Field(..., description="Memory ID to update")
    new_scope: str = Field(..., description="New consent scope")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    disclaimer: str
    timestamp: float


# ============================================================================
# Application Setup
# ============================================================================

# Global instances (in production, use dependency injection)
entropy_analyzer: EntropyAnalyzer | None = None
state_router: StateRouter | None = None
pattern_recognizer: ProtectivePatternRecognizer | None = None
memory_engine: RecursiveIdentityMemoryEngine | None = None
dialogue_companion: MirrorLinkDialogueCompanion | None = None
regime_controller: RegimeController | None = None
apostasis: Apostasis | None = None
regeneration: Regeneration | None = None
lattice_graph: LatticeMemoryGraph | None = None
encrypted_storage: EncryptedStorage | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global entropy_analyzer, state_router, pattern_recognizer
    global memory_engine, dialogue_companion, regime_controller
    global apostasis, regeneration, lattice_graph, encrypted_storage

    # Initialize components
    entropy_analyzer = EntropyAnalyzer()
    state_router = StateRouter()
    pattern_recognizer = ProtectivePatternRecognizer()
    memory_engine = RecursiveIdentityMemoryEngine()
    dialogue_companion = MirrorLinkDialogueCompanion()
    regime_controller = RegimeController()
    apostasis = Apostasis()
    regeneration = Regeneration()
    lattice_graph = LatticeMemoryGraph()

    # Initialize encrypted storage
    storage_path = Path(os.environ.get("REUNITY_STORAGE_PATH", "./data"))
    storage_config = StorageConfig(storage_path=storage_path)
    encrypted_storage = EncryptedStorage(storage_config)

    yield

    # Cleanup
    if encrypted_storage:
        encrypted_storage.close()


app = FastAPI(
    title="ReUnity API",
    description=f"""
Trauma-aware AI support system API.

{DISCLAIMER}
""",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include extended endpoints
if HAS_EXTENDED_ENDPOINTS:
    app.include_router(
        extended_router,
        prefix="/v1",
        tags=["extended"],
    )


# ============================================================================
# Dependency Injection
# ============================================================================

def get_entropy_analyzer() -> EntropyAnalyzer:
    """Get entropy analyzer instance."""
    if entropy_analyzer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return entropy_analyzer


def get_state_router() -> StateRouter:
    """Get state router instance."""
    if state_router is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return state_router


def get_pattern_recognizer() -> ProtectivePatternRecognizer:
    """Get pattern recognizer instance."""
    if pattern_recognizer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return pattern_recognizer


def get_memory_engine() -> RecursiveIdentityMemoryEngine:
    """Get memory engine instance."""
    if memory_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return memory_engine


def get_dialogue_companion() -> MirrorLinkDialogueCompanion:
    """Get dialogue companion instance."""
    if dialogue_companion is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return dialogue_companion


def get_regime_controller() -> RegimeController:
    """Get regime controller instance."""
    if regime_controller is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return regime_controller


# ============================================================================
# Health and Info Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check and disclaimer."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        disclaimer=DISCLAIMER,
        timestamp=time.time(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        disclaimer=DISCLAIMER,
        timestamp=time.time(),
    )


@app.get("/disclaimer")
async def get_disclaimer():
    """Get full disclaimer text."""
    return {"disclaimer": DISCLAIMER}


# ============================================================================
# Entropy Analysis Endpoints
# ============================================================================

@app.post("/entropy/analyze", response_model=EntropyResponse)
async def analyze_entropy(
    request: EntropyRequest,
    analyzer: EntropyAnalyzer = Depends(get_entropy_analyzer),
    router: StateRouter = Depends(get_state_router),
):
    """
    Analyze text for entropy state.

    Detects emotional/cognitive entropy using Shannon entropy,
    Jensen-Shannon divergence, and stability analysis.
    """
    # Convert text to probability distribution
    # Simple word frequency distribution
    words = request.text.lower().split()
    if not words:
        raise HTTPException(status_code=400, detail="Empty text provided")

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    total = sum(word_counts.values())
    distribution = np.array([c / total for c in word_counts.values()])

    # Analyze entropy
    metrics = analyzer.analyze(distribution)

    # Get policy recommendations
    policy = router.route(metrics)

    return EntropyResponse(
        state=metrics.state.value,
        normalized_entropy=metrics.normalized_entropy,
        confidence=metrics.confidence,
        is_stable=metrics.is_stable,
        lyapunov_exponent=metrics.lyapunov_exponent,
        recommendations=policy.recommendations,
    )


@app.get("/entropy/states")
async def get_entropy_states():
    """Get available entropy states and their descriptions."""
    return {
        "states": [
            {
                "name": state.value,
                "description": _get_state_description(state),
            }
            for state in EntropyState
        ]
    }


def _get_state_description(state: EntropyState) -> str:
    """Get description for entropy state."""
    descriptions = {
        EntropyState.LOW: "Low entropy - highly predictable, possibly rigid",
        EntropyState.STABLE: "Stable entropy - healthy variability",
        EntropyState.ELEVATED: "Elevated entropy - increased uncertainty",
        EntropyState.HIGH: "High entropy - significant instability",
        EntropyState.CRISIS: "Crisis level entropy - immediate support needed",
    }
    return descriptions.get(state, "Unknown state")


# ============================================================================
# Memory Management Endpoints
# ============================================================================

@app.post("/memory/add", response_model=MemoryResponse)
async def add_memory(
    request: MemoryRequest,
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
    analyzer: EntropyAnalyzer = Depends(get_entropy_analyzer),
):
    """
    Add a memory to the continuity store.

    Memories are tagged with entropy level at creation and
    respect consent scope settings.
    """
    # Get current entropy
    words = request.content.lower().split()
    if words:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        total = sum(word_counts.values())
        distribution = np.array([c / total for c in word_counts.values()])
        metrics = analyzer.analyze(distribution)
        entropy = metrics.normalized_entropy
    else:
        entropy = 0.5

    # Parse memory type
    try:
        mem_type = MemoryType(request.memory_type)
    except ValueError:
        mem_type = MemoryType.EPISODIC

    # Parse consent scope
    try:
        scope = ConsentScope(request.consent_scope)
    except ValueError:
        scope = ConsentScope.PRIVATE

    # Add memory
    memory = engine.add_memory(
        identity=request.identity,
        content=request.content,
        memory_type=mem_type,
        tags=request.tags,
        entropy=entropy,
        consent_scope=scope,
        emotional_valence=request.emotional_valence,
        importance=request.importance,
    )

    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        memory_type=memory.memory_type.value,
        identity_state=memory.identity_state,
        timestamp=memory.timestamp,
        tags=memory.tags,
        consent_scope=memory.consent_scope.value,
    )


@app.post("/memory/retrieve", response_model=RetrievalResponse)
async def retrieve_memories(
    request: RetrievalRequest,
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
):
    """
    Retrieve memories with grounding support.

    Prioritizes safe/grounding memories during crisis states.
    Respects consent scope settings.
    """
    result = engine.retrieve_grounding(
        current_identity=request.identity,
        query=request.query,
        crisis_level=request.crisis_level,
        max_results=request.max_results,
    )

    return RetrievalResponse(
        memories=[
            MemoryResponse(
                id=m.id,
                content=m.content,
                memory_type=m.memory_type.value,
                identity_state=m.identity_state,
                timestamp=m.timestamp,
                tags=m.tags,
                consent_scope=m.consent_scope.value,
            )
            for m in result.memories
        ],
        total_found=result.total_found,
        filtered_by_consent=result.filtered_by_consent,
        retrieval_method=result.retrieval_method,
    )


@app.put("/memory/consent", response_model=dict)
async def update_consent(
    request: ConsentUpdateRequest,
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
):
    """Update consent scope for a memory."""
    try:
        new_scope = ConsentScope(request.new_scope)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid consent scope")

    success = engine.set_consent_scope(request.memory_id, new_scope)

    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"status": "updated", "memory_id": request.memory_id}


@app.get("/memory/stats")
async def get_memory_stats(
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
):
    """Get memory store statistics."""
    return engine.get_statistics()


# ============================================================================
# Pattern Recognition Endpoints
# ============================================================================

@app.post("/patterns/analyze", response_model=PatternResponse)
async def analyze_patterns(
    request: PatternRequest,
    recognizer: ProtectivePatternRecognizer = Depends(get_pattern_recognizer),
):
    """
    Analyze interactions for harmful patterns.

    Detects gaslighting, hot-cold cycles, isolation attempts,
    and other potentially harmful relationship dynamics.
    """
    analysis = recognizer.analyze_interactions(
        interactions=request.interactions,
        person_id=request.person_id,
    )

    return PatternResponse(
        patterns_detected=[
            {
                "type": p.pattern_type.value,
                "severity": p.severity.value,
                "confidence": p.confidence,
                "message": p.message,
                "recommendation": p.recommendation,
            }
            for p in analysis.patterns_detected
        ],
        overall_risk=analysis.overall_risk,
        sentiment_variance=analysis.sentiment_variance,
        stability_assessment=analysis.stability_assessment,
        recommendations=analysis.recommendations,
    )


# ============================================================================
# Reflection Endpoints
# ============================================================================

@app.post("/reflection/generate", response_model=ReflectionResponse)
async def generate_reflection(
    request: ReflectionRequest,
    companion: MirrorLinkDialogueCompanion = Depends(get_dialogue_companion),
    analyzer: EntropyAnalyzer = Depends(get_entropy_analyzer),
):
    """
    Generate a reflection based on current emotion and past context.

    Reflects contradictions without invalidation, helping users
    hold multiple truths simultaneously.
    """
    # Get current entropy state
    words = request.current_emotion.lower().split()
    if words:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        total = sum(word_counts.values())
        distribution = np.array([c / total for c in word_counts.values()])
        metrics = analyzer.analyze(distribution)
        entropy_state = metrics.state
    else:
        entropy_state = EntropyState.STABLE

    # Parse style
    try:
        style = CommunicationStyle(request.style)
    except ValueError:
        style = CommunicationStyle.GENTLE

    # Generate reflection
    reflection = companion.reflect(
        current_emotion=request.current_emotion,
        past_context=request.past_context,
        entropy_state=entropy_state,
        style=style,
    )

    return ReflectionResponse(
        content=reflection.content,
        reflection_type=reflection.reflection_type.value,
        is_contradiction=reflection.is_contradiction,
        follow_up_question=reflection.follow_up_question,
        grounding_prompt=reflection.grounding_prompt,
    )


# ============================================================================
# Journal Endpoints
# ============================================================================

@app.post("/journal/add", response_model=JournalResponse)
async def add_journal_entry(
    request: JournalRequest,
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
    analyzer: EntropyAnalyzer = Depends(get_entropy_analyzer),
):
    """Add a journal entry."""
    # Get current entropy
    words = request.content.lower().split()
    if words:
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        total = sum(word_counts.values())
        distribution = np.array([c / total for c in word_counts.values()])
        metrics = analyzer.analyze(distribution)
        entropy = metrics.normalized_entropy
    else:
        entropy = 0.5

    # Parse consent scope
    try:
        scope = ConsentScope(request.consent_scope)
    except ValueError:
        scope = ConsentScope.PRIVATE

    entry = engine.add_journal_entry(
        title=request.title,
        content=request.content,
        identity=request.identity,
        mood=request.mood,
        energy_level=request.energy_level,
        entropy_level=entropy,
        tags=request.tags,
        consent_scope=scope,
    )

    return JournalResponse(
        id=entry.id,
        title=entry.title,
        identity_state=entry.identity_state,
        timestamp=entry.timestamp,
        mood=entry.mood,
    )


@app.get("/journal/list")
async def list_journals(
    identity: str | None = Query(default=None),
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
):
    """List journal entries."""
    entries = engine.get_journals(identity=identity)

    return {
        "entries": [
            {
                "id": e.id,
                "title": e.title,
                "identity_state": e.identity_state,
                "timestamp": e.timestamp,
                "mood": e.mood,
            }
            for e in entries
        ]
    }


# ============================================================================
# Regime Endpoints
# ============================================================================

@app.get("/regime/status", response_model=RegimeResponse)
async def get_regime_status(
    controller: RegimeController = Depends(get_regime_controller),
    analyzer: EntropyAnalyzer = Depends(get_entropy_analyzer),
):
    """Get current regime status."""
    # Create dummy metrics for status check
    metrics = EntropyMetrics(
        shannon_entropy=0.5,
        normalized_entropy=0.5,
        state=EntropyState.STABLE,
        confidence=0.8,
        is_stable=True,
    )

    state = controller.update(metrics)

    return RegimeResponse(
        regime=state.regime.value,
        entropy_band=state.entropy_band.value,
        confidence=state.confidence,
        time_in_regime=state.time_in_regime,
        apostasis_active=state.apostasis_active,
        regeneration_active=state.regeneration_active,
    )


@app.get("/regime/history")
async def get_regime_history(
    controller: RegimeController = Depends(get_regime_controller),
):
    """Get regime transition history."""
    history = controller.get_regime_history()

    return {
        "history": [
            {"regime": r.value, "timestamp": t}
            for r, t in history
        ]
    }


# ============================================================================
# Export Endpoints
# ============================================================================

@app.post("/export/bundle")
async def export_bundle(
    request: ExportRequest,
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
):
    """
    Export data as a portability bundle.

    Includes provenance information and respects consent scopes.
    """
    export_data = engine.export_memories(identity=request.identity)

    # Add provenance
    export_data["provenance"] = {
        "exported_at": time.time(),
        "export_version": "1.0.0",
        "system": "ReUnity",
        "disclaimer": DISCLAIMER,
    }

    return export_data


@app.get("/export/timeline")
async def export_timeline(
    identity: str | None = Query(default=None),
    engine: RecursiveIdentityMemoryEngine = Depends(get_memory_engine),
):
    """Export timeline events."""
    timeline = engine.get_timeline(identity=identity)

    return {
        "timeline": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "description": e.description,
                "timestamp": e.timestamp,
                "identity_state": e.identity_state,
                "entropy_level": e.entropy_level,
            }
            for e in timeline
        ],
        "provenance": {
            "exported_at": time.time(),
            "system": "ReUnity",
        },
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "disclaimer": "If you are in crisis, please contact emergency services.",
        },
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "reunity.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
