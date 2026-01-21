"""
ReUnity Extended API Endpoints

Additional endpoints for:
- Alter-Aware Subsystem (AAS) operations
- Clinician/Caregiver Interface (CCI)
- Safety Assessment
- Grounding Techniques
- Free Energy Analysis

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Import components
from reunity.alter.alter_aware import (
    AlterAwareSubsystem,
    AlterProfile,
    AlterState,
    CommunicationType,
)
from reunity.clinician.caregiver_interface import (
    ClinicianCaregiverInterface,
    AccessLevel,
    ProviderType,
)
from reunity.protective.safety_assessment import (
    SafetyAssessor,
    RiskLevel,
)
from reunity.grounding.techniques import (
    GroundingTechniquesLibrary,
    GroundingCategory,
    IntensityLevel,
)
from reunity.core.free_energy import (
    FreeEnergyMinimizer,
    PredictiveProcessor,
)


# ============================================================================
# Pydantic Models
# ============================================================================

class AlterProfileRequest(BaseModel):
    """Request to register an alter profile."""

    name: str = Field(..., description="Alter name")
    pronouns: str = Field(default="they/them", description="Pronouns")
    age_presentation: str | None = Field(default=None, description="Age presentation")
    role: str | None = Field(default=None, description="System role")
    communication_style: str = Field(default="neutral", description="Communication style")


class AlterProfileResponse(BaseModel):
    """Response for alter profile operations."""

    alter_id: str
    name: str
    pronouns: str
    role: str | None
    communication_style: str


class SwitchEventRequest(BaseModel):
    """Request to record a switch event."""

    from_alter_ids: list[str] = Field(..., description="Alters who were fronting")
    to_alter_ids: list[str] = Field(..., description="Alters now fronting")
    trigger: str | None = Field(default=None, description="Switch trigger")
    smoothness: float = Field(default=0.5, ge=0.0, le=1.0, description="Switch smoothness")
    notes: str = Field(default="", description="Notes about the switch")


class InternalMessageRequest(BaseModel):
    """Request to send an internal message."""

    sender_id: str = Field(..., description="Sender alter ID")
    recipient_ids: list[str] = Field(..., description="Recipient alter IDs")
    content: str = Field(..., description="Message content")
    message_type: str = Field(default="direct", description="Message type")


class ProviderRegistrationRequest(BaseModel):
    """Request to register a care provider."""

    name: str = Field(..., description="Provider name")
    provider_type: str = Field(..., description="Provider type")
    organization: str | None = Field(default=None, description="Organization")
    credentials: str | None = Field(default=None, description="Credentials")


class ConsentGrantRequest(BaseModel):
    """Request to grant consent to a provider."""

    provider_id: str = Field(..., description="Provider ID")
    access_level: str = Field(..., description="Access level")
    data_types: list[str] = Field(..., description="Data types to share")
    duration_days: int | None = Field(default=None, description="Duration in days")


class SafetyAssessmentRequest(BaseModel):
    """Request for safety assessment."""

    text: str = Field(..., description="Text to assess")
    entropy_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Current entropy")


class SafetyAssessmentResponse(BaseModel):
    """Response from safety assessment."""

    risk_level: str
    crisis_types: list[str]
    risk_factors: list[str]
    protective_factors: list[str]
    recommended_actions: list[str]
    entropy_level: float


class GroundingRequest(BaseModel):
    """Request for grounding technique recommendation."""

    entropy_level: float = Field(..., ge=0.0, le=1.0, description="Current entropy")
    category_preference: str | None = Field(default=None, description="Preferred category")


class GroundingResponse(BaseModel):
    """Response with grounding technique."""

    technique_id: str
    name: str
    category: str
    intensity: str
    description: str
    instructions: list[str]
    duration_minutes: float


class GroundingSessionRequest(BaseModel):
    """Request to start/complete a grounding session."""

    technique_id: str = Field(..., description="Technique ID")
    entropy_before: float = Field(..., ge=0.0, le=1.0, description="Entropy before")
    entropy_after: float | None = Field(default=None, description="Entropy after (for completion)")
    effectiveness_rating: float | None = Field(default=None, ge=0.0, le=1.0, description="Effectiveness rating")


# ============================================================================
# Global Instances
# ============================================================================

# These would be properly dependency-injected in production
_alter_subsystem: AlterAwareSubsystem | None = None
_clinician_interface: ClinicianCaregiverInterface | None = None
_safety_assessor: SafetyAssessor | None = None
_grounding_library: GroundingTechniquesLibrary | None = None
_free_energy_minimizer: FreeEnergyMinimizer | None = None
_predictive_processor: PredictiveProcessor | None = None


def get_alter_subsystem() -> AlterAwareSubsystem:
    """Get alter subsystem instance."""
    global _alter_subsystem
    if _alter_subsystem is None:
        _alter_subsystem = AlterAwareSubsystem()
    return _alter_subsystem


def get_clinician_interface() -> ClinicianCaregiverInterface:
    """Get clinician interface instance."""
    global _clinician_interface
    if _clinician_interface is None:
        _clinician_interface = ClinicianCaregiverInterface(user_id="default_user")
    return _clinician_interface


def get_safety_assessor() -> SafetyAssessor:
    """Get safety assessor instance."""
    global _safety_assessor
    if _safety_assessor is None:
        _safety_assessor = SafetyAssessor()
    return _safety_assessor


def get_grounding_library() -> GroundingTechniquesLibrary:
    """Get grounding library instance."""
    global _grounding_library
    if _grounding_library is None:
        _grounding_library = GroundingTechniquesLibrary()
    return _grounding_library


def get_free_energy_minimizer() -> FreeEnergyMinimizer:
    """Get free energy minimizer instance."""
    global _free_energy_minimizer
    if _free_energy_minimizer is None:
        _free_energy_minimizer = FreeEnergyMinimizer()
    return _free_energy_minimizer


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter()


# ============================================================================
# Alter-Aware Subsystem Endpoints
# ============================================================================

@router.post("/alter/register", response_model=AlterProfileResponse)
async def register_alter(
    request: AlterProfileRequest,
    subsystem: AlterAwareSubsystem = Depends(get_alter_subsystem),
):
    """
    Register a new alter in the system.

    The Alter-Aware Subsystem recognizes and validates the existence
    of multiple identity states while promoting healthy internal
    communication and cooperation.
    """
    profile = AlterProfile(
        alter_id="",  # Will be generated
        name=request.name,
        pronouns=request.pronouns,
        age_presentation=request.age_presentation,
        role=request.role,
        communication_style=request.communication_style,
    )

    alter_id = subsystem.register_alter(profile)

    return AlterProfileResponse(
        alter_id=alter_id,
        name=profile.name,
        pronouns=profile.pronouns,
        role=profile.role,
        communication_style=profile.communication_style,
    )


@router.get("/alter/list")
async def list_alters(
    subsystem: AlterAwareSubsystem = Depends(get_alter_subsystem),
):
    """List all registered alters."""
    alters = subsystem.list_alters()

    return {
        "alters": [
            {
                "alter_id": a.alter_id,
                "name": a.name,
                "pronouns": a.pronouns,
                "role": a.role,
                "last_active": a.last_active,
            }
            for a in alters
        ]
    }


@router.post("/alter/switch")
async def record_switch(
    request: SwitchEventRequest,
    subsystem: AlterAwareSubsystem = Depends(get_alter_subsystem),
):
    """Record an alter switch event."""
    event = subsystem.record_switch(
        from_alter_ids=request.from_alter_ids,
        to_alter_ids=request.to_alter_ids,
        trigger=request.trigger,
        smoothness=request.smoothness,
        notes=request.notes,
    )

    return {
        "event_id": event.event_id,
        "from_alters": event.from_alter_ids,
        "to_alters": event.to_alter_ids,
        "smoothness": event.smoothness,
        "timestamp": event.timestamp,
    }


@router.post("/alter/message")
async def send_internal_message(
    request: InternalMessageRequest,
    subsystem: AlterAwareSubsystem = Depends(get_alter_subsystem),
):
    """Send an internal message between alters."""
    try:
        msg_type = CommunicationType(request.message_type)
    except ValueError:
        msg_type = CommunicationType.DIRECT

    message = subsystem.send_internal_message(
        sender_id=request.sender_id,
        recipient_ids=request.recipient_ids,
        content=request.content,
        message_type=msg_type,
    )

    return {
        "message_id": message.message_id,
        "sender_id": message.sender_id,
        "recipient_ids": message.recipient_ids,
        "timestamp": message.timestamp,
    }


@router.get("/alter/messages/{alter_id}")
async def get_messages_for_alter(
    alter_id: str,
    unread_only: bool = False,
    subsystem: AlterAwareSubsystem = Depends(get_alter_subsystem),
):
    """Get messages for a specific alter."""
    messages = subsystem.get_messages_for_alter(
        alter_id=alter_id,
        unread_only=unread_only,
    )

    return {
        "messages": [
            {
                "message_id": m.message_id,
                "sender_id": m.sender_id,
                "content": m.content,
                "timestamp": m.timestamp,
                "read": alter_id in m.read_by,
            }
            for m in messages
        ]
    }


@router.get("/alter/system-report")
async def get_system_report(
    subsystem: AlterAwareSubsystem = Depends(get_alter_subsystem),
):
    """Get a report on system functioning."""
    return subsystem.generate_system_report()


# ============================================================================
# Clinician/Caregiver Interface Endpoints
# ============================================================================

@router.post("/clinician/register-provider")
async def register_provider(
    request: ProviderRegistrationRequest,
    interface: ClinicianCaregiverInterface = Depends(get_clinician_interface),
):
    """Register a new care provider."""
    try:
        provider_type = ProviderType(request.provider_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid provider type")

    provider = interface.register_provider(
        name=request.name,
        provider_type=provider_type,
        organization=request.organization,
        credentials=request.credentials,
    )

    return {
        "provider_id": provider.provider_id,
        "name": provider.name,
        "provider_type": provider.provider_type.value,
        "verified": provider.verified,
    }


@router.post("/clinician/grant-consent")
async def grant_consent(
    request: ConsentGrantRequest,
    interface: ClinicianCaregiverInterface = Depends(get_clinician_interface),
):
    """Grant consent for a provider to access data."""
    try:
        access_level = AccessLevel(request.access_level)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid access level")

    consent = interface.grant_consent(
        provider_id=request.provider_id,
        access_level=access_level,
        data_types=request.data_types,
        duration_days=request.duration_days,
    )

    if consent is None:
        raise HTTPException(status_code=404, detail="Provider not found")

    return {
        "consent_id": consent.consent_id,
        "provider_id": consent.provider_id,
        "access_level": consent.access_level.value,
        "status": consent.status.value,
        "expires_at": consent.expires_at,
    }


@router.post("/clinician/revoke-consent/{consent_id}")
async def revoke_consent(
    consent_id: str,
    interface: ClinicianCaregiverInterface = Depends(get_clinician_interface),
):
    """Revoke a previously granted consent."""
    success = interface.revoke_consent(consent_id)

    if not success:
        raise HTTPException(status_code=404, detail="Consent not found")

    return {"status": "revoked", "consent_id": consent_id}


@router.get("/clinician/consent-records")
async def get_consent_records(
    interface: ClinicianCaregiverInterface = Depends(get_clinician_interface),
):
    """Get all consent records for user review."""
    return {"records": interface.export_consent_records()}


@router.get("/clinician/access-log")
async def get_access_log(
    provider_id: str | None = None,
    interface: ClinicianCaregiverInterface = Depends(get_clinician_interface),
):
    """Get the access log."""
    return {"log": interface.get_access_log(provider_id=provider_id)}


# ============================================================================
# Safety Assessment Endpoints
# ============================================================================

@router.post("/safety/assess", response_model=SafetyAssessmentResponse)
async def assess_safety(
    request: SafetyAssessmentRequest,
    assessor: SafetyAssessor = Depends(get_safety_assessor),
):
    """
    Perform a comprehensive safety assessment.

    DISCLAIMER: This is not a clinical tool. If you are in crisis,
    please contact emergency services or a crisis hotline.
    """
    assessment = assessor.assess_safety(
        text=request.text,
        entropy_level=request.entropy_level,
    )

    return SafetyAssessmentResponse(
        risk_level=assessment.risk_level.value,
        crisis_types=[ct.value for ct in assessment.crisis_types],
        risk_factors=assessment.risk_factors,
        protective_factors=assessment.protective_factors,
        recommended_actions=assessment.recommended_actions,
        entropy_level=assessment.entropy_level,
    )


@router.get("/safety/resources")
async def get_crisis_resources(
    assessor: SafetyAssessor = Depends(get_safety_assessor),
):
    """Get crisis resources."""
    return assessor.get_crisis_resources()


@router.get("/safety/risk-trend")
async def get_risk_trend(
    assessor: SafetyAssessor = Depends(get_safety_assessor),
):
    """Get risk level trend over recent assessments."""
    return {"trend": assessor.get_risk_trend()}


# ============================================================================
# Grounding Techniques Endpoints
# ============================================================================

@router.post("/grounding/recommend", response_model=GroundingResponse)
async def recommend_grounding(
    request: GroundingRequest,
    library: GroundingTechniquesLibrary = Depends(get_grounding_library),
):
    """
    Get a recommended grounding technique based on current state.

    DISCLAIMER: These techniques are supportive tools, not treatment.
    """
    category = None
    if request.category_preference:
        try:
            category = GroundingCategory(request.category_preference)
        except ValueError:
            pass

    technique = library.recommend_technique(
        entropy_level=request.entropy_level,
        category_preference=category,
    )

    if technique is None:
        raise HTTPException(status_code=404, detail="No suitable technique found")

    return GroundingResponse(
        technique_id=technique.technique_id,
        name=technique.name,
        category=technique.category.value,
        intensity=technique.intensity.value,
        description=technique.description,
        instructions=technique.instructions,
        duration_minutes=technique.duration_minutes,
    )


@router.get("/grounding/list")
async def list_grounding_techniques(
    category: str | None = None,
    intensity: str | None = None,
    library: GroundingTechniquesLibrary = Depends(get_grounding_library),
):
    """List available grounding techniques."""
    cat = None
    if category:
        try:
            cat = GroundingCategory(category)
        except ValueError:
            pass

    intens = None
    if intensity:
        try:
            intens = IntensityLevel(intensity)
        except ValueError:
            pass

    techniques = library.list_techniques(category=cat, intensity=intens)

    return {
        "techniques": [
            {
                "technique_id": t.technique_id,
                "name": t.name,
                "category": t.category.value,
                "intensity": t.intensity.value,
                "duration_minutes": t.duration_minutes,
            }
            for t in techniques
        ]
    }


@router.get("/grounding/technique/{technique_id}")
async def get_grounding_technique(
    technique_id: str,
    library: GroundingTechniquesLibrary = Depends(get_grounding_library),
):
    """Get details of a specific grounding technique."""
    technique = library.get_technique(technique_id)

    if technique is None:
        raise HTTPException(status_code=404, detail="Technique not found")

    return {
        "technique_id": technique.technique_id,
        "name": technique.name,
        "category": technique.category.value,
        "intensity": technique.intensity.value,
        "description": technique.description,
        "instructions": technique.instructions,
        "duration_minutes": technique.duration_minutes,
        "adaptations": technique.adaptations,
        "contraindications": technique.contraindications,
    }


@router.post("/grounding/session/start")
async def start_grounding_session(
    request: GroundingSessionRequest,
    library: GroundingTechniquesLibrary = Depends(get_grounding_library),
):
    """Start a grounding session."""
    session = library.start_session(
        technique_id=request.technique_id,
        entropy_before=request.entropy_before,
    )

    if session is None:
        raise HTTPException(status_code=404, detail="Technique not found")

    return {
        "session_id": session.session_id,
        "technique_id": session.technique_id,
        "started_at": session.started_at,
    }


@router.post("/grounding/session/complete/{session_id}")
async def complete_grounding_session(
    session_id: str,
    entropy_after: float,
    effectiveness_rating: float | None = None,
    notes: str = "",
    library: GroundingTechniquesLibrary = Depends(get_grounding_library),
):
    """Complete a grounding session."""
    success = library.complete_session(
        session_id=session_id,
        entropy_after=entropy_after,
        effectiveness_rating=effectiveness_rating,
        notes=notes,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "completed", "session_id": session_id}


@router.get("/grounding/effectiveness")
async def get_grounding_effectiveness(
    library: GroundingTechniquesLibrary = Depends(get_grounding_library),
):
    """Get effectiveness summary for grounding techniques."""
    return library.get_effectiveness_summary()


# ============================================================================
# Free Energy Analysis Endpoints
# ============================================================================

@router.post("/free-energy/analyze")
async def analyze_free_energy(
    observation: list[float],
    precision: float = 1.0,
    minimizer: FreeEnergyMinimizer = Depends(get_free_energy_minimizer),
):
    """
    Analyze free energy for an observation.

    The Free Energy Principle provides a mathematical framework for
    understanding how the system minimizes surprise and maintains stability.
    """
    import numpy as np
    from reunity.core.free_energy import Observation

    obs = Observation(
        value=np.array(observation),
        precision=np.ones(len(observation)) * precision,
        timestamp=time.time(),
    )

    metrics = minimizer.calculate_variational_free_energy(obs)

    return {
        "variational_free_energy": metrics.variational_free_energy,
        "expected_free_energy": metrics.expected_free_energy,
        "surprise": metrics.surprise,
        "complexity": metrics.complexity,
        "accuracy": metrics.accuracy,
        "precision_weighted_error": metrics.precision_weighted_error,
    }


@router.post("/free-energy/update-beliefs")
async def update_beliefs(
    observation: list[float],
    precision: float = 1.0,
    minimizer: FreeEnergyMinimizer = Depends(get_free_energy_minimizer),
):
    """Update beliefs based on new observation."""
    import numpy as np
    from reunity.core.free_energy import Observation

    obs = Observation(
        value=np.array(observation),
        precision=np.ones(len(observation)) * precision,
        timestamp=time.time(),
    )

    beliefs = minimizer.update_beliefs(obs)

    return {
        "beliefs_mean": beliefs.mean.tolist(),
        "beliefs_precision": beliefs.precision.tolist(),
        "entropy": beliefs.entropy,
        "timestamp": beliefs.timestamp,
    }


@router.get("/free-energy/surprise-trajectory")
async def get_surprise_trajectory(
    minimizer: FreeEnergyMinimizer = Depends(get_free_energy_minimizer),
):
    """Get the history of surprise values."""
    return {"trajectory": minimizer.get_surprise_trajectory()}
