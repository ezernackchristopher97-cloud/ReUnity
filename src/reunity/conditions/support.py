"""
ReUnity Condition-Specific Support Module

Provides specialized support for various mental health conditions including
DID, PTSD, C-PTSD, BPD, Bipolar, and Schizophrenia/Schizoaffective disorders.

DISCLAIMER: ReUnity is NOT a clinical or treatment tool. It is a theoretical
and support framework only. This module does not diagnose or treat any condition.
Always work with qualified mental health professionals.

If you are in crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import time
import uuid


class ConditionType(Enum):
    """Supported condition types."""
    DID = "dissociative_identity_disorder"
    PTSD = "post_traumatic_stress_disorder"
    CPTSD = "complex_ptsd"
    BPD = "borderline_personality_disorder"
    BIPOLAR_I = "bipolar_i_disorder"
    BIPOLAR_II = "bipolar_ii_disorder"
    SCHIZOPHRENIA = "schizophrenia"
    SCHIZOAFFECTIVE = "schizoaffective_disorder"
    GENERAL = "general_support"


class SupportMode(Enum):
    """Support interaction modes."""
    GROUNDING = "grounding"
    CONTINUITY = "continuity"
    REALITY_TESTING = "reality_testing"
    DIALECTICAL = "dialectical"
    TRANSITION = "transition"
    PROTECTIVE = "protective"


@dataclass
class SupportContext:
    """Context for condition-specific support."""
    condition_type: ConditionType
    current_mode: SupportMode
    entropy_level: float
    stability_score: float
    recent_patterns: List[str] = field(default_factory=list)
    active_interventions: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupportResponse:
    """Response from support module."""
    message: str
    mode: SupportMode
    techniques: List[str]
    follow_up_prompts: List[str]
    grounding_exercises: List[str]
    safety_check: bool
    escalation_needed: bool
    resources: List[str] = field(default_factory=list)


class DIDSupport:
    """
    Dissociative Identity Disorder support.
    
    RIME links alters through tagged memory systems preserving
    continuity across identity switches.
    """
    
    def __init__(self):
        self.alter_profiles: Dict[str, Dict[str, Any]] = {}
        self.switch_history: List[Dict[str, Any]] = []
        self.shared_memories: List[str] = []
        self.communication_log: List[Dict[str, Any]] = []
    
    def register_alter(
        self,
        name: str,
        pronouns: str,
        role: str,
        preferences: Dict[str, Any]
    ) -> str:
        """Register an alter with personalized profile."""
        alter_id = str(uuid.uuid4())[:8]
        self.alter_profiles[alter_id] = {
            "name": name,
            "pronouns": pronouns,
            "role": role,
            "preferences": preferences,
            "created_at": time.time(),
            "interaction_count": 0,
        }
        return alter_id
    
    def recognize_alter(
        self,
        linguistic_patterns: List[str],
        emotional_markers: List[str],
        behavioral_indicators: List[str]
    ) -> Optional[str]:
        """
        Recognize current alter based on patterns.
        
        AAS_recognition = f(linguistic_patterns, emotional_markers, behavioral_indicators)
        """
        # Score each alter based on pattern matching
        scores = {}
        for alter_id, profile in self.alter_profiles.items():
            score = 0
            prefs = profile.get("preferences", {})
            
            # Check linguistic patterns
            if "speech_patterns" in prefs:
                for pattern in linguistic_patterns:
                    if pattern.lower() in str(prefs["speech_patterns"]).lower():
                        score += 1
            
            # Check emotional markers
            if "emotional_style" in prefs:
                for marker in emotional_markers:
                    if marker.lower() in str(prefs["emotional_style"]).lower():
                        score += 1
            
            scores[alter_id] = score
        
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0:
                return best_match
        
        return None
    
    def facilitate_communication(
        self,
        sender_id: str,
        recipient_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Facilitate inter-alter communication."""
        comm = {
            "id": str(uuid.uuid4())[:8],
            "sender": sender_id,
            "recipient": recipient_id,
            "message": message,
            "timestamp": time.time(),
            "read": False,
        }
        self.communication_log.append(comm)
        return comm
    
    def get_continuity_thread(self, alter_id: str) -> List[Dict[str, Any]]:
        """Get continuity thread for an alter."""
        return [
            comm for comm in self.communication_log
            if comm["sender"] == alter_id or comm["recipient"] == alter_id
        ]
    
    def support_co_consciousness(
        self,
        alter_ids: List[str],
        shared_context: str
    ) -> Dict[str, Any]:
        """Support co-consciousness development."""
        return {
            "participants": alter_ids,
            "shared_context": shared_context,
            "timestamp": time.time(),
            "techniques": [
                "Shared journaling exercise",
                "Internal meeting visualization",
                "Collaborative decision-making prompt",
            ],
        }


class PTSDSupport:
    """
    PTSD and Complex PTSD support.
    
    Predicts dissociation through entropy analysis and provides
    preemptive grounding before crisis states develop.
    """
    
    def __init__(self):
        self.entropy_history: List[float] = []
        self.grounding_sessions: List[Dict[str, Any]] = []
        self.trigger_patterns: List[str] = []
    
    def predict_dissociation(
        self,
        current_entropy: float,
        text_patterns: List[str],
        voice_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict dissociation risk through entropy analysis.
        
        Monitors emotional entropy via text and voice patterns.
        """
        self.entropy_history.append(current_entropy)
        
        # Calculate trend
        if len(self.entropy_history) >= 3:
            recent = self.entropy_history[-3:]
            trend = (recent[-1] - recent[0]) / 2
        else:
            trend = 0
        
        # Risk assessment
        risk_level = "low"
        if current_entropy > 0.7:
            risk_level = "high"
        elif current_entropy > 0.5:
            risk_level = "moderate"
        elif trend > 0.1:
            risk_level = "increasing"
        
        # Check for dissociation indicators in text
        dissociation_indicators = [
            "floating", "unreal", "watching myself",
            "numb", "disconnected", "foggy", "far away",
        ]
        indicator_count = sum(
            1 for pattern in text_patterns
            if any(ind in pattern.lower() for ind in dissociation_indicators)
        )
        
        return {
            "risk_level": risk_level,
            "current_entropy": current_entropy,
            "trend": trend,
            "indicator_count": indicator_count,
            "preemptive_grounding_recommended": risk_level in ["moderate", "high", "increasing"],
        }
    
    def get_grounding_intervention(
        self,
        risk_level: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get appropriate grounding intervention."""
        interventions = {
            "low": {
                "type": "maintenance",
                "exercises": [
                    "5-4-3-2-1 sensory awareness",
                    "Deep breathing (4-7-8 pattern)",
                ],
                "duration_minutes": 5,
            },
            "moderate": {
                "type": "active_grounding",
                "exercises": [
                    "Cold water on wrists",
                    "Strong scent (peppermint, citrus)",
                    "Physical movement (walking, stretching)",
                    "Name 5 things you can see right now",
                ],
                "duration_minutes": 10,
            },
            "high": {
                "type": "intensive_grounding",
                "exercises": [
                    "Ice cube in hand",
                    "Feet firmly on ground - feel the floor",
                    "Say your name and today's date out loud",
                    "Describe your immediate surroundings in detail",
                    "Hold a familiar object",
                ],
                "duration_minutes": 15,
                "follow_up_required": True,
            },
            "increasing": {
                "type": "preventive",
                "exercises": [
                    "Body scan meditation",
                    "Progressive muscle relaxation",
                    "Safe place visualization",
                ],
                "duration_minutes": 10,
            },
        }
        
        return interventions.get(risk_level, interventions["low"])
    
    def reality_anchoring(self) -> List[str]:
        """Provide reality anchoring prompts."""
        return [
            "What is today's date?",
            "Where are you right now?",
            "Name three things you can touch right now.",
            "What did you have for your last meal?",
            "Who is someone safe you could call?",
        ]


class BPDSupport:
    """
    Borderline Personality Disorder support.
    
    Reflects contradictions in emotional states and relationships
    without invalidation. Supports dialectical thinking.
    """
    
    def __init__(self):
        self.emotional_history: List[Dict[str, Any]] = []
        self.relationship_threads: Dict[str, List[Dict[str, Any]]] = {}
        self.dialectical_exercises: List[Dict[str, Any]] = []
    
    def reflect_contradiction(
        self,
        current_feeling: str,
        past_feeling: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Reflect contradictions without invalidation.
        
        Allows multiple truths to coexist.
        """
        reflection = {
            "current": current_feeling,
            "past": past_feeling,
            "context": context,
            "timestamp": time.time(),
        }
        
        # Generate dialectical reflection
        dialectical_response = (
            f"You're feeling {current_feeling} now, and you also felt "
            f"{past_feeling} before. Both of these feelings are real and valid. "
            f"Can both be true at the same time?"
        )
        
        follow_ups = [
            "What might explain this difference?",
            "What do you notice in your body right now?",
            "Is there a 'both/and' here instead of 'either/or'?",
        ]
        
        return {
            "reflection": dialectical_response,
            "follow_ups": follow_ups,
            "validation": "Your feelings are valid, even when they seem contradictory.",
            "dialectical_principle": "Two opposite things can both be true.",
        }
    
    def preserve_relationship_thread(
        self,
        relationship_id: str,
        event: str,
        emotional_state: str
    ) -> None:
        """Preserve relationship context during splitting episodes."""
        if relationship_id not in self.relationship_threads:
            self.relationship_threads[relationship_id] = []
        
        self.relationship_threads[relationship_id].append({
            "event": event,
            "emotional_state": emotional_state,
            "timestamp": time.time(),
        })
    
    def get_relationship_continuity(
        self,
        relationship_id: str
    ) -> Dict[str, Any]:
        """Get full relationship thread for continuity."""
        thread = self.relationship_threads.get(relationship_id, [])
        
        if not thread:
            return {"message": "No history recorded for this relationship."}
        
        # Analyze emotional patterns
        emotions = [e["emotional_state"] for e in thread]
        
        return {
            "thread": thread,
            "total_interactions": len(thread),
            "emotional_range": list(set(emotions)),
            "continuity_message": (
                "This relationship has had many moments. "
                "Here's what you've recorded over time."
            ),
        }
    
    def dialectical_thinking_exercise(
        self,
        black_thought: str,
        white_thought: str
    ) -> Dict[str, Any]:
        """Guide dialectical thinking exercise."""
        return {
            "thesis": black_thought,
            "antithesis": white_thought,
            "synthesis_prompts": [
                f"What if both '{black_thought}' AND '{white_thought}' contain some truth?",
                "What's the middle ground between these two views?",
                "How might someone who loves you see this situation?",
            ],
            "dbt_skill": "Walking the Middle Path",
            "reminder": "Extreme thinking often misses important nuances.",
        }


class BipolarSupport:
    """
    Bipolar I and II Disorder support.
    
    Preserves continuity during manic and depressive transitions.
    Tracks patterns preceding mood episodes.
    """
    
    def __init__(self):
        self.mood_log: List[Dict[str, Any]] = []
        self.episode_patterns: List[Dict[str, Any]] = []
        self.early_warning_signs: Dict[str, List[str]] = {
            "mania": [],
            "depression": [],
            "hypomania": [],
        }
    
    def log_mood(
        self,
        mood_score: float,  # -1 (depressed) to 1 (manic)
        sleep_hours: float,
        energy_level: float,
        notes: str
    ) -> Dict[str, Any]:
        """Log mood data point."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "mood_score": mood_score,
            "sleep_hours": sleep_hours,
            "energy_level": energy_level,
            "notes": notes,
            "timestamp": time.time(),
        }
        self.mood_log.append(entry)
        
        # Check for warning signs
        warnings = self._check_warning_signs(entry)
        
        return {
            "logged": entry,
            "warnings": warnings,
        }
    
    def _check_warning_signs(self, entry: Dict[str, Any]) -> List[str]:
        """Check for early warning signs of episode."""
        warnings = []
        
        # Mania indicators
        if entry["sleep_hours"] < 4 and entry["energy_level"] > 0.8:
            warnings.append("Reduced sleep with high energy - possible mania warning")
        if entry["mood_score"] > 0.7:
            warnings.append("Elevated mood detected")
        
        # Depression indicators
        if entry["sleep_hours"] > 10 and entry["energy_level"] < 0.3:
            warnings.append("Increased sleep with low energy - possible depression warning")
        if entry["mood_score"] < -0.7:
            warnings.append("Low mood detected")
        
        return warnings
    
    def get_transition_support(
        self,
        from_state: str,
        to_state: str
    ) -> Dict[str, Any]:
        """Get support for mood state transitions."""
        transitions = {
            ("manic", "depressed"): {
                "message": "Transitioning from high energy to lower mood is common. Be gentle with yourself.",
                "strategies": [
                    "Maintain regular sleep schedule",
                    "Avoid major decisions during transition",
                    "Reach out to support system",
                    "Continue medication as prescribed",
                ],
                "continuity_reminder": "You've navigated transitions before. Your identity remains constant.",
            },
            ("depressed", "manic"): {
                "message": "Energy returning can feel good but watch for escalation.",
                "strategies": [
                    "Monitor sleep carefully",
                    "Avoid stimulants",
                    "Check in with treatment team",
                    "Pace activities",
                ],
                "continuity_reminder": "This energy is part of you, but so is balance.",
            },
        }
        
        key = (from_state.lower(), to_state.lower())
        return transitions.get(key, {
            "message": "Mood changes are part of your experience.",
            "strategies": ["Maintain routines", "Stay connected to support"],
            "continuity_reminder": "You are more than your mood states.",
        })
    
    def pattern_analysis(self) -> Dict[str, Any]:
        """Analyze mood patterns for early warning."""
        if len(self.mood_log) < 7:
            return {"message": "Need more data for pattern analysis."}
        
        recent = self.mood_log[-7:]
        avg_mood = sum(e["mood_score"] for e in recent) / len(recent)
        avg_sleep = sum(e["sleep_hours"] for e in recent) / len(recent)
        
        trend = "stable"
        if recent[-1]["mood_score"] - recent[0]["mood_score"] > 0.3:
            trend = "increasing"
        elif recent[-1]["mood_score"] - recent[0]["mood_score"] < -0.3:
            trend = "decreasing"
        
        return {
            "average_mood": avg_mood,
            "average_sleep": avg_sleep,
            "trend": trend,
            "recommendation": self._get_trend_recommendation(trend, avg_mood),
        }
    
    def _get_trend_recommendation(self, trend: str, avg_mood: float) -> str:
        """Get recommendation based on trend."""
        if trend == "increasing" and avg_mood > 0.3:
            return "Mood trending up. Consider checking in with your treatment team."
        elif trend == "decreasing" and avg_mood < -0.3:
            return "Mood trending down. Prioritize self-care and support."
        return "Mood appears stable. Continue current routines."


class SchizophreniaSupport:
    """
    Schizophrenia and Schizoaffective Disorder support.
    
    MirrorLink differentiates reality from projection through
    pattern recognition algorithms.
    """
    
    def __init__(self):
        self.reality_checks: List[Dict[str, Any]] = []
        self.consistency_log: List[Dict[str, Any]] = []
        self.lyapunov_history: List[float] = []
    
    def reality_testing(
        self,
        perception: str,
        context: str,
        time_of_occurrence: float
    ) -> Dict[str, Any]:
        """
        Differentiate reality from projection.
        
        Uses pattern recognition tracking consistency across time and context.
        """
        check = {
            "id": str(uuid.uuid4())[:8],
            "perception": perception,
            "context": context,
            "time": time_of_occurrence,
            "timestamp": time.time(),
        }
        self.reality_checks.append(check)
        
        # Check consistency with previous perceptions
        consistency = self._check_consistency(perception, context)
        
        # Generate non-invalidating reflection
        reflection = self._generate_reflection(perception, consistency)
        
        return {
            "check": check,
            "consistency_score": consistency["score"],
            "reflection": reflection,
            "grounding_suggested": consistency["score"] < 0.5,
        }
    
    def _check_consistency(
        self,
        perception: str,
        context: str
    ) -> Dict[str, Any]:
        """Check consistency of perception across time and context."""
        if len(self.reality_checks) < 2:
            return {"score": 0.5, "message": "Building baseline."}
        
        # Simple consistency check based on similar contexts
        similar_contexts = [
            r for r in self.reality_checks[-10:]
            if context.lower() in r["context"].lower()
        ]
        
        if not similar_contexts:
            return {"score": 0.5, "message": "New context, no comparison available."}
        
        # Check if perception is consistent
        similar_perceptions = [
            r for r in similar_contexts
            if perception.lower() in r["perception"].lower()
        ]
        
        score = len(similar_perceptions) / len(similar_contexts)
        
        return {
            "score": score,
            "message": f"This perception has appeared {len(similar_perceptions)} times in similar contexts.",
        }
    
    def _generate_reflection(
        self,
        perception: str,
        consistency: Dict[str, Any]
    ) -> str:
        """Generate non-invalidating reflection."""
        if consistency["score"] > 0.7:
            return (
                f"You've noticed '{perception}' consistently. "
                "Let's explore what this experience means to you."
            )
        elif consistency["score"] > 0.3:
            return (
                f"You're experiencing '{perception}'. "
                "This is sometimes present and sometimes not. "
                "What do you notice about when it appears?"
            )
        else:
            return (
                f"You're noticing '{perception}' right now. "
                "This seems different from your usual experience. "
                "Would a grounding exercise be helpful?"
            )
    
    def predict_episode_onset(
        self,
        lyapunov_exponent: float
    ) -> Dict[str, Any]:
        """
        Predict episode onset using Lyapunov exponent analysis.
        
        λ > 0: instability (potential episode)
        λ < 0: stability
        λ ≈ 0: marginal
        """
        self.lyapunov_history.append(lyapunov_exponent)
        
        if lyapunov_exponent > 0.5:
            risk = "high"
            message = "System showing instability. Consider reaching out to support."
        elif lyapunov_exponent > 0:
            risk = "moderate"
            message = "Some instability detected. Grounding recommended."
        elif lyapunov_exponent > -0.5:
            risk = "low"
            message = "System relatively stable."
        else:
            risk = "very_low"
            message = "Good stability. Continue current approach."
        
        return {
            "lyapunov": lyapunov_exponent,
            "risk_level": risk,
            "message": message,
            "trend": self._calculate_trend(),
        }
    
    def _calculate_trend(self) -> str:
        """Calculate stability trend."""
        if len(self.lyapunov_history) < 3:
            return "insufficient_data"
        
        recent = self.lyapunov_history[-3:]
        if recent[-1] > recent[0] + 0.2:
            return "destabilizing"
        elif recent[-1] < recent[0] - 0.2:
            return "stabilizing"
        return "stable"


class ConditionSupportManager:
    """
    Manager for condition-specific support.
    
    Coordinates support across different conditions and modes.
    """
    
    def __init__(self):
        self.did_support = DIDSupport()
        self.ptsd_support = PTSDSupport()
        self.bpd_support = BPDSupport()
        self.bipolar_support = BipolarSupport()
        self.schizophrenia_support = SchizophreniaSupport()
    
    def get_support(
        self,
        condition: ConditionType,
        context: SupportContext
    ) -> SupportResponse:
        """Get appropriate support based on condition and context."""
        handlers = {
            ConditionType.DID: self._handle_did,
            ConditionType.PTSD: self._handle_ptsd,
            ConditionType.CPTSD: self._handle_ptsd,  # Similar handling
            ConditionType.BPD: self._handle_bpd,
            ConditionType.BIPOLAR_I: self._handle_bipolar,
            ConditionType.BIPOLAR_II: self._handle_bipolar,
            ConditionType.SCHIZOPHRENIA: self._handle_schizophrenia,
            ConditionType.SCHIZOAFFECTIVE: self._handle_schizophrenia,
            ConditionType.GENERAL: self._handle_general,
        }
        
        handler = handlers.get(condition, self._handle_general)
        return handler(context)
    
    def _handle_did(self, context: SupportContext) -> SupportResponse:
        """Handle DID support request."""
        return SupportResponse(
            message="Supporting system continuity and communication.",
            mode=SupportMode.CONTINUITY,
            techniques=["Internal communication", "Shared journaling", "Co-consciousness development"],
            follow_up_prompts=["Who is present right now?", "Would you like to leave a message for others in the system?"],
            grounding_exercises=["5-4-3-2-1 sensory grounding"],
            safety_check=context.entropy_level > 0.7,
            escalation_needed=context.entropy_level > 0.85,
        )
    
    def _handle_ptsd(self, context: SupportContext) -> SupportResponse:
        """Handle PTSD/C-PTSD support request."""
        prediction = self.ptsd_support.predict_dissociation(
            context.entropy_level,
            context.recent_patterns,
        )
        
        intervention = self.ptsd_support.get_grounding_intervention(
            prediction["risk_level"]
        )
        
        return SupportResponse(
            message=f"Dissociation risk: {prediction['risk_level']}. Grounding support available.",
            mode=SupportMode.GROUNDING,
            techniques=intervention["exercises"],
            follow_up_prompts=self.ptsd_support.reality_anchoring(),
            grounding_exercises=intervention["exercises"],
            safety_check=prediction["risk_level"] in ["moderate", "high"],
            escalation_needed=prediction["risk_level"] == "high",
        )
    
    def _handle_bpd(self, context: SupportContext) -> SupportResponse:
        """Handle BPD support request."""
        return SupportResponse(
            message="Supporting emotional continuity and dialectical thinking.",
            mode=SupportMode.DIALECTICAL,
            techniques=["Dialectical thinking", "Relationship thread review", "Both/and perspective"],
            follow_up_prompts=["What are you feeling right now?", "Can both feelings be true?"],
            grounding_exercises=["TIPP skills", "Opposite action"],
            safety_check=context.entropy_level > 0.7,
            escalation_needed=context.entropy_level > 0.85,
        )
    
    def _handle_bipolar(self, context: SupportContext) -> SupportResponse:
        """Handle Bipolar support request."""
        return SupportResponse(
            message="Supporting mood continuity and pattern awareness.",
            mode=SupportMode.TRANSITION,
            techniques=["Mood logging", "Sleep tracking", "Pattern recognition"],
            follow_up_prompts=["How many hours did you sleep?", "What's your energy level?"],
            grounding_exercises=["Routine maintenance", "Sleep hygiene"],
            safety_check=context.entropy_level > 0.6,
            escalation_needed=context.entropy_level > 0.8,
        )
    
    def _handle_schizophrenia(self, context: SupportContext) -> SupportResponse:
        """Handle Schizophrenia/Schizoaffective support request."""
        return SupportResponse(
            message="Supporting reality testing and stability.",
            mode=SupportMode.REALITY_TESTING,
            techniques=["Consistency checking", "Grounding", "Pattern tracking"],
            follow_up_prompts=["What are you noticing right now?", "Has this happened before?"],
            grounding_exercises=["5-4-3-2-1 grounding", "Name objects in the room"],
            safety_check=True,
            escalation_needed=context.entropy_level > 0.75,
        )
    
    def _handle_general(self, context: SupportContext) -> SupportResponse:
        """Handle general support request."""
        return SupportResponse(
            message="General support and grounding available.",
            mode=SupportMode.GROUNDING,
            techniques=["Breathing exercises", "Mindfulness", "Journaling"],
            follow_up_prompts=["How are you feeling?", "What brought you here today?"],
            grounding_exercises=["Deep breathing", "Body scan"],
            safety_check=context.entropy_level > 0.7,
            escalation_needed=context.entropy_level > 0.85,
            resources=[
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
            ],
        )
