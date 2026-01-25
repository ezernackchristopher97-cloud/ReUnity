"""
ReUnity Safety Assessment Module

This module provides comprehensive safety assessment tools for detecting
crisis states, evaluating risk levels, and coordinating appropriate
responses. It integrates with entropy monitoring and protective pattern
recognition to provide holistic safety support.

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only. This system is NOT a substitute for professional
crisis intervention. If you or someone you know is in immediate danger,
please contact emergency services or a crisis hotline.

Crisis Resources:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- National Domestic Violence Hotline: 1-800-799-7233 (US)

Author: Christopher Ezernack
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(Enum):
    """Risk level classifications."""

    MINIMAL = "minimal"  # No significant risk indicators
    LOW = "low"  # Some risk factors present
    MODERATE = "moderate"  # Multiple risk factors
    HIGH = "high"  # Significant risk, intervention recommended
    CRITICAL = "critical"  # Immediate intervention needed


class CrisisType(Enum):
    """Types of crisis situations."""

    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    DISSOCIATIVE_CRISIS = "dissociative_crisis"
    PANIC_ATTACK = "panic_attack"
    FLASHBACK = "flashback"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    DOMESTIC_VIOLENCE = "domestic_violence"
    SUBSTANCE_CRISIS = "substance_crisis"
    EMOTIONAL_OVERWHELM = "emotional_overwhelm"


@dataclass
class SafetyIndicator:
    """An individual safety indicator."""

    indicator_type: str
    severity: float  # 0-1
    confidence: float  # 0-1
    evidence: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyAssessment:
    """Complete safety assessment result."""

    risk_level: RiskLevel
    crisis_types: list[CrisisType]
    indicators: list[SafetyIndicator]
    protective_factors: list[str]
    risk_factors: list[str]
    recommended_actions: list[str]
    entropy_level: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyPlan:
    """A personalized safety plan."""

    plan_id: str
    warning_signs: list[str]
    coping_strategies: list[str]
    safe_people: list[dict[str, str]]  # name, contact
    safe_places: list[str]
    professional_contacts: list[dict[str, str]]
    crisis_lines: list[dict[str, str]]
    reasons_for_living: list[str]
    environment_safety: list[str]
    created_at: float = field(default_factory=time.time)
    last_reviewed: float = field(default_factory=time.time)


# Crisis language patterns
SUICIDAL_PATTERNS = [
    r"(want|wish) to (die|end it|not exist|disappear)",
    r"(kill|hurt|harm) myself",
    r"(better off|world would be better) without me",
    r"no (reason|point) (to|in) (live|living|go on)",
    r"can'?t (take|do) (it|this) anymore",
    r"(end|ending) (my|this) (life|suffering|pain)",
    r"(suicide|suicidal)",
    r"(goodbye|farewell) (forever|for good)",
]

SELF_HARM_PATTERNS = [
    r"(cut|cutting|burn|burning) myself",
    r"(hurt|hurting|harm|harming) myself",
    r"(deserve|need) (to be|the) pain",
    r"(punish|punishing) myself",
    r"(scratch|scratching|hit|hitting) myself",
]

DISSOCIATION_PATTERNS = [
    r"(don'?t|can'?t) feel (real|anything|my body)",
    r"(feel|feeling) (disconnected|detached|unreal|numb)",
    r"(watching|outside) (myself|my body)",
    r"(lost|losing) (time|hours|days)",
    r"(don'?t know|can'?t remember) (who|where|what) (i am|i'?m)",
    r"(everything|world) (feels|seems|looks) (fake|unreal|strange)",
]

PANIC_PATTERNS = [
    r"(can'?t|cannot) breathe",
    r"(heart|chest) (racing|pounding|pain)",
    r"(going|gonna) (die|have a heart attack)",
    r"(losing|lose) (control|my mind)",
    r"(feel|feeling) (trapped|suffocating)",
]

DOMESTIC_VIOLENCE_PATTERNS = [
    r"(hit|hits|hitting|beat|beats|beating) me",
    r"(afraid|scared) (of|to leave)",
    r"(threatens?|threatening) (to kill|to hurt|me|my)",
    r"(won'?t let|doesn'?t let|can'?t) (me|leave|go)",
    r"(controls?|controlling) (everything|my|where)",
    r"(chokes?|choking|strangl)",
]


class SafetyAssessor:
    """
    Safety assessment system for crisis detection and response.

    This system monitors for crisis indicators and provides appropriate
    responses while respecting user autonomy. It is designed to support,
    not replace, professional crisis intervention.

    DISCLAIMER: This is not a clinical or treatment tool. If you are
    in crisis, please contact a mental health professional or crisis line.
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        enable_alerts: bool = True,
    ) -> None:
        """
        Initialize the safety assessor.

        Args:
            sensitivity: Detection sensitivity (0-1).
            enable_alerts: Whether to enable crisis alerts.
        """
        self.sensitivity = sensitivity
        self.enable_alerts = enable_alerts

        # Compile patterns
        self._suicidal_patterns = [
            re.compile(p, re.IGNORECASE) for p in SUICIDAL_PATTERNS
        ]
        self._self_harm_patterns = [
            re.compile(p, re.IGNORECASE) for p in SELF_HARM_PATTERNS
        ]
        self._dissociation_patterns = [
            re.compile(p, re.IGNORECASE) for p in DISSOCIATION_PATTERNS
        ]
        self._panic_patterns = [
            re.compile(p, re.IGNORECASE) for p in PANIC_PATTERNS
        ]
        self._dv_patterns = [
            re.compile(p, re.IGNORECASE) for p in DOMESTIC_VIOLENCE_PATTERNS
        ]

        # User's safety plan
        self._safety_plan: SafetyPlan | None = None

        # Assessment history
        self._assessment_history: list[SafetyAssessment] = []

    def assess_safety(
        self,
        text: str,
        entropy_level: float = 0.5,
        additional_context: dict[str, Any] | None = None,
    ) -> SafetyAssessment:
        """
        Perform a comprehensive safety assessment.

        Args:
            text: Text to analyze for safety concerns.
            entropy_level: Current entropy level from monitoring.
            additional_context: Additional context for assessment.

        Returns:
            SafetyAssessment with risk level and recommendations.
        """
        indicators = []
        crisis_types = []
        risk_factors = []
        protective_factors = []

        # Check for suicidal ideation
        suicidal_matches = self._check_patterns(text, self._suicidal_patterns)
        if suicidal_matches:
            severity = min(1.0, len(suicidal_matches) * 0.4)
            indicators.append(SafetyIndicator(
                indicator_type="suicidal_ideation",
                severity=severity,
                confidence=0.8,
                evidence=suicidal_matches,
            ))
            crisis_types.append(CrisisType.SUICIDAL_IDEATION)
            risk_factors.append("Expressions of suicidal ideation detected")

        # Check for self-harm
        self_harm_matches = self._check_patterns(text, self._self_harm_patterns)
        if self_harm_matches:
            severity = min(1.0, len(self_harm_matches) * 0.3)
            indicators.append(SafetyIndicator(
                indicator_type="self_harm",
                severity=severity,
                confidence=0.7,
                evidence=self_harm_matches,
            ))
            crisis_types.append(CrisisType.SELF_HARM)
            risk_factors.append("Self-harm language detected")

        # Check for dissociation
        dissociation_matches = self._check_patterns(text, self._dissociation_patterns)
        if dissociation_matches:
            severity = min(1.0, len(dissociation_matches) * 0.25)
            indicators.append(SafetyIndicator(
                indicator_type="dissociation",
                severity=severity,
                confidence=0.6,
                evidence=dissociation_matches,
            ))
            crisis_types.append(CrisisType.DISSOCIATIVE_CRISIS)
            risk_factors.append("Dissociative symptoms detected")

        # Check for panic
        panic_matches = self._check_patterns(text, self._panic_patterns)
        if panic_matches:
            severity = min(1.0, len(panic_matches) * 0.25)
            indicators.append(SafetyIndicator(
                indicator_type="panic",
                severity=severity,
                confidence=0.7,
                evidence=panic_matches,
            ))
            crisis_types.append(CrisisType.PANIC_ATTACK)
            risk_factors.append("Panic symptoms detected")

        # Check for domestic violence
        dv_matches = self._check_patterns(text, self._dv_patterns)
        if dv_matches:
            severity = min(1.0, len(dv_matches) * 0.4)
            indicators.append(SafetyIndicator(
                indicator_type="domestic_violence",
                severity=severity,
                confidence=0.8,
                evidence=dv_matches,
            ))
            crisis_types.append(CrisisType.DOMESTIC_VIOLENCE)
            risk_factors.append("Domestic violence indicators detected")

        # High entropy is a risk factor
        if entropy_level > 0.7:
            risk_factors.append("High emotional entropy")
        elif entropy_level < 0.3:
            protective_factors.append("Stable emotional state")

        # Check for protective factors in text
        protective_keywords = [
            "support", "help", "hope", "future", "plan", "safe",
            "therapy", "therapist", "friend", "family", "love"
        ]
        for keyword in protective_keywords:
            if keyword in text.lower():
                protective_factors.append(f"Mention of {keyword}")
                break

        # Calculate overall risk level
        risk_level = self._calculate_risk_level(
            indicators,
            entropy_level,
            len(protective_factors),
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level,
            crisis_types,
            protective_factors,
        )

        assessment = SafetyAssessment(
            risk_level=risk_level,
            crisis_types=crisis_types,
            indicators=indicators,
            protective_factors=protective_factors,
            risk_factors=risk_factors,
            recommended_actions=recommendations,
            entropy_level=entropy_level,
            metadata=additional_context or {},
        )

        self._assessment_history.append(assessment)

        return assessment

    def _check_patterns(
        self,
        text: str,
        patterns: list[re.Pattern],
    ) -> list[str]:
        """Check text against a list of patterns."""
        matches = []
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                matches.append(match.group())
        return matches

    def _calculate_risk_level(
        self,
        indicators: list[SafetyIndicator],
        entropy_level: float,
        protective_count: int,
    ) -> RiskLevel:
        """Calculate overall risk level."""
        if not indicators:
            if entropy_level > 0.8:
                return RiskLevel.LOW
            return RiskLevel.MINIMAL

        # Calculate weighted severity
        total_severity = sum(
            i.severity * i.confidence for i in indicators
        )
        avg_severity = total_severity / len(indicators)

        # Adjust for entropy
        adjusted_severity = avg_severity * (0.5 + entropy_level * 0.5)

        # Protective factors reduce risk
        protective_reduction = min(0.2, protective_count * 0.05)
        adjusted_severity -= protective_reduction

        # Check for critical indicators
        has_suicidal = any(
            i.indicator_type == "suicidal_ideation" and i.severity > 0.5
            for i in indicators
        )
        has_dv = any(
            i.indicator_type == "domestic_violence" and i.severity > 0.5
            for i in indicators
        )

        if has_suicidal or has_dv:
            if adjusted_severity > 0.6:
                return RiskLevel.CRITICAL
            return RiskLevel.HIGH

        if adjusted_severity > 0.7:
            return RiskLevel.HIGH
        elif adjusted_severity > 0.5:
            return RiskLevel.MODERATE
        elif adjusted_severity > 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        crisis_types: list[CrisisType],
        protective_factors: list[str],
    ) -> list[str]:
        """Generate safety recommendations."""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append(
                "IMPORTANT: Please reach out to a crisis line or emergency services. "
                "You don't have to face this alone."
            )
            recommendations.append(
                "National Suicide Prevention Lifeline: 988 (US)"
            )
            recommendations.append(
                "Crisis Text Line: Text HOME to 741741"
            )

        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append(
                "Consider reaching out to a trusted person right now."
            )
            if self._safety_plan:
                recommendations.append(
                    "Review your safety plan and use your coping strategies."
                )

        if CrisisType.DISSOCIATIVE_CRISIS in crisis_types:
            recommendations.append(
                "Grounding exercise: Name 5 things you can see, 4 you can touch, "
                "3 you can hear, 2 you can smell, 1 you can taste."
            )

        if CrisisType.PANIC_ATTACK in crisis_types:
            recommendations.append(
                "Try box breathing: Breathe in for 4 counts, hold for 4, "
                "out for 4, hold for 4. Repeat."
            )

        if CrisisType.DOMESTIC_VIOLENCE in crisis_types:
            recommendations.append(
                "National Domestic Violence Hotline: 1-800-799-7233"
            )
            recommendations.append(
                "Your safety matters. Help is available."
            )

        if risk_level == RiskLevel.MODERATE:
            recommendations.append(
                "Consider using a grounding technique or reaching out to support."
            )

        if not recommendations:
            recommendations.append(
                "Continue monitoring how you're feeling."
            )

        return recommendations

    def create_safety_plan(
        self,
        warning_signs: list[str],
        coping_strategies: list[str],
        safe_people: list[dict[str, str]],
        safe_places: list[str],
        professional_contacts: list[dict[str, str]],
        reasons_for_living: list[str],
        environment_safety: list[str] | None = None,
    ) -> SafetyPlan:
        """
        Create a personalized safety plan.

        Args:
            warning_signs: Personal warning signs of crisis.
            coping_strategies: Internal coping strategies.
            safe_people: People to contact for support.
            safe_places: Safe places to go.
            professional_contacts: Professional help contacts.
            reasons_for_living: Personal reasons for living.
            environment_safety: Steps to make environment safe.

        Returns:
            The created SafetyPlan.
        """
        plan = SafetyPlan(
            plan_id=f"sp_{int(time.time())}",
            warning_signs=warning_signs,
            coping_strategies=coping_strategies,
            safe_people=safe_people,
            safe_places=safe_places,
            professional_contacts=professional_contacts,
            crisis_lines=[
                {"name": "National Suicide Prevention Lifeline", "number": "988"},
                {"name": "Crisis Text Line", "number": "Text HOME to 741741"},
                {"name": "National Domestic Violence Hotline", "number": "1-800-799-7233"},
            ],
            reasons_for_living=reasons_for_living,
            environment_safety=environment_safety or [],
        )

        self._safety_plan = plan
        return plan

    def get_safety_plan(self) -> SafetyPlan | None:
        """Get the current safety plan."""
        return self._safety_plan

    def get_crisis_resources(self) -> dict[str, list[dict[str, str]]]:
        """
        Get crisis resources.

        Returns:
            Dictionary of crisis resources by category.
        """
        return {
            "suicide_prevention": [
                {"name": "National Suicide Prevention Lifeline", "number": "988", "country": "US"},
                {"name": "Crisis Text Line", "number": "Text HOME to 741741", "country": "US"},
                {"name": "International Association for Suicide Prevention",
                 "url": "https://www.iasp.info/resources/Crisis_Centres/", "country": "International"},
            ],
            "domestic_violence": [
                {"name": "National Domestic Violence Hotline", "number": "1-800-799-7233", "country": "US"},
                {"name": "National Dating Abuse Helpline", "number": "1-866-331-9474", "country": "US"},
            ],
            "mental_health": [
                {"name": "SAMHSA National Helpline", "number": "1-800-662-4357", "country": "US"},
                {"name": "NAMI Helpline", "number": "1-800-950-6264", "country": "US"},
            ],
            "disclaimer": (
                "This is not a clinical or treatment tool. These resources are "
                "provided for informational purposes. If you are in immediate "
                "danger, please contact emergency services (911 in the US)."
            ),
        }

    def get_assessment_history(
        self,
        limit: int = 50,
    ) -> list[SafetyAssessment]:
        """Get recent assessment history."""
        return self._assessment_history[-limit:]

    def get_risk_trend(self) -> str:
        """
        Analyze risk level trend over recent assessments.

        Returns:
            Trend description: "improving", "stable", "worsening", or "insufficient_data"
        """
        if len(self._assessment_history) < 3:
            return "insufficient_data"

        recent = self._assessment_history[-5:]

        risk_values = {
            RiskLevel.MINIMAL: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        values = [risk_values[a.risk_level] for a in recent]

        # Simple trend analysis
        if values[-1] < values[0]:
            return "improving"
        elif values[-1] > values[0]:
            return "worsening"
        else:
            return "stable"
