"""
ReUnity Protective Logic Module (PLM)

This module implements sophisticated pattern recognition algorithms to identify
potentially harmful relationship dynamics, gaslighting attempts, manipulation
tactics, and other destabilizing interaction patterns that may not be immediately
apparent to users experiencing emotional dysregulation.

The module provides gentle warnings and reality-checking support without
invalidating the user's emotional experience, recognizing that survivors are
the experts on their own safety and circumstances.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PatternType(Enum):
    """Types of potentially harmful patterns."""

    HOT_COLD_CYCLE = "hot_cold_cycle"
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    ABANDONMENT_THREAT = "abandonment_threat"
    ISOLATION_ATTEMPT = "isolation_attempt"
    FINANCIAL_CONTROL = "financial_control"
    REALITY_CONTRADICTION = "reality_contradiction"
    EMOTIONAL_BAITING = "emotional_baiting"
    INVALIDATION = "invalidation"
    BLAME_SHIFTING = "blame_shifting"
    TRIANGULATION = "triangulation"
    SILENT_TREATMENT = "silent_treatment"
    BOUNDARY_VIOLATION = "boundary_violation"


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectedPattern:
    """Container for a detected harmful pattern."""

    pattern_type: PatternType
    severity: PatternSeverity
    confidence: float
    evidence: list[str]
    message: str
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionAnalysis:
    """Analysis results for a set of interactions."""

    patterns_detected: list[DetectedPattern]
    overall_risk: float
    sentiment_variance: float
    contradiction_score: float
    stability_assessment: str
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipContext:
    """Context for analyzing relationship patterns."""

    person_id: str
    person_name: str
    relationship_type: str
    interaction_history: list[dict[str, Any]]
    sentiment_history: list[float]
    pattern_history: list[DetectedPattern]
    trust_score: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)


# Sentiment word lists for basic analysis
POSITIVE_WORDS = frozenset([
    "good", "love", "happy", "safe", "calm", "anchor", "trust", "support",
    "kind", "caring", "gentle", "understanding", "patient", "reliable",
    "stable", "consistent", "warm", "comfort", "peace", "joy", "grateful",
    "appreciated", "valued", "respected", "heard", "seen", "accepted",
])

NEGATIVE_WORDS = frozenset([
    "bad", "hate", "angry", "unsafe", "scared", "betrayed", "hurt", "pain",
    "cruel", "mean", "harsh", "dismissive", "impatient", "unreliable",
    "unstable", "inconsistent", "cold", "fear", "anxiety", "shame", "guilt",
    "worthless", "invisible", "rejected", "abandoned", "alone", "trapped",
])

# Pattern indicators
GASLIGHTING_PHRASES = [
    r"you'?re (crazy|insane|imagining|overreacting|too sensitive)",
    r"that (never|didn'?t) happen",
    r"you'?re making (that|this|it) up",
    r"no one (else|will) (believes?|love|understand) you",
    r"you'?re (remembering|misremembering) (it|that|things) wrong",
    r"i never said that",
    r"you'?re (being|so) (dramatic|paranoid)",
    r"everyone (thinks|knows) you'?re",
    r"you should (be grateful|thank me)",
]

LOVE_BOMBING_INDICATORS = [
    r"you'?re (the only one|my everything|perfect)",
    r"i'?ve never (felt|loved|met) (anyone|someone) like",
    r"we'?re (meant to be|soulmates|destined)",
    r"i can'?t live without you",
    r"you complete me",
    r"no one (will ever|could) love you like i do",
]

ISOLATION_PHRASES = [
    r"(they|your friends|your family) (don'?t|doesn'?t) (understand|care|love) you",
    r"you don'?t need (them|anyone else)",
    r"i'?m the only one who",
    r"(they'?re|everyone is) (against|jealous of) (you|us)",
    r"you spend too much time with",
]


class ProtectivePatternRecognizer:
    """
    Protective Logic Module (PLM) for detecting harmful relationship patterns.

    This module implements pattern recognition to identify potentially harmful
    dynamics including gaslighting, hot-cold cycles, abandonment threats, and
    other manipulation tactics. It provides gentle warnings and reality-checking
    support without invalidating the user's emotional experience.

    The module recognizes that survivors are the experts on their own safety
    and provides information and perspective rather than making decisions
    for users.

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        history_window: int = 20,
        variance_threshold: float = 0.5,
        contradiction_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the protective pattern recognizer.

        Args:
            sensitivity: Detection sensitivity (0-1, higher = more sensitive).
            history_window: Number of interactions to analyze.
            variance_threshold: Threshold for flagging high sentiment variance.
            contradiction_threshold: Threshold for reality contradiction detection.
        """
        self.sensitivity = sensitivity
        self.history_window = history_window
        self.variance_threshold = variance_threshold
        self.contradiction_threshold = contradiction_threshold

        # Compile regex patterns
        self._gaslighting_patterns = [
            re.compile(p, re.IGNORECASE) for p in GASLIGHTING_PHRASES
        ]
        self._love_bombing_patterns = [
            re.compile(p, re.IGNORECASE) for p in LOVE_BOMBING_INDICATORS
        ]
        self._isolation_patterns = [
            re.compile(p, re.IGNORECASE) for p in ISOLATION_PHRASES
        ]

        # Relationship contexts
        self._contexts: dict[str, RelationshipContext] = {}

    def analyze_interactions(
        self,
        interactions: list[dict[str, Any]],
        person_id: str | None = None,
    ) -> InteractionAnalysis:
        """
        Analyze a set of interactions for harmful patterns.

        Args:
            interactions: List of interaction records with 'text' and 'timestamp'.
            person_id: Optional person identifier for context tracking.

        Returns:
            InteractionAnalysis with detected patterns and recommendations.
        """
        if not interactions:
            return InteractionAnalysis(
                patterns_detected=[],
                overall_risk=0.0,
                sentiment_variance=0.0,
                contradiction_score=0.0,
                stability_assessment="insufficient_data",
                recommendations=["More interaction data needed for analysis."],
            )

        # Extract text content
        texts = [i.get("text", "") for i in interactions]

        # Analyze sentiments
        sentiments = [self._analyze_sentiment(t) for t in texts]
        sentiment_variance = float(np.std(sentiments)) if len(sentiments) > 1 else 0.0

        # Detect patterns
        patterns = []

        # Check for hot-cold cycles
        hot_cold = self._detect_hot_cold_cycle(sentiments)
        if hot_cold:
            patterns.append(hot_cold)

        # Check for gaslighting
        for text in texts:
            gaslighting = self._detect_gaslighting(text)
            if gaslighting:
                patterns.append(gaslighting)

        # Check for love bombing
        love_bombing = self._detect_love_bombing(texts)
        if love_bombing:
            patterns.append(love_bombing)

        # Check for isolation attempts
        for text in texts:
            isolation = self._detect_isolation_attempt(text)
            if isolation:
                patterns.append(isolation)

        # Check for reality contradictions
        contradiction_score = self._detect_contradictions(interactions)

        if contradiction_score > self.contradiction_threshold:
            patterns.append(DetectedPattern(
                pattern_type=PatternType.REALITY_CONTRADICTION,
                severity=self._score_to_severity(contradiction_score),
                confidence=contradiction_score,
                evidence=["Contradictory statements detected in interaction history"],
                message="Reality contradictions detected; trust your memory threads.",
                recommendation="Review your memory records for this relationship.",
            ))

        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(patterns, sentiment_variance)

        # Generate stability assessment
        stability = self._assess_stability(sentiment_variance, len(patterns))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            patterns,
            sentiment_variance,
            overall_risk,
        )

        # Update context if person_id provided
        if person_id:
            self._update_context(person_id, interactions, sentiments, patterns)

        return InteractionAnalysis(
            patterns_detected=patterns,
            overall_risk=overall_risk,
            sentiment_variance=sentiment_variance,
            contradiction_score=contradiction_score,
            stability_assessment=stability,
            recommendations=recommendations,
            metadata={
                "interaction_count": len(interactions),
                "pattern_count": len(patterns),
            },
        )

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text (simplified implementation).

        Returns value from -1 (negative) to 1 (positive).
        """
        if not text:
            return 0.0

        words = text.lower().split()
        positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
        negative_count = sum(1 for w in words if w in NEGATIVE_WORDS)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _detect_hot_cold_cycle(
        self,
        sentiments: list[float],
    ) -> DetectedPattern | None:
        """
        Detect hot-cold cycling patterns in sentiment history.

        Hot-cold cycles indicate inconsistent behavior that can be
        destabilizing for trauma survivors.
        """
        if len(sentiments) < 4:
            return None

        # Calculate variance
        variance = np.std(sentiments)

        if variance <= self.variance_threshold:
            return None

        # Check for oscillation pattern
        changes = np.diff(sentiments)
        sign_changes = np.sum(np.diff(np.sign(changes)) != 0)

        # High variance with frequent sign changes indicates hot-cold
        oscillation_score = sign_changes / (len(changes) - 1) if len(changes) > 1 else 0

        if oscillation_score > 0.5 and variance > self.variance_threshold:
            severity = self._score_to_severity(
                (variance + oscillation_score) / 2 * self.sensitivity
            )

            return DetectedPattern(
                pattern_type=PatternType.HOT_COLD_CYCLE,
                severity=severity,
                confidence=float((variance + oscillation_score) / 2),
                evidence=[
                    f"Sentiment variance: {variance:.2f}",
                    f"Oscillation frequency: {oscillation_score:.2f}",
                ],
                message="Potential hot-cold cycling detected; reflect on past anchors.",
                recommendation="Consider reviewing your memory threads for this "
                "relationship to see patterns over time.",
            )

        return None

    def _detect_gaslighting(self, text: str) -> DetectedPattern | None:
        """
        Detect gaslighting language patterns.

        Gaslighting involves making someone question their own reality,
        memory, or perceptions.
        """
        if not text:
            return None

        matches = []
        for pattern in self._gaslighting_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)

        if not matches:
            return None

        confidence = min(1.0, len(matches) * 0.3 * self.sensitivity)
        severity = self._score_to_severity(confidence)

        return DetectedPattern(
            pattern_type=PatternType.GASLIGHTING,
            severity=severity,
            confidence=confidence,
            evidence=matches,
            message="Reality contradictions detected; trust your memory threads.",
            recommendation="Your memories and perceptions are valid. Consider "
            "documenting your experiences in your journal.",
        )

    def _detect_love_bombing(
        self,
        texts: list[str],
    ) -> DetectedPattern | None:
        """
        Detect love bombing patterns.

        Love bombing involves overwhelming someone with affection and
        attention, often early in a relationship or after conflict.
        """
        matches = []
        for text in texts:
            for pattern in self._love_bombing_patterns:
                if pattern.search(text):
                    matches.append(pattern.pattern)

        if len(matches) < 2:
            return None

        confidence = min(1.0, len(matches) * 0.2 * self.sensitivity)
        severity = self._score_to_severity(confidence)

        return DetectedPattern(
            pattern_type=PatternType.LOVE_BOMBING,
            severity=severity,
            confidence=confidence,
            evidence=matches[:5],  # Limit evidence list
            message="Intense affection patterns detected; healthy love builds gradually.",
            recommendation="Consider the pace and consistency of this relationship "
            "over time, not just intense moments.",
        )

    def _detect_isolation_attempt(self, text: str) -> DetectedPattern | None:
        """
        Detect isolation attempts.

        Isolation involves separating someone from their support network.
        """
        if not text:
            return None

        matches = []
        for pattern in self._isolation_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)

        if not matches:
            return None

        confidence = min(1.0, len(matches) * 0.4 * self.sensitivity)
        severity = self._score_to_severity(confidence)

        return DetectedPattern(
            pattern_type=PatternType.ISOLATION_ATTEMPT,
            severity=severity,
            confidence=confidence,
            evidence=matches,
            message="Isolation language detected; your support network matters.",
            recommendation="Maintaining connections with trusted friends and family "
            "is important for your wellbeing.",
        )

    def _detect_contradictions(
        self,
        interactions: list[dict[str, Any]],
    ) -> float:
        """
        Detect contradictions between statements in interactions.

        Returns a contradiction score from 0 to 1.
        """
        if len(interactions) < 2:
            return 0.0

        # Simplified contradiction detection based on sentiment shifts
        # about the same topics
        texts = [i.get("text", "") for i in interactions]
        sentiments = [self._analyze_sentiment(t) for t in texts]

        # Look for dramatic sentiment reversals
        max_shift = 0.0
        for i in range(len(sentiments) - 1):
            shift = abs(sentiments[i + 1] - sentiments[i])
            max_shift = max(max_shift, shift)

        # Normalize to 0-1 range
        return min(1.0, max_shift / 2.0)

    def _calculate_overall_risk(
        self,
        patterns: list[DetectedPattern],
        sentiment_variance: float,
    ) -> float:
        """Calculate overall relationship risk score."""
        if not patterns:
            return sentiment_variance * 0.3

        # Weight patterns by severity
        severity_weights = {
            PatternSeverity.LOW: 0.2,
            PatternSeverity.MODERATE: 0.4,
            PatternSeverity.HIGH: 0.7,
            PatternSeverity.CRITICAL: 1.0,
        }

        pattern_risk = sum(
            severity_weights[p.severity] * p.confidence
            for p in patterns
        ) / len(patterns)

        # Combine with sentiment variance
        overall = (pattern_risk * 0.7) + (sentiment_variance * 0.3)

        return min(1.0, overall)

    def _assess_stability(
        self,
        sentiment_variance: float,
        pattern_count: int,
    ) -> str:
        """Assess relationship stability."""
        if pattern_count == 0 and sentiment_variance < 0.3:
            return "stable"
        elif pattern_count <= 1 and sentiment_variance < 0.5:
            return "mostly_stable"
        elif pattern_count <= 2 or sentiment_variance < 0.7:
            return "variable"
        else:
            return "unstable"

    def _score_to_severity(self, score: float) -> PatternSeverity:
        """Convert a score to severity level."""
        if score < 0.3:
            return PatternSeverity.LOW
        elif score < 0.5:
            return PatternSeverity.MODERATE
        elif score < 0.7:
            return PatternSeverity.HIGH
        else:
            return PatternSeverity.CRITICAL

    def _generate_recommendations(
        self,
        patterns: list[DetectedPattern],
        sentiment_variance: float,
        overall_risk: float,
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if overall_risk > 0.7:
            recommendations.append(
                "Consider discussing these patterns with a trusted person or therapist."
            )

        if sentiment_variance > self.variance_threshold:
            recommendations.append(
                "High emotional variability detected. Grounding exercises may help."
            )

        pattern_types = {p.pattern_type for p in patterns}

        if PatternType.GASLIGHTING in pattern_types:
            recommendations.append(
                "Document your experiences and memories to maintain your sense of reality."
            )

        if PatternType.ISOLATION_ATTEMPT in pattern_types:
            recommendations.append(
                "Maintain connections with your support network."
            )

        if PatternType.HOT_COLD_CYCLE in pattern_types:
            recommendations.append(
                "Consistent, predictable relationships support healing."
            )

        if not recommendations:
            recommendations.append(
                "Continue monitoring relationship patterns over time."
            )

        return recommendations

    def _update_context(
        self,
        person_id: str,
        interactions: list[dict[str, Any]],
        sentiments: list[float],
        patterns: list[DetectedPattern],
    ) -> None:
        """Update relationship context with new analysis."""
        if person_id not in self._contexts:
            self._contexts[person_id] = RelationshipContext(
                person_id=person_id,
                person_name="Unknown",
                relationship_type="unknown",
                interaction_history=[],
                sentiment_history=[],
                pattern_history=[],
            )

        context = self._contexts[person_id]
        context.interaction_history.extend(interactions)
        context.sentiment_history.extend(sentiments)
        context.pattern_history.extend(patterns)
        context.last_updated = datetime.now()

        # Trim history to window size
        if len(context.interaction_history) > self.history_window:
            context.interaction_history = context.interaction_history[-self.history_window:]
            context.sentiment_history = context.sentiment_history[-self.history_window:]

        # Update trust score based on patterns
        if patterns:
            risk_factor = sum(
                0.1 if p.severity == PatternSeverity.LOW else
                0.2 if p.severity == PatternSeverity.MODERATE else
                0.3 if p.severity == PatternSeverity.HIGH else 0.4
                for p in patterns
            )
            context.trust_score = max(0.0, context.trust_score - risk_factor)
        else:
            # Slowly rebuild trust if no patterns detected
            context.trust_score = min(1.0, context.trust_score + 0.05)

    def get_context(self, person_id: str) -> RelationshipContext | None:
        """Get relationship context for a person."""
        return self._contexts.get(person_id)

    def set_context_name(self, person_id: str, name: str) -> None:
        """Set the name for a relationship context."""
        if person_id in self._contexts:
            self._contexts[person_id].person_name = name

    def detect_harmful_patterns(
        self,
        interactions: list[dict[str, Any]],
    ) -> InteractionAnalysis:
        """
        Main entry point for detecting harmful patterns.

        This is an alias for analyze_interactions for API consistency.
        """
        return self.analyze_interactions(interactions)

    def mirror_link_reflection(
        self,
        current_emotion: str,
        past_context: str,
    ) -> str:
        """
        MirrorLink component for reflecting contradictions without invalidation.

        This provides the user with a gentle reflection that acknowledges
        both their current emotional state and past context, helping them
        hold both realities without invalidation.

        Example:
            "You feel betrayed now, but you also called them your anchor
            last week. Can both be real?"

        Args:
            current_emotion: The user's current emotional state/feeling.
            past_context: Relevant past context that may seem contradictory.

        Returns:
            A reflection message that holds both realities.
        """
        # Check if there's an apparent contradiction
        current_sentiment = self._analyze_sentiment(current_emotion)
        past_sentiment = self._analyze_sentiment(past_context)

        if abs(current_sentiment - past_sentiment) > 0.5:
            # Significant difference - reflect the contradiction
            return (
                f"You feel {current_emotion} now, but {past_context}. "
                "Can both be real? What might explain this difference?"
            )
        else:
            # Consistent - validate the continuity
            return (
                f"Your feeling of {current_emotion} seems consistent with "
                f"your recent experiences. Your emotional response makes sense."
            )
