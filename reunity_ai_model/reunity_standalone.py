#!/usr/bin/env python3
"""
ReUnity Standalone - Complete Trauma-Aware AI System
=====================================================

A consolidated single-file implementation of the ReUnity system for easy
deployment and testing. This file contains all core components and can be
run directly in any Python environment.

Author: Christopher Ezernack, REOP Solutions
License: MIT

IMPORTANT DISCLAIMER
====================
ReUnity is NOT a clinical or treatment tool. It is a theoretical and support
framework only. This software is not intended to diagnose, treat, cure, or
prevent any medical or psychological condition. It should not be used as a
substitute for professional mental health care.

If you are in crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

Usage:
    python reunity_standalone.py

Requirements:
    pip install numpy cryptography

"""

from __future__ import annotations

import os
import sys
import json
import time
import uuid
import hashlib
import secrets
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Tuple, Any, Callable, 
    TypeVar, Generic, Union, Set
)
from collections import defaultdict
import math
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReUnity")

# Try to import numpy, fall back to pure Python if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not available, using pure Python implementations")

# Try to import cryptography, fall back to basic implementation if not available
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("Cryptography library not available, using basic encryption")


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

VERSION = "1.0.0"
EPSILON = 1e-10  # Small value to prevent log(0)

# Entropy thresholds for state detection
ENTROPY_THRESHOLDS = {
    "crisis": 0.85,
    "high": 0.70,
    "moderate": 0.50,
    "low": 0.30,
    "stable": 0.15,
}

# Lyapunov stability thresholds
LYAPUNOV_THRESHOLDS = {
    "chaotic": 0.5,
    "unstable": 0.1,
    "marginal": 0.0,
    "stable": -0.1,
    "very_stable": -0.5,
}


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EntropyState(Enum):
    """Entropy-based emotional states."""
    CRISIS = "crisis"
    HIGH_ENTROPY = "high_entropy"
    MODERATE = "moderate"
    LOW_ENTROPY = "low_entropy"
    STABLE = "stable"


class PolicyType(Enum):
    """Policy types for state router."""
    CRISIS_INTERVENTION = "crisis_intervention"
    STABILIZATION = "stabilization"
    SUPPORT = "support"
    MAINTENANCE = "maintenance"
    ENGAGEMENT = "engagement"


class ConsentScope(Enum):
    """Consent scopes for data access."""
    PRIVATE = "private"
    SELF_ONLY = "self_only"
    TRUSTED_CONTACTS = "trusted_contacts"
    CLINICIAN = "clinician"
    RESEARCH_ANONYMIZED = "research_anonymized"


class PatternType(Enum):
    """Types of harmful patterns."""
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    HOT_COLD_CYCLE = "hot_cold_cycle"
    ISOLATION = "isolation"
    EMOTIONAL_BAITING = "emotional_baiting"
    ABANDONMENT_TRIGGER = "abandonment_trigger"
    INVALIDATION = "invalidation"
    TRIANGULATION = "triangulation"
    SILENT_TREATMENT = "silent_treatment"
    BLAME_SHIFTING = "blame_shifting"
    FUTURE_FAKING = "future_faking"
    HOOVERING = "hoovering"
    DEVALUATION = "devaluation"


class RegimeType(Enum):
    """Regime types for behavior switching."""
    NORMAL = "normal"
    PROTECTIVE = "protective"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    GROWTH = "growth"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EntropyAnalysis:
    """Result of entropy analysis."""
    shannon_entropy: float
    js_divergence: Optional[float]
    mutual_information: Optional[float]
    lyapunov_exponent: Optional[float]
    state: EntropyState
    stability: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "state": self.state.value,
        }


@dataclass
class MemoryEntry:
    """A single memory entry in the continuity store."""
    id: str
    content: str
    emotional_state: Dict[str, float]
    entropy_at_creation: float
    timestamp: float
    tags: List[str]
    consent_scope: ConsentScope
    linked_memories: List[str] = field(default_factory=list)
    alter_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "consent_scope": self.consent_scope.value,
        }


@dataclass
class PatternDetection:
    """Result of pattern detection."""
    pattern_type: PatternType
    confidence: float
    evidence: List[str]
    timestamp: float
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "pattern_type": self.pattern_type.value,
        }


@dataclass
class Contradiction:
    """A detected contradiction for reflection."""
    id: str
    statement_a: str
    statement_b: str
    context: str
    detected_at: float
    reflection: str
    resolved: bool = False


@dataclass
class Policy:
    """A policy for the state router."""
    type: PolicyType
    actions: List[str]
    grounding_techniques: List[str]
    escalation_threshold: float
    de_escalation_threshold: float


# =============================================================================
# MATHEMATICAL FOUNDATIONS
# =============================================================================

class MathUtils:
    """Mathematical utility functions for entropy calculations."""
    
    @staticmethod
    def shannon_entropy(probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy.
        
        S = -Σ(i=1 to n) p_i * log_2(p_i)
        
        Args:
            probabilities: List of probability values (must sum to 1)
            
        Returns:
            Shannon entropy value
        """
        if HAS_NUMPY:
            p = np.array(probabilities)
            p = p[p > EPSILON]  # Remove zeros
            return -np.sum(p * np.log2(p + EPSILON))
        else:
            entropy = 0.0
            for p in probabilities:
                if p > EPSILON:
                    entropy -= p * math.log2(p + EPSILON)
            return entropy
    
    @staticmethod
    def kl_divergence(p: List[float], q: List[float]) -> float:
        """
        Calculate Kullback-Leibler divergence.
        
        D_KL(P||Q) = Σ p(x) * log(p(x)/q(x))
        """
        if HAS_NUMPY:
            p_arr = np.array(p) + EPSILON
            q_arr = np.array(q) + EPSILON
            return np.sum(p_arr * np.log2(p_arr / q_arr))
        else:
            kl = 0.0
            for pi, qi in zip(p, q):
                pi = pi + EPSILON
                qi = qi + EPSILON
                kl += pi * math.log2(pi / qi)
            return kl
    
    @staticmethod
    def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        JS(P,Q) = (1/2)*D_KL(P||M) + (1/2)*D_KL(Q||M)
        where M = (1/2)*(P + Q)
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            JS divergence (0 to 1, symmetric)
        """
        if HAS_NUMPY:
            p_arr = np.array(p)
            q_arr = np.array(q)
            m = 0.5 * (p_arr + q_arr)
        else:
            m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
        
        return 0.5 * MathUtils.kl_divergence(p, m) + 0.5 * MathUtils.kl_divergence(q, m)
    
    @staticmethod
    def mutual_information(
        joint_probs: List[List[float]],
        marginal_x: List[float],
        marginal_y: List[float]
    ) -> float:
        """
        Calculate mutual information.
        
        MI(X;Y) = Σ(x,y) p(x,y) * log_2(p(x,y) / (p(x)*p(y)))
        
        Args:
            joint_probs: Joint probability matrix p(x,y)
            marginal_x: Marginal probability p(x)
            marginal_y: Marginal probability p(y)
            
        Returns:
            Mutual information value
        """
        mi = 0.0
        for i, px in enumerate(marginal_x):
            for j, py in enumerate(marginal_y):
                pxy = joint_probs[i][j] if i < len(joint_probs) and j < len(joint_probs[i]) else 0
                if pxy > EPSILON and px > EPSILON and py > EPSILON:
                    mi += pxy * math.log2(pxy / (px * py + EPSILON) + EPSILON)
        return mi
    
    @staticmethod
    def lyapunov_exponent(entropy_history: List[float], dt: float = 1.0) -> float:
        """
        Estimate Lyapunov exponent from entropy time series.
        
        λ = lim(n→∞) (1/n) * Σ(i=1 to n) log_2|dS/dt|_{t_i}
        
        Args:
            entropy_history: Time series of entropy values
            dt: Time step between measurements
            
        Returns:
            Estimated Lyapunov exponent
            λ > 0: chaos/instability
            λ < 0: stability
            λ ≈ 0: marginal stability
        """
        if len(entropy_history) < 2:
            return 0.0
        
        # Calculate derivatives
        derivatives = []
        for i in range(1, len(entropy_history)):
            dS = abs(entropy_history[i] - entropy_history[i-1])
            if dS > EPSILON:
                derivatives.append(math.log2(dS / dt + EPSILON))
        
        if not derivatives:
            return 0.0
        
        return sum(derivatives) / len(derivatives)
    
    @staticmethod
    def normalize_distribution(values: List[float]) -> List[float]:
        """Normalize values to form a probability distribution."""
        total = sum(values)
        if total < EPSILON:
            n = len(values)
            return [1.0 / n] * n
        return [v / total for v in values]


# =============================================================================
# ENTROPY-BASED EMOTIONAL STATE ANALYZER (EESA)
# =============================================================================

class EntropyAnalyzer:
    """
    Entropy-Based Emotional State Analyzer (EESA).
    
    Monitors Shannon entropy of emotional states, detects state transitions
    using JS divergence, and tracks stability using Lyapunov exponents.
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.entropy_history: List[float] = []
        self.state_history: List[Dict[str, float]] = []
        self.current_state: Optional[EntropyState] = None
    
    def analyze(
        self,
        emotional_state: Dict[str, float],
        previous_state: Optional[Dict[str, float]] = None
    ) -> EntropyAnalysis:
        """
        Perform comprehensive entropy analysis.
        
        Args:
            emotional_state: Current emotional state as {emotion: intensity}
            previous_state: Previous emotional state for comparison
            
        Returns:
            EntropyAnalysis with all metrics
        """
        # Normalize to probability distribution
        probs = MathUtils.normalize_distribution(list(emotional_state.values()))
        
        # Calculate Shannon entropy
        shannon = MathUtils.shannon_entropy(probs)
        self.entropy_history.append(shannon)
        self.state_history.append(emotional_state)
        
        # Trim history
        if len(self.entropy_history) > self.history_size:
            self.entropy_history = self.entropy_history[-self.history_size:]
            self.state_history = self.state_history[-self.history_size:]
        
        # Calculate JS divergence if we have previous state
        js_div = None
        if previous_state:
            prev_probs = MathUtils.normalize_distribution(list(previous_state.values()))
            # Align distributions
            all_emotions = set(emotional_state.keys()) | set(previous_state.keys())
            p = [emotional_state.get(e, 0) for e in all_emotions]
            q = [previous_state.get(e, 0) for e in all_emotions]
            p = MathUtils.normalize_distribution(p)
            q = MathUtils.normalize_distribution(q)
            js_div = MathUtils.jensen_shannon_divergence(p, q)
        
        # Calculate Lyapunov exponent
        lyapunov = None
        if len(self.entropy_history) >= 3:
            lyapunov = MathUtils.lyapunov_exponent(self.entropy_history[-10:])
        
        # Determine state
        state = self._classify_state(shannon, emotional_state)
        self.current_state = state
        
        # Determine stability
        stability = self._assess_stability(lyapunov)
        
        return EntropyAnalysis(
            shannon_entropy=shannon,
            js_divergence=js_div,
            mutual_information=None,  # Requires relationship data
            lyapunov_exponent=lyapunov,
            state=state,
            stability=stability,
        )
    
    def _classify_state(self, entropy: float, emotional_state: Optional[Dict[str, float]] = None) -> EntropyState:
        """Classify state based on emotional distribution and entropy."""
        # Check for crisis marker in emotional state (set by keyword detection)
        if emotional_state and "crisis" in emotional_state:
            if emotional_state["crisis"] > 0.5:
                return EntropyState.CRISIS
        
        if emotional_state:
            # Check for dominant negative emotions
            negative_emotions = ["fear", "sadness", "anger", "disgust"]
            negative_sum = sum(emotional_state.get(e, 0) for e in negative_emotions)
            positive_emotions = ["joy", "trust", "anticipation"]
            positive_sum = sum(emotional_state.get(e, 0) for e in positive_emotions)
            
            # Single dominant negative emotion (like just "anxious") = HIGH, not CRISIS
            # Crisis requires crisis keywords or multiple strong negatives
            max_negative = max(emotional_state.get(e, 0) for e in negative_emotions)
            
            # If one emotion is completely dominant (>0.9), it's HIGH distress
            if max_negative > 0.9:
                return EntropyState.HIGH_ENTROPY
            
            # If negative emotions dominate overall
            if negative_sum > 0.7:
                return EntropyState.HIGH_ENTROPY
            elif negative_sum > 0.5:
                return EntropyState.MODERATE
            elif negative_sum > 0.3:
                return EntropyState.LOW_ENTROPY
            elif positive_sum > 0.5:
                return EntropyState.STABLE
        
        # Fallback to entropy-based classification
        if entropy < 1.0:
            return EntropyState.MODERATE  # Very concentrated state
        elif entropy < 2.0:
            return EntropyState.LOW_ENTROPY
        else:
            return EntropyState.STABLE  # High entropy = balanced = stable
    
    def _assess_stability(self, lyapunov: Optional[float]) -> str:
        """Assess system stability from Lyapunov exponent."""
        if lyapunov is None:
            return "unknown"
        
        if lyapunov > LYAPUNOV_THRESHOLDS["chaotic"]:
            return "chaotic"
        elif lyapunov > LYAPUNOV_THRESHOLDS["unstable"]:
            return "unstable"
        elif lyapunov > LYAPUNOV_THRESHOLDS["marginal"]:
            return "marginal"
        elif lyapunov > LYAPUNOV_THRESHOLDS["stable"]:
            return "stable"
        else:
            return "very_stable"
    
    def get_trend(self, window: int = 5) -> str:
        """Get entropy trend over recent history."""
        if len(self.entropy_history) < window:
            return "insufficient_data"
        
        recent = self.entropy_history[-window:]
        slope = (recent[-1] - recent[0]) / (window - 1)
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"


# =============================================================================
# STATE ROUTER
# =============================================================================

class StateRouter:
    """
    State Router for policy selection.
    
    Selects appropriate policies based on entropy state, stability,
    and other contextual factors.
    """
    
    def __init__(self):
        self.policies = self._initialize_policies()
        self.current_policy: Optional[Policy] = None
        self.policy_history: List[Tuple[float, PolicyType]] = []
    
    def _initialize_policies(self) -> Dict[PolicyType, Policy]:
        """Initialize default policies."""
        return {
            PolicyType.CRISIS_INTERVENTION: Policy(
                type=PolicyType.CRISIS_INTERVENTION,
                actions=[
                    "Activate immediate grounding",
                    "Provide crisis resources",
                    "Simplify interface",
                    "Enable emergency contacts",
                ],
                grounding_techniques=[
                    "5-4-3-2-1 sensory grounding",
                    "Ice cube technique",
                    "Deep breathing 4-7-8",
                    "Name 5 things you can see",
                ],
                escalation_threshold=0.95,
                de_escalation_threshold=0.70,
            ),
            PolicyType.STABILIZATION: Policy(
                type=PolicyType.STABILIZATION,
                actions=[
                    "Gentle grounding prompts",
                    "Reduce cognitive load",
                    "Offer structured choices",
                    "Maintain calm presence",
                ],
                grounding_techniques=[
                    "Body scan",
                    "Progressive muscle relaxation",
                    "Safe place visualization",
                ],
                escalation_threshold=0.85,
                de_escalation_threshold=0.50,
            ),
            PolicyType.SUPPORT: Policy(
                type=PolicyType.SUPPORT,
                actions=[
                    "Active listening responses",
                    "Validation without judgment",
                    "Gentle exploration prompts",
                    "Resource suggestions",
                ],
                grounding_techniques=[
                    "Mindful breathing",
                    "Present moment awareness",
                ],
                escalation_threshold=0.70,
                de_escalation_threshold=0.30,
            ),
            PolicyType.MAINTENANCE: Policy(
                type=PolicyType.MAINTENANCE,
                actions=[
                    "Check-in prompts",
                    "Journaling suggestions",
                    "Progress reflection",
                    "Goal setting support",
                ],
                grounding_techniques=[
                    "Gratitude practice",
                    "Mindfulness meditation",
                ],
                escalation_threshold=0.50,
                de_escalation_threshold=0.15,
            ),
            PolicyType.ENGAGEMENT: Policy(
                type=PolicyType.ENGAGEMENT,
                actions=[
                    "Open exploration",
                    "Growth-oriented prompts",
                    "Creative exercises",
                    "Future planning",
                ],
                grounding_techniques=[
                    "Visualization",
                    "Values exploration",
                ],
                escalation_threshold=0.30,
                de_escalation_threshold=0.0,
            ),
        }
    
    def route(self, analysis: EntropyAnalysis) -> Policy:
        """
        Route to appropriate policy based on analysis.
        
        Args:
            analysis: Current entropy analysis
            
        Returns:
            Selected policy
        """
        # Map entropy state to policy type
        state_to_policy = {
            EntropyState.CRISIS: PolicyType.CRISIS_INTERVENTION,
            EntropyState.HIGH_ENTROPY: PolicyType.STABILIZATION,
            EntropyState.MODERATE: PolicyType.SUPPORT,
            EntropyState.LOW_ENTROPY: PolicyType.MAINTENANCE,
            EntropyState.STABLE: PolicyType.ENGAGEMENT,
        }
        
        policy_type = state_to_policy[analysis.state]
        
        # Check for stability override
        if analysis.stability == "chaotic":
            policy_type = PolicyType.CRISIS_INTERVENTION
        elif analysis.stability == "unstable" and policy_type == PolicyType.ENGAGEMENT:
            policy_type = PolicyType.MAINTENANCE
        
        policy = self.policies[policy_type]
        self.current_policy = policy
        self.policy_history.append((time.time(), policy_type))
        
        return policy
    
    def get_grounding_technique(self) -> Optional[str]:
        """Get a grounding technique from current policy."""
        if self.current_policy and self.current_policy.grounding_techniques:
            import random
            return random.choice(self.current_policy.grounding_techniques)
        return None


# =============================================================================
# PROTECTIVE PATTERN RECOGNIZER (PLM)
# =============================================================================

class PatternRecognizer:
    """
    Protective Logic Module (PLM).
    
    Detects harmful relational patterns including gaslighting,
    hot-cold cycles, isolation attempts, and other abuse patterns.
    """
    
    def __init__(self):
        self.pattern_indicators = self._initialize_indicators()
        self.detection_history: List[PatternDetection] = []
    
    def _initialize_indicators(self) -> Dict[PatternType, List[str]]:
        """Initialize pattern indicators."""
        return {
            PatternType.GASLIGHTING: [
                "that never happened",
                "never happened",
                "it never happened",
                "you're imagining things",
                "imagining things",
                "imagining it",
                "making it up",
                "didn't happen",
                "never said that",
                "you're wrong",
                "that's not true",
                "you're confused",
                "misremember",
                "your memory",
                "you're too sensitive",
                "you're crazy",
                "no one else thinks that",
                "you're remembering wrong",
                "i never said that",
                "you're making things up",
            ],
            PatternType.LOVE_BOMBING: [
                "soulmate",
                "never felt this way",
                "perfect for each other",
                "meant to be",
                "can't live without",
                "obsessed with you",
                "you complete me",
            ],
            PatternType.HOT_COLD_CYCLE: [
                "intense then distant",
                "loving then cold",
                "available then gone",
                "interested then ignoring",
                "pursuing then withdrawing",
            ],
            PatternType.ISOLATION: [
                "don't need anyone else",
                "your friends don't understand",
                "family is toxic",
                "only i understand you",
                "they're jealous of us",
                "spend all time together",
            ],
            PatternType.EMOTIONAL_BAITING: [
                "if you loved me",
                "prove your love",
                "you don't care",
                "you never",
                "you always",
                "testing you",
            ],
            PatternType.ABANDONMENT_TRIGGER: [
                "maybe we should break up",
                "i'm not sure about us",
                "need space",
                "thinking of leaving",
                "not sure i can do this",
            ],
            PatternType.INVALIDATION: [
                "you shouldn't feel",
                "get over it",
                "stop being dramatic",
                "it's not a big deal",
                "you're overreacting",
            ],
            PatternType.TRIANGULATION: [
                "my ex would",
                "other people think",
                "everyone agrees",
                "someone else would",
                "compared to others",
            ],
            PatternType.SILENT_TREATMENT: [
                "not talking to you",
                "ignoring",
                "won't respond",
                "giving you space",
                "need to think",
            ],
            PatternType.BLAME_SHIFTING: [
                "your fault",
                "you made me",
                "because of you",
                "if you hadn't",
                "you caused this",
            ],
            PatternType.FUTURE_FAKING: [
                "someday we'll",
                "when things settle",
                "i promise eventually",
                "just wait until",
                "things will change",
            ],
            PatternType.HOOVERING: [
                "i've changed",
                "give me another chance",
                "it will be different",
                "i can't live without you",
                "please come back",
            ],
            PatternType.DEVALUATION: [
                "you're not as",
                "you used to be",
                "disappointed in you",
                "not good enough",
                "you've changed",
            ],
        }
    
    def analyze(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[PatternDetection]:
        """
        Analyze text for harmful patterns.
        
        Args:
            text: Text to analyze
            context: Additional context
            
        Returns:
            List of detected patterns
        """
        detections = []
        text_lower = text.lower()
        
        for pattern_type, indicators in self.pattern_indicators.items():
            evidence = []
            for indicator in indicators:
                if indicator in text_lower:
                    evidence.append(indicator)
            
            if evidence:
                confidence = min(len(evidence) / 3, 1.0)  # Cap at 1.0
                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    evidence=evidence,
                    timestamp=time.time(),
                    recommendation=self._get_recommendation(pattern_type),
                )
                detections.append(detection)
                self.detection_history.append(detection)
        
        return detections
    
    def _get_recommendation(self, pattern_type: PatternType) -> str:
        """Get recommendation for detected pattern."""
        recommendations = {
            PatternType.GASLIGHTING: (
                "Trust your own perceptions and memories. Consider documenting "
                "events as they happen. This pattern can make you doubt yourself."
            ),
            PatternType.LOVE_BOMBING: (
                "Healthy relationships develop gradually. Intense early attention "
                "can be a warning sign. Take time to observe consistent behavior."
            ),
            PatternType.HOT_COLD_CYCLE: (
                "Inconsistent behavior creates anxiety and attachment. Notice the "
                "pattern and consider whether this meets your needs for stability."
            ),
            PatternType.ISOLATION: (
                "Healthy partners encourage your other relationships. Isolation "
                "is a control tactic. Maintain connections with trusted people."
            ),
            PatternType.EMOTIONAL_BAITING: (
                "You don't need to prove your feelings through tests. Healthy "
                "relationships are built on trust, not trials."
            ),
            PatternType.ABANDONMENT_TRIGGER: (
                "Threats of leaving can be used to control. Notice if this "
                "happens during conflicts or when you express needs."
            ),
            PatternType.INVALIDATION: (
                "Your feelings are valid. Someone who cares will try to "
                "understand, not dismiss your experience."
            ),
            PatternType.TRIANGULATION: (
                "Comparisons to others are meant to create insecurity. Your "
                "worth isn't determined by comparison."
            ),
            PatternType.SILENT_TREATMENT: (
                "Silence as punishment is different from needing space. Notice "
                "if it's used to control or punish."
            ),
            PatternType.BLAME_SHIFTING: (
                "Healthy people take responsibility for their actions. Constant "
                "blame-shifting avoids accountability."
            ),
            PatternType.FUTURE_FAKING: (
                "Actions matter more than promises. Notice if promises "
                "consistently go unfulfilled."
            ),
            PatternType.HOOVERING: (
                "Past patterns often repeat. Consider whether real change "
                "has occurred before re-engaging."
            ),
            PatternType.DEVALUATION: (
                "Your worth is constant. Devaluation often follows idealization "
                "in unhealthy relationship cycles."
            ),
        }
        return recommendations.get(pattern_type, "Consider discussing this pattern with a trusted person.")
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of detected patterns."""
        if not self.detection_history:
            return {"message": "No patterns detected yet."}
        
        pattern_counts = defaultdict(int)
        for detection in self.detection_history:
            pattern_counts[detection.pattern_type.value] += 1
        
        return {
            "total_detections": len(self.detection_history),
            "pattern_counts": dict(pattern_counts),
            "most_common": max(pattern_counts, key=pattern_counts.get) if pattern_counts else None,
        }


# =============================================================================
# CONTINUITY MEMORY STORE (RIME)
# =============================================================================

class ContinuityMemoryStore:
    """
    Recursive Identity Memory Engine (RIME).
    
    Maintains continuous identity thread across dissociative episodes.
    Stores and retrieves identity-relevant memories with consent controls.
    
    RIME(t) = α · M_episodic(t) + β · M_semantic(t) + γ · C_context(t)
    """
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.memories: Dict[str, MemoryEntry] = {}
        self.timeline: List[str] = []  # Memory IDs in chronological order
        self.tags_index: Dict[str, Set[str]] = defaultdict(set)
        self.alter_index: Dict[str, Set[str]] = defaultdict(set)
        
        # RIME weights
        self.alpha = 0.4  # Episodic memory weight
        self.beta = 0.35  # Semantic memory weight
        self.gamma = 0.25  # Context weight
        
        # Encryption
        self.encryption_key = encryption_key or secrets.token_bytes(32)
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption if available."""
        if HAS_CRYPTO:
            self.cipher = AESGCM(self.encryption_key)
        else:
            self.cipher = None
    
    def store(
        self,
        content: str,
        emotional_state: Dict[str, float],
        entropy: float,
        tags: List[str],
        consent_scope: ConsentScope = ConsentScope.PRIVATE,
        alter_id: Optional[str] = None,
        linked_memories: Optional[List[str]] = None,
    ) -> str:
        """
        Store a new memory entry.
        
        Args:
            content: Memory content
            emotional_state: Emotional state at creation
            entropy: Entropy level at creation
            tags: Tags for indexing
            consent_scope: Access consent level
            alter_id: ID of alter if applicable
            linked_memories: IDs of related memories
            
        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            emotional_state=emotional_state,
            entropy_at_creation=entropy,
            timestamp=time.time(),
            tags=tags,
            consent_scope=consent_scope,
            linked_memories=linked_memories or [],
            alter_id=alter_id,
        )
        
        self.memories[memory_id] = entry
        self.timeline.append(memory_id)
        
        # Update indices
        for tag in tags:
            self.tags_index[tag].add(memory_id)
        
        if alter_id:
            self.alter_index[alter_id].add(memory_id)
        
        logger.info(f"Stored memory {memory_id} with scope {consent_scope.value}")
        return memory_id
    
    def retrieve(
        self,
        memory_id: str,
        requester_scope: ConsentScope = ConsentScope.SELF_ONLY
    ) -> Optional[MemoryEntry]:
        """
        Retrieve a memory by ID with consent check.
        
        Args:
            memory_id: ID of memory to retrieve
            requester_scope: Scope of the requester
            
        Returns:
            Memory entry if access granted, None otherwise
        """
        if memory_id not in self.memories:
            return None
        
        entry = self.memories[memory_id]
        
        # Check consent
        if not self._check_consent(entry.consent_scope, requester_scope):
            logger.warning(f"Access denied to memory {memory_id}")
            return None
        
        return entry
    
    def _check_consent(
        self,
        memory_scope: ConsentScope,
        requester_scope: ConsentScope
    ) -> bool:
        """Check if requester has access to memory."""
        scope_hierarchy = {
            ConsentScope.PRIVATE: 0,
            ConsentScope.SELF_ONLY: 1,
            ConsentScope.TRUSTED_CONTACTS: 2,
            ConsentScope.CLINICIAN: 3,
            ConsentScope.RESEARCH_ANONYMIZED: 4,
        }
        
        return scope_hierarchy[requester_scope] >= scope_hierarchy[memory_scope]
    
    def search_by_tags(
        self,
        tags: List[str],
        requester_scope: ConsentScope = ConsentScope.SELF_ONLY
    ) -> List[MemoryEntry]:
        """Search memories by tags."""
        matching_ids = set()
        for tag in tags:
            matching_ids.update(self.tags_index.get(tag, set()))
        
        results = []
        for memory_id in matching_ids:
            entry = self.retrieve(memory_id, requester_scope)
            if entry:
                results.append(entry)
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def get_timeline(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        requester_scope: ConsentScope = ConsentScope.SELF_ONLY
    ) -> List[MemoryEntry]:
        """Get memories in chronological order."""
        results = []
        for memory_id in self.timeline:
            entry = self.retrieve(memory_id, requester_scope)
            if entry:
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue
                results.append(entry)
        
        return results
    
    def get_alter_memories(
        self,
        alter_id: str,
        requester_scope: ConsentScope = ConsentScope.SELF_ONLY
    ) -> List[MemoryEntry]:
        """Get memories for a specific alter."""
        memory_ids = self.alter_index.get(alter_id, set())
        results = []
        for memory_id in memory_ids:
            entry = self.retrieve(memory_id, requester_scope)
            if entry:
                results.append(entry)
        
        return sorted(results, key=lambda x: x.timestamp, reverse=True)
    
    def calculate_rime_score(
        self,
        memory_id: str,
        current_context: Dict[str, Any]
    ) -> float:
        """
        Calculate RIME relevance score for a memory.
        
        RIME(t) = α · M_episodic(t) + β · M_semantic(t) + γ · C_context(t)
        """
        entry = self.memories.get(memory_id)
        if not entry:
            return 0.0
        
        # Episodic component (recency)
        time_diff = time.time() - entry.timestamp
        episodic_score = math.exp(-time_diff / (86400 * 7))  # Decay over week
        
        # Semantic component (tag overlap)
        context_tags = set(current_context.get("tags", []))
        memory_tags = set(entry.tags)
        semantic_score = len(context_tags & memory_tags) / max(len(context_tags | memory_tags), 1)
        
        # Context component (emotional similarity)
        context_emotions = current_context.get("emotional_state", {})
        if context_emotions and entry.emotional_state:
            emotion_diff = sum(
                abs(context_emotions.get(e, 0) - entry.emotional_state.get(e, 0))
                for e in set(context_emotions.keys()) | set(entry.emotional_state.keys())
            )
            context_score = 1 / (1 + emotion_diff)
        else:
            context_score = 0.5
        
        return (
            self.alpha * episodic_score +
            self.beta * semantic_score +
            self.gamma * context_score
        )
    
    def export_bundle(
        self,
        requester_scope: ConsentScope = ConsentScope.SELF_ONLY,
        include_provenance: bool = True
    ) -> Dict[str, Any]:
        """Export memories as a portable bundle."""
        memories_data = []
        for memory_id in self.timeline:
            entry = self.retrieve(memory_id, requester_scope)
            if entry:
                memories_data.append(entry.to_dict())
        
        bundle = {
            "version": VERSION,
            "exported_at": time.time(),
            "memory_count": len(memories_data),
            "memories": memories_data,
        }
        
        if include_provenance:
            bundle["provenance"] = {
                "source": "ReUnity",
                "export_scope": requester_scope.value,
                "hash": hashlib.sha256(
                    json.dumps(memories_data, sort_keys=True).encode()
                ).hexdigest(),
            }
        
        return bundle


# =============================================================================
# MIRRORLINK REFLECTION LAYER
# =============================================================================

class MirrorLinkReflection:
    """
    MirrorLink Dialogue Companion (MLDC).
    
    Surfaces contradictions without invalidation. Provides controlled
    structured reflections rather than open-ended text generation.
    """
    
    def __init__(self):
        self.contradictions: List[Contradiction] = []
        self.reflections_given: List[Dict[str, Any]] = []
    
    def detect_contradiction(
        self,
        statement_a: str,
        statement_b: str,
        context: str
    ) -> Optional[Contradiction]:
        """
        Detect and record a contradiction.
        
        Args:
            statement_a: First statement
            statement_b: Potentially contradicting statement
            context: Context of the statements
            
        Returns:
            Contradiction object if detected
        """
        # Generate non-invalidating reflection
        reflection = self._generate_reflection(statement_a, statement_b)
        
        contradiction = Contradiction(
            id=str(uuid.uuid4())[:8],
            statement_a=statement_a,
            statement_b=statement_b,
            context=context,
            detected_at=time.time(),
            reflection=reflection,
        )
        
        self.contradictions.append(contradiction)
        return contradiction
    
    def _generate_reflection(self, statement_a: str, statement_b: str) -> str:
        """Generate a non-invalidating reflection on contradiction."""
        templates = [
            (
                f"I notice you mentioned '{statement_a}' and also '{statement_b}'. "
                "Both of these can be true at the same time. What do you notice about "
                "holding both of these?"
            ),
            (
                f"You've shared '{statement_a}' and '{statement_b}'. "
                "It's okay for feelings and thoughts to seem contradictory. "
                "What feels most true right now?"
            ),
            (
                f"I'm hearing '{statement_a}' and also '{statement_b}'. "
                "People are complex, and holding different truths is part of being human. "
                "Would you like to explore either of these more?"
            ),
        ]
        
        import random
        return random.choice(templates)
    
    def reflect_on_pattern(
        self,
        pattern: str,
        instances: List[str]
    ) -> str:
        """
        Reflect on a recurring pattern without judgment.
        
        Args:
            pattern: Description of the pattern
            instances: Specific instances of the pattern
            
        Returns:
            Reflective response
        """
        reflection = (
            f"I've noticed a pattern: {pattern}. "
            f"This has come up {len(instances)} times. "
            "Patterns often carry important information. "
            "What do you think this pattern might be telling you?"
        )
        
        self.reflections_given.append({
            "type": "pattern",
            "pattern": pattern,
            "instance_count": len(instances),
            "timestamp": time.time(),
        })
        
        return reflection
    
    def dialectical_reflection(
        self,
        thesis: str,
        antithesis: str
    ) -> Dict[str, str]:
        """
        Provide dialectical reflection for opposing views.
        
        Supports BPD continuity by allowing multiple truths.
        """
        return {
            "thesis": thesis,
            "antithesis": antithesis,
            "reflection": (
                f"You're holding '{thesis}' and also '{antithesis}'. "
                "In dialectical thinking, we look for the 'both/and' rather than 'either/or'. "
                "What might be true about both of these?"
            ),
            "synthesis_prompt": (
                "What would it look like to honor both of these truths?"
            ),
            "validation": (
                "It's okay to feel pulled in different directions. "
                "This is part of being human, not a flaw."
            ),
        }
    
    def get_unresolved_contradictions(self) -> List[Contradiction]:
        """Get list of unresolved contradictions."""
        return [c for c in self.contradictions if not c.resolved]
    
    def resolve_contradiction(self, contradiction_id: str) -> bool:
        """Mark a contradiction as resolved."""
        for c in self.contradictions:
            if c.id == contradiction_id:
                c.resolved = True
                return True
        return False


# =============================================================================
# REGIME CONTROLLER
# =============================================================================

class RegimeController:
    """
    Regime Controller for behavior switching.
    
    Implements:
    - Regime Logic: Switches behavior based on entropy bands and confidence
    - Apostasis: Pruning/forgetting during stable states
    - Regeneration: Controlled restoration when stability returns
    - Lattice Function: Discrete state graph with divergence-constrained edges
    """
    
    def __init__(self):
        self.current_regime: RegimeType = RegimeType.NORMAL
        self.regime_history: List[Tuple[float, RegimeType]] = []
        self.lattice_nodes: Dict[str, Dict[str, Any]] = {}
        self.lattice_edges: List[Tuple[str, str, float]] = []
        self.pruned_features: List[Dict[str, Any]] = []
        
        # Thresholds
        self.apostasis_threshold = 0.3  # Prune when entropy below this
        self.regeneration_threshold = 0.5  # Regenerate when entropy above this
        self.divergence_constraint = 0.7  # Max JS divergence for edge
    
    def evaluate_regime(self, analysis: EntropyAnalysis) -> RegimeType:
        """
        Evaluate and potentially switch regime based on entropy analysis.
        
        Args:
            analysis: Current entropy analysis
            
        Returns:
            New regime type
        """
        entropy = analysis.shannon_entropy
        stability = analysis.stability
        
        # Determine regime based on entropy and stability
        if entropy >= ENTROPY_THRESHOLDS["crisis"] or stability == "chaotic":
            new_regime = RegimeType.CRISIS
        elif entropy >= ENTROPY_THRESHOLDS["high"] or stability == "unstable":
            new_regime = RegimeType.PROTECTIVE
        elif entropy <= ENTROPY_THRESHOLDS["low"] and stability in ["stable", "very_stable"]:
            new_regime = RegimeType.GROWTH
        elif entropy <= ENTROPY_THRESHOLDS["stable"]:
            new_regime = RegimeType.RECOVERY
        else:
            new_regime = RegimeType.NORMAL
        
        # Record regime change
        if new_regime != self.current_regime:
            self.regime_history.append((time.time(), new_regime))
            logger.info(f"Regime change: {self.current_regime.value} -> {new_regime.value}")
            self.current_regime = new_regime
        
        return new_regime
    
    def apostasis(
        self,
        memory_store: ContinuityMemoryStore,
        current_entropy: float
    ) -> List[str]:
        """
        Perform apostasis (pruning) during stable states.
        
        Removes low-utility memory features to reduce cognitive load.
        Only operates when entropy is below threshold.
        
        Args:
            memory_store: Memory store to prune
            current_entropy: Current entropy level
            
        Returns:
            List of pruned memory IDs
        """
        if current_entropy > self.apostasis_threshold:
            return []
        
        pruned = []
        current_time = time.time()
        
        for memory_id, entry in list(memory_store.memories.items()):
            # Calculate utility score
            age_days = (current_time - entry.timestamp) / 86400
            utility = self._calculate_utility(entry, age_days)
            
            if utility < 0.2:  # Low utility threshold
                # Don't delete, but mark as pruned
                self.pruned_features.append({
                    "memory_id": memory_id,
                    "pruned_at": current_time,
                    "utility_score": utility,
                    "can_regenerate": True,
                })
                pruned.append(memory_id)
        
        logger.info(f"Apostasis: Marked {len(pruned)} memories for pruning")
        return pruned
    
    def _calculate_utility(self, entry: MemoryEntry, age_days: float) -> float:
        """Calculate utility score for a memory."""
        # Recency factor
        recency = math.exp(-age_days / 30)  # Decay over month
        
        # Emotional significance
        if entry.emotional_state:
            intensity = max(entry.emotional_state.values())
        else:
            intensity = 0.5
        
        # Link factor (more connected = more useful)
        link_factor = min(len(entry.linked_memories) / 5, 1.0)
        
        return 0.4 * recency + 0.4 * intensity + 0.2 * link_factor
    
    def regeneration(
        self,
        current_entropy: float,
        stability: str
    ) -> List[Dict[str, Any]]:
        """
        Perform regeneration when stability returns.
        
        Restores previously pruned features in a controlled manner.
        
        Args:
            current_entropy: Current entropy level
            stability: Current stability assessment
            
        Returns:
            List of regenerated features
        """
        if current_entropy < self.regeneration_threshold:
            return []
        
        if stability not in ["stable", "very_stable"]:
            return []
        
        regenerated = []
        for feature in list(self.pruned_features):
            if feature["can_regenerate"]:
                regenerated.append(feature)
                self.pruned_features.remove(feature)
        
        logger.info(f"Regeneration: Restored {len(regenerated)} features")
        return regenerated
    
    def add_lattice_node(
        self,
        node_id: str,
        node_type: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Add a node to the lattice memory graph.
        
        Nodes represent identity, memory, or relationship states.
        """
        self.lattice_nodes[node_id] = {
            "type": node_type,
            "data": data,
            "created_at": time.time(),
        }
    
    def add_lattice_edge(
        self,
        source_id: str,
        target_id: str,
        js_divergence: float
    ) -> bool:
        """
        Add an edge to the lattice graph if divergence constraint is met.
        
        Edges are constrained by JS divergence to maintain coherence.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            js_divergence: JS divergence between nodes
            
        Returns:
            True if edge was added, False if constraint violated
        """
        if js_divergence > self.divergence_constraint:
            logger.warning(
                f"Edge rejected: JS divergence {js_divergence:.3f} > "
                f"constraint {self.divergence_constraint}"
            )
            return False
        
        self.lattice_edges.append((source_id, target_id, js_divergence))
        return True
    
    def get_connected_nodes(self, node_id: str) -> List[str]:
        """Get all nodes connected to a given node."""
        connected = []
        for source, target, _ in self.lattice_edges:
            if source == node_id:
                connected.append(target)
            elif target == node_id:
                connected.append(source)
        return connected
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of regime history and current state."""
        return {
            "current_regime": self.current_regime.value,
            "regime_changes": len(self.regime_history),
            "lattice_nodes": len(self.lattice_nodes),
            "lattice_edges": len(self.lattice_edges),
            "pruned_features": len(self.pruned_features),
        }


# =============================================================================
# GROUNDING TECHNIQUES
# =============================================================================

class GroundingTechniques:
    """
    Collection of grounding techniques for crisis support.
    
    Provides entropy-based recommendations for grounding exercises.
    """
    
    def __init__(self):
        self.techniques = self._initialize_techniques()
        self.usage_history: List[Dict[str, Any]] = []
    
    def _initialize_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize grounding techniques library."""
        return {
            "5-4-3-2-1": {
                "name": "5-4-3-2-1 Sensory Grounding",
                "description": (
                    "Name 5 things you can see, 4 things you can touch, "
                    "3 things you can hear, 2 things you can smell, "
                    "1 thing you can taste."
                ),
                "entropy_range": (0.5, 1.0),
                "duration_minutes": 5,
                "category": "sensory",
            },
            "deep_breathing": {
                "name": "4-7-8 Deep Breathing",
                "description": (
                    "Breathe in for 4 counts, hold for 7 counts, "
                    "exhale for 8 counts. Repeat 4 times."
                ),
                "entropy_range": (0.3, 0.8),
                "duration_minutes": 3,
                "category": "breathing",
            },
            "ice_cube": {
                "name": "Ice Cube Technique",
                "description": (
                    "Hold an ice cube in your hand. Focus on the sensation. "
                    "Notice the cold, the melting, the texture."
                ),
                "entropy_range": (0.7, 1.0),
                "duration_minutes": 5,
                "category": "physical",
            },
            "body_scan": {
                "name": "Body Scan Meditation",
                "description": (
                    "Starting from your toes, slowly move attention up through "
                    "your body. Notice sensations without judgment."
                ),
                "entropy_range": (0.2, 0.6),
                "duration_minutes": 10,
                "category": "mindfulness",
            },
            "safe_place": {
                "name": "Safe Place Visualization",
                "description": (
                    "Imagine a place where you feel completely safe. "
                    "Notice the details: colors, sounds, smells, textures."
                ),
                "entropy_range": (0.3, 0.7),
                "duration_minutes": 10,
                "category": "visualization",
            },
            "grounding_objects": {
                "name": "Grounding Object",
                "description": (
                    "Hold a familiar, comforting object. Focus on its weight, "
                    "texture, temperature. Let it anchor you to the present."
                ),
                "entropy_range": (0.4, 0.9),
                "duration_minutes": 5,
                "category": "physical",
            },
            "cold_water": {
                "name": "Cold Water on Wrists",
                "description": (
                    "Run cold water over your wrists for 30 seconds. "
                    "Focus on the sensation."
                ),
                "entropy_range": (0.6, 1.0),
                "duration_minutes": 2,
                "category": "physical",
            },
            "naming": {
                "name": "Naming Exercise",
                "description": (
                    "Say your name out loud. Say today's date. "
                    "Name where you are. Name one safe person."
                ),
                "entropy_range": (0.7, 1.0),
                "duration_minutes": 2,
                "category": "cognitive",
            },
            "progressive_relaxation": {
                "name": "Progressive Muscle Relaxation",
                "description": (
                    "Tense each muscle group for 5 seconds, then release. "
                    "Start with feet, move up to face."
                ),
                "entropy_range": (0.3, 0.7),
                "duration_minutes": 15,
                "category": "physical",
            },
            "butterfly_hug": {
                "name": "Butterfly Hug",
                "description": (
                    "Cross arms over chest, hands on shoulders. "
                    "Alternately tap shoulders while breathing slowly."
                ),
                "entropy_range": (0.5, 0.9),
                "duration_minutes": 5,
                "category": "bilateral",
            },
        }
    
    def recommend(
        self,
        current_entropy: float,
        category_preference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend grounding techniques based on entropy level.
        
        Args:
            current_entropy: Current entropy level
            category_preference: Optional preferred category
            
        Returns:
            List of recommended techniques
        """
        recommendations = []
        
        for tech_id, tech in self.techniques.items():
            min_entropy, max_entropy = tech["entropy_range"]
            
            if min_entropy <= current_entropy <= max_entropy:
                if category_preference is None or tech["category"] == category_preference:
                    recommendations.append({
                        "id": tech_id,
                        **tech,
                    })
        
        # Sort by how well entropy matches the middle of the range
        recommendations.sort(
            key=lambda x: abs(
                current_entropy - (x["entropy_range"][0] + x["entropy_range"][1]) / 2
            )
        )
        
        return recommendations[:3]  # Return top 3
    
    def log_usage(self, technique_id: str, effectiveness: float) -> None:
        """Log technique usage for personalization."""
        self.usage_history.append({
            "technique_id": technique_id,
            "effectiveness": effectiveness,
            "timestamp": time.time(),
        })
    
    def get_personalized_recommendation(
        self,
        current_entropy: float
    ) -> Optional[Dict[str, Any]]:
        """Get personalized recommendation based on usage history."""
        if not self.usage_history:
            recommendations = self.recommend(current_entropy)
            return recommendations[0] if recommendations else None
        
        # Calculate effectiveness scores
        effectiveness_scores = defaultdict(list)
        for usage in self.usage_history:
            effectiveness_scores[usage["technique_id"]].append(usage["effectiveness"])
        
        avg_scores = {
            tech_id: sum(scores) / len(scores)
            for tech_id, scores in effectiveness_scores.items()
        }
        
        # Get recommendations and sort by effectiveness
        recommendations = self.recommend(current_entropy)
        for rec in recommendations:
            rec["personal_score"] = avg_scores.get(rec["id"], 0.5)
        
        recommendations.sort(key=lambda x: x["personal_score"], reverse=True)
        return recommendations[0] if recommendations else None


# =============================================================================
# ENCRYPTED STORAGE
# =============================================================================

class EncryptedStorage:
    """
    Encrypted local-first storage.
    
    Uses AES-256-GCM for encryption when cryptography library is available.
    """
    
    def __init__(self, storage_path: str = "reunity_data"):
        self.storage_path = storage_path
        self.encryption_key = secrets.token_bytes(32)
        
        if HAS_CRYPTO:
            self.cipher = AESGCM(self.encryption_key)
        else:
            self.cipher = None
            logger.warning("Using basic XOR encryption (not secure for production)")
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        if self.cipher:
            nonce = secrets.token_bytes(12)
            ciphertext = self.cipher.encrypt(nonce, data, None)
            return nonce + ciphertext
        else:
            # Basic XOR (not secure, just for demo)
            key_extended = (self.encryption_key * (len(data) // 32 + 1))[:len(data)]
            return bytes(a ^ b for a, b in zip(data, key_extended))
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if self.cipher:
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            return self.cipher.decrypt(nonce, ciphertext, None)
        else:
            # Basic XOR
            key_extended = (self.encryption_key * (len(encrypted_data) // 32 + 1))[:len(encrypted_data)]
            return bytes(a ^ b for a, b in zip(encrypted_data, key_extended))
    
    def save(self, key: str, data: Dict[str, Any]) -> bool:
        """Save encrypted data."""
        try:
            json_data = json.dumps(data).encode('utf-8')
            encrypted = self.encrypt(json_data)
            
            # In production, save to file
            # For standalone, store in memory
            if not hasattr(self, '_storage'):
                self._storage = {}
            self._storage[key] = encrypted
            
            return True
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt data."""
        try:
            if not hasattr(self, '_storage') or key not in self._storage:
                return None
            
            encrypted = self._storage[key]
            decrypted = self.decrypt(encrypted)
            return json.loads(decrypted.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None


# =============================================================================
# REUNITY CORE SYSTEM
# =============================================================================

class ReUnity:
    """
    ReUnity Core System.
    
    Integrates all components into a cohesive trauma-aware AI system.
    
    DISCLAIMER: ReUnity is NOT a clinical or treatment tool. It is a theoretical
    and support framework only.
    """
    
    def __init__(self):
        # Core components
        self.entropy_analyzer = EntropyAnalyzer()
        self.state_router = StateRouter()
        self.pattern_recognizer = PatternRecognizer()
        self.memory_store = ContinuityMemoryStore()
        self.reflection = MirrorLinkReflection()
        self.regime_controller = RegimeController()
        self.grounding = GroundingTechniques()
        self.storage = EncryptedStorage()
        
        # State
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        self.interaction_count = 0
        
        logger.info(f"ReUnity initialized. Session: {self.session_id}")
        self._print_disclaimer()
    
    def _print_disclaimer(self):
        """Print important disclaimer."""
        disclaimer = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           IMPORTANT DISCLAIMER                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ReUnity is NOT a clinical or treatment tool. It is a theoretical and        ║
║  support framework only. This software is not intended to diagnose, treat,   ║
║  cure, or prevent any medical or psychological condition.                    ║
║                                                                              ║
║  If you are in crisis, please contact:                                       ║
║  • National Suicide Prevention Lifeline: 988 (US)                            ║
║  • Crisis Text Line: Text HOME to 741741 (US)                                ║
║  • International: https://www.iasp.info/resources/Crisis_Centres/            ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(disclaimer)
    
    def process_input(
        self,
        text: str,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Process user input through the ReUnity system.
        
        Args:
            text: User input text
            emotional_state: Optional emotional state dict
            
        Returns:
            Comprehensive response with analysis and support
        """
        self.interaction_count += 1
        
        # Default emotional state if not provided
        if emotional_state is None:
            emotional_state = self._infer_emotional_state(text)
        
        # Get previous state for comparison
        previous_state = None
        if self.entropy_analyzer.state_history:
            previous_state = self.entropy_analyzer.state_history[-1]
        
        # Entropy analysis
        analysis = self.entropy_analyzer.analyze(emotional_state, previous_state)
        
        # Route to appropriate policy
        policy = self.state_router.route(analysis)
        
        # Evaluate regime
        regime = self.regime_controller.evaluate_regime(analysis)
        
        # Pattern detection
        patterns = self.pattern_recognizer.analyze(text)
        
        # Get grounding recommendation if needed
        grounding_rec = None
        if analysis.state in [EntropyState.CRISIS, EntropyState.HIGH_ENTROPY]:
            grounding_rec = self.grounding.get_personalized_recommendation(
                analysis.shannon_entropy
            )
        
        # Store memory
        memory_id = self.memory_store.store(
            content=text,
            emotional_state=emotional_state,
            entropy=analysis.shannon_entropy,
            tags=self._extract_tags(text),
            consent_scope=ConsentScope.SELF_ONLY,
        )
        
        # Build response
        response = {
            "session_id": self.session_id,
            "interaction": self.interaction_count,
            "timestamp": time.time(),
            "analysis": analysis.to_dict(),
            "policy": {
                "type": policy.type.value,
                "actions": policy.actions,
            },
            "regime": regime.value,
            "patterns_detected": [p.to_dict() for p in patterns],
            "grounding_recommendation": grounding_rec,
            "memory_id": memory_id,
            "support_message": self._generate_support_message(analysis, policy, patterns),
        }
        
        return response
    
    def _infer_emotional_state(self, text: str) -> Dict[str, float]:
        """Infer emotional state from text with crisis detection."""
        text_lower = text.lower()
        
        # CRISIS KEYWORDS - These MUST trigger crisis state
        crisis_keywords = {
            "dissociating", "dissociate", "dissociated", "dissociation",
            "depersonalization", "derealization", "not real", "unreal",
            "floating", "detached", "out of body", "watching myself",
            "numb", "empty inside", "disconnected from myself",
            "suicidal", "suicide", "kill myself", "end it", "end my life",
            "want to die", "better off dead", "no reason to live",
            "can't go on", "give up", "hopeless",
            "hurt myself", "cutting", "self harm", "self-harm",
            "panic", "panicking", "panic attack", "can't breathe",
            "heart racing", "going to die", "losing my mind",
            "breaking down", "falling apart", "can't take it",
            "overwhelmed", "drowning", "suffocating",
            "flashback", "triggered", "ptsd",
        }
        
        # HIGH DISTRESS KEYWORDS
        high_keywords = {
            "scared", "afraid", "terrified", "terror", "frightened",
            "anxious", "anxiety", "worried", "nervous", "on edge",
            "angry", "furious", "rage", "hate", "frustrated",
            "sad", "depressed", "crying", "tears", "grief",
            "confused", "lost", "uncertain", "doubt",
            "alone", "lonely", "isolated", "abandoned",
            "hurt", "pain", "suffering", "struggling",
            "stressed", "tense", "restless", "agitated",
            "ashamed", "guilty", "worthless", "failure",
        }
        
        # STABLE KEYWORDS
        stable_keywords = {
            "calm", "peaceful", "relaxed", "okay", "fine", "good",
            "happy", "content", "grateful", "hopeful", "better",
            "safe", "secure", "grounded", "present", "centered",
            "strong", "capable", "confident", "clear",
        }
        
        # Check for crisis keywords FIRST
        crisis_found = [kw for kw in crisis_keywords if kw in text_lower]
        if crisis_found:
            # Return distribution that will calculate to HIGH entropy (crisis)
            return {
                "crisis": 0.95,
                "fear": 0.02,
                "sadness": 0.01,
                "anger": 0.01,
                "joy": 0.005,
                "trust": 0.005,
            }
        
        # Check for high distress keywords
        high_found = [kw for kw in high_keywords if kw in text_lower]
        stable_found = [kw for kw in stable_keywords if kw in text_lower]
        
        # Build emotion distribution based on what was found
        emotions = {
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.0,
            "anticipation": 0.0,
        }
        
        # Score based on specific emotion words
        joy_words = ["happy", "glad", "excited", "wonderful", "great", "love", "grateful", "hopeful"]
        sad_words = ["sad", "depressed", "hopeless", "crying", "hurt", "lonely", "grief", "tears"]
        anger_words = ["angry", "furious", "mad", "frustrated", "annoyed", "hate", "rage"]
        fear_words = ["scared", "afraid", "anxious", "worried", "terrified", "panic", "nervous", "frightened"]
        trust_words = ["safe", "secure", "calm", "peaceful", "okay", "fine", "good", "relaxed"]
        
        for w in joy_words:
            if w in text_lower:
                emotions["joy"] += 0.2
        for w in sad_words:
            if w in text_lower:
                emotions["sadness"] += 0.2
        for w in anger_words:
            if w in text_lower:
                emotions["anger"] += 0.2
        for w in fear_words:
            if w in text_lower:
                emotions["fear"] += 0.2
        for w in trust_words:
            if w in text_lower:
                emotions["trust"] += 0.2
        
        # If high distress keywords found but no specific emotions, boost fear/sadness
        if high_found and sum(emotions.values()) < 0.1:
            emotions["fear"] = 0.4
            emotions["sadness"] = 0.3
            emotions["anger"] = 0.2
            emotions["disgust"] = 0.1
        
        # If stable keywords found and no distress, boost trust/joy
        if stable_found and not high_found:
            emotions["trust"] += 0.3
            emotions["joy"] += 0.2
            emotions["anticipation"] += 0.1
        
        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}
        else:
            # Default to slightly positive neutral state (low entropy = stable)
            emotions = {
                "trust": 0.3,
                "joy": 0.2,
                "anticipation": 0.2,
                "surprise": 0.1,
                "sadness": 0.1,
                "fear": 0.05,
                "anger": 0.03,
                "disgust": 0.02,
            }
        
        return emotions
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text (simplified)."""
        tags = []
        
        # Simple keyword extraction
        keywords = [
            "relationship", "work", "family", "health", "anxiety",
            "depression", "trauma", "memory", "identity", "emotion",
        ]
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword)
        
        return tags
    
    def _generate_support_message(
        self,
        analysis: EntropyAnalysis,
        policy: Policy,
        patterns: List[PatternDetection]
    ) -> str:
        """Generate appropriate support message."""
        messages = {
            EntropyState.CRISIS: (
                "I notice you may be in distress right now. Your safety matters. "
                "Would you like to try a grounding exercise together? "
                "If you're in immediate danger, please reach out to a crisis line: 988 (US)."
            ),
            EntropyState.HIGH_ENTROPY: (
                "Things seem intense right now. That's okay. "
                "Let's take a moment to ground together. "
                "What do you notice in your body right now?"
            ),
            EntropyState.MODERATE: (
                "I'm here with you. How can I support you right now? "
                "Would you like to explore what you're feeling, or would a grounding exercise help?"
            ),
            EntropyState.LOW_ENTROPY: (
                "You seem relatively settled. This is a good time for reflection or planning. "
                "What would be most helpful for you right now?"
            ),
            EntropyState.STABLE: (
                "Things seem stable. This could be a good time for growth-oriented work. "
                "What would you like to explore?"
            ),
        }
        
        message = messages.get(analysis.state, "I'm here to support you.")
        
        # Add pattern warning if detected
        if patterns:
            most_confident = max(patterns, key=lambda p: p.confidence)
            if most_confident.confidence > 0.2:  # Lower threshold to catch more patterns
                message = f"I noticed something in what you shared that might be worth reflecting on.\n\n{most_confident.recommendation}\n\n{message}"
        
        return message
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_id": self.session_id,
            "duration_minutes": (time.time() - self.session_start) / 60,
            "interactions": self.interaction_count,
            "current_state": self.entropy_analyzer.current_state.value if self.entropy_analyzer.current_state else "unknown",
            "current_regime": self.regime_controller.current_regime.value,
            "memories_stored": len(self.memory_store.memories),
            "patterns_detected": len(self.pattern_recognizer.detection_history),
            "entropy_trend": self.entropy_analyzer.get_trend(),
        }
    
    def export_data(
        self,
        include_memories: bool = True,
        include_patterns: bool = True
    ) -> Dict[str, Any]:
        """Export session data as portable bundle."""
        bundle = {
            "version": VERSION,
            "exported_at": time.time(),
            "session_summary": self.get_session_summary(),
        }
        
        if include_memories:
            bundle["memories"] = self.memory_store.export_bundle()
        
        if include_patterns:
            bundle["pattern_summary"] = self.pattern_recognizer.get_pattern_summary()
        
        # Add provenance
        bundle["provenance"] = {
            "source": "ReUnity",
            "hash": hashlib.sha256(
                json.dumps(bundle, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        
        return bundle


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def run_interactive():
    """Run interactive CLI session."""
    print("\n" + "="*80)
    print("ReUnity Interactive Session")
    print("="*80)
    print("\nCommands:")
    print("  /status  - Show session status")
    print("  /ground  - Get grounding technique")
    print("  /export  - Export session data")
    print("  /help    - Show help")
    print("  /quit    - Exit session")
    print("\nType anything else to interact with ReUnity.\n")
    
    reunity = ReUnity()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "/quit":
                print("\nThank you for using ReUnity. Take care of yourself.")
                break
            
            elif user_input.lower() == "/status":
                summary = reunity.get_session_summary()
                print("\n--- Session Status ---")
                for key, value in summary.items():
                    print(f"  {key}: {value}")
            
            elif user_input.lower() == "/ground":
                entropy = reunity.entropy_analyzer.entropy_history[-1] if reunity.entropy_analyzer.entropy_history else 0.5
                rec = reunity.grounding.get_personalized_recommendation(entropy)
                if rec:
                    print(f"\n--- Grounding Technique: {rec['name']} ---")
                    print(f"  {rec['description']}")
                    print(f"  Duration: {rec['duration_minutes']} minutes")
                else:
                    print("\nNo grounding technique available.")
            
            elif user_input.lower() == "/export":
                data = reunity.export_data()
                filename = f"reunity_export_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(f"\nExported to {filename}")
            
            elif user_input.lower() == "/help":
                print("\n--- Help ---")
                print("ReUnity is a trauma-aware support system.")
                print("Share what's on your mind, and ReUnity will provide")
                print("appropriate support based on your emotional state.")
                print("\nRemember: This is NOT a replacement for professional help.")
            
            else:
                response = reunity.process_input(user_input)
                
                print(f"\n--- ReUnity ---")
                print(f"State: {response['analysis']['state']} | "
                      f"Entropy: {response['analysis']['shannon_entropy']:.3f} | "
                      f"Regime: {response['regime']}")
                print(f"\n{response['support_message']}")
                
                if response['patterns_detected']:
                    print("\n[Pattern detected - see recommendations above]")
                
                if response['grounding_recommendation']:
                    print(f"\n[Grounding available: {response['grounding_recommendation']['name']}]")
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Take care of yourself.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception("Error in interactive session")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print(f"\nReUnity v{VERSION}")
    print("A Trauma-Aware AI Support Framework")
    print("By Christopher Ezernack, REOP Solutions\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Run basic test
            print("Running basic test...")
            reunity = ReUnity()
            
            test_inputs = [
                "I'm feeling anxious today",
                "My partner said I was imagining things again",
                "I feel disconnected from myself",
                "Things are actually going okay",
            ]
            
            for text in test_inputs:
                print(f"\nInput: {text}")
                response = reunity.process_input(text)
                print(f"State: {response['analysis']['state']}")
                print(f"Entropy: {response['analysis']['shannon_entropy']:.3f}")
            
            print("\n--- Session Summary ---")
            print(json.dumps(reunity.get_session_summary(), indent=2))
            
        elif sys.argv[1] == "--version":
            print(f"Version: {VERSION}")
        
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python reunity_standalone.py [--test|--version]")
    else:
        # Run interactive session
        run_interactive()


if __name__ == "__main__":
    main()
