# ReUnity AI Model: Complete Code for Faraz

## Copy and Paste Guide

Hello Faraz,

This document contains ALL the code for the ReUnity AI model in order. Just follow each step and copy/paste the code into the files as instructed.

---

## STEP 1: Open GitHub Codespaces

1. Go to: https://github.com/ezernackchristopher97-cloud/ReUnity
2. Click the green "Code" button
3. Click "Codespaces" tab
4. Click "Create codespace on main"
5. Wait for it to load (2-3 minutes)

---

## STEP 2: Create the Project Folder

In the terminal at the bottom, type this and press Enter:

```bash
mkdir -p reunity_model && cd reunity_model
```

---

## STEP 3: Create requirements.txt

Create a new file called `requirements.txt` and paste this:

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
fastapi>=0.68.0
pydantic>=1.8.0
uvicorn>=0.15.0
python-multipart>=0.0.5
```

---

## STEP 4: Install Dependencies

In the terminal, run:

```bash
pip install -r requirements.txt
```

---

## STEP 5: Create the Main Model File

Create a new file called `reunity_model.py` and paste ALL of the following code:

```python
#!/usr/bin/env python3
"""
ReUnity AI Model - Complete Implementation
Author: Christopher Ezernack

DISCLAIMER: This is NOT a clinical or treatment tool. 
It is a theoretical and support framework only.
"""

import math
import time
import uuid
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, List, Dict
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# SECTION 1: ENTROPY STATES AND METRICS
# =============================================================================

class EntropyState(Enum):
    """Enumeration of entropy-based emotional states."""
    LOW = "low"           # Emotional rigidity or suppression
    STABLE = "stable"     # Healthy emotional range
    ELEVATED = "elevated" # Increased emotional variability
    HIGH = "high"         # Significant fragmentation
    CRISIS = "crisis"     # Crisis-level instability


@dataclass
class EntropyMetrics:
    """Container for entropy-related metrics."""
    shannon_entropy: float
    normalized_entropy: float
    state: EntropyState
    confidence: float
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergenceMetrics:
    """Container for divergence metrics between states."""
    js_divergence: float
    kl_divergence_pq: float
    kl_divergence_qp: float
    symmetric: bool = True
    transition_detected: bool = False


@dataclass
class StabilityMetrics:
    """Container for Lyapunov stability metrics."""
    lyapunov_exponent: float
    is_stable: bool
    is_chaotic: bool
    stability_trend: str
    confidence_interval: tuple = (0.0, 0.0)


# =============================================================================
# SECTION 2: CORE ENTROPY CALCULATIONS
# =============================================================================

EPSILON = 1e-10  # Prevent log(0) errors

DEFAULT_ENTROPY_THRESHOLDS = {
    EntropyState.LOW: 0.3,
    EntropyState.STABLE: 0.5,
    EntropyState.ELEVATED: 0.7,
    EntropyState.HIGH: 0.85,
    EntropyState.CRISIS: 1.0,
}


def calculate_shannon_entropy(probabilities: NDArray) -> float:
    """
    Calculate Shannon entropy for emotional state distribution.
    
    Formula: S = -Σ(i=1 to n) p_i * log_2(p_i)
    
    Args:
        probabilities: Array of emotional state probabilities. Must sum to 1.
    
    Returns:
        Shannon entropy value in bits.
    """
    if len(probabilities) == 0:
        return 0.0
    
    probabilities = np.asarray(probabilities, dtype=np.float64)
    
    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative")
    
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        return 0.0
    
    p_safe = np.where(probabilities > 0, probabilities, EPSILON)
    entropy = -np.sum(probabilities * np.log2(p_safe))
    
    return float(entropy)


def calculate_normalized_entropy(probabilities: NDArray) -> float:
    """Calculate normalized Shannon entropy (0 to 1 scale)."""
    n_states = len(probabilities)
    if n_states <= 1:
        return 0.0
    
    entropy = calculate_shannon_entropy(probabilities)
    max_entropy = np.log2(n_states)
    
    if max_entropy == 0:
        return 0.0
    
    return float(entropy / max_entropy)


def calculate_kl_divergence(p: NDArray, q: NDArray) -> float:
    """
    Calculate Kullback-Leibler divergence D_KL(P||Q).
    
    Formula: D_KL(P||Q) = Σ p(x) * log_2(p(x) / q(x))
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")
    
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    p_safe = np.where(p > 0, p, EPSILON)
    q_safe = np.where(q > 0, q, EPSILON)
    
    kl_div = np.sum(np.where(p > 0, p * np.log2(p_safe / q_safe), 0))
    
    return float(max(0.0, kl_div))


def calculate_jensen_shannon_divergence(p: NDArray, q: NDArray) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    Formula: JS(P,Q) = (1/2)*D_KL(P||M) + (1/2)*D_KL(Q||M)
    where M = (1/2)*(P + Q)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    m = 0.5 * (p + q)
    js_div = 0.5 * calculate_kl_divergence(p, m) + 0.5 * calculate_kl_divergence(q, m)
    
    return float(js_div)


def calculate_lyapunov_exponent(state_sequence: NDArray, delta_t: float = 1.0) -> StabilityMetrics:
    """
    Calculate Lyapunov exponent for emotional state stability analysis.
    
    Formula: λ = lim(n→∞) (1/n) * Σ log_2|dS/dt|
    
    Interpretation:
    - λ > 0: Chaos/instability (crisis intervention may be needed)
    - λ < 0: Stability (therapeutic progress)
    - λ ≈ 0: Marginal stability
    """
    states = np.asarray(state_sequence, dtype=np.float64)
    
    if len(states) < 3:
        return StabilityMetrics(
            lyapunov_exponent=0.0,
            is_stable=True,
            is_chaotic=False,
            stability_trend="stable",
            confidence_interval=(0.0, 0.0),
        )
    
    derivatives = np.diff(states) / delta_t
    derivatives_safe = np.where(np.abs(derivatives) > EPSILON, np.abs(derivatives), EPSILON)
    log_sensitivities = np.log2(derivatives_safe)
    lyapunov = float(np.mean(log_sensitivities))
    
    if len(log_sensitivities) >= 10:
        std_err = np.std(log_sensitivities) / np.sqrt(len(log_sensitivities))
        ci_low = lyapunov - 1.96 * std_err
        ci_high = lyapunov + 1.96 * std_err
    else:
        ci_low = lyapunov - 0.5
        ci_high = lyapunov + 0.5
    
    is_stable = lyapunov < 0
    is_chaotic = lyapunov > 0.5
    
    if len(log_sensitivities) >= 4:
        mid = len(log_sensitivities) // 2
        early_mean = np.mean(log_sensitivities[:mid])
        late_mean = np.mean(log_sensitivities[mid:])
        
        if late_mean < early_mean - 0.1:
            trend = "improving"
        elif late_mean > early_mean + 0.1:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    return StabilityMetrics(
        lyapunov_exponent=lyapunov,
        is_stable=is_stable,
        is_chaotic=is_chaotic,
        stability_trend=trend,
        confidence_interval=(float(ci_low), float(ci_high)),
    )


def classify_entropy_state(entropy: float, thresholds: Dict = None) -> EntropyState:
    """Classify entropy value into an emotional state category."""
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS
    
    if entropy < thresholds[EntropyState.LOW]:
        return EntropyState.LOW
    elif entropy < thresholds[EntropyState.STABLE]:
        return EntropyState.STABLE
    elif entropy < thresholds[EntropyState.ELEVATED]:
        return EntropyState.ELEVATED
    elif entropy < thresholds[EntropyState.HIGH]:
        return EntropyState.HIGH
    else:
        return EntropyState.CRISIS


# =============================================================================
# SECTION 3: ENTROPY ANALYZER
# =============================================================================

class EntropyAnalyzer:
    """
    Analyzes emotional state entropy from text or probability distributions.
    """
    
    def __init__(self, window_size: int = 10, thresholds: Dict = None):
        self.window_size = window_size
        self.thresholds = thresholds or DEFAULT_ENTROPY_THRESHOLDS
        self._history: List[EntropyMetrics] = []
        self._state_sequence: List[float] = []
    
    def analyze(self, probabilities: NDArray) -> EntropyMetrics:
        """Analyze entropy from probability distribution."""
        shannon = calculate_shannon_entropy(probabilities)
        normalized = calculate_normalized_entropy(probabilities)
        state = classify_entropy_state(normalized, self.thresholds)
        
        confidence = self._calculate_confidence(normalized, state)
        
        metrics = EntropyMetrics(
            shannon_entropy=shannon,
            normalized_entropy=normalized,
            state=state,
            confidence=confidence,
            timestamp=time.time(),
        )
        
        self._history.append(metrics)
        self._state_sequence.append(normalized)
        
        if len(self._history) > self.window_size * 10:
            self._history = self._history[-self.window_size * 10:]
            self._state_sequence = self._state_sequence[-self.window_size * 10:]
        
        return metrics
    
    def analyze_text(self, text: str) -> EntropyMetrics:
        """Analyze entropy from text using simple sentiment analysis."""
        probabilities = self._text_to_probabilities(text)
        return self.analyze(probabilities)
    
    def _text_to_probabilities(self, text: str) -> NDArray:
        """Convert text to emotion probability distribution."""
        positive_words = {
            "good", "love", "happy", "safe", "calm", "trust", "support",
            "kind", "caring", "gentle", "peace", "joy", "grateful"
        }
        negative_words = {
            "bad", "hate", "angry", "unsafe", "scared", "hurt", "pain",
            "cruel", "mean", "fear", "anxiety", "shame", "guilt", "alone"
        }
        neutral_words = {"okay", "fine", "normal", "regular", "usual"}
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        neu_count = sum(1 for w in words if w in neutral_words)
        other_count = len(words) - pos_count - neg_count - neu_count
        
        total = max(1, pos_count + neg_count + neu_count + other_count)
        
        probs = np.array([
            pos_count / total,
            neg_count / total,
            neu_count / total,
            other_count / total,
            0.1  # baseline uncertainty
        ])
        
        return probs / np.sum(probs)
    
    def _calculate_confidence(self, normalized: float, state: EntropyState) -> float:
        """Calculate confidence in state classification."""
        thresholds = list(self.thresholds.values())
        
        distances = [abs(normalized - t) for t in thresholds]
        min_distance = min(distances)
        
        confidence = 1.0 - min(1.0, min_distance * 2)
        return max(0.3, confidence)
    
    def get_stability(self) -> StabilityMetrics:
        """Get stability metrics from recent history."""
        if len(self._state_sequence) < 3:
            return StabilityMetrics(
                lyapunov_exponent=0.0,
                is_stable=True,
                is_chaotic=False,
                stability_trend="stable",
            )
        
        recent = self._state_sequence[-self.window_size:]
        return calculate_lyapunov_exponent(np.array(recent))


# =============================================================================
# SECTION 4: STATE ROUTER
# =============================================================================

class PolicyMode(Enum):
    """Operating modes for the system based on emotional state."""
    SUPPORTIVE = "supportive"
    GROUNDING = "grounding"
    PROTECTIVE = "protective"
    CRISIS = "crisis"
    REFLECTIVE = "reflective"


class ResponseConstraint(Enum):
    """Constraints on system responses."""
    FULL = "full"
    SIMPLIFIED = "simplified"
    GROUNDING_ONLY = "grounding_only"
    SAFETY_FOCUSED = "safety_focused"
    MINIMAL = "minimal"


@dataclass
class PolicyConfig:
    """Configuration for a specific policy mode."""
    mode: PolicyMode
    response_constraint: ResponseConstraint
    max_response_length: int
    allow_deep_reflection: bool
    allow_memory_retrieval: bool
    require_grounding_prompts: bool
    crisis_resources_visible: bool
    description: str


@dataclass
class RoutingDecision:
    """Decision made by the state router."""
    policy: PolicyConfig
    entropy_state: EntropyState
    confidence: float
    reasoning: str
    recommended_actions: List[str]
    warnings: List[str]


DEFAULT_POLICIES = {
    EntropyState.LOW: PolicyConfig(
        mode=PolicyMode.REFLECTIVE,
        response_constraint=ResponseConstraint.FULL,
        max_response_length=2000,
        allow_deep_reflection=True,
        allow_memory_retrieval=True,
        require_grounding_prompts=False,
        crisis_resources_visible=False,
        description="Low entropy: emotional rigidity. Enable deep reflection.",
    ),
    EntropyState.STABLE: PolicyConfig(
        mode=PolicyMode.SUPPORTIVE,
        response_constraint=ResponseConstraint.FULL,
        max_response_length=1500,
        allow_deep_reflection=True,
        allow_memory_retrieval=True,
        require_grounding_prompts=False,
        crisis_resources_visible=False,
        description="Stable entropy: healthy range. Full support enabled.",
    ),
    EntropyState.ELEVATED: PolicyConfig(
        mode=PolicyMode.SUPPORTIVE,
        response_constraint=ResponseConstraint.SIMPLIFIED,
        max_response_length=1000,
        allow_deep_reflection=False,
        allow_memory_retrieval=True,
        require_grounding_prompts=True,
        crisis_resources_visible=False,
        description="Elevated entropy: increased variability. Simplified responses.",
    ),
    EntropyState.HIGH: PolicyConfig(
        mode=PolicyMode.GROUNDING,
        response_constraint=ResponseConstraint.GROUNDING_ONLY,
        max_response_length=500,
        allow_deep_reflection=False,
        allow_memory_retrieval=True,
        require_grounding_prompts=True,
        crisis_resources_visible=True,
        description="High entropy: fragmentation. Focus on grounding.",
    ),
    EntropyState.CRISIS: PolicyConfig(
        mode=PolicyMode.CRISIS,
        response_constraint=ResponseConstraint.SAFETY_FOCUSED,
        max_response_length=300,
        allow_deep_reflection=False,
        allow_memory_retrieval=False,
        require_grounding_prompts=True,
        crisis_resources_visible=True,
        description="Crisis entropy: immediate support needed. Safety focus.",
    ),
}


class StateRouter:
    """
    State Router for policy selection based on entropy state.
    """
    
    def __init__(self, policies: Dict = None, transition_smoothing: float = 0.3):
        self.policies = policies or DEFAULT_POLICIES.copy()
        self.transition_smoothing = transition_smoothing
        self._current_policy: Optional[PolicyConfig] = None
        self._previous_state: Optional[EntropyState] = None
        self._transition_count: int = 0
    
    def route(self, entropy_metrics: EntropyMetrics, 
              stability_metrics: StabilityMetrics = None) -> RoutingDecision:
        """Route to appropriate policy based on current metrics."""
        target_state = entropy_metrics.state
        
        if self._previous_state is not None and self._previous_state != target_state:
            if entropy_metrics.confidence < 0.6:
                target_state = self._previous_state
        
        policy = self.policies.get(target_state, self.policies[EntropyState.STABLE])
        
        reasoning = f"Entropy state: {target_state.value}. {policy.description}"
        
        actions = self._generate_actions(target_state, entropy_metrics, stability_metrics)
        warnings = self._generate_warnings(entropy_metrics, stability_metrics, target_state)
        
        if self._previous_state != target_state:
            self._transition_count += 1
        self._previous_state = target_state
        self._current_policy = policy
        
        return RoutingDecision(
            policy=policy,
            entropy_state=target_state,
            confidence=entropy_metrics.confidence,
            reasoning=reasoning,
            recommended_actions=actions,
            warnings=warnings,
        )
    
    def _generate_actions(self, state: EntropyState, 
                          entropy: EntropyMetrics,
                          stability: StabilityMetrics = None) -> List[str]:
        """Generate recommended actions."""
        actions = []
        
        if state == EntropyState.CRISIS:
            actions.append("Provide crisis resources")
            actions.append("Use grounding techniques")
            actions.append("Keep responses short and clear")
        elif state == EntropyState.HIGH:
            actions.append("Offer grounding exercises")
            actions.append("Simplify language")
            actions.append("Check in frequently")
        elif state == EntropyState.ELEVATED:
            actions.append("Include grounding prompts")
            actions.append("Monitor for escalation")
        elif state == EntropyState.LOW:
            actions.append("Encourage emotional exploration")
            actions.append("Use open-ended questions")
        else:
            actions.append("Continue supportive interaction")
        
        return actions
    
    def _generate_warnings(self, entropy: EntropyMetrics,
                           stability: StabilityMetrics,
                           state: EntropyState) -> List[str]:
        """Generate warnings based on metrics."""
        warnings = []
        
        if state in [EntropyState.HIGH, EntropyState.CRISIS]:
            warnings.append("Elevated distress detected")
        
        if stability and stability.is_chaotic:
            warnings.append("Emotional instability detected")
        
        if stability and stability.stability_trend == "degrading":
            warnings.append("Stability is degrading")
        
        return warnings


# =============================================================================
# SECTION 5: PROTECTIVE PATTERN RECOGNITION
# =============================================================================

class PatternType(Enum):
    """Types of potentially harmful patterns."""
    HOT_COLD_CYCLE = "hot_cold_cycle"
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    ABANDONMENT_THREAT = "abandonment_threat"
    ISOLATION_ATTEMPT = "isolation_attempt"
    INVALIDATION = "invalidation"
    BLAME_SHIFTING = "blame_shifting"


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
    evidence: List[str]
    message: str
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


GASLIGHTING_PHRASES = [
    r"you'?re (crazy|insane|imagining|overreacting|too sensitive)",
    r"that (never|didn'?t) happen",
    r"you'?re making (that|this|it) up",
    r"no one (else|will) (believes?|love|understand) you",
    r"i never said that",
    r"you'?re (being|so) (dramatic|paranoid)",
]

LOVE_BOMBING_INDICATORS = [
    r"you'?re (the only one|my everything|perfect)",
    r"i can'?t live without you",
    r"no one (will ever|could) love you like i do",
]

ISOLATION_PHRASES = [
    r"(they|your friends|your family) (don'?t|doesn'?t) (understand|care|love) you",
    r"you don'?t need (them|anyone else)",
    r"i'?m the only one who",
]


class ProtectivePatternRecognizer:
    """
    Protective Logic Module (PLM) for detecting harmful relationship patterns.
    """
    
    def __init__(self, sensitivity: float = 0.5, variance_threshold: float = 0.5):
        self.sensitivity = sensitivity
        self.variance_threshold = variance_threshold
        
        self._gaslighting_patterns = [re.compile(p, re.IGNORECASE) for p in GASLIGHTING_PHRASES]
        self._love_bombing_patterns = [re.compile(p, re.IGNORECASE) for p in LOVE_BOMBING_INDICATORS]
        self._isolation_patterns = [re.compile(p, re.IGNORECASE) for p in ISOLATION_PHRASES]
    
    def analyze_text(self, text: str) -> List[DetectedPattern]:
        """Analyze text for harmful patterns."""
        patterns = []
        
        gaslighting = self._detect_gaslighting(text)
        if gaslighting:
            patterns.append(gaslighting)
        
        love_bombing = self._detect_love_bombing(text)
        if love_bombing:
            patterns.append(love_bombing)
        
        isolation = self._detect_isolation(text)
        if isolation:
            patterns.append(isolation)
        
        return patterns
    
    def analyze_sentiment_sequence(self, sentiments: List[float]) -> Optional[DetectedPattern]:
        """Detect hot-cold cycles from sentiment sequence."""
        if len(sentiments) < 4:
            return None
        
        variance = np.std(sentiments)
        
        if variance > self.variance_threshold:
            changes = np.diff(sentiments)
            sign_changes = np.sum(np.abs(np.diff(np.sign(changes)))) / 2
            
            if sign_changes >= 2:
                return DetectedPattern(
                    pattern_type=PatternType.HOT_COLD_CYCLE,
                    severity=PatternSeverity.MODERATE if variance < 0.7 else PatternSeverity.HIGH,
                    confidence=min(1.0, variance),
                    evidence=[f"Sentiment variance: {variance:.2f}", f"Sign changes: {sign_changes}"],
                    message="Hot-cold emotional cycling detected. This pattern can be destabilizing.",
                    recommendation="Trust your perception. Consider documenting these patterns.",
                )
        
        return None
    
    def _detect_gaslighting(self, text: str) -> Optional[DetectedPattern]:
        """Detect gaslighting patterns."""
        matches = []
        for pattern in self._gaslighting_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        if matches:
            return DetectedPattern(
                pattern_type=PatternType.GASLIGHTING,
                severity=PatternSeverity.HIGH,
                confidence=min(1.0, len(matches) * 0.3 + 0.4),
                evidence=matches,
                message="Language patterns associated with gaslighting detected.",
                recommendation="Trust your memory. Consider reviewing your records.",
            )
        return None
    
    def _detect_love_bombing(self, text: str) -> Optional[DetectedPattern]:
        """Detect love bombing patterns."""
        matches = []
        for pattern in self._love_bombing_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        if matches:
            return DetectedPattern(
                pattern_type=PatternType.LOVE_BOMBING,
                severity=PatternSeverity.MODERATE,
                confidence=min(1.0, len(matches) * 0.3 + 0.3),
                evidence=matches,
                message="Intense idealization language detected.",
                recommendation="Healthy relationships develop gradually over time.",
            )
        return None
    
    def _detect_isolation(self, text: str) -> Optional[DetectedPattern]:
        """Detect isolation attempt patterns."""
        matches = []
        for pattern in self._isolation_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        if matches:
            return DetectedPattern(
                pattern_type=PatternType.ISOLATION_ATTEMPT,
                severity=PatternSeverity.HIGH,
                confidence=min(1.0, len(matches) * 0.3 + 0.4),
                evidence=matches,
                message="Language attempting to isolate you from support systems detected.",
                recommendation="Maintain connections with trusted friends and family.",
            )
        return None


# =============================================================================
# SECTION 6: MEMORY CONTINUITY STORE
# =============================================================================

class ConsentScope(Enum):
    """Consent scopes for memory access."""
    PRIVATE = "private"
    SELF_ONLY = "self_only"
    THERAPIST = "therapist"
    CAREGIVER = "caregiver"
    EMERGENCY = "emergency"


class MemoryType(Enum):
    """Types of memories stored in the system."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    RELATIONAL = "relational"
    GROUNDING = "grounding"
    IDENTITY = "identity"
    JOURNAL = "journal"


@dataclass
class MemoryEntry:
    """A single memory entry in the store."""
    id: str
    content: str
    memory_type: MemoryType
    identity_state: str
    timestamp: float
    tags: List[str]
    entropy_at_creation: float
    consent_scope: ConsentScope
    emotional_valence: float  # -1 to 1
    importance: float  # 0 to 1
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    linked_memories: List[str] = field(default_factory=list)


class RecursiveIdentityMemoryEngine:
    """
    RIME - Recursive Identity Memory Engine.
    
    Provides external memory support during dissociation and emotional amnesia.
    
    Formula: RIME(t) = α · M_episodic(t) + β · M_semantic(t) + γ · C_context(t)
    """
    
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
                 max_memories: int = 10000, grounding_priority: float = 0.8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_memories = max_memories
        self.grounding_priority = grounding_priority
        
        self._memory_threads: Dict[str, List[MemoryEntry]] = {}
        self._emotional_states: Dict[str, float] = {}
    
    def add_memory(self, identity: str, content: str,
                   memory_type: MemoryType = MemoryType.EPISODIC,
                   tags: List[str] = None, entropy: float = 0.5,
                   consent_scope: ConsentScope = ConsentScope.PRIVATE,
                   emotional_valence: float = 0.0,
                   importance: float = 0.5) -> MemoryEntry:
        """Add a tagged memory for a specific identity state."""
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
        )
        
        self._memory_threads[identity].append(memory)
        self._update_emotional_state(identity, memory)
        self._enforce_memory_limit()
        
        return memory
    
    def retrieve_grounding(self, current_identity: str, query: str,
                           crisis_level: float = 0.0,
                           max_results: int = 5) -> List[MemoryEntry]:
        """Retrieve grounding memories during fragmentation or crisis."""
        all_memories = []
        for identity, memories in self._memory_threads.items():
            all_memories.extend(memories)
        
        if crisis_level > 0.7:
            grounding_memories = [m for m in all_memories 
                                  if m.memory_type == MemoryType.GROUNDING]
            if grounding_memories:
                return sorted(grounding_memories, 
                              key=lambda m: m.importance, 
                              reverse=True)[:max_results]
        
        scored_memories = []
        query_words = set(query.lower().split())
        
        for memory in all_memories:
            score = 0.0
            
            content_words = set(memory.content.lower().split())
            overlap = len(query_words & content_words)
            score += overlap * 0.3
            
            tag_overlap = len(query_words & set(t.lower() for t in memory.tags))
            score += tag_overlap * 0.4
            
            score += memory.importance * 0.2
            
            if memory.memory_type == MemoryType.GROUNDING:
                score += self.grounding_priority * crisis_level
            
            scored_memories.append((memory, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        results = [m for m, s in scored_memories[:max_results]]
        for memory in results:
            memory.retrieval_count += 1
            memory.last_retrieved = time.time()
        
        return results
    
    def get_identity_summary(self, identity: str) -> Dict[str, Any]:
        """Get summary of memories for an identity."""
        if identity not in self._memory_threads:
            return {"identity": identity, "memory_count": 0}
        
        memories = self._memory_threads[identity]
        
        return {
            "identity": identity,
            "memory_count": len(memories),
            "emotional_state": self._emotional_states.get(identity, 0.0),
            "memory_types": {mt.value: sum(1 for m in memories if m.memory_type == mt) 
                            for mt in MemoryType},
            "average_importance": np.mean([m.importance for m in memories]) if memories else 0.0,
        }
    
    def _update_emotional_state(self, identity: str, memory: MemoryEntry):
        """Update emotional state tracking."""
        current = self._emotional_states.get(identity, 0.0)
        self._emotional_states[identity] = 0.7 * current + 0.3 * memory.emotional_valence
    
    def _enforce_memory_limit(self):
        """Enforce maximum memory limit."""
        total = sum(len(m) for m in self._memory_threads.values())
        
        if total > self.max_memories:
            all_memories = []
            for identity, memories in self._memory_threads.items():
                for memory in memories:
                    all_memories.append((identity, memory))
            
            all_memories.sort(key=lambda x: x[1].importance + 
                              (0.5 if x[1].memory_type == MemoryType.GROUNDING else 0))
            
            to_remove = total - self.max_memories
            for identity, memory in all_memories[:to_remove]:
                self._memory_threads[identity].remove(memory)


# =============================================================================
# SECTION 7: GROUNDING TECHNIQUES
# =============================================================================

class GroundingCategory(Enum):
    """Categories of grounding techniques."""
    SENSORY = "sensory"
    COGNITIVE = "cognitive"
    PHYSICAL = "physical"
    EMOTIONAL = "emotional"


class IntensityLevel(Enum):
    """Intensity level of grounding techniques."""
    GENTLE = "gentle"
    MODERATE = "moderate"
    STRONG = "strong"


@dataclass
class GroundingTechnique:
    """A grounding technique with instructions."""
    technique_id: str
    name: str
    category: GroundingCategory
    intensity: IntensityLevel
    description: str
    instructions: List[str]
    duration_minutes: float


class GroundingTechniquesLibrary:
    """Library of grounding techniques."""
    
    def __init__(self):
        self._techniques: Dict[str, GroundingTechnique] = {}
        self._load_default_techniques()
    
    def _load_default_techniques(self):
        """Load default grounding techniques."""
        
        self._techniques["5-4-3-2-1"] = GroundingTechnique(
            technique_id="5-4-3-2-1",
            name="5-4-3-2-1 Sensory Grounding",
            category=GroundingCategory.SENSORY,
            intensity=IntensityLevel.GENTLE,
            description="Uses all five senses to anchor you in the present moment.",
            instructions=[
                "Take a slow, deep breath.",
                "Look around and name 5 things you can SEE.",
                "Notice 4 things you can TOUCH or feel.",
                "Listen for 3 things you can HEAR.",
                "Identify 2 things you can SMELL.",
                "Notice 1 thing you can TASTE.",
                "Take another deep breath and notice how you feel.",
            ],
            duration_minutes=3.0,
        )
        
        self._techniques["box_breathing"] = GroundingTechnique(
            technique_id="box_breathing",
            name="Box Breathing",
            category=GroundingCategory.PHYSICAL,
            intensity=IntensityLevel.GENTLE,
            description="Structured breathing to regulate the nervous system.",
            instructions=[
                "Find a comfortable position.",
                "Breathe IN slowly for 4 counts.",
                "HOLD your breath for 4 counts.",
                "Breathe OUT slowly for 4 counts.",
                "HOLD empty for 4 counts.",
                "Repeat this cycle 4-6 times.",
                "Return to normal breathing when ready.",
            ],
            duration_minutes=2.0,
        )
        
        self._techniques["safe_place"] = GroundingTechnique(
            technique_id="safe_place",
            name="Safe Place Visualization",
            category=GroundingCategory.EMOTIONAL,
            intensity=IntensityLevel.MODERATE,
            description="Create or recall a mental image of a safe place.",
            instructions=[
                "Close your eyes if comfortable, or soften your gaze.",
                "Imagine a place where you feel completely safe.",
                "This can be real or imaginary.",
                "Notice what you see in this place.",
                "What sounds are present?",
                "What does the air feel like?",
                "Allow yourself to feel the safety of this space.",
                "When ready, slowly return to the present.",
            ],
            duration_minutes=5.0,
        )
        
        self._techniques["categories"] = GroundingTechnique(
            technique_id="categories",
            name="Categories Mental Game",
            category=GroundingCategory.COGNITIVE,
            intensity=IntensityLevel.GENTLE,
            description="Cognitive exercise to interrupt distressing thoughts.",
            instructions=[
                "Choose a category (colors, animals, countries, etc.).",
                "Name items in that category for each letter of the alphabet.",
                "A is for... B is for... and so on.",
                "If you get stuck, skip to the next letter.",
                "Try to get through as many letters as you can.",
                "Notice how your mind shifts focus.",
            ],
            duration_minutes=3.0,
        )
    
    def get_technique(self, technique_id: str) -> Optional[GroundingTechnique]:
        """Get a specific technique by ID."""
        return self._techniques.get(technique_id)
    
    def get_techniques_for_state(self, entropy_state: EntropyState) -> List[GroundingTechnique]:
        """Get appropriate techniques for an entropy state."""
        if entropy_state == EntropyState.CRISIS:
            return [t for t in self._techniques.values() 
                    if t.intensity == IntensityLevel.GENTLE]
        elif entropy_state == EntropyState.HIGH:
            return [t for t in self._techniques.values() 
                    if t.intensity in [IntensityLevel.GENTLE, IntensityLevel.MODERATE]]
        else:
            return list(self._techniques.values())
    
    def list_all(self) -> List[GroundingTechnique]:
        """List all available techniques."""
        return list(self._techniques.values())


# =============================================================================
# SECTION 8: MIRRORLINK DIALOGUE COMPANION
# =============================================================================

class ReflectionType(Enum):
    """Types of reflections the system can provide."""
    CONTRADICTION_HOLDING = "contradiction_holding"
    VALIDATION = "validation"
    GROUNDING = "grounding"
    PERSPECTIVE = "perspective"
    MEMORY_BRIDGE = "memory_bridge"
    EMOTIONAL_NAMING = "emotional_naming"


class CommunicationStyle(Enum):
    """Communication style preferences."""
    DIRECT = "direct"
    GENTLE = "gentle"
    MINIMAL = "minimal"
    DETAILED = "detailed"


@dataclass
class Reflection:
    """A reflection generated by the system."""
    reflection_type: ReflectionType
    content: str
    current_context: str
    past_context: Optional[str]
    entropy_state: EntropyState
    confidence: float
    is_contradiction: bool
    follow_up_question: Optional[str] = None
    grounding_prompt: Optional[str] = None


class MirrorLinkDialogueCompanion:
    """
    MirrorLink Dialogue Companion (MLDC).
    
    Provides empathetic, trauma-informed conversational support.
    
    Core principle: Reflect contradictions without invalidation.
    Example: "You feel betrayed now, but you also called them your anchor 
    last week. Can both be real?"
    """
    
    def __init__(self, default_style: CommunicationStyle = CommunicationStyle.GENTLE,
                 include_grounding: bool = True):
        self.default_style = default_style
        self.include_grounding = include_grounding
        self._reflection_history: List[Reflection] = []
    
    def reflect(self, current_emotion: str, past_context: str = None,
                entropy_state: EntropyState = EntropyState.STABLE,
                style: CommunicationStyle = None) -> Reflection:
        """Generate a reflection based on current emotion and past context."""
        style = style or self.default_style
        
        is_contradiction = self._detect_contradiction(current_emotion, past_context)
        
        if is_contradiction:
            reflection = self._generate_contradiction_reflection(
                current_emotion, past_context, style, entropy_state
            )
        else:
            reflection = self._generate_validation_reflection(
                current_emotion, style, entropy_state
            )
        
        if self.include_grounding and entropy_state in [EntropyState.HIGH, EntropyState.CRISIS]:
            reflection.grounding_prompt = self._generate_grounding_prompt(entropy_state)
        
        self._reflection_history.append(reflection)
        return reflection
    
    def _detect_contradiction(self, current: str, past: str) -> bool:
        """Detect if there's a contradiction between current and past."""
        if not past:
            return False
        
        current_sentiment = self._analyze_sentiment(current)
        past_sentiment = self._analyze_sentiment(past)
        
        return abs(current_sentiment - past_sentiment) > 0.5
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text. Returns -1 to 1."""
        positive = {"love", "safe", "happy", "trust", "good", "calm", "peace"}
        negative = {"hate", "unsafe", "angry", "betrayed", "hurt", "bad", "scared"}
        
        words = text.lower().split()
        pos = sum(1 for w in words if w in positive)
        neg = sum(1 for w in words if w in negative)
        
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total
    
    def _generate_contradiction_reflection(self, current: str, past: str,
                                            style: CommunicationStyle,
                                            entropy: EntropyState) -> Reflection:
        """Generate a reflection that holds contradictory truths."""
        templates = {
            CommunicationStyle.DIRECT: (
                f"You feel {current} now, but {past}. "
                "Both can be true. What might explain this difference?"
            ),
            CommunicationStyle.GENTLE: (
                f"Right now, you're experiencing {current}. "
                f"I also remember that {past}. "
                "It's okay for both of these to be real at the same time. "
                "Would you like to explore what's different now?"
            ),
            CommunicationStyle.MINIMAL: (
                f"Now: {current}. Before: {past}. Both real. What changed?"
            ),
            CommunicationStyle.DETAILED: (
                f"I notice you're feeling {current} right now. "
                f"This seems different from before, where {past}. "
                "In trauma recovery, feelings can shift dramatically. "
                "Both experiences are valid. "
                "Would it help to look at what triggered this shift?"
            ),
        }
        
        content = templates.get(style, templates[CommunicationStyle.GENTLE])
        
        if entropy in [EntropyState.HIGH, EntropyState.CRISIS]:
            content = self._simplify_for_crisis(content)
        
        return Reflection(
            reflection_type=ReflectionType.CONTRADICTION_HOLDING,
            content=content,
            current_context=current,
            past_context=past,
            entropy_state=entropy,
            confidence=0.8,
            is_contradiction=True,
            follow_up_question="What feels most true for you right now?",
        )
    
    def _generate_validation_reflection(self, current: str,
                                         style: CommunicationStyle,
                                         entropy: EntropyState) -> Reflection:
        """Generate a validating reflection."""
        templates = {
            CommunicationStyle.DIRECT: f"Your feeling of {current} makes sense.",
            CommunicationStyle.GENTLE: (
                f"I hear that you're feeling {current}. "
                "That's a valid response to what you're experiencing."
            ),
            CommunicationStyle.MINIMAL: f"Feeling {current}. Valid.",
            CommunicationStyle.DETAILED: (
                f"You're experiencing {current} right now. "
                "This is a natural response. Your feelings are valid."
            ),
        }
        
        content = templates.get(style, templates[CommunicationStyle.GENTLE])
        
        return Reflection(
            reflection_type=ReflectionType.VALIDATION,
            content=content,
            current_context=current,
            past_context=None,
            entropy_state=entropy,
            confidence=0.9,
            is_contradiction=False,
        )
    
    def _simplify_for_crisis(self, content: str) -> str:
        """Simplify content for crisis states."""
        sentences = content.split(". ")
        if len(sentences) > 2:
            return ". ".join(sentences[:2]) + "."
        return content
    
    def _generate_grounding_prompt(self, entropy: EntropyState) -> str:
        """Generate a grounding prompt."""
        if entropy == EntropyState.CRISIS:
            return "Take a slow breath. Feel your feet on the ground. You are safe right now."
        else:
            return "If you need to, take a moment to ground yourself. What can you see around you?"


# =============================================================================
# SECTION 9: MAIN REUNITY CLASS
# =============================================================================

class ReUnity:
    """
    Main ReUnity AI Model class.
    
    Integrates all components:
    - Entropy analysis
    - State routing
    - Protective pattern recognition
    - Memory continuity
    - Grounding techniques
    - MirrorLink dialogue
    
    DISCLAIMER: This is NOT a clinical or treatment tool.
    """
    
    def __init__(self):
        self.entropy_analyzer = EntropyAnalyzer()
        self.state_router = StateRouter()
        self.pattern_recognizer = ProtectivePatternRecognizer()
        self.memory_engine = RecursiveIdentityMemoryEngine()
        self.grounding_library = GroundingTechniquesLibrary()
        self.dialogue_companion = MirrorLinkDialogueCompanion()
        
        self._current_identity = "primary"
        self._session_start = time.time()
    
    def process_input(self, text: str, identity: str = None) -> Dict[str, Any]:
        """
        Process user input through the full ReUnity pipeline.
        
        Args:
            text: User input text
            identity: Optional identity state identifier
        
        Returns:
            Dictionary with analysis results and recommended response
        """
        identity = identity or self._current_identity
        
        # Step 1: Analyze entropy
        entropy_metrics = self.entropy_analyzer.analyze_text(text)
        stability_metrics = self.entropy_analyzer.get_stability()
        
        # Step 2: Route to appropriate policy
        routing_decision = self.state_router.route(entropy_metrics, stability_metrics)
        
        # Step 3: Check for harmful patterns
        detected_patterns = self.pattern_recognizer.analyze_text(text)
        
        # Step 4: Store in memory
        memory = self.memory_engine.add_memory(
            identity=identity,
            content=text,
            memory_type=MemoryType.EPISODIC,
            entropy=entropy_metrics.normalized_entropy,
            emotional_valence=self._estimate_valence(text),
        )
        
        # Step 5: Get grounding techniques if needed
        grounding_techniques = []
        if entropy_metrics.state in [EntropyState.HIGH, EntropyState.CRISIS]:
            grounding_techniques = self.grounding_library.get_techniques_for_state(
                entropy_metrics.state
            )
        
        # Step 6: Generate reflection
        past_context = self._get_relevant_past_context(identity, text)
        reflection = self.dialogue_companion.reflect(
            current_emotion=text,
            past_context=past_context,
            entropy_state=entropy_metrics.state,
        )
        
        return {
            "entropy": {
                "state": entropy_metrics.state.value,
                "normalized": entropy_metrics.normalized_entropy,
                "confidence": entropy_metrics.confidence,
            },
            "stability": {
                "lyapunov": stability_metrics.lyapunov_exponent,
                "is_stable": stability_metrics.is_stable,
                "trend": stability_metrics.stability_trend,
            },
            "policy": {
                "mode": routing_decision.policy.mode.value,
                "actions": routing_decision.recommended_actions,
                "warnings": routing_decision.warnings,
            },
            "patterns": [
                {
                    "type": p.pattern_type.value,
                    "severity": p.severity.value,
                    "message": p.message,
                    "recommendation": p.recommendation,
                }
                for p in detected_patterns
            ],
            "reflection": {
                "content": reflection.content,
                "type": reflection.reflection_type.value,
                "grounding_prompt": reflection.grounding_prompt,
            },
            "grounding_techniques": [
                {"name": t.name, "instructions": t.instructions}
                for t in grounding_techniques[:3]
            ],
            "memory_id": memory.id,
        }
    
    def _estimate_valence(self, text: str) -> float:
        """Estimate emotional valence of text."""
        positive = {"love", "happy", "safe", "good", "calm", "peace", "joy"}
        negative = {"hate", "angry", "scared", "hurt", "sad", "fear", "pain"}
        
        words = text.lower().split()
        pos = sum(1 for w in words if w in positive)
        neg = sum(1 for w in words if w in negative)
        
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    
    def _get_relevant_past_context(self, identity: str, current: str) -> Optional[str]:
        """Get relevant past context for reflection."""
        memories = self.memory_engine.retrieve_grounding(
            current_identity=identity,
            query=current,
            max_results=1,
        )
        
        if memories:
            return memories[0].content
        return None
    
    def get_crisis_resources(self) -> Dict[str, str]:
        """Get crisis resources."""
        return {
            "National Suicide Prevention Lifeline": "988 (US)",
            "Crisis Text Line": "Text HOME to 741741 (US)",
            "International Association for Suicide Prevention": "https://www.iasp.info/",
        }


# =============================================================================
# SECTION 10: MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for testing."""
    print("=" * 60)
    print("ReUnity AI Model - Test Mode")
    print("=" * 60)
    print()
    print("DISCLAIMER: This is NOT a clinical or treatment tool.")
    print("It is a theoretical and support framework only.")
    print()
    
    # Initialize ReUnity
    reunity = ReUnity()
    
    # Test inputs
    test_inputs = [
        "I feel calm and peaceful today",
        "I'm feeling anxious and scared about everything",
        "They told me I'm crazy and imagining things",
        "I feel so alone, no one understands me",
    ]
    
    for text in test_inputs:
        print("-" * 60)
        print(f"Input: {text}")
        print()
        
        result = reunity.process_input(text)
        
        print(f"Entropy State: {result['entropy']['state']}")
        print(f"Entropy Value: {result['entropy']['normalized']:.3f}")
        print(f"Policy Mode: {result['policy']['mode']}")
        print(f"Actions: {', '.join(result['policy']['actions'])}")
        
        if result['patterns']:
            print(f"Patterns Detected: {len(result['patterns'])}")
            for p in result['patterns']:
                print(f"  - {p['type']}: {p['message']}")
        
        print(f"Reflection: {result['reflection']['content']}")
        
        if result['reflection']['grounding_prompt']:
            print(f"Grounding: {result['reflection']['grounding_prompt']}")
        
        print()
    
    print("=" * 60)
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## STEP 6: Run the Model

In the terminal, run:

```bash
python reunity_model.py
```

You should see output showing the model processing test inputs.

---

## STEP 7: Use the Model in Your Own Code

To use ReUnity in your own code:

```python
from reunity_model import ReUnity

# Create instance
reunity = ReUnity()

# Process user input
result = reunity.process_input("I feel scared and alone")

# Access results
print(result['entropy']['state'])        # e.g., "elevated"
print(result['policy']['mode'])           # e.g., "supportive"
print(result['reflection']['content'])    # The AI's response
```

---

## What Each Section Does

| Section | Purpose |
|---------|---------|
| 1. Entropy States | Defines the 5 emotional states (LOW, STABLE, ELEVATED, HIGH, CRISIS) |
| 2. Entropy Calculations | Shannon entropy, JS divergence, Lyapunov exponents |
| 3. Entropy Analyzer | Analyzes text to determine emotional state |
| 4. State Router | Selects appropriate response policy based on state |
| 5. Pattern Recognition | Detects gaslighting, love bombing, isolation attempts |
| 6. Memory Store | RIME engine for identity continuity |
| 7. Grounding Techniques | 5-4-3-2-1, box breathing, safe place, etc. |
| 8. MirrorLink Dialogue | Generates empathetic reflections |
| 9. Main ReUnity Class | Integrates all components |
| 10. Entry Point | Test function to verify everything works |

---

## Crisis Resources

If you or someone you know is in crisis:

- **National Suicide Prevention Lifeline**: 988 (US)
- **Crisis Text Line**: Text HOME to 741741 (US)
- **International Association for Suicide Prevention**: https://www.iasp.info/

---

## Contact

For questions, contact Christopher Ezernack.

---

*Author: Christopher Ezernack*
*Version: 1.0.0*
