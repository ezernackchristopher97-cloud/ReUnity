"""
ReUnity Core Entropy Module

This module implements the mathematical foundations for entropy-based emotional
state detection, including Shannon entropy, Jensen-Shannon divergence, mutual
information, and Lyapunov exponents for system stability analysis.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.

Mathematical Foundations:
- Shannon Entropy: S = -Σ p_i * log_2(p_i)
- Jensen-Shannon Divergence: JS(P,Q) = (1/2)*D_KL(P||M) + (1/2)*D_KL(Q||M)
- Mutual Information: MI(X;Y) = Σ p(x,y) * log_2(p(x,y) / (p(x)*p(y)))
- Lyapunov Exponents: λ = lim(n→∞) (1/n) * Σ log_2|dS/dt|

Author: Christopher Ezernack
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EntropyState(Enum):
    """Enumeration of entropy-based emotional states."""

    LOW = "low"  # Emotional rigidity or suppression
    STABLE = "stable"  # Healthy emotional range
    ELEVATED = "elevated"  # Increased emotional variability
    HIGH = "high"  # Significant fragmentation
    CRISIS = "crisis"  # Crisis-level instability


@dataclass
class EntropyMetrics:
    """Container for entropy-related metrics."""

    shannon_entropy: float
    normalized_entropy: float
    state: EntropyState
    confidence: float
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DivergenceMetrics:
    """Container for divergence metrics between states."""

    js_divergence: float
    kl_divergence_pq: float
    kl_divergence_qp: float
    symmetric: bool = True
    transition_detected: bool = False


@dataclass
class MutualInformationMetrics:
    """Container for mutual information metrics."""

    mutual_information: float
    normalized_mi: float
    dependency_strength: str  # "none", "weak", "moderate", "strong"


@dataclass
class StabilityMetrics:
    """Container for Lyapunov stability metrics."""

    lyapunov_exponent: float
    is_stable: bool
    is_chaotic: bool
    stability_trend: str  # "improving", "stable", "degrading"
    confidence_interval: tuple[float, float] = (0.0, 0.0)


# Constants
EPSILON = 1e-10  # Small value to prevent log(0) errors
DEFAULT_ENTROPY_THRESHOLDS = {
    EntropyState.LOW: 0.3,
    EntropyState.STABLE: 0.5,
    EntropyState.ELEVATED: 0.7,
    EntropyState.HIGH: 0.85,
    EntropyState.CRISIS: 1.0,
}


def calculate_shannon_entropy(probabilities: NDArray[np.floating]) -> float:
    """
    Calculate Shannon entropy for emotional state distribution.

    The Shannon entropy measures the uncertainty or randomness in the emotional
    state distribution. Higher entropy indicates greater emotional fragmentation
    and instability, while lower values may indicate emotional rigidity or
    suppression.

    Formula: S = -Σ(i=1 to n) p_i * log_2(p_i)

    Args:
        probabilities: Array of emotional state probabilities. Must sum to 1.

    Returns:
        Shannon entropy value in bits.

    Raises:
        ValueError: If probabilities are invalid (negative or don't sum to ~1).

    Example:
        >>> probs = np.array([0.4, 0.3, 0.3])
        >>> entropy = calculate_shannon_entropy(probs)
        >>> print(f"Entropy: {entropy:.3f} bits")
    """
    # Input validation
    if len(probabilities) == 0:
        return 0.0

    probabilities = np.asarray(probabilities, dtype=np.float64)

    # Check for negative probabilities
    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative")

    # Normalize probabilities to ensure they sum to 1
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        return 0.0

    # Handle zero probabilities with epsilon to avoid log(0)
    p_safe = np.where(probabilities > 0, probabilities, EPSILON)

    # Calculate Shannon entropy: S = -Σ p_i * log_2(p_i)
    entropy = -np.sum(probabilities * np.log2(p_safe))

    return float(entropy)


def calculate_normalized_entropy(
    probabilities: NDArray[np.floating],
) -> float:
    """
    Calculate normalized Shannon entropy (0 to 1 scale).

    Normalizes entropy by dividing by the maximum possible entropy for the
    given number of states (log_2(n)).

    Args:
        probabilities: Array of emotional state probabilities.

    Returns:
        Normalized entropy value between 0 and 1.
    """
    n_states = len(probabilities)
    if n_states <= 1:
        return 0.0

    entropy = calculate_shannon_entropy(probabilities)
    max_entropy = np.log2(n_states)

    if max_entropy == 0:
        return 0.0

    return float(entropy / max_entropy)


def calculate_kl_divergence(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> float:
    """
    Calculate Kullback-Leibler divergence D_KL(P||Q).

    The KL divergence measures how one probability distribution diverges from
    a reference distribution. It is asymmetric: D_KL(P||Q) ≠ D_KL(Q||P).

    Formula: D_KL(P||Q) = Σ p(x) * log_2(p(x) / q(x))

    Args:
        p: First probability distribution (reference).
        q: Second probability distribution.

    Returns:
        KL divergence value (non-negative).

    Raises:
        ValueError: If distributions have different lengths.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if len(p) != len(q):
        raise ValueError("Distributions must have the same length")

    # Normalize distributions
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    # Handle zeros with epsilon
    p_safe = np.where(p > 0, p, EPSILON)
    q_safe = np.where(q > 0, q, EPSILON)

    # Calculate KL divergence
    kl_div = np.sum(np.where(p > 0, p * np.log2(p_safe / q_safe), 0))

    return float(max(0.0, kl_div))


def calculate_jensen_shannon_divergence(
    p: NDArray[np.floating],
    q: NDArray[np.floating],
) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    The JS divergence is a symmetric measure of the difference between two
    probability distributions, making it ideal for tracking state transitions
    without bias toward particular emotional configurations.

    Formula: JS(P,Q) = (1/2)*D_KL(P||M) + (1/2)*D_KL(Q||M)
    where M = (1/2)*(P + Q)

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        Jensen-Shannon divergence value (0 = identical, 1 = completely different).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize distributions
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q

    # Calculate midpoint distribution M = (P + Q) / 2
    m = 0.5 * (p + q)

    # Calculate JS divergence
    js_div = 0.5 * calculate_kl_divergence(p, m) + 0.5 * calculate_kl_divergence(q, m)

    return float(js_div)


def compare_emotional_states(
    state1_probs: NDArray[np.floating],
    state2_probs: NDArray[np.floating],
    transition_threshold: float = 0.3,
) -> DivergenceMetrics:
    """
    Compare two emotional state distributions using JS divergence.

    This function detects when someone is transitioning between dramatically
    different psychological states, indicating potential splitting episodes
    or dissociative periods that require additional support.

    Args:
        state1_probs: First emotional state probability distribution.
        state2_probs: Second emotional state probability distribution.
        transition_threshold: Threshold for detecting significant transitions.

    Returns:
        DivergenceMetrics containing divergence values and transition detection.
    """
    js_div = calculate_jensen_shannon_divergence(state1_probs, state2_probs)
    kl_pq = calculate_kl_divergence(state1_probs, state2_probs)
    kl_qp = calculate_kl_divergence(state2_probs, state1_probs)

    return DivergenceMetrics(
        js_divergence=js_div,
        kl_divergence_pq=kl_pq,
        kl_divergence_qp=kl_qp,
        symmetric=True,
        transition_detected=js_div > transition_threshold,
    )


def calculate_mutual_information(
    joint_distribution: NDArray[np.floating],
) -> MutualInformationMetrics:
    """
    Calculate mutual information from a joint probability distribution.

    Mutual information measures the amount of information obtained about one
    emotional variable through observing another, enabling the system to
    understand how different aspects of emotional experience influence each other.

    Formula: MI(X;Y) = Σ p(x,y) * log_2(p(x,y) / (p(x)*p(y)))

    Alternative: MI(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

    Args:
        joint_distribution: 2D array representing joint probability p(x,y).

    Returns:
        MutualInformationMetrics containing MI value and dependency assessment.
    """
    joint = np.asarray(joint_distribution, dtype=np.float64)

    if joint.ndim != 2:
        raise ValueError("Joint distribution must be 2-dimensional")

    # Normalize joint distribution
    joint_sum = np.sum(joint)
    if joint_sum > 0:
        joint = joint / joint_sum
    else:
        return MutualInformationMetrics(
            mutual_information=0.0,
            normalized_mi=0.0,
            dependency_strength="none",
        )

    # Compute marginal distributions
    p_x = np.sum(joint, axis=1)  # Sum over y
    p_y = np.sum(joint, axis=0)  # Sum over x

    # Calculate mutual information
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > EPSILON and p_x[i] > EPSILON and p_y[j] > EPSILON:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j]))

    # Calculate entropies for normalization
    h_x = calculate_shannon_entropy(p_x)
    h_y = calculate_shannon_entropy(p_y)
    max_mi = min(h_x, h_y) if min(h_x, h_y) > 0 else 1.0

    normalized_mi = mi / max_mi if max_mi > 0 else 0.0

    # Determine dependency strength
    if normalized_mi < 0.1:
        strength = "none"
    elif normalized_mi < 0.3:
        strength = "weak"
    elif normalized_mi < 0.6:
        strength = "moderate"
    else:
        strength = "strong"

    return MutualInformationMetrics(
        mutual_information=float(mi),
        normalized_mi=float(normalized_mi),
        dependency_strength=strength,
    )


def calculate_mutual_information_from_marginals(
    p_x: NDArray[np.floating],
    p_y: NDArray[np.floating],
    p_xy: NDArray[np.floating],
) -> float:
    """
    Calculate mutual information from marginal and joint distributions.

    Args:
        p_x: Marginal distribution of X.
        p_y: Marginal distribution of Y.
        p_xy: Joint distribution p(x,y) as 2D array.

    Returns:
        Mutual information value in bits.
    """
    p_x = np.asarray(p_x, dtype=np.float64)
    p_y = np.asarray(p_y, dtype=np.float64)
    p_xy = np.asarray(p_xy, dtype=np.float64)

    # Normalize
    p_x = p_x / np.sum(p_x) if np.sum(p_x) > 0 else p_x
    p_y = p_y / np.sum(p_y) if np.sum(p_y) > 0 else p_y
    p_xy = p_xy / np.sum(p_xy) if np.sum(p_xy) > 0 else p_xy

    mi = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > EPSILON and p_x[i] > EPSILON and p_y[j] > EPSILON:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

    return float(max(0.0, mi))


def calculate_lyapunov_exponent(
    state_sequence: NDArray[np.floating],
    delta_t: float = 1.0,
) -> StabilityMetrics:
    """
    Calculate Lyapunov exponent for emotional state stability analysis.

    The Lyapunov exponent quantifies the rate of divergence of nearby trajectories
    in the emotional state space. Positive exponents indicate chaotic, unpredictable
    behavior patterns, while negative exponents suggest stable, convergent dynamics
    that may indicate successful therapeutic progress.

    Formula: λ = lim(n→∞) (1/n) * Σ log_2|dS/dt|

    Practical approximation: dS/dt ≈ (S_t - S_{t-1}) / Δt

    Interpretation:
    - λ > 0: Chaos/instability (crisis intervention may be needed)
    - λ < 0: Stability (therapeutic progress)
    - λ ≈ 0: Marginal stability

    Args:
        state_sequence: Time series of emotional state values.
        delta_t: Time step between observations.

    Returns:
        StabilityMetrics containing Lyapunov exponent and stability assessment.
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

    # Calculate derivatives (finite differences)
    derivatives = np.diff(states) / delta_t

    # Handle zero derivatives with epsilon
    derivatives_safe = np.where(
        np.abs(derivatives) > EPSILON,
        np.abs(derivatives),
        EPSILON,
    )

    # Calculate log sensitivities
    log_sensitivities = np.log2(derivatives_safe)

    # Calculate Lyapunov exponent as average
    lyapunov = float(np.mean(log_sensitivities))

    # Calculate confidence interval using bootstrap-like approach
    if len(log_sensitivities) >= 10:
        std_err = np.std(log_sensitivities) / np.sqrt(len(log_sensitivities))
        ci_low = lyapunov - 1.96 * std_err
        ci_high = lyapunov + 1.96 * std_err
    else:
        ci_low = lyapunov - 0.5
        ci_high = lyapunov + 0.5

    # Determine stability characteristics
    is_stable = lyapunov < 0
    is_chaotic = lyapunov > 0.5

    # Determine trend by comparing recent vs earlier values
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


def classify_entropy_state(
    entropy: float,
    thresholds: dict[EntropyState, float] | None = None,
) -> EntropyState:
    """
    Classify entropy value into an emotional state category.

    Args:
        entropy: Normalized entropy value (0 to 1).
        thresholds: Custom thresholds for state classification.

    Returns:
        EntropyState classification.
    """
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


def emotional_state_entropy(emotional_states: list[str]) -> float:
    """
    Calculate entropy for a sequence of emotional state labels.

    This is a convenience function that converts text labels to probabilities
    and calculates entropy.

    Args:
        emotional_states: List of emotional state labels (e.g., ["anxious", "calm"]).

    Returns:
        Entropy of the emotional state distribution in bits.

    Example:
        >>> states = ["stable", "anxious", "stable", "fragmented", "stable"]
        >>> entropy = emotional_state_entropy(states)
    """
    if not emotional_states:
        return 0.0

    # Count frequencies
    unique_states, counts = np.unique(emotional_states, return_counts=True)

    # Convert to probabilities
    probabilities = counts / len(emotional_states)

    return calculate_shannon_entropy(probabilities)


class EntropyStateDetector:
    """
    Entropy-Based Emotional State Analyzer (EESA).

    This class continuously monitors emotional entropy levels using the
    mathematical frameworks defined above. When entropy exceeds threshold
    values, the system activates protective protocols designed to prevent
    crisis escalation while maintaining user autonomy and choice.

    The analyzer processes multiple data streams including text input,
    behavioral patterns, and maintains baseline entropy profiles for each
    user, triggering alerts when current entropy levels exceed personalized
    thresholds that indicate increased risk of fragmentation or crisis.

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        baseline_entropy: float = 0.5,
        crisis_threshold: float = 0.85,
        stability_window: int = 10,
        custom_thresholds: dict[EntropyState, float] | None = None,
    ) -> None:
        """
        Initialize the entropy state detector.

        Args:
            baseline_entropy: Expected baseline entropy for the user.
            crisis_threshold: Threshold for crisis-level entropy.
            stability_window: Number of observations for stability analysis.
            custom_thresholds: Custom state classification thresholds.
        """
        self.baseline_entropy = baseline_entropy
        self.crisis_threshold = crisis_threshold
        self.stability_window = stability_window
        self.thresholds = custom_thresholds or DEFAULT_ENTROPY_THRESHOLDS

        # State history for tracking
        self._entropy_history: list[float] = []
        self._state_history: list[EntropyState] = []
        self._timestamp_history: list[float] = []

    def analyze_state(
        self,
        probabilities: NDArray[np.floating],
        timestamp: float = 0.0,
    ) -> EntropyMetrics:
        """
        Analyze emotional state from probability distribution.

        Args:
            probabilities: Current emotional state probabilities.
            timestamp: Current timestamp for tracking.

        Returns:
            EntropyMetrics with analysis results.
        """
        entropy = calculate_shannon_entropy(probabilities)
        normalized = calculate_normalized_entropy(probabilities)
        state = classify_entropy_state(normalized, self.thresholds)

        # Calculate confidence based on deviation from baseline
        deviation = abs(normalized - self.baseline_entropy)
        confidence = max(0.0, 1.0 - deviation)

        # Update history
        self._entropy_history.append(normalized)
        self._state_history.append(state)
        self._timestamp_history.append(timestamp)

        # Trim history to window size
        if len(self._entropy_history) > self.stability_window * 2:
            self._entropy_history = self._entropy_history[-self.stability_window * 2 :]
            self._state_history = self._state_history[-self.stability_window * 2 :]
            self._timestamp_history = self._timestamp_history[-self.stability_window * 2 :]

        return EntropyMetrics(
            shannon_entropy=entropy,
            normalized_entropy=normalized,
            state=state,
            confidence=confidence,
            timestamp=timestamp,
            metadata={
                "baseline_deviation": deviation,
                "above_crisis_threshold": normalized > self.crisis_threshold,
            },
        )

    def analyze_text_entropy(
        self,
        text: str,
        timestamp: float = 0.0,
    ) -> EntropyMetrics:
        """
        Analyze entropy from text content.

        Simplified entropy calculation from text features.

        Args:
            text: Input text to analyze.
            timestamp: Current timestamp.

        Returns:
            EntropyMetrics with analysis results.
        """
        if not text.strip():
            return EntropyMetrics(
                shannon_entropy=0.0,
                normalized_entropy=0.0,
                state=EntropyState.LOW,
                confidence=0.0,
                timestamp=timestamp,
            )

        # Tokenize and count word frequencies
        words = text.lower().split()
        unique_words, counts = np.unique(words, return_counts=True)
        probabilities = counts / len(words)

        return self.analyze_state(probabilities, timestamp)

    def detect_transition(
        self,
        current_probs: NDArray[np.floating],
        previous_probs: NDArray[np.floating],
        threshold: float = 0.3,
    ) -> DivergenceMetrics:
        """
        Detect state transition using Jensen-Shannon divergence.

        Args:
            current_probs: Current state probabilities.
            previous_probs: Previous state probabilities.
            threshold: Transition detection threshold.

        Returns:
            DivergenceMetrics with transition analysis.
        """
        return compare_emotional_states(current_probs, previous_probs, threshold)

    def assess_stability(self) -> StabilityMetrics:
        """
        Assess overall system stability using Lyapunov analysis.

        Returns:
            StabilityMetrics for the current entropy history.
        """
        if len(self._entropy_history) < 3:
            return StabilityMetrics(
                lyapunov_exponent=0.0,
                is_stable=True,
                is_chaotic=False,
                stability_trend="stable",
            )

        return calculate_lyapunov_exponent(
            np.array(self._entropy_history[-self.stability_window :])
        )

    def get_crisis_risk(self) -> float:
        """
        Calculate current crisis risk level (0 to 1).

        Returns:
            Crisis risk score based on recent entropy patterns.
        """
        if not self._entropy_history:
            return 0.0

        recent = self._entropy_history[-min(5, len(self._entropy_history)) :]
        avg_entropy = np.mean(recent)

        # Risk increases with entropy above baseline
        risk = max(0.0, (avg_entropy - self.baseline_entropy) / (1.0 - self.baseline_entropy))

        # Increase risk if trending upward
        if len(recent) >= 3:
            trend = recent[-1] - recent[0]
            if trend > 0:
                risk = min(1.0, risk + trend * 0.5)

        return float(min(1.0, risk))

    def reset_baseline(self, new_baseline: float) -> None:
        """
        Update the baseline entropy for the user.

        Args:
            new_baseline: New baseline entropy value.
        """
        self.baseline_entropy = new_baseline

    def clear_history(self) -> None:
        """Clear all stored history."""
        self._entropy_history.clear()
        self._state_history.clear()
        self._timestamp_history.clear()

    @property
    def entropy_history(self) -> list[float]:
        """Get entropy history."""
        return self._entropy_history.copy()

    @property
    def state_history(self) -> list[EntropyState]:
        """Get state history."""
        return self._state_history.copy()
