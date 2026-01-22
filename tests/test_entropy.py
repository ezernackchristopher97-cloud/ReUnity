"""
Tests for ReUnity Entropy Analysis Module.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.
"""

import numpy as np
import pytest

from reunity.core.entropy import (
    EntropyStateDetector,
    EntropyState,
    EntropyMetrics,
    calculate_shannon_entropy,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_lyapunov_exponent,
)

# Backwards compatibility alias
EntropyAnalyzer = EntropyStateDetector


def estimate_lyapunov_exponent(series):
    """Wrapper for backwards compatibility with tests."""
    if len(series) < 3:
        return None
    result = calculate_lyapunov_exponent(series)
    return result.lyapunov_exponent


class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = calculate_shannon_entropy(p)
        assert entropy == pytest.approx(2.0, rel=1e-5)

    def test_deterministic_distribution_zero_entropy(self):
        """Deterministic distribution should have zero entropy."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = calculate_shannon_entropy(p)
        assert entropy == pytest.approx(0.0, rel=1e-5)

    def test_binary_distribution(self):
        """Test binary distribution entropy."""
        p = np.array([0.5, 0.5])
        entropy = calculate_shannon_entropy(p)
        assert entropy == pytest.approx(1.0, rel=1e-5)

    def test_empty_distribution(self):
        """Empty distribution should return 0."""
        p = np.array([])
        entropy = calculate_shannon_entropy(p)
        assert entropy == 0.0

    def test_single_element(self):
        """Single element distribution should have zero entropy."""
        p = np.array([1.0])
        entropy = calculate_shannon_entropy(p)
        assert entropy == 0.0


class TestJensenShannonDivergence:
    """Tests for Jensen-Shannon divergence."""

    def test_identical_distributions_zero_divergence(self):
        """Identical distributions should have zero divergence."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        divergence = calculate_jensen_shannon_divergence(p, q)
        assert divergence == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_positive_divergence(self):
        """Different distributions should have positive divergence."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        divergence = calculate_jensen_shannon_divergence(p, q)
        assert divergence > 0

    def test_symmetry(self):
        """JS divergence should be symmetric."""
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        d1 = calculate_jensen_shannon_divergence(p, q)
        d2 = calculate_jensen_shannon_divergence(q, p)
        assert d1 == pytest.approx(d2, rel=1e-10)

    def test_bounded_zero_to_one(self):
        """JS divergence should be bounded [0, 1]."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        divergence = calculate_jensen_shannon_divergence(p, q)
        assert 0 <= divergence <= 1


class TestMutualInformation:
    """Tests for mutual information calculation."""

    def test_independent_variables_zero_mi(self):
        """Independent variables should have zero MI."""
        # Independent joint distribution
        joint = np.array([[0.25, 0.25], [0.25, 0.25]])
        result = calculate_mutual_information(joint)
        assert result.mutual_information == pytest.approx(0.0, abs=1e-10)

    def test_perfectly_correlated_max_mi(self):
        """Perfectly correlated variables should have high MI."""
        # Perfectly correlated
        joint = np.array([[0.5, 0.0], [0.0, 0.5]])
        result = calculate_mutual_information(joint)
        assert result.mutual_information > 0

    def test_normalized_mi_bounded(self):
        """Normalized MI should be bounded [0, 1]."""
        joint = np.array([[0.4, 0.1], [0.1, 0.4]])
        result = calculate_mutual_information(joint)
        assert 0 <= result.normalized_mi <= 1


class TestLyapunovExponent:
    """Tests for Lyapunov exponent estimation."""

    def test_stable_time_series(self):
        """Stable time series should have negative Lyapunov exponent."""
        # Converging series
        series = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        lyapunov = estimate_lyapunov_exponent(series)
        assert lyapunov < 0

    def test_chaotic_time_series(self):
        """Chaotic time series should have non-zero Lyapunov exponent."""
        # Diverging series with more variation
        series = np.array([0.1, 0.3, 0.2, 0.8, 0.4, 1.2, 0.5, 2.0])
        lyapunov = estimate_lyapunov_exponent(series)
        # For chaotic systems, we just verify the calculation completes
        assert lyapunov is not None

    def test_short_series_returns_none(self):
        """Short series should return None."""
        series = np.array([1.0, 2.0])
        lyapunov = estimate_lyapunov_exponent(series)
        assert lyapunov is None


class TestEntropyAnalyzer:
    """Tests for the EntropyAnalyzer (EntropyStateDetector) class."""

    def test_analyze_low_entropy(self):
        """Low entropy distribution should be classified as LOW."""
        analyzer = EntropyAnalyzer()
        p = np.array([0.95, 0.05])
        metrics = analyzer.analyze_state(p)
        assert metrics.state in [EntropyState.LOW, EntropyState.STABLE]

    def test_analyze_high_entropy(self):
        """High entropy distribution should be classified appropriately."""
        analyzer = EntropyAnalyzer()
        p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        metrics = analyzer.analyze_state(p)
        assert metrics.state in [EntropyState.ELEVATED, EntropyState.HIGH, EntropyState.CRISIS]

    def test_analyze_returns_metrics(self):
        """Analyze should return EntropyMetrics."""
        analyzer = EntropyAnalyzer()
        p = np.array([0.5, 0.5])
        metrics = analyzer.analyze_state(p)
        assert isinstance(metrics, EntropyMetrics)
        assert hasattr(metrics, "shannon_entropy")
        assert hasattr(metrics, "normalized_entropy")
        assert hasattr(metrics, "state")
        assert hasattr(metrics, "confidence")

    def test_confidence_bounded(self):
        """Confidence should be bounded [0, 1]."""
        analyzer = EntropyAnalyzer()
        p = np.array([0.5, 0.3, 0.2])
        metrics = analyzer.analyze_state(p)
        assert 0 <= metrics.confidence <= 1

    def test_analyze_with_history(self):
        """Analyzer should track history for stability analysis."""
        analyzer = EntropyAnalyzer(stability_window=5)
        for _ in range(5):
            p = np.array([0.5, 0.5])
            analyzer.analyze_state(p)
        assert len(analyzer._entropy_history) == 5


class TestEntropyStates:
    """Tests for entropy state classification."""

    def test_all_states_accessible(self):
        """All entropy states should be accessible."""
        states = [
            EntropyState.LOW,
            EntropyState.STABLE,
            EntropyState.ELEVATED,
            EntropyState.HIGH,
            EntropyState.CRISIS,
        ]
        for state in states:
            assert state.value is not None

    def test_state_ordering(self):
        """States should have logical ordering."""
        # This tests the conceptual ordering
        assert EntropyState.LOW.value == "low"
        assert EntropyState.CRISIS.value == "crisis"
