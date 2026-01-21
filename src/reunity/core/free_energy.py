"""
ReUnity Free Energy Principle Applications

This module implements the Free Energy Principle (FEP) for modeling
predictive processing in trauma-aware AI systems. The FEP provides
a mathematical framework for understanding how the system minimizes
surprise and maintains stability.

Key Concepts:
- Variational Free Energy: Upper bound on surprise
- Active Inference: Action selection to minimize expected free energy
- Precision Weighting: Confidence in predictions vs. sensory data
- Belief Updating: Bayesian inference for state estimation

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class BeliefState:
    """Represents a belief state with uncertainty."""

    mean: NDArray[np.floating]  # Expected value
    precision: NDArray[np.floating]  # Inverse variance (confidence)
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def variance(self) -> NDArray[np.floating]:
        """Get variance from precision."""
        return 1.0 / (self.precision + 1e-10)

    @property
    def entropy(self) -> float:
        """Calculate entropy of the belief distribution."""
        # For Gaussian: H = 0.5 * log(2 * pi * e * variance)
        return float(0.5 * np.sum(np.log(2 * np.pi * np.e * self.variance)))


@dataclass
class Observation:
    """Sensory observation with precision."""

    value: NDArray[np.floating]
    precision: NDArray[np.floating]  # Confidence in observation
    timestamp: float = 0.0


@dataclass
class FreeEnergyMetrics:
    """Container for free energy calculations."""

    variational_free_energy: float
    expected_free_energy: float
    surprise: float
    complexity: float
    accuracy: float
    precision_weighted_error: float


class FreeEnergyMinimizer:
    """
    Free Energy Principle implementation for predictive processing.

    The Free Energy Principle states that biological systems minimize
    variational free energy, which is an upper bound on surprise
    (negative log probability of observations).

    F = E_q[log q(s) - log p(o,s)]
      = D_KL[q(s) || p(s)] - E_q[log p(o|s)]
      = Complexity - Accuracy

    Where:
    - F = Variational Free Energy
    - q(s) = Approximate posterior (beliefs about hidden states)
    - p(o,s) = Generative model (joint probability of observations and states)
    - D_KL = Kullback-Leibler divergence

    DISCLAIMER: This is not a clinical or treatment tool.
    """

    def __init__(
        self,
        state_dim: int = 8,
        learning_rate: float = 0.1,
        precision_decay: float = 0.99,
    ) -> None:
        """
        Initialize the Free Energy Minimizer.

        Args:
            state_dim: Dimensionality of the state space.
            learning_rate: Learning rate for belief updates.
            precision_decay: Decay factor for precision over time.
        """
        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.precision_decay = precision_decay

        # Initialize beliefs with high uncertainty
        self._beliefs = BeliefState(
            mean=np.zeros(state_dim),
            precision=np.ones(state_dim) * 0.1,
        )

        # Prior beliefs (baseline)
        self._prior = BeliefState(
            mean=np.zeros(state_dim),
            precision=np.ones(state_dim) * 0.5,
        )

        # History for tracking
        self._observation_history: list[Observation] = []
        self._free_energy_history: list[float] = []

    def calculate_variational_free_energy(
        self,
        observation: Observation,
    ) -> FreeEnergyMetrics:
        """
        Calculate variational free energy given an observation.

        F = Complexity - Accuracy
        F = D_KL[q(s) || p(s)] - E_q[log p(o|s)]

        Args:
            observation: Current sensory observation.

        Returns:
            FreeEnergyMetrics containing all components.
        """
        # Complexity: KL divergence from prior
        complexity = self._calculate_kl_divergence(
            self._beliefs.mean,
            self._beliefs.precision,
            self._prior.mean,
            self._prior.precision,
        )

        # Prediction error
        prediction_error = observation.value - self._beliefs.mean

        # Precision-weighted prediction error
        precision_weighted_error = float(np.sum(
            observation.precision * prediction_error ** 2
        ))

        # Accuracy: Expected log likelihood (negative of weighted error)
        accuracy = -0.5 * precision_weighted_error

        # Variational Free Energy
        vfe = complexity - accuracy

        # Surprise: -log p(o) ≈ prediction error magnitude
        surprise = float(np.sum(prediction_error ** 2))

        # Expected Free Energy (for action selection)
        efe = self._calculate_expected_free_energy(observation)

        return FreeEnergyMetrics(
            variational_free_energy=vfe,
            expected_free_energy=efe,
            surprise=surprise,
            complexity=complexity,
            accuracy=accuracy,
            precision_weighted_error=precision_weighted_error,
        )

    def _calculate_kl_divergence(
        self,
        mean_q: NDArray[np.floating],
        precision_q: NDArray[np.floating],
        mean_p: NDArray[np.floating],
        precision_p: NDArray[np.floating],
    ) -> float:
        """
        Calculate KL divergence between two Gaussian distributions.

        D_KL[q || p] = 0.5 * (tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q)
                       - k + ln(|Σ_p| / |Σ_q|))
        """
        var_q = 1.0 / (precision_q + 1e-10)
        var_p = 1.0 / (precision_p + 1e-10)

        # For diagonal covariance
        trace_term = np.sum(precision_p * var_q)
        mean_diff = mean_p - mean_q
        mahalanobis = np.sum(precision_p * mean_diff ** 2)
        log_det_ratio = np.sum(np.log(var_p + 1e-10) - np.log(var_q + 1e-10))

        kl = 0.5 * (trace_term + mahalanobis - self.state_dim + log_det_ratio)
        return float(max(0.0, kl))

    def _calculate_expected_free_energy(
        self,
        observation: Observation,
    ) -> float:
        """
        Calculate expected free energy for active inference.

        G = E_q[log q(s') - log p(o', s')]
          = Expected Complexity - Expected Accuracy
          + Epistemic Value (information gain)
          + Pragmatic Value (goal achievement)
        """
        # Epistemic value: Expected information gain
        epistemic = float(np.sum(np.log(
            self._beliefs.precision / (self._beliefs.precision + observation.precision)
        )))

        # Pragmatic value: Distance from preferred state (prior)
        pragmatic = -float(np.sum(
            (self._beliefs.mean - self._prior.mean) ** 2
        ))

        return -(epistemic + pragmatic)

    def update_beliefs(
        self,
        observation: Observation,
    ) -> BeliefState:
        """
        Update beliefs using Bayesian inference.

        Implements precision-weighted belief updating:
        μ_new = μ_old + κ * π_o * (o - μ_old)

        Where:
        - κ = learning rate
        - π_o = observation precision
        - o = observation value

        Args:
            observation: New observation.

        Returns:
            Updated belief state.
        """
        # Prediction error
        error = observation.value - self._beliefs.mean

        # Precision-weighted update
        precision_ratio = observation.precision / (
            self._beliefs.precision + observation.precision
        )

        # Update mean
        new_mean = self._beliefs.mean + self.learning_rate * precision_ratio * error

        # Update precision (combine precisions)
        new_precision = self._beliefs.precision + observation.precision

        # Apply precision decay
        new_precision = new_precision * self.precision_decay

        # Update beliefs
        self._beliefs = BeliefState(
            mean=new_mean,
            precision=new_precision,
            timestamp=observation.timestamp,
        )

        # Store observation
        self._observation_history.append(observation)

        # Calculate and store free energy
        metrics = self.calculate_variational_free_energy(observation)
        self._free_energy_history.append(metrics.variational_free_energy)

        return self._beliefs

    def predict_next_state(
        self,
        time_horizon: int = 1,
    ) -> BeliefState:
        """
        Predict future state using generative model.

        Args:
            time_horizon: How many steps ahead to predict.

        Returns:
            Predicted belief state.
        """
        # Simple autoregressive prediction
        if len(self._observation_history) < 2:
            return self._beliefs

        # Calculate trend from recent observations
        recent = self._observation_history[-min(10, len(self._observation_history)):]
        if len(recent) >= 2:
            trend = (recent[-1].value - recent[0].value) / len(recent)
        else:
            trend = np.zeros(self.state_dim)

        # Predict future state
        predicted_mean = self._beliefs.mean + trend * time_horizon

        # Uncertainty increases with prediction horizon
        predicted_precision = self._beliefs.precision / (1 + 0.1 * time_horizon)

        return BeliefState(
            mean=predicted_mean,
            precision=predicted_precision,
            timestamp=self._beliefs.timestamp + time_horizon,
        )

    def select_action(
        self,
        possible_actions: list[NDArray[np.floating]],
    ) -> tuple[int, float]:
        """
        Select action that minimizes expected free energy.

        Active inference: Choose actions that minimize expected
        surprise and maximize information gain.

        Args:
            possible_actions: List of possible action vectors.

        Returns:
            Tuple of (action_index, expected_free_energy).
        """
        best_action = 0
        best_efe = float("inf")

        for i, action in enumerate(possible_actions):
            # Simulate effect of action on beliefs
            simulated_state = self._beliefs.mean + action

            # Calculate expected free energy for this action
            efe = self._evaluate_action_efe(simulated_state)

            if efe < best_efe:
                best_efe = efe
                best_action = i

        return best_action, best_efe

    def _evaluate_action_efe(
        self,
        simulated_state: NDArray[np.floating],
    ) -> float:
        """Evaluate expected free energy for a simulated state."""
        # Distance from preferred state
        pragmatic_cost = float(np.sum(
            (simulated_state - self._prior.mean) ** 2
        ))

        # Epistemic value (information gain potential)
        uncertainty = float(np.sum(1.0 / (self._beliefs.precision + 1e-10)))

        return pragmatic_cost - 0.5 * uncertainty

    def get_surprise_trajectory(self) -> list[float]:
        """Get history of surprise values."""
        return self._free_energy_history.copy()

    def reset_to_prior(self) -> None:
        """Reset beliefs to prior."""
        self._beliefs = BeliefState(
            mean=self._prior.mean.copy(),
            precision=self._prior.precision.copy(),
        )

    def set_prior(self, mean: NDArray[np.floating], precision: NDArray[np.floating]) -> None:
        """
        Set the prior belief state.

        Args:
            mean: Prior mean.
            precision: Prior precision.
        """
        self._prior = BeliefState(mean=mean, precision=precision)

    @property
    def current_beliefs(self) -> BeliefState:
        """Get current belief state."""
        return self._beliefs

    @property
    def current_uncertainty(self) -> float:
        """Get current total uncertainty."""
        return float(np.sum(1.0 / (self._beliefs.precision + 1e-10)))


class PredictiveProcessor:
    """
    Predictive Processing implementation for emotional state modeling.

    This class implements a hierarchical predictive processing model
    where higher levels predict lower-level states, and prediction
    errors propagate upward to update beliefs.

    DISCLAIMER: This is not a clinical or treatment tool.
    """

    def __init__(
        self,
        n_levels: int = 3,
        state_dims: list[int] | None = None,
    ) -> None:
        """
        Initialize hierarchical predictive processor.

        Args:
            n_levels: Number of hierarchical levels.
            state_dims: Dimensionality at each level.
        """
        self.n_levels = n_levels

        if state_dims is None:
            state_dims = [8, 4, 2]  # Default: decreasing dimensions

        self.state_dims = state_dims[:n_levels]

        # Create free energy minimizer for each level
        self._levels = [
            FreeEnergyMinimizer(state_dim=dim)
            for dim in self.state_dims
        ]

        # Precision weights for each level
        self._level_precisions = [1.0] * n_levels

    def process_observation(
        self,
        observation: NDArray[np.floating],
        observation_precision: float = 1.0,
    ) -> dict[str, Any]:
        """
        Process an observation through the hierarchy.

        Args:
            observation: Raw observation vector.
            observation_precision: Confidence in observation.

        Returns:
            Dictionary containing processing results at each level.
        """
        results = {
            "levels": [],
            "total_free_energy": 0.0,
            "prediction_errors": [],
        }

        current_input = observation
        current_precision = observation_precision

        for level_idx, level in enumerate(self._levels):
            # Create observation for this level
            obs = Observation(
                value=current_input[:level.state_dim] if len(current_input) >= level.state_dim
                else np.pad(current_input, (0, level.state_dim - len(current_input))),
                precision=np.ones(level.state_dim) * current_precision * self._level_precisions[level_idx],
            )

            # Calculate free energy
            metrics = level.calculate_variational_free_energy(obs)

            # Update beliefs
            updated_beliefs = level.update_beliefs(obs)

            # Store results
            results["levels"].append({
                "level": level_idx,
                "free_energy": metrics.variational_free_energy,
                "surprise": metrics.surprise,
                "beliefs_mean": updated_beliefs.mean.tolist(),
                "beliefs_precision": updated_beliefs.precision.tolist(),
            })

            results["total_free_energy"] += metrics.variational_free_energy
            results["prediction_errors"].append(metrics.precision_weighted_error)

            # Pass beliefs to next level as input
            current_input = updated_beliefs.mean
            current_precision = float(np.mean(updated_beliefs.precision))

        return results

    def get_integrated_state(self) -> NDArray[np.floating]:
        """
        Get integrated state across all levels.

        Returns:
            Weighted combination of beliefs at all levels.
        """
        states = []
        weights = []

        for level_idx, level in enumerate(self._levels):
            states.append(level.current_beliefs.mean)
            weights.append(self._level_precisions[level_idx])

        # Weighted average (pad to same size)
        max_dim = max(len(s) for s in states)
        padded_states = [
            np.pad(s, (0, max_dim - len(s)))
            for s in states
        ]

        total_weight = sum(weights)
        integrated = sum(
            w * s for w, s in zip(weights, padded_states)
        ) / total_weight

        return integrated

    def set_level_precision(self, level: int, precision: float) -> None:
        """
        Set precision weight for a hierarchical level.

        Higher precision = more influence on processing.

        Args:
            level: Level index.
            precision: Precision weight.
        """
        if 0 <= level < self.n_levels:
            self._level_precisions[level] = precision

    def get_total_uncertainty(self) -> float:
        """Get total uncertainty across all levels."""
        return sum(level.current_uncertainty for level in self._levels)
