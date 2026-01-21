"""
ReUnity State Router Module

This module implements the state router that selects appropriate policies and
response strategies based on the current entropy state. The router ensures
that system behavior adapts to the user's emotional state while maintaining
safety and supportive interactions.

DISCLAIMER: This is not a clinical or treatment document. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from reunity.core.entropy import EntropyMetrics, EntropyState, StabilityMetrics


class PolicyMode(Enum):
    """Operating modes for the system based on emotional state."""

    SUPPORTIVE = "supportive"  # Normal supportive interaction
    GROUNDING = "grounding"  # Grounding and stabilization focus
    PROTECTIVE = "protective"  # Protective mode with safety checks
    CRISIS = "crisis"  # Crisis intervention mode
    REFLECTIVE = "reflective"  # Deep reflection and processing mode


class ResponseConstraint(Enum):
    """Constraints on system responses."""

    FULL = "full"  # Full response capability
    SIMPLIFIED = "simplified"  # Simplified, clear responses
    GROUNDING_ONLY = "grounding_only"  # Only grounding responses
    SAFETY_FOCUSED = "safety_focused"  # Safety-focused responses only
    MINIMAL = "minimal"  # Minimal interaction


@dataclass
class PolicyConfig:
    """Configuration for a specific policy mode."""

    mode: PolicyMode
    response_constraint: ResponseConstraint
    max_response_length: int
    allow_deep_reflection: bool
    allow_memory_retrieval: bool
    allow_relationship_analysis: bool
    require_grounding_prompts: bool
    crisis_resources_visible: bool
    confidence_threshold: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision made by the state router."""

    policy: PolicyConfig
    entropy_state: EntropyState
    confidence: float
    reasoning: str
    recommended_actions: list[str]
    warnings: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# Default policy configurations for each entropy state
DEFAULT_POLICIES: dict[EntropyState, PolicyConfig] = {
    EntropyState.LOW: PolicyConfig(
        mode=PolicyMode.REFLECTIVE,
        response_constraint=ResponseConstraint.FULL,
        max_response_length=2000,
        allow_deep_reflection=True,
        allow_memory_retrieval=True,
        allow_relationship_analysis=True,
        require_grounding_prompts=False,
        crisis_resources_visible=False,
        confidence_threshold=0.7,
        description="Low entropy state - emotional rigidity or suppression. "
        "System enables deep reflection to explore emotional range.",
    ),
    EntropyState.STABLE: PolicyConfig(
        mode=PolicyMode.SUPPORTIVE,
        response_constraint=ResponseConstraint.FULL,
        max_response_length=1500,
        allow_deep_reflection=True,
        allow_memory_retrieval=True,
        allow_relationship_analysis=True,
        require_grounding_prompts=False,
        crisis_resources_visible=False,
        confidence_threshold=0.6,
        description="Stable entropy state - healthy emotional range. "
        "Full supportive interaction enabled.",
    ),
    EntropyState.ELEVATED: PolicyConfig(
        mode=PolicyMode.SUPPORTIVE,
        response_constraint=ResponseConstraint.SIMPLIFIED,
        max_response_length=1000,
        allow_deep_reflection=False,
        allow_memory_retrieval=True,
        allow_relationship_analysis=False,
        require_grounding_prompts=True,
        crisis_resources_visible=False,
        confidence_threshold=0.5,
        description="Elevated entropy state - increased emotional variability. "
        "Simplified responses with grounding prompts.",
    ),
    EntropyState.HIGH: PolicyConfig(
        mode=PolicyMode.GROUNDING,
        response_constraint=ResponseConstraint.GROUNDING_ONLY,
        max_response_length=500,
        allow_deep_reflection=False,
        allow_memory_retrieval=True,
        allow_relationship_analysis=False,
        require_grounding_prompts=True,
        crisis_resources_visible=True,
        confidence_threshold=0.4,
        description="High entropy state - significant fragmentation. "
        "Focus on grounding and stabilization.",
    ),
    EntropyState.CRISIS: PolicyConfig(
        mode=PolicyMode.CRISIS,
        response_constraint=ResponseConstraint.SAFETY_FOCUSED,
        max_response_length=300,
        allow_deep_reflection=False,
        allow_memory_retrieval=False,
        allow_relationship_analysis=False,
        require_grounding_prompts=True,
        crisis_resources_visible=True,
        confidence_threshold=0.3,
        description="Crisis entropy state - immediate support needed. "
        "Safety-focused responses with crisis resources.",
    ),
}


class StateRouter:
    """
    State Router for policy selection based on entropy state.

    The router analyzes the current entropy metrics and stability assessment
    to select the appropriate policy configuration. It ensures smooth
    transitions between policies and maintains consistency in system behavior.

    DISCLAIMER: This is not a clinical or treatment document. It is a
    theoretical and support framework only.
    """

    def __init__(
        self,
        policies: dict[EntropyState, PolicyConfig] | None = None,
        transition_smoothing: float = 0.3,
        confidence_weight: float = 0.5,
    ) -> None:
        """
        Initialize the state router.

        Args:
            policies: Custom policy configurations per state.
            transition_smoothing: Smoothing factor for state transitions (0-1).
            confidence_weight: Weight given to confidence in routing decisions.
        """
        self.policies = policies or DEFAULT_POLICIES.copy()
        self.transition_smoothing = transition_smoothing
        self.confidence_weight = confidence_weight

        # State tracking
        self._current_policy: PolicyConfig | None = None
        self._previous_state: EntropyState | None = None
        self._transition_count: int = 0
        self._routing_history: list[RoutingDecision] = []

        # Custom handlers
        self._state_handlers: dict[EntropyState, list[Callable]] = {
            state: [] for state in EntropyState
        }

    def route(
        self,
        entropy_metrics: EntropyMetrics,
        stability_metrics: StabilityMetrics | None = None,
        override_state: EntropyState | None = None,
    ) -> RoutingDecision:
        """
        Route to appropriate policy based on current metrics.

        Args:
            entropy_metrics: Current entropy analysis results.
            stability_metrics: Optional stability analysis results.
            override_state: Optional state override for testing.

        Returns:
            RoutingDecision with selected policy and reasoning.
        """
        # Determine target state
        target_state = override_state or entropy_metrics.state

        # Apply transition smoothing if we have a previous state
        if self._previous_state is not None and self._previous_state != target_state:
            target_state = self._apply_transition_smoothing(
                self._previous_state,
                target_state,
                entropy_metrics.confidence,
            )

        # Get policy for target state
        policy = self.policies.get(target_state, self.policies[EntropyState.STABLE])

        # Build reasoning
        reasoning = self._build_reasoning(
            entropy_metrics,
            stability_metrics,
            target_state,
            policy,
        )

        # Generate recommended actions
        actions = self._generate_recommended_actions(
            target_state,
            entropy_metrics,
            stability_metrics,
        )

        # Generate warnings
        warnings = self._generate_warnings(
            entropy_metrics,
            stability_metrics,
            target_state,
        )

        # Calculate routing confidence
        routing_confidence = self._calculate_routing_confidence(
            entropy_metrics,
            stability_metrics,
            target_state,
        )

        # Create decision
        decision = RoutingDecision(
            policy=policy,
            entropy_state=target_state,
            confidence=routing_confidence,
            reasoning=reasoning,
            recommended_actions=actions,
            warnings=warnings,
            metadata={
                "transition_from": self._previous_state.value if self._previous_state else None,
                "transition_count": self._transition_count,
                "entropy_value": entropy_metrics.normalized_entropy,
            },
        )

        # Update state tracking
        if self._previous_state != target_state:
            self._transition_count += 1
        self._previous_state = target_state
        self._current_policy = policy
        self._routing_history.append(decision)

        # Trim history
        if len(self._routing_history) > 100:
            self._routing_history = self._routing_history[-100:]

        # Execute state handlers
        self._execute_handlers(target_state, decision)

        return decision

    def _apply_transition_smoothing(
        self,
        previous: EntropyState,
        target: EntropyState,
        confidence: float,
    ) -> EntropyState:
        """
        Apply smoothing to state transitions to prevent rapid oscillation.

        Args:
            previous: Previous entropy state.
            target: Target entropy state.
            confidence: Confidence in the target state.

        Returns:
            Smoothed target state.
        """
        # Define state ordering for smoothing
        state_order = [
            EntropyState.LOW,
            EntropyState.STABLE,
            EntropyState.ELEVATED,
            EntropyState.HIGH,
            EntropyState.CRISIS,
        ]

        prev_idx = state_order.index(previous)
        target_idx = state_order.index(target)

        # If confidence is low and transition is large, stay at intermediate state
        if confidence < self.transition_smoothing and abs(target_idx - prev_idx) > 1:
            # Move one step toward target
            if target_idx > prev_idx:
                return state_order[prev_idx + 1]
            else:
                return state_order[prev_idx - 1]

        return target

    def _build_reasoning(
        self,
        entropy_metrics: EntropyMetrics,
        stability_metrics: StabilityMetrics | None,
        state: EntropyState,
        policy: PolicyConfig,
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        parts = [
            f"Entropy state detected: {state.value}.",
            f"Normalized entropy: {entropy_metrics.normalized_entropy:.3f}.",
            f"Confidence: {entropy_metrics.confidence:.3f}.",
        ]

        if stability_metrics:
            parts.append(
                f"Stability trend: {stability_metrics.stability_trend}."
            )
            if stability_metrics.is_chaotic:
                parts.append("Warning: Chaotic dynamics detected.")

        parts.append(f"Selected policy: {policy.mode.value}.")
        parts.append(policy.description)

        return " ".join(parts)

    def _generate_recommended_actions(
        self,
        state: EntropyState,
        entropy_metrics: EntropyMetrics,
        stability_metrics: StabilityMetrics | None,
    ) -> list[str]:
        """Generate recommended actions based on current state."""
        actions = []

        if state == EntropyState.CRISIS:
            actions.extend([
                "Provide immediate grounding support",
                "Display crisis resources",
                "Offer breathing exercises",
                "Remind user of safe contacts",
            ])
        elif state == EntropyState.HIGH:
            actions.extend([
                "Focus on grounding techniques",
                "Retrieve safe memories",
                "Simplify communication",
                "Check in on physical needs",
            ])
        elif state == EntropyState.ELEVATED:
            actions.extend([
                "Include grounding prompts",
                "Monitor for escalation",
                "Offer stabilization options",
            ])
        elif state == EntropyState.STABLE:
            actions.extend([
                "Continue supportive interaction",
                "Enable full feature access",
            ])
        elif state == EntropyState.LOW:
            actions.extend([
                "Encourage emotional exploration",
                "Offer reflection prompts",
                "Support emotional range expansion",
            ])

        if stability_metrics and stability_metrics.stability_trend == "degrading":
            actions.insert(0, "Monitor for continued destabilization")

        return actions

    def _generate_warnings(
        self,
        entropy_metrics: EntropyMetrics,
        stability_metrics: StabilityMetrics | None,
        state: EntropyState,
    ) -> list[str]:
        """Generate warnings based on current metrics."""
        warnings = []

        if entropy_metrics.metadata.get("above_crisis_threshold"):
            warnings.append("Entropy above crisis threshold")

        if stability_metrics:
            if stability_metrics.is_chaotic:
                warnings.append("Chaotic dynamics detected - high unpredictability")
            if stability_metrics.stability_trend == "degrading":
                warnings.append("Stability trend is degrading")

        if state == EntropyState.CRISIS:
            warnings.append("User may need immediate support")

        if self._transition_count > 5:
            recent_transitions = len([
                d for d in self._routing_history[-10:]
                if d.metadata.get("transition_from") is not None
            ])
            if recent_transitions > 3:
                warnings.append("Frequent state transitions detected")

        return warnings

    def _calculate_routing_confidence(
        self,
        entropy_metrics: EntropyMetrics,
        stability_metrics: StabilityMetrics | None,
        state: EntropyState,
    ) -> float:
        """Calculate confidence in the routing decision."""
        confidence = entropy_metrics.confidence

        # Adjust based on stability
        if stability_metrics:
            if stability_metrics.is_stable:
                confidence = min(1.0, confidence + 0.1)
            elif stability_metrics.is_chaotic:
                confidence = max(0.0, confidence - 0.2)

        # Adjust based on policy threshold
        policy = self.policies.get(state)
        if policy and confidence < policy.confidence_threshold:
            confidence *= 0.8  # Reduce confidence if below threshold

        return float(confidence)

    def _execute_handlers(
        self,
        state: EntropyState,
        decision: RoutingDecision,
    ) -> None:
        """Execute registered handlers for the state."""
        for handler in self._state_handlers.get(state, []):
            try:
                handler(decision)
            except Exception:
                pass  # Handlers should not break routing

    def register_handler(
        self,
        state: EntropyState,
        handler: Callable[[RoutingDecision], None],
    ) -> None:
        """
        Register a handler to be called when entering a state.

        Args:
            state: The entropy state to handle.
            handler: Callback function receiving the routing decision.
        """
        self._state_handlers[state].append(handler)

    def update_policy(
        self,
        state: EntropyState,
        policy: PolicyConfig,
    ) -> None:
        """
        Update the policy for a specific state.

        Args:
            state: The entropy state to update.
            policy: New policy configuration.
        """
        self.policies[state] = policy

    def get_current_policy(self) -> PolicyConfig | None:
        """Get the currently active policy."""
        return self._current_policy

    def get_routing_history(self) -> list[RoutingDecision]:
        """Get routing decision history."""
        return self._routing_history.copy()

    def reset(self) -> None:
        """Reset router state."""
        self._current_policy = None
        self._previous_state = None
        self._transition_count = 0
        self._routing_history.clear()


class ResponseGenerator:
    """
    Generate responses constrained by the current policy.

    This class ensures that all responses adhere to the constraints
    defined by the active policy configuration.
    """

    def __init__(self, router: StateRouter) -> None:
        """
        Initialize the response generator.

        Args:
            router: The state router to use for policy decisions.
        """
        self.router = router

    def constrain_response(
        self,
        response: str,
        decision: RoutingDecision,
    ) -> str:
        """
        Constrain a response based on the routing decision.

        Args:
            response: The original response text.
            decision: The routing decision with policy constraints.

        Returns:
            Constrained response text.
        """
        policy = decision.policy

        # Apply length constraint
        if len(response) > policy.max_response_length:
            response = response[: policy.max_response_length - 3] + "..."

        # Add grounding prompts if required
        if policy.require_grounding_prompts:
            response = self._add_grounding_prompt(response, decision.entropy_state)

        # Add crisis resources if visible
        if policy.crisis_resources_visible:
            response = self._add_crisis_resources(response)

        return response

    def _add_grounding_prompt(self, response: str, state: EntropyState) -> str:
        """Add appropriate grounding prompt to response."""
        prompts = {
            EntropyState.ELEVATED: "\n\nTake a moment to notice your breathing.",
            EntropyState.HIGH: "\n\nLet's pause. Can you feel your feet on the ground?",
            EntropyState.CRISIS: "\n\nYou are safe right now. Let's focus on this moment.",
        }
        prompt = prompts.get(state, "")
        return response + prompt

    def _add_crisis_resources(self, response: str) -> str:
        """Add crisis resources to response."""
        resources = (
            "\n\n---\n"
            "If you're in crisis, please reach out:\n"
            "- National Suicide Prevention Lifeline: 988\n"
            "- Crisis Text Line: Text HOME to 741741\n"
            "- International Association for Suicide Prevention: "
            "https://www.iasp.info/resources/Crisis_Centres/"
        )
        return response + resources

    def can_perform_action(
        self,
        action: str,
        decision: RoutingDecision,
    ) -> bool:
        """
        Check if an action is allowed under the current policy.

        Args:
            action: The action to check.
            decision: The current routing decision.

        Returns:
            True if the action is allowed.
        """
        policy = decision.policy

        action_permissions = {
            "deep_reflection": policy.allow_deep_reflection,
            "memory_retrieval": policy.allow_memory_retrieval,
            "relationship_analysis": policy.allow_relationship_analysis,
        }

        return action_permissions.get(action, True)
