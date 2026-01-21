#!/usr/bin/env python3
"""
ReUnity Basic Usage Example

This example demonstrates the core functionality of the ReUnity system,
including entropy analysis, memory management, pattern recognition,
and reflection generation.

DISCLAIMER: ReUnity is NOT a clinical or treatment tool. It is a theoretical
and support framework only. If you are in crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
"""

import numpy as np
from datetime import datetime

# Import ReUnity components
from reunity.core.entropy import EntropyAnalyzer, EntropyState
from reunity.router.state_router import StateRouter
from reunity.protective.pattern_recognizer import ProtectivePatternRecognizer
from reunity.memory.continuity_store import (
    RecursiveIdentityMemoryEngine,
    ConsentScope,
    MemoryType,
)
from reunity.reflection.mirror_link import (
    MirrorLinkDialogueCompanion,
    CommunicationStyle,
)
from reunity.regime.regime_controller import RegimeController


def text_to_distribution(text: str) -> np.ndarray:
    """Convert text to a probability distribution for entropy analysis."""
    words = text.lower().split()
    if not words:
        return np.array([1.0])

    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    total = sum(word_counts.values())
    return np.array([c / total for c in word_counts.values()])


def main():
    print("=" * 60)
    print("ReUnity Basic Usage Example")
    print("=" * 60)
    print()
    print("DISCLAIMER: This is not a clinical or treatment tool.")
    print()

    # Initialize components
    entropy_analyzer = EntropyAnalyzer()
    state_router = StateRouter()
    pattern_recognizer = ProtectivePatternRecognizer()
    memory_engine = RecursiveIdentityMemoryEngine()
    dialogue_companion = MirrorLinkDialogueCompanion()
    regime_controller = RegimeController()

    # Example 1: Entropy Analysis
    print("-" * 40)
    print("1. Entropy Analysis")
    print("-" * 40)

    sample_text = "I feel calm and centered today. The morning was peaceful."
    distribution = text_to_distribution(sample_text)
    metrics = entropy_analyzer.analyze(distribution)

    print(f"Text: '{sample_text}'")
    print(f"Entropy State: {metrics.state.value}")
    print(f"Normalized Entropy: {metrics.normalized_entropy:.3f}")
    print(f"Confidence: {metrics.confidence:.3f}")
    print(f"Is Stable: {metrics.is_stable}")
    print()

    # Example 2: State Routing
    print("-" * 40)
    print("2. State-Based Policy Routing")
    print("-" * 40)

    policy = state_router.route(metrics)
    print(f"Policy Type: {policy.policy_type.value}")
    print("Recommendations:")
    for rec in policy.recommendations[:3]:
        print(f"  - {rec}")
    print()

    # Example 3: Memory Management
    print("-" * 40)
    print("3. Memory Management with Consent")
    print("-" * 40)

    # Add an anchor memory
    anchor_memory = memory_engine.add_memory(
        identity="host",
        content="My grandmother's garden was always a safe place. "
                "The smell of roses and the sound of wind chimes.",
        memory_type=MemoryType.ANCHOR,
        tags=["safe_place", "grandmother", "garden"],
        consent_scope=ConsentScope.PRIVATE,
        emotional_valence=0.8,
        importance=0.9,
    )
    print(f"Added anchor memory: {anchor_memory.id[:8]}...")

    # Add a journal entry
    journal = memory_engine.add_journal_entry(
        title="Morning Reflection",
        content="Woke up feeling rested. Practiced breathing exercises.",
        identity="host",
        mood="calm",
        energy_level=0.7,
        entropy_level=metrics.normalized_entropy,
    )
    print(f"Added journal entry: {journal.id[:8]}...")

    # Retrieve grounding memories
    result = memory_engine.retrieve_grounding(
        current_identity="host",
        query="safe place",
        crisis_level=0.3,
        max_results=3,
    )
    print(f"Retrieved {len(result.memories)} grounding memories")
    print()

    # Example 4: Pattern Recognition
    print("-" * 40)
    print("4. Protective Pattern Recognition")
    print("-" * 40)

    # Analyze a series of interactions
    interactions = [
        {"text": "You're so amazing, I've never met anyone like you", "timestamp": 1000},
        {"text": "We're meant to be together forever", "timestamp": 2000},
        {"text": "Why were you talking to them? You don't need other friends", "timestamp": 3000},
        {"text": "I'm the only one who really understands you", "timestamp": 4000},
    ]

    analysis = pattern_recognizer.analyze_interactions(interactions)
    print(f"Overall Risk: {analysis.overall_risk:.2f}")
    print(f"Stability Assessment: {analysis.stability_assessment}")
    if analysis.patterns_detected:
        print("Detected Patterns:")
        for pattern in analysis.patterns_detected[:3]:
            print(f"  - {pattern.pattern_type.value}: {pattern.message}")
    print()

    # Example 5: MirrorLink Reflection
    print("-" * 40)
    print("5. MirrorLink Reflection")
    print("-" * 40)

    reflection = dialogue_companion.reflect(
        current_emotion="confused and hurt",
        past_context="Last week I felt so loved and valued by them",
        entropy_state=EntropyState.ELEVATED,
        style=CommunicationStyle.GENTLE,
    )

    print(f"Reflection Type: {reflection.reflection_type.value}")
    print(f"Content: {reflection.content}")
    if reflection.follow_up_question:
        print(f"Follow-up: {reflection.follow_up_question}")
    if reflection.grounding_prompt:
        print(f"Grounding: {reflection.grounding_prompt}")
    print()

    # Example 6: Regime Control
    print("-" * 40)
    print("6. Regime Control")
    print("-" * 40)

    regime_state = regime_controller.update(metrics)
    print(f"Current Regime: {regime_state.regime.value}")
    print(f"Entropy Band: {regime_state.entropy_band.value}")
    print(f"Apostasis Active: {regime_state.apostasis_active}")
    print(f"Regeneration Active: {regime_state.regeneration_active}")
    print()

    # Example 7: Export Data
    print("-" * 40)
    print("7. Data Export with Provenance")
    print("-" * 40)

    export_data = memory_engine.export_memories(identity="host")
    print(f"Exported {export_data['statistics']['total_memories']} memories")
    print(f"Exported {export_data['statistics']['total_journals']} journal entries")
    print()

    print("=" * 60)
    print("Example Complete")
    print("=" * 60)
    print()
    print("Remember: ReUnity is a support tool, not a replacement for")
    print("professional mental health care. If you need help, please")
    print("reach out to a qualified mental health professional.")


if __name__ == "__main__":
    main()
