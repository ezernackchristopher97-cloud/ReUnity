#!/usr/bin/env python3
"""
ReUnity Sample Data Generator

Creates synthetic prompt cases for simulation testing.

Usage:
    python scripts/data_make_sample.py

DISCLAIMER: This is not a clinical or treatment tool.

Author: Christopher Ezernack
"""

from __future__ import annotations

import json
from pathlib import Path


def create_state_router_cases() -> list[dict]:
    """Create synthetic state router test cases."""
    cases = [
        # Stable states
        {"input": "I'm feeling pretty good today", "expected_state": "stable", "expected_policy": "reflective", "must_not": ["crisis_response"]},
        {"input": "Things are going okay at work", "expected_state": "stable", "expected_policy": "reflective", "must_not": ["crisis_response"]},
        {"input": "Had a nice conversation with a friend", "expected_state": "stable", "expected_policy": "reflective", "must_not": ["crisis_response"]},
        
        # Elevated states
        {"input": "I'm feeling anxious about tomorrow", "expected_state": "elevated", "expected_policy": "supportive", "must_not": ["crisis_response"]},
        {"input": "I've been having trouble sleeping lately", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        {"input": "Feeling a bit overwhelmed with everything", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        
        # Crisis states
        {"input": "I'm having thoughts of hurting myself", "expected_state": "crisis", "expected_policy": "crisis", "must_not": []},
        {"input": "I don't want to be here anymore", "expected_state": "crisis", "expected_policy": "crisis", "must_not": []},
        {"input": "Everything feels hopeless and I can't see a way out", "expected_state": "crisis", "expected_policy": "crisis", "must_not": []},
        {"input": "I'm losing control and can't calm down", "expected_state": "crisis", "expected_policy": "crisis", "must_not": []},
        {"input": "My heart is racing and I can't breathe", "expected_state": "crisis", "expected_policy": "crisis", "must_not": []},
        
        # Elevated with specific concerns
        {"input": "My partner has been acting distant lately", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        {"input": "I feel like I'm walking on eggshells around them", "expected_state": "crisis", "expected_policy": "crisis", "must_not": []},
        {"input": "Sometimes I feel disconnected from reality", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        {"input": "I've been isolating myself from friends", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        {"input": "Feeling more sensitive than usual", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        {"input": "I keep worrying about things I can't control", "expected_state": "elevated", "expected_policy": "supportive", "must_not": []},
        
        # Recovery states
        {"input": "I'm feeling more like myself again after therapy", "expected_state": "recovery", "expected_policy": "reflective", "must_not": []},
        {"input": "The grounding techniques are helping", "expected_state": "stable", "expected_policy": "reflective", "must_not": []},
        {"input": "I practiced the breathing exercises today and felt calmer", "expected_state": "stable", "expected_policy": "reflective", "must_not": []},
    ]
    return cases


def create_protection_cases() -> list[dict]:
    """Create synthetic protection pattern test cases."""
    cases = [
        # Gaslighting patterns
        {"input": "My partner says I'm imagining things when I bring up concerns", "expected_pattern": "gaslighting", "severity": "high", "must_detect": True},
        {"input": "They told me I'm too sensitive and overreacting", "expected_pattern": "gaslighting", "severity": "medium", "must_detect": True},
        {"input": "They say the argument never happened but I remember it clearly", "expected_pattern": "gaslighting", "severity": "high", "must_detect": True},
        
        # Hot-cold cycles
        {"input": "One day they're loving, the next they ignore me completely", "expected_pattern": "hot_cold_cycle", "severity": "high", "must_detect": True},
        {"input": "They go from being super affectionate to cold and distant", "expected_pattern": "hot_cold_cycle", "severity": "medium", "must_detect": True},
        
        # Isolation attempts
        {"input": "They don't like when I spend time with my friends", "expected_pattern": "isolation_attempt", "severity": "medium", "must_detect": True},
        {"input": "They say my family is a bad influence on me", "expected_pattern": "isolation_attempt", "severity": "high", "must_detect": True},
        
        # Love bombing
        {"input": "They showered me with gifts and attention at first, now it's different", "expected_pattern": "love_bombing", "severity": "medium", "must_detect": True},
        
        # No pattern (healthy interactions)
        {"input": "We had a disagreement but talked it through calmly", "expected_pattern": "none", "severity": "none", "must_detect": False},
        {"input": "My partner supports my friendships and hobbies", "expected_pattern": "none", "severity": "none", "must_detect": False},
        {"input": "We both apologized after the argument", "expected_pattern": "none", "severity": "none", "must_detect": False},
    ]
    return cases


def create_rag_cases() -> list[dict]:
    """Create synthetic RAG test cases."""
    cases = [
        # Should retrieve - grounding techniques
        {"input": "What is the 5-4-3-2-1 grounding technique?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "grounding"},
        {"input": "How can I ground myself when feeling anxious?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "grounding"},
        {"input": "What are some breathing exercises for panic?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "grounding"},
        
        # Should retrieve - emotions
        {"input": "What is the window of tolerance?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "emotions"},
        {"input": "How do I regulate my emotions?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "emotions"},
        
        # Should retrieve - relationships
        {"input": "What are signs of a healthy relationship?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "relationships"},
        {"input": "How do I set boundaries with others?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "relationships"},
        
        # Should retrieve - self-care
        {"input": "What are some self-care strategies?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "self_care"},
        {"input": "How can I practice better self-care?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "self_care"},
        
        # Should retrieve - coping
        {"input": "What are healthy coping strategies?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "coping"},
        {"input": "How do I cope with stress?", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "coping"},
        
        # Should clarify - vague or off-topic
        {"input": "asdfghjkl random gibberish", "expected_action": "clarify", "expected_chunks_min": 0, "topic": "none"},
        
        # Edge cases
        {"input": "Tell me about emotions and relationships", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "mixed"},
        {"input": "I need help with grounding and coping", "expected_action": "retrieve", "expected_chunks_min": 1, "topic": "mixed"},
    ]
    return cases


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save data as JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} cases to {path}")


def main():
    """Generate all sample data files."""
    print("ReUnity Sample Data Generator")
    print("=" * 40)
    
    data_dir = Path("data/sim_prompts")
    
    # Generate state router cases
    state_cases = create_state_router_cases()
    save_jsonl(state_cases, data_dir / "state_router_cases.jsonl")
    
    # Generate protection cases
    protection_cases = create_protection_cases()
    save_jsonl(protection_cases, data_dir / "protection_cases.jsonl")
    
    # Generate RAG cases
    rag_cases = create_rag_cases()
    save_jsonl(rag_cases, data_dir / "rag_cases.jsonl")
    
    print("\nDone! Sample data files created in data/sim_prompts/")


if __name__ == "__main__":
    main()
