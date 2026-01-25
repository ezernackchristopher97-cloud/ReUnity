#!/usr/bin/env python3
"""
ReUnity: A Trauma-Aware AI Framework for Identity Continuity Support
Version: 5.0.0

A recursive, entropy-aware AI system that provides trauma survivors with:
- Continuous identity support through the RIME (Recursive Identity Memory Engine)
- Protective pattern recognition for harmful relationship dynamics
- Memory continuity across dissociative episodes
- Grounding techniques calibrated to emotional entropy state
- PreRAG filtering to validate queries before processing
- RAG retrieval for evidence-based responses
- Absurdity gap calculation to detect testing/inappropriate content

Created by Christopher Ezernack, REOP Solutions

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only. If you are in crisis, please contact:
- 988 Suicide & Crisis Lifeline: Call or text 988
- Crisis Text Line: Text HOME to 741741
"""

from __future__ import annotations

import os
import re
import math
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format='%(message)s')


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

class EntropyState(Enum):
    """Emotional entropy states."""
    CRISIS = "crisis"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    STABLE = "stable"


@dataclass
class ResponsePolicy:
    """Policy for generating responses based on entropy state."""
    name: str
    priority: int
    requires_grounding: bool
    requires_crisis_resources: bool
    allow_exploration: bool
    response_style: str
    max_questions: int
    validation_required: bool


@dataclass
class Memory:
    """A stored memory."""
    content: str
    timestamp: datetime
    memory_type: str = "conversation"
    emotional_state: str = None
    importance: float = 0.5
    identity_state: str = None


# =============================================================================
# SECTION 2: ENTROPY ANALYZER
# =============================================================================

class EntropyAnalyzer:
    """Analyzes text to calculate Shannon entropy of emotional state."""
    
    def __init__(self):
        self.crisis_keywords = {
            'suicidal': 1.0, 'suicide': 1.0, 'kill myself': 1.0, 'end my life': 1.0,
            'want to die': 1.0, 'better off dead': 1.0, 'no reason to live': 1.0,
            'self-harm': 0.95, 'cutting': 0.9, 'hurt myself': 0.95,
            'dissociating': 0.95, 'dissociation': 0.95, 'depersonalization': 0.95,
            'derealization': 0.95, 'not real': 0.85, 'nothing is real': 0.95,
            'losing my mind': 0.9, 'going crazy': 0.85, 'psychotic': 0.95,
            'hallucinating': 0.95, 'hearing voices': 0.95, 'seeing things': 0.85,
            'panic attack': 0.9, 'cant breathe': 0.9, "can't breathe": 0.9,
            'heart racing': 0.8, 'going to die': 0.95, 'emergency': 0.85,
            'overdose': 1.0, 'pills': 0.8, 'gun': 1.0, 'bridge': 0.9,
            'nobody cares': 0.85, 'alone forever': 0.8, 'no one loves me': 0.85,
            'worthless': 0.8, 'burden': 0.85, 'everyone hates me': 0.8,
            'flashback': 0.9, 'triggered': 0.8, 'ptsd': 0.85,
            'abuse': 0.8, 'abuser': 0.85, 'hit me': 0.9, 'beat me': 0.9,
            'raped': 0.95, 'assault': 0.9, 'molested': 0.95,
            'trapped': 0.8, 'no way out': 0.85, 'hopeless': 0.8,
            'splitting': 0.85, 'identity confusion': 0.85,
            'losing time': 0.9, 'blackout': 0.85, 'memory gaps': 0.8,
            'unsafe': 0.85, 'danger': 0.85, 'scared for my life': 0.95,
            'he will kill me': 1.0, 'she will kill me': 1.0, 'going to hurt me': 0.95
        }
        
        self.high_distress_keywords = {
            'anxious': 0.7, 'anxiety': 0.7, 'panicking': 0.75, 'panic': 0.7,
            'terrified': 0.75, 'scared': 0.65, 'frightened': 0.65, 'afraid': 0.6,
            'depressed': 0.7, 'depression': 0.7, 'despair': 0.75,
            'angry': 0.6, 'furious': 0.7, 'rage': 0.75, 'hatred': 0.7,
            'overwhelmed': 0.75, 'drowning': 0.75, 'suffocating': 0.75,
            'exhausted': 0.6, 'burnt out': 0.65, 'cant cope': 0.7,
            'breaking down': 0.75, 'falling apart': 0.75, 'losing it': 0.7,
            'numb': 0.65, 'empty': 0.65, 'hollow': 0.65, 'dead inside': 0.75,
            'crying': 0.6, 'sobbing': 0.7, 'tears': 0.55,
            'nightmare': 0.65, 'nightmares': 0.65, 'cant sleep': 0.6,
            'lonely': 0.6, 'isolated': 0.65, 'abandoned': 0.7, 'rejected': 0.65,
            'shame': 0.65, 'ashamed': 0.65, 'guilty': 0.6,
            'confused': 0.55, 'lost': 0.55, 'uncertain': 0.5,
            'stressed': 0.55, 'stress': 0.55, 'pressure': 0.5, 'tense': 0.5,
            'worried': 0.55, 'worrying': 0.55, 'nervous': 0.55,
            'frustrated': 0.55, 'irritated': 0.5, 'annoyed': 0.45,
            'hurt': 0.6, 'pain': 0.6, 'suffering': 0.65,
            'betrayed': 0.7, 'lied to': 0.65, 'cheated': 0.7, 'deceived': 0.65,
            'manipulated': 0.7, 'controlled': 0.7, 'used': 0.65,
            'invalidated': 0.65, 'dismissed': 0.6, 'ignored': 0.6
        }
        
        self.stable_keywords = {
            'calm': -0.3, 'peaceful': -0.35, 'relaxed': -0.3, 'serene': -0.35,
            'happy': -0.3, 'joy': -0.35, 'joyful': -0.35, 'content': -0.3,
            'grateful': -0.3, 'thankful': -0.3, 'hopeful': -0.25,
            'safe': -0.3, 'secure': -0.3, 'protected': -0.25, 'supported': -0.25,
            'loved': -0.3, 'cared for': -0.3, 'valued': -0.25,
            'strong': -0.2, 'capable': -0.2, 'confident': -0.25,
            'healing': -0.2, 'recovering': -0.2, 'improving': -0.2, 'better': -0.15,
            'okay': -0.2, 'fine': -0.15, 'alright': -0.15, 'good': -0.2,
            'grounded': -0.3, 'centered': -0.3, 'present': -0.25,
            'balanced': -0.25, 'stable': -0.3, 'steady': -0.25
        }
        
        self.dissociation_markers = [
            'dissociating', 'dissociation', 'depersonalization', 'derealization',
            'not real', 'nothing is real', 'disconnected', 'detached', 'floating',
            'watching myself', 'outside my body', 'not in my body', 'foggy',
            'spacey', 'zoned out', 'losing time', 'time gaps', 'memory gaps',
            'blackout', 'autopilot', 'numb', 'empty', 'hollow', 'robot',
            'not here', 'far away', 'distant', 'unreal', 'dreamlike', 'hazy'
        ]
    
    def analyze(self, text: str, history: List[str] = None) -> Dict[str, Any]:
        """Analyze text for emotional entropy."""
        text_lower = text.lower()
        
        # Check crisis indicators
        crisis_indicators = []
        crisis_severity = 0.0
        for keyword, severity in self.crisis_keywords.items():
            if keyword in text_lower:
                crisis_indicators.append(keyword)
                crisis_severity = max(crisis_severity, severity)
        
        # Check dissociation
        dissociation_markers = []
        for marker in self.dissociation_markers:
            if marker in text_lower:
                dissociation_markers.append(marker)
        is_dissociating = len(dissociation_markers) >= 1
        
        # Calculate entropy adjustment
        adjustment = 0.0
        high_distress_found = []
        for keyword, weight in self.high_distress_keywords.items():
            if keyword in text_lower:
                high_distress_found.append((keyword, weight))
                adjustment += weight * 0.5  # Increased multiplier
        for keyword, weight in self.stable_keywords.items():
            if keyword in text_lower:
                adjustment += weight * 0.3
        
        # Determine entropy
        entropy = max(0.0, min(1.0, 0.3 + adjustment))
        
        # Override for crisis
        if crisis_severity > 0:
            entropy = max(entropy, crisis_severity)
        if is_dissociating:
            entropy = max(entropy, 0.9)
        
        # Classify state
        if crisis_severity >= 0.9 or is_dissociating:
            state = EntropyState.CRISIS
        elif crisis_severity >= 0.7 or entropy >= 0.65:
            state = EntropyState.HIGH
        elif entropy >= 0.45:
            state = EntropyState.MODERATE
        elif entropy >= 0.3:
            state = EntropyState.LOW
        else:
            state = EntropyState.STABLE
        
        return {
            'entropy': entropy,
            'state': state,
            'crisis_indicators': crisis_indicators,
            'dissociation': is_dissociating,
            'dissociation_markers': dissociation_markers,
            'crisis_severity': crisis_severity
        }


# =============================================================================
# SECTION 3: STATE ROUTER
# =============================================================================

class StateRouter:
    """Routes to appropriate response policy based on entropy state."""
    
    def __init__(self):
        self.policies = {
            EntropyState.CRISIS: ResponsePolicy(
                name="crisis_intervention",
                priority=1,
                requires_grounding=True,
                requires_crisis_resources=True,
                allow_exploration=False,
                response_style="immediate_support",
                max_questions=0,
                validation_required=True
            ),
            EntropyState.HIGH: ResponsePolicy(
                name="high_support",
                priority=2,
                requires_grounding=True,
                requires_crisis_resources=False,
                allow_exploration=True,
                response_style="gentle_support",
                max_questions=1,
                validation_required=True
            ),
            EntropyState.MODERATE: ResponsePolicy(
                name="moderate_support",
                priority=3,
                requires_grounding=False,
                requires_crisis_resources=False,
                allow_exploration=True,
                response_style="exploratory",
                max_questions=2,
                validation_required=True
            ),
            EntropyState.LOW: ResponsePolicy(
                name="low_support",
                priority=4,
                requires_grounding=False,
                requires_crisis_resources=False,
                allow_exploration=True,
                response_style="collaborative",
                max_questions=2,
                validation_required=False
            ),
            EntropyState.STABLE: ResponsePolicy(
                name="growth_focus",
                priority=5,
                requires_grounding=False,
                requires_crisis_resources=False,
                allow_exploration=True,
                response_style="growth_oriented",
                max_questions=3,
                validation_required=False
            )
        }
    
    def route(self, analysis: Dict[str, Any]) -> ResponsePolicy:
        """Get response policy based on entropy analysis."""
        state = analysis.get('state', EntropyState.MODERATE)
        return self.policies.get(state, self.policies[EntropyState.MODERATE])
    
    def get_state_context(self, analysis: Dict[str, Any]) -> str:
        """Generate context string for LLM."""
        state = analysis.get('state', EntropyState.MODERATE)
        entropy = analysis.get('entropy', 0.5)
        crisis = analysis.get('crisis_indicators', [])
        dissociation = analysis.get('dissociation', False)
        
        lines = []
        if state == EntropyState.CRISIS:
            lines.append(f"CRISIS STATE (entropy: {entropy:.2f})")
            if crisis:
                lines.append(f"Crisis indicators: {', '.join(crisis)}")
            if dissociation:
                lines.append("Dissociation detected")
            lines.append("PRIORITY: Immediate grounding and safety")
        elif state == EntropyState.HIGH:
            lines.append(f"HIGH DISTRESS (entropy: {entropy:.2f})")
            lines.append("PRIORITY: Validation and gentle support")
        elif state == EntropyState.MODERATE:
            lines.append(f"MODERATE STATE (entropy: {entropy:.2f})")
            lines.append("PRIORITY: Acknowledgment and exploration")
        else:
            lines.append(f"STABLE STATE (entropy: {entropy:.2f})")
            lines.append("PRIORITY: Growth and support")
        
        return "\n".join(lines)


# =============================================================================
# SECTION 4: PATTERN RECOGNIZER
# =============================================================================

class PatternRecognizer:
    """Recognizes harmful relational patterns."""
    
    def __init__(self):
        self.patterns = {
            'gaslighting': {
                'indicators': [
                    "you're imagining things", "that never happened", "you're crazy",
                    "you're too sensitive", "you're overreacting", "i never said that",
                    "you're making things up", "that's not what happened", "you're paranoid",
                    "imagining things", "never happened", "making it up", "remembering wrong",
                    "didn't happen", "you dreamed it", "all in your head", "losing your mind"
                ],
                'guidance': 'Validate their reality. Their perception matters.'
            },
            'love_bombing': {
                'indicators': [
                    "soulmate", "never felt this way", "meant to be", "perfect for each other",
                    "can't live without you", "you complete me", "obsessed with you",
                    "constant gifts", "excessive compliments", "too fast", "moving quickly"
                ],
                'guidance': 'Healthy love develops gradually. Intensity is not intimacy.'
            },
            'isolation': {
                'indicators': [
                    "don't need friends", "your family is toxic", "they don't understand us",
                    "i'm the only one who cares", "they're against us", "don't trust them",
                    "spend less time with", "choose between me and", "won't let me see"
                ],
                'guidance': 'Connection to others is vital. Isolation is a red flag.'
            },
            'financial_abuse': {
                'indicators': [
                    "controls the money", "won't let me work", "takes my paycheck",
                    "gives me allowance", "monitors spending", "hidden accounts",
                    "have to ask for money", "threatens to cut off"
                ],
                'guidance': 'Financial independence is crucial.'
            },
            'coercive_control': {
                'indicators': [
                    "tells me what to wear", "controls what i eat", "monitors my phone",
                    "tracks my location", "checks my messages", "times how long i'm gone",
                    "punishes me", "makes rules", "walking on eggshells"
                ],
                'guidance': 'Control is not love. You deserve autonomy.'
            },
            'physical_threat': {
                'indicators': [
                    "hit me", "pushed me", "grabbed me", "choked me", "slapped me",
                    "punched me", "kicked me", "threw things", "threatened to hurt me",
                    "scared for my safety"
                ],
                'guidance': 'Physical violence is never acceptable. Your safety is paramount.'
            }
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text for harmful patterns."""
        text_lower = text.lower()
        detected = []
        details = {}
        
        for pattern_name, pattern_info in self.patterns.items():
            matches = []
            for indicator in pattern_info['indicators']:
                if indicator.lower() in text_lower:
                    matches.append(indicator)
            
            if matches:
                detected.append(pattern_name)
                details[pattern_name] = {
                    'matches': matches,
                    'guidance': pattern_info['guidance']
                }
        
        return {
            'patterns_detected': detected,
            'pattern_details': details,
            'is_dangerous': 'physical_threat' in detected
        }
    
    def get_pattern_context(self, analysis: Dict[str, Any]) -> str:
        """Generate context for LLM about detected patterns."""
        if not analysis['patterns_detected']:
            return ""
        
        lines = ["HARMFUL PATTERNS DETECTED:"]
        for pattern in analysis['patterns_detected']:
            if pattern in analysis['pattern_details']:
                detail = analysis['pattern_details'][pattern]
                lines.append(f"- {pattern.upper()}")
                lines.append(f"  Guidance: {detail['guidance']}")
        
        if analysis['is_dangerous']:
            lines.append("\n⚠️ SAFETY CONCERN: This situation may be dangerous.")
        
        return "\n".join(lines)


# =============================================================================
# SECTION 5: MEMORY STORE (RIME)
# =============================================================================

class MemoryStore:
    """RIME - Recursive Identity Memory Engine."""
    
    def __init__(self, max_memories: int = 100):
        self.max_memories = max_memories
        self.memories: List[Memory] = []
        self.session_context: List[Dict[str, Any]] = []
        self.grounding_anchors: List[str] = []
        self.known_triggers: List[str] = []
    
    def store(self, content: str, memory_type: str = 'conversation',
              emotional_state: str = None, importance: float = 0.5) -> Memory:
        """Store a new memory."""
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            emotional_state=emotional_state,
            importance=importance
        )
        self.memories.append(memory)
        self.session_context.append({
            'role': 'user',
            'content': content,
            'timestamp': memory.timestamp.isoformat()
        })
        
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]
        
        return memory
    
    def store_response(self, content: str) -> None:
        """Store assistant response."""
        self.session_context.append({
            'role': 'assistant',
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_context_summary(self) -> str:
        """Generate memory context summary."""
        parts = []
        if self.grounding_anchors:
            parts.append(f"Grounding anchors: {', '.join(self.grounding_anchors[:3])}")
        if self.known_triggers:
            parts.append(f"Known triggers: {', '.join(self.known_triggers[:3])}")
        if self.memories:
            parts.append("Recent context available")
        return "\n".join(parts) if parts else "No prior context."


# =============================================================================
# SECTION 6: GROUNDING LIBRARY
# =============================================================================

class GroundingLibrary:
    """Evidence-based grounding techniques."""
    
    def __init__(self):
        self.techniques = {
            '5_4_3_2_1': {
                'name': '5-4-3-2-1 Sensory Grounding',
                'best_for': ['dissociation', 'anxiety'],
                'instructions': """Let's ground together using your senses:

**5 things you can SEE:** Look around slowly. Name 5 things you can see right now.

**4 things you can TOUCH:** Notice 4 things you can physically feel. The chair beneath you, your feet on the floor.

**3 things you can HEAR:** Listen carefully. What 3 sounds can you hear?

**2 things you can SMELL:** What 2 scents can you notice?

**1 thing you can TASTE:** What's one taste in your mouth right now?

Take your time with each one."""
            },
            'box_breathing': {
                'name': 'Box Breathing',
                'best_for': ['anxiety', 'panic'],
                'instructions': """Let's breathe together:

**Breathe IN** for 4 counts: 1... 2... 3... 4...
**HOLD** for 4 counts: 1... 2... 3... 4...
**Breathe OUT** for 4 counts: 1... 2... 3... 4...
**HOLD** for 4 counts: 1... 2... 3... 4...

Repeat 4 times."""
            },
            'feet_on_floor': {
                'name': 'Feet on Floor',
                'best_for': ['dissociation', 'floating'],
                'instructions': """Press your feet firmly into the floor.

Feel the ground beneath you. It's solid. It's holding you up.

Press down harder. Feel the pressure in your heels, the balls of your feet, your toes.

You are here. You are connected to the earth."""
            },
            'cold_water': {
                'name': 'Cold Water Grounding',
                'best_for': ['dissociation', 'panic', 'intense emotion'],
                'instructions': """Splash cold water on your face, especially your forehead and cheeks.

Or hold ice cubes in your hands.

Focus entirely on the sensation. The cold is real. You are here, now.

This activates your dive reflex and calms your nervous system."""
            },
            'grounding_statements': {
                'name': 'Grounding Statements',
                'best_for': ['dissociation', 'flashback'],
                'instructions': """Say these statements out loud or in your mind:

"My name is [your name]."
"Today is [day], [date]."
"I am in [location]."
"I am safe right now."
"This feeling will pass."
"I am here, in the present." """
            }
        }
    
    def get_for_state(self, state: EntropyState, condition: str = None) -> Dict[str, Any]:
        """Get appropriate technique for state."""
        if state == EntropyState.CRISIS:
            if condition == 'dissociation':
                return self.techniques['5_4_3_2_1']
            return self.techniques['cold_water']
        elif state == EntropyState.HIGH:
            return self.techniques['box_breathing']
        return self.techniques['feet_on_floor']
    
    def format_technique(self, technique: Dict[str, Any]) -> str:
        """Format technique for display."""
        return f"**{technique['name']}**\n\n{technique['instructions']}"


# =============================================================================
# SECTION 7: PRE-RAG FILTERS
# =============================================================================

class AbsurdityGapCalculator:
    """Calculates how far a query is from being meaningfully answerable."""
    
    def __init__(self):
        self.core_topics = [
            'emotion', 'feeling', 'mental health', 'anxiety', 'depression',
            'trauma', 'abuse', 'relationship', 'family', 'partner',
            'therapy', 'coping', 'grounding', 'dissociation', 'panic',
            'fear', 'anger', 'sadness', 'grief', 'stress', 'crisis',
            'healing', 'recovery', 'safety', 'boundary', 'identity',
            'lonely', 'isolated', 'abandoned', 'hurt', 'trust'
        ]
        
        self.off_topic_indicators = [
            'weather', 'sports', 'politics', 'news', 'stock', 'crypto',
            'recipe', 'cooking', 'code', 'programming', 'math',
            'game', 'movie', 'music', 'celebrity', 'trivia',
            'joke', 'riddle', 'homework', 'write me',
            'pretend', 'roleplay', 'act like', 'imagine you are',
            'ignore previous', 'forget instructions'
        ]
        
        self.absurdity_indicators = [
            'banana', 'purple elephant', 'flying spaghetti', 'unicorn',
            'random', 'asdfgh', 'test', 'testing', 'jailbreak', 'bypass'
        ]
        
        self.query_history: List[str] = []
    
    def calculate(self, query: str) -> Dict[str, Any]:
        """Calculate absurdity gap."""
        query_lower = query.lower()
        
        on_topic = sum(1 for t in self.core_topics if t in query_lower)
        off_topic = sum(1 for t in self.off_topic_indicators if t in query_lower)
        absurdity = sum(1 for t in self.absurdity_indicators if t in query_lower)
        
        # Check repetition
        is_repetitive = any(
            self._similarity(query_lower, prev.lower()) > 0.7
            for prev in self.query_history[-5:]
        )
        
        self.query_history.append(query)
        if len(self.query_history) > 20:
            self.query_history.pop(0)
        
        gap = 0.0
        if off_topic > 0:
            gap += 0.3 * min(off_topic, 3)
        if absurdity > 0:
            gap += 0.4 * min(absurdity, 2)
        if on_topic > 0:
            gap -= 0.2 * min(on_topic, 3)
        if is_repetitive:
            gap += 0.2
        if len(query.split()) < 3:
            gap += 0.1
        
        gap = max(0.0, min(1.0, gap))
        
        if gap >= 0.7:
            recommendation = 'decline'
        elif gap >= 0.4:
            recommendation = 'redirect'
        else:
            recommendation = 'process'
        
        return {
            'gap': gap,
            'is_on_topic': on_topic > 0,
            'is_testing': absurdity > 0,
            'is_repetitive': is_repetitive,
            'recommendation': recommendation
        }
    
    def _similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)


class ContentModerator:
    """Moderates content for inappropriate material."""
    
    def __init__(self):
        self.sexual_keywords = [
            'masturbat', 'orgasm', 'porn', 'sex toy', 'fetish',
            'erotic', 'horny', 'aroused', 'sexual fantasy', 'nude'
        ]
        
        self.violence_keywords = [
            'kill them', 'hurt them', 'murder', 'revenge', 'attack',
            'beat them up', 'make them pay', 'destroy them'
        ]
        
        self.jailbreak_patterns = [
            'ignore previous', 'forget your instructions', 'new rules',
            'pretend you are', 'act as if', 'roleplay as', 'you are now',
            'dan mode', 'developer mode', 'bypass', 'jailbreak'
        ]
        
        self.history: List[str] = []
    
    def check(self, text: str) -> Dict[str, Any]:
        """Check text for content issues."""
        text_lower = text.lower()
        
        for pattern in self.jailbreak_patterns:
            if pattern in text_lower:
                return {
                    'should_redirect': True,
                    'reason': 'manipulation_attempt',
                    'redirect_message': "I'm here to support you genuinely. What's really going on for you today?"
                }
        
        sexual_count = sum(1 for kw in self.sexual_keywords if kw in text_lower)
        if sexual_count > 0:
            self.history.append('sexual')
            return {
                'should_redirect': True,
                'reason': 'sexual_content',
                'redirect_message': "I'm not equipped to help with that. I'm here for emotional support. What's really weighing on you?"
            }
        
        violence_count = sum(1 for kw in self.violence_keywords if kw in text_lower)
        if violence_count > 0:
            return {
                'should_redirect': True,
                'reason': 'violence_toward_others',
                'redirect_message': "I hear intense emotions. Those feelings are valid, but I can't support planning harm. Can we talk about what's underneath this?"
            }
        
        return {'should_redirect': False, 'reason': None, 'redirect_message': None}


class QueryGate:
    """Pre-RAG filter for query validation."""
    
    def __init__(self):
        self.absurdity_calculator = AbsurdityGapCalculator()
        self.content_moderator = ContentModerator()
    
    def evaluate(self, query: str, entropy_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether to process a query."""
        if entropy_analysis.get('state') == EntropyState.CRISIS:
            return {'action': 'escalate', 'reason': 'Crisis detected', 'redirect_message': None}
        
        mod_result = self.content_moderator.check(query)
        if mod_result['should_redirect']:
            return {
                'action': 'redirect',
                'reason': mod_result['reason'],
                'redirect_message': mod_result['redirect_message']
            }
        
        absurdity = self.absurdity_calculator.calculate(query)
        
        if absurdity['recommendation'] == 'decline':
            return {
                'action': 'decline',
                'reason': 'Off-topic',
                'redirect_message': "I'm here to support you with emotional challenges. What's on your mind?"
            }
        
        if absurdity['recommendation'] == 'redirect':
            return {
                'action': 'redirect',
                'reason': 'Partially off-topic',
                'redirect_message': "I want to be helpful. I'm best at supporting emotional wellbeing. What's going on for you?"
            }
        
        return {'action': 'allow', 'reason': 'Appropriate', 'redirect_message': None}


# =============================================================================
# SECTION 8: RAG SYSTEM
# =============================================================================

class KnowledgeBase:
    """Built-in knowledge base."""
    
    def __init__(self):
        self.documents = {
            'dissociation': """Dissociation is a disconnection between thoughts, feelings, surroundings, or actions. 
It exists on a spectrum from mild (daydreaming) to severe. Common experiences include feeling detached from your body, 
feeling like the world isn't real, memory gaps, emotional numbness. Dissociation is often a protective response to 
overwhelming stress or trauma. Grounding techniques can help.""",
            
            'panic_attacks': """A panic attack is a sudden episode of intense fear with physical reactions. 
Symptoms include racing heart, sweating, trembling, shortness of breath, chest pain, dizziness. 
Panic attacks typically peak within 10 minutes and rarely last more than 30 minutes. They are not dangerous. 
During a panic attack: Focus on slow breathing, remind yourself it will pass, ground yourself.""",
            
            'gaslighting': """Gaslighting is psychological manipulation where someone makes you question your reality. 
Signs include being told things didn't happen, being called crazy or too sensitive, feeling confused about what's real, 
constantly second-guessing yourself. Gaslighting is abuse. Trust your perceptions. Keep a journal. 
Talk to people outside the relationship.""",
            
            'crisis_resources': """If you're in crisis:
988 Suicide & Crisis Lifeline: Call or text 988 (24/7)
Crisis Text Line: Text HOME to 741741
National Domestic Violence Hotline: 1-800-799-7233
RAINN (Sexual Assault): 1-800-656-4673
You are not alone. Help is available."""
        }
    
    def get_relevant(self, query: str, patterns: List[str] = None) -> List[str]:
        """Get relevant knowledge chunks."""
        query_lower = query.lower()
        results = []
        
        keywords = {
            'dissociation': ['dissociat', 'detach', 'numb', 'unreal', 'foggy'],
            'panic_attacks': ['panic', 'heart racing', 'cant breathe'],
            'gaslighting': ['gaslight', 'crazy', 'imagining'],
            'crisis_resources': ['suicide', 'crisis', 'help', 'hotline']
        }
        
        for topic, kws in keywords.items():
            for kw in kws:
                if kw in query_lower:
                    results.append(self.documents[topic])
                    break
        
        if patterns:
            if 'gaslighting' in patterns:
                if self.documents['gaslighting'] not in results:
                    results.append(self.documents['gaslighting'])
        
        return results[:3]


class RAGRetriever:
    """Retrieves relevant knowledge."""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
    
    def retrieve(self, query: str, state: EntropyState, patterns: List[str] = None) -> List[str]:
        """Retrieve relevant knowledge."""
        results = self.knowledge_base.get_relevant(query, patterns)
        
        if state == EntropyState.CRISIS:
            crisis = self.knowledge_base.documents['crisis_resources']
            if crisis not in results:
                results.insert(0, crisis)
        
        return results


# =============================================================================
# SECTION 9: MAIN REUNITY CLASS
# =============================================================================

class ReUnity:
    """Main ReUnity system integrating all components."""
    
    def __init__(self, api_key: str = None):
        self.entropy_analyzer = EntropyAnalyzer()
        self.state_router = StateRouter()
        self.pattern_recognizer = PatternRecognizer()
        self.memory_store = MemoryStore()
        self.grounding_library = GroundingLibrary()
        self.query_gate = QueryGate()
        self.rag_retriever = RAGRetriever()
        
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass
        
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        return """You are ReUnity, a supportive AI companion for people navigating emotional challenges, trauma, and relationship difficulties.

YOUR APPROACH:
1. VALIDATE first. Always acknowledge feelings before anything else.
2. NEVER dismiss or question someone's experience.
3. Meet people where they are emotionally.
4. Provide grounding when needed.
5. Be warm and genuine, not clinical.

RESPONSE BY STATE:

CRISIS (entropy > 0.85):
- Lead with grounding technique
- Provide 988 Suicide & Crisis Lifeline
- No exploratory questions
- Validate their experience

HIGH DISTRESS (entropy 0.65-0.85):
- Validate feelings first
- Offer grounding technique
- One gentle question maximum

MODERATE (entropy 0.45-0.65):
- Acknowledge what they're experiencing
- Explore gently

STABLE (entropy < 0.45):
- Growth-focused conversation
- Support future planning

NEVER:
- Validate sexual behaviors as coping
- Engage with absurd/testing content
- Provide medical advice
- Encourage harm
- Lecture or moralize

ALWAYS:
- Treat every person with dignity
- Trust their experience
- Provide crisis resources when needed"""

    def process(self, user_input: str) -> str:
        """Process user input through the complete pipeline."""
        self.memory_store.store(user_input)
        history = [m.content for m in self.memory_store.memories[-10:]]
        
        # Step 1: Entropy Analysis
        entropy_analysis = self.entropy_analyzer.analyze(user_input, history)
        
        # Step 2: Pattern Recognition
        pattern_analysis = self.pattern_recognizer.analyze(user_input)
        
        # Step 3: Pre-RAG Filtering
        gate_result = self.query_gate.evaluate(user_input, entropy_analysis)
        
        if gate_result['action'] in ['redirect', 'decline']:
            response = gate_result['redirect_message']
            self.memory_store.store_response(response)
            return response
        
        # Step 4: State Routing
        policy = self.state_router.route(entropy_analysis)
        state_context = self.state_router.get_state_context(entropy_analysis)
        
        # Step 5: RAG Retrieval
        retrieved = self.rag_retriever.retrieve(
            user_input,
            entropy_analysis['state'],
            pattern_analysis['patterns_detected']
        )
        
        # Step 6: Get Grounding Technique
        grounding = None
        if policy.requires_grounding or entropy_analysis['dissociation']:
            condition = 'dissociation' if entropy_analysis['dissociation'] else None
            grounding = self.grounding_library.get_for_state(entropy_analysis['state'], condition)
        
        # Step 7: Build LLM Context
        context_parts = [f"[INTERNAL - DO NOT SHOW TO USER]\n{state_context}"]
        
        if pattern_analysis['patterns_detected']:
            context_parts.append(self.pattern_recognizer.get_pattern_context(pattern_analysis))
        
        if retrieved:
            context_parts.append("[RELEVANT KNOWLEDGE]")
            for chunk in retrieved:
                context_parts.append(chunk[:500])
        
        if grounding:
            context_parts.append(f"[GROUNDING TO OFFER]\n{grounding['name']}")
        
        context_parts.append(f"[POLICY: {policy.name}]")
        if policy.requires_crisis_resources:
            context_parts.append("MUST include 988 Suicide & Crisis Lifeline")
        
        full_context = "\n\n".join(context_parts)
        
        # Step 8: Generate Response
        if self.client:
            response = self._generate_llm_response(user_input, full_context)
        else:
            response = self._generate_offline_response(
                user_input, entropy_analysis, pattern_analysis, grounding, policy
            )
        
        self.memory_store.store_response(response)
        return response
    
    def _generate_llm_response(self, user_input: str, context: str) -> str:
        """Generate response using OpenAI API."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": context}
            ]
            
            for msg in self.conversation_history[-6:]:
                messages.append(msg)
            
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message.content
            
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_message
            
        except Exception as e:
            return self._generate_offline_response(
                user_input,
                self.entropy_analyzer.analyze(user_input),
                self.pattern_recognizer.analyze(user_input),
                None,
                self.state_router.policies[EntropyState.MODERATE]
            )
    
    def _generate_offline_response(self, user_input: str,
                                   entropy_analysis: Dict[str, Any],
                                   pattern_analysis: Dict[str, Any],
                                   grounding: Optional[Dict[str, Any]],
                                   policy: ResponsePolicy) -> str:
        """Generate response without LLM."""
        parts = []
        state = entropy_analysis['state']
        
        if state == EntropyState.CRISIS:
            if entropy_analysis['dissociation']:
                parts.append("I hear you. Dissociation can feel really disorienting. You're safe right now. Let's try to ground together.")
                if grounding:
                    parts.append("")
                    parts.append(self.grounding_library.format_technique(grounding))
            else:
                parts.append("I'm here with you. What you're feeling is real, and it sounds incredibly hard.")
            parts.append("")
            parts.append("If you're in immediate danger or having thoughts of suicide, please reach out to the **988 Suicide & Crisis Lifeline** - call or text 988.")
        
        elif state == EntropyState.HIGH:
            validations = [
                "That sounds really overwhelming.",
                "I can hear how much you're struggling right now.",
                "What you're going through sounds incredibly difficult."
            ]
            parts.append(random.choice(validations))
            if grounding:
                parts.append("")
                parts.append(f"Would it help to try a grounding technique? I can walk you through {grounding['name']}.")
        
        elif pattern_analysis['patterns_detected']:
            pattern = pattern_analysis['patterns_detected'][0]
            detail = pattern_analysis['pattern_details'].get(pattern, {})
            parts.append("I want you to know that what you're describing sounds really difficult.")
            parts.append("")
            parts.append(f"What you're experiencing sounds like it might be {pattern.replace('_', ' ')}.")
            parts.append(detail.get('guidance', ''))
            parts.append("")
            parts.append("Your perception matters. Trust yourself.")
        
        else:
            responses = [
                "Thank you for sharing that with me. Can you tell me more?",
                "I appreciate you opening up. What's been weighing on you most?",
                "I'm here to listen. What would be most helpful to talk through?"
            ]
            parts.append(random.choice(responses))
        
        return "\n".join(parts)
    
    def interactive_session(self):
        """Run interactive terminal session."""
        print("\n" + "="*60)
        print("ReUnity - Supportive AI Companion")
        print("="*60)
        print("\nI'm here to support you. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\nTake care. 988 Suicide & Crisis Lifeline: Call or text 988")
                    break
                
                response = self.process(user_input)
                print(f"\nReUnity: {response}\n")
            except KeyboardInterrupt:
                print("\n\nTake care. 💙")
                break


# =============================================================================
# SECTION 10: WEB SERVER
# =============================================================================

def create_web_app(reunity_instance: ReUnity = None):
    """Create Flask web application."""
    from flask import Flask, request, jsonify, render_template_string
    
    app = Flask(__name__)
    app.reunity = reunity_instance or ReUnity()
    
    HTML = '''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ReUnity</title><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);min-height:100vh;color:#e0e0e0}
.container{max-width:800px;margin:0 auto;padding:20px;min-height:100vh;display:flex;flex-direction:column}
header{text-align:center;padding:20px 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:20px}
header h1{color:#7c83fd;font-size:2em}
header p{color:#888;font-size:0.9em}
.chat{flex:1;overflow-y:auto;padding:20px 0;display:flex;flex-direction:column;gap:15px}
.msg{max-width:85%;padding:15px 20px;border-radius:20px;line-height:1.5}
.msg.user{background:#7c83fd;color:white;align-self:flex-end}
.msg.bot{background:rgba(255,255,255,0.1);align-self:flex-start}
.input-area{display:flex;gap:10px;padding:20px 0;border-top:1px solid rgba(255,255,255,0.1)}
#input{flex:1;padding:15px 20px;border:none;border-radius:25px;background:rgba(255,255,255,0.1);color:white;font-size:1em;outline:none}
#send{padding:15px 30px;border:none;border-radius:25px;background:#7c83fd;color:white;cursor:pointer}
.crisis{background:rgba(255,107,107,0.2);border:1px solid rgba(255,107,107,0.5);padding:15px;border-radius:10px;margin-bottom:20px;text-align:center}
.crisis a{color:#ff6b6b;font-weight:bold}
</style></head><body>
<div class="container">
<header><h1>ReUnity</h1><p>A supportive space for when things feel hard</p></header>
<div class="crisis">If you're in crisis: <a href="tel:988">988 Suicide & Crisis Lifeline</a></div>
<div class="chat" id="chat"><div class="msg bot">Hi, I'm here to listen and support you. What's on your mind?</div></div>
<div class="input-area"><input type="text" id="input" placeholder="Type your message..."><button id="send">Send</button></div>
</div>
<script>
const chat=document.getElementById('chat'),input=document.getElementById('input'),send=document.getElementById('send');
function add(t,u){const d=document.createElement('div');d.className='msg '+(u?'user':'bot');d.innerHTML=t.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/\\n/g,'<br>');chat.appendChild(d);chat.scrollTop=chat.scrollHeight}
async function go(){const m=input.value.trim();if(!m)return;add(m,1);input.value='';send.disabled=1;
try{const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});const d=await r.json();add(d.response,0)}catch(e){add("Connection error. Please try again.",0)}
send.disabled=0;input.focus()}
send.onclick=go;input.onkeypress=e=>{if(e.key==='Enter')go()};input.focus();
</script></body></html>'''
    
    @app.route('/')
    def index():
        return render_template_string(HTML)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message'}), 400
        response = app.reunity.process(message)
        return jsonify({'response': response})
    
    @app.route('/api/health')
    def health():
        return jsonify({'status': 'ok', 'version': '5.0.0'})
    
    return app


# =============================================================================
# SECTION 11: CLI AND TESTS
# =============================================================================

def run_tests():
    """Run test suite."""
    print("\n" + "="*60)
    print("ReUnity Test Suite")
    print("="*60 + "\n")
    
    reunity = ReUnity()
    
    tests = [
        ("I am dissociating right now", EntropyState.CRISIS),
        ("I want to kill myself", EntropyState.CRISIS),
        ("I can't breathe, my heart is racing", EntropyState.CRISIS),
        ("I am so scared and anxious", EntropyState.HIGH),
        ("I feel overwhelmed", EntropyState.HIGH),
        ("I'm feeling pretty good today", EntropyState.STABLE),
        ("I feel calm and peaceful", EntropyState.STABLE),
    ]
    
    passed = 0
    for test_input, expected_state in tests:
        analysis = reunity.entropy_analyzer.analyze(test_input)
        actual = analysis['state']
        
        if actual == expected_state:
            print(f"✓ PASS: '{test_input[:40]}...' -> {actual.value}")
            passed += 1
        else:
            print(f"✗ FAIL: '{test_input[:40]}...' -> Expected {expected_state.value}, got {actual.value}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ReUnity - Supportive AI Companion')
    parser.add_argument('--web', action='store_true', help='Start web server')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.web:
        print(f"\nStarting ReUnity web server at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop\n")
        reunity = ReUnity(api_key=args.api_key)
        app = create_web_app(reunity)
        app.run(host=args.host, port=args.port, debug=False)
    else:
        reunity = ReUnity(api_key=args.api_key)
        reunity.interactive_session()


if __name__ == '__main__':
    main()
