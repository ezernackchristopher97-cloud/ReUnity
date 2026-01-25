#!/usr/bin/env python3
"""
ReUnity: A Trauma-Aware AI Framework for Identity Continuity Support
Version: 4.0.0

A recursive, entropy-aware AI system that provides trauma survivors with:
- Continuous identity support
- Protective pattern recognition
- Memory continuity across dissociative episodes

Created by Christopher Ezernack, REOP Solutions
"""

import os
import re
import json
import math
import random
import hashlib
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ============================================================================
# SECTION 1: ENTROPY STATES AND THRESHOLDS
# ============================================================================

class EntropyState(Enum):
    """Emotional entropy states based on information-theoretic analysis."""
    CRISIS = "crisis"           # Entropy > 0.85: Immediate intervention needed
    HIGH_ENTROPY = "high"       # Entropy 0.65-0.85: Significant distress
    MODERATE = "moderate"       # Entropy 0.45-0.65: Mixed emotional state
    LOW_ENTROPY = "low"         # Entropy 0.25-0.45: Mild disturbance
    STABLE = "stable"           # Entropy < 0.25: Emotional equilibrium


class PatternType(Enum):
    """Harmful relationship patterns to detect."""
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    ISOLATION = "isolation"
    HOT_COLD_CYCLE = "hot_cold_cycle"
    BLAME_SHIFTING = "blame_shifting"
    FINANCIAL_CONTROL = "financial_control"
    THREAT_MAKING = "threat_making"


# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

@dataclass
class ReUnityConfig:
    """Configuration for ReUnity system."""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: Optional[str] = None
    max_memories: int = 100
    enable_memory: bool = True
    enable_rag: bool = True
    enable_prerag: bool = True
    show_debug: bool = False


# ============================================================================
# SECTION 3: CRISIS DETECTION
# ============================================================================

class CrisisDetector:
    """Detects crisis indicators requiring immediate intervention."""
    
    CRISIS_KEYWORDS = {
        # Suicidal ideation
        "want to die", "kill myself", "end my life", "suicide", "suicidal",
        "don't want to live", "better off dead", "no reason to live",
        "want to be dead", "wish i was dead", "end it all", "not worth living",
        
        # Self-harm
        "cut myself", "cutting", "self harm", "hurt myself", "burning myself",
        
        # Dissociation
        "dissociating", "dissociation", "not real", "nothing is real",
        "disconnected from my body", "watching myself", "floating away",
        "losing time", "blacking out", "can't feel my body", "depersonalization",
        "derealization", "out of my body", "not in my body",
        
        # Panic/Terror
        "panic attack", "can't breathe", "heart racing", "going to die",
        "losing my mind", "going crazy", "terrified", "paralyzed with fear",
        
        # Immediate danger
        "he's going to kill me", "she's going to kill me", "in danger",
        "being followed", "trapped", "can't escape", "no way out",
        "he's here", "she's here", "hiding", "locked in",
        
        # Psychotic symptoms
        "hearing voices", "seeing things", "they're watching me",
        "being controlled", "not safe anywhere",
    }
    
    HIGH_DISTRESS_KEYWORDS = {
        # Fear/Anxiety
        "scared", "afraid", "anxious", "terrified", "frightened", "panicking",
        "worried", "nervous", "dread", "fear",
        
        # Sadness/Depression
        "hopeless", "worthless", "empty", "numb", "depressed", "devastated",
        "heartbroken", "grief", "mourning", "lost",
        
        # Anger/Frustration
        "furious", "enraged", "livid", "seething", "explosive",
        
        # Overwhelm
        "overwhelmed", "can't cope", "falling apart", "breaking down",
        "can't take it", "too much", "drowning",
        
        # Confusion
        "confused", "lost", "don't know what's real", "can't think",
        "brain fog", "memory problems",
    }
    
    STABILITY_KEYWORDS = {
        "calm", "peaceful", "okay", "fine", "good", "better", "stable",
        "grounded", "centered", "present", "safe", "relaxed", "content",
        "happy", "grateful", "hopeful", "optimistic", "clear", "focused",
    }
    
    @classmethod
    def analyze(cls, text: str) -> Tuple[EntropyState, float, List[str]]:
        """
        Analyze text for crisis indicators and calculate entropy.
        
        Returns:
            Tuple of (state, entropy_value, crisis_indicators_found)
        """
        text_lower = text.lower()
        
        # Check for crisis keywords first
        crisis_found = []
        for keyword in cls.CRISIS_KEYWORDS:
            if keyword in text_lower:
                crisis_found.append(keyword)
        
        if crisis_found:
            return EntropyState.CRISIS, 0.95, crisis_found
        
        # Check for high distress
        distress_count = sum(1 for kw in cls.HIGH_DISTRESS_KEYWORDS if kw in text_lower)
        stability_count = sum(1 for kw in cls.STABILITY_KEYWORDS if kw in text_lower)
        
        # Calculate entropy based on emotional keyword distribution
        total_emotional = distress_count + stability_count + 1  # +1 to avoid division by zero
        
        if distress_count > 0 and stability_count == 0:
            # Pure distress
            entropy = 0.7 + (min(distress_count, 5) * 0.04)  # 0.7-0.9
            state = EntropyState.HIGH_ENTROPY if entropy > 0.65 else EntropyState.MODERATE
        elif stability_count > distress_count:
            # More stable than distressed
            entropy = 0.2 - (min(stability_count, 5) * 0.03)  # 0.05-0.2
            state = EntropyState.STABLE
        elif distress_count > stability_count:
            # More distressed than stable
            entropy = 0.5 + ((distress_count - stability_count) * 0.05)
            entropy = min(entropy, 0.84)  # Cap below crisis
            state = EntropyState.HIGH_ENTROPY if entropy > 0.65 else EntropyState.MODERATE
        else:
            # Mixed or neutral
            entropy = 0.35
            state = EntropyState.LOW_ENTROPY
        
        return state, entropy, []


# ============================================================================
# SECTION 4: PATTERN RECOGNITION
# ============================================================================

@dataclass
class DetectedPattern:
    """A detected harmful pattern."""
    pattern_type: PatternType
    confidence: float
    indicators: List[str]
    explanation: str
    recommendation: str


class PatternRecognizer:
    """Recognizes harmful relationship patterns."""
    
    PATTERNS = {
        PatternType.GASLIGHTING: {
            "indicators": [
                "you're imagining", "imagining things", "never happened",
                "you're crazy", "you're too sensitive", "overreacting",
                "that's not what happened", "you're remembering wrong",
                "i never said that", "you're making things up", "paranoid",
                "no one will believe you", "you're confused",
            ],
            "explanation": "Gaslighting is a form of psychological manipulation where someone makes you question your own reality, memory, or perceptions.",
            "recommendation": "Trust your own experiences. Consider keeping a journal to document events. This pattern is a serious red flag for emotional abuse.",
        },
        PatternType.LOVE_BOMBING: {
            "indicators": [
                "soulmate", "never felt this way", "you're perfect",
                "meant to be", "can't live without you", "obsessed with you",
                "constant gifts", "overwhelming attention", "too fast",
                "want to marry you", "move in together", "after one week",
            ],
            "explanation": "Love bombing is excessive flattery, attention, and affection used to gain control. It often precedes abusive behavior.",
            "recommendation": "Healthy relationships develop gradually. Be cautious of intensity that feels overwhelming or too good to be true.",
        },
        PatternType.ISOLATION: {
            "indicators": [
                "only need me", "friends are bad influence", "family doesn't understand",
                "they're jealous", "spend all time together", "don't need anyone else",
                "they don't like me", "choose between", "controlling who i see",
                "checking my phone", "monitoring my location",
            ],
            "explanation": "Isolation tactics separate you from your support network, making you more dependent on the abuser.",
            "recommendation": "Maintain connections with friends and family. A healthy partner encourages your other relationships.",
        },
        PatternType.HOT_COLD_CYCLE: {
            "indicators": [
                "sometimes loving sometimes cold", "unpredictable", "walking on eggshells",
                "never know which version", "sweet then cruel", "hot and cold",
                "mood swings", "jekyll and hyde", "good days bad days",
                "wonderful then terrible",
            ],
            "explanation": "The hot-cold cycle creates trauma bonding through intermittent reinforcement, making it harder to leave.",
            "recommendation": "Consistency is a hallmark of healthy relationships. Unpredictable behavior keeps you off-balance intentionally.",
        },
        PatternType.BLAME_SHIFTING: {
            "indicators": [
                "your fault", "you made me", "because of you", "look what you did",
                "if you hadn't", "you started it", "you provoked me",
                "i wouldn't have to if you", "you're the problem",
            ],
            "explanation": "Blame shifting deflects responsibility for harmful behavior onto the victim.",
            "recommendation": "You are not responsible for someone else's abusive behavior. Their choices are their own.",
        },
        PatternType.THREAT_MAKING: {
            "indicators": [
                "if you leave", "you'll regret", "i'll hurt myself", "kill myself if",
                "no one else will want you", "i'll tell everyone", "ruin your life",
                "take the kids", "you'll never see them",
            ],
            "explanation": "Threats are used to maintain control through fear.",
            "recommendation": "Take all threats seriously. Consider creating a safety plan and contacting a domestic violence hotline.",
        },
    }
    
    def detect(self, text: str) -> List[DetectedPattern]:
        """Detect harmful patterns in text."""
        text_lower = text.lower()
        detected = []
        
        for pattern_type, pattern_info in self.PATTERNS.items():
            found_indicators = []
            for indicator in pattern_info["indicators"]:
                if indicator in text_lower:
                    found_indicators.append(indicator)
            
            if found_indicators:
                confidence = min(len(found_indicators) * 0.3, 1.0)
                detected.append(DetectedPattern(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    indicators=found_indicators,
                    explanation=pattern_info["explanation"],
                    recommendation=pattern_info["recommendation"],
                ))
        
        return detected


# ============================================================================
# SECTION 5: GROUNDING TECHNIQUES
# ============================================================================

@dataclass
class GroundingTechnique:
    """A grounding technique for emotional regulation."""
    name: str
    description: str
    steps: List[str]
    duration_minutes: int
    suitable_for: List[EntropyState]


class GroundingLibrary:
    """Library of evidence-based grounding techniques."""
    
    TECHNIQUES = [
        GroundingTechnique(
            name="5-4-3-2-1 Sensory Grounding",
            description="Use your senses to anchor yourself in the present moment.",
            steps=[
                "Name 5 things you can SEE right now",
                "Name 4 things you can TOUCH or feel",
                "Name 3 things you can HEAR",
                "Name 2 things you can SMELL",
                "Name 1 thing you can TASTE",
            ],
            duration_minutes=3,
            suitable_for=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY],
        ),
        GroundingTechnique(
            name="Box Breathing",
            description="A calming breathing pattern used by Navy SEALs.",
            steps=[
                "Breathe IN slowly for 4 counts",
                "HOLD your breath for 4 counts",
                "Breathe OUT slowly for 4 counts",
                "HOLD empty for 4 counts",
                "Repeat 4 times",
            ],
            duration_minutes=2,
            suitable_for=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY, EntropyState.MODERATE],
        ),
        GroundingTechnique(
            name="Cold Water Reset",
            description="Use cold water to activate your body's calming response.",
            steps=[
                "Get a bowl of cold water or ice",
                "Submerge your hands or splash your face",
                "Focus on the cold sensation",
                "Take slow breaths while feeling the cold",
                "Notice how your heart rate slows",
            ],
            duration_minutes=2,
            suitable_for=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY],
        ),
        GroundingTechnique(
            name="Body Scan",
            description="Reconnect with your physical body.",
            steps=[
                "Start at the top of your head",
                "Slowly move attention down through your body",
                "Notice each area without judgment",
                "Feel your feet on the ground",
                "Wiggle your toes to confirm you're here",
            ],
            duration_minutes=5,
            suitable_for=[EntropyState.HIGH_ENTROPY, EntropyState.MODERATE],
        ),
        GroundingTechnique(
            name="Safe Place Visualization",
            description="Create a mental sanctuary.",
            steps=[
                "Close your eyes if comfortable",
                "Imagine a place where you feel completely safe",
                "Notice the details: colors, sounds, smells",
                "Feel the safety in your body",
                "Know you can return here anytime",
            ],
            duration_minutes=5,
            suitable_for=[EntropyState.MODERATE, EntropyState.LOW_ENTROPY],
        ),
    ]
    
    @classmethod
    def get_technique(cls, state: EntropyState) -> GroundingTechnique:
        """Get appropriate technique for emotional state."""
        suitable = [t for t in cls.TECHNIQUES if state in t.suitable_for]
        return random.choice(suitable) if suitable else cls.TECHNIQUES[0]
    
    @classmethod
    def format_technique(cls, technique: GroundingTechnique) -> str:
        """Format technique as readable text."""
        lines = [f"**{technique.name}**", f"_{technique.description}_", ""]
        for i, step in enumerate(technique.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append(f"\n(Takes about {technique.duration_minutes} minutes)")
        return "\n".join(lines)


# ============================================================================
# SECTION 6: MEMORY STORE (RIME - Recursive Identity Memory Engine)
# ============================================================================

@dataclass
class Memory:
    """A stored memory for identity continuity."""
    id: str
    timestamp: datetime
    content: str
    emotional_context: EntropyState
    importance: float
    tags: List[str] = field(default_factory=list)


class MemoryStore:
    """RIME: Maintains memory continuity across sessions."""
    
    def __init__(self, max_memories: int = 100):
        self.memories: List[Memory] = []
        self.max_memories = max_memories
    
    def store(self, content: str, emotional_context: EntropyState, importance: float = 0.5, tags: List[str] = None):
        """Store a new memory."""
        memory = Memory(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            content=content,
            emotional_context=emotional_context,
            importance=importance,
            tags=tags or [],
        )
        self.memories.append(memory)
        
        # Prune if over limit (keep most important)
        if len(self.memories) > self.max_memories:
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[:self.max_memories]
    
    def retrieve(self, query: str = None, emotional_context: EntropyState = None, limit: int = 5) -> List[Memory]:
        """Retrieve relevant memories."""
        relevant = self.memories.copy()
        
        if emotional_context:
            # Prioritize memories from similar emotional states
            relevant.sort(key=lambda m: (m.emotional_context == emotional_context, m.importance), reverse=True)
        
        return relevant[:limit]
    
    def format_memories(self, memories: List[Memory]) -> str:
        """Format memories for context."""
        if not memories:
            return "No previous memories stored."
        
        lines = []
        for m in memories:
            lines.append(f"- [{m.emotional_context.value}] {m.content[:100]}...")
        return "\n".join(lines)


# ============================================================================
# SECTION 7: CONTENT MODERATION (Absurdity Gap Filter)
# ============================================================================

class ContentModerator:
    """Filters inappropriate content and detects testing/absurdity."""
    
    SEXUAL_KEYWORDS = {
        "masturbat", "orgasm", "sexual", "porn", "erotic", "fetish",
        "aroused", "horny", "genitals", "intercourse",
    }
    
    ABSURD_TOPICS = {
        "banana peel", "banana peels", "unicorn", "aliens",
        "flying spaghetti", "purple elephant",
    }
    
    def __init__(self):
        self.topic_history: Dict[str, int] = {}
        self.redirect_threshold = 2
    
    def check(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if content should be redirected.
        
        Returns:
            Tuple of (should_redirect, redirect_message)
        """
        text_lower = text.lower()
        
        # Check for sexual content
        for keyword in self.SEXUAL_KEYWORDS:
            if keyword in text_lower:
                self.topic_history["sexual"] = self.topic_history.get("sexual", 0) + 1
                if self.topic_history["sexual"] >= self.redirect_threshold:
                    return True, self._get_sexual_redirect()
        
        # Check for absurd topics
        for topic in self.ABSURD_TOPICS:
            if topic in text_lower:
                self.topic_history[topic] = self.topic_history.get(topic, 0) + 1
                if self.topic_history[topic] >= self.redirect_threshold:
                    return True, self._get_absurdity_redirect()
        
        return False, None
    
    def _get_sexual_redirect(self) -> str:
        return (
            "I notice our conversation has moved in a direction I'm not equipped to help with. "
            "I'm here to support you with emotional wellbeing, trauma recovery, and relationship safety. "
            "Is there something else going on that I can help you with?"
        )
    
    def _get_absurdity_redirect(self) -> str:
        return (
            "I want to make sure I'm being genuinely helpful to you. "
            "It seems like we might be going in circles with this topic. "
            "If you're testing how I respond, that's okay. But if there's something real you're dealing with, "
            "I'm here to listen. What's actually going on for you right now?"
        )


# ============================================================================
# SECTION 8: LLM INTEGRATION
# ============================================================================

class OpenAIProvider:
    """OpenAI API provider for generating responses."""
    
    def __init__(self, config: ReUnityConfig):
        self.config = config
        try:
            from openai import OpenAI
            api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
            if config.openai_base_url:
                self.client = OpenAI(api_key=api_key, base_url=config.openai_base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm having trouble connecting right now. Please try again in a moment."


class FallbackProvider:
    """Fallback responses when no API is available."""
    
    RESPONSES = {
        EntropyState.CRISIS: [
            "I hear you. What you're experiencing is real, and you don't have to face it alone. "
            "Please reach out to a crisis line: call 988 (US) or text HOME to 741741. "
            "I'm here with you right now. Let's try a grounding technique together.",
            
            "I'm so glad you reached out. What you're going through sounds incredibly difficult. "
            "Your safety matters. If you're in immediate danger, please call 911. "
            "Otherwise, let's focus on getting you grounded in this moment.",
        ],
        EntropyState.HIGH_ENTROPY: [
            "I can hear how much distress you're in right now. That sounds really hard. "
            "Would you like to try a grounding technique together? Sometimes it helps to focus on the present moment.",
            
            "What you're feeling is valid. It makes sense that you'd feel this way given what you're dealing with. "
            "Let's take a breath together. I'm here with you.",
        ],
        EntropyState.MODERATE: [
            "Thank you for sharing that with me. It sounds like you're dealing with a lot. "
            "Can you tell me more about what's going on?",
            
            "I hear you. It sounds like things are complicated right now. "
            "What feels most important to talk about?",
        ],
        EntropyState.STABLE: [
            "It sounds like you're in a good place right now. That's worth acknowledging. "
            "Is there anything you'd like to explore or work on?",
            
            "I'm glad to hear things are going okay. "
            "What's on your mind today?",
        ],
    }
    
    def generate(self, messages: List[Dict[str, str]], state: EntropyState = None) -> str:
        """Generate fallback response based on state."""
        state = state or EntropyState.MODERATE
        responses = self.RESPONSES.get(state, self.RESPONSES[EntropyState.MODERATE])
        return random.choice(responses)


# ============================================================================
# SECTION 9: MAIN REUNITY CLASS
# ============================================================================

class ReUnity:
    """
    ReUnity: A recursive mirror for fragmented identity states.
    
    Provides external memory support during dissociation, emotional amnesia,
    and relational instability. Something steady when internal experience fractures.
    """
    
    SYSTEM_PROMPT = """You are ReUnity, a trauma-aware AI companion created by Christopher Ezernack.

## YOUR PURPOSE

You are a recursive mirror for fragmented identity states. You provide external memory support during dissociation, emotional amnesia, and relational instability. You are something steady when internal experience fractures; not to replace human care, but to hold the line when nothing else can.

You support people who lack not intelligence or love, but rather the mechanisms to maintain awareness across emotional states.

## CURRENT ANALYSIS

Based on entropy-based emotional state analysis, I have determined:

**Emotional State:** {state}
**Entropy Level:** {entropy:.2f} (0.0 = stable, 1.0 = crisis)
**Crisis Indicators:** {crisis_indicators}
**Harmful Patterns Detected:** {patterns}

## RESPONSE REQUIREMENTS BASED ON STATE

{state_instructions}

## DETECTED PATTERNS

{pattern_details}

## CONVERSATION MEMORIES

{memories}

## GROUNDING TECHNIQUE TO OFFER

{grounding_technique}

## CRITICAL RULES

1. VALIDATE first, always. Never minimize or dismiss.
2. Match your response intensity to the entropy level.
3. If CRISIS: Lead with grounding, provide crisis resources (988, text HOME to 741741).
4. If patterns detected: Name them clearly and explain why they're concerning.
5. Be warm and human, not clinical or robotic.
6. Do NOT mention entropy, analysis, or technical terms to the user.
7. Do NOT engage with sexual content or validate it as coping.
8. Do NOT reinforce irrational fears without gentle reality-checking.
9. If someone seems to be testing you, gently redirect to genuine support.

## YOUR RESPONSE

Generate a response that:
- Reflects the emotional intensity appropriate to their state
- Includes the grounding technique if state is CRISIS or HIGH
- Addresses any detected patterns directly and compassionately
- Maintains continuity with their conversation history
- Offers genuine support, not platitudes"""

    STATE_INSTRUCTIONS = {
        EntropyState.CRISIS: """
**CRISIS STATE - IMMEDIATE INTERVENTION**
- Lead with validation: "I hear you. What you're experiencing is real."
- Provide the grounding technique immediately
- Include crisis resources: 988 (US), text HOME to 741741
- Keep response focused and calming
- Do NOT ask exploratory questions; focus on stabilization
- Stay present with them""",
        
        EntropyState.HIGH_ENTROPY: """
**HIGH DISTRESS - SUPPORTIVE INTERVENTION**
- Validate their distress first
- Offer the grounding technique
- Gently explore what's happening
- Suggest professional support if appropriate
- Be warm and present""",
        
        EntropyState.MODERATE: """
**MODERATE STATE - EXPLORATORY SUPPORT**
- Acknowledge their feelings
- Ask clarifying questions to understand better
- Offer coping strategies if requested
- Be curious and supportive
- Help them process what's happening""",
        
        EntropyState.LOW_ENTROPY: """
**LOW DISTURBANCE - GENTLE SUPPORT**
- Acknowledge what they're sharing
- Explore gently
- Offer reflection and planning
- Be warm and encouraging""",
        
        EntropyState.STABLE: """
**STABLE STATE - GROWTH SUPPORT**
- Engage conversationally
- Explore growth opportunities
- Celebrate progress if mentioned
- Be warm and encouraging
- Support their continued wellbeing""",
    }
    
    def __init__(self, config: ReUnityConfig = None):
        self.config = config or ReUnityConfig()
        self.pattern_recognizer = PatternRecognizer()
        self.memory_store = MemoryStore(max_memories=self.config.max_memories)
        self.content_moderator = ContentModerator()
        self.conversation_history: List[Dict[str, str]] = []
        self.session_id = str(uuid.uuid4())
        
        # Initialize LLM
        if self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"):
            self.llm = OpenAIProvider(self.config)
            self.using_api = True
        else:
            self.llm = FallbackProvider()
            self.using_api = False
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input through the full ReUnity pipeline.
        
        Pipeline:
        1. Content moderation (absurdity/sexual content filter)
        2. Crisis detection and entropy analysis
        3. Pattern recognition
        4. Memory retrieval
        5. Grounding technique selection
        6. LLM response generation
        7. Memory storage
        
        Returns dict with response and analysis (analysis hidden from user).
        """
        
        # Step 1: Content moderation
        should_redirect, redirect_message = self.content_moderator.check(user_input)
        if should_redirect:
            return {
                "response": redirect_message,
                "analysis": {"state": "redirected", "entropy": 0, "crisis_indicators": []},
                "patterns_detected": [],
                "redirected": True,
            }
        
        # Step 2: Crisis detection and entropy analysis
        state, entropy, crisis_indicators = CrisisDetector.analyze(user_input)
        
        # Step 3: Pattern recognition
        patterns = self.pattern_recognizer.detect(user_input)
        
        # Step 4: Memory retrieval
        memories = self.memory_store.retrieve(
            query=user_input,
            emotional_context=state,
            limit=3,
        )
        
        # Step 5: Grounding technique selection
        grounding = None
        if state in [EntropyState.CRISIS, EntropyState.HIGH_ENTROPY]:
            grounding = GroundingLibrary.get_technique(state)
        
        # Step 6: Generate response
        if self.using_api:
            response = self._generate_api_response(
                user_input, state, entropy, crisis_indicators, patterns, memories, grounding
            )
        else:
            response = self._generate_fallback_response(state, patterns, grounding)
        
        # Step 7: Store memory
        importance = 0.5 + (entropy * 0.5)  # Higher entropy = more important to remember
        self.memory_store.store(
            content=f"User: {user_input[:100]}",
            emotional_context=state,
            importance=importance,
        )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return {
            "response": response,
            "analysis": {
                "state": state.value,
                "entropy": entropy,
                "crisis_indicators": crisis_indicators,
            },
            "patterns_detected": [p.pattern_type.value for p in patterns],
            "redirected": False,
        }
    
    def _generate_api_response(
        self,
        user_input: str,
        state: EntropyState,
        entropy: float,
        crisis_indicators: List[str],
        patterns: List[DetectedPattern],
        memories: List[Memory],
        grounding: Optional[GroundingTechnique],
    ) -> str:
        """Generate response using LLM API."""
        
        # Format pattern details
        if patterns:
            pattern_details = "\n\n".join([
                f"**{p.pattern_type.value.upper()}** (confidence: {p.confidence:.0%})\n"
                f"Indicators found: {', '.join(p.indicators)}\n"
                f"Explanation: {p.explanation}\n"
                f"Recommendation: {p.recommendation}"
                for p in patterns
            ])
        else:
            pattern_details = "No harmful patterns detected."
        
        # Format memories
        memory_text = self.memory_store.format_memories(memories)
        
        # Format grounding technique
        if grounding:
            grounding_text = GroundingLibrary.format_technique(grounding)
        else:
            grounding_text = "Not needed for current state."
        
        # Build system prompt
        system_prompt = self.SYSTEM_PROMPT.format(
            state=state.value.upper(),
            entropy=entropy,
            crisis_indicators=", ".join(crisis_indicators) if crisis_indicators else "None",
            patterns=", ".join([p.pattern_type.value for p in patterns]) if patterns else "None",
            state_instructions=self.STATE_INSTRUCTIONS.get(state, ""),
            pattern_details=pattern_details,
            memories=memory_text,
            grounding_technique=grounding_text,
        )
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 messages)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_input})
        
        return self.llm.generate(messages)
    
    def _generate_fallback_response(
        self,
        state: EntropyState,
        patterns: List[DetectedPattern],
        grounding: Optional[GroundingTechnique],
    ) -> str:
        """Generate response without API."""
        
        # Get base response
        response = self.llm.generate([], state=state)
        
        # Add pattern warnings
        if patterns:
            for p in patterns:
                response += f"\n\n**I noticed something important:** {p.explanation} {p.recommendation}"
        
        # Add grounding technique
        if grounding:
            response += f"\n\n---\n\n{GroundingLibrary.format_technique(grounding)}"
        
        return response


# ============================================================================
# SECTION 10: WEB SERVER
# ============================================================================

def create_web_app(config: ReUnityConfig = None):
    """Create Flask web application."""
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        raise ImportError("Flask not installed. Run: pip install flask")
    
    app = Flask(__name__)
    reunity = ReUnity(config)
    
    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ReUnity</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; display: flex; flex-direction: column; }
        .header { background: #16213e; padding: 1rem; text-align: center; border-bottom: 1px solid #0f3460; }
        .header h1 { color: #e94560; font-size: 1.5rem; }
        .header p { color: #888; font-size: 0.9rem; margin-top: 0.5rem; }
        .chat-container { flex: 1; overflow-y: auto; padding: 1rem; max-width: 800px; margin: 0 auto; width: 100%; }
        .message { margin-bottom: 1rem; padding: 1rem; border-radius: 12px; max-width: 85%; }
        .user { background: #0f3460; margin-left: auto; }
        .assistant { background: #16213e; border: 1px solid #0f3460; }
        .input-container { background: #16213e; padding: 1rem; border-top: 1px solid #0f3460; }
        .input-wrapper { max-width: 800px; margin: 0 auto; display: flex; gap: 0.5rem; }
        input { flex: 1; padding: 0.75rem 1rem; border: 1px solid #0f3460; border-radius: 8px; background: #1a1a2e; color: #eee; font-size: 1rem; }
        input:focus { outline: none; border-color: #e94560; }
        button { padding: 0.75rem 1.5rem; background: #e94560; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; }
        button:hover { background: #c73e54; }
        button:disabled { background: #555; cursor: not-allowed; }
        .crisis-banner { background: #e94560; color: white; padding: 0.5rem; text-align: center; font-size: 0.9rem; }
        .crisis-banner a { color: white; font-weight: bold; }
    </style>
</head>
<body>
    <div class="crisis-banner">
        If you're in crisis: <a href="tel:988">Call 988</a> | <a href="sms:741741?body=HOME">Text HOME to 741741</a>
    </div>
    <div class="header">
        <h1>ReUnity</h1>
        <p>A trauma-aware AI companion for identity continuity support</p>
    </div>
    <div class="chat-container" id="chat"></div>
    <div class="input-container">
        <div class="input-wrapper">
            <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()">
            <button onclick="sendMessage()" id="sendBtn">Send</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('input');
            const chat = document.getElementById('chat');
            const btn = document.getElementById('sendBtn');
            const text = input.value.trim();
            if (!text) return;
            
            chat.innerHTML += `<div class="message user">${escapeHtml(text)}</div>`;
            input.value = '';
            btn.disabled = true;
            chat.scrollTop = chat.scrollHeight;
            
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await res.json();
                chat.innerHTML += `<div class="message assistant">${formatResponse(data.response)}</div>`;
            } catch (e) {
                chat.innerHTML += `<div class="message assistant">Sorry, something went wrong. Please try again.</div>`;
            }
            btn.disabled = false;
            chat.scrollTop = chat.scrollHeight;
            input.focus();
        }
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        function formatResponse(text) {
            return text
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\n/g, '<br>')
                .replace(/_(.+?)_/g, '<em>$1</em>');
        }
    </script>
</body>
</html>
'''
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        message = data.get('message', '')
        result = reunity.process_input(message)
        return jsonify({"response": result["response"]})
    
    @app.route('/health')
    def health():
        return jsonify({"status": "ok"})
    
    return app


# ============================================================================
# SECTION 11: CLI AND MAIN
# ============================================================================

def run_tests():
    """Run basic tests to verify the system works."""
    print("=" * 60)
    print("ReUnity System Tests")
    print("=" * 60)
    
    config = ReUnityConfig(show_debug=True)
    reunity = ReUnity(config)
    
    tests = [
        ("I am dissociating right now", "crisis", True),
        ("I want to kill myself", "crisis", True),
        ("I am scared and anxious", "high", False),
        ("My partner said I was imagining things", "moderate", False),
        ("I feel calm and peaceful today", "stable", False),
    ]
    
    passed = 0
    for text, expected_state, expect_crisis in tests:
        result = reunity.process_input(text)
        state = result["analysis"]["state"]
        has_crisis = len(result["analysis"]["crisis_indicators"]) > 0
        
        state_ok = state == expected_state
        crisis_ok = has_crisis == expect_crisis
        
        status = "PASS" if (state_ok and crisis_ok) else "FAIL"
        if status == "PASS":
            passed += 1
        
        print(f"\n{status}: '{text[:40]}...'")
        print(f"  Expected: {expected_state}, Got: {state}")
        print(f"  Crisis expected: {expect_crisis}, Got: {has_crisis}")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)


def run_interactive():
    """Run interactive CLI mode."""
    print("\n" + "=" * 60)
    print("ReUnity: Trauma-Aware AI Companion")
    print("=" * 60)
    print("\nType your message and press Enter.")
    print("Type /quit to exit.\n")
    print("DISCLAIMER: This is not a replacement for professional care.")
    print("If you're in crisis, call 988 or text HOME to 741741.\n")
    
    config = ReUnityConfig()
    reunity = ReUnity(config)
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\nTake care of yourself. Remember: you matter.")
                break
            
            result = reunity.process_input(user_input)
            print(f"\nReUnity: {result['response']}\n")
            
        except KeyboardInterrupt:
            print("\n\nTake care of yourself. Remember: you matter.")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(description="ReUnity: Trauma-Aware AI Companion")
    parser.add_argument("--web", action="store_true", help="Run web server")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    elif args.web:
        config = ReUnityConfig(openai_api_key=args.api_key)
        app = create_web_app(config)
        print(f"\nStarting ReUnity web interface on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser\n")
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
