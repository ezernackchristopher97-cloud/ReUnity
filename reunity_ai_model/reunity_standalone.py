"""
ReUnity AI Model v3.1.0
Author: Christopher Ezernack
January 2026

A trauma-aware AI support framework with:
- OpenAI API integration for real contextual responses
- PreRAG filters (QueryGate, EvidenceGate, AbsurdityGap)
- RAG retrieval system
- Entropy-based state detection
- Pattern recognition for harmful dynamics
- RIME memory continuity
- Grounding techniques library
- Content moderation and absurdity detection

IMPORTANT DISCLAIMER: This is NOT a clinical or treatment tool.
If you are in crisis, please call 988 (US) or your local crisis line.
"""

import os
import sys
import json
import time
import uuid
import math
import random
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

# Suppress HTTP request logging
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class ReUnityConfig:
    """Configuration for ReUnity AI Model."""
    
    # API Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = ""
    
    # Entropy Thresholds
    crisis_threshold: float = 0.85
    high_threshold: float = 0.65
    moderate_threshold: float = 0.45
    low_threshold: float = 0.25
    
    # PreRAG Configuration
    enable_prerag: bool = True
    retrieve_threshold: float = 0.4
    clarify_threshold: float = 0.7
    
    # RAG Configuration
    enable_rag: bool = True
    top_k_chunks: int = 5
    
    # Memory Configuration
    enable_memory: bool = True
    max_memories: int = 100
    
    # Content Moderation
    enable_content_moderation: bool = True
    
    # Web Server Configuration
    web_host: str = "0.0.0.0"
    web_port: int = 5000
    
    # Debug (hide backend output in production)
    show_debug: bool = False


# ============================================================================
# SECTION 2: ENTROPY STATES AND ANALYSIS
# ============================================================================

class EntropyState(Enum):
    """Emotional entropy states."""
    CRISIS = "crisis"
    HIGH_ENTROPY = "high_entropy"
    MODERATE = "moderate"
    LOW_ENTROPY = "low_entropy"
    STABLE = "stable"


@dataclass
class EntropyAnalysis:
    """Result of entropy analysis."""
    state: EntropyState
    entropy_value: float
    confidence: float
    dominant_emotions: Dict[str, float]
    crisis_indicators: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class EntropyAnalyzer:
    """Analyzes text for emotional entropy and crisis indicators."""
    
    CRISIS_KEYWORDS = [
        "suicidal", "suicide", "kill myself", "end my life", "want to die",
        "don't want to live", "better off dead", "no reason to live",
        "dissociating", "dissociation", "disconnected from myself",
        "not real", "nothing is real", "losing my mind",
        "panic attack", "can't breathe", "heart racing",
        "flashback", "triggered", "ptsd",
        "self harm", "cutting", "hurting myself",
        "overdose", "pills", "jump off",
        "abuse", "abusing me", "hitting me", "hurting me",
        "trapped", "no way out", "hopeless",
        "voices", "hearing things", "seeing things",
    ]
    
    HIGH_DISTRESS_KEYWORDS = [
        "scared", "terrified", "afraid", "fear", "frightened",
        "anxious", "anxiety", "panic", "worried", "nervous",
        "angry", "furious", "rage", "hate", "livid",
        "depressed", "depression", "sad", "crying", "tears",
        "lonely", "alone", "isolated", "abandoned",
        "overwhelmed", "can't cope", "falling apart",
        "numb", "empty", "hollow", "dead inside",
        "ashamed", "shame", "guilty", "worthless",
        "confused", "lost", "don't know what to do",
    ]
    
    STABLE_KEYWORDS = [
        "okay", "fine", "good", "great", "happy", "calm",
        "peaceful", "relaxed", "content", "hopeful",
        "better", "improving", "progress", "grateful",
        "safe", "secure", "supported", "loved",
    ]
    
    def __init__(self, config: ReUnityConfig):
        self.config = config
        self.current_state: Optional[EntropyState] = None
        self.state_history: List[EntropyAnalysis] = []
    
    def analyze(self, text: str) -> EntropyAnalysis:
        """Analyze text for emotional entropy."""
        text_lower = text.lower()
        
        crisis_indicators = []
        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                crisis_indicators.append(keyword)
        
        if crisis_indicators:
            analysis = EntropyAnalysis(
                state=EntropyState.CRISIS,
                entropy_value=0.95,
                confidence=0.95,
                dominant_emotions={"crisis": 1.0},
                crisis_indicators=crisis_indicators,
            )
            self.current_state = EntropyState.CRISIS
            self.state_history.append(analysis)
            return analysis
        
        high_count = sum(1 for kw in self.HIGH_DISTRESS_KEYWORDS if kw in text_lower)
        stable_count = sum(1 for kw in self.STABLE_KEYWORDS if kw in text_lower)
        total_keywords = high_count + stable_count + 1
        
        if high_count > stable_count:
            ratio = high_count / total_keywords
            if ratio > 0.6:
                state = EntropyState.HIGH_ENTROPY
                entropy = 0.7 + (ratio * 0.2)
            else:
                state = EntropyState.MODERATE
                entropy = 0.45 + (ratio * 0.2)
        elif stable_count > high_count:
            ratio = stable_count / total_keywords
            if ratio > 0.6:
                state = EntropyState.STABLE
                entropy = 0.1 + (ratio * 0.1)
            else:
                state = EntropyState.LOW_ENTROPY
                entropy = 0.25 + (ratio * 0.1)
        else:
            state = EntropyState.MODERATE
            entropy = 0.5
        
        emotions = {}
        if high_count > 0:
            emotions["distress"] = high_count / total_keywords
        if stable_count > 0:
            emotions["stability"] = stable_count / total_keywords
        if not emotions:
            emotions["neutral"] = 1.0
        
        analysis = EntropyAnalysis(
            state=state,
            entropy_value=entropy,
            confidence=0.8,
            dominant_emotions=emotions,
            crisis_indicators=[],
        )
        
        self.current_state = state
        self.state_history.append(analysis)
        return analysis


# ============================================================================
# SECTION 3: PATTERN RECOGNITION
# ============================================================================

class PatternType(Enum):
    """Types of harmful patterns."""
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    ISOLATION = "isolation"
    HOT_COLD = "hot_cold"
    BLAME_SHIFTING = "blame_shifting"
    TRIANGULATION = "triangulation"
    STONEWALLING = "stonewalling"
    FINANCIAL_CONTROL = "financial_control"


@dataclass
class PatternDetection:
    """Result of pattern detection."""
    pattern_type: PatternType
    confidence: float
    indicators: List[str]
    recommendation: str


class PatternRecognizer:
    """Recognizes harmful relationship patterns."""
    
    PATTERNS = {
        PatternType.GASLIGHTING: {
            "indicators": [
                "that never happened", "never happened", "it never happened",
                "you're imagining things", "imagining things", "imagining it",
                "you're crazy", "you're too sensitive", "overreacting",
                "you're remembering wrong", "that's not what happened",
                "no one else thinks that", "you're confused",
                "making it up", "didn't happen", "never said that",
            ],
            "recommendation": "Trust your own perceptions and memories. Consider documenting events as they happen. This pattern can make you doubt yourself, but your experiences are valid.",
        },
        PatternType.LOVE_BOMBING: {
            "indicators": [
                "you're perfect", "never felt this way", "soulmates",
                "meant to be", "love at first sight", "can't live without you",
                "you're the only one", "no one understands me like you",
            ],
            "recommendation": "Healthy relationships develop gradually. Intense early affection can be a way to create dependency. Take time to build trust slowly.",
        },
        PatternType.ISOLATION: {
            "indicators": [
                "you only need me", "they're jealous", "they don't understand",
                "spend all time together", "don't need friends",
                "your family is toxic", "they're trying to break us up",
            ],
            "recommendation": "Healthy relationships encourage connections with friends and family. Isolation is a warning sign. Maintain your support network.",
        },
        PatternType.BLAME_SHIFTING: {
            "indicators": [
                "your fault", "you made me", "because of you",
                "if you hadn't", "look what you made me do",
                "you're the reason", "this is on you",
            ],
            "recommendation": "Each person is responsible for their own actions. Blame shifting is a way to avoid accountability. You are not responsible for someone else's behavior.",
        },
    }
    
    def __init__(self):
        self.detection_history: List[PatternDetection] = []
    
    def detect(self, text: str) -> List[PatternDetection]:
        """Detect harmful patterns in text."""
        text_lower = text.lower()
        detected = []
        
        for pattern_type, pattern_data in self.PATTERNS.items():
            found_indicators = []
            for indicator in pattern_data["indicators"]:
                if indicator in text_lower:
                    found_indicators.append(indicator)
            
            if found_indicators:
                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=min(0.9, 0.3 + (len(found_indicators) * 0.2)),
                    indicators=found_indicators,
                    recommendation=pattern_data["recommendation"],
                )
                detected.append(detection)
                self.detection_history.append(detection)
        
        return detected


# ============================================================================
# SECTION 4: CONTENT MODERATION AND ABSURDITY DETECTION
# ============================================================================

class ContentModerator:
    """Moderates content for inappropriate or absurd requests."""
    
    # Sexual content indicators
    SEXUAL_KEYWORDS = [
        "masturbat", "sex", "orgasm", "erotic", "porn", "nude",
        "genital", "penis", "vagina", "breast", "nipple",
        "horny", "aroused", "turn me on", "sexual",
    ]
    
    # Absurdity indicators (testing/trolling patterns)
    ABSURDITY_PATTERNS = [
        # Repetitive nonsense
        r"banana.{0,20}peel",
        r"(slip|slipp).{0,10}(banana|peel)",
        # Nonsense keyboard patterns
        r"asdf", r"qwerty", r"aaaaa+", r"12345",
        # Testing patterns
        r"test.{0,5}test", r"blah.{0,5}blah",
    ]
    
    # Escalation tracking
    ESCALATION_KEYWORDS = [
        "everywhere", "can't stop", "ruining my life",
        "all i think about", "obsessed", "can't escape",
    ]
    
    def __init__(self):
        self.absurdity_count = 0
        self.sexual_count = 0
        self.session_topics: List[str] = []
    
    def check_content(self, text: str) -> Dict[str, Any]:
        """Check content for moderation issues."""
        text_lower = text.lower()
        
        result = {
            "is_sexual": False,
            "is_absurd": False,
            "is_escalating_absurdity": False,
            "should_redirect": False,
            "redirect_message": None,
        }
        
        # Check for sexual content
        for keyword in self.SEXUAL_KEYWORDS:
            if keyword in text_lower:
                result["is_sexual"] = True
                self.sexual_count += 1
                break
        
        # Check for absurdity patterns
        for pattern in self.ABSURDITY_PATTERNS:
            if re.search(pattern, text_lower):
                result["is_absurd"] = True
                self.absurdity_count += 1
                break
        
        # Check for escalating absurdity (same nonsense topic repeated)
        if result["is_absurd"]:
            # Extract the absurd topic
            banana_match = re.search(r"banana", text_lower)
            if banana_match:
                self.session_topics.append("banana")
                if self.session_topics.count("banana") >= 3:
                    result["is_escalating_absurdity"] = True
        
        # Determine if we should redirect
        if result["is_sexual"] and self.sexual_count >= 2:
            result["should_redirect"] = True
            result["redirect_message"] = (
                "I notice our conversation has moved in a direction that I'm not able to "
                "support you with. I'm here to help with emotional wellbeing, trauma support, "
                "and coping strategies. Is there something else on your mind that I can help with?"
            )
        
        if result["is_escalating_absurdity"]:
            result["should_redirect"] = True
            result["redirect_message"] = (
                "I want to make sure I'm being genuinely helpful to you. It seems like we might "
                "be going in circles with this topic. If you're testing the system, that's okay. "
                "But if there's something real you're struggling with, I'm here to listen. "
                "What's actually going on for you today?"
            )
        
        return result


@dataclass
class AbsurdityGapMetrics:
    """Metrics from absurdity gap calculation."""
    gap_score: float
    confidence: float
    method: str
    is_testing: bool = False


class AbsurdityGapCalculator:
    """Calculates the absurdity gap for queries."""
    
    def __init__(self):
        self.query_history: List[str] = []
    
    def calculate_gap(self, query: str) -> AbsurdityGapMetrics:
        """Calculate absurdity gap for a query."""
        query_lower = query.lower()
        self.query_history.append(query_lower)
        
        # Check for obvious testing/trolling
        testing_indicators = [
            "banana peel", "slip on banana", "bananas everywhere",
            "asdfgh", "qwerty", "test test", "blah blah",
        ]
        
        for indicator in testing_indicators:
            if indicator in query_lower:
                return AbsurdityGapMetrics(
                    gap_score=0.9,
                    confidence=0.9,
                    method="testing_detection",
                    is_testing=True,
                )
        
        # Check for repetitive patterns in history
        if len(self.query_history) >= 3:
            recent = self.query_history[-3:]
            # Check if same unusual topic repeated
            banana_count = sum(1 for q in recent if "banana" in q)
            if banana_count >= 2:
                return AbsurdityGapMetrics(
                    gap_score=0.85,
                    confidence=0.85,
                    method="repetition_detection",
                    is_testing=True,
                )
        
        # Check query coherence
        words = query.split()
        if len(words) < 2:
            return AbsurdityGapMetrics(
                gap_score=0.5,
                confidence=0.6,
                method="length_check",
            )
        
        return AbsurdityGapMetrics(
            gap_score=0.2,
            confidence=0.8,
            method="default",
        )


# ============================================================================
# SECTION 5: PRERAG FILTERS
# ============================================================================

class QueryGateAction(Enum):
    """Actions the QueryGate can take."""
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    NO_RETRIEVE = "no_retrieve"
    REDIRECT = "redirect"


@dataclass
class QueryGateDecision:
    """Decision from QueryGate."""
    action: QueryGateAction
    normalized_query: str
    absurdity_gap: AbsurdityGapMetrics
    reasoning: str
    redirect_message: Optional[str] = None


class QueryGate:
    """Pre-retrieval gate that validates queries."""
    
    def __init__(
        self,
        retrieve_threshold: float = 0.4,
        clarify_threshold: float = 0.7,
    ):
        self.retrieve_threshold = retrieve_threshold
        self.clarify_threshold = clarify_threshold
        self.absurdity_calculator = AbsurdityGapCalculator()
        self.content_moderator = ContentModerator()
        
        self.blocked_patterns = [
            r"how to (harm|hurt|kill)",
            r"ways to (die|suicide)",
            r"methods of (self.harm|cutting)",
        ]
    
    def process(self, query: str) -> QueryGateDecision:
        """Process a query through the gate."""
        normalized = query.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Check for blocked patterns (self-harm requests)
        for pattern in self.blocked_patterns:
            if re.search(pattern, normalized.lower()):
                return QueryGateDecision(
                    action=QueryGateAction.NO_RETRIEVE,
                    normalized_query=normalized,
                    absurdity_gap=AbsurdityGapMetrics(1.0, 1.0, "blocked"),
                    reasoning="crisis_redirect",
                    redirect_message=(
                        "I'm concerned about what you've shared. If you're thinking about "
                        "hurting yourself, please reach out for support right now:\n\n"
                        "• National Suicide Prevention Lifeline: 988 (US)\n"
                        "• Crisis Text Line: Text HOME to 741741\n"
                        "• International: https://www.iasp.info/resources/Crisis_Centres/\n\n"
                        "You matter, and there are people who want to help."
                    ),
                )
        
        # Check content moderation
        moderation = self.content_moderator.check_content(normalized)
        if moderation["should_redirect"]:
            return QueryGateDecision(
                action=QueryGateAction.REDIRECT,
                normalized_query=normalized,
                absurdity_gap=AbsurdityGapMetrics(0.8, 0.9, "moderation"),
                reasoning="content_moderation",
                redirect_message=moderation["redirect_message"],
            )
        
        # Calculate absurdity gap
        gap = self.absurdity_calculator.calculate_gap(normalized)
        
        if gap.is_testing:
            return QueryGateDecision(
                action=QueryGateAction.REDIRECT,
                normalized_query=normalized,
                absurdity_gap=gap,
                reasoning="absurdity_detected",
                redirect_message=(
                    "I want to make sure I'm being genuinely helpful. It seems like we might "
                    "be exploring some unusual territory. If you're testing how I respond, "
                    "that's okay. But if there's something real you're dealing with, I'm here "
                    "to listen without judgment. What's actually on your mind?"
                ),
            )
        
        # Decide action
        if gap.gap_score < self.retrieve_threshold:
            action = QueryGateAction.RETRIEVE
            reasoning = "valid_query"
        elif gap.gap_score > self.clarify_threshold:
            action = QueryGateAction.CLARIFY
            reasoning = "needs_clarification"
        else:
            action = QueryGateAction.RETRIEVE
            reasoning = "acceptable_query"
        
        return QueryGateDecision(
            action=action,
            normalized_query=normalized,
            absurdity_gap=gap,
            reasoning=reasoning,
        )


class EvidenceGateAction(Enum):
    """Actions the EvidenceGate can take."""
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"


@dataclass
class EvidenceGateDecision:
    """Decision from EvidenceGate."""
    action: EvidenceGateAction
    selected_chunks: List[str]
    confidence: float
    reasoning: str


class EvidenceGate:
    """Post-retrieval gate that validates evidence."""
    
    def __init__(
        self,
        answer_threshold: float = 0.3,
        refuse_threshold: float = 0.85,
    ):
        self.answer_threshold = answer_threshold
        self.refuse_threshold = refuse_threshold
    
    def process(
        self,
        query: str,
        chunks: List[str],
    ) -> EvidenceGateDecision:
        """Process retrieved evidence."""
        if not chunks:
            return EvidenceGateDecision(
                action=EvidenceGateAction.REFUSE,
                selected_chunks=[],
                confidence=0.9,
                reasoning="no_relevant_information",
            )
        
        query_words = set(query.lower().split())
        relevant_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                relevant_chunks.append(chunk)
        
        if relevant_chunks:
            return EvidenceGateDecision(
                action=EvidenceGateAction.ANSWER,
                selected_chunks=relevant_chunks[:5],
                confidence=0.8,
                reasoning="found_relevant_information",
            )
        else:
            return EvidenceGateDecision(
                action=EvidenceGateAction.CLARIFY,
                selected_chunks=[],
                confidence=0.6,
                reasoning="no_direct_match",
            )


# ============================================================================
# SECTION 6: RAG SYSTEM
# ============================================================================

@dataclass
class Chunk:
    """A chunk of text for RAG."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGSystem:
    """Simple RAG system for knowledge retrieval."""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize with trauma-informed knowledge."""
        knowledge = [
            "The 5-4-3-2-1 grounding technique helps with dissociation and anxiety. Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
            "Box breathing is a calming technique: breathe in for 4 counts, hold for 4 counts, breathe out for 4 counts, hold for 4 counts. Repeat 4 times.",
            "Progressive muscle relaxation involves tensing and releasing muscle groups from your toes to your head, helping release physical tension from stress.",
            "The TIPP skill from DBT helps in crisis: Temperature (cold water on face), Intense exercise, Paced breathing, Paired muscle relaxation.",
            "Dissociation is a protective response to overwhelming stress. It is not dangerous, though it can feel scary. Grounding techniques can help you return to the present moment.",
            "Flashbacks are memories that feel like they are happening now. You are safe in the present moment. Try naming where you are and what year it is.",
            "Panic attacks, while terrifying, are not dangerous. They typically peak within 10 minutes. Focus on slow breathing and remember it will pass.",
            "Gaslighting is a form of emotional abuse where someone makes you question your own reality. Trust your perceptions and document events if helpful.",
            "Love bombing is excessive attention and affection early in a relationship, often used to create dependency. Healthy relationships develop gradually.",
            "The cycle of abuse often includes tension building, an incident, reconciliation (honeymoon phase), and calm before the cycle repeats.",
            "Self-compassion means treating yourself with the same kindness you would offer a friend. You deserve care and understanding.",
            "Setting boundaries is healthy and necessary. You have the right to say no and to protect your wellbeing.",
            "Healing is not linear. Having difficult days does not mean you are going backward. Each moment is a new opportunity.",
            "If you are in immediate danger, please call 911 or your local emergency number.",
            "The National Suicide Prevention Lifeline is available 24/7 at 988 in the US.",
            "The Crisis Text Line is available by texting HOME to 741741 in the US.",
            "The National Domestic Violence Hotline is 1-800-799-7233.",
        ]
        
        for i, text in enumerate(knowledge):
            self.chunks.append(Chunk(
                id=f"chunk_{i}",
                text=text,
                metadata={"source": "knowledge_base"},
            ))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Retrieve relevant chunks for a query."""
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(chunk.text.lower().split())
            score = len(query_words & chunk_words)
            if score > 0:
                scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:top_k]]


# ============================================================================
# SECTION 7: MEMORY SYSTEM (RIME)
# ============================================================================

@dataclass
class Memory:
    """A memory entry."""
    id: str
    content: str
    emotional_context: EntropyState
    timestamp: datetime
    importance: float
    tags: List[str] = field(default_factory=list)


class MemoryStore:
    """RIME: Recursive Identity Memory Engine."""
    
    def __init__(self, max_memories: int = 100):
        self.memories: List[Memory] = []
        self.max_memories = max_memories
    
    def store(
        self,
        content: str,
        emotional_context: EntropyState,
        importance: float = 0.5,
        tags: List[str] = None,
    ) -> Memory:
        """Store a new memory."""
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            emotional_context=emotional_context,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or [],
        )
        
        self.memories.append(memory)
        
        if len(self.memories) > self.max_memories:
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[:self.max_memories]
        
        return memory
    
    def retrieve(
        self,
        query: str = None,
        emotional_context: EntropyState = None,
        limit: int = 5,
    ) -> List[Memory]:
        """Retrieve relevant memories."""
        results = self.memories.copy()
        
        if emotional_context:
            results = [m for m in results if m.emotional_context == emotional_context]
        
        if query:
            query_words = set(query.lower().split())
            scored = []
            for memory in results:
                memory_words = set(memory.content.lower().split())
                score = len(query_words & memory_words)
                if score > 0:
                    scored.append((memory, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            results = [m for m, _ in scored]
        
        results.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return results[:limit]


# ============================================================================
# SECTION 8: GROUNDING TECHNIQUES
# ============================================================================

@dataclass
class GroundingTechnique:
    """A grounding technique."""
    name: str
    description: str
    steps: List[str]
    duration_minutes: int
    suitable_for: List[EntropyState]


class GroundingLibrary:
    """Library of grounding techniques."""
    
    TECHNIQUES = [
        GroundingTechnique(
            name="5-4-3-2-1 Sensory Grounding",
            description="Use your senses to anchor yourself in the present moment.",
            steps=[
                "Name 5 things you can SEE around you right now.",
                "Name 4 things you can TOUCH or feel.",
                "Name 3 things you can HEAR.",
                "Name 2 things you can SMELL.",
                "Name 1 thing you can TASTE.",
            ],
            duration_minutes=5,
            suitable_for=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY],
        ),
        GroundingTechnique(
            name="Box Breathing",
            description="A calming breathing pattern used by Navy SEALs.",
            steps=[
                "Breathe IN slowly for 4 counts.",
                "HOLD your breath for 4 counts.",
                "Breathe OUT slowly for 4 counts.",
                "HOLD empty for 4 counts.",
                "Repeat this cycle 4 times.",
            ],
            duration_minutes=3,
            suitable_for=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY, EntropyState.MODERATE],
        ),
        GroundingTechnique(
            name="Cold Water Reset",
            description="Use cold temperature to activate your dive reflex and calm your nervous system.",
            steps=[
                "Get a bowl of cold water or ice cubes.",
                "Hold your breath and put your face in the cold water for 30 seconds.",
                "Or hold ice cubes in your hands.",
                "Focus on the cold sensation.",
                "This activates your parasympathetic nervous system.",
            ],
            duration_minutes=2,
            suitable_for=[EntropyState.CRISIS],
        ),
        GroundingTechnique(
            name="Body Scan",
            description="Reconnect with your physical body.",
            steps=[
                "Close your eyes if comfortable.",
                "Notice your feet on the ground.",
                "Feel the weight of your body in your seat.",
                "Notice your hands. Wiggle your fingers.",
                "Take three slow breaths.",
                "Open your eyes when ready.",
            ],
            duration_minutes=3,
            suitable_for=[EntropyState.HIGH_ENTROPY, EntropyState.MODERATE],
        ),
    ]
    
    def get_technique(self, state: EntropyState) -> GroundingTechnique:
        """Get appropriate technique for state."""
        suitable = [t for t in self.TECHNIQUES if state in t.suitable_for]
        if suitable:
            return random.choice(suitable)
        return self.TECHNIQUES[0]
    
    def format_technique(self, technique: GroundingTechnique) -> str:
        """Format technique for display."""
        lines = [
            f"**{technique.name}**",
            f"_{technique.description}_",
            "",
        ]
        for i, step in enumerate(technique.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append(f"\n(Takes about {technique.duration_minutes} minutes)")
        return "\n".join(lines)


# ============================================================================
# SECTION 9: LLM PROVIDERS
# ============================================================================

class OpenAIProvider:
    """OpenAI API provider."""
    
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
            return f"I'm having trouble connecting right now. Please try again. (Error: {str(e)[:50]})"


class FallbackProvider:
    """Fallback provider when no API is available."""
    
    RESPONSES = {
        EntropyState.CRISIS: [
            "I hear that you're going through something really difficult right now. Your safety matters. Please reach out to a crisis line: 988 (US) or text HOME to 741741.",
            "What you're experiencing sounds overwhelming. You don't have to face this alone. Please consider calling 988 or texting HOME to 741741 for immediate support.",
        ],
        EntropyState.HIGH_ENTROPY: [
            "It sounds like you're dealing with a lot right now. Would you like to try a grounding technique together?",
            "I can hear how distressed you're feeling. Let's take a moment to breathe together.",
        ],
        EntropyState.MODERATE: [
            "Thank you for sharing that with me. Can you tell me more about what's going on?",
            "I'm here to listen. What feels most important to talk about right now?",
        ],
        EntropyState.STABLE: [
            "It sounds like you're in a good place right now. Is there anything you'd like to explore or work on?",
            "I'm glad to hear things are going okay. What's on your mind?",
        ],
    }
    
    def generate(self, messages: List[Dict[str, str]], entropy_state: EntropyState = None) -> str:
        """Generate fallback response."""
        state = entropy_state or EntropyState.MODERATE
        responses = self.RESPONSES.get(state, self.RESPONSES[EntropyState.MODERATE])
        return random.choice(responses)


# ============================================================================
# SECTION 10: MAIN REUNITY CLASS
# ============================================================================

class ReUnity:
    """Main ReUnity AI Model class."""
    
    SYSTEM_PROMPT = """You are ReUnity, a trauma-informed AI support companion created by Christopher Ezernack.

Your role is to provide empathetic, validating support while maintaining appropriate boundaries. You are NOT a therapist or crisis counselor, but you can offer:
- Emotional validation and active listening
- Grounding techniques and coping strategies
- Recognition of harmful relationship patterns
- Gentle psychoeducation about trauma responses
- Encouragement to seek professional support when appropriate

CRITICAL GUIDELINES:
1. ALWAYS validate feelings before offering suggestions
2. NEVER minimize or dismiss someone's experience
3. Use warm, compassionate language without being patronizing
4. Recognize signs of crisis and provide crisis resources (988, Crisis Text Line)
5. Acknowledge harmful patterns (gaslighting, love bombing, etc.) when detected
6. Offer grounding techniques when someone is dysregulated
7. Be honest about your limitations as an AI

CONTENT BOUNDARIES:
- Do NOT engage with sexual content or validate sexual behaviors as coping mechanisms
- Do NOT reinforce obsessive or irrational fears without gently reality-checking
- If someone seems to be testing the system with absurd scenarios, gently redirect to genuine support
- If a topic seems disconnected from real distress, ask what's really going on

REALITY CHECKING:
- If someone describes an irrational fear (like being afraid of banana peels everywhere), acknowledge their distress but gently explore what might really be underneath
- Help them distinguish between genuine anxiety and avoidance patterns
- Encourage professional support for persistent irrational fears

CRISIS PROTOCOL:
If someone expresses suicidal ideation, self-harm, or immediate danger:
1. Express care and concern
2. Provide crisis resources: 988 (US), Crisis Text Line (text HOME to 741741)
3. Encourage them to reach out to a trusted person or professional
4. Stay with them in the conversation

CURRENT USER STATE: {entropy_state}
DETECTED PATTERNS: {patterns}
RELEVANT CONTEXT: {context}

Respond with empathy, validation, and appropriate support for their current state. Keep responses focused and helpful."""

    def __init__(self, config: ReUnityConfig = None):
        self.config = config or ReUnityConfig()
        
        self.entropy_analyzer = EntropyAnalyzer(self.config)
        self.pattern_recognizer = PatternRecognizer()
        self.query_gate = QueryGate(
            retrieve_threshold=self.config.retrieve_threshold,
            clarify_threshold=self.config.clarify_threshold,
        )
        self.evidence_gate = EvidenceGate()
        self.rag_system = RAGSystem()
        self.memory_store = MemoryStore(max_memories=self.config.max_memories)
        self.grounding_library = GroundingLibrary()
        
        if self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"):
            self.llm = OpenAIProvider(self.config)
        else:
            self.llm = FallbackProvider()
        
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        self.conversation_history: List[Dict[str, str]] = []
        self.interaction_count = 0
    
    def _print_disclaimer(self):
        """Print the important disclaimer."""
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
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the full pipeline."""
        self.interaction_count += 1
        
        # Step 1: PreRAG Query Gate (includes content moderation and absurdity detection)
        if self.config.enable_prerag:
            gate_decision = self.query_gate.process(user_input)
            
            if gate_decision.action == QueryGateAction.NO_RETRIEVE:
                return {
                    "response": gate_decision.redirect_message or gate_decision.reasoning,
                    "analysis": None,
                    "patterns_detected": [],
                    "grounding_recommendation": None,
                    "blocked": True,
                }
            
            if gate_decision.action == QueryGateAction.REDIRECT:
                return {
                    "response": gate_decision.redirect_message,
                    "analysis": None,
                    "patterns_detected": [],
                    "grounding_recommendation": None,
                    "blocked": False,
                    "redirected": True,
                }
        
        # Step 2: Entropy Analysis
        analysis = self.entropy_analyzer.analyze(user_input)
        
        # Step 3: Pattern Recognition
        patterns = self.pattern_recognizer.detect(user_input)
        
        # Step 4: RAG Retrieval
        context_chunks = []
        if self.config.enable_rag:
            retrieved = self.rag_system.retrieve(user_input, top_k=self.config.top_k_chunks)
            context_chunks = [chunk.text for chunk in retrieved]
            
            if self.config.enable_prerag:
                evidence_decision = self.evidence_gate.process(user_input, context_chunks)
                if evidence_decision.action == EvidenceGateAction.ANSWER:
                    context_chunks = evidence_decision.selected_chunks
        
        # Step 5: Memory Retrieval
        memories = []
        if self.config.enable_memory:
            memories = self.memory_store.retrieve(
                query=user_input,
                emotional_context=analysis.state,
                limit=3,
            )
        
        # Step 6: Build context for LLM
        context_text = "\n".join(context_chunks) if context_chunks else "No specific context retrieved."
        pattern_text = ", ".join([p.pattern_type.value for p in patterns]) if patterns else "None detected"
        
        # Step 7: Generate response
        system_prompt = self.SYSTEM_PROMPT.format(
            entropy_state=analysis.state.value,
            patterns=pattern_text,
            context=context_text,
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_input})
        
        if isinstance(self.llm, FallbackProvider):
            response = self.llm.generate(messages, entropy_state=analysis.state)
        else:
            response = self.llm.generate(messages)
        
        # Step 8: Add pattern warnings if detected
        if patterns:
            pattern_warnings = []
            for p in patterns:
                pattern_warnings.append(f"\n\n**I noticed something important:** {p.recommendation}")
            response = response + "".join(pattern_warnings)
        
        # Step 9: Add grounding technique for high entropy states
        grounding_recommendation = None
        if analysis.state in [EntropyState.CRISIS, EntropyState.HIGH_ENTROPY]:
            technique = self.grounding_library.get_technique(analysis.state)
            grounding_recommendation = {
                "technique": technique.name,
                "formatted": self.grounding_library.format_technique(technique),
            }
            response = response + f"\n\n---\n\n**Would you like to try a grounding technique?**\n\n{grounding_recommendation['formatted']}"
        
        # Step 10: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Step 11: Store memory
        if self.config.enable_memory:
            importance = 0.8 if analysis.state == EntropyState.CRISIS else 0.5
            self.memory_store.store(
                content=user_input,
                emotional_context=analysis.state,
                importance=importance,
            )
        
        return {
            "response": response,
            "analysis": {
                "state": analysis.state.value,
                "entropy": analysis.entropy_value,
                "confidence": analysis.confidence,
                "crisis_indicators": analysis.crisis_indicators,
            },
            "patterns_detected": [
                {
                    "pattern_type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "recommendation": p.recommendation,
                }
                for p in patterns
            ],
            "grounding_recommendation": grounding_recommendation,
            "blocked": False,
        }
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            "session_id": self.session_id,
            "duration_minutes": (time.time() - self.session_start) / 60,
            "interactions": self.interaction_count,
            "current_state": self.entropy_analyzer.current_state.value if self.entropy_analyzer.current_state else "unknown",
            "memories_stored": len(self.memory_store.memories),
            "patterns_detected": len(self.pattern_recognizer.detection_history),
        }


# ============================================================================
# SECTION 11: WEB SERVER
# ============================================================================

def create_web_app(reunity: ReUnity = None):
    """Create Flask web application."""
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return None
    
    app = Flask(__name__)
    
    if reunity is None:
        reunity = ReUnity()
    
    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>ReUnity AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 600px;
            height: 80vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 20px 20px 0 0;
            text-align: center;
        }
        .header h1 { font-size: 24px; margin-bottom: 5px; }
        .header p { font-size: 12px; opacity: 0.8; }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .message {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        .user {
            background: #667eea;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        .assistant {
            background: #f0f0f0;
            color: #333;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }
        .input-area {
            padding: 20px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
        }
        .input-area input:focus { border-color: #667eea; }
        .input-area button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
        }
        .input-area button:hover { opacity: 0.9; }
        .disclaimer {
            font-size: 11px;
            color: #666;
            text-align: center;
            padding: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ReUnity</h1>
            <p>Trauma-Aware AI Support</p>
        </div>
        <div class="messages" id="messages">
            <div class="message assistant">Hello. I'm here to listen and support you. How are you feeling today?</div>
        </div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
        <div class="disclaimer">
            Not a clinical tool. If in crisis, call 988 (US) or text HOME to 741741.
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('input');
            const messages = document.getElementById('messages');
            const text = input.value.trim();
            if (!text) return;
            
            messages.innerHTML += `<div class="message user">${text}</div>`;
            input.value = '';
            messages.scrollTop = messages.scrollHeight;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await response.json();
                messages.innerHTML += `<div class="message assistant">${data.response}</div>`;
            } catch (e) {
                messages.innerHTML += `<div class="message assistant">Sorry, something went wrong. Please try again.</div>`;
            }
            messages.scrollTop = messages.scrollHeight;
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
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        result = reunity.process_input(message)
        
        return jsonify({
            "response": result["response"],
        })
    
    @app.route('/api/status', methods=['GET'])
    def status():
        return jsonify(reunity.get_session_summary())
    
    return app


# ============================================================================
# SECTION 12: CLI AND MAIN
# ============================================================================

def run_interactive():
    """Run interactive CLI mode."""
    print("\n" + "="*60)
    print("ReUnity AI Model v3.1.0")
    print("A Trauma-Aware AI Support Framework")
    print("By Christopher Ezernack, REOP Solutions")
    print("="*60)
    print("\nType your message and press Enter.")
    print("Type /quit to exit, /status for session info.\n")
    
    config = ReUnityConfig(show_debug=False)
    reunity = ReUnity(config)
    reunity._print_disclaimer()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/quit':
                print("\nThank you for using ReUnity. Take care of yourself.")
                break
            
            if user_input.lower() == '/status':
                summary = reunity.get_session_summary()
                print(f"\nSession: {summary['session_id'][:8]}...")
                print(f"Duration: {summary['duration_minutes']:.1f} minutes")
                print(f"Interactions: {summary['interactions']}")
                continue
            
            result = reunity.process_input(user_input)
            
            # Only show the response, no backend info
            print(f"\nReUnity: {result['response']}")
        
        except KeyboardInterrupt:
            print("\n\nSession ended. Take care.")
            break
        except Exception as e:
            print(f"\nI'm having trouble processing that. Please try again.")
            continue


def run_test():
    """Run basic tests."""
    print("\n" + "="*60)
    print("ReUnity AI Model v3.1.0 - Test Mode")
    print("="*60 + "\n")
    
    config = ReUnityConfig(show_debug=True)
    reunity = ReUnity(config)
    
    test_cases = [
        ("I'm feeling anxious today", "Should detect HIGH_ENTROPY"),
        ("My partner said I was imagining things again", "Should detect gaslighting pattern"),
        ("I am dissociating right now", "Should detect CRISIS"),
        ("I feel calm and peaceful", "Should detect STABLE"),
        ("I want to kill myself", "Should detect CRISIS with crisis indicators"),
        ("banana peel banana peel banana peel", "Should detect absurdity and redirect"),
        ("I am comfortable when I masturbate", "Should redirect away from sexual content"),
    ]
    
    for text, expected in test_cases:
        print(f"\nInput: {text}")
        print(f"Expected: {expected}")
        
        result = reunity.process_input(text)
        
        if result.get('redirected'):
            print("Result: REDIRECTED")
            print(f"Message: {result['response'][:100]}...")
        elif result.get('blocked'):
            print("Result: BLOCKED")
        elif result['analysis']:
            print(f"State: {result['analysis']['state']}")
            if result['patterns_detected']:
                patterns = [p['pattern_type'] for p in result['patterns_detected']]
                print(f"Patterns: {', '.join(patterns)}")
            if result['analysis']['crisis_indicators']:
                print(f"Crisis indicators: {result['analysis']['crisis_indicators']}")
        print("-" * 40)
    
    print("\nTest complete.")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ReUnity AI Model")
    parser.add_argument("--web", action="store_true", help="Run web server")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    
    args = parser.parse_args()
    
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    if args.test:
        run_test()
    elif args.web:
        app = create_web_app()
        if app:
            print(f"\nStarting ReUnity web interface on port {args.port}...")
            print(f"Open http://localhost:{args.port} in your browser\n")
            app.run(host="0.0.0.0", port=args.port, debug=False)
        else:
            print("Failed to create web app. Install Flask: pip install flask")
    else:
        run_interactive()


if __name__ == "__main__":
    main()
