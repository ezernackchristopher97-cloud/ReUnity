"""
ReUnity AI Model v3.0.0
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
import logging
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

@dataclass
class ReUnityConfig:
    """Configuration for ReUnity AI Model."""
    
    # API Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = ""  # Leave empty for default
    
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
    
    # Web Server Configuration
    web_host: str = "0.0.0.0"
    web_port: int = 5000
    
    # Debug
    debug: bool = False


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
    
    # Crisis keywords that immediately trigger CRISIS state
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
    
    # High distress keywords
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
    
    # Positive/stable keywords
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
        
        # Check for crisis indicators first
        crisis_indicators = []
        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                crisis_indicators.append(keyword)
        
        # If any crisis indicators, immediately return CRISIS
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
        
        # Count high distress keywords
        high_count = sum(1 for kw in self.HIGH_DISTRESS_KEYWORDS if kw in text_lower)
        
        # Count stable keywords
        stable_count = sum(1 for kw in self.STABLE_KEYWORDS if kw in text_lower)
        
        # Calculate entropy based on keyword balance
        total_keywords = high_count + stable_count + 1  # +1 to avoid division by zero
        
        if high_count > stable_count:
            # More distress than stability
            ratio = high_count / total_keywords
            if ratio > 0.6:
                state = EntropyState.HIGH_ENTROPY
                entropy = 0.7 + (ratio * 0.2)
            else:
                state = EntropyState.MODERATE
                entropy = 0.45 + (ratio * 0.2)
        elif stable_count > high_count:
            # More stability than distress
            ratio = stable_count / total_keywords
            if ratio > 0.6:
                state = EntropyState.STABLE
                entropy = 0.1 + (ratio * 0.1)
            else:
                state = EntropyState.LOW_ENTROPY
                entropy = 0.25 + (ratio * 0.1)
        else:
            # Balanced or neutral
            state = EntropyState.MODERATE
            entropy = 0.5
        
        # Build dominant emotions
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
                "you're wrong", "that's not true", "misremember", "your memory",
            ],
            "recommendation": "Trust your own perceptions and memories. Consider documenting events as they happen. This pattern can make you doubt yourself, but your experiences are valid.",
        },
        PatternType.LOVE_BOMBING: {
            "indicators": [
                "you're perfect", "never felt this way", "soulmates",
                "meant to be", "love at first sight", "can't live without you",
                "you're the only one", "no one understands me like you",
                "we should move in together", "let's get married",
                "constant gifts", "overwhelming attention",
            ],
            "recommendation": "Healthy relationships develop gradually. Intense early affection can be a way to create dependency. Take time to build trust slowly.",
        },
        PatternType.ISOLATION: {
            "indicators": [
                "you only need me", "they're jealous", "they don't understand",
                "spend all time together", "don't need friends",
                "your family is toxic", "they're trying to break us up",
                "why do you need to see them", "I should be enough",
            ],
            "recommendation": "Healthy relationships encourage connections with friends and family. Isolation is a warning sign. Maintain your support network.",
        },
        PatternType.HOT_COLD: {
            "indicators": [
                "sometimes loving sometimes cold", "unpredictable",
                "never know which version", "walking on eggshells",
                "one day sweet next day cruel", "mood swings",
                "hot and cold", "push and pull",
            ],
            "recommendation": "Inconsistent treatment creates anxiety and dependency. You deserve consistent respect and kindness.",
        },
        PatternType.BLAME_SHIFTING: {
            "indicators": [
                "your fault", "you made me", "because of you",
                "if you hadn't", "you always", "you never",
                "look what you made me do", "you drove me to this",
            ],
            "recommendation": "Each person is responsible for their own actions. Being blamed for someone else's behavior is a manipulation tactic.",
        },
    }
    
    def __init__(self):
        self.detection_history: List[PatternDetection] = []
    
    def detect(self, text: str) -> List[PatternDetection]:
        """Detect harmful patterns in text."""
        text_lower = text.lower()
        detections = []
        
        for pattern_type, pattern_info in self.PATTERNS.items():
            found_indicators = []
            for indicator in pattern_info["indicators"]:
                if indicator in text_lower:
                    found_indicators.append(indicator)
            
            if found_indicators:
                confidence = min(len(found_indicators) * 0.3, 0.95)
                detection = PatternDetection(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    indicators=found_indicators,
                    recommendation=pattern_info["recommendation"],
                )
                detections.append(detection)
                self.detection_history.append(detection)
        
        return detections


# ============================================================================
# SECTION 4: PRERAG FILTERS
# ============================================================================

@dataclass
class AbsurdityGapMetrics:
    """Metrics from absurdity gap calculation."""
    gap_score: float
    confidence: float
    method: str


class AbsurdityGapCalculator:
    """Calculates the absurdity gap for queries."""
    
    def __init__(self, anchors: List[str] = None):
        self.anchors = anchors or []
    
    def calculate_gap(self, query: str) -> AbsurdityGapMetrics:
        """Calculate absurdity gap for a query."""
        # Simple heuristic-based calculation
        query_lower = query.lower()
        
        # Check for nonsensical patterns
        nonsense_indicators = [
            "asdfgh", "qwerty", "aaaaa", "12345",
            "blah blah", "whatever whatever",
        ]
        
        for indicator in nonsense_indicators:
            if indicator in query_lower:
                return AbsurdityGapMetrics(
                    gap_score=0.9,
                    confidence=0.8,
                    method="nonsense_detection",
                )
        
        # Check query length
        words = query.split()
        if len(words) < 2:
            return AbsurdityGapMetrics(
                gap_score=0.6,
                confidence=0.7,
                method="length_check",
            )
        
        if len(words) > 100:
            return AbsurdityGapMetrics(
                gap_score=0.5,
                confidence=0.6,
                method="length_check",
            )
        
        # Default: reasonable query
        return AbsurdityGapMetrics(
            gap_score=0.2,
            confidence=0.8,
            method="default",
        )


class QueryGateAction(Enum):
    """Actions the QueryGate can take."""
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    NO_RETRIEVE = "no_retrieve"


@dataclass
class QueryGateDecision:
    """Decision from QueryGate."""
    action: QueryGateAction
    normalized_query: str
    absurdity_gap: AbsurdityGapMetrics
    reasoning: str
    suggestions: List[str] = field(default_factory=list)


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
        
        # Blocked patterns (harmful requests)
        self.blocked_patterns = [
            r"how to (harm|hurt|kill)",
            r"ways to (die|suicide)",
            r"methods of (self.harm|cutting)",
        ]
    
    def process(self, query: str) -> QueryGateDecision:
        """Process a query through the gate."""
        # Normalize query
        normalized = query.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, normalized.lower()):
                return QueryGateDecision(
                    action=QueryGateAction.NO_RETRIEVE,
                    normalized_query=normalized,
                    absurdity_gap=AbsurdityGapMetrics(1.0, 1.0, "blocked"),
                    reasoning="This query cannot be processed. If you are in crisis, please call 988.",
                    suggestions=["Please reach out to a crisis line: 988 (US)"],
                )
        
        # Calculate absurdity gap
        gap = self.absurdity_calculator.calculate_gap(normalized)
        
        # Decide action
        if gap.gap_score < self.retrieve_threshold:
            action = QueryGateAction.RETRIEVE
            reasoning = "Query is valid, proceeding"
        elif gap.gap_score > self.clarify_threshold:
            action = QueryGateAction.CLARIFY
            reasoning = "Query needs clarification"
        else:
            action = QueryGateAction.RETRIEVE
            reasoning = "Query is acceptable"
        
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
                reasoning="No relevant information found",
            )
        
        # Simple relevance check
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
                reasoning="Found relevant information",
            )
        else:
            return EvidenceGateDecision(
                action=EvidenceGateAction.CLARIFY,
                selected_chunks=[],
                confidence=0.6,
                reasoning="Could not find directly relevant information",
            )


# ============================================================================
# SECTION 5: RAG SYSTEM
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
            # Grounding techniques
            "The 5-4-3-2-1 grounding technique helps with dissociation and anxiety. Name 5 things you can see, 4 things you can touch, 3 things you can hear, 2 things you can smell, and 1 thing you can taste.",
            "Box breathing is a calming technique: breathe in for 4 counts, hold for 4 counts, breathe out for 4 counts, hold for 4 counts. Repeat 4 times.",
            "Progressive muscle relaxation involves tensing and releasing muscle groups from your toes to your head, helping release physical tension from stress.",
            "The TIPP skill from DBT helps in crisis: Temperature (cold water on face), Intense exercise, Paced breathing, Paired muscle relaxation.",
            
            # Trauma-informed responses
            "Dissociation is a protective response to overwhelming stress. It is not dangerous, though it can feel scary. Grounding techniques can help you return to the present moment.",
            "Flashbacks are memories that feel like they are happening now. You are safe in the present moment. Try naming where you are and what year it is.",
            "Panic attacks, while terrifying, are not dangerous. They typically peak within 10 minutes. Focus on slow breathing and remember it will pass.",
            
            # Relationship patterns
            "Gaslighting is a form of emotional abuse where someone makes you question your own reality. Trust your perceptions and document events if helpful.",
            "Love bombing is excessive attention and affection early in a relationship, often used to create dependency. Healthy relationships develop gradually.",
            "The cycle of abuse often includes tension building, an incident, reconciliation (honeymoon phase), and calm before the cycle repeats.",
            
            # Self-care
            "Self-compassion means treating yourself with the same kindness you would offer a friend. You deserve care and understanding.",
            "Setting boundaries is healthy and necessary. You have the right to say no and to protect your wellbeing.",
            "Healing is not linear. Having difficult days does not mean you are going backward. Each moment is a new opportunity.",
            
            # Crisis resources
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
    
    def add_chunk(self, text: str, metadata: Dict[str, Any] = None):
        """Add a chunk to the knowledge base."""
        chunk = Chunk(
            id=f"chunk_{len(self.chunks)}",
            text=text,
            metadata=metadata or {},
        )
        self.chunks.append(chunk)


# ============================================================================
# SECTION 6: MEMORY SYSTEM (RIME)
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
        
        # Prune if over limit
        if len(self.memories) > self.max_memories:
            # Remove lowest importance memories
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
        
        # Filter by emotional context if specified
        if emotional_context:
            results = [m for m in results if m.emotional_context == emotional_context]
        
        # Filter by query if specified
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
        
        # Sort by recency and importance
        results.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        return results[:limit]


# ============================================================================
# SECTION 7: GROUNDING TECHNIQUES
# ============================================================================

@dataclass
class GroundingTechnique:
    """A grounding technique."""
    name: str
    description: str
    steps: List[str]
    duration_minutes: int
    suitable_states: List[EntropyState]


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
            suitable_states=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY],
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
            suitable_states=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY, EntropyState.MODERATE],
        ),
        GroundingTechnique(
            name="Cold Water Reset",
            description="Use cold temperature to activate your dive reflex and calm your nervous system.",
            steps=[
                "Get a bowl of cold water or ice water.",
                "Take a deep breath and hold it.",
                "Submerge your face in the cold water for 15-30 seconds.",
                "This activates your parasympathetic nervous system.",
                "Repeat if needed.",
            ],
            duration_minutes=2,
            suitable_states=[EntropyState.CRISIS, EntropyState.HIGH_ENTROPY],
        ),
        GroundingTechnique(
            name="Body Scan",
            description="Systematically notice sensations throughout your body.",
            steps=[
                "Close your eyes or soften your gaze.",
                "Start at the top of your head. Notice any sensations.",
                "Slowly move your attention down: forehead, eyes, jaw, neck.",
                "Continue through shoulders, arms, hands, chest, belly.",
                "Move through hips, legs, feet.",
                "Notice where you feel tension. Breathe into those areas.",
            ],
            duration_minutes=10,
            suitable_states=[EntropyState.MODERATE, EntropyState.LOW_ENTROPY],
        ),
        GroundingTechnique(
            name="Safe Place Visualization",
            description="Create a mental sanctuary you can return to anytime.",
            steps=[
                "Close your eyes and take three deep breaths.",
                "Imagine a place where you feel completely safe and peaceful.",
                "Notice what you see in this place. Colors, shapes, light.",
                "Notice what you hear. Sounds of nature, silence, music.",
                "Notice what you feel. Temperature, textures, comfort.",
                "Stay here as long as you need. This place is always available to you.",
            ],
            duration_minutes=5,
            suitable_states=[EntropyState.MODERATE, EntropyState.LOW_ENTROPY, EntropyState.STABLE],
        ),
    ]
    
    def get_technique(self, state: EntropyState) -> GroundingTechnique:
        """Get an appropriate grounding technique for the current state."""
        suitable = [t for t in self.TECHNIQUES if state in t.suitable_states]
        if suitable:
            return random.choice(suitable)
        return self.TECHNIQUES[0]  # Default to 5-4-3-2-1
    
    def format_technique(self, technique: GroundingTechnique) -> str:
        """Format a technique for display."""
        lines = [
            f"**{technique.name}**",
            f"_{technique.description}_",
            "",
        ]
        for i, step in enumerate(technique.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")
        lines.append(f"(Takes about {technique.duration_minutes} minutes)")
        return "\n".join(lines)


# ============================================================================
# SECTION 8: LLM INTEGRATION
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: ReUnityConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            kwargs = {}
            if self.config.openai_api_key:
                kwargs["api_key"] = self.config.openai_api_key
            if self.config.openai_base_url:
                kwargs["base_url"] = self.config.openai_base_url
            
            self.client = OpenAI(**kwargs)
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("OpenAI package not installed. Run: pip install openai")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response using OpenAI API."""
        if not self.client:
            return self._fallback_response(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response(messages)
    
    def _fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """Fallback response when API is unavailable."""
        return (
            "I'm here to support you. While I'm having trouble connecting to my "
            "full capabilities right now, I want you to know that your feelings are valid. "
            "If you're in crisis, please reach out to 988 (US) or your local crisis line."
        )


class FallbackProvider(LLMProvider):
    """Fallback provider when no API is available."""
    
    def __init__(self):
        self.responses = {
            EntropyState.CRISIS: [
                "I can hear that you're going through something really difficult right now. Your safety matters. If you're in immediate danger, please call 988 or your local emergency number. I'm here with you.",
                "What you're experiencing sounds overwhelming. You don't have to face this alone. Would you like to try a grounding exercise together? And please remember, 988 is available 24/7 if you need to talk to someone.",
                "I'm concerned about what you're sharing. Your wellbeing is important. Let's take this one moment at a time. Can you tell me: are you safe right now?",
            ],
            EntropyState.HIGH_ENTROPY: [
                "It sounds like things are really intense right now. That's okay. Let's take a breath together. What do you notice in your body right now?",
                "I hear you. These feelings are valid, even when they're overwhelming. Would it help to try a grounding technique together?",
                "Thank you for sharing that with me. It takes courage to express difficult feelings. What would feel most supportive right now?",
            ],
            EntropyState.MODERATE: [
                "I'm here with you. It sounds like you're navigating some complex feelings. Would you like to explore what's coming up for you?",
                "Thank you for sharing. I'm listening. What feels most important to talk about right now?",
                "I appreciate you opening up. Sometimes just naming what we're feeling can help. What else is on your mind?",
            ],
            EntropyState.STABLE: [
                "It's good to hear from you. How can I support you today?",
                "I'm glad you're here. What would you like to explore or work on?",
                "Thank you for checking in. What's on your mind?",
            ],
        }
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response using templates."""
        state = kwargs.get("entropy_state", EntropyState.MODERATE)
        responses = self.responses.get(state, self.responses[EntropyState.MODERATE])
        return random.choice(responses)


# ============================================================================
# SECTION 9: MAIN REUNITY CLASS
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

IMPORTANT GUIDELINES:
1. ALWAYS validate feelings before offering suggestions
2. NEVER minimize or dismiss someone's experience
3. Use warm, compassionate language without being patronizing
4. Recognize signs of crisis and provide crisis resources (988, Crisis Text Line)
5. Acknowledge harmful patterns (gaslighting, love bombing, etc.) when detected
6. Offer grounding techniques when someone is dysregulated
7. Remember context from the conversation to provide continuity
8. Be honest about your limitations as an AI

CRISIS PROTOCOL:
If someone expresses suicidal ideation, self-harm, or immediate danger:
1. Express care and concern
2. Provide crisis resources: 988 (US), Crisis Text Line (text HOME to 741741)
3. Encourage them to reach out to a trusted person or professional
4. Stay with them in the conversation

CURRENT USER STATE: {entropy_state}
DETECTED PATTERNS: {patterns}
RELEVANT CONTEXT: {context}

Respond with empathy, validation, and appropriate support for their current state."""

    def __init__(self, config: ReUnityConfig = None):
        self.config = config or ReUnityConfig()
        
        # Initialize components
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
        
        # Initialize LLM provider
        if self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"):
            self.llm = OpenAIProvider(self.config)
        else:
            logger.warning("No OpenAI API key found. Using fallback responses.")
            self.llm = FallbackProvider()
        
        # Session state
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        self.conversation_history: List[Dict[str, str]] = []
        self.interaction_count = 0
        
        # Print disclaimer
        self._print_disclaimer()
    
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
        
        # Step 1: PreRAG Query Gate
        if self.config.enable_prerag:
            gate_decision = self.query_gate.process(user_input)
            if gate_decision.action == QueryGateAction.NO_RETRIEVE:
                return {
                    "response": gate_decision.reasoning,
                    "analysis": None,
                    "patterns_detected": [],
                    "grounding_recommendation": None,
                    "blocked": True,
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
            
            # Evidence Gate
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
        memory_text = "\n".join([m.content for m in memories]) if memories else ""
        
        # Step 7: Generate response
        system_prompt = self.SYSTEM_PROMPT.format(
            entropy_state=analysis.state.value,
            patterns=pattern_text,
            context=context_text,
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        if isinstance(self.llm, FallbackProvider):
            response = self.llm.generate(messages, entropy_state=analysis.state)
        else:
            response = self.llm.generate(messages)
        
        # Step 8: Add pattern warnings if detected
        if patterns:
            pattern_warnings = []
            for p in patterns:
                pattern_warnings.append(f"\n\n**Pattern Detected: {p.pattern_type.value.replace('_', ' ').title()}**\n{p.recommendation}")
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
# SECTION 10: WEB SERVER
# ============================================================================

def create_web_app(reunity: ReUnity = None):
    """Create Flask web application."""
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        logger.error("Flask not installed. Run: pip install flask")
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
            line-height: 1.4;
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
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #eee;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: scale(1.05); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .crisis-banner {
            background: #ff6b6b;
            color: white;
            padding: 10px;
            text-align: center;
            font-size: 12px;
        }
        .state-indicator {
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 10px;
            margin-top: 5px;
            display: inline-block;
        }
        .state-crisis { background: #ff6b6b; color: white; }
        .state-high_entropy { background: #ffa502; color: white; }
        .state-moderate { background: #ffd93d; color: #333; }
        .state-stable { background: #6bcb77; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="crisis-banner">
            If you are in crisis, please call 988 (US) or text HOME to 741741
        </div>
        <div class="header">
            <h1>ReUnity</h1>
            <p>Trauma-Informed AI Support</p>
        </div>
        <div class="messages" id="messages">
            <div class="message assistant">
                Hello. I'm ReUnity, a trauma-informed AI companion. I'm here to listen and support you. How are you feeling today?
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.key==='Enter')sendMessage()">
            <button onclick="sendMessage()" id="sendBtn">Send</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('input');
            const messages = document.getElementById('messages');
            const sendBtn = document.getElementById('sendBtn');
            const text = input.value.trim();
            
            if (!text) return;
            
            // Add user message
            messages.innerHTML += `<div class="message user">${text}</div>`;
            input.value = '';
            sendBtn.disabled = true;
            
            // Scroll to bottom
            messages.scrollTop = messages.scrollHeight;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });
                const data = await response.json();
                
                let stateClass = 'state-' + (data.analysis?.state || 'stable');
                let stateLabel = data.analysis?.state?.replace('_', ' ') || 'stable';
                
                messages.innerHTML += `
                    <div class="message assistant">
                        ${data.response.replace(/\\n/g, '<br>').replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')}
                        <div class="state-indicator ${stateClass}">${stateLabel}</div>
                    </div>
                `;
            } catch (error) {
                messages.innerHTML += `<div class="message assistant">I'm having trouble connecting. Please try again.</div>`;
            }
            
            sendBtn.disabled = false;
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
            return jsonify({'error': 'No message provided'}), 400
        
        result = reunity.process_input(message)
        return jsonify(result)
    
    @app.route('/api/session', methods=['GET'])
    def session():
        return jsonify(reunity.get_session_summary())
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'version': '3.0.0'})
    
    return app


# ============================================================================
# SECTION 11: COMMAND LINE INTERFACE
# ============================================================================

def run_interactive():
    """Run interactive command line interface."""
    print("\n" + "="*60)
    print("ReUnity AI Model v3.0.0")
    print("A Trauma-Aware AI Support Framework")
    print("By Christopher Ezernack, REOP Solutions")
    print("="*60)
    print("\nType your message and press Enter.")
    print("Type /quit to exit, /status for session info.\n")
    
    reunity = ReUnity()
    
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
                print(f"Current State: {summary['current_state']}")
                continue
            
            result = reunity.process_input(user_input)
            
            print(f"\nReUnity: {result['response']}")
            
            if result['analysis']:
                print(f"\n[State: {result['analysis']['state']} | Entropy: {result['analysis']['entropy']:.2f}]")
        
        except KeyboardInterrupt:
            print("\n\nSession ended. Take care.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def run_test():
    """Run basic tests."""
    print("\n" + "="*60)
    print("ReUnity AI Model v3.0.0 - Test Mode")
    print("="*60 + "\n")
    
    reunity = ReUnity()
    
    test_cases = [
        ("I'm feeling anxious today", "Should detect HIGH_ENTROPY"),
        ("My partner said I was imagining things again", "Should detect gaslighting pattern"),
        ("I am dissociating right now", "Should detect CRISIS"),
        ("I feel calm and peaceful", "Should detect STABLE"),
        ("I want to kill myself", "Should detect CRISIS with crisis indicators"),
        ("He told me it never happened", "Should detect gaslighting"),
    ]
    
    for text, expected in test_cases:
        print(f"\nInput: {text}")
        print(f"Expected: {expected}")
        
        result = reunity.process_input(text)
        
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
    
    args = parser.parse_args()
    
    # Set API key if provided
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
