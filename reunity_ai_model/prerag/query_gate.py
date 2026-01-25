"""
QueryGate - Pre-Retrieval Filter

Layer 1 of the two-layer Pre-RAG filter. Runs before retrieval to:
- Normalize query
- Compute absurdity gap prior using anchors
- Decide: retrieve, clarify, or no_retrieve

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reunity.prerag.absurdity_gap import AbsurdityGapCalculator, AbsurdityGapMetrics

logger = logging.getLogger(__name__)


class QueryGateAction(Enum):
    """Actions the QueryGate can take."""
    
    RETRIEVE = "retrieve"  # Proceed with retrieval
    CLARIFY = "clarify"  # Ask for clarification
    NO_RETRIEVE = "no_retrieve"  # Skip retrieval, answer directly or refuse


@dataclass
class QueryGateDecision:
    """Decision made by the QueryGate."""
    
    action: QueryGateAction
    normalized_query: str
    absurdity_gap: AbsurdityGapMetrics
    confidence: float
    reasoning: str
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryGate:
    """
    Pre-retrieval gate that decides whether to proceed with retrieval,
    ask for clarification, or skip retrieval entirely.
    
    This is Layer 1 of the two-layer Pre-RAG filter.
    """
    
    def __init__(
        self,
        absurdity_calculator: AbsurdityGapCalculator | None = None,
        retrieve_threshold: float = 0.4,
        clarify_threshold: float = 0.7,
        min_query_words: int = 2,
        max_query_words: int = 100,
        blocked_patterns: list[str] | None = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize the QueryGate.
        
        Args:
            absurdity_calculator: Calculator for absurdity gap.
            retrieve_threshold: Below this gap, proceed with retrieval.
            clarify_threshold: Above this gap, ask for clarification.
            min_query_words: Minimum words for valid query.
            max_query_words: Maximum words for valid query.
            blocked_patterns: Regex patterns to block.
            debug: Enable debug logging.
        """
        self.absurdity_calculator = absurdity_calculator or AbsurdityGapCalculator(
            debug=debug
        )
        self.retrieve_threshold = retrieve_threshold
        self.clarify_threshold = clarify_threshold
        self.min_query_words = min_query_words
        self.max_query_words = max_query_words
        self.blocked_patterns = blocked_patterns or []
        self.debug = debug
        
        # Compile blocked patterns
        self._blocked_regexes = [
            re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
        ]
        
        logger.info(
            f"QueryGate initialized: "
            f"retrieve_threshold={retrieve_threshold}, "
            f"clarify_threshold={clarify_threshold}"
        )
    
    def process(
        self,
        query: str,
        query_embedding: NDArray[np.floating] | None = None,
        context: dict[str, Any] | None = None,
    ) -> QueryGateDecision:
        """
        Process a query through the gate.
        
        Args:
            query: Raw input query.
            query_embedding: Pre-computed query embedding.
            context: Additional context (user state, session info).
        
        Returns:
            QueryGateDecision with action and details.
        """
        context = context or {}
        
        # Step 1: Normalize query
        normalized = self._normalize_query(query)
        
        # Step 2: Check for blocked patterns
        blocked_match = self._check_blocked_patterns(normalized)
        if blocked_match:
            return QueryGateDecision(
                action=QueryGateAction.NO_RETRIEVE,
                normalized_query=normalized,
                absurdity_gap=AbsurdityGapMetrics(
                    gap_score=1.0,
                    confidence=1.0,
                    method="blocked",
                    best_anchor_similarity=0.0,
                    mean_anchor_similarity=0.0,
                    query_length_factor=0.0,
                ),
                confidence=1.0,
                reasoning=f"Query blocked by pattern: {blocked_match}",
                metadata={"blocked_pattern": blocked_match},
            )
        
        # Step 3: Check query length
        word_count = len(normalized.split())
        if word_count < self.min_query_words:
            return QueryGateDecision(
                action=QueryGateAction.CLARIFY,
                normalized_query=normalized,
                absurdity_gap=AbsurdityGapMetrics(
                    gap_score=0.8,
                    confidence=0.9,
                    method="length_check",
                    best_anchor_similarity=0.0,
                    mean_anchor_similarity=0.0,
                    query_length_factor=1.0,
                ),
                confidence=0.9,
                reasoning="Query too short for meaningful retrieval",
                suggestions=["Please provide more detail about what you're looking for"],
            )
        
        if word_count > self.max_query_words:
            return QueryGateDecision(
                action=QueryGateAction.CLARIFY,
                normalized_query=normalized,
                absurdity_gap=AbsurdityGapMetrics(
                    gap_score=0.6,
                    confidence=0.7,
                    method="length_check",
                    best_anchor_similarity=0.0,
                    mean_anchor_similarity=0.0,
                    query_length_factor=0.8,
                ),
                confidence=0.7,
                reasoning="Query too long - please focus on the main question",
                suggestions=["Try breaking your question into smaller parts"],
            )
        
        # Step 4: Calculate absurdity gap
        gap_metrics = self.absurdity_calculator.calculate_gap(
            query=normalized,
            query_embedding=query_embedding,
        )
        
        # Step 5: Decide action based on gap
        if gap_metrics.gap_score < self.retrieve_threshold:
            action = QueryGateAction.RETRIEVE
            reasoning = "Query is well-grounded, proceeding with retrieval"
            suggestions = []
        elif gap_metrics.gap_score > self.clarify_threshold:
            action = QueryGateAction.CLARIFY
            reasoning = "Query has high absurdity gap, clarification needed"
            suggestions = self._generate_clarification_suggestions(normalized, context)
        else:
            # Middle ground - retrieve but with caution
            action = QueryGateAction.RETRIEVE
            reasoning = "Query has moderate absurdity gap, retrieving with caution"
            suggestions = []
        
        decision = QueryGateDecision(
            action=action,
            normalized_query=normalized,
            absurdity_gap=gap_metrics,
            confidence=gap_metrics.confidence,
            reasoning=reasoning,
            suggestions=suggestions,
            metadata={
                "word_count": word_count,
                "context_provided": bool(context),
            },
        )
        
        if self.debug:
            logger.debug(
                f"QueryGate decision: {action.value} "
                f"(gap={gap_metrics.gap_score:.3f})"
            )
        
        return decision
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text."""
        # Strip whitespace
        normalized = query.strip()
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing punctuation (but keep internal)
        normalized = normalized.strip('.,;:!?')
        
        return normalized
    
    def _check_blocked_patterns(self, query: str) -> str | None:
        """Check if query matches any blocked patterns."""
        for regex in self._blocked_regexes:
            if regex.search(query):
                return regex.pattern
        return None
    
    def _generate_clarification_suggestions(
        self,
        query: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Generate suggestions for clarifying the query."""
        suggestions = []
        
        # Check for vague terms
        vague_terms = ["it", "this", "that", "thing", "stuff"]
        query_lower = query.lower()
        for term in vague_terms:
            if f" {term} " in f" {query_lower} ":
                suggestions.append(
                    f"Could you specify what '{term}' refers to?"
                )
                break
        
        # Check for missing context
        if not context:
            suggestions.append(
                "Providing more context might help me understand better"
            )
        
        # Default suggestion
        if not suggestions:
            suggestions.append(
                "Could you rephrase or provide more specific details?"
            )
        
        return suggestions
    
    def add_anchor(self, text: str, embedding: NDArray[np.floating] | None = None) -> None:
        """Add an anchor to the absurdity calculator."""
        self.absurdity_calculator.add_anchor(text, embedding)
    
    def add_blocked_pattern(self, pattern: str) -> None:
        """Add a blocked pattern."""
        self.blocked_patterns.append(pattern)
        self._blocked_regexes.append(re.compile(pattern, re.IGNORECASE))
    
    def get_statistics(self) -> dict[str, Any]:
        """Get gate statistics."""
        return {
            "retrieve_threshold": self.retrieve_threshold,
            "clarify_threshold": self.clarify_threshold,
            "blocked_patterns_count": len(self.blocked_patterns),
            "absurdity_calculator": self.absurdity_calculator.get_statistics(),
        }
