"""
EvidenceGate - Post-Retrieval Filter

Layer 2 of the two-layer Pre-RAG filter. Runs after retrieval to:
- Score retrieved chunks (relevance + coverage)
- Compute absurdity gap posterior
- Decide: answer, clarify, or refuse

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reunity.prerag.absurdity_gap import AbsurdityGapCalculator, AbsurdityGapMetrics

logger = logging.getLogger(__name__)


class EvidenceGateAction(Enum):
    """Actions the EvidenceGate can take."""
    
    ANSWER = "answer"  # Proceed to generate answer
    CLARIFY = "clarify"  # Ask for clarification
    REFUSE = "refuse"  # Refuse to answer (insufficient evidence)


@dataclass
class ChunkScore:
    """Score for a retrieved chunk."""
    
    chunk_id: str
    text: str
    relevance_score: float
    coverage_score: float
    combined_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGateDecision:
    """Decision made by the EvidenceGate."""
    
    action: EvidenceGateAction
    absurdity_gap_prior: AbsurdityGapMetrics
    absurdity_gap_posterior: AbsurdityGapMetrics
    chunk_scores: list[ChunkScore]
    selected_chunks: list[str]
    confidence: float
    reasoning: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvidenceGate:
    """
    Post-retrieval gate that validates retrieved evidence and decides
    whether to answer, ask for clarification, or refuse.
    
    This is Layer 2 of the two-layer Pre-RAG filter.
    """
    
    def __init__(
        self,
        absurdity_calculator: AbsurdityGapCalculator | None = None,
        answer_threshold: float = 0.3,
        clarify_threshold: float = 0.6,
        refuse_threshold: float = 0.85,
        min_relevance: float = 0.3,
        min_chunks_for_answer: int = 1,
        max_chunks_to_use: int = 5,
        coverage_weight: float = 0.3,
        relevance_weight: float = 0.7,
        debug: bool = False,
    ) -> None:
        """
        Initialize the EvidenceGate.
        
        Args:
            absurdity_calculator: Calculator for absurdity gap.
            answer_threshold: Below this posterior gap, proceed to answer.
            clarify_threshold: Above this, ask for clarification.
            refuse_threshold: Above this, refuse to answer.
            min_relevance: Minimum relevance score for chunk selection.
            min_chunks_for_answer: Minimum chunks needed to answer.
            max_chunks_to_use: Maximum chunks to include in context.
            coverage_weight: Weight for coverage in combined score.
            relevance_weight: Weight for relevance in combined score.
            debug: Enable debug logging.
        """
        self.absurdity_calculator = absurdity_calculator or AbsurdityGapCalculator(
            debug=debug
        )
        self.answer_threshold = answer_threshold
        self.clarify_threshold = clarify_threshold
        self.refuse_threshold = refuse_threshold
        self.min_relevance = min_relevance
        self.min_chunks_for_answer = min_chunks_for_answer
        self.max_chunks_to_use = max_chunks_to_use
        self.coverage_weight = coverage_weight
        self.relevance_weight = relevance_weight
        self.debug = debug
        
        logger.info(
            f"EvidenceGate initialized: "
            f"answer_threshold={answer_threshold}, "
            f"refuse_threshold={refuse_threshold}"
        )
    
    def process(
        self,
        query: str,
        query_embedding: NDArray[np.floating] | None,
        retrieved_chunks: list[dict[str, Any]],
        chunk_embeddings: list[NDArray[np.floating]] | None = None,
        prior_gap: AbsurdityGapMetrics | None = None,
    ) -> EvidenceGateDecision:
        """
        Process retrieved evidence through the gate.
        
        Args:
            query: The original query.
            query_embedding: Query embedding vector.
            retrieved_chunks: List of retrieved chunks with text and metadata.
            chunk_embeddings: Embeddings for each chunk.
            prior_gap: Absurdity gap from QueryGate (prior).
        
        Returns:
            EvidenceGateDecision with action and details.
        """
        # Calculate prior if not provided
        if prior_gap is None:
            prior_gap = self.absurdity_calculator.calculate_gap(
                query=query,
                query_embedding=query_embedding,
            )
        
        # Handle empty retrieval
        if not retrieved_chunks:
            return EvidenceGateDecision(
                action=EvidenceGateAction.REFUSE,
                absurdity_gap_prior=prior_gap,
                absurdity_gap_posterior=AbsurdityGapMetrics(
                    gap_score=1.0,
                    confidence=0.9,
                    method="no_evidence",
                    best_anchor_similarity=0.0,
                    mean_anchor_similarity=0.0,
                    query_length_factor=0.0,
                ),
                chunk_scores=[],
                selected_chunks=[],
                confidence=0.9,
                reasoning="No evidence retrieved",
                warnings=["No relevant documents found"],
            )
        
        # Score each chunk
        chunk_scores = self._score_chunks(
            query=query,
            query_embedding=query_embedding,
            chunks=retrieved_chunks,
            chunk_embeddings=chunk_embeddings,
        )
        
        # Filter by minimum relevance
        relevant_chunks = [
            cs for cs in chunk_scores
            if cs.relevance_score >= self.min_relevance
        ]
        
        # Select top chunks
        relevant_chunks.sort(key=lambda x: x.combined_score, reverse=True)
        selected = relevant_chunks[:self.max_chunks_to_use]
        selected_texts = [cs.text for cs in selected]
        
        # Calculate posterior absurdity gap with evidence
        posterior_gap = self.absurdity_calculator.calculate_gap(
            query=query,
            query_embedding=query_embedding,
            retrieved_chunks=selected_texts,
            chunk_embeddings=(
                chunk_embeddings[:len(selected)]
                if chunk_embeddings
                else None
            ),
        )
        
        # Decide action based on posterior gap and evidence quality
        action, reasoning, warnings = self._decide_action(
            prior_gap=prior_gap,
            posterior_gap=posterior_gap,
            relevant_chunk_count=len(relevant_chunks),
            chunk_scores=chunk_scores,
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            posterior_gap=posterior_gap,
            relevant_chunk_count=len(relevant_chunks),
            chunk_scores=chunk_scores,
        )
        
        decision = EvidenceGateDecision(
            action=action,
            absurdity_gap_prior=prior_gap,
            absurdity_gap_posterior=posterior_gap,
            chunk_scores=chunk_scores,
            selected_chunks=selected_texts,
            confidence=confidence,
            reasoning=reasoning,
            warnings=warnings,
            metadata={
                "total_chunks_retrieved": len(retrieved_chunks),
                "relevant_chunks": len(relevant_chunks),
                "selected_chunks": len(selected),
                "gap_reduction": prior_gap.gap_score - posterior_gap.gap_score,
            },
        )
        
        if self.debug:
            logger.debug(
                f"EvidenceGate decision: {action.value} "
                f"(prior_gap={prior_gap.gap_score:.3f}, "
                f"posterior_gap={posterior_gap.gap_score:.3f})"
            )
        
        return decision
    
    def _score_chunks(
        self,
        query: str,
        query_embedding: NDArray[np.floating] | None,
        chunks: list[dict[str, Any]],
        chunk_embeddings: list[NDArray[np.floating]] | None,
    ) -> list[ChunkScore]:
        """Score each retrieved chunk."""
        scores = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", chunk.get("content", ""))
            chunk_id = chunk.get("id", f"chunk_{i}")
            
            # Calculate relevance
            if query_embedding is not None and chunk_embeddings and i < len(chunk_embeddings):
                relevance = self._embedding_similarity(
                    query_embedding,
                    chunk_embeddings[i],
                )
            else:
                relevance = self._text_similarity(query, chunk_text)
            
            # Calculate coverage (how much of query is covered)
            coverage = self._calculate_coverage(query, chunk_text)
            
            # Combined score
            combined = (
                self.relevance_weight * relevance +
                self.coverage_weight * coverage
            )
            
            scores.append(ChunkScore(
                chunk_id=chunk_id,
                text=chunk_text,
                relevance_score=relevance,
                coverage_score=coverage,
                combined_score=combined,
                metadata=chunk.get("metadata", {}),
            ))
        
        return scores
    
    def _embedding_similarity(
        self,
        emb1: NDArray[np.floating],
        emb2: NDArray[np.floating],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def _text_similarity(self, query: str, chunk: str) -> float:
        """Compute text similarity using word overlap."""
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())
        
        if not query_words or not chunk_words:
            return 0.0
        
        intersection = query_words & chunk_words
        union = query_words | chunk_words
        
        return len(intersection) / len(union)
    
    def _calculate_coverage(self, query: str, chunk: str) -> float:
        """Calculate how much of the query is covered by the chunk."""
        query_words = set(query.lower().split())
        chunk_words = set(chunk.lower().split())
        
        if not query_words:
            return 0.0
        
        covered = query_words & chunk_words
        return len(covered) / len(query_words)
    
    def _decide_action(
        self,
        prior_gap: AbsurdityGapMetrics,
        posterior_gap: AbsurdityGapMetrics,
        relevant_chunk_count: int,
        chunk_scores: list[ChunkScore],
    ) -> tuple[EvidenceGateAction, str, list[str]]:
        """Decide action based on evidence analysis."""
        warnings = []
        
        # Check for high posterior gap
        if posterior_gap.gap_score > self.refuse_threshold:
            return (
                EvidenceGateAction.REFUSE,
                "Evidence insufficient - absurdity gap remains too high",
                ["Retrieved evidence does not adequately address the query"],
            )
        
        # Check for insufficient relevant chunks
        if relevant_chunk_count < self.min_chunks_for_answer:
            return (
                EvidenceGateAction.CLARIFY,
                f"Only {relevant_chunk_count} relevant chunks found",
                ["Could you provide more specific details?"],
            )
        
        # Check if gap improved
        gap_improvement = prior_gap.gap_score - posterior_gap.gap_score
        if gap_improvement < 0:
            warnings.append("Evidence did not reduce absurdity gap")
        
        # Check posterior gap thresholds
        if posterior_gap.gap_score < self.answer_threshold:
            return (
                EvidenceGateAction.ANSWER,
                "Evidence is sufficient to answer",
                warnings,
            )
        elif posterior_gap.gap_score < self.clarify_threshold:
            return (
                EvidenceGateAction.ANSWER,
                "Evidence is moderately sufficient",
                warnings + ["Answer may be incomplete"],
            )
        else:
            return (
                EvidenceGateAction.CLARIFY,
                "Evidence quality is borderline",
                warnings + ["Please clarify your question for a better answer"],
            )
    
    def _calculate_confidence(
        self,
        posterior_gap: AbsurdityGapMetrics,
        relevant_chunk_count: int,
        chunk_scores: list[ChunkScore],
    ) -> float:
        """Calculate confidence in the decision."""
        # Base confidence from gap
        gap_confidence = 1.0 - posterior_gap.gap_score
        
        # Chunk count factor
        chunk_factor = min(1.0, relevant_chunk_count / 3.0)
        
        # Average chunk quality
        if chunk_scores:
            avg_quality = sum(cs.combined_score for cs in chunk_scores) / len(chunk_scores)
        else:
            avg_quality = 0.0
        
        # Combined confidence
        confidence = (
            0.4 * gap_confidence +
            0.3 * chunk_factor +
            0.3 * avg_quality
        )
        
        return max(0.0, min(1.0, confidence))
    
    def get_statistics(self) -> dict[str, Any]:
        """Get gate statistics."""
        return {
            "answer_threshold": self.answer_threshold,
            "clarify_threshold": self.clarify_threshold,
            "refuse_threshold": self.refuse_threshold,
            "min_relevance": self.min_relevance,
            "min_chunks_for_answer": self.min_chunks_for_answer,
            "absurdity_calculator": self.absurdity_calculator.get_statistics(),
        }
