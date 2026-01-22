"""
Retriever

RAG retriever with Pre-RAG gate integration.

DISCLAIMER: This is not a clinical or treatment tool.

Author: Christopher Ezernack
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from reunity.rag.chunker import Chunk
from reunity.rag.indexer import FAISSIndexer
from reunity.prerag.query_gate import QueryGate, QueryGateDecision, QueryGateAction
from reunity.prerag.evidence_gate import EvidenceGate, EvidenceGateDecision, EvidenceGateAction

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    
    query: str
    chunks: list[Chunk]
    scores: list[float]
    query_gate_decision: QueryGateDecision | None
    evidence_gate_decision: EvidenceGateDecision | None
    final_action: str
    should_answer: bool
    clarification_needed: bool
    clarification_message: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever:
    """
    RAG retriever with Pre-RAG gate integration.
    
    Integrates QueryGate (pre-retrieval) and EvidenceGate (post-retrieval)
    to validate queries and evidence before answering.
    """
    
    def __init__(
        self,
        indexer: FAISSIndexer,
        query_gate: QueryGate | None = None,
        evidence_gate: EvidenceGate | None = None,
        embed_fn: Callable[[str], NDArray[np.floating]] | None = None,
        top_k: int = 5,
        enable_prerag: bool = True,
        debug: bool = False,
    ) -> None:
        """
        Initialize the retriever.
        
        Args:
            indexer: FAISS indexer with indexed chunks.
            query_gate: Pre-retrieval gate (optional).
            evidence_gate: Post-retrieval gate (optional).
            embed_fn: Function to embed text. If None, uses simple hashing.
            top_k: Number of chunks to retrieve.
            enable_prerag: Enable Pre-RAG gates.
            debug: Enable debug logging.
        """
        self.indexer = indexer
        self.query_gate = query_gate
        self.evidence_gate = evidence_gate
        self.embed_fn = embed_fn or self._simple_embed
        self.top_k = top_k
        self.enable_prerag = enable_prerag
        self.debug = debug
        
        logger.info(
            f"Retriever initialized: "
            f"top_k={top_k}, prerag={enable_prerag}"
        )
    
    def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text.
            context: Additional context for gates.
        
        Returns:
            RetrievalResult with chunks and gate decisions.
        """
        context = context or {}
        
        # Embed query
        query_embedding = self.embed_fn(query)
        
        # Layer 1: QueryGate (pre-retrieval)
        query_gate_decision = None
        if self.enable_prerag and self.query_gate:
            query_gate_decision = self.query_gate.process(
                query=query,
                query_embedding=query_embedding,
                context=context,
            )
            
            if self.debug:
                logger.debug(
                    f"QueryGate: {query_gate_decision.action.value} "
                    f"(gap={query_gate_decision.absurdity_gap.gap_score:.3f})"
                )
            
            # Handle QueryGate decisions
            if query_gate_decision.action == QueryGateAction.NO_RETRIEVE:
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    scores=[],
                    query_gate_decision=query_gate_decision,
                    evidence_gate_decision=None,
                    final_action="no_retrieve",
                    should_answer=False,
                    clarification_needed=False,
                    clarification_message=query_gate_decision.reasoning,
                    metadata={"blocked_by": "query_gate"},
                )
            
            if query_gate_decision.action == QueryGateAction.CLARIFY:
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    scores=[],
                    query_gate_decision=query_gate_decision,
                    evidence_gate_decision=None,
                    final_action="clarify",
                    should_answer=False,
                    clarification_needed=True,
                    clarification_message=(
                        query_gate_decision.suggestions[0]
                        if query_gate_decision.suggestions
                        else "Could you please clarify your question?"
                    ),
                    metadata={"blocked_by": "query_gate"},
                )
        
        # Retrieve chunks
        results = self.indexer.search(query_embedding, k=self.top_k)
        chunks = [chunk for chunk, _ in results]
        scores = [score for _, score in results]
        
        # Layer 2: EvidenceGate (post-retrieval)
        evidence_gate_decision = None
        if self.enable_prerag and self.evidence_gate:
            # Prepare chunks for evidence gate
            retrieved_chunks = [
                {"text": chunk.text, "id": chunk.id, "metadata": chunk.metadata}
                for chunk in chunks
            ]
            
            # Get chunk embeddings
            chunk_embeddings = [self.embed_fn(chunk.text) for chunk in chunks]
            
            evidence_gate_decision = self.evidence_gate.process(
                query=query,
                query_embedding=query_embedding,
                retrieved_chunks=retrieved_chunks,
                chunk_embeddings=chunk_embeddings,
                prior_gap=(
                    query_gate_decision.absurdity_gap
                    if query_gate_decision
                    else None
                ),
            )
            
            if self.debug:
                logger.debug(
                    f"EvidenceGate: {evidence_gate_decision.action.value} "
                    f"(posterior_gap={evidence_gate_decision.absurdity_gap_posterior.gap_score:.3f})"
                )
            
            # Handle EvidenceGate decisions
            if evidence_gate_decision.action == EvidenceGateAction.REFUSE:
                return RetrievalResult(
                    query=query,
                    chunks=chunks,
                    scores=scores,
                    query_gate_decision=query_gate_decision,
                    evidence_gate_decision=evidence_gate_decision,
                    final_action="refuse",
                    should_answer=False,
                    clarification_needed=False,
                    clarification_message=evidence_gate_decision.reasoning,
                    metadata={"blocked_by": "evidence_gate"},
                )
            
            if evidence_gate_decision.action == EvidenceGateAction.CLARIFY:
                return RetrievalResult(
                    query=query,
                    chunks=chunks,
                    scores=scores,
                    query_gate_decision=query_gate_decision,
                    evidence_gate_decision=evidence_gate_decision,
                    final_action="clarify",
                    should_answer=False,
                    clarification_needed=True,
                    clarification_message=(
                        evidence_gate_decision.warnings[0]
                        if evidence_gate_decision.warnings
                        else "Could you provide more details?"
                    ),
                    metadata={"blocked_by": "evidence_gate"},
                )
        
        # Success - proceed to answer
        return RetrievalResult(
            query=query,
            chunks=chunks,
            scores=scores,
            query_gate_decision=query_gate_decision,
            evidence_gate_decision=evidence_gate_decision,
            final_action="answer",
            should_answer=True,
            clarification_needed=False,
            clarification_message=None,
            metadata={
                "num_chunks": len(chunks),
                "prerag_enabled": self.enable_prerag,
            },
        )
    
    def _simple_embed(self, text: str) -> NDArray[np.floating]:
        """
        Simple embedding function using character n-grams.
        This is a fallback when no embedding model is available.
        """
        # Create a simple hash-based embedding
        dim = self.indexer.embedding_dim
        embedding = np.zeros(dim, dtype=np.float32)
        
        # Use character trigrams
        text = text.lower()
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            # Hash to index
            idx = hash(trigram) % dim
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_anchors_from_index(self) -> None:
        """Add indexed chunks as anchors to the query gate."""
        if self.query_gate:
            for chunk in self.indexer._chunks:
                self.query_gate.add_anchor(chunk.text)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "indexer": self.indexer.get_statistics(),
            "top_k": self.top_k,
            "prerag_enabled": self.enable_prerag,
        }
        
        if self.query_gate:
            stats["query_gate"] = self.query_gate.get_statistics()
        
        if self.evidence_gate:
            stats["evidence_gate"] = self.evidence_gate.get_statistics()
        
        return stats
