"""
Absurdity Gap Calculator

Estimates how far an input is from being answerable or grounded using:
- Embedding similarity against corpus anchors and retrieved evidence
- Fallback TF-IDF cosine similarity if embeddings unavailable

DISCLAIMER: This is not a clinical or treatment tool. It is a theoretical
and support framework only.

Author: Christopher Ezernack
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class AbsurdityGapMetrics:
    """Metrics from absurdity gap calculation."""
    
    gap_score: float  # 0.0 = highly grounded, 1.0 = highly absurd
    confidence: float  # Confidence in the gap score
    method: str  # "embedding" or "tfidf"
    best_anchor_similarity: float
    mean_anchor_similarity: float
    query_length_factor: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AbsurdityGapCalculator:
    """
    Calculates absurdity gap - a scalar score estimating how far
    an input is from being answerable or grounded.
    
    The gap is computed using:
    1. Embedding similarity against corpus anchors (preferred)
    2. TF-IDF cosine similarity (fallback)
    
    A higher gap score indicates the query is further from
    being answerable with available knowledge.
    """
    
    def __init__(
        self,
        use_embeddings: bool = True,
        embedding_dim: int = 384,
        min_query_length: int = 3,
        max_query_length: int = 500,
        gap_threshold_low: float = 0.3,
        gap_threshold_high: float = 0.7,
        debug: bool = False,
    ) -> None:
        """
        Initialize the absurdity gap calculator.
        
        Args:
            use_embeddings: Whether to use embeddings (vs TF-IDF only).
            embedding_dim: Dimension of embedding vectors.
            min_query_length: Minimum query length in characters.
            max_query_length: Maximum query length in characters.
            gap_threshold_low: Below this, query is well-grounded.
            gap_threshold_high: Above this, query is likely absurd.
            debug: Enable debug logging.
        """
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.gap_threshold_low = gap_threshold_low
        self.gap_threshold_high = gap_threshold_high
        self.debug = debug
        
        # Corpus anchors (embeddings or TF-IDF vectors)
        self._anchor_embeddings: list[NDArray[np.floating]] = []
        self._anchor_texts: list[str] = []
        self._anchor_tfidf: list[dict[str, float]] = []
        
        # Vocabulary for TF-IDF
        self._vocabulary: set[str] = set()
        self._idf: dict[str, float] = {}
        
        logger.info(
            f"AbsurdityGapCalculator initialized: "
            f"embeddings={use_embeddings}, debug={debug}"
        )
    
    def add_anchor(
        self,
        text: str,
        embedding: NDArray[np.floating] | None = None,
    ) -> None:
        """
        Add a corpus anchor for comparison.
        
        Args:
            text: Anchor text content.
            embedding: Pre-computed embedding (optional).
        """
        self._anchor_texts.append(text)
        
        # Store embedding if provided
        if embedding is not None:
            self._anchor_embeddings.append(embedding)
        
        # Always compute TF-IDF representation as fallback
        tfidf = self._compute_tfidf(text)
        self._anchor_tfidf.append(tfidf)
        
        # Update vocabulary
        tokens = self._tokenize(text)
        self._vocabulary.update(tokens)
        
        # Recompute IDF when anchors change
        self._recompute_idf()
        
        if self.debug:
            logger.debug(f"Added anchor: {text[:50]}...")
    
    def add_anchors_from_texts(self, texts: list[str]) -> None:
        """Add multiple anchors from text list."""
        for text in texts:
            self.add_anchor(text)
    
    def calculate_gap(
        self,
        query: str,
        query_embedding: NDArray[np.floating] | None = None,
        retrieved_chunks: list[str] | None = None,
        chunk_embeddings: list[NDArray[np.floating]] | None = None,
    ) -> AbsurdityGapMetrics:
        """
        Calculate the absurdity gap for a query.
        
        Args:
            query: The input query text.
            query_embedding: Pre-computed query embedding (optional).
            retrieved_chunks: Retrieved evidence chunks (for posterior).
            chunk_embeddings: Embeddings of retrieved chunks.
        
        Returns:
            AbsurdityGapMetrics with gap score and details.
        """
        # Validate query
        query_length_factor = self._compute_query_length_factor(query)
        
        # Choose method based on availability
        if (
            self.use_embeddings
            and query_embedding is not None
            and len(self._anchor_embeddings) > 0
        ):
            method = "embedding"
            similarities = self._compute_embedding_similarities(
                query_embedding,
                self._anchor_embeddings,
            )
        else:
            method = "tfidf"
            query_tfidf = self._compute_tfidf(query)
            similarities = self._compute_tfidf_similarities(
                query_tfidf,
                self._anchor_tfidf,
            )
        
        # Handle empty anchors
        if len(similarities) == 0:
            return AbsurdityGapMetrics(
                gap_score=0.5,  # Neutral when no anchors
                confidence=0.0,
                method=method,
                best_anchor_similarity=0.0,
                mean_anchor_similarity=0.0,
                query_length_factor=query_length_factor,
                metadata={"warning": "no_anchors"},
            )
        
        # Compute gap from similarities
        best_sim = float(np.max(similarities))
        mean_sim = float(np.mean(similarities))
        
        # Gap is inverse of best similarity, adjusted by query length
        raw_gap = 1.0 - best_sim
        adjusted_gap = raw_gap * (0.5 + 0.5 * query_length_factor)
        
        # Include retrieved evidence in posterior calculation
        if retrieved_chunks and len(retrieved_chunks) > 0:
            if chunk_embeddings and query_embedding is not None:
                evidence_sims = self._compute_embedding_similarities(
                    query_embedding,
                    chunk_embeddings,
                )
            else:
                query_tfidf = self._compute_tfidf(query)
                chunk_tfidfs = [self._compute_tfidf(c) for c in retrieved_chunks]
                evidence_sims = self._compute_tfidf_similarities(
                    query_tfidf,
                    chunk_tfidfs,
                )
            
            if len(evidence_sims) > 0:
                evidence_factor = float(np.max(evidence_sims))
                # Reduce gap if good evidence found
                adjusted_gap *= (1.0 - 0.5 * evidence_factor)
        
        # Clamp to [0, 1]
        gap_score = max(0.0, min(1.0, adjusted_gap))
        
        # Confidence based on anchor coverage
        confidence = min(1.0, len(self._anchor_texts) / 10.0)
        
        metrics = AbsurdityGapMetrics(
            gap_score=gap_score,
            confidence=confidence,
            method=method,
            best_anchor_similarity=best_sim,
            mean_anchor_similarity=mean_sim,
            query_length_factor=query_length_factor,
            metadata={
                "num_anchors": len(self._anchor_texts),
                "num_evidence_chunks": len(retrieved_chunks) if retrieved_chunks else 0,
            },
        )
        
        if self.debug:
            logger.debug(
                f"Absurdity gap: {gap_score:.3f} "
                f"(method={method}, best_sim={best_sim:.3f})"
            )
        
        return metrics
    
    def is_grounded(self, gap_metrics: AbsurdityGapMetrics) -> bool:
        """Check if query is well-grounded (low gap)."""
        return gap_metrics.gap_score < self.gap_threshold_low
    
    def is_absurd(self, gap_metrics: AbsurdityGapMetrics) -> bool:
        """Check if query is likely absurd (high gap)."""
        return gap_metrics.gap_score > self.gap_threshold_high
    
    def _compute_query_length_factor(self, query: str) -> float:
        """Compute factor based on query length (0=too short, 1=too long)."""
        length = len(query.strip())
        
        if length < self.min_query_length:
            return 1.0  # Too short is suspicious
        elif length > self.max_query_length:
            return 0.8  # Too long is somewhat suspicious
        else:
            # Optimal range
            return 0.0
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for TF-IDF."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def _compute_tfidf(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        
        if not tokens:
            return {}
        
        # Term frequency
        tf = Counter(tokens)
        total = len(tokens)
        
        # TF-IDF
        tfidf = {}
        for term, count in tf.items():
            tf_score = count / total
            idf_score = self._idf.get(term, 1.0)
            tfidf[term] = tf_score * idf_score
        
        return tfidf
    
    def _recompute_idf(self) -> None:
        """Recompute IDF scores from anchors."""
        if not self._anchor_texts:
            return
        
        # Document frequency
        df = Counter()
        for text in self._anchor_texts:
            tokens = set(self._tokenize(text))
            df.update(tokens)
        
        # IDF
        n_docs = len(self._anchor_texts)
        self._idf = {
            term: math.log(n_docs / (1 + count))
            for term, count in df.items()
        }
    
    def _compute_embedding_similarities(
        self,
        query_emb: NDArray[np.floating],
        anchor_embs: list[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Compute cosine similarities between query and anchors."""
        if not anchor_embs:
            return np.array([])
        
        # Normalize query
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        similarities = []
        for anchor_emb in anchor_embs:
            anchor_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
            sim = float(np.dot(query_norm, anchor_norm))
            similarities.append(sim)
        
        return np.array(similarities)
    
    def _compute_tfidf_similarities(
        self,
        query_tfidf: dict[str, float],
        anchor_tfidfs: list[dict[str, float]],
    ) -> NDArray[np.floating]:
        """Compute cosine similarities using TF-IDF vectors."""
        if not anchor_tfidfs or not query_tfidf:
            return np.array([])
        
        similarities = []
        for anchor_tfidf in anchor_tfidfs:
            sim = self._cosine_similarity_dicts(query_tfidf, anchor_tfidf)
            similarities.append(sim)
        
        return np.array(similarities)
    
    def _cosine_similarity_dicts(
        self,
        vec1: dict[str, float],
        vec2: dict[str, float],
    ) -> float:
        """Compute cosine similarity between two sparse vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Dot product
        common_keys = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[k] * vec2[k] for k in common_keys)
        
        # Norms
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get calculator statistics."""
        return {
            "num_anchors": len(self._anchor_texts),
            "vocabulary_size": len(self._vocabulary),
            "use_embeddings": self.use_embeddings,
            "gap_threshold_low": self.gap_threshold_low,
            "gap_threshold_high": self.gap_threshold_high,
        }
