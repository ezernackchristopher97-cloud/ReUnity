"""
FAISS Indexer

Vector index for RAG retrieval using FAISS.
CPU-safe implementation for Codespaces compatibility.

DISCLAIMER: This is not a clinical or treatment tool.

Author: Christopher Ezernack
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reunity.rag.chunker import Chunk

logger = logging.getLogger(__name__)

# Try to import FAISS, fall back to simple numpy-based index
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not installed, using numpy-based fallback index")


@dataclass
class IndexedChunk:
    """A chunk with its embedding stored in the index."""
    
    chunk: Chunk
    embedding: NDArray[np.floating]
    index_id: int


class FAISSIndexer:
    """
    FAISS-based vector index for RAG.
    
    Falls back to numpy-based similarity search if FAISS is not available.
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "flat",
        use_gpu: bool = False,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize the indexer.
        
        Args:
            embedding_dim: Dimension of embedding vectors.
            index_type: FAISS index type ("flat", "ivf", "hnsw").
            use_gpu: Use GPU for FAISS (if available).
            normalize_embeddings: L2 normalize embeddings.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and HAS_FAISS
        self.normalize_embeddings = normalize_embeddings
        
        # Storage
        self._chunks: list[Chunk] = []
        self._embeddings: list[NDArray[np.floating]] = []
        self._index = None
        self._is_trained = False
        
        # Initialize index
        self._init_index()
        
        logger.info(
            f"FAISSIndexer initialized: "
            f"dim={embedding_dim}, type={index_type}, "
            f"faiss_available={HAS_FAISS}"
        )
    
    def _init_index(self) -> None:
        """Initialize the FAISS index."""
        if not HAS_FAISS:
            self._index = None
            return
        
        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._is_trained = True
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self._index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                100,  # nlist
                faiss.METRIC_INNER_PRODUCT,
            )
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(
                self.embedding_dim,
                32,  # M parameter
                faiss.METRIC_INNER_PRODUCT,
            )
            self._is_trained = True
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._is_trained = True
    
    def add(
        self,
        chunk: Chunk,
        embedding: NDArray[np.floating],
    ) -> int:
        """
        Add a chunk to the index.
        
        Args:
            chunk: Chunk to add.
            embedding: Embedding vector for the chunk.
        
        Returns:
            Index ID of the added chunk.
        """
        # Normalize if needed
        if self.normalize_embeddings:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Store
        index_id = len(self._chunks)
        self._chunks.append(chunk)
        self._embeddings.append(embedding)
        
        # Add to FAISS index if available and trained
        if HAS_FAISS and self._index is not None and self._is_trained:
            self._index.add(embedding.reshape(1, -1).astype(np.float32))
        
        return index_id
    
    def add_batch(
        self,
        chunks: list[Chunk],
        embeddings: list[NDArray[np.floating]],
    ) -> list[int]:
        """
        Add multiple chunks to the index.
        
        Args:
            chunks: List of chunks.
            embeddings: List of embedding vectors.
        
        Returns:
            List of index IDs.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")
        
        index_ids = []
        for chunk, emb in zip(chunks, embeddings):
            index_id = self.add(chunk, emb)
            index_ids.append(index_id)
        
        return index_ids
    
    def train(self, embeddings: NDArray[np.floating] | None = None) -> None:
        """
        Train the index (for IVF-type indexes).
        
        Args:
            embeddings: Training embeddings. Uses stored embeddings if None.
        """
        if not HAS_FAISS or self._index is None:
            self._is_trained = True
            return
        
        if self._is_trained:
            return
        
        if embeddings is None:
            if not self._embeddings:
                raise ValueError("No embeddings to train on")
            embeddings = np.array(self._embeddings).astype(np.float32)
        
        self._index.train(embeddings)
        self._is_trained = True
        
        # Add all stored embeddings
        if self._embeddings:
            all_embs = np.array(self._embeddings).astype(np.float32)
            self._index.add(all_embs)
        
        logger.info(f"Index trained with {len(self._embeddings)} vectors")
    
    def search(
        self,
        query_embedding: NDArray[np.floating],
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.
        
        Returns:
            List of (chunk, score) tuples.
        """
        if not self._chunks:
            return []
        
        # Normalize query
        if self.normalize_embeddings:
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        k = min(k, len(self._chunks))
        
        if HAS_FAISS and self._index is not None and self._is_trained:
            # FAISS search
            query = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(query, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._chunks):
                    results.append((self._chunks[idx], float(score)))
            
            return results
        else:
            # Numpy fallback
            return self._numpy_search(query_embedding, k)
    
    def _numpy_search(
        self,
        query_embedding: NDArray[np.floating],
        k: int,
    ) -> list[tuple[Chunk, float]]:
        """Fallback numpy-based search."""
        if not self._embeddings:
            return []
        
        # Compute similarities
        embeddings = np.array(self._embeddings)
        similarities = np.dot(embeddings, query_embedding)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((self._chunks[idx], float(similarities[idx])))
        
        return results
    
    def save(self, path: str | Path) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Directory path to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_data = [
            {
                "id": c.id,
                "text": c.text,
                "source": c.source,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        with open(path / "chunks.json", "w") as f:
            json.dump(chunks_data, f)
        
        # Save embeddings
        np.save(path / "embeddings.npy", np.array(self._embeddings))
        
        # Save FAISS index
        if HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, str(path / "index.faiss"))
        
        # Save config
        config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "normalize_embeddings": self.normalize_embeddings,
            "is_trained": self._is_trained,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str | Path) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Directory path to load from.
        """
        path = Path(path)
        
        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)
        
        self.embedding_dim = config["embedding_dim"]
        self.index_type = config["index_type"]
        self.normalize_embeddings = config["normalize_embeddings"]
        self._is_trained = config["is_trained"]
        
        # Load chunks
        with open(path / "chunks.json") as f:
            chunks_data = json.load(f)
        
        self._chunks = [
            Chunk(
                id=c["id"],
                text=c["text"],
                source=c["source"],
                start_char=c["start_char"],
                end_char=c["end_char"],
                metadata=c["metadata"],
            )
            for c in chunks_data
        ]
        
        # Load embeddings
        self._embeddings = list(np.load(path / "embeddings.npy"))
        
        # Load FAISS index
        if HAS_FAISS and (path / "index.faiss").exists():
            self._index = faiss.read_index(str(path / "index.faiss"))
        else:
            self._init_index()
            if self._embeddings:
                all_embs = np.array(self._embeddings).astype(np.float32)
                if self._index is not None:
                    self._index.add(all_embs)
        
        logger.info(f"Index loaded from {path}: {len(self._chunks)} chunks")
    
    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        return {
            "size": self.size,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "is_trained": self._is_trained,
            "has_faiss": HAS_FAISS,
        }
