"""
ReUnity RAG Module

Minimal local-first RAG implementation with:
- Document chunking
- FAISS vector indexing
- Retrieval with Pre-RAG integration

DISCLAIMER: This is not a clinical or treatment tool.
"""

from reunity.rag.chunker import DocumentChunker, Chunk
from reunity.rag.indexer import FAISSIndexer
from reunity.rag.retriever import Retriever, RetrievalResult

__all__ = [
    "DocumentChunker",
    "Chunk",
    "FAISSIndexer",
    "Retriever",
    "RetrievalResult",
]
