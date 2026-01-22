# RAG Setup Guide

This document describes how to set up and use the Retrieval-Augmented Generation (RAG) system in ReUnity.

## Overview

ReUnity's RAG system provides context-aware retrieval with a two-layer Pre-RAG filter that validates queries and evidence before generating responses. The system is designed to be CPU-safe and works in resource-constrained environments like GitHub Codespaces.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Query ──► QueryGate ──► Retriever ──► EvidenceGate ──► Response   │
│              (Layer 1)                    (Layer 2)                  │
│                                                                      │
│   QueryGate:                                                         │
│   - Normalizes query                                                 │
│   - Computes absurdity gap (prior)                                  │
│   - Decides: RETRIEVE / CLARIFY / NO_RETRIEVE                       │
│                                                                      │
│   EvidenceGate:                                                      │
│   - Scores retrieved chunks                                          │
│   - Computes absurdity gap (posterior)                              │
│   - Decides: ANSWER / CLARIFY / REFUSE                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### Document Chunker

Splits documents into overlapping chunks for indexing.

```python
from reunity.rag.chunker import DocumentChunker

chunker = DocumentChunker(
    chunk_size=500,      # Target chunk size in characters
    chunk_overlap=50,    # Overlap between chunks
    strategy="fixed",    # "fixed", "sentence", or "paragraph"
)

# Chunk a single file
chunks = chunker.chunk_file("document.md")

# Chunk a directory
chunks = chunker.chunk_directory("data/sample_docs/", extensions=[".md", ".txt"])
```

### FAISS Indexer

Vector index for similarity search. Falls back to numpy if FAISS is not installed.

```python
from reunity.rag.indexer import FAISSIndexer

indexer = FAISSIndexer(
    embedding_dim=384,           # Embedding dimension
    index_type="flat",           # "flat", "ivf", or "hnsw"
    normalize_embeddings=True,   # L2 normalize embeddings
)

# Add chunks with embeddings
indexer.add_batch(chunks, embeddings)

# Save/load index
indexer.save("data/index")
indexer.load("data/index")

# Search
results = indexer.search(query_embedding, k=5)
```

### Retriever

Combines indexing with Pre-RAG gates.

```python
from reunity.rag.retriever import Retriever
from reunity.prerag.query_gate import QueryGate
from reunity.prerag.evidence_gate import EvidenceGate

retriever = Retriever(
    indexer=indexer,
    query_gate=QueryGate(),
    evidence_gate=EvidenceGate(),
    embed_fn=your_embedding_function,
    top_k=5,
    enable_prerag=True,
)

result = retriever.retrieve("What is grounding?")
if result.should_answer:
    # Use result.chunks for context
    pass
elif result.clarification_needed:
    # Ask for clarification
    print(result.clarification_message)
```

## Quick Start

### 1. Build the Index

```bash
make rag-index
```

Or programmatically:

```python
from pathlib import Path
from reunity.rag.chunker import DocumentChunker
from reunity.rag.indexer import FAISSIndexer
import numpy as np

# Chunk documents
chunker = DocumentChunker(chunk_size=300)
chunks = chunker.chunk_directory(Path("data/sample_docs"))

# Create embeddings (simple example)
def embed(text):
    dim = 128
    emb = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for i in range(len(text) - 2):
        idx = hash(text[i:i+3]) % dim
        emb[idx] += 1.0
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

embeddings = [embed(chunk.text) for chunk in chunks]

# Build index
indexer = FAISSIndexer(embedding_dim=128)
indexer.add_batch(chunks, embeddings)
indexer.save(Path("data/index"))
```

### 2. Query the Index

```bash
make rag-demo
```

Or programmatically:

```python
from pathlib import Path
from reunity.rag.indexer import FAISSIndexer
from reunity.rag.retriever import Retriever

# Load index
indexer = FAISSIndexer(embedding_dim=128)
indexer.load(Path("data/index"))

# Create retriever
retriever = Retriever(
    indexer=indexer,
    embed_fn=embed,  # Same embedding function used for indexing
    enable_prerag=False,  # Disable gates for simple retrieval
)

# Query
result = retriever.retrieve("What is the 5-4-3-2-1 technique?")
for chunk in result.chunks:
    print(f"Score: {result.scores[result.chunks.index(chunk)]:.3f}")
    print(f"Text: {chunk.text[:200]}...")
    print()
```

## Pre-RAG Gates

### Absurdity Gap

The absurdity gap is a scalar score (0.0 to 1.0) that estimates how far a query is from being answerable with the available knowledge.

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.3 | Well-grounded, proceed with retrieval |
| 0.3 - 0.7 | Moderate uncertainty, retrieve with caution |
| 0.7 - 1.0 | High uncertainty, likely needs clarification |

### QueryGate (Layer 1)

Pre-retrieval filter that decides whether to proceed with retrieval.

```python
from reunity.prerag.query_gate import QueryGate, QueryGateAction

gate = QueryGate(
    retrieve_threshold=0.4,   # Below this, proceed with retrieval
    clarify_threshold=0.7,    # Above this, ask for clarification
)

decision = gate.process(query="What is grounding?")
if decision.action == QueryGateAction.RETRIEVE:
    # Proceed with retrieval
    pass
elif decision.action == QueryGateAction.CLARIFY:
    # Ask for clarification
    print(decision.suggestions)
```

### EvidenceGate (Layer 2)

Post-retrieval filter that validates retrieved evidence.

```python
from reunity.prerag.evidence_gate import EvidenceGate, EvidenceGateAction

gate = EvidenceGate(
    answer_threshold=0.3,    # Below this, proceed to answer
    refuse_threshold=0.85,   # Above this, refuse to answer
)

decision = gate.process(
    query=query,
    query_embedding=query_emb,
    retrieved_chunks=chunks,
)
if decision.action == EvidenceGateAction.ANSWER:
    # Generate response using decision.selected_chunks
    pass
```

## Adding Custom Documents

1. Add documents to `data/sample_docs/` (Markdown, text, or RST files)
2. Rebuild the index: `make rag-index`
3. Test with: `make rag-demo`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REUNITY_RAG_ENABLED` | `true` | Enable/disable RAG |
| `REUNITY_PRERAG_ENABLED` | `true` | Enable/disable Pre-RAG gates |
| `REUNITY_RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `REUNITY_DEBUG` | `false` | Enable debug logging |

### Tuning Parameters

For better retrieval quality:

1. **Chunk size**: Smaller chunks (200-400 chars) for precise retrieval, larger (500-800) for more context
2. **Overlap**: 10-20% of chunk size prevents information loss at boundaries
3. **Top-k**: Start with 3-5, increase if answers seem incomplete
4. **Thresholds**: Adjust based on your corpus and query patterns

## Troubleshooting

### No Results Returned

- Check that the index was built successfully
- Verify documents exist in `data/sample_docs/`
- Try lowering the `min_relevance` threshold

### Poor Relevance

- Increase chunk overlap
- Try different chunking strategies
- Consider using better embeddings (sentence-transformers)

### Memory Issues

- Use smaller chunk sizes
- Process documents in batches
- Use IVF index type for large corpora

## Advanced Usage

### Using Sentence Transformers

For better embeddings, install sentence-transformers:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(text):
    return model.encode(text, normalize_embeddings=True)

# Use with indexer (embedding_dim=384 for this model)
indexer = FAISSIndexer(embedding_dim=384)
```

### Custom Embedding Functions

Any function that takes a string and returns a numpy array can be used:

```python
def custom_embed(text: str) -> np.ndarray:
    # Your embedding logic here
    return embedding_vector
```

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.
