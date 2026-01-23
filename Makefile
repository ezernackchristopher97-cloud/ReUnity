# ReUnity Makefile
# Provides one-command runs for setup, testing, demos, and simulations
#
# DISCLAIMER: This is not a clinical or treatment tool.

.PHONY: all setup test demo clean help
.PHONY: sim-stage1 sim-stage2 sim-stage3 sim-all
.PHONY: rag-index rag-demo
.PHONY: lint format typecheck

# Default target
all: help

# ============================================================================
# Setup
# ============================================================================

setup:
	@echo "Setting up ReUnity..."
	python3 -m venv .venv || true
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e .
	. .venv/bin/activate && pip install pytest pytest-cov black isort mypy
	@echo "Setup complete. Activate with: source .venv/bin/activate"

setup-dev: setup
	. .venv/bin/activate && pip install faiss-cpu sentence-transformers

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "Running tests..."
	. .venv/bin/activate && pytest -q

test-verbose:
	@echo "Running tests with verbose output..."
	. .venv/bin/activate && pytest -v

test-cov:
	@echo "Running tests with coverage..."
	. .venv/bin/activate && pytest --cov=reunity --cov-report=html

# ============================================================================
# Demo
# ============================================================================

demo:
	@echo "Running ReUnity demo..."
	. .venv/bin/activate && python reunity_standalone.py --test

demo-interactive:
	@echo "Running ReUnity interactive demo..."
	. .venv/bin/activate && python reunity_standalone.py

# ============================================================================
# Simulation Stages
# ============================================================================

sim-stage1:
	@echo "Running Stage 1: Pipeline Sanity Test..."
	. .venv/bin/activate && python scripts/run_sim_tests.py --stage 1

sim-stage2:
	@echo "Running Stage 2: Pre-RAG Gates Test..."
	. .venv/bin/activate && python scripts/run_sim_tests.py --stage 2

sim-stage3: rag-index
	@echo "Running Stage 3: Full RAG Test..."
	. .venv/bin/activate && python scripts/run_sim_tests.py --stage 3

sim-all: sim-stage1 sim-stage2 sim-stage3
	@echo "All simulation stages complete."

# ============================================================================
# RAG
# ============================================================================

rag-index:
	@echo "Building RAG index..."
	@mkdir -p data/index
	. .venv/bin/activate && python -c "\
from pathlib import Path; \
from reunity.rag.chunker import DocumentChunker; \
from reunity.rag.indexer import FAISSIndexer; \
import numpy as np; \
chunker = DocumentChunker(chunk_size=300); \
chunks = chunker.chunk_directory(Path('data/sample_docs')); \
print(f'Chunked {len(chunks)} chunks'); \
indexer = FAISSIndexer(embedding_dim=128); \
def embed(t): e=np.zeros(128,dtype=np.float32); [e.__setitem__(hash(t[i:i+3])%128, e[hash(t[i:i+3])%128]+1) for i in range(len(t)-2)]; n=np.linalg.norm(e); return e/n if n>0 else e; \
embeddings = [embed(c.text.lower()) for c in chunks]; \
indexer.add_batch(chunks, embeddings); \
indexer.save(Path('data/index')); \
print('Index saved to data/index')"

rag-demo:
	@echo "Running RAG demo..."
	. .venv/bin/activate && python -c "\
from pathlib import Path; \
from reunity.rag.indexer import FAISSIndexer; \
from reunity.rag.retriever import Retriever; \
import numpy as np; \
indexer = FAISSIndexer(embedding_dim=128); \
indexer.load(Path('data/index')); \
print(f'Loaded index with {indexer.size} chunks'); \
def embed(t): e=np.zeros(128,dtype=np.float32); [e.__setitem__(hash(t[i:i+3])%128, e[hash(t[i:i+3])%128]+1) for i in range(len(t)-2)]; n=np.linalg.norm(e); return e/n if n>0 else e; \
retriever = Retriever(indexer=indexer, embed_fn=embed, enable_prerag=False); \
result = retriever.retrieve('What is the 5-4-3-2-1 grounding technique?'); \
print(f'Query: What is the 5-4-3-2-1 grounding technique?'); \
print(f'Retrieved {len(result.chunks)} chunks'); \
for i, chunk in enumerate(result.chunks[:2]): print(f'Chunk {i+1}: {chunk.text[:200]}...')"

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "Running linter..."
	. .venv/bin/activate && black --check src/ tests/ scripts/
	. .venv/bin/activate && isort --check-only src/ tests/ scripts/

format:
	@echo "Formatting code..."
	. .venv/bin/activate && black src/ tests/ scripts/
	. .venv/bin/activate && isort src/ tests/ scripts/

typecheck:
	@echo "Running type checker..."
	. .venv/bin/activate && mypy src/reunity --ignore-missing-imports

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf __pycache__ .pytest_cache .mypy_cache
	rm -rf htmlcov .coverage
	rm -rf data/index
	rm -rf reports/*.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-index:
	@echo "Cleaning RAG index..."
	rm -rf data/index

# ============================================================================
# Help
# ============================================================================

help:
	@echo "ReUnity Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create venv and install dependencies"
	@echo "  make setup-dev      - Setup with optional dev dependencies (FAISS)"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run unit tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo ""
	@echo "Demo:"
	@echo "  make demo           - Run test demo"
	@echo "  make demo-interactive - Run interactive demo"
	@echo ""
	@echo "Simulation Stages:"
	@echo "  make sim-stage1     - Pipeline sanity test (no RAG)"
	@echo "  make sim-stage2     - Pre-RAG gates test"
	@echo "  make sim-stage3     - Full RAG test"
	@echo "  make sim-all        - Run all simulation stages"
	@echo ""
	@echo "RAG:"
	@echo "  make rag-index      - Build RAG index from sample docs"
	@echo "  make rag-demo       - Run RAG retrieval demo"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Check code formatting"
	@echo "  make format         - Auto-format code"
	@echo "  make typecheck      - Run type checker"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove all generated files"
	@echo "  make clean-index    - Remove RAG index only"

# Real data simulations
sim-real:
@echo "Running real data simulations with GoEmotions dataset..."
python scripts/run_real_simulations.py

sim-download-data:
@echo "Downloading GoEmotions dataset..."
mkdir -p data/goemotions
cd data/goemotions && \
wget -q https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv && \
wget -q https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/dev.tsv && \
wget -q https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/test.tsv && \
wget -q https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt
@echo "Downloaded GoEmotions dataset (54k+ comments, 27 emotions)"

sim-all-real: sim-download-data sim-real
@echo "All real data simulations complete!"
