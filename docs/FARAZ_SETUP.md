# Faraz Setup Guide

This guide provides step-by-step instructions for setting up and running ReUnity in a fresh environment, including GitHub Codespaces.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ezernackchristopher97-cloud/ReUnity.git
cd ReUnity

# Setup (creates venv, installs dependencies)
make setup

# Activate virtual environment
source .venv/bin/activate

# Run tests to verify installation
make test

# Run demo
make demo
```

## Prerequisites

ReUnity requires Python 3.10 or higher. The following are automatically installed during setup:

| Package | Purpose |
|---------|---------|
| numpy | Numerical computations |
| fastapi | REST API backend |
| pydantic | Data validation |
| pytest | Testing framework |

Optional packages for enhanced functionality:

| Package | Purpose |
|---------|---------|
| faiss-cpu | Vector similarity search |
| sentence-transformers | Text embeddings |

## Environment Setup

### Option 1: GitHub Codespaces (Recommended)

1. Open the repository in GitHub
2. Click "Code" → "Codespaces" → "Create codespace on main"
3. Wait for the environment to initialize
4. Run setup commands in the terminal

### Option 2: Local Development

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -e .
pip install pytest
```

## Running the Application

### Demo Mode

```bash
# Quick test demo
make demo

# Interactive mode
make demo-interactive

# Or directly
python reunity_standalone.py --test
python reunity_standalone.py
```

### API Server

```bash
# Start the FastAPI server
uvicorn reunity.api.main:app --reload --host 0.0.0.0 --port 8000

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Running Tests

```bash
# All tests
make test

# Verbose output
make test-verbose

# With coverage
make test-cov
```

## Simulation Stages

ReUnity includes three simulation stages for validation:

### Stage 1: Pipeline Sanity

Tests basic state routing without RAG.

```bash
make sim-stage1
# Or: python scripts/run_sim_tests.py --stage 1
```

**What it tests:**
- Entropy state detection
- Policy routing
- State transitions

**Output:** `reports/sim_stage1_metrics.json`

### Stage 2: Pre-RAG Gates

Tests the QueryGate and EvidenceGate without full retrieval.

```bash
make sim-stage2
# Or: python scripts/run_sim_tests.py --stage 2
```

**What it tests:**
- Absurdity gap calculation
- Query validation
- Gate decisions (retrieve/clarify/refuse)

**Output:** `reports/sim_stage2_prerag.json`

### Stage 3: Full RAG

Tests complete retrieval pipeline with gating.

```bash
# Build index first
make rag-index

# Run stage 3
make sim-stage3
# Or: python scripts/run_sim_tests.py --stage 3
```

**What it tests:**
- Document chunking
- FAISS indexing
- Retrieval with Pre-RAG integration
- Evidence validation

**Output:** `reports/sim_stage3_rag.json`

### Run All Stages

```bash
make sim-all
```

## RAG System

### Building the Index

```bash
make rag-index
```

This processes documents in `data/sample_docs/` and creates a FAISS index in `data/index/`.

### Running RAG Demo

```bash
make rag-demo
```

### Custom Documents

Add your own documents to `data/sample_docs/` (Markdown or text files), then rebuild the index.

## Troubleshooting

### Import Errors

If you see import errors, ensure you've activated the virtual environment:

```bash
source .venv/bin/activate
```

### FAISS Not Available

The system works without FAISS using a numpy fallback. For better performance:

```bash
pip install faiss-cpu
```

### Test Failures

Check the test output for specific failures:

```bash
pytest -v --tb=long
```

### Codespaces Memory Issues

If Codespaces runs out of memory, try:
- Using a larger machine type
- Running stages individually instead of `sim-all`

## File Structure

```
ReUnity/
├── src/reunity/          # Main package
│   ├── core/             # Entropy analysis
│   ├── router/           # State routing
│   ├── protective/       # Pattern recognition
│   ├── memory/           # Memory store
│   ├── prerag/           # Pre-RAG gates
│   ├── rag/              # RAG system
│   └── api/              # FastAPI backend
├── data/
│   ├── sample_docs/      # Sample documents for RAG
│   ├── sim_prompts/      # Simulation test cases
│   └── index/            # FAISS index (generated)
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── reports/              # Simulation reports (generated)
└── docs/                 # Documentation
```

## Next Steps

1. Run all simulation stages to verify the system
2. Explore the API documentation at `/docs`
3. Add custom documents to the RAG system
4. Review the architecture documentation

## Support

For issues or questions, please open a GitHub issue.

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.
