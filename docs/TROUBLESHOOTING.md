# Troubleshooting Guide

This document provides solutions for common issues when setting up and running ReUnity.

## Setup Issues

### Virtual Environment Creation Fails

**Error:**
```
python -m venv .venv
Error: Command 'python' not found
```

**Solution:**
```bash
# Try python3 instead
python3 -m venv .venv

# Or install Python if missing
sudo apt update && sudo apt install python3.11 python3.11-venv
```

### Package Installation Fails

**Error:**
```
pip install -e .
ERROR: Could not find a version that satisfies the requirement...
```

**Solutions:**

1. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

2. Install from requirements.txt first:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. Check Python version (requires 3.9+):
   ```bash
   python --version
   ```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'reunity'
```

**Solutions:**

1. Ensure package is installed:
   ```bash
   pip install -e .
   ```

2. Check PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. Verify installation:
   ```bash
   pip show reunity
   ```

## Test Failures

### Pytest Not Found

**Error:**
```
pytest: command not found
```

**Solution:**
```bash
pip install pytest pytest-cov
```

### Test Import Errors

**Error:**
```
ImportError: cannot import name 'EntropyStateDetector' from 'reunity.core.entropy'
```

**Solution:**
Ensure you're running tests from the repo root:
```bash
cd /path/to/ReUnity
pytest -v
```

### Entropy Test Failures

**Error:**
```
AssertionError: Expected state 'stable' but got 'elevated'
```

**Explanation:**
Entropy state detection depends on the distribution of input data. The thresholds may need tuning for your specific use case.

**Solution:**
Check the entropy thresholds in `src/reunity/core/entropy.py` and adjust if needed.

## RAG Issues

### FAISS Not Installed

**Warning:**
```
FAISS not installed, using numpy-based fallback index
```

**Explanation:**
This is expected in CPU-only environments. The numpy fallback works but is slower for large datasets.

**Solution (optional):**
```bash
# For CPU-only FAISS
pip install faiss-cpu

# For GPU FAISS (requires CUDA)
pip install faiss-gpu
```

### Index Build Fails

**Error:**
```
FileNotFoundError: data/sample_docs/ not found
```

**Solution:**
Ensure sample documents exist:
```bash
ls data/sample_docs/
# Should show .md files
```

### Retrieval Returns Empty

**Issue:**
Retrieval returns no chunks for valid queries.

**Solutions:**

1. Rebuild the index:
   ```bash
   make rag-index
   ```

2. Check chunk size settings:
   ```python
   chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
   ```

3. Verify documents were indexed:
   ```bash
   cat data/index/config.json
   ```

## Simulation Test Issues

### Stage 1 Failures

**Error:**
```
Results: 15/20 passed (75.0%)
```

**Explanation:**
State detection is based on keyword matching and entropy analysis. Some edge cases may not match expected states.

**Solutions:**

1. Review the test cases in `data/sim_prompts/state_router_cases.jsonl`
2. Adjust keyword lists in the simulation script
3. Tune entropy thresholds

### Stage 2 Pre-RAG Failures

**Error:**
```
absurdity_gap is NaN
```

**Solution:**
Ensure anchor documents are loaded:
```python
calculator = AbsurdityGapCalculator(use_embeddings=False)
for doc_path in sample_docs_dir.glob("*.md"):
    calculator.add_anchor(doc_path.read_text())
```

### Stage 3 RAG Failures

**Error:**
```
Index not found at data/index/
```

**Solution:**
Build the index first:
```bash
make rag-index
# Then run stage 3
make sim-stage3
```

## API Issues

### FastAPI Won't Start

**Error:**
```
ModuleNotFoundError: No module named 'uvicorn'
```

**Solution:**
```bash
pip install uvicorn[standard]
```

### Port Already in Use

**Error:**
```
ERROR: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find and kill the process
lsof -i :8000
kill -9 <PID>

# Or use a different port
uvicorn reunity.api.main:app --port 8001
```

### CORS Errors

**Error:**
```
Access to fetch at 'http://localhost:8000' has been blocked by CORS policy
```

**Solution:**
CORS is already configured in the API. If you need to add more origins:
```python
# In src/reunity/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://your-domain.com"],
    ...
)
```

## Memory Issues

### Out of Memory During Indexing

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. Reduce batch size:
   ```python
   indexer = FAISSIndexer(embedding_dim=384, batch_size=16)
   ```

2. Use smaller chunk sizes:
   ```python
   chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
   ```

3. Process documents in batches:
   ```python
   for batch in chunks_batched(chunks, batch_size=100):
       indexer.add_batch(batch, embeddings)
   ```

## Codespaces-Specific Issues

### Slow Performance

**Explanation:**
Codespaces may have limited CPU resources.

**Solutions:**

1. Use mini mode for tests:
   ```bash
   python scripts/run_sim_tests.py --stage 1 --mini
   ```

2. Reduce dataset size
3. Use numpy fallback instead of FAISS

### Disk Space

**Error:**
```
No space left on device
```

**Solutions:**

1. Clean up caches:
   ```bash
   pip cache purge
   rm -rf .pytest_cache __pycache__
   ```

2. Remove old indexes:
   ```bash
   rm -rf data/index/
   ```

## Getting Help

If you encounter issues not covered here:

1. Check the GitHub Issues page
2. Review the documentation in `docs/`
3. Run with debug logging:
   ```bash
   REUNITY_DEBUG=true python scripts/run_sim_tests.py --stage 1
   ```

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.
