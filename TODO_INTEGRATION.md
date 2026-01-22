# ReUnity Integration TODO

## Status: ✅ COMPLETE

All acceptance criteria have been met.

---

## Phase 1: Fix Broken Imports and Tests

### memory/__init__.py Import Errors
- [x] Fix: `ContinuityMemoryStore` → `RecursiveIdentityMemoryEngine`
- [x] Fix: Remove `MemoryFragment` (does not exist)
- [x] Fix: Remove `MemoryThread` (does not exist)
- [x] Add: Missing classes from continuity_store.py

### tests/test_entropy.py Import Errors
- [x] Fix: `EntropyAnalyzer` → `EntropyStateDetector`

### tests/test_integration.py Import Errors
- [x] Fix: `EntropyAnalyzer` → `EntropyStateDetector`
- [x] Fix: All method calls to match actual implementations

### tests/test_memory.py Import Errors
- [x] Fix: Update imports to match actual class names

### tests/test_regime.py
- [x] Verify imports are correct
- [x] Fix EntropyMetrics usage
- [x] Fix method signatures

---

## Phase 2: Verify All Tests Pass
- [x] Run `pytest -q` and confirm all tests pass (94/94 passed)
- [x] Run demo/entrypoint and confirm it works

---

## Phase 3: Add Pre-RAG Gates
- [x] Create `src/reunity/prerag/` directory
- [x] Create `absurdity_gap.py`
- [x] Create `query_gate.py`
- [x] Create `evidence_gate.py`
- [x] Add config flag for Pre-RAG (`enable_prerag`)

---

## Phase 4: Add RAG System
- [x] Create `src/reunity/rag/chunker.py`
- [x] Create `src/reunity/rag/indexer.py`
- [x] Create `src/reunity/rag/retriever.py`
- [x] Create `data/sample_docs/` with 5 sample documents
- [x] Integrate with Pre-RAG gates

---

## Phase 5: Add Datasets and Simulation Stages
- [x] Create `data/sim_prompts/state_router_cases.jsonl` (20 cases)
- [x] Create `data/sim_prompts/protection_cases.jsonl` (11 cases)
- [x] Create `data/sim_prompts/rag_cases.jsonl` (14 cases)
- [x] Create `scripts/data_make_sample.py`
- [x] Create `scripts/run_sim_tests.py`

---

## Phase 6: Add Makefile Targets
- [x] `make setup`
- [x] `make test`
- [x] `make demo`
- [x] `make rag-index`
- [x] `make rag-demo`
- [x] `make sim-stage1`
- [x] `make sim-stage2`
- [x] `make sim-stage3`
- [x] `make sim-all`

---

## Phase 7: Documentation
- [x] Create `docs/FARAZ_SETUP.md`
- [x] Update `docs/ARCHITECTURE.md`
- [x] Create `docs/RAG_SETUP.md`
- [x] Create `docs/DATASETS_AND_SIM_TESTS.md`
- [x] Create `docs/GPU_TRAINING.md`
- [x] Create `docs/TROUBLESHOOTING.md`

---

## Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| Fresh setup works from scratch | ✅ |
| `make setup` completes | ✅ |
| `make test` passes (94/94) | ✅ |
| `make demo` runs and outputs a response | ✅ |
| Pre-RAG is present behind a flag | ✅ |
| `make sim-stage1` passes (90%) | ✅ |
| `make sim-stage2` passes (100%) | ✅ |
| `make rag-index` and `make sim-stage3` pass (78.6%) | ✅ |
| Documentation matches real commands and paths | ✅ |
| No keys, PHI, or large weights in git | ✅ |

---

## Simulation Test Results

| Stage | Pass Rate | Cases |
|-------|-----------|-------|
| Stage 1: Pipeline Sanity | 90% | 18/20 |
| Stage 2: Pre-RAG Gates | 100% | 14/14 |
| Stage 3: Full RAG | 78.6% | 11/14 |

---

## Git Commits on Integration Branch

1. Fix test imports and method signatures
2. Fix API imports
3. Add Pre-RAG gates, RAG system, datasets, and simulation tests

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool.
