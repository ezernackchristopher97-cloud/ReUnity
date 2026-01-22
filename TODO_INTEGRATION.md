# ReUnity Integration TODO

## Phase 1: Fix Broken Imports and Tests

### memory/__init__.py Import Errors
- [ ] Fix: `ContinuityMemoryStore` → `RecursiveIdentityMemoryEngine`
- [ ] Fix: Remove `MemoryFragment` (does not exist)
- [ ] Fix: Remove `MemoryThread` (does not exist)
- [ ] Add: Missing classes from continuity_store.py

### tests/test_entropy.py Import Errors
- [ ] Fix: `EntropyAnalyzer` → `EntropyStateDetector`

### tests/test_integration.py Import Errors
- [ ] Fix: `EntropyAnalyzer` → `EntropyStateDetector`

### tests/test_memory.py Import Errors
- [ ] Fix: Update imports to match actual class names

### tests/test_regime.py
- [ ] Verify imports are correct

---

## Phase 2: Verify All Tests Pass
- [ ] Run `pytest -q` and confirm all tests pass
- [ ] Run demo/entrypoint and confirm it works

---

## Phase 3: Add Pre-RAG Gates
- [ ] Create `src/reunity/prerag/` directory
- [ ] Create `absurdity_gap.py`
- [ ] Create `query_gate.py`
- [ ] Create `evidence_gate.py`
- [ ] Add config flag for Pre-RAG

---

## Phase 4: Add RAG System
- [ ] Create `scripts/rag_ingest_docs.py`
- [ ] Create `scripts/rag_build_index.py`
- [ ] Create `scripts/rag_query_demo.py`
- [ ] Create `data/sample_docs/` with sample content
- [ ] Integrate with Pre-RAG gates

---

## Phase 5: Add Datasets and Simulation Stages
- [ ] Create `data/sim_prompts/state_router_cases.jsonl`
- [ ] Create `data/sim_prompts/protection_cases.jsonl`
- [ ] Create `data/sim_prompts/rag_cases.jsonl`
- [ ] Create `scripts/data_make_sample.py`
- [ ] Create `scripts/run_sim_tests.py`

---

## Phase 6: Add Makefile Targets
- [ ] `make setup`
- [ ] `make test`
- [ ] `make demo`
- [ ] `make rag-index`
- [ ] `make rag-demo`
- [ ] `make sim-stage1`
- [ ] `make sim-stage2`
- [ ] `make sim-stage3`

---

## Phase 7: Documentation
- [ ] Create/update `docs/FARAZ_SETUP.md`
- [ ] Update `docs/ARCHITECTURE.md`
- [ ] Create `docs/RAG_SETUP.md`
- [ ] Create `docs/DATASETS_AND_SIM_TESTS.md`
- [ ] Create `docs/GPU_TRAINING.md`
- [ ] Create `docs/TROUBLESHOOTING.md`

---

## Acceptance Criteria
- [ ] Fresh setup works from scratch
- [ ] `make setup` completes
- [ ] `make test` passes
- [ ] `make demo` runs and outputs a response
- [ ] Pre-RAG is present behind a flag
- [ ] `make sim-stage1` passes
- [ ] `make sim-stage2` passes
- [ ] `make rag-index` and `make sim-stage3` pass
- [ ] Documentation matches real commands and paths
- [ ] No keys, PHI, or large weights in git
