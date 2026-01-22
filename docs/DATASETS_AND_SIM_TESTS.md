# Datasets and Simulation Tests

This document describes the datasets and simulation test stages used to validate ReUnity's functionality.

## Overview

ReUnity uses a three-stage simulation testing approach to validate the system from basic pipeline functionality through full RAG integration. Each stage builds on the previous one, ensuring that components work correctly before adding complexity.

## Datasets

### Sample Documents

Located in `data/sample_docs/`, these documents provide the knowledge base for RAG retrieval.

| Document | Content |
|----------|---------|
| `01_understanding_emotions.md` | Emotional types, regulation, window of tolerance |
| `02_grounding_techniques.md` | 5-4-3-2-1 technique, physical/mental grounding |
| `03_healthy_relationships.md` | Relationship foundations, warning signs |
| `04_self_care.md` | Physical, emotional, social, mental self-care |
| `05_coping_strategies.md` | Adaptive vs maladaptive coping, crisis coping |

### Simulation Prompts

Located in `data/sim_prompts/`, these JSONL files contain test cases for each simulation stage.

#### State Router Cases (`state_router_cases.jsonl`)

Tests for entropy state detection and policy routing.

```json
{
  "input": "I'm feeling pretty good today",
  "expected_state": "stable",
  "expected_policy": "reflective",
  "must_not": ["crisis_response"]
}
```

| Field | Description |
|-------|-------------|
| `input` | User input text |
| `expected_state` | Expected entropy state (stable/elevated/crisis/recovery) |
| `expected_policy` | Expected policy type |
| `must_not` | Actions that should NOT be taken |

#### Protection Cases (`protection_cases.jsonl`)

Tests for harmful pattern detection.

```json
{
  "input": "My partner says I'm crazy for thinking they were flirting",
  "expected_pattern": "gaslighting",
  "severity": "high",
  "must_detect": true
}
```

| Field | Description |
|-------|-------------|
| `input` | User input describing situation |
| `expected_pattern` | Pattern that should be detected |
| `severity` | Expected severity level |
| `must_detect` | Whether pattern must be detected (true) or must NOT be detected (false) |

#### RAG Cases (`rag_cases.jsonl`)

Tests for retrieval and Pre-RAG gating.

```json
{
  "input": "What is the 5-4-3-2-1 grounding technique?",
  "expected_action": "retrieve",
  "expected_chunks_min": 1,
  "topic": "grounding"
}
```

| Field | Description |
|-------|-------------|
| `input` | Query text |
| `expected_action` | Expected action (retrieve/clarify/refuse) |
| `expected_chunks_min` | Minimum chunks that should be retrieved |
| `topic` | Topic category for analysis |

## Simulation Stages

### Stage 1: Pipeline Sanity

**Purpose:** Verify that the basic state routing pipeline works without RAG.

**Command:**
```bash
make sim-stage1
# Or: python scripts/run_sim_tests.py --stage 1
```

**What it tests:**
1. Text-to-distribution conversion
2. Entropy state detection
3. Policy routing based on state
4. State transition logic

**Test cases:** `data/sim_prompts/state_router_cases.jsonl`

**Output:** `reports/sim_stage1_metrics.json`

**Example output:**
```json
{
  "total": 20,
  "passed": 18,
  "failed": 2,
  "pass_rate": 0.9,
  "elapsed_seconds": 0.15,
  "details": [...]
}
```

**Pass criteria:**
- State detection matches expected state category
- No `must_not` violations

### Stage 2: Pre-RAG Gates

**Purpose:** Verify that QueryGate and EvidenceGate produce valid absurdity gap scores.

**Command:**
```bash
make sim-stage2
# Or: python scripts/run_sim_tests.py --stage 2
```

**What it tests:**
1. Absurdity gap calculation
2. Query normalization
3. Gate decision logic (retrieve/clarify/no_retrieve)
4. Anchor-based similarity

**Test cases:** `data/sim_prompts/rag_cases.jsonl`

**Output:** `reports/sim_stage2_prerag.json`

**Example output:**
```json
{
  "total": 14,
  "passed": 12,
  "failed": 2,
  "pass_rate": 0.857,
  "mean_gap": 0.45,
  "absurdity_gaps": [0.2, 0.3, ...],
  "details": [...]
}
```

**Pass criteria:**
- Absurdity gap is numeric and bounded [0.0, 1.0]
- High-gap queries trigger clarify/refuse
- Low-gap queries proceed to retrieve

### Stage 3: Full RAG

**Purpose:** Verify complete retrieval pipeline with gating.

**Command:**
```bash
make sim-stage3
# Or: python scripts/run_sim_tests.py --stage 3
```

**Prerequisites:**
- Index must be built first: `make rag-index`

**What it tests:**
1. Document chunking
2. FAISS indexing
3. Retrieval with embedding similarity
4. Pre-RAG gate integration
5. Evidence validation

**Test cases:** `data/sim_prompts/rag_cases.jsonl`

**Output:** `reports/sim_stage3_rag.json`

**Example output:**
```json
{
  "total": 14,
  "passed": 11,
  "failed": 3,
  "pass_rate": 0.786,
  "elapsed_seconds": 2.3,
  "details": [
    {
      "input": "What is the 5-4-3-2-1 grounding technique?",
      "expected_action": "retrieve",
      "actual_action": "answer",
      "num_chunks": 3,
      "prior_gap": 0.25,
      "posterior_gap": 0.18,
      "passed": true
    }
  ]
}
```

**Pass criteria:**
- Retrieved chunk count meets minimum
- Action matches expected (with flexibility for borderline cases)
- Prior and posterior gaps are valid

### Running All Stages

```bash
make sim-all
```

This runs stages 1, 2, and 3 in sequence, stopping if any stage fails.

## Adding Custom Test Cases

### Format

All test case files use JSONL format (one JSON object per line).

### State Router Cases

```json
{"input": "Your test input", "expected_state": "stable|elevated|crisis|recovery", "expected_policy": "policy_name", "must_not": ["action1", "action2"]}
```

### Protection Cases

```json
{"input": "Your test input", "expected_pattern": "pattern_name|none", "severity": "high|medium|low|none", "must_detect": true|false}
```

### RAG Cases

```json
{"input": "Your query", "expected_action": "retrieve|clarify|refuse", "expected_chunks_min": 0, "topic": "topic_name"}
```

## Interpreting Results

### Pass Rate Guidelines

| Pass Rate | Interpretation |
|-----------|----------------|
| > 90% | Excellent - system is working well |
| 80-90% | Good - minor issues to investigate |
| 70-80% | Acceptable - some edge cases failing |
| < 70% | Needs attention - significant issues |

### Common Failure Patterns

1. **State detection mismatch**: Entropy thresholds may need tuning
2. **High absurdity gap for valid queries**: Add more anchors to knowledge base
3. **Low retrieval count**: Adjust chunk size or top-k parameter
4. **Pattern not detected**: Add pattern variants to recognizer

### Debugging Tips

1. Run with verbose output:
   ```bash
   python scripts/run_sim_tests.py --stage 1 2>&1 | tee debug.log
   ```

2. Check individual test details in the JSON reports

3. Add debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

## Extending the Test Suite

### Adding New Test Categories

1. Create a new JSONL file in `data/sim_prompts/`
2. Add a new function in `scripts/run_sim_tests.py`
3. Add a new Makefile target

### Adding New Documents

1. Add documents to `data/sample_docs/`
2. Rebuild index: `make rag-index`
3. Update test cases to cover new content

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.
