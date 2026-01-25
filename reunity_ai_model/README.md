# ReUnity AI Model

## For Faraz: Complete Implementation Guide

Hello Faraz,

This folder contains the complete ReUnity AI model code. This README provides step-by-step instructions to implement everything using GitHub Codespaces. No prior coding experience is required; just follow each step exactly as written.

---

## Table of Contents

1. [What is ReUnity?](#what-is-reunity)
2. [File Structure Overview](#file-structure-overview)
3. [Getting Started with GitHub Codespaces](#getting-started-with-github-codespaces)
4. [Step-by-Step Setup Instructions](#step-by-step-setup-instructions)
5. [Running the AI Model](#running-the-ai-model)
6. [Module Descriptions](#module-descriptions)
7. [Troubleshooting](#troubleshooting)
8. [Contact](#contact)

---

## What is ReUnity?

ReUnity is a trauma-aware AI support framework that uses entropy-based emotional state detection to provide adaptive, supportive interactions. The system includes:

- **Entropy Analysis**: Measures emotional state variability using Shannon entropy
- **State Router**: Selects appropriate response policies based on detected emotional state
- **Protective Pattern Recognition**: Identifies potentially harmful relationship dynamics
- **Memory Continuity**: Maintains identity continuity across fragmented states
- **Grounding Techniques**: Provides evidence-based grounding during distress
- **Alter-Aware Subsystem**: Supports individuals with DID/plural consciousness

**DISCLAIMER**: ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.

---

## File Structure Overview

```
reunity_ai_model/
│
├── README.md                 # This file (you are here)
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── reunity_standalone.py    # Complete standalone implementation (all-in-one)
├── __init__.py              # Package initialization
├── config.py                # Configuration settings
├── utils.py                 # Utility functions
│
├── core/                    # Core mathematical modules
│   ├── entropy.py           # Shannon entropy, JS divergence, Lyapunov exponents
│   └── free_energy.py       # Free Energy Principle implementation
│
├── router/                  # State routing
│   └── state_router.py      # Policy selection based on entropy state
│
├── protective/              # Protective logic
│   ├── pattern_recognizer.py # Harmful pattern detection
│   └── safety_assessment.py  # Safety evaluation
│
├── memory/                  # Memory systems
│   ├── continuity_store.py  # RIME (Recursive Identity Memory Engine)
│   └── timeline_threading.py # Timeline management
│
├── reflection/              # Reflection layer
│   └── mirror_link.py       # MirrorLink Dialogue Companion
│
├── regime/                  # Regime control
│   └── regime_controller.py # Apostasis and regeneration
│
├── grounding/               # Grounding techniques
│   └── techniques.py        # Evidence-based grounding library
│
├── alter/                   # Alter-aware subsystem
│   └── alter_aware.py       # DID/plural consciousness support
│
├── prerag/                  # Pre-RAG filtering
│   ├── query_gate.py        # Pre-retrieval filtering
│   ├── evidence_gate.py     # Post-retrieval validation
│   └── absurdity_gap.py     # Absurdity gap calculation
│
└── rag/                     # Retrieval-Augmented Generation
    ├── chunker.py           # Document chunking
    ├── indexer.py           # FAISS indexing
    └── retriever.py         # RAG retrieval with Pre-RAG integration
```

---

## Getting Started with GitHub Codespaces

GitHub Codespaces provides a complete development environment in your browser. No installation required on your computer.

### Step 1: Open the Repository in GitHub

1. Go to: `https://github.com/ezernackchristopher97-cloud/ReUnity`
2. Make sure you are logged into your GitHub account

### Step 2: Create a Codespace

1. Click the green **"Code"** button (top right of the file list)
2. Click the **"Codespaces"** tab
3. Click **"Create codespace on main"**
4. Wait 2-3 minutes for the environment to initialize
5. A VS Code editor will open in your browser

### Step 3: Open the Terminal

1. In the Codespace, look at the bottom of the screen
2. You should see a **"Terminal"** tab
3. If not visible, click **"Terminal"** in the top menu, then **"New Terminal"**

---

## Step-by-Step Setup Instructions

Copy and paste each command into the terminal, then press Enter. Wait for each command to complete before running the next one.

### Step 1: Navigate to the AI Model Folder

```bash
cd reunity_ai_model
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- numpy (numerical computations)
- scipy (scientific computing)
- pandas (data handling)
- fastapi (API framework)
- pydantic (data validation)

### Step 5: Install Optional Dependencies (Recommended)

For full functionality including vector search:

```bash
pip install torch faiss-cpu sentence-transformers
```

---

## Running the AI Model

### Option A: Run the Standalone Demo

The simplest way to test the system:

```bash
python reunity_standalone.py --test
```

This runs a complete test of all components and shows you the output.

### Option B: Run Interactive Mode

For an interactive session:

```bash
python reunity_standalone.py
```

Type messages and the system will respond based on detected emotional state.

### Option C: Start the API Server

To run the full API:

```bash
cd ..
uvicorn reunity.api.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: `http://localhost:8000/docs` to see the API documentation.

---

## Module Descriptions

### Core Modules

#### `core/entropy.py`
Implements the mathematical foundations:
- **Shannon Entropy**: Measures emotional state variability
- **Jensen-Shannon Divergence**: Detects state transitions
- **Mutual Information**: Analyzes emotion co-occurrence
- **Lyapunov Exponents**: Assesses system stability

```python
# Example usage
from core.entropy import calculate_shannon_entropy, EntropyState
import numpy as np

probs = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
entropy = calculate_shannon_entropy(probs)
print(f"Entropy: {entropy:.3f} bits")
```

#### `core/free_energy.py`
Implements the Free Energy Principle for predictive processing:
- Variational free energy calculation
- Belief updating via Bayesian inference
- Active inference for action selection

### Router Module

#### `router/state_router.py`
Selects appropriate policies based on entropy state:
- **LOW**: Reflective mode (emotional rigidity)
- **STABLE**: Full supportive interaction
- **ELEVATED**: Simplified responses with grounding
- **HIGH**: Grounding-focused responses
- **CRISIS**: Safety-focused with crisis resources

### Protective Module

#### `protective/pattern_recognizer.py`
Detects harmful relationship patterns:
- Hot-cold cycles
- Gaslighting attempts
- Love bombing
- Isolation attempts
- Reality contradictions

### Memory Module

#### `memory/continuity_store.py`
The RIME (Recursive Identity Memory Engine):
- Episodic and semantic memory storage
- Consent-scoped access controls
- Grounding memory retrieval during crisis
- Timeline threading for identity continuity

### Reflection Module

#### `reflection/mirror_link.py`
MirrorLink Dialogue Companion:
- Reflects contradictions without invalidation
- Holds multiple truths simultaneously
- Adapts communication style to user preference
- Provides grounding prompts when needed

### Regime Module

#### `regime/regime_controller.py`
Controls system behavior based on state:
- Apostasis (memory pruning during stable periods)
- Regeneration (restoration after crisis)
- Lattice memory graph management

### Grounding Module

#### `grounding/techniques.py`
Evidence-based grounding techniques:
- 5-4-3-2-1 Sensory Grounding
- Box Breathing
- Safe Place Visualization
- Cold Water Grounding
- Movement-based Grounding
- Categories Mental Game

### Alter-Aware Module

#### `alter/alter_aware.py`
Support for DID/plural consciousness:
- Individual alter recognition
- Inter-alter communication facilitation
- Shared memory with consent controls
- Switch event tracking

### Pre-RAG Modules

#### `prerag/query_gate.py`
Pre-retrieval filtering:
- Query normalization
- Absurdity gap calculation
- Decide: retrieve, clarify, or refuse

#### `prerag/evidence_gate.py`
Post-retrieval validation:
- Evidence quality assessment
- Hallucination prevention

### RAG Modules

#### `rag/retriever.py`
Retrieval-Augmented Generation:
- FAISS vector indexing
- Pre-RAG gate integration
- Semantic chunk retrieval

---

## Troubleshooting

### "ModuleNotFoundError"

Make sure you activated the virtual environment:
```bash
source venv/bin/activate
```

### "No module named 'numpy'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "FAISS not available"

The system works without FAISS using a numpy fallback. For better performance:
```bash
pip install faiss-cpu
```

### Codespace Times Out

Codespaces have a timeout. If it stops:
1. Go back to the repository
2. Click "Code" then "Codespaces"
3. Click on your existing codespace to restart it

### Need More Memory

If you run out of memory:
1. Stop the codespace
2. Create a new one with a larger machine type
3. Click the three dots next to "Create codespace"
4. Select "Configure and create codespace"
5. Choose a larger machine size

---

## Using the GitHub Copilot SDK (Optional)

If you want to integrate with GitHub Copilot:

1. Install the Copilot extension in your Codespace
2. The ReUnity modules can be used as context for Copilot suggestions
3. See `docs/COPILOT_SDK.md` in the main repository for integration details

---

## Quick Reference Commands

```bash
# Navigate to folder
cd reunity_ai_model

# Activate environment
source venv/bin/activate

# Run test
python reunity_standalone.py --test

# Run interactive
python reunity_standalone.py

# Start API server
uvicorn reunity.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Contact

For questions or issues, please contact Christopher Ezernack or open a GitHub issue.

---

**DISCLAIMER**: ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only. If you are experiencing a mental health crisis, please contact:
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/

---

*Author: Christopher Ezernack*
*Version: 1.0.0*
