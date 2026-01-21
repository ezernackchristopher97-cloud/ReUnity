# ReUnity

## A Trauma-Aware AI Extension for Unshackled

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

---

## ⚠️ IMPORTANT DISCLAIMER

**ReUnity is NOT a clinical or treatment tool.** It is a theoretical and support framework only. ReUnity is not intended to diagnose, treat, cure, or prevent any medical or psychological condition.

This system is designed to provide supportive tools for individuals working with mental health professionals. It should be used as a complement to, not a replacement for, professional mental health care.

**If you are experiencing a mental health crisis, please contact:**
- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741 (US)
- **International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

Always consult with qualified mental health professionals for diagnosis and treatment of mental health conditions.

---

## Overview

ReUnity is a trauma-aware AI support system that implements entropy-based state detection, protective pattern recognition, continuity memory, and reflection capabilities. It is designed to provide supportive tools for individuals navigating complex emotional and relational dynamics, particularly those with trauma histories.

### Key Features

- **Entropy-Based State Detection**: Uses Shannon entropy, Jensen-Shannon divergence, and Lyapunov exponents to detect emotional/cognitive states
- **Protective Pattern Recognition**: Identifies potentially harmful relationship dynamics (gaslighting, hot-cold cycles, isolation attempts)
- **Continuity Memory (RIME)**: Recursive Identity Memory Engine for maintaining identity continuity across fragmented states
- **MirrorLink Reflection**: Surfaces contradictions without invalidation, helping users hold multiple truths
- **Regime Logic**: Adaptive behavior based on entropy bands with apostasis (pruning) and regeneration
- **Consent-Scoped Access**: Fine-grained privacy controls for all stored data
- **Local-First Architecture**: Encrypted storage with user data sovereignty

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ReUnity System                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Entropy   │  │    State    │  │      Protective         │ │
│  │  Analyzer   │──│   Router    │──│   Pattern Recognizer    │ │
│  │  (H, JS, λ) │  │  (Policies) │  │   (PLM)                 │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Regime Controller                        ││
│  │  ┌──────────┐  ┌─────────────┐  ┌────────────────────────┐ ││
│  │  │Apostasis │  │Regeneration │  │   Lattice Memory Graph │ ││
│  │  │(Pruning) │  │ (Restore)   │  │   (Divergence-Bounded) │ ││
│  │  └──────────┘  └─────────────┘  └────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                │                      │               │
│         ▼                ▼                      ▼               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Continuity Memory Store (RIME)                 ││
│  │  ┌──────────┐  ┌─────────────┐  ┌────────────────────────┐ ││
│  │  │ Episodic │  │  Semantic   │  │     Consent Scopes     │ ││
│  │  │ Memory   │  │  Patterns   │  │  (Private→Emergency)   │ ││
│  │  └──────────┘  └─────────────┘  └────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│         │                                       │               │
│         ▼                                       ▼               │
│  ┌─────────────────┐                  ┌─────────────────────┐  │
│  │    MirrorLink   │                  │  Encrypted Storage  │  │
│  │    Reflection   │                  │   (AES-256-GCM)     │  │
│  └─────────────────┘                  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundations

### Shannon Entropy

```
H(X) = -Σ p(x) log₂ p(x)
```

Measures the uncertainty or information content in a probability distribution.

### Jensen-Shannon Divergence

```
JS(P||Q) = ½ KL(P||M) + ½ KL(Q||M)
```

Where M = ½(P + Q). Measures the similarity between two probability distributions.

### Mutual Information

```
I(X;Y) = Σ p(x,y) log₂(p(x,y) / (p(x)p(y)))
```

Measures the mutual dependence between two variables.

### RIME Formula

```
RIME(t) = α · M_episodic(t) + β · M_semantic(t) + γ · C_context(t)
```

Where:
- M_episodic = episodic memory activation
- M_semantic = semantic memory patterns
- C_context = current contextual factors
- α, β, γ = dynamically adjusted weights

### Lyapunov Exponent

```
λ = lim(n→∞) (1/n) Σ log|f'(xᵢ)|
```

Measures the rate of separation of infinitesimally close trajectories (stability).

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/reunity.git
cd reunity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t reunity .
docker run -p 8000:8000 reunity
```

---

## Quick Start

### Running the API Server

```bash
# Start the FastAPI server
uvicorn reunity.api.main:app --reload

# Or use the module directly
python -m reunity.api.main
```

The API will be available at `http://localhost:8000`. Access the interactive documentation at `http://localhost:8000/docs`.

### Basic Usage

```python
from reunity import (
    EntropyAnalyzer,
    StateRouter,
    ProtectivePatternRecognizer,
    RecursiveIdentityMemoryEngine,
    MirrorLinkDialogueCompanion,
)
import numpy as np

# Initialize components
analyzer = EntropyAnalyzer()
router = StateRouter()
recognizer = ProtectivePatternRecognizer()
memory = RecursiveIdentityMemoryEngine()
companion = MirrorLinkDialogueCompanion()

# Analyze entropy state
distribution = np.array([0.3, 0.3, 0.2, 0.2])
metrics = analyzer.analyze(distribution)
print(f"State: {metrics.state.value}, Confidence: {metrics.confidence}")

# Get policy for current state
policy = router.route(metrics)
print(f"Policy: {policy.policy_type.value}")

# Add a memory
mem = memory.add_memory(
    identity="primary",
    content="Felt safe at the beach today",
    tags=["safe", "grounding"],
)

# Generate a reflection
reflection = companion.reflect(
    current_emotion="feeling anxious",
    past_context="you felt calm yesterday",
)
print(reflection.content)
```

---

## API Endpoints

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check with disclaimer |
| `/health` | GET | Health status |
| `/disclaimer` | GET | Full disclaimer text |

### Entropy Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/entropy/analyze` | POST | Analyze text for entropy state |
| `/entropy/states` | GET | List available entropy states |

### Memory Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/memory/add` | POST | Add a memory |
| `/memory/retrieve` | POST | Retrieve memories with grounding support |
| `/memory/consent` | PUT | Update consent scope |
| `/memory/stats` | GET | Get memory statistics |

### Pattern Recognition

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/patterns/analyze` | POST | Analyze interactions for harmful patterns |

### Reflection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reflection/generate` | POST | Generate a reflection |

### Journal

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/journal/add` | POST | Add journal entry |
| `/journal/list` | GET | List journal entries |

### Export

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/export/bundle` | POST | Export data bundle with provenance |
| `/export/timeline` | GET | Export timeline events |

---

## Consent Scopes

ReUnity implements fine-grained consent controls:

| Scope | Description |
|-------|-------------|
| `private` | Only the user can access |
| `self_only` | User and their alters (for DID support) |
| `therapist` | Shared with designated therapist |
| `caregiver` | Shared with designated caregiver |
| `emergency` | Accessible in crisis situations |
| `research` | Anonymized for research (with consent) |

---

## Entropy States

| State | Description | System Response |
|-------|-------------|-----------------|
| `low` | Highly predictable, possibly rigid | Monitor for flexibility |
| `stable` | Healthy variability | Normal operation |
| `elevated` | Increased uncertainty | Enhanced monitoring |
| `high` | Significant instability | Protective measures |
| `crisis` | Immediate support needed | Crisis protocols |

---

## Regime Logic

The system operates in different regimes based on entropy and confidence:

| Regime | Trigger | Behavior |
|--------|---------|----------|
| `stable` | Low entropy, high confidence | Apostasis active, normal operation |
| `elevated` | Elevated entropy | Increased monitoring |
| `protective` | High entropy | Protective measures, grounding prioritized |
| `crisis` | Crisis entropy | Crisis protocols, regeneration paused |
| `recovery` | Improving from crisis | Regeneration active |

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=reunity

# Run specific test file
pytest tests/test_entropy.py
```

### Code Quality

```bash
# Format code
black src tests

# Type checking
mypy src

# Linting
ruff check src tests
```

---

## Project Structure

```
reunity/
├── src/
│   └── reunity/
│       ├── __init__.py          # Package initialization
│       ├── api/
│       │   └── main.py          # FastAPI application
│       ├── core/
│       │   └── entropy.py       # Entropy analysis
│       ├── router/
│       │   └── state_router.py  # Policy routing
│       ├── protective/
│       │   └── pattern_recognizer.py  # Pattern detection
│       ├── memory/
│       │   └── continuity_store.py    # RIME implementation
│       ├── reflection/
│       │   └── mirror_link.py   # Dialogue companion
│       ├── regime/
│       │   └── regime_controller.py   # Regime logic
│       ├── storage/
│       │   └── encrypted_store.py     # Encrypted storage
│       └── export/
│           └── portability.py   # Export bundles
├── tests/
│   ├── conftest.py
│   ├── test_entropy.py
│   ├── test_memory.py
│   └── test_regime.py
├── docs/
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## GitHub Deployment

### Initial Setup

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ReUnity trauma-aware AI system"

# Add remote (replace with your repository URL)
git remote add origin https://github.com/yourusername/reunity.git

# Push to GitHub
git push -u origin main
```

### GitHub Actions (CI/CD)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Run tests
        run: pytest --cov=reunity
```

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct before submitting pull requests.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Christopher Ezernack**

---

## Acknowledgments

- Built on the theoretical foundations of trauma-informed care
- Inspired by research in dissociative disorders and complex trauma
- Designed with input from mental health professionals

---

## References

1. Shannon, C. E. (1948). A Mathematical Theory of Communication
2. Lin, J. (1991). Divergence Measures Based on the Shannon Entropy
3. Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory
4. Van der Kolk, B. (2014). The Body Keeps the Score

---

*Remember: This tool is meant to support, not replace, professional mental health care.*
