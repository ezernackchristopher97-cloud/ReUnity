# ReUnity

## A Trauma-Aware AI Support Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

---

## IMPORTANT DISCLAIMER

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

- **Entropy-Based State Detection**: Uses Shannon entropy, Jensen-Shannon divergence, mutual information, and Lyapunov exponents to detect emotional/cognitive states
- **Adaptive Policy Routing**: Automatically adjusts support strategies based on detected states
- **Protective Pattern Recognition**: Identifies 13+ potentially harmful relationship dynamics (gaslighting, love bombing, hot-cold cycles, isolation attempts, etc.)
- **Continuity Memory (RIME)**: Recursive Identity Memory Engine for maintaining identity continuity across fragmented states
- **MirrorLink Reflection**: Surfaces contradictions without invalidation, helping users hold multiple truths
- **Regime Logic**: Adaptive behavior based on entropy bands with apostasis (pruning) and regeneration
- **Alter-Aware Subsystem**: Specialized support for dissociative identity experiences
- **Clinician Interface**: Professional access with consent controls and audit logging
- **Condition-Specific Support**: Specialized modules for DID, PTSD, BPD, Bipolar, and Schizophrenia
- **Grounding Techniques Library**: 20+ entropy-based grounding recommendations
- **Quantum-Resistant Cryptography**: Future-proof security implementations
- **Consent-Scoped Access**: Fine-grained privacy controls for all stored data
- **Local-First Architecture**: Encrypted storage with AES-256-GCM and user data sovereignty
- **Portable Export Bundles**: Data portability with provenance tracking and hash verification

---

## Quick Start

### Option 1: Standalone Single-File Version (Easiest)

The easiest way to try ReUnity is with the standalone file that contains all core functionality:

```bash
# Download or copy reunity_standalone.py
python reunity_standalone.py

# Or run with test mode
python reunity_standalone.py --test
```

This single file runs an interactive CLI session with all core components.

### Option 2: Full Installation

```bash
# Clone the repository
git clone https://github.com/ezernackchristopher97-cloud/ReUnity.git
cd ReUnity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run the API server
uvicorn reunity.api.main:app --reload
```

The API will be available at `http://localhost:8000`. Access the interactive documentation at `http://localhost:8000/docs`.

### Option 3: Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t reunity .
docker run -p 8000:8000 reunity
```

---

## Architecture

ReUnity implements a seven-component AI Mirror System architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│              (Web / Mobile / CLI / Crisis Mode)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                         Core AI Layer                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  EESA   │ │  RIME   │ │   PLM   │ │   RCT   │ │  MLDC   │   │
│  │Entropy  │ │ Memory  │ │Protect  │ │Relation │ │ Mirror  │   │
│  │Analyzer │ │ Engine  │ │ Logic   │ │ Thread  │ │  Link   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐                                        │
│  │   AAS   │ │   CCI   │                                        │
│  │ Alter   │ │Clinician│                                        │
│  │ Aware   │ │Interface│                                        │
│  └─────────┘ └─────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Regime Controller                           │
│  ┌──────────┐  ┌─────────────┐  ┌────────────────────────────┐  │
│  │Apostasis │  │Regeneration │  │   Lattice Memory Graph     │  │
│  │(Pruning) │  │ (Restore)   │  │   (Divergence-Bounded)     │  │
│  └──────────┘  └─────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                                │
│     Encrypted Storage │ Consent Management │ Export Bundles      │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **EESA** | Entropy-Based Emotional State Analyzer - monitors emotional entropy and stability using Shannon entropy, JS divergence, and Lyapunov exponents |
| **RIME** | Recursive Identity Memory Engine - maintains continuity across sessions with consent-scoped access |
| **PLM** | Protective Logic Module - detects 13+ harmful relational patterns |
| **RCT** | Relationship Continuity Threader - tracks relational context across time |
| **MLDC** | MirrorLink Dialogue Companion - provides structured reflections without invalidation |
| **AAS** | Alter-Aware Subsystem - supports dissociative identity experiences |
| **CCI** | Clinician and Caregiver Interface - professional access with consent and audit logging |

---

## Mathematical Foundations

### Shannon Entropy for Emotional State Detection

```
S = -Σ(i=1 to n) p_i × log₂(p_i)
```

Where:
- S = entropy of emotional state system
- p_i = probability of emotional state i
- Higher entropy = greater emotional fragmentation/chaos
- Lower entropy = emotional rigidity or stability

### Jensen-Shannon Divergence for State Transitions

```
JS(P,Q) = ½ × D_KL(P||M) + ½ × D_KL(Q||M)
```

Where M = ½(P + Q) is the midpoint distribution. Used to detect splitting episodes, dissociative transitions, or significant state changes.

### Mutual Information

```
I(X;Y) = Σ p(x,y) log₂(p(x,y) / (p(x)p(y)))
```

Measures the mutual dependence between emotional states and external triggers.

### Lyapunov Exponents for System Stability

```
λ = lim(n→∞) (1/n) × Σ log₂|dS/dt|
```

Where:
- λ > 0: chaos/instability (crisis intervention needed)
- λ < 0: stability (therapeutic progress)
- λ ≈ 0: marginal stability

### RIME Memory Scoring

```
RIME(t) = α × M_episodic(t) + β × M_semantic(t) + γ × C_context(t)
```

Where:
- M_episodic = episodic memory activation (recency-weighted)
- M_semantic = semantic memory patterns (tag overlap)
- C_context = current contextual factors (emotional similarity)
- α, β, γ = dynamically adjusted weights (default: 0.4, 0.35, 0.25)

### Free Energy Principle

```
F = E_q[log q(s) - log p(o,s)]
```

Variational free energy minimization for predictive processing and surprise reduction.

---

## Entropy States and Policy Routing

| State | Entropy Range | System Response |
|-------|---------------|-----------------|
| `stable` | < 0.15 | Engagement policy, growth-oriented prompts |
| `low` | 0.15 - 0.30 | Maintenance policy, check-ins and journaling |
| `moderate` | 0.30 - 0.50 | Support policy, active listening and validation |
| `elevated` | 0.50 - 0.70 | Stabilization policy, grounding prompts |
| `high` | 0.70 - 0.85 | Protective measures, simplified interface |
| `crisis` | > 0.85 | Crisis intervention, immediate grounding, emergency resources |

---

## Protective Pattern Recognition

ReUnity detects 13+ harmful relational patterns:

| Pattern | Description |
|---------|-------------|
| **Gaslighting** | Reality denial, memory questioning |
| **Love Bombing** | Excessive early attention and idealization |
| **Hot-Cold Cycle** | Inconsistent availability and affection |
| **Isolation** | Attempts to separate from support network |
| **Emotional Baiting** | Provocations to test or manipulate |
| **Abandonment Trigger** | Threats of leaving during conflicts |
| **Invalidation** | Dismissing feelings and experiences |
| **Triangulation** | Using others to create insecurity |
| **Silent Treatment** | Punishment through withdrawal |
| **Blame Shifting** | Avoiding accountability |
| **Future Faking** | Empty promises about change |
| **Hoovering** | Attempts to re-engage after separation |
| **Devaluation** | Criticism following idealization |

---

## Condition-Specific Support

ReUnity provides specialized support modules:

### Dissociative Identity Disorder (DID)
- Alter recognition and profile management
- Inter-alter communication facilitation
- Shared memory systems with consent
- Switch tracking and co-consciousness support

### PTSD and Complex PTSD
- Dissociation prediction through entropy analysis
- Preemptive grounding interventions
- Reality anchoring tools
- Trigger pattern tracking

### Borderline Personality Disorder (BPD)
- Dialectical thinking support (both/and perspectives)
- Contradiction reflection without invalidation
- Relationship thread preservation during splitting
- Identity continuity maintenance

### Bipolar Disorder
- Mood continuity preservation
- Episode pattern tracking
- Early warning detection through entropy trends
- Transition support between states

### Schizophrenia/Schizoaffective
- Reality testing without invalidation
- Consistency tracking across time
- Lyapunov-based episode prediction
- Grounding recommendations

---

## Regime Logic

The system operates in different regimes based on entropy and stability:

| Regime | Trigger | Behavior |
|--------|---------|----------|
| **Normal** | Moderate entropy, stable | Standard support interactions |
| **Protective** | High entropy, unstable | Increased grounding, simplified interface |
| **Crisis** | Critical entropy, chaotic | Immediate intervention, crisis resources |
| **Recovery** | Low entropy, stabilizing | Gentle restoration, reflection |
| **Growth** | Very low entropy, very stable | Exploration, future planning |

### Apostasis (Pruning)
During stable states (entropy < 0.3), low-utility memory features are marked for pruning to reduce cognitive load while preserving essential identity threads.

### Regeneration
When stability returns (entropy > 0.5 with stable Lyapunov), previously pruned features can be restored in a controlled manner.

### Lattice Memory Graph
A discrete state graph over identity/memory/relationship nodes with edges constrained by JS divergence (< 0.7), ensuring coherent memory connections.

---

## API Endpoints

### Health and Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check with disclaimer |
| `/health` | GET | Health status |
| `/disclaimer` | GET | Full disclaimer text |

### Entropy Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/entropy/analyze` | POST | Analyze text for entropy state |
| `/api/v1/entropy/states` | GET | List available entropy states |

### Memory Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory` | POST | Store a memory |
| `/api/v1/memory/{id}` | GET | Retrieve a memory |
| `/api/v1/memory/search` | POST | Search memories by tags |
| `/api/v1/memory/timeline` | GET | Get memory timeline |
| `/api/v1/memory/consent` | PUT | Update consent scope |

### Pattern Recognition

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/patterns/analyze` | POST | Analyze for harmful patterns |
| `/api/v1/patterns/summary` | GET | Get pattern detection summary |

### Reflection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/reflection/generate` | POST | Generate a reflection |
| `/api/v1/reflection/dialectical` | POST | Get dialectical reflection |

### Grounding

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/grounding/recommend` | GET | Get grounding recommendations |
| `/api/v1/grounding/techniques` | GET | List all techniques |

### Alter-Aware (DID Support)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/alter/register` | POST | Register an alter profile |
| `/api/v1/alter/switch` | POST | Record alter switch |
| `/api/v1/alter/profiles` | GET | List alter profiles |

### Export

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/export/bundle` | POST | Export data bundle with provenance |
| `/api/v1/export/timeline` | GET | Export timeline events |

---

## Privacy and Security

### Encryption
- AES-256-GCM for data at rest
- Quantum-resistant cryptography (CRYSTALS-Kyber/Dilithium ready)
- Zero-knowledge proofs for verification
- Anti-forensic secure deletion

### Consent Scopes

| Scope | Description |
|-------|-------------|
| `private` | Only the user can access |
| `self_only` | User and their alters (for DID support) |
| `trusted_contacts` | Shared with designated trusted contacts |
| `clinician` | Shared with designated clinician/therapist |
| `research_anonymized` | Anonymized for research (with explicit consent) |

### Data Portability
- Export bundles with provenance tracking
- SHA-256 hash verification for integrity
- Anonymization options for research sharing

---

## Basic Usage Example

```python
from reunity import ReUnity

# Initialize the system
system = ReUnity()

# Process user input
response = system.process_input(
    text="I'm feeling anxious about my relationship",
    emotional_state={"anxiety": 0.7, "sadness": 0.2, "hope": 0.1}
)

# Access analysis results
print(f"Entropy State: {response['analysis']['state']}")
print(f"Shannon Entropy: {response['analysis']['shannon_entropy']:.3f}")
print(f"Stability: {response['analysis']['stability']}")
print(f"Current Regime: {response['regime']}")
print(f"Support Message: {response['support_message']}")

# Check for detected patterns
for pattern in response['patterns_detected']:
    print(f"Pattern: {pattern['pattern_type']} ({pattern['confidence']:.0%})")
    print(f"Recommendation: {pattern['recommendation']}")

# Get grounding recommendation if in high entropy state
if response['grounding_recommendation']:
    print(f"Grounding: {response['grounding_recommendation']['name']}")
    print(f"Instructions: {response['grounding_recommendation']['description']}")
```

---

## Project Structure

```
reunity/
├── src/reunity/
│   ├── __init__.py              # Package initialization
│   ├── core/
│   │   ├── entropy.py           # Entropy analysis (Shannon, JS, MI, Lyapunov)
│   │   └── free_energy.py       # Free Energy Principle
│   ├── router/
│   │   └── state_router.py      # Policy routing
│   ├── protective/
│   │   ├── pattern_recognizer.py # Pattern detection (13+ patterns)
│   │   └── safety_assessment.py  # Crisis and risk assessment
│   ├── memory/
│   │   ├── continuity_store.py  # RIME implementation
│   │   └── timeline_threading.py # Timeline with gap detection
│   ├── reflection/
│   │   └── mirror_link.py       # MirrorLink dialogue companion
│   ├── regime/
│   │   └── regime_controller.py # Apostasis, Regeneration, Lattice
│   ├── alter/
│   │   └── alter_aware.py       # DID support
│   ├── clinician/
│   │   └── caregiver_interface.py # Professional interface
│   ├── conditions/
│   │   └── support.py           # Condition-specific support
│   ├── grounding/
│   │   └── techniques.py        # 20+ grounding techniques
│   ├── crypto/
│   │   └── quantum_resistant.py # Future-proof cryptography
│   ├── storage/
│   │   └── encrypted_store.py   # AES-256-GCM encryption
│   ├── export/
│   │   └── portability.py       # Export bundles with provenance
│   └── api/
│       ├── main.py              # FastAPI application
│       └── endpoints_extended.py # Extended endpoints
├── tests/
│   ├── conftest.py
│   ├── test_entropy.py
│   ├── test_memory.py
│   ├── test_regime.py
│   └── test_integration.py
├── examples/
│   ├── basic_usage.py
│   └── alter_aware_example.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── API.md
├── reunity_standalone.py        # Single-file version (copy & paste ready)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── CONTRIBUTORS.md
├── CONTRIBUTING.md
├── TODO.md
└── README.md
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

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
mypy src/reunity

# Linting
ruff check src tests
```

---

## GitHub Deployment

### Initial Setup

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: ReUnity trauma-aware AI system"

# Add remote
git remote add origin https://github.com/ezernackchristopher97-cloud/ReUnity.git

# Push to GitHub
git push -u origin main
```

### GitHub Actions (CI/CD)

The repository includes a CI workflow at `.github/workflows/ci.yml` that runs tests on every push and pull request.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Author

**Christopher Ezernack**, REOP Solutions

---

## Acknowledgments

### Theoretical Foundations
- Karl Friston - Free Energy Principle
- Claude Shannon - Information Theory
- Judith Herman - Trauma and Recovery
- Bessel van der Kolk - Body-based trauma understanding
- Marsha Linehan - Dialectical Behavior Therapy

### References
1. Shannon, C. E. (1948). A Mathematical Theory of Communication
2. Lin, J. (1991). Divergence Measures Based on the Shannon Entropy
3. Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory
4. Van der Kolk, B. (2014). The Body Keeps the Score
5. Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory?

---

**Remember: This tool is meant to support, not replace, professional mental health care. If you are in crisis, please reach out to a crisis line.**
