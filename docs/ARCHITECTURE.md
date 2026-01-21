# ReUnity System Architecture

**Version:** 1.0.0  
**Author:** Christopher Ezernack  
**Last Updated:** January 2026

> **DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only. This system is not intended to diagnose, treat, cure, or prevent any medical or psychological condition.

## Table of Contents

1. [Overview](#overview)
2. [Core Principles](#core-principles)
3. [System Components](#system-components)
4. [Mathematical Foundations](#mathematical-foundations)
5. [Data Flow](#data-flow)
6. [Security & Privacy](#security--privacy)
7. [Deployment Architecture](#deployment-architecture)

---

## Overview

ReUnity is a trauma-aware AI support system designed to provide continuity, stability, and protective support for individuals navigating complex emotional and psychological experiences. The system is built on principles from information theory, dynamical systems, and trauma-informed care.

### Design Philosophy

1. **User Sovereignty**: Users maintain complete control over their data and interactions
2. **Non-Pathologizing**: The system validates experiences without labeling or diagnosing
3. **Continuity Preservation**: Maintains narrative threads across fragmented states
4. **Protective Intelligence**: Recognizes potentially harmful patterns without judgment
5. **Adaptive Response**: Adjusts behavior based on detected emotional/cognitive state

---

## Core Principles

### The Free Energy Principle

ReUnity is grounded in the Free Energy Principle (FEP), which posits that adaptive systems minimize surprise (or free energy) to maintain their integrity. In the context of ReUnity:

- **Variational Free Energy**: Measures the divergence between the system's model and actual observations
- **Expected Free Energy**: Guides action selection to minimize future surprise
- **Active Inference**: The system acts to confirm its predictions or update its model

### Entropy as State Indicator

Shannon entropy serves as the primary indicator of cognitive/emotional state:

```
H(X) = -Σ p(x) log₂ p(x)
```

- **Low Entropy (H < 0.3)**: Rigid, potentially dissociative states
- **Stable Entropy (0.3 ≤ H < 0.5)**: Healthy variability
- **Elevated Entropy (0.5 ≤ H < 0.7)**: Increased uncertainty
- **High Entropy (0.7 ≤ H < 0.85)**: Significant instability
- **Crisis Entropy (H ≥ 0.85)**: Immediate support needed

### Jensen-Shannon Divergence

Used to measure the difference between probability distributions:

```
JSD(P || Q) = ½ KL(P || M) + ½ KL(Q || M)
```

Where M = ½(P + Q). This metric helps detect:
- Shifts in emotional state
- Contradictions in narrative
- Changes in relationship dynamics

---

## System Components

### 1. Entropy Analysis Core (`core/entropy.py`)

The foundation of state detection and monitoring.

```
┌─────────────────────────────────────────────────────────┐
│                   EntropyAnalyzer                       │
├─────────────────────────────────────────────────────────┤
│ • Shannon Entropy Calculation                           │
│ • Jensen-Shannon Divergence                             │
│ • Mutual Information                                    │
│ • Lyapunov Exponent (Stability)                        │
│ • State Classification                                  │
└─────────────────────────────────────────────────────────┘
```

**Key Methods:**
- `analyze(distribution)` → EntropyMetrics
- `calculate_js_divergence(p, q)` → float
- `estimate_lyapunov(time_series)` → float

### 2. State Router (`router/state_router.py`)

Routes system behavior based on detected entropy state.

```
┌─────────────────────────────────────────────────────────┐
│                    StateRouter                          │
├─────────────────────────────────────────────────────────┤
│ EntropyState → PolicyType                               │
│                                                         │
│ LOW      → ENGAGE (encourage exploration)               │
│ STABLE   → MAINTAIN (continue current approach)         │
│ ELEVATED → SUPPORT (increase support)                   │
│ HIGH     → STABILIZE (grounding focus)                  │
│ CRISIS   → CRISIS (immediate safety)                    │
└─────────────────────────────────────────────────────────┘
```

### 3. Protective Pattern Recognizer (`protective/pattern_recognizer.py`)

Detects potentially harmful relationship dynamics.

**Detected Patterns:**
- Hot-Cold Cycles (intermittent reinforcement)
- Gaslighting (reality contradiction)
- Love Bombing (excessive affection)
- Isolation Attempts
- Boundary Violations
- Financial Control
- Triangulation

**Output:**
```python
InteractionAnalysis(
    patterns_detected: List[DetectedPattern],
    overall_risk: float,
    sentiment_variance: float,
    stability_assessment: str,
    recommendations: List[str]
)
```

### 4. Continuity Memory Store (`memory/continuity_store.py`)

Implements the Recursive Identity Memory Engine (RIME).

```
┌─────────────────────────────────────────────────────────┐
│              RecursiveIdentityMemoryEngine              │
├─────────────────────────────────────────────────────────┤
│ Memory Types:                                           │
│ • EPISODIC - Event memories                             │
│ • SEMANTIC - Factual knowledge                          │
│ • ANCHOR - Stabilizing memories                         │
│ • PROCEDURAL - Skills and habits                        │
│ • EMOTIONAL - Feeling states                            │
│                                                         │
│ Consent Scopes:                                         │
│ • PRIVATE - Only accessible to creating identity        │
│ • SYSTEM_SHARED - Shared within system                  │
│ • THERAPIST_SHARED - Shared with therapist              │
│ • CAREGIVER_SHARED - Shared with caregivers             │
│ • RESEARCH_ANONYMIZED - Anonymized for research         │
└─────────────────────────────────────────────────────────┘
```

### 5. MirrorLink Reflection (`reflection/mirror_link.py`)

Provides reflective dialogue that surfaces contradictions without invalidation.

**Core Principle:**
> "You feel betrayed now, but you also called them your anchor last week. Can both be real?"

**Reflection Types:**
- VALIDATION - Pure emotional validation
- CONTRADICTION - Gentle contradiction surfacing
- CONTINUITY - Connection to past experiences
- GROUNDING - Grounding prompts during crisis
- EXPLORATION - Open-ended exploration

### 6. Regime Controller (`regime/regime_controller.py`)

Manages system behavior regimes based on entropy bands.

```
┌─────────────────────────────────────────────────────────┐
│                  RegimeController                       │
├─────────────────────────────────────────────────────────┤
│ Regimes:                                                │
│ • EXPLORATION - Low entropy, encourage growth           │
│ • MAINTENANCE - Stable, maintain current state          │
│ • SUPPORT - Elevated, increase support                  │
│ • STABILIZATION - High, focus on grounding              │
│ • CRISIS - Crisis, immediate safety focus               │
│                                                         │
│ Special Processes:                                      │
│ • Apostasis - Pruning during stable states              │
│ • Regeneration - Restoration during recovery            │
└─────────────────────────────────────────────────────────┘
```

### 7. Lattice Memory Graph (`regime/regime_controller.py`)

Graph-based memory organization constrained by divergence.

```
Nodes: Memory entries
Edges: Weighted by JS divergence
Constraint: edge_weight < divergence_threshold

Operations:
• add_node(memory_id, content, importance)
• add_edge(source, target, weight)
• get_neighbors(node_id, max_divergence)
• find_path(source, target)
```

### 8. Alter-Aware Subsystem (`alter/alter_aware.py`)

Supports individuals with dissociative identity experiences.

**Features:**
- Alter profile registration
- Switch event tracking
- Internal communication (direct, broadcast, bulletin)
- Identity-scoped memory access
- System functioning reports

### 9. Clinician/Caregiver Interface (`clinician/caregiver_interface.py`)

Controlled access for care providers.

**Access Levels:**
- VIEW_ONLY - Read access to shared data
- ANNOTATE - Can add clinical notes
- COLLABORATE - Can suggest interventions
- FULL_ACCESS - Complete access (rare)

**Consent Management:**
- Explicit consent required for all access
- Time-limited grants
- Audit logging of all access
- User can revoke at any time

---

## Mathematical Foundations

### Shannon Entropy

```
H(X) = -Σᵢ p(xᵢ) log₂ p(xᵢ)
```

Normalized entropy (0-1 scale):
```
H_norm = H(X) / log₂(n)
```

### Jensen-Shannon Divergence

```
JSD(P || Q) = ½ D_KL(P || M) + ½ D_KL(Q || M)
```

Where:
- M = ½(P + Q)
- D_KL is Kullback-Leibler divergence

### Mutual Information

```
I(X; Y) = H(X) + H(Y) - H(X, Y)
```

### Lyapunov Exponent (Stability)

```
λ = lim(n→∞) (1/n) Σᵢ log|f'(xᵢ)|
```

- λ < 0: Stable (converging)
- λ ≈ 0: Neutral
- λ > 0: Unstable (diverging/chaotic)

### Free Energy

Variational Free Energy:
```
F = D_KL(q(s) || p(s|o)) - log p(o)
```

Expected Free Energy:
```
G = E_q[log q(s) - log p(o, s)]
```

---

## Data Flow

### Request Processing Flow

```
User Input
    │
    ▼
┌─────────────┐
│   FastAPI   │
│   Endpoint  │
└─────────────┘
    │
    ▼
┌─────────────┐     ┌─────────────┐
│   Entropy   │────▶│    State    │
│   Analyzer  │     │   Router    │
└─────────────┘     └─────────────┘
    │                     │
    ▼                     ▼
┌─────────────┐     ┌─────────────┐
│   Pattern   │     │   Regime    │
│ Recognizer  │     │ Controller  │
└─────────────┘     └─────────────┘
    │                     │
    └──────────┬──────────┘
               │
               ▼
        ┌─────────────┐
        │   Memory    │
        │   Engine    │
        └─────────────┘
               │
               ▼
        ┌─────────────┐
        │  Response   │
        │ Generation  │
        └─────────────┘
               │
               ▼
          Response
```

### Memory Retrieval Flow

```
Query + Identity + Crisis Level
           │
           ▼
    ┌─────────────┐
    │   Consent   │
    │   Filter    │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  Relevance  │
    │   Scoring   │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   Crisis    │
    │  Priority   │──▶ If crisis_level > 0.5:
    └─────────────┘    Prioritize anchor memories
           │
           ▼
    Retrieved Memories
```

---

## Security & Privacy

### Encryption

- **At Rest**: AES-256-GCM encryption for all stored data
- **In Transit**: TLS 1.3 for all API communications
- **Key Management**: User-controlled encryption keys

### Consent Model

```
┌─────────────────────────────────────────────────────────┐
│                    Consent Record                       │
├─────────────────────────────────────────────────────────┤
│ consent_id: str                                         │
│ user_id: str                                            │
│ provider_id: str                                        │
│ access_level: AccessLevel                               │
│ data_types: List[str]                                   │
│ granted_at: float                                       │
│ expires_at: float | None                                │
│ status: ConsentStatus (ACTIVE, REVOKED, EXPIRED)        │
└─────────────────────────────────────────────────────────┘
```

### Audit Logging

All data access is logged:
```
AccessLog(
    log_id: str,
    provider_id: str,
    action: str,
    data_accessed: List[str],
    timestamp: float,
    ip_address: str | None
)
```

### Data Portability

Export bundles include:
- All user data in standard formats
- Provenance information
- Hash verification
- Consent records

---

## Deployment Architecture

### Local-First Mode

```
┌─────────────────────────────────────────────────────────┐
│                    User Device                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   FastAPI   │  │  SQLite DB  │  │  Encrypted  │     │
│  │   Server    │  │  (Local)    │  │   Storage   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          │                              │
│                    Local Only                           │
│                    No Cloud                             │
└─────────────────────────────────────────────────────────┘
```

### Docker Deployment

```yaml
services:
  reunity:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - REUNITY_STORAGE_PATH=/app/data
      - REUNITY_ENCRYPTION_KEY_FILE=/app/data/.key
```

### Production Deployment

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                        │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  ReUnity  │   │  ReUnity  │   │  ReUnity  │
    │ Instance  │   │ Instance  │   │ Instance  │
    └───────────┘   └───────────┘   └───────────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                          ▼
                ┌─────────────────┐
                │   Encrypted     │
                │   Database      │
                └─────────────────┘
```

---

## API Reference

See [API.md](./API.md) for complete API documentation.

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/entropy/analyze` | POST | Analyze text for entropy state |
| `/memory/add` | POST | Add a memory with consent scope |
| `/memory/retrieve` | POST | Retrieve memories with grounding support |
| `/patterns/analyze` | POST | Analyze interactions for harmful patterns |
| `/reflection/generate` | POST | Generate MirrorLink reflection |
| `/regime/status` | GET | Get current regime status |
| `/export/bundle` | POST | Export data with provenance |

---

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory?
2. Shannon, C. E. (1948). A Mathematical Theory of Communication
3. Herman, J. (1992). Trauma and Recovery
4. Van der Kolk, B. (2014). The Body Keeps the Score

---

*This documentation is part of the ReUnity project. For questions or contributions, please see CONTRIBUTING.md.*
