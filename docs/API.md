# ReUnity API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`

> **DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. It is a theoretical and support framework only.

## Authentication

Currently, the API does not require authentication for local deployment. For production deployments, implement appropriate authentication mechanisms.

## Response Format

All responses are JSON formatted with the following structure:

```json
{
  "field1": "value1",
  "field2": "value2"
}
```

Error responses:
```json
{
  "detail": "Error message"
}
```

---

## Endpoints

### Health & Info

#### GET /
Root endpoint with health check and disclaimer.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "disclaimer": "...",
  "timestamp": 1706000000.0
}
```

#### GET /health
Health check endpoint.

#### GET /disclaimer
Get full disclaimer text.

---

### Entropy Analysis

#### POST /entropy/analyze
Analyze text for entropy state.

**Request:**
```json
{
  "text": "I feel calm and centered today",
  "include_stability": true
}
```

**Response:**
```json
{
  "state": "stable",
  "normalized_entropy": 0.42,
  "confidence": 0.85,
  "is_stable": true,
  "lyapunov_exponent": -0.15,
  "recommendations": [
    "Current state is stable",
    "Continue current approach"
  ]
}
```

**Entropy States:**
| State | Range | Description |
|-------|-------|-------------|
| low | < 0.3 | Highly predictable, possibly rigid |
| stable | 0.3 - 0.5 | Healthy variability |
| elevated | 0.5 - 0.7 | Increased uncertainty |
| high | 0.7 - 0.85 | Significant instability |
| crisis | ≥ 0.85 | Immediate support needed |

#### GET /entropy/states
Get available entropy states and descriptions.

---

### Memory Management

#### POST /memory/add
Add a memory to the continuity store.

**Request:**
```json
{
  "identity": "host",
  "content": "A peaceful memory from my grandmother's garden",
  "memory_type": "anchor",
  "tags": ["safe_place", "family"],
  "consent_scope": "private",
  "emotional_valence": 0.8,
  "importance": 0.9
}
```

**Response:**
```json
{
  "id": "mem_abc123",
  "content": "A peaceful memory...",
  "memory_type": "anchor",
  "identity_state": "host",
  "timestamp": 1706000000.0,
  "tags": ["safe_place", "family"],
  "consent_scope": "private"
}
```

**Memory Types:**
- `episodic` - Event memories
- `semantic` - Factual knowledge
- `anchor` - Stabilizing memories
- `procedural` - Skills and habits
- `emotional` - Feeling states

**Consent Scopes:**
- `private` - Only accessible to creating identity
- `system_shared` - Shared within system
- `therapist_shared` - Shared with therapist
- `caregiver_shared` - Shared with caregivers
- `research_anonymized` - Anonymized for research

#### POST /memory/retrieve
Retrieve memories with grounding support.

**Request:**
```json
{
  "identity": "host",
  "query": "safe place",
  "crisis_level": 0.3,
  "max_results": 5
}
```

**Response:**
```json
{
  "memories": [...],
  "total_found": 10,
  "filtered_by_consent": 2,
  "retrieval_method": "grounding_priority"
}
```

#### PUT /memory/consent
Update consent scope for a memory.

**Request:**
```json
{
  "memory_id": "mem_abc123",
  "new_scope": "therapist_shared"
}
```

#### GET /memory/stats
Get memory store statistics.

---

### Pattern Recognition

#### POST /patterns/analyze
Analyze interactions for harmful patterns.

**Request:**
```json
{
  "interactions": [
    {"text": "You're imagining things", "timestamp": 1706000000},
    {"text": "No one else believes you", "timestamp": 1706001000}
  ],
  "person_id": "person_123"
}
```

**Response:**
```json
{
  "patterns_detected": [
    {
      "type": "gaslighting",
      "severity": "high",
      "confidence": 0.85,
      "message": "Reality contradictions detected",
      "recommendation": "Trust your memory threads"
    }
  ],
  "overall_risk": 0.72,
  "sentiment_variance": 0.45,
  "stability_assessment": "unstable",
  "recommendations": [
    "Document your experiences",
    "Consider discussing with a trusted person"
  ]
}
```

**Pattern Types:**
- `hot_cold_cycle` - Intermittent reinforcement
- `gaslighting` - Reality contradiction
- `love_bombing` - Excessive affection
- `abandonment_threat` - Threats of leaving
- `isolation_attempt` - Separating from support
- `financial_control` - Money manipulation
- `reality_contradiction` - Denying events
- `emotional_baiting` - Provoking reactions
- `invalidation` - Dismissing feelings
- `blame_shifting` - Deflecting responsibility
- `triangulation` - Using third parties
- `silent_treatment` - Withdrawal as punishment
- `boundary_violation` - Ignoring limits

---

### Reflection

#### POST /reflection/generate
Generate a MirrorLink reflection.

**Request:**
```json
{
  "current_emotion": "confused and hurt",
  "past_context": "Last week I felt loved by them",
  "style": "gentle"
}
```

**Response:**
```json
{
  "content": "You feel confused and hurt now, but last week you felt loved. Can both be real? What might explain this difference?",
  "reflection_type": "contradiction",
  "is_contradiction": true,
  "follow_up_question": "What do you notice in your body right now?",
  "grounding_prompt": null
}
```

**Communication Styles:**
- `gentle` - Soft, supportive tone
- `direct` - Clear, straightforward
- `curious` - Exploratory, questioning
- `validating` - Emphasizes validation
- `grounding` - Focus on present moment

---

### Journal

#### POST /journal/add
Add a journal entry.

**Request:**
```json
{
  "title": "Morning Reflection",
  "content": "Woke up feeling rested...",
  "identity": "host",
  "mood": "calm",
  "energy_level": 0.7,
  "tags": ["morning", "positive"],
  "consent_scope": "private"
}
```

#### GET /journal/list
List journal entries.

**Query Parameters:**
- `identity` (optional): Filter by identity state

---

### Regime Control

#### GET /regime/status
Get current regime status.

**Response:**
```json
{
  "regime": "maintenance",
  "entropy_band": "stable",
  "confidence": 0.85,
  "time_in_regime": 3600.0,
  "apostasis_active": false,
  "regeneration_active": false
}
```

**Regimes:**
- `exploration` - Low entropy, encourage growth
- `maintenance` - Stable, maintain current state
- `support` - Elevated, increase support
- `stabilization` - High, focus on grounding
- `crisis` - Crisis, immediate safety focus

#### GET /regime/history
Get regime transition history.

---

### Export

#### POST /export/bundle
Export data as a portability bundle.

**Request:**
```json
{
  "identity": "host",
  "include_memories": true,
  "include_journals": true,
  "include_timeline": true
}
```

**Response:**
```json
{
  "memories": [...],
  "journals": [...],
  "timeline": [...],
  "statistics": {...},
  "provenance": {
    "exported_at": 1706000000.0,
    "export_version": "1.0.0",
    "system": "ReUnity",
    "disclaimer": "..."
  }
}
```

#### GET /export/timeline
Export timeline events.

---

## Extended Endpoints (v1)

### Alter-Aware Subsystem

#### POST /v1/alter/register
Register a new alter.

#### GET /v1/alter/list
List all registered alters.

#### POST /v1/alter/switch
Record a switch event.

#### POST /v1/alter/message
Send an internal message.

#### GET /v1/alter/messages/{alter_id}
Get messages for an alter.

#### GET /v1/alter/system-report
Get system functioning report.

### Clinician Interface

#### POST /v1/clinician/register-provider
Register a care provider.

#### POST /v1/clinician/grant-consent
Grant consent to a provider.

#### POST /v1/clinician/revoke-consent/{consent_id}
Revoke a consent grant.

#### GET /v1/clinician/consent-records
Get all consent records.

#### GET /v1/clinician/access-log
Get the access log.

### Safety Assessment

#### POST /v1/safety/assess
Perform safety assessment.

**Request:**
```json
{
  "text": "I'm feeling overwhelmed",
  "entropy_level": 0.6
}
```

**Response:**
```json
{
  "risk_level": "moderate",
  "crisis_types": ["emotional_overwhelm"],
  "risk_factors": ["High emotional entropy"],
  "protective_factors": ["Mention of support"],
  "recommended_actions": [
    "Consider using a grounding technique"
  ],
  "entropy_level": 0.6
}
```

#### GET /v1/safety/resources
Get crisis resources.

### Grounding Techniques

#### POST /v1/grounding/recommend
Get recommended grounding technique.

#### GET /v1/grounding/list
List available techniques.

#### GET /v1/grounding/technique/{technique_id}
Get technique details.

#### POST /v1/grounding/session/start
Start a grounding session.

#### POST /v1/grounding/session/complete/{session_id}
Complete a grounding session.

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Not initialized |

---

## Rate Limiting

For production deployments, implement appropriate rate limiting. Suggested limits:
- 100 requests per minute for analysis endpoints
- 1000 requests per minute for retrieval endpoints

---

## Webhooks (Future)

Planned webhook support for:
- Crisis state detection
- Pattern detection alerts
- Regime transitions

---

*For more information, see the [Architecture Documentation](./ARCHITECTURE.md).*
