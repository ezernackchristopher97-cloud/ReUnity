# ReUnity System Architecture

## Overview

ReUnity is a full-stack trauma-aware AI mental health support platform built with:
- **Frontend**: React 19 + TypeScript + Tailwind CSS 4 + shadcn/ui
- **Backend**: Express 4 + tRPC 11 + Drizzle ORM
- **Database**: MySQL/TiDB
- **AI**: Custom ReUnity AI Engine with LLM integration

---

## Core Architecture Components

### 1. ReUnity AI Engine (`server/reunity.ts`)

The heart of the system - a 3000+ line recursive, entropy-aware AI framework:

**Key Classes:**
- `EntropyAnalyzer` - Analyzes emotional entropy (0-1 scale) from text
- `PatternRecognizer` - Detects harmful relationship patterns (gaslighting, love-bombing, etc.)
- `GroundingLibrary` - 50+ grounding techniques for different conditions
- `MemoryStore` - RIME (Recursive Identity Memory Engine) for session continuity
- `PreRAGFilter` - Validates queries before processing
- `RAGRetriever` - Evidence-based knowledge retrieval

**Entropy States:**
- CRISIS (0.85-1.0) - Immediate intervention needed
- HIGH (0.65-0.84) - High distress, grounding required
- MODERATE (0.45-0.64) - Moderate distress
- LOW (0.25-0.44) - Low distress
- STABLE (0-0.24) - Stable state

**Mental Health Conditions Covered:**
- Anxiety, Depression, Trauma/PTSD, Dissociative disorders
- BPD, Bipolar, OCD, Eating disorders, Substance use
- Grief, ADHD, Autism, Psychosis, General distress

### 2. Specialized AI Modules

| Module | File | Purpose |
|--------|------|---------|
| Geometric Processing | `geometric.ts` | Torus-based emotional state modeling |
| Vicsek Flocking | `vicsek.ts` | Trajectory prediction for mood patterns |
| BPD Splitting | `bpd-splitting.ts` | Black/white thinking detection |
| Rural Support | `rural-support.ts` | Rural-specific resources and isolation support |
| Existential Support | `existential-support.ts` | Meaning crisis, death anxiety support |
| OCD/Phobias | `ocd-phobias.ts` | OCD subtypes and phobia detection |
| Belief Systems | `belief-systems.ts` | 22+ religious/philosophical worldviews |
| Languages | `languages.ts` | 30+ languages including Native American |
| Immigrant Support | `immigrant-support.ts` | Immigration anxiety, media literacy |
| Context Awareness | `context-awareness.ts` | Environmental/cultural adaptation |
| Techniques | `techniques.ts` | 50+ grounding and coping techniques |
| Resources | `resources.ts` | Crisis resources and hotlines |

### 3. Database Schema (`drizzle/schema.ts`)

**Core Tables:**
| Table | Purpose |
|-------|---------|
| `users` | User accounts with email/password auth |
| `sessions` | Active user sessions |
| `conversations` | Chat sessions with state tracking |
| `messages` | Individual messages with entropy metadata |
| `userMemory` | RIME memory persistence |
| `sessionAnalytics` | Session-level metrics |

**Feature Tables:**
| Table | Purpose |
|-------|---------|
| `safetyPlans` | Encrypted DV safety plans |
| `peerProfiles` | Anonymous peer support profiles |
| `peerConnections` | Peer-to-peer connections |
| `peerMessages` | Peer chat messages |
| `moderationActions` | Content moderation |
| `journalEntries` | User journal with entropy tracking |
| `journalInsights` | AI-generated journal insights |
| `therapistProfiles` | Licensed therapist info |
| `therapistClientRelationships` | Consent-based monitoring |

### 4. API Routes (`server/routers.ts`)

**Main Router Groups:**

```
appRouter
├── auth (login, logout, register, password reset)
├── reunity
│   ├── chat - Main AI chat endpoint
│   ├── processImage - OCR for images
│   ├── chatWithImage - Combined OCR + chat
│   ├── analyzeConversation - Pattern detection in screenshots
│   ├── analyzeJournal - Journal entry analysis
│   ├── loadMemory / saveMemory - RIME memory
│   ├── createConversation / getConversations
│   ├── exportConversation - HTML export
│   └── getCrisisResources
├── safetyPlan (get, save, export)
├── peerSupport
│   ├── getProfile / saveProfile
│   ├── getMatches / getConnections
│   ├── requestConnection / respondToConnection
│   └── getMessages / sendMessage
├── journal
│   ├── getEntries / createEntry / updateEntry
│   ├── getInsights / dismissInsight
│   └── getTrajectory - Vicsek-based predictions
└── system (notifyOwner)
```

---

## Frontend Architecture

### Pages (19 total)

| Page | Route | Purpose |
|------|-------|---------|
| Home | `/` | Landing page with feature overview |
| Chat | `/chat` | Main AI chat interface |
| Dashboard | `/dashboard` | User dashboard with all features |
| Journal | `/journal` | Journal entries with mood tracking |
| SafetyPlan | `/safety-plan` | DV safety plan builder |
| PeerSupport | `/peer-support` | Peer matching and chat |
| Resources | `/resources` | Crisis resources directory |
| TherapistPortal | `/therapist` | Therapist dashboard |
| Settings | `/settings` | User preferences |
| OfflineGrounding | `/grounding` | Offline grounding techniques |
| Login/Register | `/login`, `/register` | Authentication |
| LearnMore | `/learn-more` | About ReUnity |

### Components (53 total)

**Core Components:**
- `VoiceChat.tsx` - Voice chat with 5 personas (Gentle Woman, Gentle Man, Neutral, Warm Elder, Calm Friend)
- `AIChatBox.tsx` - Full-featured chat interface
- `DashboardLayout.tsx` - Sidebar navigation layout
- `PanicButton.tsx` - Emergency crisis button
- `ConsentDialog.tsx` - Terms acceptance

**Feature Components:**
- `DailyAffirmationsEnhanced.tsx` - Mood-based affirmations
- `SymptomTracker.tsx` - Physical symptom tracking
- `SocialConnectionPrompts.tsx` - Isolation detection
- `BreathingExercises.tsx` - Guided breathing
- `GuidedMeditation.tsx` / `GuidedMeditationLibrary.tsx`
- `MoodCalendar.tsx` / `MoodPrediction.tsx`
- `SleepTracker.tsx` / `SleepTracking.tsx`
- `MedicationReminder.tsx` / `MedicationInteractionChecker.tsx`
- `JournalWithSentiment.tsx`
- `CrisisInterventionTimeline.tsx`
- `EmergencyContacts.tsx`
- `ShelterFinder.tsx`
- `WearableIntegration.tsx`
- `Gamification.tsx` / `ProgressBadges.tsx`
- `LanguageSelector.tsx`

**Peer Support Components:**
- `PeerSupportMatching.tsx`
- `CommunityForum.tsx`
- `CommunitySupportGroups.tsx`
- `GroupTherapySessions.tsx`
- `FamilyGroupChat.tsx`

**Therapist Components:**
- `TherapistNotesSync.tsx`
- `TherapistScheduling.tsx`
- `AppointmentScheduler.tsx`
- `CaregiverDashboard.tsx`

**Security Components:**
- `BiometricAuth.tsx` / `BiometricLock.tsx`
- `TrustedDevicePairing.tsx`

---

## Data Flow

### Chat Message Flow

```
User Input
    ↓
Chat.tsx (frontend)
    ↓
trpc.reunity.chat.useMutation()
    ↓
routers.ts → reunity.chat procedure
    ↓
Load RIME memory (if authenticated)
    ↓
reunity.processMessage()
    ├── EntropyAnalyzer.analyze()
    ├── PatternRecognizer.analyze()
    ├── PreRAGFilter.shouldProcess()
    ├── RAGRetriever.retrieve()
    ├── GroundingLibrary.getForState()
    ├── Specialized modules (Vicsek, BPD, Rural, etc.)
    ├── Belief system detection
    ├── Language detection
    ├── Immigration anxiety detection
    └── LLM invocation with full context
    ↓
Save to database (messages, analytics)
    ↓
Return response to frontend
    ↓
Display with Streamdown markdown rendering
```

### Memory Persistence Flow

```
User mentions personal detail
    ↓
ReUnity detects memory-worthy content
    ↓
MemoryStore.addMemory()
    ↓
Response includes memoryUpdated: true
    ↓
Frontend calls saveRIMEMemory
    ↓
userMemory table updated
    ↓
Next session loads memory automatically
```

---

## Security Features

1. **Authentication**: Email/password with bcrypt hashing
2. **Sessions**: JWT tokens with expiry
3. **Biometric**: Optional fingerprint/face lock
4. **Encryption**: Safety plans encrypted at rest
5. **Consent**: Explicit consent for data sharing
6. **Moderation**: Peer chat monitoring for safety

---

## Crisis Handling

When crisis is detected (entropy > 0.85 or crisis keywords):

1. Immediate crisis resources displayed
2. 988 Suicide & Crisis Lifeline prominently shown
3. Grounding technique automatically offered
4. Response policy switches to "crisis" mode
5. If therapist connected, alert sent
6. Session flagged for review

---

## Test Coverage

443 tests across 16 test files covering:
- Entropy analysis
- Pattern recognition
- Grounding techniques
- Belief systems (58 tests)
- Geometric processing
- Context awareness
- Export functionality
- All feature modules

---

## File Structure

```
reunity-app/
├── client/
│   ├── src/
│   │   ├── pages/          # 19 page components
│   │   ├── components/     # 53 UI components
│   │   ├── contexts/       # Auth, Theme, Language contexts
│   │   ├── hooks/          # Custom React hooks
│   │   └── lib/            # tRPC client, utilities
│   └── public/             # Static assets
├── server/
│   ├── reunity.ts          # Main AI engine (3000+ lines)
│   ├── routers.ts          # tRPC API routes
│   ├── db.ts               # Database queries
│   ├── geometric.ts        # Torus emotional modeling
│   ├── vicsek.ts           # Trajectory prediction
│   ├── bpd-splitting.ts    # BPD support
│   ├── rural-support.ts    # Rural community support
│   ├── existential-support.ts
│   ├── ocd-phobias.ts
│   ├── belief-systems.ts   # 22+ worldviews
│   ├── languages.ts        # 30+ languages
│   ├── immigrant-support.ts
│   ├── context-awareness.ts
│   ├── techniques.ts       # 50+ techniques
│   ├── resources.ts        # Crisis resources
│   └── *.test.ts           # Test files
├── drizzle/
│   └── schema.ts           # Database schema
└── docs/
    ├── SYSTEM_ARCHITECTURE.md
    ├── DEPLOYMENT_GUIDE_WEB.md
    ├── DEPLOYMENT_GUIDE_IOS.md
    ├── DEPLOYMENT_GUIDE_ANDROID.md
    └── STRIPE_GUIDE.md
```

---

## Version History

- v1.0: Initial ReUnity AI with entropy analysis
- v2.0: Added pattern recognition, RIME memory
- v3.0: Database persistence, user accounts
- v4.0: Peer support, safety plans
- v5.0: Journal, therapist portal
- v6.0: Full mental health spectrum coverage
- v7.0: Geometric processing, Vicsek model
- v8.0: BPD, rural, existential support
- v9.0: OCD/phobias, enhanced grounding
- v10.0: Affirmations, symptoms, social prompts
- v11.0: Belief systems (22+ worldviews)
- v12.0: Multi-language (30+), immigrant support

---

*Created by Christopher Ezernack, REOP Solutions*
*This is a support tool, not a clinical treatment system.*
