# ReUnity

**A recursive AI companion for fragmented identity states.**

Built from physics. Built from pain. It does not surveil you. It mirrors you.

---

## Overview

ReUnity is a trauma-aware mental health support application that uses entropy physics principles to understand and support users experiencing dissociation, fragmented identity states, and mental health challenges.

### Core Features

- **Entropy-Based Mood Tracking** - Visualize mental state stability using physics-inspired metrics
- **Voice-Activated Check-ins** - Hands-free wellness confirmation during crisis situations
- **Trusted Device Pairing** - Share location and wellness data with family in emergencies
- **Therapist Portal** - Licensed providers can monitor consenting clients
- **Crisis Intervention Timeline** - Visual patterns to identify triggers and prevention strategies
- **Guided Meditation Library** - Curated sessions for anxiety, depression, trauma, and more
- **Peer Support Matching** - Anonymous connections with others who share similar experiences
- **Family Group Chat** - Coordinated support between linked family members

### AI Capabilities

- Multi-regime detection (algebraic, geometric, creative, planning)
- Image generation with DALL-E 3
- OCR for journal entries and conversation screenshots
- Sentiment analysis for journaling
- Mood prediction from HRV trends

---

## Quick Start

### Self-Hosted Deployment

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for complete Railway + OpenAI setup instructions.

```bash
# Clone repository
git clone https://github.com/yourusername/reunity-app.git
cd reunity-app

# Install dependencies
pnpm install

# Configure environment
cp docs/env.example.txt .env
# Edit .env with your OpenAI API key and database URL

# Run migrations
pnpm db:push

# Start development server
pnpm dev
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o and DALL-E 3 |
| `DATABASE_URL` | Yes | MySQL connection string |
| `JWT_SECRET` | Yes | 64-character random string |
| `NODE_ENV` | Yes | `production` or `development` |

---

## Technology Stack

- **Frontend**: React 19, Tailwind CSS 4, shadcn/ui
- **Backend**: Express 4, tRPC 11
- **Database**: MySQL/TiDB with Drizzle ORM
- **AI**: OpenAI GPT-4o, DALL-E 3
- **Deployment**: Railway, Vercel, or any Node.js host

---

## Project Structure

```
reunity-app/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   └── lib/            # Utilities and tRPC client
├── server/                 # Express + tRPC backend
│   ├── _core/              # Core services (LLM, auth, etc.)
│   ├── routers.ts          # tRPC procedures
│   └── db.ts               # Database helpers
├── drizzle/                # Database schema
├── docs/                   # Documentation
└── DEPLOYMENT_GUIDE.md     # Self-hosting instructions
```

---

## Testing

```bash
# Run all tests
pnpm test

# Run tests in watch mode
pnpm test:watch
```

---

## Mobile App

The React Native mobile app is available in a separate package. See `reunity-mobile-v10.zip` for the latest version.

---

## Links

- **Main Website**: [entropy-physics-ai.com](https://entropy-physics-ai.com)
- **Documentation**: [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

---

## License

ReUnity is proprietary software by REOP Solutions.

**Created by Christopher Ezernack**

© 2026 REOP Solutions. All rights reserved.

---

*"We don't disappear. We reorganize."*
