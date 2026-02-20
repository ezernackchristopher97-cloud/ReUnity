# Migration Instructions from Christopher

## Context
Christopher is moving away from Manus hosting due to billing issues. Need to prepare ReUnity for self-hosting on Railway + OpenAI.

## Critical Instructions

### 1. Self-Hosting Migration Required
- Create a complete tar.gz package of the ReUnity web app
- Update the app to use OpenAI API directly instead of Manus Forge API
- Create a step-by-step deployment guide (similar to REOP-AI)
- Push all code to the GitHub repository

### 2. Deployment Configuration
```
Platform: Railway (https://railway.app)
LLM Provider: OpenAI API (GPT-4o + DALL-E 3)
Database: Railway MySQL or PostgreSQL
Estimated Cost: ~$55/month total
```

### 3. Required Environment Variables
```
OPENAI_API_KEY=sk-proj-xxx (user will provide)
DATABASE_URL=(Railway auto-generates)
JWT_SECRET=(generate 64-char random string)
NODE_ENV=production
```

### 4. Domain Setup
- DNS configuration instructions
- CNAME record setup
- SSL/HTTPS enablement
- Custom domain verification

### 5. Code Quality Requirements
- Remove all AI narration/meta-language from code
- Update version number
- Ensure all features match the web app
- Test image generation functionality
- Verify regime detection works correctly

### 6. Feature Parity Checklist (same as REOP-AI)
- [ ] Multi-regime detection (algebraic, geometric, creative, planning, etc.)
- [ ] Image generation with architecture details in collapsible section
- [ ] Mobile-responsive design
- [ ] Email templates with proper styling
- [ ] User authentication
- [ ] Chat history persistence
- [ ] OCR/image analysis

### 7. Deliverables Required
1. tar.gz package - Complete app ready for self-hosting
2. Deployment guide - Step-by-step for Railway + OpenAI
3. README - Professional documentation
4. GitHub push - All code in repository

## Technical Reference

### OpenAI API Integration
Replace Manus Forge API calls with:
```typescript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// For chat completions
const response = await openai.chat.completions.create({
  model: 'gpt-4o',
  messages: messages,
});

// For image generation
const image = await openai.images.generate({
  model: 'dall-e-3',
  prompt: prompt,
  size: '1024x1024',
});
```

### Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link
railway login
railway link

# Deploy
railway up
```

## Priority Level: HIGH
Christopher wants this done ASAP. Do not ask questions - make decisions and execute.
The goal is a fully functional, self-hosted product.
