# ReUnity Complete Deployment Guide

**A recursive AI companion for fragmented identity states.**

Built from physics. Built from pain. It does not surveil you. It mirrors you.

**Author:** Christopher Ezernack, REOP Solutions

**68,333+ lines of code | 443 tests passing | 55 server modules | 106 components | 19 pages**

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Prerequisites for Deployment](#prerequisites-for-deployment)
4. [Part 1: Web App Deployment to Railway](#part-1-web-app-deployment-to-railway)
5. [Part 2: iOS App Store Deployment](#part-2-ios-app-store-deployment)
6. [Part 3: Google Play Store Deployment](#part-3-google-play-store-deployment)
7. [Part 4: Stripe Payment Integration](#part-4-stripe-payment-integration)
8. [Module Reference](#module-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

ReUnity is a trauma-aware mental health support application that uses entropy physics principles to understand and support users experiencing dissociation, fragmented identity states, and mental health challenges.

### Technology Stack

- **Frontend:** React 19, TypeScript, Tailwind CSS 4, shadcn/ui
- **Backend:** Express 4, tRPC 11, Drizzle ORM
- **Database:** MySQL/TiDB
- **AI Engine:** Custom ReUnity AI with entropy-based emotional analysis
- **Mobile:** React Native with Expo

### Core Features

- **Entropy-Based Mood Tracking** - Visualize mental state stability using physics-inspired metrics
- **Voice-Activated Check-ins** - Hands-free wellness confirmation during crisis situations
- **Trusted Device Pairing** - Share location and wellness data with family in emergencies
- **Therapist Portal** - Licensed providers can monitor consenting clients
- **Crisis Intervention Timeline** - Visual patterns to identify triggers and prevention strategies
- **Guided Meditation Library** - Curated sessions for anxiety, depression, trauma, and more
- **Peer Support Matching** - Anonymous connections with others who share similar experiences
- **Family Group Chat** - Coordinated support between linked family members
- **30+ Language Support** - Including Native American languages (Navajo, Cherokee, Lakota, Ojibwe, Apache)
- **22+ Belief Systems** - Religious, spiritual, and philosophical frameworks
- **Immigrant/Refugee Support** - Grounding techniques and media literacy
- **Voice Chat** - 5 voice personas (Gentle Woman, Gentle Man, Neutral, Warm Elder, Calm Friend)

### AI Capabilities

- Multi-regime detection (algebraic, geometric, creative, planning)
- RIME memory continuity across sessions
- Entropy-based emotional state analysis
- Pattern detection and intervention
- Image generation with DALL-E 3
- OCR for journal entries and conversation screenshots
- Sentiment analysis for journaling
- Mood prediction from HRV trends

---

## System Architecture

### Server Modules (55 files, 40,000+ lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| reunity.ts | 3,000+ | Core AI engine with entropy analysis, pattern detection, RAG, grounding |
| routers.ts | 1,658 | Full tRPC API with 50+ endpoints |
| geometric.ts | 800+ | Geometric regime detection and Vicsek model integration |
| belief-systems.ts | 1,200+ | 22+ worldviews with culturally-sensitive support |
| languages.ts | 1,500+ | 30+ languages with greetings and comforting phrases |
| immigrant-support.ts | 800+ | Immigration anxiety support and media literacy |
| bpd-splitting.ts | 600+ | BPD-specific interventions |
| vicsek.ts | 500+ | Collective behavior modeling |
| rural-support.ts | 400+ | Rural healthcare desert support |
| existential-support.ts | 400+ | Existential crisis support |
| ocd-phobias.ts | 400+ | OCD and phobia-specific techniques |
| techniques.ts | 600+ | 50+ therapeutic techniques |
| context-awareness.ts | 500+ | Context detection and adaptation |
| safety-planning.ts | 400+ | Crisis safety planning |
| peer-support.ts | 400+ | Peer matching algorithms |
| journaling.ts | 300+ | Journal analysis and insights |

### Client Components (106 files, 25,000+ lines)

| Category | Count | Examples |
|----------|-------|----------|
| Core UI | 53 | VoiceChat, SymptomTracker, SocialConnectionPrompts |
| Dashboard | 8 | DashboardLayout, MoodCalendar, ProgressBadges |
| Crisis | 6 | PanicButton, EmergencyContacts, OfflineCrisisCard |
| Wellness | 12 | BreathingExercises, GuidedMeditation, SleepTracker |
| Social | 8 | PeerSupportMatching, CommunityForum, FamilyGroupChat |
| Settings | 5 | AccessibilitySettings, LanguageSelector, BiometricAuth |
| shadcn/ui | 53 | Button, Card, Dialog, Select, etc. |

### Client Pages (19 files)

| Page | Purpose |
|------|---------|
| Home.tsx | Landing page with feature overview |
| Chat.tsx | Main AI conversation interface |
| Dashboard.tsx | User dashboard with mood tracking, symptoms, social |
| Journal.tsx | Journaling with sentiment analysis |
| SafetyPlan.tsx | Crisis safety plan creation |
| PeerSupport.tsx | Peer matching and messaging |
| Resources.tsx | Mental health resources |
| TherapistPortal.tsx | Therapist dashboard |
| Settings.tsx | User preferences |
| OfflineGrounding.tsx | Offline grounding techniques |

### Database Schema (15 tables)

```
users              - User accounts and profiles
sessions           - Authentication sessions
conversations      - Chat conversation metadata
messages           - Individual chat messages
userMemory         - RIME memory storage for AI continuity
safetyPlans        - Crisis safety plans
peerProfiles       - Peer support profiles
peerConnections    - Peer matching connections
peerMessages       - Peer-to-peer messages
journalEntries     - Journal entries
journalInsights    - AI-generated journal insights
therapistProfiles  - Therapist information
appointments       - Scheduled appointments
medicationReminders - Medication tracking
symptomLogs        - Physical symptom tracking
```

---

## Prerequisites for Deployment

### What You Need Before Starting

| Item | Cost | Where to Get It |
|------|------|-----------------|
| Mac Computer | You have this | - |
| Apple Developer Account | $99/year | https://developer.apple.com |
| Google Play Developer Account | $25 one-time | https://play.google.com/console |
| Railway Account | Free tier available | https://railway.app |
| Stripe Account | Free (2.9% + $0.30 per transaction) | https://stripe.com |
| Expo Account | Free | https://expo.dev |

---

## Part 1: Web App Deployment to Railway

### Step 1: Install Homebrew (Mac Package Manager)

Open Terminal on your Mac. To do this:
1. Press Command + Space on your keyboard
2. Type "Terminal"
3. Press Enter

You should see a window with a command prompt that looks like this:
```
yourusername@yourcomputer ~ %
```

Now copy and paste this entire command (including the quotes), then press Enter:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

What happens next:
- It will ask for your Mac login password
- Type your password (you won't see any characters appear - that's normal and secure)
- Press Enter
- Wait for it to finish (about 2-5 minutes)
- You'll see "Installation successful!" when done

### Step 2: Install Node.js

Node.js is what runs the server code. In Terminal, copy and paste this command:

```bash
brew install node
```

Press Enter and wait for it to finish (1-2 minutes).

To verify it worked, type:

```bash
node --version
```

Press Enter. You should see something like `v22.13.0` (the numbers might be slightly different).

### Step 3: Install pnpm (Package Manager)

pnpm is a tool that downloads and manages code packages. In Terminal:

```bash
npm install -g pnpm
```

Press Enter and wait for it to finish.

Verify it worked:

```bash
pnpm --version
```

You should see a version number like `9.15.0`.

### Step 4: Install Git (Version Control)

Git tracks changes to your code. In Terminal:

```bash
brew install git
```

Press Enter and wait.

Verify:

```bash
git --version
```

### Step 5: Install Railway CLI

Railway CLI lets you deploy from Terminal. In Terminal:

```bash
npm install -g @railway/cli
```

Press Enter and wait.

### Step 6: Download the ReUnity Code

First, create a folder for your projects:

```bash
mkdir -p ~/Projects
cd ~/Projects
```

The first command creates a "Projects" folder in your home directory.
The second command moves you into that folder.

**Option A: If you have the zip file:**

If you downloaded reunity-app.zip, find where it is (probably Downloads folder) and run:

```bash
unzip ~/Downloads/reunity-app.zip -d ~/Projects/
cd ~/Projects/reunity-app
```

**Option B: If cloning from GitHub:**

```bash
git clone https://github.com/ezernackchristopher97-cloud/ReUnity.git reunity-app
cd reunity-app
```

### Step 7: Install Project Dependencies

This downloads all the code packages that ReUnity needs:

```bash
pnpm install
```

Press Enter and wait. This takes 2-5 minutes. You'll see a lot of text scrolling by - that's normal.

When it's done, you'll see your command prompt again.

### Step 8: Create Railway Account

1. Open your web browser (Safari, Chrome, etc.)
2. Go to https://railway.app
3. Click "Login" in the top right corner
4. Click "Login with GitHub" (this is the easiest option)
5. If you don't have a GitHub account, click "Create an account" and follow the steps
6. Authorize Railway to access your GitHub account

### Step 9: Connect Terminal to Railway

Back in Terminal, type:

```bash
railway login
```

Press Enter. This opens a browser window. Click "Authorize" to connect your Terminal to Railway.

You'll see "Logged in as [your username]" in Terminal when successful.

### Step 10: Create a New Railway Project

In Terminal:

```bash
railway init
```

When it asks what to do:
- Use arrow keys to select "Empty Project"
- Press Enter
- Type a name: `reunity-app`
- Press Enter

### Step 11: Add a Database

Your app needs a database to store user data. In Terminal:

```bash
railway add
```

Use arrow keys to select "MySQL" from the list, then press Enter.

Wait 1-2 minutes for the database to be created.

### Step 12: Link Your Code to Railway

```bash
railway link
```

Select your project from the list.

### Step 13: Set Up Environment Variables

Environment variables are secret settings your app needs. Go to Railway in your browser:

1. Go to https://railway.app/dashboard
2. Click on your "reunity-app" project
3. You'll see two boxes - one for your app, one for MySQL
4. Click on the app box (not MySQL)
5. Click the "Variables" tab

Now add these variables one at a time. For each one:
- Click "New Variable"
- Type the name in the first box
- Type the value in the second box
- Click the checkmark

| Variable Name | What to Put |
|--------------|-------------|
| NODE_ENV | production |
| JWT_SECRET | Make up a random string of 32+ characters like: xK9mP2nQ5rT8vW1yB4cF7hJ0kL3oS6uA |

The DATABASE_URL is automatically set by Railway when you added MySQL.

### Step 14: Deploy Your App

This is the big moment! In Terminal:

```bash
railway up
```

Press Enter and wait. This uploads your code to Railway and starts your app. It takes 3-5 minutes.

You'll see a lot of text. When it's done, you'll see "Deployment successful" or similar.

### Step 15: Get Your Public URL

Your app needs a web address people can visit. In Terminal:

```bash
railway domain
```

This creates a URL like `reunity-app-production.up.railway.app`.

Copy this URL - you'll need it!

### Step 16: Run Database Setup

Your database needs tables created. In Terminal:

```bash
railway run pnpm db:push
```

This creates all the database tables your app needs.

### Step 17: Test Your Deployment

1. Open your web browser
2. Go to the URL from Step 15 (like `reunity-app-production.up.railway.app`)
3. You should see the ReUnity welcome page!

**Congratulations! Your web app is now live on the internet!**

---

## Part 2: iOS App Store Deployment

### Step 1: Create Apple Developer Account

This costs $99/year but is required to publish on the App Store.

1. Go to https://developer.apple.com
2. Click "Account" in the top right
3. Sign in with your Apple ID
   - If you don't have an Apple ID, click "Create Apple ID" and follow the steps
4. Once signed in, click "Join the Apple Developer Program"
5. Click "Enroll"
6. Choose "Individual" (unless you have a company)
7. Fill in your information
8. Pay $99
9. Wait for approval (usually 24-48 hours - Apple will email you)

### Step 2: Install Xcode

Xcode is Apple's app development software. It's free but large (12+ GB).

1. Open the App Store on your Mac (click the blue "A" icon in your dock)
2. Search for "Xcode"
3. Click "Get" then "Install"
4. Wait for download (this can take 30-60 minutes depending on your internet)
5. Once installed, open Xcode from your Applications folder
6. Accept the license agreement when prompted
7. Let it install additional components (another 5-10 minutes)

### Step 3: Install Expo CLI Tools

Expo makes building mobile apps easier. In Terminal:

```bash
npm install -g expo-cli eas-cli
```

Press Enter and wait.

### Step 4: Create Expo Account

1. Go to https://expo.dev in your browser
2. Click "Sign Up"
3. Create an account with your email

Now connect Terminal to Expo:

```bash
eas login
```

Enter your Expo email and password when prompted.

### Step 5: Navigate to Mobile App Folder

If you have the mobile app zip file:

```bash
cd ~/Projects
unzip ~/Downloads/reunity-mobile-v12.zip
cd reunity-mobile/reunity-mobile
```

Or if it's already extracted:

```bash
cd ~/Projects/reunity-mobile/reunity-mobile
```

### Step 6: Install Mobile App Dependencies

```bash
pnpm install
```

Wait for it to finish (2-3 minutes).

### Step 7: Update App Configuration

You need to edit a file called `app.json`. 

Open it in a text editor. You can use TextEdit on Mac:

```bash
open -a TextEdit app.json
```

Find and update these values:

```json
{
  "expo": {
    "name": "ReUnity",
    "slug": "reunity",
    "version": "1.0.0",
    "ios": {
      "bundleIdentifier": "com.reopsolutions.reunity",
      "buildNumber": "1"
    },
    "android": {
      "package": "com.reopsolutions.reunity",
      "versionCode": 1
    }
  }
}
```

Save the file (Command + S) and close TextEdit.

### Step 8: Configure EAS Build

In Terminal:

```bash
eas build:configure
```

When it asks which platforms, select "All" using arrow keys and press Enter.

### Step 9: Build for iOS

This creates the app file Apple needs:

```bash
eas build --platform ios
```

When prompted:
- "Would you like to log in to your Apple Developer account?" - Type `Y` and press Enter
- Enter your Apple ID email
- Enter your Apple ID password
- If asked about two-factor authentication, enter the code from your phone
- "Select a team" - Choose your name/company
- "Would you like EAS to manage your certificates?" - Type `Y` and press Enter

Now wait. This takes 15-30 minutes. EAS is building your app on their servers.

When done, you'll see a URL to download your app file. You don't need to download it - EAS will use it directly.

### Step 10: Submit to App Store

```bash
eas submit --platform ios
```

When prompted:
- Select the build you just created
- Enter your App Store Connect credentials (same as Apple Developer account)

### Step 11: Complete App Store Listing

Now you need to fill in information about your app on Apple's website.

1. Go to https://appstoreconnect.apple.com
2. Sign in with your Apple ID
3. Click "My Apps"
4. Click on "ReUnity" (it should appear after the submit)
5. Click "App Information" in the left sidebar

Fill in these fields:

**Name:** ReUnity

**Subtitle:** Trauma-Aware AI Companion

**Category:** Health & Fitness

**Secondary Category:** Medical

**Privacy Policy URL:** https://[your-railway-url]/privacy
(Replace [your-railway-url] with your actual Railway URL from Part 1)

Now click "Prepare for Submission" in the left sidebar.

**Description (copy this entire text):**
```
ReUnity is a trauma-aware AI companion designed to support individuals experiencing fragmented identity states, emotional dysregulation, and mental health challenges.

Built from physics. Built from pain. It does not surveil you. It mirrors you.

FEATURES:

• AI Companion with Memory
ReUnity remembers your conversations across sessions, providing continuity and understanding that grows over time.

• 30+ Languages
Support in English, Spanish, Hindi, Arabic, Mandarin, and Native American languages including Navajo, Cherokee, and Lakota.

• Crisis Support
Immediate access to crisis resources, safety planning tools, and grounding techniques when you need them most.

• Voice Chat
Talk to ReUnity using your voice. Choose from 5 different voice personas that feel comfortable to you.

• Guided Meditations
Curated meditation sessions for anxiety, depression, trauma recovery, and more.

• Peer Support
Connect anonymously with others who share similar experiences.

• Mood Tracking
Visualize your emotional patterns over time with entropy-based mood analysis.

• Symptom Tracking
Track physical symptoms and see how they correlate with your emotional state.

• 22+ Belief Systems
Culturally-sensitive support honoring your religious, spiritual, or philosophical background.

IMPORTANT DISCLAIMERS:

• ReUnity is NOT a substitute for professional mental health care, therapy, or medical treatment.

• The AI companion is NOT a licensed therapist, counselor, or healthcare provider.

• Nothing in this app constitutes medical advice, diagnosis, or treatment.

• If you are in crisis, please contact emergency services or call 988 (Suicide & Crisis Lifeline).

Created by Christopher Ezernack, REOP Solutions.

"We don't disappear. We reorganize."
```

**Keywords:** mental health, therapy, AI companion, trauma, anxiety, depression, meditation, mood tracker, crisis support, peer support

**Support URL:** https://[your-railway-url]/support

**Marketing URL:** https://[your-railway-url]

### Step 12: Add Screenshots

You need screenshots of your app. The required sizes are:
- 6.5 inch iPhone (1284 x 2778 pixels)
- 5.5 inch iPhone (1242 x 2208 pixels)

To take screenshots:
1. Run the app in Xcode Simulator
2. Press Command + S to save a screenshot
3. Upload to App Store Connect

### Step 13: Submit for Review

1. Make sure all required fields have green checkmarks
2. Click "Submit for Review"
3. Answer the questions:
   - "Does this app use the Advertising Identifier?" - No
   - "Does this app use encryption?" - No (unless you added custom encryption)
4. Click "Submit"

Apple will review your app within 24-48 hours. They'll email you when it's approved (or if they need changes).

---

## Part 3: Google Play Store Deployment

### Step 1: Create Google Play Developer Account

This costs $25 one-time.

1. Go to https://play.google.com/console
2. Sign in with your Google account
3. Click "Create account" or "Get started"
4. Pay the $25 registration fee
5. Fill in your developer information
6. Wait for verification (can take 24-48 hours)

### Step 2: Build for Android

In Terminal, make sure you're in the mobile app folder:

```bash
cd ~/Projects/reunity-mobile/reunity-mobile
```

Now build:

```bash
eas build --platform android
```

When prompted:
- "Generate a new Android Keystore?" - Type `Y` and press Enter
- Wait for the build (15-30 minutes)

When done, you'll get a URL to download the .aab file. Copy this URL.

### Step 3: Download the Build File

```bash
cd ~/Downloads
curl -o reunity.aab "[paste the URL here]"
```

Replace `[paste the URL here]` with the actual URL from the previous step.

### Step 4: Create App in Play Console

1. Go to https://play.google.com/console
2. Click "Create app"
3. Fill in:
   - App name: ReUnity
   - Default language: English (United States)
   - App or game: App
   - Free or paid: Free
4. Check the boxes for declarations
5. Click "Create app"

### Step 5: Set Up Store Listing

In the left sidebar, click "Main store listing"

**App name:** ReUnity

**Short description (80 characters max):**
```
Trauma-aware AI companion for mental health support and emotional wellness.
```

**Full description (copy the same description from iOS above)**

### Step 6: Add Graphics

You need:
- **App icon:** 512 x 512 pixels PNG
- **Feature graphic:** 1024 x 500 pixels PNG  
- **Phone screenshots:** At least 2 screenshots, minimum 320px, maximum 3840px

Upload these in the "Graphics" section.

### Step 7: Complete Content Rating

1. In left sidebar, click "Content rating"
2. Click "Start questionnaire"
3. Answer the questions honestly:
   - Violence: No
   - Sexual content: No
   - Language: No
   - Controlled substances: No
   - User-generated content: Yes (if you have peer messaging)
4. Click "Save" then "Submit"

### Step 8: Set Up Pricing and Distribution

1. Click "Countries/regions" in left sidebar
2. Click "Add countries/regions"
3. Select all countries where you want the app available
4. Click "Add countries/regions"

### Step 9: Upload Your App

1. In left sidebar, click "Production" under "Release"
2. Click "Create new release"
3. Click "Upload" and select the .aab file you downloaded
4. Wait for upload to complete
5. In "Release notes" type: "Initial release of ReUnity"
6. Click "Save"
7. Click "Review release"
8. Click "Start rollout to Production"

### Step 10: Submit for Review

Google will automatically review your app. This typically takes 3-7 days for new apps.

You'll receive an email when your app is approved and live on the Play Store.

---

## Part 4: Stripe Payment Integration

### Step 1: Create Stripe Account

1. Go to https://stripe.com
2. Click "Start now"
3. Enter your email and create a password
4. Verify your email
5. Fill in your business information

### Step 2: Get Your API Keys

1. In Stripe Dashboard, click "Developers" in the left sidebar
2. Click "API keys"
3. You'll see two keys:
   - **Publishable key** - starts with `pk_test_`
   - **Secret key** - starts with `sk_test_` (click "Reveal" to see it)
4. Copy both keys somewhere safe

### Step 3: Add Keys to Railway

1. Go to https://railway.app/dashboard
2. Click on your reunity-app project
3. Click on your app service
4. Click "Variables" tab
5. Add these new variables:

| Variable Name | Value |
|--------------|-------|
| STRIPE_SECRET_KEY | sk_test_... (your secret key) |
| VITE_STRIPE_PUBLISHABLE_KEY | pk_test_... (your publishable key) |

6. Click "Deploy" to restart with new variables

### Step 4: Create Products in Stripe

1. In Stripe Dashboard, click "Products" in left sidebar
2. Click "Add product"

**Product 1: Free Tier**
- Name: ReUnity Free
- Description: Basic access to ReUnity
- Click "Add product"
- Click "Add price"
- Price: $0.00
- Billing period: Monthly
- Click "Add price"

**Product 2: Premium**
- Name: ReUnity Premium  
- Description: Full access to all features
- Click "Add product"
- Click "Add price"
- Price: $9.99
- Billing period: Monthly
- Click "Add price"

**Product 3: Professional**
- Name: ReUnity Professional
- Description: Premium plus therapist portal
- Click "Add product"
- Click "Add price"
- Price: $19.99
- Billing period: Monthly
- Click "Add price"

### Step 5: Set Up Webhooks

Webhooks let Stripe notify your app when payments happen.

1. In Stripe Dashboard, click "Developers" > "Webhooks"
2. Click "Add endpoint"
3. Endpoint URL: `https://[your-railway-url]/api/stripe/webhook`
   (Replace [your-railway-url] with your actual URL)
4. Click "Select events"
5. Check these events:
   - checkout.session.completed
   - customer.subscription.created
   - customer.subscription.updated
   - customer.subscription.deleted
6. Click "Add events"
7. Click "Add endpoint"
8. Click on your new endpoint
9. Under "Signing secret", click "Reveal"
10. Copy the secret (starts with `whsec_`)

### Step 6: Add Webhook Secret to Railway

1. Go to Railway dashboard
2. Add new variable:

| Variable Name | Value |
|--------------|-------|
| STRIPE_WEBHOOK_SECRET | whsec_... (your webhook secret) |

3. Redeploy

### Step 7: Test Payments

1. Go to your app
2. Try to subscribe to Premium
3. Use test card number: 4242 4242 4242 4242
4. Any future expiry date
5. Any 3-digit CVC
6. Any billing address

### Step 8: Go Live with Real Payments

When you're ready for real money:

1. Complete Stripe account verification (they'll ask for ID and bank info)
2. In Stripe Dashboard, toggle from "Test mode" to "Live mode"
3. Get new live API keys (they start with `pk_live_` and `sk_live_`)
4. Update Railway variables with live keys
5. Create new webhook endpoint for live mode
6. Update STRIPE_WEBHOOK_SECRET with live webhook secret

---

## Module Reference

### AI Core Modules

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| ReUnity AI Engine | reunity.ts | 3,000+ | Main AI processing with entropy analysis, pattern detection, RIME memory |
| Geometric Processing | geometric.ts | 800+ | Regime detection using geometric models |
| Vicsek Model | vicsek.ts | 500+ | Collective behavior modeling for group dynamics |
| Context Awareness | context-awareness.ts | 500+ | Detects conversation context and adapts responses |

### Therapeutic Modules

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Techniques | techniques.ts | 600+ | 50+ therapeutic techniques (CBT, DBT, ACT, etc.) |
| BPD Support | bpd-splitting.ts | 600+ | Borderline personality disorder interventions |
| OCD/Phobias | ocd-phobias.ts | 400+ | OCD and phobia-specific support |
| Existential Support | existential-support.ts | 400+ | Existential crisis and meaning-making |

### Cultural Modules

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Belief Systems | belief-systems.ts | 1,200+ | 22+ religious/philosophical frameworks |
| Languages | languages.ts | 1,500+ | 30+ languages with native phrases |
| Immigrant Support | immigrant-support.ts | 800+ | Immigration anxiety and media literacy |
| Rural Support | rural-support.ts | 400+ | Healthcare desert support |

### Safety Modules

| Module | File | Lines | Description |
|--------|------|-------|-------------|
| Safety Planning | safety-planning.ts | 400+ | Crisis safety plan creation |
| Resources | resources.ts | 300+ | Mental health resource database |
| Shelter API | shelter-api.ts | 200+ | Domestic violence shelter finder |

### Complete Module List

```
Server Modules (55 files):
├── _core/
│   ├── context.ts
│   ├── cookies.ts
│   ├── dataApi.ts
│   ├── env.ts
│   ├── imageGeneration.ts
│   ├── index.ts
│   ├── llm.ts
│   ├── map.ts
│   ├── notification.ts
│   ├── oauth.ts
│   ├── sdk.ts
│   ├── systemRouter.ts
│   ├── trpc.ts
│   ├── vite.ts
│   └── voiceTranscription.ts
├── auth.ts
├── belief-systems.ts
├── bpd-splitting.ts
├── context-awareness.ts
├── db.ts
├── existential-support.ts
├── geometric.ts
├── immigrant-support.ts
├── journaling.ts
├── languages.ts
├── ocd-phobias.ts
├── ocr.ts
├── peer-support.ts
├── resources.ts
├── reunity.ts
├── routers.ts
├── rural-support.ts
├── safety-planning.ts
├── shelter-api.ts
├── storage.ts
├── techniques.ts
└── vicsek.ts

Test Files (16 files, 443 tests):
├── auth.logout.test.ts
├── belief-systems.test.ts
├── context-awareness.test.ts
├── export.test.ts
├── geometric.test.ts
├── new-features.test.ts
├── new-modules.test.ts
├── phase3-features.test.ts
├── phase4-features.test.ts
├── phase5-features.test.ts
├── phase6-features.test.ts
├── phase7-features.test.ts
├── phase8-features.test.ts
├── phase9-features.test.ts
├── phase10-features.test.ts
└── techniques.test.ts
```

---

## Troubleshooting

### "Command not found" Errors

If Terminal says a command isn't found:

```bash
# Close Terminal and reopen it, OR run:
source ~/.zshrc
```

### Railway Deployment Fails

Check what went wrong:

```bash
railway logs
```

This shows the error messages. Common fixes:

```bash
# Redeploy:
railway up --detach

# Check your variables are set:
railway variables
```

### Database Connection Errors

```bash
# Make sure DATABASE_URL is set:
railway variables

# Re-run database setup:
railway run pnpm db:push
```

### iOS Build Fails

```bash
# Clear the cache and try again:
eas build --platform ios --clear-cache
```

### Android Build Fails

```bash
# Clear the cache and try again:
eas build --platform android --clear-cache
```

### App Crashes on Startup

Check the logs:
- Railway: `railway logs`
- iOS: Open Xcode > Window > Devices and Simulators > Select your device > View Device Logs
- Android: `adb logcat`

### Stripe Payments Not Working

1. Make sure you're using test mode keys for testing
2. Check webhook endpoint URL is correct
3. Verify webhook secret is set in Railway
4. Check Stripe Dashboard > Developers > Webhooks for errors

### Getting More Help

- Railway Documentation: https://docs.railway.app
- Expo Documentation: https://docs.expo.dev
- Apple Developer Documentation: https://developer.apple.com/documentation
- Google Play Documentation: https://developer.android.com/distribute
- Stripe Documentation: https://stripe.com/docs

---

## Running Tests

To verify everything works:

```bash
cd ~/Projects/reunity-app
pnpm test
```

You should see:
```
Test Files  16 passed (16)
     Tests  443 passed (443)
```

---

## License

Copyright 2024-2026 Christopher Ezernack, REOP Solutions. All rights reserved.

**DISCLAIMER:** This document describes a research framework and is not a clinical or treatment tool. It does not provide medical advice, diagnosis, or treatment. If you are in crisis, please contact emergency services or call 988 (Suicide & Crisis Lifeline).

---

*"We don't disappear. We reorganize."*
