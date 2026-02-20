# ReUnity iOS App Store Deployment Guide

**Complete Step-by-Step Instructions for Publishing to Apple App Store (No Coding Experience Required)**

This guide walks you through building the ReUnity mobile app and publishing it to the Apple App Store. Every step is explained in detail for Mac users.

---

## Table of Contents

1. [Prerequisites and Costs](#part-1-prerequisites-and-costs)
2. [Setting Up Your Apple Developer Account](#part-2-setting-up-your-apple-developer-account)
3. [Installing Development Tools](#part-3-installing-development-tools)
4. [Preparing the Mobile App Code](#part-4-preparing-the-mobile-app-code)
5. [Configuring Your App for iOS](#part-5-configuring-your-app-for-ios)
6. [Building the iOS App](#part-6-building-the-ios-app)
7. [Testing with TestFlight](#part-7-testing-with-testflight)
8. [Submitting to the App Store](#part-8-submitting-to-the-app-store)
9. [App Store Review Process](#part-9-app-store-review-process)

---

## Part 1: Prerequisites and Costs

Before you begin, here's what you need:

| Requirement | Details | Cost |
|-------------|---------|------|
| Mac Computer | Required for iOS development | You already have this |
| Apple ID | Your regular Apple account | Free |
| Apple Developer Account | Required to publish apps | $99/year |
| Xcode | Apple's development software | Free (from App Store) |
| The ReUnity mobile app code | The reunity-mobile folder | You already have this |

**Important**: You cannot publish iOS apps without paying the $99/year Apple Developer fee. There is no way around this - it's Apple's requirement.

---

## Part 2: Setting Up Your Apple Developer Account

### Step 2.1: Enroll in the Apple Developer Program

1. Open Safari and go to [developer.apple.com/programs/enroll](https://developer.apple.com/programs/enroll).

2. Click **Start Your Enrollment**.

3. Sign in with your Apple ID (the same one you use for your iPhone/Mac).

4. If you don't have an Apple ID, click **Create Apple ID** and follow the steps.

5. You'll be asked to verify your identity. Apple may require:
   - A phone number for verification
   - A credit card on file
   - In some cases, a government ID

6. Select **Individual** enrollment (unless you have a registered business).

7. Review and accept the Apple Developer Agreement.

8. Pay the $99 annual fee with a credit card.

9. **Wait 24-48 hours** for Apple to process your enrollment. You'll receive an email when approved.

### Step 2.2: Verify Your Enrollment

Once approved:

1. Go to [developer.apple.com](https://developer.apple.com).
2. Click **Account** in the top right.
3. Sign in with your Apple ID.
4. You should see your Developer dashboard with options like "Certificates, IDs & Profiles".

If you see this dashboard, your account is ready!

---

## Part 3: Installing Development Tools

### Step 3.1: Install Xcode

Xcode is Apple's official development software. It's large (about 12GB) so this will take a while.

1. Open the **App Store** on your Mac (click the blue "A" icon in your Dock, or search for "App Store" in Spotlight).

2. In the search bar, type **Xcode** and press Enter.

3. Find **Xcode** by Apple (it has a hammer icon on a blue background).

4. Click **Get**, then **Install**.

5. Enter your Apple ID password if prompted.

6. **Wait 30-60 minutes** for the download and installation (depends on your internet speed).

7. Once installed, open Xcode from your Applications folder.

8. Xcode will ask to install additional components. Click **Install** and enter your Mac password.

9. Wait another 5-10 minutes for components to install.

### Step 3.2: Accept Xcode License

Open Terminal and type:

```bash
sudo xcodebuild -license accept
```

Enter your Mac password when prompted.

### Step 3.3: Install Xcode Command Line Tools

In Terminal:

```bash
xcode-select --install
```

A popup will appear. Click **Install**, then **Agree** to the license.

### Step 3.4: Install Node.js and pnpm (If Not Already Installed)

If you followed the web deployment guide, skip this. Otherwise:

```bash
brew install node
```

```bash
npm install -g pnpm
```

### Step 3.5: Install Expo CLI

Expo is the framework that makes building React Native apps easier:

```bash
npm install -g expo-cli eas-cli
```

Verify installation:

```bash
eas --version
```

You should see something like `eas-cli/x.x.x`.

### Step 3.6: Log Into Expo

Create an Expo account if you don't have one:

1. Go to [expo.dev](https://expo.dev) in your browser.
2. Click **Sign Up** and create an account.
3. Return to Terminal and log in:

```bash
eas login
```

Enter your Expo username and password.

---

## Part 4: Preparing the Mobile App Code

### Step 4.1: Navigate to the Mobile App Folder

If you have the reunity-mobile ZIP file:

1. Find the ZIP file in Finder.
2. Double-click to unzip it.
3. Move the unzipped folder to your Projects folder.

In Terminal:

```bash
cd ~/Projects
```

If the folder is named `reunity-mobile`:

```bash
cd reunity-mobile/reunity-mobile
```

### Step 4.2: Install Dependencies

```bash
pnpm install
```

Wait 2-3 minutes for all packages to download.

### Step 4.3: Verify the App Runs

Test that everything works:

```bash
npx expo start
```

You should see a QR code in Terminal. Press **Ctrl + C** to stop for now.

---

## Part 5: Configuring Your App for iOS

### Step 5.1: Update app.json

Open the `app.json` file in a text editor. You can use:

```bash
nano app.json
```

Or open it in TextEdit/VS Code.

Find and update these fields:

```json
{
  "expo": {
    "name": "ReUnity",
    "slug": "reunity",
    "version": "1.0.0",
    "ios": {
      "bundleIdentifier": "com.yourcompany.reunity",
      "buildNumber": "1",
      "supportsTablet": true
    }
  }
}
```

**Important changes to make:**

- `bundleIdentifier`: Change `com.yourcompany.reunity` to something unique, like `com.reopsolutions.reunity` or `com.yourname.reunity`. This must be unique across all apps on the App Store.

- `version`: This is what users see (1.0.0 for your first release).

- `buildNumber`: This increments with each build you upload (start with "1").

Save the file (Ctrl + X, then Y, then Enter if using nano).

### Step 5.2: Create App Icons

Your app needs icons in specific sizes. The easiest way:

1. Create a 1024x1024 pixel PNG image of your app icon.
2. Go to [appicon.co](https://appicon.co) in your browser.
3. Upload your 1024x1024 image.
4. Select **iPhone** and **iPad**.
5. Click **Generate**.
6. Download the ZIP file.
7. Extract and copy the icons to your project's `assets` folder.

Alternatively, if you already have icons in the assets folder, make sure you have:
- `icon.png` (1024x1024)
- `adaptive-icon.png` (1024x1024)
- `splash.png` (1284x2778 recommended)

### Step 5.3: Configure EAS Build

Create an EAS configuration file:

```bash
eas build:configure
```

When prompted:
- Select **All** for platforms.
- This creates an `eas.json` file.

Open `eas.json` and make sure it looks like this:

```json
{
  "cli": {
    "version": ">= 5.0.0"
  },
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal"
    },
    "preview": {
      "distribution": "internal"
    },
    "production": {
      "autoIncrement": true
    }
  },
  "submit": {
    "production": {}
  }
}
```

---

## Part 6: Building the iOS App

### Step 6.1: Register Your App with Apple

This creates your app's identity in Apple's system:

```bash
eas build --platform ios --profile production
```

The first time you run this:

1. EAS will ask if you want to log in to your Apple Developer account. Type **Y** and press Enter.

2. Enter your Apple ID email and password.

3. If you have two-factor authentication (you probably do), you'll receive a code on your iPhone. Enter it.

4. EAS will ask about creating certificates and provisioning profiles. Select **Yes** to let EAS handle this automatically.

5. When asked about the bundle identifier, confirm it matches what you put in app.json.

### Step 6.2: Wait for the Build

The build process happens on Expo's servers. This takes **15-30 minutes**.

You'll see a URL like:
```
Build details: https://expo.dev/accounts/yourname/projects/reunity/builds/xxxxx
```

You can:
- Watch the progress in Terminal, OR
- Open that URL in your browser to see detailed logs

### Step 6.3: Download the Build (Optional)

When the build completes, you'll see:
```
✔ Build finished
📦 Build artifact: https://expo.dev/artifacts/eas/xxxxx.ipa
```

The `.ipa` file is your iOS app. You don't need to download it manually - we'll submit it directly.

---

## Part 7: Testing with TestFlight

TestFlight lets you test your app before it goes public.

### Step 7.1: Submit to TestFlight

```bash
eas submit --platform ios --latest
```

When prompted:
1. Select **App Store Connect** as the destination.
2. Log in with your Apple ID again if asked.
3. Wait 5-10 minutes for the upload.

### Step 7.2: Complete App Information in App Store Connect

1. Go to [appstoreconnect.apple.com](https://appstoreconnect.apple.com) in your browser.

2. Sign in with your Apple ID.

3. Click **My Apps**.

4. You should see your app (ReUnity). Click on it.

5. If you don't see it, click the **+** button and select **New App**:
   - Platform: iOS
   - Name: ReUnity
   - Primary Language: English (U.S.)
   - Bundle ID: Select the one you created
   - SKU: reunity-ios-001 (any unique identifier)

### Step 7.3: Fill In App Information

In App Store Connect, you need to provide:

**App Information Tab:**
- Name: ReUnity
- Subtitle: Mental Health Support Companion (30 characters max)
- Category: Health & Fitness (Primary), Medical (Secondary)
- Content Rights: Select "This app does not contain third-party content"

**Pricing and Availability Tab:**
- Price: Free (or set a price)
- Availability: Select countries where you want the app available

**App Privacy Tab:**
This is important for App Store approval:

1. Click **Get Started** under App Privacy.
2. For "Data Collection": Select **Yes, we collect data**.
3. Add the data types you collect:
   - **Health & Fitness**: Mental health data (for mood tracking)
   - **Contact Info**: Email (if users create accounts)
   - **Identifiers**: User ID
4. For each data type, specify:
   - Used for: App Functionality
   - Linked to User: Yes
   - Used for Tracking: No

### Step 7.4: Add TestFlight Testers

1. In App Store Connect, click the **TestFlight** tab.
2. Wait for your build to finish processing (can take 15-30 minutes after upload).
3. Once processed, click on the build number.
4. Click **Manage** under "Missing Compliance".
5. Answer the export compliance question:
   - "Does your app use encryption?" - Select **No** (unless you added custom encryption).
6. Click **Internal Testing** in the left sidebar.
7. Click **+** to create a new group.
8. Name it "Beta Testers".
9. Add testers by email (they must have Apple IDs).
10. Testers will receive an email invitation to download TestFlight and test your app.

### Step 7.5: Test Your App

1. On your iPhone, download **TestFlight** from the App Store.
2. Open the invitation email on your iPhone.
3. Tap the link to open TestFlight.
4. Install ReUnity and test all features.

---

## Part 8: Submitting to the App Store

### Step 8.1: Prepare Screenshots

You need screenshots of your app for the App Store listing. Required sizes:

| Device | Size (pixels) | Required |
|--------|---------------|----------|
| iPhone 6.7" | 1290 x 2796 | Yes |
| iPhone 6.5" | 1284 x 2778 | Yes |
| iPhone 5.5" | 1242 x 2208 | Yes |
| iPad Pro 12.9" | 2048 x 2732 | If supporting iPad |

**How to take screenshots:**

1. Run your app in the iOS Simulator:
   ```bash
   npx expo start --ios
   ```

2. In the Simulator, press **Cmd + S** to save a screenshot.

3. Screenshots are saved to your Desktop.

4. Alternatively, take screenshots on your real iPhone and AirDrop them to your Mac.

You need **3-10 screenshots** showing your app's main features:
- Home/Landing screen
- Chat interface
- Voice chat
- Grounding exercises
- Settings/preferences

### Step 8.2: Write Your App Description

Prepare this text before going to App Store Connect:

**App Name**: ReUnity (30 characters max)

**Subtitle**: Mental Health Support Companion (30 characters max)

**Promotional Text** (170 characters max):
```
Your compassionate AI companion for mental health support. Available 24/7 with grounding techniques, crisis resources, and personalized care.
```

**Description** (4000 characters max):
```
ReUnity is your personal mental health companion, designed to provide compassionate support whenever you need it.

KEY FEATURES:

• 24/7 AI Support - Talk to a caring AI companion anytime, day or night. ReUnity listens without judgment and responds with empathy.

• Crisis Detection - Intelligent monitoring recognizes when you're struggling and provides immediate access to crisis resources including the 988 Suicide & Crisis Lifeline.

• Grounding Techniques - Evidence-based exercises including 5-4-3-2-1 sensory grounding, box breathing, and progressive muscle relaxation to help you feel centered.

• Voice Conversations - Speak naturally with ReUnity using voice chat. Choose from 5 different voice personas that feel comfortable for you.

• Multi-Language Support - Available in 30+ languages including Spanish, Hindi, Arabic, Mandarin, and Native American languages.

• Culturally Sensitive - Respects your religious, spiritual, or philosophical background with personalized support.

• Mood Tracking - Track your emotional patterns over time with visual calendars and insights.

• Symptom Tracking - Monitor physical symptoms that correlate with your mental health.

• Guided Meditations - Curated library of calming audio sessions for anxiety, sleep, and stress relief.

• Privacy First - Your conversations are private and secure.

IMPORTANT DISCLAIMER:
ReUnity is NOT a substitute for professional mental health care, therapy, or medical treatment. If you are experiencing a mental health emergency, please contact emergency services (911) or the 988 Suicide & Crisis Lifeline immediately.

ReUnity is designed to complement, not replace, professional care. Always consult with qualified healthcare providers for mental health concerns.

Download ReUnity today and take the first step toward feeling supported.
```

**Keywords** (100 characters max, comma-separated):
```
mental health,anxiety,depression,therapy,meditation,mood,wellness,support,crisis,grounding
```

### Step 8.3: Submit for Review

1. In App Store Connect, go to your app.

2. Click **App Store** tab in the left sidebar.

3. Click the **+** next to "iOS App" to create a new version (or edit the existing 1.0).

4. Fill in all required fields:
   - Version: 1.0
   - What's New: "Initial release"

5. Upload screenshots:
   - Scroll to "App Preview and Screenshots"
   - Drag and drop your screenshots for each device size

6. Fill in:
   - Promotional Text
   - Description
   - Keywords
   - Support URL (your website or a contact page)
   - Marketing URL (optional)

7. Under "Build", click **+** and select your TestFlight build.

8. Under "App Review Information":
   - Contact info (your name, email, phone)
   - Notes for reviewer: "This is a mental health support app. Test account not required - the app works without login."

9. Under "Version Release":
   - Select "Automatically release this version"

10. Click **Save** in the top right.

11. Click **Submit for Review**.

---

## Part 9: App Store Review Process

### What to Expect

After submitting:

| Stage | Duration | What Happens |
|-------|----------|--------------|
| Waiting for Review | 24-48 hours | Your app is in the queue |
| In Review | 24-48 hours | Apple reviewers test your app |
| Approved | Immediate | Your app goes live! |
| Rejected | Immediate | You receive feedback to fix |

### Common Rejection Reasons and Fixes

**1. Metadata Rejected**
- Issue: Screenshots don't match app, or description is misleading.
- Fix: Update screenshots and description to accurately reflect your app.

**2. Guideline 4.2 - Minimum Functionality**
- Issue: App doesn't have enough features.
- Fix: ReUnity has plenty of features, but make sure all features work correctly.

**3. Guideline 5.1.1 - Data Collection**
- Issue: Privacy policy missing or incomplete.
- Fix: Add a privacy policy URL and complete the App Privacy section.

**4. Guideline 1.2 - User Generated Content**
- Issue: Apps with chat features need content moderation.
- Fix: ReUnity has built-in content moderation. Mention this in reviewer notes.

**5. Guideline 5.6.1 - Health Apps**
- Issue: Health apps need clear disclaimers.
- Fix: Make sure your disclaimer is prominent and clearly states the app is not a substitute for professional care.

### If Your App is Rejected

1. Read the rejection reason carefully in App Store Connect.
2. Fix the issue in your code or metadata.
3. If it's a code change, create a new build:
   ```bash
   eas build --platform ios --profile production
   eas submit --platform ios --latest
   ```
4. If it's just metadata, update it in App Store Connect.
5. Submit for review again.

### After Approval

Once approved:
- Your app will be live on the App Store within 24 hours.
- You can search for "ReUnity" in the App Store to find it.
- Share the App Store link with users!

---

## Updating Your App

When you want to release a new version:

1. Update the `version` in app.json (e.g., "1.0.0" → "1.1.0").

2. Build a new version:
   ```bash
   eas build --platform ios --profile production
   ```

3. Submit to App Store:
   ```bash
   eas submit --platform ios --latest
   ```

4. In App Store Connect, create a new version and submit for review.

---

## Costs Summary

| Item | Cost | Frequency |
|------|------|-----------|
| Apple Developer Account | $99 | Annual |
| Expo/EAS Build | Free tier: 30 builds/month | Monthly |
| App Store Hosting | Free | Ongoing |

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Install dependencies | `pnpm install` |
| Run app locally | `npx expo start` |
| Run on iOS simulator | `npx expo start --ios` |
| Build for iOS | `eas build --platform ios --profile production` |
| Submit to App Store | `eas submit --platform ios --latest` |
| Check EAS login | `eas whoami` |

---

**Congratulations!** You've published your app to the Apple App Store! 🎉🍎
