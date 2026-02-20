# ReUnity Google Play Store Deployment Guide

**Complete Step-by-Step Instructions for Publishing to Google Play Store (No Coding Experience Required)**

This guide walks you through building the ReUnity mobile app and publishing it to the Google Play Store. Every step is explained in detail for Mac users.

---

## Table of Contents

1. [Prerequisites and Costs](#part-1-prerequisites-and-costs)
2. [Setting Up Your Google Play Developer Account](#part-2-setting-up-your-google-play-developer-account)
3. [Installing Development Tools](#part-3-installing-development-tools)
4. [Preparing the Mobile App Code](#part-4-preparing-the-mobile-app-code)
5. [Configuring Your App for Android](#part-5-configuring-your-app-for-android)
6. [Building the Android App](#part-6-building-the-android-app)
7. [Creating Your Play Store Listing](#part-7-creating-your-play-store-listing)
8. [Uploading and Publishing](#part-8-uploading-and-publishing)
9. [Play Store Review Process](#part-9-play-store-review-process)

---

## Part 1: Prerequisites and Costs

Before you begin, here's what you need:

| Requirement | Details | Cost |
|-------------|---------|------|
| Mac Computer | You'll use Terminal for commands | You already have this |
| Google Account | Your regular Gmail account | Free |
| Google Play Developer Account | Required to publish apps | $25 one-time fee |
| The ReUnity mobile app code | The reunity-mobile folder | You already have this |

**Good news**: Unlike Apple's $99/year fee, Google only charges a one-time $25 fee that never expires.

---

## Part 2: Setting Up Your Google Play Developer Account

### Step 2.1: Create a Google Play Developer Account

1. Open your web browser and go to [play.google.com/console](https://play.google.com/console).

2. Click **Go to Play Console**.

3. Sign in with your Google account (Gmail).

4. Click **Create a developer account**.

5. Choose your account type:
   - **Personal**: For individual developers (simpler setup)
   - **Organization**: For businesses (requires more verification)
   
   Select **Personal** unless you have a registered business.

6. Fill in your developer details:
   - **Developer name**: This appears publicly on the Play Store (e.g., "REOP Solutions" or your name)
   - **Email address**: For Google to contact you
   - **Phone number**: For verification
   - **Website**: Optional (you can add your web app URL)

7. Accept the Developer Distribution Agreement.

8. Pay the $25 registration fee with a credit/debit card.

9. **Wait 24-48 hours** for Google to verify your account. You'll receive an email when approved.

### Step 2.2: Verify Your Identity (Required Since 2023)

Google now requires identity verification for all developers:

1. After your account is created, go to [play.google.com/console](https://play.google.com/console).

2. Click on your account name in the top right.

3. Go to **Developer account** → **Account details**.

4. Under "Identity verification", click **Verify identity**.

5. You'll need to provide:
   - A government-issued ID (driver's license or passport)
   - A selfie holding your ID
   - Your address

6. Upload the required documents.

7. **Wait 2-5 business days** for verification. You cannot publish apps until verified.

---

## Part 3: Installing Development Tools

If you already completed the iOS guide, you have most tools installed. Skip to Step 3.4.

### Step 3.1: Open Terminal

1. Press **Command (⌘) + Spacebar** to open Spotlight.
2. Type **Terminal** and press **Enter**.

### Step 3.2: Install Homebrew (If Not Already Installed)

Check if Homebrew is installed:

```bash
brew --version
```

If you see "command not found", install it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the prompts and enter your Mac password when asked.

### Step 3.3: Install Node.js and pnpm (If Not Already Installed)

```bash
brew install node
```

```bash
npm install -g pnpm
```

Verify:

```bash
node --version
pnpm --version
```

### Step 3.4: Install Expo CLI and EAS CLI

```bash
npm install -g expo-cli eas-cli
```

Verify:

```bash
eas --version
```

### Step 3.5: Log Into Expo

If you haven't created an Expo account:

1. Go to [expo.dev](https://expo.dev).
2. Click **Sign Up** and create an account.

Then log in via Terminal:

```bash
eas login
```

Enter your Expo username and password.

---

## Part 4: Preparing the Mobile App Code

### Step 4.1: Navigate to the Mobile App Folder

```bash
cd ~/Projects/reunity-mobile/reunity-mobile
```

If your folder is in a different location, adjust the path accordingly.

### Step 4.2: Install Dependencies

```bash
pnpm install
```

Wait 2-3 minutes for all packages to download.

### Step 4.3: Verify the App Works

```bash
npx expo start
```

You should see a QR code. Press **Ctrl + C** to stop.

---

## Part 5: Configuring Your App for Android

### Step 5.1: Update app.json

Open the `app.json` file:

```bash
nano app.json
```

Find the `android` section and update it:

```json
{
  "expo": {
    "name": "ReUnity",
    "slug": "reunity",
    "version": "1.0.0",
    "android": {
      "package": "com.reopsolutions.reunity",
      "versionCode": 1,
      "adaptiveIcon": {
        "foregroundImage": "./assets/adaptive-icon.png",
        "backgroundColor": "#1a1a2e"
      },
      "permissions": [
        "android.permission.RECORD_AUDIO",
        "android.permission.VIBRATE"
      ]
    }
  }
}
```

**Important changes:**

- `package`: This is your unique app identifier. Change `com.reopsolutions.reunity` to something unique like `com.yourname.reunity`. This must be unique across all Play Store apps.

- `versionCode`: This must increase with each update (1, 2, 3, etc.). Google uses this to track versions.

- `version`: This is what users see ("1.0.0").

Save and exit: Press **Ctrl + X**, then **Y**, then **Enter**.

### Step 5.2: Create App Icons

Your app needs icons. Required:

| Icon | Size | Purpose |
|------|------|---------|
| `icon.png` | 1024x1024 | Main app icon |
| `adaptive-icon.png` | 1024x1024 | Android adaptive icon foreground |
| `splash.png` | 1284x2778 | Loading screen |

If you need to create icons:

1. Create or obtain a 1024x1024 PNG of your logo.
2. Go to [appicon.co](https://appicon.co).
3. Upload your image.
4. Select **Android**.
5. Download and extract the icons.
6. Copy them to your `assets` folder.

### Step 5.3: Configure EAS Build

If you haven't already configured EAS:

```bash
eas build:configure
```

Select **All** when prompted.

Open `eas.json` and ensure it has:

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
      "distribution": "internal",
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "android": {
        "buildType": "app-bundle"
      }
    }
  },
  "submit": {
    "production": {}
  }
}
```

The `app-bundle` format is required for Play Store submission.

---

## Part 6: Building the Android App

### Step 6.1: Build for Production

Run this command:

```bash
eas build --platform android --profile production
```

What happens:

1. EAS uploads your code to Expo's build servers.
2. The build process takes **10-20 minutes**.
3. You'll see progress updates in Terminal.

When prompted about creating a new Android Keystore:
- Select **Yes** to let EAS generate one automatically.
- EAS stores this securely. You need this keystore for all future updates.

### Step 6.2: Wait for the Build

You'll see a URL like:
```
Build details: https://expo.dev/accounts/yourname/projects/reunity/builds/xxxxx
```

You can watch progress in Terminal or open that URL in your browser.

### Step 6.3: Download the Build

When complete, you'll see:
```
✔ Build finished
📦 Build artifact: https://expo.dev/artifacts/eas/xxxxx.aab
```

Download the `.aab` file:

1. Click the artifact URL or go to [expo.dev](https://expo.dev).
2. Navigate to your project → Builds.
3. Click on the completed build.
4. Click **Download** to save the `.aab` file.

Save it somewhere easy to find, like your Desktop.

---

## Part 7: Creating Your Play Store Listing

### Step 7.1: Create Your App in Play Console

1. Go to [play.google.com/console](https://play.google.com/console).

2. Sign in with your Google account.

3. Click **Create app** (blue button).

4. Fill in the details:
   - **App name**: ReUnity
   - **Default language**: English (United States)
   - **App or game**: App
   - **Free or paid**: Free (or Paid if you want to charge)

5. Check the boxes for:
   - Developer Program Policies
   - US export laws

6. Click **Create app**.

### Step 7.2: Set Up Your Store Listing

In the left sidebar, go to **Grow** → **Store presence** → **Main store listing**.

Fill in each section:

**App Details:**

- **App name**: ReUnity (30 characters max)

- **Short description** (80 characters max):
```
Your 24/7 AI mental health companion with crisis support and grounding techniques.
```

- **Full description** (4000 characters max):
```
ReUnity is your personal mental health companion, designed to provide compassionate support whenever you need it.

KEY FEATURES:

★ 24/7 AI Support - Talk to a caring AI companion anytime, day or night. ReUnity listens without judgment and responds with empathy.

★ Crisis Detection - Intelligent monitoring recognizes when you're struggling and provides immediate access to crisis resources including the 988 Suicide & Crisis Lifeline.

★ Grounding Techniques - Evidence-based exercises including 5-4-3-2-1 sensory grounding, box breathing, and progressive muscle relaxation to help you feel centered.

★ Voice Conversations - Speak naturally with ReUnity using voice chat. Choose from 5 different voice personas that feel comfortable for you.

★ Multi-Language Support - Available in 30+ languages including Spanish, Hindi, Arabic, Mandarin, and Native American languages.

★ Culturally Sensitive - Respects your religious, spiritual, or philosophical background with personalized support.

★ Mood Tracking - Track your emotional patterns over time with visual calendars and insights.

★ Symptom Tracking - Monitor physical symptoms that correlate with your mental health.

★ Guided Meditations - Curated library of calming audio sessions for anxiety, sleep, and stress relief.

★ Privacy First - Your conversations are private and secure.

IMPORTANT DISCLAIMER:
ReUnity is NOT a substitute for professional mental health care, therapy, or medical treatment. If you are experiencing a mental health emergency, please contact emergency services (911) or the 988 Suicide & Crisis Lifeline immediately.

ReUnity is designed to complement, not replace, professional care. Always consult with qualified healthcare providers for mental health concerns.

Download ReUnity today and take the first step toward feeling supported.
```

### Step 7.3: Add Screenshots

You need screenshots for different device sizes:

| Type | Minimum | Maximum | Size |
|------|---------|---------|------|
| Phone | 2 | 8 | 1080x1920 to 1080x2400 |
| 7-inch tablet | 0 | 8 | 1200x1920 |
| 10-inch tablet | 0 | 8 | 1920x1200 |

**How to take Android screenshots:**

Option 1 - Use Android Emulator:
1. Install Android Studio from [developer.android.com/studio](https://developer.android.com/studio).
2. Open Android Studio → Tools → Device Manager.
3. Create a virtual device (Pixel 6 recommended).
4. Run your app: `npx expo start --android`
5. Take screenshots with the camera button in the emulator.

Option 2 - Use a real Android phone:
1. Install Expo Go from Play Store on your phone.
2. Run `npx expo start` in Terminal.
3. Scan the QR code with your phone.
4. Take screenshots (usually Power + Volume Down).
5. Transfer to your Mac via Google Drive, email, or USB.

**Required screenshots:**
- Home/Landing screen
- Chat interface
- Voice chat
- Grounding exercises
- Settings/preferences

Upload screenshots by clicking **Add phone screenshot** in Play Console.

### Step 7.4: Add App Icon

Upload your 512x512 PNG app icon in the **App icon** section.

### Step 7.5: Add Feature Graphic

The feature graphic appears at the top of your Play Store listing.

- Size: 1024x500 pixels
- Format: PNG or JPEG

Create one using:
- [Canva](https://canva.com) - Free online design tool
- Include your app name, logo, and a tagline

Upload in the **Feature graphic** section.

### Step 7.6: Categorization

In the left sidebar, go to **Grow** → **Store presence** → **Store settings**.

- **App category**: Health & Fitness
- **Tags**: Add relevant tags like "mental health", "meditation", "wellness"

### Step 7.7: Contact Details

Still in Store settings:

- **Email**: Your support email
- **Phone**: Optional
- **Website**: Your web app URL

---

## Part 8: Uploading and Publishing

### Step 8.1: Complete App Content Section

In the left sidebar, go to **Policy** → **App content**.

You must complete ALL sections:

**1. Privacy Policy**
- Click **Start** under Privacy policy.
- Enter your privacy policy URL.
- If you don't have one, create a simple one at [privacypolicygenerator.info](https://www.privacypolicygenerator.info/).

**2. Ads**
- Click **Start**.
- Select **No, my app does not contain ads**.

**3. App Access**
- Click **Start**.
- Select **All functionality is available without special access** (if no login required) OR
- Select **All or some functionality is restricted** and provide test credentials if login is required.

**4. Content Rating**
- Click **Start**.
- Fill out the questionnaire honestly.
- For ReUnity, answer:
  - Violence: No
  - Sexual content: No
  - Language: No (or mild if you allow user input)
  - Controlled substances: References only (for substance abuse support)
  - User interaction: Yes (chat feature)
- Click **Save** → **Next** → **Submit**.
- You'll receive ratings like "Everyone" or "Teen".

**5. Target Audience**
- Click **Start**.
- Select age groups: **18 and over** (recommended for mental health apps).
- Confirm the app is not designed for children.

**6. News Apps**
- Click **Start**.
- Select **No, my app is not a news app**.

**7. COVID-19 Contact Tracing**
- Click **Start**.
- Select **My app is not a COVID-19 contact tracing or status app**.

**8. Data Safety**
- Click **Start**.
- This is important! Answer honestly about data collection:
  - **Does your app collect or share data?** Yes
  - **Data types collected**:
    - Personal info: Email (if accounts exist)
    - Health info: Mental health data (mood tracking)
    - App activity: App interactions
  - **Data usage**: App functionality
  - **Data sharing**: Not shared with third parties
  - **Security practices**: Data encrypted in transit
- Review and submit.

**9. Government Apps**
- Click **Start**.
- Select **No**.

**10. Financial Features**
- Click **Start**.
- Select **No** (unless you have payment features).

### Step 8.2: Create a Release

1. In the left sidebar, go to **Release** → **Production**.

2. Click **Create new release**.

3. Under "App bundles", click **Upload** and select your `.aab` file from Step 6.3.

4. Wait for the upload to complete (1-5 minutes depending on file size).

5. Under "Release details":
   - **Release name**: 1.0.0 (or leave default)
   - **Release notes**: 
   ```
   Initial release of ReUnity - Your Mental Health Companion
   
   Features:
   • 24/7 AI chat support
   • Crisis detection and resources
   • Grounding techniques
   • Voice conversations
   • Multi-language support
   • Mood and symptom tracking
   • Guided meditations
   ```

6. Click **Save**.

7. Click **Review release**.

8. Review any warnings. Common warnings:
   - "This release will be available to 0 users" - This is normal for first release.
   - Fix any errors (red) before proceeding. Warnings (yellow) are usually okay.

9. Click **Start rollout to Production**.

10. Confirm by clicking **Rollout**.

---

## Part 9: Play Store Review Process

### What to Expect

| Stage | Duration | What Happens |
|-------|----------|--------------|
| Processing | 1-2 hours | Google processes your upload |
| In Review | 1-7 days | Google reviews your app |
| Approved | Within 24 hours | Your app goes live |
| Rejected | Immediate | You receive feedback |

**Note**: First-time apps typically take longer (3-7 days). Updates are usually faster (1-3 days).

### Checking Review Status

1. Go to [play.google.com/console](https://play.google.com/console).
2. Click on your app.
3. Go to **Release** → **Production**.
4. Check the status of your release.

### Common Rejection Reasons and Fixes

**1. Metadata Policy Violation**
- Issue: Screenshots or description don't match app functionality.
- Fix: Update screenshots to accurately show your app.

**2. Broken Functionality**
- Issue: App crashes or features don't work.
- Fix: Test thoroughly on multiple devices before resubmitting.

**3. Privacy Policy Issues**
- Issue: Missing or inadequate privacy policy.
- Fix: Create a comprehensive privacy policy and link it correctly.

**4. Sensitive Content**
- Issue: Mental health apps require extra scrutiny.
- Fix: Ensure disclaimers are prominent. Add crisis resources.

**5. Impersonation**
- Issue: App name or icon too similar to another app.
- Fix: Make your branding more unique.

**6. User Data**
- Issue: Data safety form incomplete or inaccurate.
- Fix: Review and update your data safety responses.

### If Your App is Rejected

1. Read the rejection email carefully - Google explains the specific issue.

2. Fix the problem:
   - For code issues: Update code, rebuild, and re-upload.
   - For metadata issues: Update in Play Console.

3. Resubmit:
   - Go to **Release** → **Production**.
   - Create a new release with the fix.
   - Submit for review.

4. If you disagree with the rejection, you can appeal:
   - Click **Contact support** in Play Console.
   - Explain why you believe the rejection was incorrect.

### After Approval

Once approved:

1. Your app will appear on the Play Store within a few hours.

2. Search for "ReUnity" in the Play Store to find it.

3. Share the Play Store link:
   ```
   https://play.google.com/store/apps/details?id=com.reopsolutions.reunity
   ```
   (Replace with your actual package name)

---

## Updating Your App

When you want to release an update:

### Step 1: Update Version Numbers

In `app.json`:
```json
{
  "expo": {
    "version": "1.1.0",
    "android": {
      "versionCode": 2
    }
  }
}
```

**Important**: `versionCode` must ALWAYS increase. If your last release was 1, the next must be 2 or higher.

### Step 2: Build New Version

```bash
eas build --platform android --profile production
```

### Step 3: Upload to Play Console

1. Go to Play Console → Your app → **Release** → **Production**.
2. Click **Create new release**.
3. Upload the new `.aab` file.
4. Add release notes describing what changed.
5. Submit for review.

---

## Testing Before Release (Optional but Recommended)

### Internal Testing Track

Before going to production, you can test with a small group:

1. In Play Console, go to **Release** → **Testing** → **Internal testing**.
2. Click **Create new release**.
3. Upload your `.aab` file.
4. Add tester emails (up to 100).
5. Testers receive a link to install the app.

### Open Testing (Beta)

For larger beta tests:

1. Go to **Release** → **Testing** → **Open testing**.
2. Create a release.
3. Anyone can join your beta by opting in on the Play Store.

---

## Costs Summary

| Item | Cost | Frequency |
|------|------|-----------|
| Google Play Developer Account | $25 | One-time (lifetime) |
| Expo/EAS Build | Free tier: 30 builds/month | Monthly |
| Play Store Hosting | Free | Ongoing |

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Install dependencies | `pnpm install` |
| Run app locally | `npx expo start` |
| Run on Android emulator | `npx expo start --android` |
| Build for Android | `eas build --platform android --profile production` |
| Build APK for testing | `eas build --platform android --profile preview` |
| Check EAS login | `eas whoami` |

---

## Differences from iOS

| Aspect | iOS (Apple) | Android (Google) |
|--------|-------------|------------------|
| Developer fee | $99/year | $25 one-time |
| Review time | 24-48 hours | 1-7 days |
| Build format | .ipa | .aab |
| Testing | TestFlight | Internal/Open testing |
| Identity verification | Apple ID | Government ID required |

---

**Congratulations!** You've published your app to the Google Play Store! 🎉🤖
