# ReUnity Web App Deployment Guide

**Complete Step-by-Step Instructions for Mac Users (No Coding Experience Required)**

This guide will walk you through deploying the ReUnity web app from start to finish. Every step is explained in detail with exact commands to type.

---

## Table of Contents

1. [Setting Up Your Mac for Development](#part-1-setting-up-your-mac-for-development)
2. [Getting the ReUnity Code](#part-2-getting-the-reunity-code)
3. [Setting Up Your Database](#part-3-setting-up-your-database)
4. [Configuring Environment Variables](#part-4-configuring-environment-variables)
5. [Deploying to Railway (Recommended Hosting)](#part-5-deploying-to-railway)
6. [Setting Up Your Custom Domain](#part-6-setting-up-your-custom-domain)
7. [Testing Your Live Site](#part-7-testing-your-live-site)

---

## Part 1: Setting Up Your Mac for Development

### Step 1.1: Open Terminal

Terminal is the application where you'll type commands. Here's how to open it:

1. Press **Command (⌘) + Spacebar** on your keyboard. This opens Spotlight Search.
2. Type **Terminal** and press **Enter**.
3. A window with a black or white background will open. This is Terminal.

You should see something like this:
```
yourusername@your-mac ~ %
```

The `%` or `$` at the end is called the "prompt" - it means Terminal is ready for you to type a command.

### Step 1.2: Install Homebrew (Mac Package Manager)

Homebrew helps you install software easily. Copy and paste this entire command into Terminal, then press **Enter**:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

What happens next:
- Terminal will ask for your Mac password. Type it (you won't see the characters - that's normal for security) and press **Enter**.
- Wait 5-10 minutes for installation to complete.
- When you see the prompt (`%` or `$`) again, it's done.

After installation, run these two commands (press **Enter** after each):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
```

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify Homebrew is installed by typing:

```bash
brew --version
```

You should see something like `Homebrew 4.x.x`.

### Step 1.3: Install Node.js

Node.js runs JavaScript code. Install it with this command:

```bash
brew install node
```

Wait 2-3 minutes. When done, verify with:

```bash
node --version
```

You should see something like `v22.x.x`.

### Step 1.4: Install pnpm (Package Manager)

pnpm manages the project's dependencies. Install it:

```bash
npm install -g pnpm
```

Verify with:

```bash
pnpm --version
```

You should see something like `9.x.x`.

### Step 1.5: Install Git

Git tracks code changes and downloads code from the internet:

```bash
brew install git
```

Verify with:

```bash
git --version
```

You should see something like `git version 2.x.x`.

---

## Part 2: Getting the ReUnity Code

### Step 2.1: Create a Folder for Your Projects

Let's create a folder to store your code. Type these commands one at a time:

```bash
cd ~
```

This goes to your home folder.

```bash
mkdir Projects
```

This creates a folder called "Projects".

```bash
cd Projects
```

This goes into the Projects folder.

### Step 2.2: Download the ReUnity Code

If you have the code as a ZIP file:

1. Find the ZIP file in Finder (probably in Downloads).
2. Double-click to unzip it.
3. Drag the unzipped folder into your Projects folder.
4. In Terminal, type:

```bash
cd ~/Projects/reunity-app
```

If you're downloading from GitHub:

```bash
git clone https://github.com/ezernackchristopher97-cloud/reunity-app.git
```

Then:

```bash
cd reunity-app
```

### Step 2.3: Install Project Dependencies

This downloads all the code libraries ReUnity needs:

```bash
pnpm install
```

Wait 3-5 minutes. You'll see lots of text scrolling - that's normal.

---

## Part 3: Setting Up Your Database

ReUnity uses a MySQL database. We'll use PlanetScale (free tier available) or Railway's built-in database.

### Option A: Using Railway's Database (Recommended - Easiest)

Skip this section for now. Railway will create the database automatically when you deploy. Jump to Part 4.

### Option B: Using PlanetScale (Alternative)

1. Go to [planetscale.com](https://planetscale.com) in your web browser.
2. Click **Sign Up** and create an account (you can use Google or GitHub to sign up).
3. Click **Create Database**.
4. Name it `reunity-db`.
5. Select the region closest to you.
6. Click **Create Database**.
7. Click **Connect** and select **Node.js** from the dropdown.
8. Copy the connection string that looks like:
   ```
   mysql://username:password@host.planetscale.com/reunity-db?ssl={"rejectUnauthorized":true}
   ```
9. Save this - you'll need it in Part 4.

---

## Part 4: Configuring Environment Variables

Environment variables are secret settings your app needs to run.

### Step 4.1: Create Your Environment File

In Terminal (make sure you're in the reunity-app folder), type:

```bash
cp docs/env.example.txt .env
```

This copies the example file to create your own `.env` file.

### Step 4.2: Open the File for Editing

```bash
nano .env
```

This opens a simple text editor in Terminal.

### Step 4.3: Fill In Your Values

You'll see something like this:

```
DATABASE_URL=your_database_url_here
JWT_SECRET=your_secret_here
OPENAI_API_KEY=your_openai_key_here
```

Replace each value:

**DATABASE_URL**: If using PlanetScale, paste the connection string from Part 3. If using Railway, leave it blank for now - Railway will provide this.

**JWT_SECRET**: This is a random string for security. Generate one by typing this in a NEW Terminal window:

```bash
openssl rand -base64 32
```

Copy the output and paste it as your JWT_SECRET.

**OPENAI_API_KEY**: 
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Click your profile icon → **View API Keys**
4. Click **Create new secret key**
5. Copy the key (starts with `sk-`) and paste it here

### Step 4.4: Save and Exit

1. Press **Control + X** to exit
2. Press **Y** to save
3. Press **Enter** to confirm

---

## Part 5: Deploying to Railway

Railway is a hosting service that makes deployment easy.

### Step 5.1: Create a Railway Account

1. Go to [railway.app](https://railway.app) in your browser.
2. Click **Login** in the top right.
3. Click **Login with GitHub** (if you don't have GitHub, create an account at github.com first).
4. Authorize Railway to access your GitHub.

### Step 5.2: Install Railway CLI

In Terminal:

```bash
brew install railway
```

Then log in:

```bash
railway login
```

A browser window will open. Click **Authorize** and return to Terminal.

### Step 5.3: Create a New Project

```bash
railway init
```

When prompted:
- Select **Empty Project**
- Name it `reunity-app`

### Step 5.4: Add a Database

```bash
railway add
```

Select **MySQL** from the list. Railway will create a database for you.

### Step 5.5: Link Your Code to Railway

Make sure you're in the reunity-app folder:

```bash
cd ~/Projects/reunity-app
```

Link the project:

```bash
railway link
```

Select the `reunity-app` project you just created.

### Step 5.6: Set Environment Variables on Railway

You need to add your secrets to Railway. Go to [railway.app](https://railway.app), click on your project, then:

1. Click on your service (not the database)
2. Go to the **Variables** tab
3. Click **Raw Editor**
4. Paste all your environment variables:

```
JWT_SECRET=your_jwt_secret_from_earlier
OPENAI_API_KEY=sk-your_openai_key
NODE_ENV=production
```

Note: Don't add DATABASE_URL - Railway automatically provides this from your MySQL service.

Click **Update Variables**.

### Step 5.7: Deploy Your App

In Terminal:

```bash
railway up
```

Wait 5-10 minutes. You'll see build logs scrolling.

When you see "Deployment successful" or similar, your app is live!

### Step 5.8: Get Your App URL

```bash
railway open
```

This opens your Railway dashboard. Click on your service and find the **Deployments** section. Click on the latest deployment to see your app's URL (something like `reunity-app-production.up.railway.app`).

---

## Part 6: Setting Up Your Custom Domain

### Step 6.1: Add Domain in Railway

1. In your Railway dashboard, click on your service.
2. Go to **Settings** tab.
3. Scroll to **Domains**.
4. Click **Generate Domain** for a free railway.app subdomain, OR
5. Click **Custom Domain** and enter your domain (e.g., `reunityai.com` or `app.reunityai.com`).

### Step 6.2: Configure DNS (If Using Custom Domain)

If you bought a domain from Namecheap, GoDaddy, Google Domains, etc.:

1. Log into your domain registrar's website.
2. Find **DNS Settings** or **DNS Management**.
3. Add a new record:
   - **Type**: CNAME
   - **Name**: `@` (or `www` or `app` depending on what you want)
   - **Value**: The Railway domain (e.g., `reunity-app-production.up.railway.app`)
   - **TTL**: 3600 (or "Auto")
4. Save the record.
5. Wait 15-30 minutes for DNS to update.

### Step 6.3: Enable HTTPS

Railway automatically provides free SSL certificates. Once your domain is connected, HTTPS will be enabled automatically within a few minutes.

---

## Part 7: Testing Your Live Site

### Step 7.1: Visit Your Site

Open your browser and go to your domain (or the Railway-provided URL).

### Step 7.2: Test Key Features

1. **Disclaimer Page**: Should appear on first visit.
2. **Chat**: Accept the disclaimer and try sending a message.
3. **Voice Chat**: Click the voice button and test speaking.
4. **Crisis Detection**: Type "I'm feeling really hopeless" - should show crisis resources.

### Step 7.3: Check for Errors

If something doesn't work:

1. Go to your Railway dashboard.
2. Click on your service.
3. Click **Deployments** → your latest deployment → **View Logs**.
4. Look for red error messages.

Common fixes:
- **Database errors**: Make sure your MySQL service is running in Railway.
- **API errors**: Double-check your OPENAI_API_KEY in Variables.
- **Build errors**: Run `pnpm install` locally and try `railway up` again.

---

## Troubleshooting

### "Command not found" errors

If Terminal says a command isn't found, the software isn't installed. Go back to Part 1 and make sure each installation completed successfully.

### "Permission denied" errors

Add `sudo` before the command:

```bash
sudo pnpm install
```

Enter your Mac password when prompted.

### App shows blank page

Check the browser console:
1. Right-click on the page
2. Click **Inspect**
3. Click **Console** tab
4. Look for red error messages

### Railway deployment fails

1. Check that all environment variables are set correctly.
2. Make sure there are no typos in your `.env` file.
3. Try running locally first:

```bash
pnpm dev
```

If it works locally but not on Railway, the issue is likely environment variables.

---

## Quick Reference: Common Commands

| What You Want to Do | Command |
|---------------------|---------|
| Go to project folder | `cd ~/Projects/reunity-app` |
| Install dependencies | `pnpm install` |
| Run locally for testing | `pnpm dev` |
| Deploy to Railway | `railway up` |
| View Railway logs | `railway logs` |
| Open Railway dashboard | `railway open` |
| Check Node version | `node --version` |
| Check if in right folder | `pwd` |

---

## Costs

| Service | Free Tier | Paid |
|---------|-----------|------|
| Railway | $5/month credit (usually enough) | $20/month for more resources |
| OpenAI API | $5 free credit for new accounts | Pay per use (~$0.01-0.03 per chat) |
| Domain | N/A | $10-15/year |

---

## Next Steps

Once your web app is live:
1. Set up Stripe for payments (see STRIPE_GUIDE.md)
2. Deploy the mobile app to App Store and Play Store
3. Set up analytics to track usage

---

**Congratulations!** Your ReUnity web app is now live on the internet. 🎉
