# ReUnity AI Model: Setup Guide for Faraz

**Author:** Christopher Ezernack  
**Version:** 3.0.0  
**January 2026**

---

## STEP 1: Open GitHub Codespaces

1. Go to https://github.com/ezernackchristopher/ReUnity
2. Click the green **Code** button
3. Click the **Codespaces** tab
4. Click **Create codespace on main**
5. Wait 2 minutes for it to load
6. You will see a code editor in your browser

---

## STEP 2: Open the Terminal

1. Look at the bottom of the screen
2. You should see a panel called **Terminal**
3. If you do not see it, click **View** in the top menu, then click **Terminal**
4. You will see a blinking cursor where you can type

---

## STEP 3: Go to the AI Model Folder

Copy this line and paste it into the terminal, then press Enter:

```
cd reunity_ai_model
```

---

## STEP 4: Install the Requirements

Copy this line and paste it into the terminal, then press Enter:

```
pip install flask openai numpy
```

Wait for it to finish. When you see the blinking cursor again, it is done.

---

## STEP 5: Get an OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Click **Create new secret key**
4. Copy the key (it starts with sk-)
5. Save it somewhere safe

---

## STEP 6: Set Your API Key

Copy this line, replace YOUR_KEY_HERE with your actual key, then paste and press Enter:

```
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

Example:
```
export OPENAI_API_KEY="sk-abc123xyz456..."
```

---

## STEP 7: Test the Model

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_standalone.py --test
```

You should see output like this:

```
Input: I am dissociating right now
Expected: Should detect CRISIS
State: crisis
Crisis indicators: ['dissociating']
```

If you see "crisis" for dissociating, the model is working correctly.

---

## STEP 8: Run Interactive Mode

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_standalone.py
```

Type anything and press Enter. The AI will respond. Type `/quit` to exit.

---

## STEP 9: Run the Web Version

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_standalone.py --web --port 5000
```

You will see:

```
Starting ReUnity web interface on port 5000...
```

---

## STEP 10: Open the Web Interface

1. Look at the bottom of the Codespaces window
2. Click the **Ports** tab (next to Terminal)
3. Find port **5000** in the list
4. Click the globe icon next to it
5. A new browser tab will open with the ReUnity web interface

---

## STEP 11: Share the Web Interface

1. In the **Ports** tab, right-click on port 5000
2. Click **Port Visibility** then **Public**
3. Copy the URL shown
4. Send this URL to anyone to let them use ReUnity

---

## STEP 12: Deploy to the Internet

To make ReUnity permanently available online:

### Option A: Render.com (Free Tier Available)

1. Go to https://render.com
2. Sign up with GitHub
3. Click **New** then **Web Service**
4. Connect your ReUnity repository
5. Set these values:
   - **Name:** reunity
   - **Root Directory:** reunity_ai_model
   - **Build Command:** `pip install flask openai numpy`
   - **Start Command:** `python reunity_standalone.py --web --port 10000`
6. Click **Environment** and add:
   - Key: `OPENAI_API_KEY`
   - Value: (paste your OpenAI API key)
7. Click **Create Web Service**
8. Wait 5-10 minutes
9. Your app will be live at `https://reunity.onrender.com`

### Option B: Railway.app

1. Go to https://railway.app
2. Sign up with GitHub
3. Click **New Project** then **Deploy from GitHub repo**
4. Select the ReUnity repository
5. Click **Variables** and add:
   - `OPENAI_API_KEY` = (your key)
6. Railway will auto-detect and deploy
7. Your app will be live in 5 minutes

---

## STEP 13: Mobile App

ReUnity works as a mobile app through your phone's browser:

**iPhone:**
1. Open the web URL in Safari
2. Tap the Share button (square with arrow)
3. Tap **Add to Home Screen**
4. Tap **Add**

**Android:**
1. Open the web URL in Chrome
2. Tap the menu (three dots)
3. Tap **Add to Home Screen**
4. Tap **Add**

ReUnity will appear as an app icon on your phone.

---

## What the Model Does

| State | What It Means | What ReUnity Does |
|-------|---------------|-------------------|
| CRISIS | Dissociation, suicidal thoughts, panic | Immediate grounding, crisis resources |
| HIGH_ENTROPY | Scared, anxious, angry, sad | Grounding techniques, validation |
| MODERATE | Mixed emotions, uncertain | Exploration, support options |
| LOW_ENTROPY | Slightly unsettled | Reflection, planning |
| STABLE | Calm, okay, peaceful | Growth work, exploration |

| Pattern | What It Detects |
|---------|-----------------|
| Gaslighting | "imagining things", "never happened", "you're crazy" |
| Love Bombing | "you're perfect", "soulmates", "never felt this way" |
| Isolation | "only need me", "they're jealous", "spend all time together" |
| Hot-Cold Cycle | "sometimes loving sometimes cold", "unpredictable" |
| Blame Shifting | "your fault", "you made me", "because of you" |

---

## How It Works

1. **User sends message** → 
2. **QueryGate (PreRAG)** checks if query is valid →
3. **EntropyAnalyzer** detects emotional state →
4. **PatternRecognizer** checks for harmful patterns →
5. **RAGSystem** retrieves relevant knowledge →
6. **EvidenceGate (PreRAG)** validates retrieved info →
7. **MemoryStore (RIME)** retrieves conversation context →
8. **OpenAI API** generates contextual response →
9. **GroundingLibrary** adds techniques if needed →
10. **Response sent to user**

---

## Files in This Folder

| File | What It Does |
|------|--------------|
| `reunity_standalone.py` | The complete AI model with all components |
| `requirements.txt` | Python packages needed |
| `README.md` | General information |
| `REUNITY_COMPLETE_CODE_FOR_FARAZ.md` | This file |

---

## Commands Reference

| Command | What It Does |
|---------|--------------|
| `python reunity_standalone.py` | Run interactive mode |
| `python reunity_standalone.py --test` | Run tests |
| `python reunity_standalone.py --web` | Run web server on port 5000 |
| `python reunity_standalone.py --web --port 8080` | Run web server on port 8080 |
| `python reunity_standalone.py --api-key YOUR_KEY` | Run with specific API key |

---

## Troubleshooting

**"No module named flask"**
```
pip install flask
```

**"No module named openai"**
```
pip install openai
```

**"OpenAI API key not found"**
```
export OPENAI_API_KEY="your-key-here"
```

**"Port already in use"**
```
python reunity_standalone.py --web --port 5001
```

**Responses are generic/fallback**
- Check that your OpenAI API key is valid
- Check that you have credits in your OpenAI account

---

## Crisis Resources

If you or someone you know is in crisis:

- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741 (US)
- **National Domestic Violence Hotline:** 1-800-799-7233
- **International:** https://www.iasp.info/resources/Crisis_Centres/

ReUnity is NOT a replacement for professional mental health care.

---

## Questions?

Contact Christopher Ezernack at REOP Solutions.
