# ReUnity AI Model: Setup Guide for Faraz

**Author:** Christopher Ezernack  
**Version:** 2.0.0  
**January 2026**

---

## STEP 1: Open GitHub Codespaces

1. Go to https://github.com/ezernackchristopher97-cloud/ReUnity
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
pip install numpy flask gunicorn
```

Wait for it to finish. When you see the blinking cursor again, it is done.

---

## STEP 5: Test the Model

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_standalone.py --test
```

You should see output like this:

```
ReUnity v1.0.0
A Trauma-Aware AI Support Framework
By Christopher Ezernack, REOP Solutions
Running basic test...

Input: I'm feeling anxious today
State: high_entropy
Entropy: ...

Input: My partner said I was imagining things again
State: stable
Patterns: gaslighting
```

---

## STEP 6: Run Interactive Mode

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_standalone.py
```

Type anything and press Enter. Type `/quit` to exit.

---

## STEP 7: Run the Web Version

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_standalone.py --web
```

You will see:

```
Starting ReUnity web interface on port 5000...
```

---

## STEP 8: Open the Web Interface

1. Look at the bottom of the Codespaces window
2. Click the **Ports** tab (next to Terminal)
3. Find port **5000** in the list
4. Click the globe icon next to it
5. A new browser tab will open with the ReUnity web interface

---

## STEP 9: Share the Web Interface

1. In the **Ports** tab, right-click on port 5000
2. Click **Port Visibility** then **Public**
3. Copy the URL shown
4. Send this URL to anyone to let them use ReUnity

---

## STEP 10: Deploy to the Internet (Optional)

To make ReUnity permanently available online:

### Option A: Render.com (Free)

1. Go to https://render.com
2. Sign up with GitHub
3. Click **New** then **Web Service**
4. Connect your ReUnity repository
5. Set these values:
   - Name: `reunity`
   - Build Command: `pip install numpy flask gunicorn`
   - Start Command: `cd reunity_ai_model && gunicorn -b 0.0.0.0:$PORT reunity_standalone:app`
6. Click **Create Web Service**
7. Wait 5 minutes
8. Your app will be live at `https://reunity.onrender.com`

### Option B: Railway.app

1. Go to https://railway.app
2. Sign up with GitHub
3. Click **New Project** then **Deploy from GitHub repo**
4. Select the ReUnity repository
5. Railway will auto-detect and deploy
6. Your app will be live in 5 minutes

---

## STEP 11: Mobile App (Optional)

ReUnity works as a mobile app through your phone's browser:

1. Open the web URL on your phone
2. On iPhone: Tap the Share button, then **Add to Home Screen**
3. On Android: Tap the menu (three dots), then **Add to Home Screen**
4. ReUnity will appear as an app icon on your phone

---

## What the Model Does

| State | What It Means | What ReUnity Does |
|-------|---------------|-------------------|
| CRISIS | Dissociation, suicidal thoughts, panic | Immediate grounding, crisis resources |
| HIGH | Scared, anxious, angry, sad | Grounding techniques, body awareness |
| MODERATE | Mixed emotions, uncertain | Exploration, support options |
| LOW | Slightly unsettled | Reflection, planning |
| STABLE | Calm, okay, peaceful | Growth work, exploration |

| Pattern | What It Detects |
|---------|-----------------|
| Gaslighting | "imagining things", "never happened", "you're crazy" |
| Love Bombing | "you're perfect", "soulmates", "never felt this way" |
| Isolation | "only need me", "they're jealous", "spend all time together" |
| Hot-Cold Cycle | "sometimes loving sometimes cold", "unpredictable" |
| Blame Shifting | "your fault", "you made me", "because of you" |

---

## Files in This Folder

| File | What It Does |
|------|--------------|
| `reunity_standalone.py` | The complete AI model (2100+ lines) |
| `requirements.txt` | Python packages needed |
| `README.md` | General information |
| `core/` | Entropy calculation modules |
| `router/` | State routing modules |
| `protective/` | Pattern detection modules |
| `memory/` | RIME memory modules |
| `grounding/` | Grounding technique modules |

---

## Troubleshooting

**"Module not found" error:**
```
pip install numpy flask gunicorn
```

**"Port already in use" error:**
```
python reunity_standalone.py --web --port 5001
```

**"Permission denied" error:**
```
chmod +x reunity_standalone.py
```

---

## Crisis Resources

If you or someone you know is in crisis:

- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741 (US)
- **International:** https://www.iasp.info/resources/Crisis_Centres/

ReUnity is NOT a replacement for professional mental health care.
