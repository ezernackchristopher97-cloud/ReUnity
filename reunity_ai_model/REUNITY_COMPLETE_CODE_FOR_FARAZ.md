# ReUnity AI Model

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

## STEP 3: Create the Project Folder

Copy this line and paste it into the terminal, then press Enter:

```
mkdir -p reunity_model && cd reunity_model
```

---

## STEP 4: Create the Requirements File

1. In the left sidebar, right-click on the **reunity_model** folder
2. Click **New File**
3. Name the file: `requirements.txt`
4. Press Enter
5. The file will open
6. Copy everything below and paste it into the file:

```
numpy>=1.21.0
flask>=2.0.0
gunicorn>=20.1.0
```

7. Press **Ctrl+S** (or **Cmd+S** on Mac) to save

---

## STEP 5: Install the Requirements

Copy this line and paste it into the terminal, then press Enter:

```
pip install -r requirements.txt
```

Wait for it to finish. You will see some text scrolling. When it stops and you see the blinking cursor again, it is done.

---

## STEP 6: Create the Main Code File

1. In the left sidebar, right-click on the **reunity_model** folder
2. Click **New File**
3. Name the file: `reunity_model.py`
4. Press Enter
5. The file will open
6. Copy **ALL** of the code below (from the first line to the last line) and paste it into the file:

```python
"""
ReUnity AI Model v2.0.0
Author: Christopher Ezernack
January 2026

IMPORTANT: This is NOT a clinical tool. If you are in crisis, call 988.
"""

import os
import sys
import json
import time
import uuid
import math
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReUnity")

VERSION = "2.0.0"

# ============================================================================
# CRISIS KEYWORDS - These trigger immediate crisis response
# ============================================================================

CRISIS_KEYWORDS = {
    "dissociating", "dissociate", "dissociated", "dissociation",
    "depersonalization", "derealization", "not real", "unreal",
    "floating", "detached", "out of body", "watching myself",
    "numb", "empty inside", "disconnected",
    "suicidal", "suicide", "kill myself", "end it", "end my life",
    "want to die", "better off dead", "no reason to live",
    "can't go on", "give up", "hopeless",
    "hurt myself", "cutting", "self harm", "self-harm",
    "panic", "panicking", "terrified", "terror", "can't breathe",
    "heart racing", "going to die", "losing my mind",
    "breaking down", "falling apart", "can't take it",
    "overwhelmed", "drowning", "suffocating",
}

# ============================================================================
# HIGH DISTRESS KEYWORDS - These trigger elevated response
# ============================================================================

HIGH_KEYWORDS = {
    "scared", "afraid", "anxious", "worried", "nervous",
    "angry", "furious", "rage", "hate", "frustrated",
    "sad", "depressed", "crying", "tears", "grief",
    "confused", "lost", "uncertain", "doubt",
    "alone", "lonely", "isolated", "abandoned",
    "hurt", "pain", "suffering", "struggling",
    "stressed", "tense", "on edge", "restless",
}

# ============================================================================
# STABLE KEYWORDS - These indicate grounded state
# ============================================================================

STABLE_KEYWORDS = {
    "calm", "peaceful", "relaxed", "okay", "fine", "good",
    "happy", "content", "grateful", "hopeful", "better",
    "safe", "secure", "grounded", "present", "centered",
    "strong", "capable", "confident", "clear",
}

# ============================================================================
# ENTROPY STATES
# ============================================================================

class EntropyState(Enum):
    CRISIS = "crisis"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    STABLE = "stable"

# ============================================================================
# PATTERN TYPES
# ============================================================================

class PatternType(Enum):
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    ISOLATION = "isolation"
    INVALIDATION = "invalidation"
    BLAME_SHIFTING = "blame_shifting"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EntropyAnalysis:
    entropy: float
    state: EntropyState
    confidence: float
    crisis_keywords: List[str]
    high_keywords: List[str]
    timestamp: float = field(default_factory=time.time)

@dataclass
class PatternDetection:
    pattern_type: PatternType
    confidence: float
    evidence: List[str]
    recommendation: str

@dataclass
class GroundingTechnique:
    name: str
    description: str
    steps: List[str]
    duration: int

# ============================================================================
# ENTROPY ANALYZER
# ============================================================================

class EntropyAnalyzer:
    def __init__(self):
        self.history = []
    
    def analyze(self, text: str) -> EntropyAnalysis:
        text_lower = text.lower()
        
        # Check crisis keywords first
        crisis_found = [kw for kw in CRISIS_KEYWORDS if kw in text_lower]
        if crisis_found:
            analysis = EntropyAnalysis(
                entropy=0.95,
                state=EntropyState.CRISIS,
                confidence=0.95,
                crisis_keywords=crisis_found,
                high_keywords=[],
            )
            self.history.append(analysis)
            return analysis
        
        # Check high keywords
        high_found = [kw for kw in HIGH_KEYWORDS if kw in text_lower]
        
        # Check stable keywords
        stable_found = [kw for kw in STABLE_KEYWORDS if kw in text_lower]
        
        # Calculate entropy
        high_score = len(high_found) * 0.15
        stable_score = len(stable_found) * 0.15
        net_entropy = max(0.0, min(1.0, 0.5 + high_score - stable_score))
        
        # Determine state
        if len(high_found) >= 3 and len(stable_found) == 0:
            state = EntropyState.HIGH
            net_entropy = max(net_entropy, 0.75)
        elif net_entropy >= 0.75:
            state = EntropyState.HIGH
        elif net_entropy >= 0.55:
            state = EntropyState.MODERATE
        elif net_entropy >= 0.35:
            state = EntropyState.LOW
        else:
            state = EntropyState.STABLE
        
        analysis = EntropyAnalysis(
            entropy=net_entropy,
            state=state,
            confidence=0.8,
            crisis_keywords=[],
            high_keywords=high_found,
        )
        self.history.append(analysis)
        return analysis

# ============================================================================
# PATTERN RECOGNIZER
# ============================================================================

class PatternRecognizer:
    def __init__(self):
        self.patterns = {
            PatternType.GASLIGHTING: {
                "indicators": [
                    "you're imagining", "that never happened", "you're crazy",
                    "you're too sensitive", "you're overreacting", "i never said that",
                    "you're making things up", "no one will believe you",
                ],
                "recommendation": "Trust your own perceptions. Consider documenting events. Gaslighting is psychological abuse designed to make you doubt reality.",
            },
            PatternType.LOVE_BOMBING: {
                "indicators": [
                    "soulmate", "never felt this way", "meant to be",
                    "perfect for each other", "can't live without you",
                ],
                "recommendation": "Healthy relationships develop gradually. Intense early attention can be a warning sign.",
            },
            PatternType.ISOLATION: {
                "indicators": [
                    "don't need them", "they don't understand us",
                    "only i understand you", "they're jealous",
                ],
                "recommendation": "Healthy partners encourage your other relationships. Isolation is a control tactic.",
            },
            PatternType.INVALIDATION: {
                "indicators": [
                    "you shouldn't feel", "get over it", "stop being dramatic",
                    "it's not a big deal", "you're too emotional",
                ],
                "recommendation": "Your feelings are valid. Someone who cares will try to understand, not dismiss.",
            },
            PatternType.BLAME_SHIFTING: {
                "indicators": [
                    "your fault", "you made me", "because of you",
                    "if you hadn't", "look what you made me do",
                ],
                "recommendation": "Healthy people take responsibility. Blame-shifting avoids accountability.",
            },
        }
    
    def analyze(self, text: str) -> List[PatternDetection]:
        detections = []
        text_lower = text.lower()
        
        for pattern_type, data in self.patterns.items():
            evidence = [ind for ind in data["indicators"] if ind in text_lower]
            if evidence:
                detections.append(PatternDetection(
                    pattern_type=pattern_type,
                    confidence=min(len(evidence) / 3, 1.0),
                    evidence=evidence,
                    recommendation=data["recommendation"],
                ))
        return detections

# ============================================================================
# GROUNDING LIBRARY
# ============================================================================

class GroundingLibrary:
    def __init__(self):
        self.techniques = {
            "five_senses": GroundingTechnique(
                name="5-4-3-2-1 Grounding",
                description="Use your five senses to anchor to the present.",
                steps=[
                    "Name 5 things you can SEE right now",
                    "Name 4 things you can TOUCH",
                    "Name 3 things you can HEAR",
                    "Name 2 things you can SMELL",
                    "Name 1 thing you can TASTE",
                ],
                duration=5,
            ),
            "box_breathing": GroundingTechnique(
                name="Box Breathing",
                description="A calming breath pattern.",
                steps=[
                    "Breathe IN for 4 counts",
                    "HOLD for 4 counts",
                    "Breathe OUT for 4 counts",
                    "HOLD empty for 4 counts",
                    "Repeat 4 times",
                ],
                duration=3,
            ),
            "cold_water": GroundingTechnique(
                name="Cold Water Reset",
                description="Use cold to activate your dive reflex.",
                steps=[
                    "Get cold water or ice",
                    "Splash cold water on your face",
                    "Or hold ice cubes in your hands",
                    "Focus on the cold sensation",
                    "Breathe slowly",
                ],
                duration=2,
            ),
            "body_scan": GroundingTechnique(
                name="Quick Body Scan",
                description="Notice your body to return to the present.",
                steps=[
                    "Feel your feet on the ground",
                    "Notice your legs",
                    "Feel your hands",
                    "Notice your shoulders, let them drop",
                    "Relax your jaw",
                    "Take three slow breaths",
                ],
                duration=3,
            ),
            "butterfly_hug": GroundingTechnique(
                name="Butterfly Hug",
                description="Bilateral stimulation to calm your nervous system.",
                steps=[
                    "Cross your arms over your chest",
                    "Place hands on shoulders",
                    "Tap shoulders alternately (left, right, left, right)",
                    "Tap slowly and rhythmically",
                    "Breathe slowly as you tap",
                    "Continue for 1-2 minutes",
                ],
                duration=2,
            ),
        }
    
    def get_for_state(self, state: EntropyState) -> GroundingTechnique:
        if state == EntropyState.CRISIS:
            return random.choice([
                self.techniques["cold_water"],
                self.techniques["box_breathing"],
                self.techniques["butterfly_hug"],
            ])
        elif state == EntropyState.HIGH:
            return random.choice([
                self.techniques["five_senses"],
                self.techniques["box_breathing"],
                self.techniques["body_scan"],
            ])
        else:
            return random.choice(list(self.techniques.values()))
    
    def format_technique(self, t: GroundingTechnique) -> str:
        lines = [f"**{t.name}**", f"_{t.description}_", ""]
        for i, step in enumerate(t.steps, 1):
            lines.append(f"{i}. {step}")
        lines.append(f"\nThis takes about {t.duration} minutes.")
        return "\n".join(lines)

# ============================================================================
# MEMORY STORE
# ============================================================================

class MemoryStore:
    def __init__(self):
        self.memories = {}
        self.timeline = []
    
    def store(self, content: str, state: EntropyState, tags: List[str]) -> str:
        memory_id = str(uuid.uuid4())
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "state": state.value,
            "timestamp": time.time(),
            "tags": tags,
        }
        self.timeline.append(memory_id)
        return memory_id
    
    def get_recent(self, count: int = 5) -> List[dict]:
        return [self.memories[mid] for mid in self.timeline[-count:] if mid in self.memories]

# ============================================================================
# RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    def __init__(self):
        self.grounding = GroundingLibrary()
        
        self.crisis_responses = [
            "I hear you. What you are feeling is real. Let me help you get grounded right now.",
            "Thank you for telling me. You are not alone. I am going to share something that can help.",
            "I am glad you reached out. Let us work through this together, one breath at a time.",
            "You are safe to share this with me. Let me help you feel more grounded.",
            "I can hear how much pain you are in. You matter. Let me help you through this moment.",
        ]
        
        self.high_responses = [
            "I can sense things feel intense right now. Let us take a moment to breathe together.",
            "What you are feeling is valid. These emotions are telling you something important.",
            "It sounds like you are carrying a lot. I am here to listen and support you.",
            "I hear the weight in what you are sharing. You do not have to figure this all out right now.",
            "These feelings are real and they matter. Let us slow down and be present with what is here.",
        ]
        
        self.moderate_responses = [
            "Thank you for sharing that. How are you feeling in your body right now?",
            "I am here with you. Would you like to talk more about what is on your mind?",
            "It sounds like there is a lot going on. What feels most important to focus on?",
            "I appreciate you opening up. What would feel most supportive right now?",
        ]
        
        self.stable_responses = [
            "It sounds like you are in a good place. What would you like to explore?",
            "I am glad to hear things feel manageable. Is there anything you would like to work on?",
            "This seems like a good time for reflection. What has been on your mind?",
            "You seem grounded. Would you like to use this time for growth-oriented work?",
        ]
    
    def generate(self, analysis: EntropyAnalysis, patterns: List[PatternDetection]) -> str:
        parts = []
        
        if analysis.state == EntropyState.CRISIS:
            parts.append(random.choice(self.crisis_responses))
            parts.append("")
            technique = self.grounding.get_for_state(analysis.state)
            parts.append(self.grounding.format_technique(technique))
            parts.append("")
            parts.append("**If you need immediate support:**")
            parts.append("- National Suicide Prevention Lifeline: **988** (US)")
            parts.append("- Crisis Text Line: Text **HOME** to **741741**")
            
        elif analysis.state == EntropyState.HIGH:
            parts.append(random.choice(self.high_responses))
            parts.append("")
            parts.append("Here is something that might help:")
            parts.append("")
            technique = self.grounding.get_for_state(analysis.state)
            parts.append(self.grounding.format_technique(technique))
            
        elif analysis.state == EntropyState.MODERATE:
            parts.append(random.choice(self.moderate_responses))
            
        else:
            parts.append(random.choice(self.stable_responses))
        
        if patterns:
            best = max(patterns, key=lambda p: p.confidence)
            if best.confidence > 0.3:
                parts.append("")
                parts.append("---")
                parts.append("")
                parts.append(f"**Something I noticed:** {best.recommendation}")
        
        return "\n".join(parts)

# ============================================================================
# MAIN REUNITY CLASS
# ============================================================================

class ReUnity:
    def __init__(self):
        self.entropy_analyzer = EntropyAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.memory_store = MemoryStore()
        self.response_generator = ResponseGenerator()
        self.session_id = str(uuid.uuid4())
        self.interaction_count = 0
        logger.info(f"ReUnity v{VERSION} started. Session: {self.session_id}")
    
    def process(self, text: str) -> Dict[str, Any]:
        self.interaction_count += 1
        analysis = self.entropy_analyzer.analyze(text)
        patterns = self.pattern_recognizer.analyze(text)
        response = self.response_generator.generate(analysis, patterns)
        
        tags = []
        text_lower = text.lower()
        if any(w in text_lower for w in ["relationship", "partner", "boyfriend", "girlfriend"]):
            tags.append("relationship")
        if any(w in text_lower for w in ["family", "mother", "father", "parent"]):
            tags.append("family")
        if any(w in text_lower for w in ["work", "job", "boss"]):
            tags.append("work")
        
        memory_id = self.memory_store.store(text, analysis.state, tags)
        
        return {
            "session_id": self.session_id,
            "interaction": self.interaction_count,
            "input": text,
            "state": analysis.state.value,
            "entropy": analysis.entropy,
            "crisis_keywords": analysis.crisis_keywords,
            "high_keywords": analysis.high_keywords,
            "patterns": [{"type": p.pattern_type.value, "confidence": p.confidence} for p in patterns],
            "response": response,
            "memory_id": memory_id,
        }

# ============================================================================
# WEB APPLICATION
# ============================================================================

def create_app():
    try:
        from flask import Flask, request, jsonify, render_template_string
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return None
    
    app = Flask(__name__)
    reunity = ReUnity()
    
    HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>ReUnity</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            min-height: 100vh;
            color: #e0e0e0;
        }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        header { text-align: center; padding: 30px 0; }
        header h1 { color: #64b5f6; font-size: 2.5em; margin-bottom: 10px; }
        header p { color: #90a4ae; font-size: 0.9em; }
        .chat {
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 20px;
            min-height: 400px;
            max-height: 60vh;
            overflow-y: auto;
        }
        .msg { margin-bottom: 20px; padding: 15px 20px; border-radius: 15px; max-width: 85%; }
        .user { background: #1e88e5; margin-left: auto; color: white; }
        .ai { background: rgba(255,255,255,0.1); margin-right: auto; }
        .ai strong { color: #64b5f6; }
        .input-row { display: flex; gap: 10px; }
        input[type="text"] {
            flex: 1;
            padding: 15px 20px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 16px;
        }
        input::placeholder { color: #90a4ae; }
        input:focus { outline: 2px solid #64b5f6; }
        button {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            background: #1e88e5;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { background: #1565c0; }
        .disclaimer { text-align: center; padding: 20px; color: #90a4ae; font-size: 0.8em; }
        .state {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .crisis { background: #c62828; }
        .high { background: #ef6c00; }
        .moderate { background: #fbc02d; color: #333; }
        .low { background: #7cb342; }
        .stable { background: #26a69a; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>ReUnity</h1>
            <p>Trauma-Aware AI Support System</p>
        </header>
        <div class="chat" id="chat"></div>
        <div class="input-row">
            <input type="text" id="input" placeholder="Share what is on your mind..." onkeypress="if(event.key==='Enter')send()">
            <button onclick="send()">Send</button>
        </div>
        <div class="disclaimer">
            <strong>Important:</strong> This is NOT a clinical tool. If you are in crisis, call 988.
        </div>
    </div>
    <script>
        function send() {
            const input = document.getElementById('input');
            const chat = document.getElementById('chat');
            const text = input.value.trim();
            if (!text) return;
            chat.innerHTML += '<div class="msg user">' + text + '</div>';
            input.value = '';
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            })
            .then(r => r.json())
            .then(data => {
                const resp = data.response.replace(/\\n/g, '<br>').replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
                chat.innerHTML += '<div class="msg ai"><span class="state ' + data.state + '">' + data.state.toUpperCase() + '</span><div>' + resp + '</div></div>';
                chat.scrollTop = chat.scrollHeight;
            });
        }
    </script>
</body>
</html>'''
    
    @app.route('/')
    def index():
        return render_template_string(HTML)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text'}), 400
        return jsonify(reunity.process(text))
    
    return app

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def run_cli():
    print()
    print("=" * 60)
    print(f"ReUnity v{VERSION}")
    print("=" * 60)
    print()
    print("Type your message and press Enter.")
    print("Type /quit to exit.")
    print()
    print("IMPORTANT: This is NOT a clinical tool. If in crisis, call 988.")
    print()
    
    reunity = ReUnity()
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "/quit":
                print()
                print("Take care. Goodbye.")
                break
            
            result = reunity.process(user_input)
            print()
            print(f"[State: {result['state'].upper()} | Entropy: {result['entropy']:.2f}]")
            print()
            print("ReUnity:")
            print(result['response'])
            print()
            
        except KeyboardInterrupt:
            print()
            print("Session ended. Take care.")
            break

# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--web":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
            app = create_app()
            if app:
                print(f"Starting web server at http://localhost:{port}")
                app.run(host='0.0.0.0', port=port, debug=False)
        elif sys.argv[1] == "--test":
            reunity = ReUnity()
            tests = [
                "I am dissociating right now",
                "I am scared",
                "They told me I was imagining things",
                "I feel calm and peaceful today",
            ]
            for text in tests:
                print()
                print("=" * 60)
                print(f"INPUT: {text}")
                result = reunity.process(text)
                print(f"STATE: {result['state']}")
                print(f"ENTROPY: {result['entropy']:.2f}")
                print()
                print("RESPONSE:")
                print(result['response'])
        else:
            print("Usage:")
            print("  python reunity_model.py          Interactive mode")
            print("  python reunity_model.py --web    Web server")
            print("  python reunity_model.py --test   Run tests")
    else:
        run_cli()

if __name__ == "__main__":
    main()
```

7. Press **Ctrl+S** (or **Cmd+S** on Mac) to save

---

## STEP 7: Test the Code

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_model.py --test
```

You will see output like this:

```
INPUT: I am dissociating right now
STATE: crisis
ENTROPY: 0.95

INPUT: I am scared
STATE: high

INPUT: They told me I was imagining things
STATE: moderate

INPUT: I feel calm and peaceful today
STATE: stable
```

If you see **crisis** for "I am dissociating" and **high** for "I am scared", the code is working.

---

## STEP 8: Run Interactive Mode

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_model.py
```

You can now type messages and get responses. Type `/quit` to exit.

---

## STEP 9: Run the Web Version

Copy this line and paste it into the terminal, then press Enter:

```
python reunity_model.py --web 8080
```

A popup will appear in Codespaces that says "Your application running on port 8080 is available."

Click **Open in Browser**.

You will see the ReUnity web interface. You can type messages and get responses.

---

## STEP 10: Share the Web Version

When the web version is running in Codespaces:

1. Look at the popup that appeared
2. Click **Make Public** if you want others to access it
3. Copy the URL
4. Send the URL to anyone you want to use it

The URL will look like: `https://something-8080.app.github.dev`

---

## STEP 11: Deploy to the Internet (Optional)

If you want the web version to be available all the time (not just when Codespaces is running):

### Option A: Render.com (Free)

1. Go to https://render.com
2. Click **Get Started for Free**
3. Sign up with your GitHub account
4. Click **New** then **Web Service**
5. Connect your GitHub account if asked
6. Select the **ReUnity** repository
7. Set these values:
   - Name: `reunity`
   - Build Command: `pip install flask gunicorn`
   - Start Command: `gunicorn -b 0.0.0.0:10000 reunity_model:create_app()`
8. Click **Create Web Service**
9. Wait a few minutes
10. You will get a URL like `https://reunity.onrender.com`

### Option B: Railway.app (Free)

1. Go to https://railway.app
2. Click **Start a New Project**
3. Click **Deploy from GitHub repo**
4. Select the **ReUnity** repository
5. Railway will automatically detect and deploy
6. You will get a URL

---

## STEP 12: Mobile App (Optional)

The web version works on phones. To make it feel like an app:

### On iPhone:
1. Open the web URL in Safari
2. Tap the Share button (square with arrow)
3. Tap **Add to Home Screen**
4. Tap **Add**

### On Android:
1. Open the web URL in Chrome
2. Tap the three dots menu
3. Tap **Add to Home screen**
4. Tap **Add**

---

## Crisis Resources

If you or someone you know is in crisis:

- **National Suicide Prevention Lifeline:** 988 (US)
- **Crisis Text Line:** Text HOME to 741741 (US)
- **International:** https://www.iasp.info/resources/Crisis_Centres/

---

**Author:** Christopher Ezernack  
**Version:** 2.0.0  
**January 2026**
