import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, Pause, SkipBack, SkipForward, Volume2, VolumeX } from "lucide-react";

// Meditation scripts for each technique
const meditationScripts = {
  boxBreathing: {
    title: "Box Breathing Meditation",
    duration: 180, // 3 minutes
    steps: [
      { time: 0, text: "Welcome to this guided box breathing meditation. Find a comfortable position and let your body relax.", duration: 8 },
      { time: 8, text: "We'll breathe together in a square pattern. Inhale for 4 counts, hold for 4, exhale for 4, and hold for 4.", duration: 10 },
      { time: 18, text: "Let's begin. Breathe in... 1... 2... 3... 4...", duration: 6 },
      { time: 24, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 29, text: "Breathe out... 1... 2... 3... 4...", duration: 6 },
      { time: 35, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 40, text: "Breathe in... 1... 2... 3... 4...", duration: 6 },
      { time: 46, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 51, text: "Breathe out... 1... 2... 3... 4...", duration: 6 },
      { time: 57, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 62, text: "You're doing wonderfully. Continue this pattern.", duration: 4 },
      { time: 66, text: "Breathe in... 1... 2... 3... 4...", duration: 6 },
      { time: 72, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 77, text: "Breathe out... 1... 2... 3... 4...", duration: 6 },
      { time: 83, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 88, text: "Breathe in... 1... 2... 3... 4...", duration: 6 },
      { time: 94, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 99, text: "Breathe out... 1... 2... 3... 4...", duration: 6 },
      { time: 105, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 110, text: "Notice how your body feels. Each breath brings more calm.", duration: 6 },
      { time: 116, text: "Breathe in... 1... 2... 3... 4...", duration: 6 },
      { time: 122, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 127, text: "Breathe out... 1... 2... 3... 4...", duration: 6 },
      { time: 133, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 138, text: "Breathe in... 1... 2... 3... 4...", duration: 6 },
      { time: 144, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 149, text: "Breathe out... 1... 2... 3... 4...", duration: 6 },
      { time: 155, text: "Hold... 1... 2... 3... 4...", duration: 5 },
      { time: 160, text: "Let your breathing return to normal. You've done beautifully.", duration: 8 },
      { time: 168, text: "Take a moment to notice the calm you've created. You can return to this practice anytime.", duration: 12 },
    ]
  },
  grounding54321: {
    title: "5-4-3-2-1 Grounding Meditation",
    duration: 240, // 4 minutes
    steps: [
      { time: 0, text: "Welcome to this grounding meditation. This technique will help bring you back to the present moment.", duration: 8 },
      { time: 8, text: "Take a deep breath in... and slowly release it.", duration: 6 },
      { time: 14, text: "We'll use your five senses to anchor you to the here and now.", duration: 5 },
      { time: 19, text: "First, look around you. Name five things you can see.", duration: 6 },
      { time: 25, text: "Maybe you see a wall... a light... a piece of furniture... a color... a shape.", duration: 10 },
      { time: 35, text: "Take your time. There's no rush.", duration: 5 },
      { time: 40, text: "Now, notice four things you can touch or feel.", duration: 5 },
      { time: 45, text: "Perhaps the texture of your clothes... the surface beneath you... the temperature of the air... your feet on the ground.", duration: 12 },
      { time: 57, text: "You're doing wonderfully. Stay present.", duration: 4 },
      { time: 61, text: "Listen carefully. What are three things you can hear?", duration: 5 },
      { time: 66, text: "Maybe distant sounds... your own breathing... the hum of electronics... or silence itself.", duration: 10 },
      { time: 76, text: "Now, two things you can smell.", duration: 4 },
      { time: 80, text: "This might be subtle. The air... a familiar scent... or nothing at all, which is okay.", duration: 10 },
      { time: 90, text: "Finally, one thing you can taste.", duration: 4 },
      { time: 94, text: "Notice what's in your mouth right now. The taste of your last drink... or simply the taste of your own mouth.", duration: 10 },
      { time: 104, text: "Take another deep breath.", duration: 4 },
      { time: 108, text: "You've just completed the 5-4-3-2-1 grounding technique.", duration: 5 },
      { time: 113, text: "Notice how you feel now compared to when we started.", duration: 5 },
      { time: 118, text: "You are here. You are present. You are safe in this moment.", duration: 7 },
      { time: 125, text: "Whenever you feel overwhelmed, you can return to this practice.", duration: 6 },
    ]
  },
  bodyRelaxation: {
    title: "Progressive Body Relaxation",
    duration: 300, // 5 minutes
    steps: [
      { time: 0, text: "Welcome to this progressive relaxation meditation. Find a comfortable position.", duration: 7 },
      { time: 7, text: "Close your eyes if that feels comfortable. Take a deep breath.", duration: 6 },
      { time: 13, text: "We'll move through your body, releasing tension as we go.", duration: 5 },
      { time: 18, text: "Start with your feet. Notice any tension there. Now, let it go.", duration: 8 },
      { time: 26, text: "Feel your feet becoming heavy and relaxed.", duration: 5 },
      { time: 31, text: "Move to your ankles and calves. Release any tightness.", duration: 6 },
      { time: 37, text: "Let the relaxation flow up to your knees.", duration: 5 },
      { time: 42, text: "Now your thighs. Let them sink into whatever surface supports you.", duration: 7 },
      { time: 49, text: "Feel the relaxation spreading through your entire lower body.", duration: 5 },
      { time: 54, text: "Move to your hips and lower back. Release any tension you're holding there.", duration: 7 },
      { time: 61, text: "Let your stomach soften. There's no need to hold it in.", duration: 6 },
      { time: 67, text: "Feel your chest rise and fall with each breath.", duration: 5 },
      { time: 72, text: "Now your shoulders. Let them drop away from your ears.", duration: 6 },
      { time: 78, text: "So much tension lives in our shoulders. Let it all go.", duration: 6 },
      { time: 84, text: "Move down your arms. Relax your upper arms... your elbows... your forearms.", duration: 8 },
      { time: 92, text: "Let your wrists go limp. Relax your hands and fingers.", duration: 6 },
      { time: 98, text: "Return to your neck. Gently release any stiffness.", duration: 5 },
      { time: 103, text: "Relax your jaw. Let your teeth part slightly.", duration: 5 },
      { time: 108, text: "Soften the muscles around your eyes.", duration: 4 },
      { time: 112, text: "Smooth your forehead. Release any furrowed lines.", duration: 5 },
      { time: 117, text: "Your entire body is now relaxed and at peace.", duration: 5 },
      { time: 122, text: "Take a moment to enjoy this feeling of complete relaxation.", duration: 8 },
      { time: 130, text: "Your body knows how to heal. Trust it.", duration: 5 },
      { time: 135, text: "When you're ready, slowly begin to bring awareness back.", duration: 6 },
      { time: 141, text: "Wiggle your fingers and toes gently.", duration: 5 },
      { time: 146, text: "Take a deep breath and open your eyes when you're ready.", duration: 7 },
    ]
  },
  safePlace: {
    title: "Safe Place Visualization",
    duration: 240, // 4 minutes
    steps: [
      { time: 0, text: "Welcome to this safe place visualization. Close your eyes and take a deep breath.", duration: 8 },
      { time: 8, text: "Imagine a place where you feel completely safe and at peace.", duration: 6 },
      { time: 14, text: "This can be a real place you've been, or somewhere from your imagination.", duration: 6 },
      { time: 20, text: "Begin to see this place clearly in your mind.", duration: 5 },
      { time: 25, text: "What do you see around you? Notice the colors, the shapes, the light.", duration: 8 },
      { time: 33, text: "Is it indoors or outdoors? Bright or softly lit?", duration: 5 },
      { time: 38, text: "Now notice what you can hear in this safe place.", duration: 5 },
      { time: 43, text: "Perhaps gentle sounds... or peaceful silence.", duration: 6 },
      { time: 49, text: "What can you feel? The temperature... the textures around you.", duration: 7 },
      { time: 56, text: "Maybe there's a comfortable place to sit or lie down.", duration: 5 },
      { time: 61, text: "Are there any pleasant smells in your safe place?", duration: 5 },
      { time: 66, text: "Let yourself fully arrive in this space.", duration: 5 },
      { time: 71, text: "In this place, nothing can harm you. You are protected.", duration: 6 },
      { time: 77, text: "Feel the safety surrounding you like a warm embrace.", duration: 6 },
      { time: 83, text: "This place exists within you. You can return here anytime.", duration: 6 },
      { time: 89, text: "Take a few moments to simply be here, in your safe place.", duration: 10 },
      { time: 99, text: "Remember this feeling of safety and peace.", duration: 5 },
      { time: 104, text: "Know that you carry this safe place with you always.", duration: 6 },
      { time: 110, text: "When you're ready, slowly begin to return to the present.", duration: 6 },
      { time: 116, text: "Bring with you the calm and safety you've found.", duration: 6 },
      { time: 122, text: "Open your eyes gently. You are safe. You are here.", duration: 8 },
    ]
  }
};

type MeditationType = keyof typeof meditationScripts;

interface GuidedMeditationProps {
  technique?: MeditationType;
  onClose?: () => void;
}

export function GuidedMeditation({ technique = "boxBreathing", onClose }: GuidedMeditationProps) {
  const [selectedTechnique, setSelectedTechnique] = useState<MeditationType>(technique);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentText, setCurrentText] = useState("");
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(0.8);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const speechRef = useRef<SpeechSynthesisUtterance | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const meditation = meditationScripts[selectedTechnique];

  // Initialize audio context
  useEffect(() => {
    audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    return () => {
      audioContextRef.current?.close();
      if (intervalRef.current) clearInterval(intervalRef.current);
      window.speechSynthesis.cancel();
    };
  }, []);

  // Play ambient background tone
  const playAmbientTone = useCallback(() => {
    if (!audioContextRef.current || isMuted) return;
    
    const ctx = audioContextRef.current;
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(174, ctx.currentTime); // Healing frequency
    
    gainNode.gain.setValueAtTime(0, ctx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.05 * volume, ctx.currentTime + 2);
    
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    oscillator.start();
    
    return { oscillator, gainNode };
  }, [isMuted, volume]);

  // Speak text using Web Speech API
  const speakText = useCallback((text: string) => {
    if (isMuted) return;
    
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.85;
    utterance.pitch = 0.95;
    utterance.volume = volume;
    
    // Try to find a calm, soothing voice
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => 
      v.name.includes("Samantha") || 
      v.name.includes("Karen") ||
      v.name.includes("Google UK English Female") ||
      v.lang.startsWith("en")
    );
    if (preferredVoice) utterance.voice = preferredVoice;
    
    speechRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  }, [isMuted, volume]);

  // Update current text based on time
  useEffect(() => {
    const step = meditation.steps.findLast(s => s.time <= currentTime);
    if (step && step.text !== currentText) {
      setCurrentText(step.text);
      if (isPlaying) {
        speakText(step.text);
      }
    }
  }, [currentTime, meditation.steps, currentText, isPlaying, speakText]);

  // Timer for meditation progress
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= meditation.duration) {
            setIsPlaying(false);
            return meditation.duration;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, meditation.duration]);

  const togglePlay = () => {
    if (!isPlaying) {
      // Resume audio context if suspended
      if (audioContextRef.current?.state === "suspended") {
        audioContextRef.current.resume();
      }
      playAmbientTone();
    } else {
      window.speechSynthesis.cancel();
    }
    setIsPlaying(!isPlaying);
  };

  const restart = () => {
    setCurrentTime(0);
    setCurrentText("");
    window.speechSynthesis.cancel();
    if (isPlaying) {
      setTimeout(() => speakText(meditation.steps[0].text), 500);
    }
  };

  const skip = (seconds: number) => {
    setCurrentTime(prev => Math.max(0, Math.min(meditation.duration, prev + seconds)));
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progress = (currentTime / meditation.duration) * 100;

  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg text-white">{meditation.title}</CardTitle>
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose} className="text-slate-400">
              ✕
            </Button>
          )}
        </div>
        
        {/* Technique Selector */}
        <div className="flex flex-wrap gap-2 mt-2">
          {(Object.keys(meditationScripts) as MeditationType[]).map((key) => (
            <Button
              key={key}
              variant={selectedTechnique === key ? "default" : "outline"}
              size="sm"
              onClick={() => {
                setSelectedTechnique(key);
                setCurrentTime(0);
                setCurrentText("");
                setIsPlaying(false);
                window.speechSynthesis.cancel();
              }}
              className={selectedTechnique === key 
                ? "bg-emerald-600 hover:bg-emerald-500" 
                : "border-slate-600 text-slate-300"
              }
            >
              {meditationScripts[key].title.split(" ")[0]}
            </Button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Current Text Display */}
        <div className="bg-slate-900/50 rounded-lg p-4 min-h-[80px] flex items-center justify-center">
          <p className="text-center text-slate-200 text-lg leading-relaxed">
            {currentText || "Press play to begin your guided meditation"}
          </p>
        </div>
        
        {/* Progress Bar */}
        <div className="space-y-1">
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-emerald-500 transition-all duration-1000"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-slate-400">
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(meditation.duration)}</span>
          </div>
        </div>
        
        {/* Controls */}
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => skip(-15)}
            className="text-slate-400 hover:text-white"
          >
            <SkipBack className="w-5 h-5" />
          </Button>
          
          <Button
            onClick={togglePlay}
            className="w-14 h-14 rounded-full bg-emerald-600 hover:bg-emerald-500"
          >
            {isPlaying ? (
              <Pause className="w-6 h-6" />
            ) : (
              <Play className="w-6 h-6 ml-1" />
            )}
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={() => skip(15)}
            className="text-slate-400 hover:text-white"
          >
            <SkipForward className="w-5 h-5" />
          </Button>
        </div>
        
        {/* Volume and Restart */}
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={restart}
            className="text-slate-400 hover:text-white"
          >
            Restart
          </Button>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMuted(!isMuted)}
              className="text-slate-400 hover:text-white"
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </Button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => setVolume(parseFloat(e.target.value))}
              className="w-20 accent-emerald-500"
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default GuidedMeditation;
