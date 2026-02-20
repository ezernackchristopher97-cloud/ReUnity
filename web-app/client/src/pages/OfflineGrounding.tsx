import { useState, useEffect, useCallback } from "react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Wifi, 
  WifiOff, 
  ChevronLeft, 
  ChevronRight, 
  Heart, 
  Wind, 
  Eye, 
  Snowflake,
  Hand,
  Brain,
  Timer,
  Volume2,
  VolumeX,
  RefreshCw,
  SkipForward,
  Pause
} from "lucide-react";

// Offline grounding techniques - stored locally, no network needed
const offlineGroundingTechniques = [
  {
    id: "5-4-3-2-1",
    name: "5-4-3-2-1 Grounding",
    icon: Eye,
    duration: "3-5 minutes",
    description: "Engage all five senses to anchor yourself in the present moment.",
    steps: [
      { sense: "SEE", count: 5, instruction: "Name 5 things you can see right now. Look around slowly - a crack in the wall, the color of the sky, your own hands." },
      { sense: "TOUCH", count: 4, instruction: "Name 4 things you can physically feel. The texture of your clothes, the ground under your feet, the air on your skin." },
      { sense: "HEAR", count: 3, instruction: "Name 3 things you can hear. Distant sounds, your own breathing, the hum of silence." },
      { sense: "SMELL", count: 2, instruction: "Name 2 things you can smell. If you can't smell anything, name 2 smells you like." },
      { sense: "TASTE", count: 1, instruction: "Name 1 thing you can taste. The inside of your mouth, your last meal, or imagine your favorite taste." }
    ],
    affirmation: "You are here. You are present. You are safe in this moment."
  },
  {
    id: "box-breathing",
    name: "Box Breathing",
    icon: Wind,
    duration: "4-8 minutes",
    description: "A powerful technique used by Navy SEALs to calm the nervous system.",
    steps: [
      { phase: "INHALE", count: 4, instruction: "Breathe in slowly through your nose for 4 seconds. Feel your belly expand." },
      { phase: "HOLD", count: 4, instruction: "Hold your breath gently for 4 seconds. Don't strain." },
      { phase: "EXHALE", count: 4, instruction: "Release slowly through your mouth for 4 seconds. Let go of tension." },
      { phase: "HOLD", count: 4, instruction: "Hold empty for 4 seconds. Rest in the stillness." }
    ],
    cycles: 4,
    affirmation: "With each breath, you are calming your nervous system. You are in control."
  },
  {
    id: "cold-water",
    name: "Cold Water Reset",
    icon: Snowflake,
    duration: "30 seconds - 2 minutes",
    description: "Activates the dive reflex to rapidly calm your nervous system.",
    steps: [
      { instruction: "Find cold water - from a tap, a water bottle, or even ice cubes." },
      { instruction: "Splash cold water on your face, especially your forehead and cheeks." },
      { instruction: "If possible, hold a cold object against your wrists or the back of your neck." },
      { instruction: "Take slow breaths while feeling the cold sensation." },
      { instruction: "Notice how your heart rate begins to slow." }
    ],
    affirmation: "The cold is bringing you back to your body. You are grounded."
  },
  {
    id: "feet-on-floor",
    name: "Feet on Floor",
    icon: Hand,
    duration: "1-3 minutes",
    description: "A simple but powerful technique to reconnect with your physical body.",
    steps: [
      { instruction: "If possible, remove your shoes. Feel the ground directly." },
      { instruction: "Press your feet firmly into the floor. Feel the pressure." },
      { instruction: "Notice the temperature of the floor - is it warm or cool?" },
      { instruction: "Wiggle your toes slowly. Feel each one individually." },
      { instruction: "Imagine roots growing from your feet deep into the earth." },
      { instruction: "You are connected to the ground. You are stable. You are here." }
    ],
    affirmation: "You are rooted. The earth is holding you. You are supported."
  },
  {
    id: "butterfly-hug",
    name: "Butterfly Hug",
    icon: Heart,
    duration: "2-5 minutes",
    description: "Bilateral stimulation to calm both hemispheres of your brain.",
    steps: [
      { instruction: "Cross your arms over your chest, fingertips resting below your collarbones." },
      { instruction: "Your hands should look like butterfly wings." },
      { instruction: "Slowly alternate tapping your hands - left, right, left, right." },
      { instruction: "Keep a steady, slow rhythm. Like a heartbeat." },
      { instruction: "Close your eyes if comfortable. Focus on the sensation." },
      { instruction: "Continue for 2-5 minutes or until you feel calmer." }
    ],
    affirmation: "You are giving yourself comfort. You deserve this care."
  },
  {
    id: "progressive-relaxation",
    name: "Progressive Muscle Relaxation",
    icon: Brain,
    duration: "10-15 minutes",
    description: "Systematically release tension from every part of your body.",
    steps: [
      { area: "FEET", instruction: "Curl your toes tightly for 5 seconds... then release. Feel the difference." },
      { area: "CALVES", instruction: "Tense your calf muscles for 5 seconds... then release. Notice the warmth." },
      { area: "THIGHS", instruction: "Squeeze your thigh muscles for 5 seconds... then release. Let them go heavy." },
      { area: "STOMACH", instruction: "Tighten your stomach muscles for 5 seconds... then release. Breathe deeply." },
      { area: "HANDS", instruction: "Make tight fists for 5 seconds... then release. Feel your fingers tingle." },
      { area: "ARMS", instruction: "Tense your biceps for 5 seconds... then release. Let your arms go limp." },
      { area: "SHOULDERS", instruction: "Raise your shoulders to your ears for 5 seconds... then drop them. Feel the release." },
      { area: "FACE", instruction: "Scrunch your face tightly for 5 seconds... then release. Soften your jaw." }
    ],
    affirmation: "Your body is releasing what it no longer needs. You are safe to relax."
  },
  {
    id: "safe-place",
    name: "Safe Place Visualization",
    icon: Eye,
    duration: "5-10 minutes",
    description: "Create a mental sanctuary you can visit anytime, anywhere.",
    steps: [
      { instruction: "Close your eyes. Take three slow, deep breaths." },
      { instruction: "Imagine a place where you feel completely safe. It can be real or imagined." },
      { instruction: "What do you SEE? Colors, shapes, light, details..." },
      { instruction: "What do you HEAR? Sounds of nature, silence, music..." },
      { instruction: "What do you FEEL? Temperature, textures, the ground beneath you..." },
      { instruction: "What do you SMELL? Fresh air, flowers, the ocean, something comforting..." },
      { instruction: "You are safe here. Nothing can harm you in this place." },
      { instruction: "Stay as long as you need. This place is always available to you." }
    ],
    affirmation: "You carry this safe place within you. It is always accessible."
  },
  {
    id: "grounding-statements",
    name: "Grounding Statements",
    icon: Brain,
    duration: "2-5 minutes",
    description: "Use words to anchor yourself in reality and the present moment.",
    statements: [
      "My name is _____. I am _____ years old.",
      "Today is _____. The date is _____.",
      "I am in _____. I am safe right now.",
      "I can feel my feet on the ground.",
      "I can feel my hands. I can move my fingers.",
      "This feeling will pass. Feelings are temporary.",
      "I have survived difficult moments before.",
      "I am doing the best I can right now.",
      "I am allowed to take up space.",
      "I am not alone, even when I feel alone.",
      "This moment is not forever.",
      "I am stronger than I feel right now."
    ],
    affirmation: "Your words have power. Speaking truth anchors you in reality."
  }
];

// Crisis resources that work offline (phone numbers)
const offlineCrisisResources = [
  { name: "National Suicide Prevention Lifeline", number: "988" },
  { name: "Crisis Text Line", number: "Text HOME to 741741" },
  { name: "National DV Hotline", number: "1-800-799-7233" },
  { name: "SAMHSA Helpline", number: "1-800-662-4357" },
  { name: "Emergency Services", number: "911" }
];

export default function OfflineGrounding() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [currentTechnique, setCurrentTechnique] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timer, setTimer] = useState(0);
  const [breathPhase, setBreathPhase] = useState(0);
  
  // Text-to-speech state
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [autoAdvance, setAutoAdvance] = useState(true); // Auto-advance when audio finishes
  
  // Audio context for breathing tones
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);

  // Check if speech synthesis is available and initialize audio context
  useEffect(() => {
    setSpeechSupported('speechSynthesis' in window);
    // Initialize AudioContext for breathing tones (lazy init on user interaction)
  }, []);

  // Initialize audio context on first interaction
  const initAudioContext = useCallback(() => {
    if (!audioContext) {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      setAudioContext(ctx);
      return ctx;
    }
    return audioContext;
  }, [audioContext]);

  // Play breathing tone
  const playBreathTone = useCallback((type: 'inhale' | 'exhale' | 'hold') => {
    if (!audioEnabled) return;
    
    const ctx = initAudioContext();
    if (!ctx) return;
    
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);
    
    // Different tones for different phases
    if (type === 'inhale') {
      // Rising tone for inhale - calming frequency
      oscillator.frequency.setValueAtTime(220, ctx.currentTime); // A3
      oscillator.frequency.linearRampToValueAtTime(330, ctx.currentTime + 0.5); // E4
      oscillator.type = 'sine';
    } else if (type === 'exhale') {
      // Falling tone for exhale
      oscillator.frequency.setValueAtTime(330, ctx.currentTime); // E4
      oscillator.frequency.linearRampToValueAtTime(220, ctx.currentTime + 0.5); // A3
      oscillator.type = 'sine';
    } else {
      // Soft sustained tone for hold
      oscillator.frequency.setValueAtTime(261.63, ctx.currentTime); // C4
      oscillator.type = 'sine';
    }
    
    // Gentle volume envelope
    gainNode.gain.setValueAtTime(0, ctx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
    gainNode.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.3);
    gainNode.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.5);
    
    oscillator.start(ctx.currentTime);
    oscillator.stop(ctx.currentTime + 0.5);
  }, [audioEnabled, initAudioContext, audioContext]);

  // Speech synthesis function
  const speak = useCallback((text: string, onEnd?: () => void) => {
    if (!speechSupported || !audioEnabled) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.85; // Slightly slower for calming effect
    utterance.pitch = 0.95;
    utterance.volume = 1;
    
    // Try to get a calm, soothing voice
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => 
      v.name.includes('Samantha') || 
      v.name.includes('Karen') || 
      v.name.includes('Google US English') ||
      v.lang.startsWith('en')
    );
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }
    
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => {
      setIsSpeaking(false);
      if (onEnd) onEnd();
    };
    utterance.onerror = () => setIsSpeaking(false);
    
    window.speechSynthesis.speak(utterance);
  }, [speechSupported, audioEnabled]);

  // Stop speaking
  const stopSpeaking = useCallback(() => {
    if (speechSupported) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  }, [speechSupported]);

  // Auto-read current step when audio is enabled with auto-advance
  useEffect(() => {
    if (!audioEnabled || !speechSupported) return;
    
    const technique = offlineGroundingTechniques[currentTechnique];
    const steps = technique.steps || technique.statements;
    
    if (steps && steps[currentStep]) {
      const step = steps[currentStep] as any;
      let textToRead = '';
      
      if (step.sense) {
        textToRead = `${step.sense}. ${step.count}. ${step.instruction}`;
      } else if (step.phase) {
        textToRead = `${step.phase}. ${step.instruction}`;
      } else if (step.area) {
        textToRead = `${step.area}. ${step.instruction}`;
      } else if (typeof step === 'string') {
        textToRead = step;
      } else if (step.instruction) {
        textToRead = step.instruction;
      }
      
      if (textToRead) {
        // Auto-advance callback when speech ends
        const onSpeechEnd = () => {
          if (autoAdvance && steps && currentStep < steps.length - 1) {
            // Wait a moment before advancing
            setTimeout(() => {
              setCurrentStep(prev => prev + 1);
            }, 1500); // 1.5 second pause between steps
          } else if (autoAdvance && steps && currentStep === steps.length - 1) {
            // Read affirmation at the end
            if (technique.affirmation) {
              setTimeout(() => {
                speak(technique.affirmation as string);
              }, 1000);
            }
          }
        };
        speak(textToRead, onSpeechEnd);
      }
    }
    
    return () => {
      stopSpeaking();
    };
  }, [currentStep, currentTechnique, audioEnabled, speechSupported, speak, stopSpeaking, autoAdvance]);

  // Toggle audio guidance
  const toggleAudio = () => {
    if (audioEnabled) {
      stopSpeaking();
      setAudioEnabled(false);
    } else {
      setAudioEnabled(true);
      // Speak introduction
      const technique = offlineGroundingTechniques[currentTechnique];
      speak(`Starting ${technique.name}. ${technique.description}`);
    }
  };

  // Monitor online status
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    
    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Timer for breathing exercises with audio cues
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying && offlineGroundingTechniques[currentTechnique].id === "box-breathing") {
      interval = setInterval(() => {
        setTimer((prev) => {
          if (prev >= 3) {
            const newPhase = (breathPhase + 1) % 4;
            setBreathPhase(newPhase);
            
            // Play breathing audio cues
            if (audioEnabled) {
              if (newPhase === 0) {
                playBreathTone('inhale');
              } else if (newPhase === 2) {
                playBreathTone('exhale');
              } else {
                playBreathTone('hold');
              }
            }
            
            return 0;
          }
          return prev + 1;
        });
      }, 1000);
    }
    
    return () => clearInterval(interval);
  }, [isPlaying, currentTechnique, breathPhase, audioEnabled, playBreathTone]);

  const technique = offlineGroundingTechniques[currentTechnique];
  const Icon = technique.icon;

  const nextTechnique = () => {
    setCurrentTechnique((prev) => (prev + 1) % offlineGroundingTechniques.length);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const prevTechnique = () => {
    setCurrentTechnique((prev) => (prev - 1 + offlineGroundingTechniques.length) % offlineGroundingTechniques.length);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const nextStep = () => {
    const steps = technique.steps || technique.statements;
    if (steps && currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const resetTechnique = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setTimer(0);
    setBreathPhase(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      {/* Offline Banner */}
      <div className={`py-2 px-4 text-center text-sm ${isOnline ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
        <div className="flex items-center justify-center gap-2">
          {isOnline ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          {isOnline ? "Online - All features available" : "Offline Mode - Grounding techniques still work"}
        </div>
      </div>

      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <Heart className="h-6 w-6 text-emerald-400" />
            <span className="text-xl font-bold text-white">Grounding Techniques</span>
          </Link>
          <div className="flex items-center gap-2">
            {speechSupported && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleAudio}
                  className={`flex items-center gap-2 ${audioEnabled ? 'text-emerald-400' : 'text-slate-400'}`}
                >
                  {audioEnabled ? (
                    <>
                      <Volume2 className={`w-4 h-4 ${isSpeaking ? 'animate-pulse' : ''}`} />
                      <span className="hidden sm:inline">Audio On</span>
                    </>
                  ) : (
                    <>
                      <VolumeX className="w-4 h-4" />
                      <span className="hidden sm:inline">Audio Off</span>
                    </>
                  )}
                </Button>
                {audioEnabled && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setAutoAdvance(!autoAdvance)}
                    className={`flex items-center gap-2 ${autoAdvance ? 'text-emerald-400' : 'text-slate-400'}`}
                    title={autoAdvance ? 'Auto-advance enabled' : 'Auto-advance disabled'}
                  >
                    {autoAdvance ? (
                      <>
                        <SkipForward className="w-4 h-4" />
                        <span className="hidden sm:inline">Auto</span>
                      </>
                    ) : (
                      <>
                        <Pause className="w-4 h-4" />
                        <span className="hidden sm:inline">Manual</span>
                      </>
                    )}
                  </Button>
                )}
              </>
            )}
            <span className="text-sm text-slate-400">Works offline</span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-2xl">
        {/* Technique Selector */}
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            size="icon"
            onClick={prevTechnique}
            className="text-white hover:bg-white/10"
          >
            <ChevronLeft className="h-6 w-6" />
          </Button>
          
          <div className="text-center">
            <p className="text-slate-400 text-sm">
              {currentTechnique + 1} of {offlineGroundingTechniques.length}
            </p>
          </div>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={nextTechnique}
            className="text-white hover:bg-white/10"
          >
            <ChevronRight className="h-6 w-6" />
          </Button>
        </div>

        {/* Main Technique Card */}
        <Card className="bg-slate-800/50 border-slate-700 mb-6">
          <CardHeader className="text-center pb-4">
            <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <Icon className="w-8 h-8 text-emerald-400" />
            </div>
            <CardTitle className="text-2xl text-white">{technique.name}</CardTitle>
            <CardDescription className="text-slate-400">
              <Timer className="w-4 h-4 inline mr-1" />
              {technique.duration}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300 text-center mb-6">{technique.description}</p>

            {/* Box Breathing Special UI */}
            {technique.id === "box-breathing" && (
              <div className="mb-6">
                <div className="relative w-48 h-48 mx-auto">
                  {/* Box visualization */}
                  <div className="absolute inset-0 border-2 border-emerald-500/30 rounded-lg" />
                  
                  {/* Animated indicator */}
                  <div 
                    className={`absolute w-4 h-4 bg-emerald-400 rounded-full transition-all duration-1000 ${
                      breathPhase === 0 ? 'top-0 left-0' :
                      breathPhase === 1 ? 'top-0 right-0' :
                      breathPhase === 2 ? 'bottom-0 right-0' :
                      'bottom-0 left-0'
                    }`}
                  />
                  
                  {/* Center text */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <p className="text-3xl font-bold text-emerald-400">{4 - timer}</p>
                    <p className="text-lg text-white">
                      {(technique.steps?.[breathPhase] as any)?.phase}
                    </p>
                  </div>
                </div>
                
                <div className="flex justify-center gap-4 mt-6">
                  <Button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className={isPlaying ? "bg-amber-600 hover:bg-amber-500" : "bg-emerald-600 hover:bg-emerald-500"}
                  >
                    {isPlaying ? <VolumeX className="w-4 h-4 mr-2" /> : <Volume2 className="w-4 h-4 mr-2" />}
                    {isPlaying ? "Pause" : "Start"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={resetTechnique}
                    className="border-slate-600 text-white hover:bg-slate-700"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Reset
                  </Button>
                </div>
              </div>
            )}

            {/* Step-by-step UI for other techniques */}
            {technique.id !== "box-breathing" && technique.steps && (
              <div className="space-y-4">
                <div className="bg-slate-900/50 rounded-xl p-6">
                  {(technique.steps[currentStep] as any)?.sense && (
                    <p className="text-emerald-400 font-bold text-lg mb-2">
                      {(technique.steps[currentStep] as any).sense}: {(technique.steps[currentStep] as any).count}
                    </p>
                  )}
                  {(technique.steps[currentStep] as any)?.phase && (
                    <p className="text-emerald-400 font-bold text-lg mb-2">
                      {(technique.steps[currentStep] as any).phase}
                    </p>
                  )}
                  {(technique.steps[currentStep] as any)?.area && (
                    <p className="text-emerald-400 font-bold text-lg mb-2">
                      {(technique.steps[currentStep] as any).area}
                    </p>
                  )}
                  <p className="text-white text-lg">
                    {technique.steps[currentStep]?.instruction}
                  </p>
                </div>
                
                <div className="flex items-center justify-between">
                  <Button
                    variant="ghost"
                    onClick={prevStep}
                    disabled={currentStep === 0}
                    className="text-white hover:bg-white/10"
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Previous
                  </Button>
                  
                  <span className="text-slate-400">
                    Step {currentStep + 1} of {technique.steps.length}
                  </span>
                  
                  <Button
                    variant="ghost"
                    onClick={nextStep}
                    disabled={currentStep === technique.steps.length - 1}
                    className="text-white hover:bg-white/10"
                  >
                    Next
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </div>
            )}

            {/* Grounding statements */}
            {technique.statements && (
              <div className="space-y-4">
                <div className="bg-slate-900/50 rounded-xl p-6">
                  <p className="text-white text-xl text-center">
                    "{technique.statements[currentStep]}"
                  </p>
                </div>
                
                <div className="flex items-center justify-between">
                  <Button
                    variant="ghost"
                    onClick={prevStep}
                    disabled={currentStep === 0}
                    className="text-white hover:bg-white/10"
                  >
                    <ChevronLeft className="w-4 h-4 mr-1" />
                    Previous
                  </Button>
                  
                  <span className="text-slate-400">
                    {currentStep + 1} of {technique.statements.length}
                  </span>
                  
                  <Button
                    variant="ghost"
                    onClick={nextStep}
                    disabled={currentStep === technique.statements.length - 1}
                    className="text-white hover:bg-white/10"
                  >
                    Next
                    <ChevronRight className="w-4 h-4 ml-1" />
                  </Button>
                </div>
              </div>
            )}

            {/* Affirmation */}
            <div className="mt-6 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
              <p className="text-emerald-300 text-center italic">
                "{technique.affirmation}"
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Quick Technique List */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-8">
          {offlineGroundingTechniques.map((t, i) => {
            const TIcon = t.icon;
            return (
              <button
                key={t.id}
                onClick={() => {
                  setCurrentTechnique(i);
                  setCurrentStep(0);
                  setIsPlaying(false);
                }}
                className={`p-3 rounded-lg text-center transition-all ${
                  i === currentTechnique 
                    ? 'bg-emerald-500/20 border border-emerald-500/50' 
                    : 'bg-slate-800/50 border border-slate-700 hover:bg-slate-700/50'
                }`}
              >
                <TIcon className={`w-5 h-5 mx-auto mb-1 ${i === currentTechnique ? 'text-emerald-400' : 'text-slate-400'}`} />
                <p className={`text-xs ${i === currentTechnique ? 'text-emerald-300' : 'text-slate-400'}`}>
                  {t.name.split(' ')[0]}
                </p>
              </button>
            );
          })}
        </div>

        {/* Crisis Resources - Always Available */}
        <Card className="bg-red-500/10 border-red-500/30">
          <CardHeader>
            <CardTitle className="text-red-300 text-lg">Crisis Resources (Always Available)</CardTitle>
            <CardDescription className="text-red-200/60">
              These numbers work even without internet
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {offlineCrisisResources.map((resource) => (
                <div key={resource.name} className="flex items-center justify-between">
                  <span className="text-slate-300">{resource.name}</span>
                  <a 
                    href={`tel:${resource.number.replace(/\D/g, '')}`}
                    className="text-red-400 font-mono hover:text-red-300"
                  >
                    {resource.number}
                  </a>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
