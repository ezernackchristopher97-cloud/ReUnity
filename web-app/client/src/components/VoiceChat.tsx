import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Phone, PhoneOff, Mic, MicOff, Volume2, VolumeX, Settings, X, Loader2 } from 'lucide-react';
import { trpc } from '@/lib/trpc';

// OpenAI TTS Voice Options - Natural human voices
const VOICE_OPTIONS = [
  // Female voices
  { id: 'nova', name: 'Nova', gender: 'female', description: 'Warm, friendly female voice', accent: 'American', tone: 'warm' },
  { id: 'shimmer', name: 'Shimmer', gender: 'female', description: 'Soft, gentle female voice', accent: 'American', tone: 'gentle' },
  // Male voices
  { id: 'echo', name: 'Echo', gender: 'male', description: 'Deep, calm male voice', accent: 'American', tone: 'calm' },
  { id: 'onyx', name: 'Onyx', gender: 'male', description: 'Rich, warm male voice', accent: 'American', tone: 'warm' },
  { id: 'fable', name: 'Fable', gender: 'male', description: 'Expressive male voice', accent: 'British', tone: 'expressive' },
  // Neutral
  { id: 'alloy', name: 'Alloy', gender: 'neutral', description: 'Balanced, neutral voice', accent: 'American', tone: 'neutral' },
] as const;

type VoiceId = typeof VOICE_OPTIONS[number]['id'];
type Gender = 'female' | 'male' | 'neutral' | 'all';

interface VoiceChatProps {
  onTranscript?: (text: string) => void;
  onSendMessage?: (text: string) => Promise<string>;
  className?: string;
  compact?: boolean;
}

export function VoiceChat({ 
  onTranscript, 
  onSendMessage,
  className = "",
  compact = false
}: VoiceChatProps) {
  const [isInCall, setIsInCall] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<VoiceId>('nova');
  const [genderFilter, setGenderFilter] = useState<Gender>('all');
  const [speed, setSpeed] = useState(1.0);
  const [volume, setVolume] = useState(0.8);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const recognitionRef = useRef<any>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  // TTS mutation
  const speakMutation = trpc.tts.speak.useMutation();
  
  // Filter voices by gender
  const filteredVoices = genderFilter === 'all' 
    ? VOICE_OPTIONS 
    : VOICE_OPTIONS.filter(v => v.gender === genderFilter);
  
  // Get current voice info
  const currentVoice = VOICE_OPTIONS.find(v => v.id === selectedVoice) || VOICE_OPTIONS[0];
  
  // Initialize audio element
  useEffect(() => {
    audioRef.current = new Audio();
    audioRef.current.volume = volume;
    
    audioRef.current.onended = () => {
      setIsSpeaking(false);
      // Resume listening after speaking
      if (isInCall && recognitionRef.current && !isMuted) {
        try {
          recognitionRef.current.start();
          setIsListening(true);
        } catch (e) {
          // Already started
        }
      }
    };
    
    audioRef.current.onerror = (e) => {
      console.error('[VoiceChat] Audio error:', e);
      setIsSpeaking(false);
      setIsLoadingAudio(false);
    };
    
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
    };
  }, []);
  
  // Update volume when changed
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);
  
  // Speak text using OpenAI TTS
  const speak = useCallback(async (text: string) => {
    if (!text || isMuted) return;
    
    // Stop listening while speaking
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
    
    // Stop any current speech
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    
    setIsLoadingAudio(true);
    setIsSpeaking(true);
    
    try {
      const result = await speakMutation.mutateAsync({
        text,
        voice: selectedVoice,
        speed
      });
      
      if (audioRef.current && result.audioUrl) {
        audioRef.current.src = result.audioUrl;
        audioRef.current.volume = volume;
        await audioRef.current.play();
      }
    } catch (error) {
      console.error('[VoiceChat] TTS error:', error);
      setIsSpeaking(false);
    } finally {
      setIsLoadingAudio(false);
    }
  }, [selectedVoice, speed, volume, isMuted, speakMutation, isListening]);
  
  // Handle sending voice message to AI
  const handleSendVoiceMessage = useCallback(async (text: string) => {
    if (!text.trim() || !onSendMessage) return;

    setIsProcessing(true);
    setTranscript("");

    try {
      const response = await onSendMessage(text);
      if (response) {
        speak(response);
      }
    } catch (err) {
      console.error("Error sending voice message:", err);
      setError("Failed to get response. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  }, [onSendMessage, speak]);
  
  // Initialize speech recognition
  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onresult = (event: any) => {
          let finalTranscript = "";
          let interimTranscript = "";

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcriptText = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcriptText;
            } else {
              interimTranscript += transcriptText;
            }
          }

          if (finalTranscript) {
            setTranscript(finalTranscript);
            onTranscript?.(finalTranscript);
            
            // Reset silence timer
            if (silenceTimerRef.current) {
              clearTimeout(silenceTimerRef.current);
            }
            
            // Auto-send after 2 seconds of silence
            silenceTimerRef.current = setTimeout(() => {
              if (finalTranscript.trim() && isInCall) {
                handleSendVoiceMessage(finalTranscript.trim());
              }
            }, 2000);
          } else if (interimTranscript) {
            setTranscript(interimTranscript);
          }
        };

        recognition.onerror = (event: any) => {
          console.error("Speech recognition error:", event.error);
          if (event.error !== "no-speech") {
            setError(`Voice recognition error: ${event.error}`);
          }
          setIsListening(false);
        };

        recognition.onend = () => {
          // Restart if call is still active and not speaking
          if (isInCall && !isSpeaking && !isMuted) {
            try {
              recognition.start();
            } catch (e) {
              // Already started
            }
          } else {
            setIsListening(false);
          }
        };

        recognitionRef.current = recognition;
      } else {
        setError("Speech recognition not supported in this browser");
      }
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
      }
    };
  }, [isInCall, isSpeaking, isMuted, onTranscript, handleSendVoiceMessage]);
  
  // Start call
  const startCall = useCallback(() => {
    setIsInCall(true);
    setError(null);
    setTranscript("");
    
    if (recognitionRef.current) {
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (e) {
        console.error("Failed to start recognition:", e);
      }
    }

    // Greeting message
    speak("Hello, I'm here with you. Take your time, and speak whenever you're ready. I'm listening.");
  }, [speak]);
  
  // End call
  const endCall = useCallback(() => {
    setIsInCall(false);
    setIsListening(false);
    setIsSpeaking(false);
    setTranscript("");
    
    // Stop recognition
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {
        // Already stopped
      }
    }
    
    // Stop audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }
  }, []);
  
  // Toggle mute
  const toggleMute = useCallback(() => {
    setIsMuted(prev => {
      const newMuted = !prev;
      
      if (newMuted) {
        // Stop listening and speaking
        if (recognitionRef.current) {
          try {
            recognitionRef.current.stop();
          } catch (e) {}
        }
        setIsListening(false);
        
        if (audioRef.current) {
          audioRef.current.pause();
        }
        setIsSpeaking(false);
      } else if (isInCall) {
        // Resume listening
        if (recognitionRef.current) {
          try {
            recognitionRef.current.start();
            setIsListening(true);
          } catch (e) {}
        }
      }
      
      return newMuted;
    });
  }, [isInCall]);
  
  // Test voice
  const testVoice = useCallback(() => {
    speak("Hello, I'm here to support you. How are you feeling today?");
  }, [speak]);
  
  // Render settings panel
  const renderSettings = () => (
    <div className="absolute bottom-full left-0 right-0 mb-2 bg-zinc-900 border border-zinc-700 rounded-lg p-4 shadow-xl z-50 max-h-[70vh] overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-medium">Choose Your Companion's Voice</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowSettings(false)}
          className="text-zinc-400 hover:text-white"
        >
          Done
        </Button>
      </div>
      
      <p className="text-zinc-400 text-sm mb-4">
        Everyone is welcome here. Choose a voice that feels comfortable for you. These are natural AI voices powered by OpenAI.
      </p>
      
      {/* Gender filter */}
      <div className="flex gap-2 mb-4">
        {(['all', 'female', 'male', 'neutral'] as Gender[]).map(g => (
          <Button
            key={g}
            variant={genderFilter === g ? 'default' : 'outline'}
            size="sm"
            onClick={() => setGenderFilter(g)}
            className={genderFilter === g ? 'bg-emerald-600' : 'border-zinc-600'}
          >
            {g.charAt(0).toUpperCase() + g.slice(1)}
          </Button>
        ))}
      </div>
      
      {/* Voice list */}
      <div className="space-y-2 mb-4">
        {filteredVoices.map(voice => (
          <button
            key={voice.id}
            onClick={() => setSelectedVoice(voice.id)}
            className={`w-full p-3 rounded-lg text-left transition-colors flex items-center gap-3 ${
              selectedVoice === voice.id 
                ? 'bg-emerald-600/20 border border-emerald-500' 
                : 'bg-zinc-800 hover:bg-zinc-700 border border-transparent'
            }`}
          >
            <div className="w-10 h-10 rounded-full bg-zinc-700 flex items-center justify-center text-lg">
              {voice.gender === 'female' ? '👩' : voice.gender === 'male' ? '👨' : '🧑'}
            </div>
            <div className="flex-1">
              <div className="text-white font-medium">{voice.name}</div>
              <div className="text-zinc-400 text-sm">{voice.description}</div>
            </div>
            <div className="text-zinc-500 text-xs">{voice.accent}</div>
          </button>
        ))}
      </div>
      
      {/* Speed control */}
      <div className="mb-4">
        <label className="text-zinc-400 text-sm block mb-2">
          Speed: {speed.toFixed(1)}x
        </label>
        <input
          type="range"
          min="0.5"
          max="2.0"
          step="0.1"
          value={speed}
          onChange={(e) => setSpeed(parseFloat(e.target.value))}
          className="w-full accent-emerald-500"
        />
      </div>
      
      {/* Volume control */}
      <div className="mb-4">
        <label className="text-zinc-400 text-sm block mb-2">
          Volume: {Math.round(volume * 100)}%
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={volume}
          onChange={(e) => setVolume(parseFloat(e.target.value))}
          className="w-full accent-emerald-500"
        />
      </div>
      
      {/* Test button */}
      <Button
        onClick={testVoice}
        disabled={isLoadingAudio || isSpeaking}
        className="w-full bg-zinc-700 hover:bg-zinc-600"
      >
        {isLoadingAudio ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Loading...
          </>
        ) : isSpeaking ? (
          <>
            <Volume2 className="w-4 h-4 mr-2 animate-pulse" />
            Speaking...
          </>
        ) : (
          <>
            <Volume2 className="w-4 h-4 mr-2" />
            Test Voice
          </>
        )}
      </Button>
    </div>
  );
  
  // Compact mode (for chat page)
  if (compact) {
    return (
      <div className={`relative flex items-center gap-2 ${className}`}>
        {/* Voice Call button */}
        {!isInCall ? (
          <Button
            onClick={startCall}
            className="bg-emerald-600 hover:bg-emerald-700 text-white"
            size="sm"
          >
            <Phone className="w-4 h-4 mr-2" />
            Voice Call
          </Button>
        ) : (
          <Button
            onClick={endCall}
            variant="destructive"
            size="sm"
          >
            <PhoneOff className="w-4 h-4 mr-2" />
            End Call
          </Button>
        )}
        
        {/* Settings button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowSettings(!showSettings)}
          className="text-zinc-400 hover:text-white"
        >
          <Settings className="w-4 h-4" />
        </Button>
        
        {/* Current voice indicator */}
        <span className="text-zinc-400 text-sm hidden sm:inline">
          {currentVoice.name} ({currentVoice.accent})
        </span>
        
        {/* Status indicators */}
        {isInCall && (
          <div className="flex items-center gap-2">
            {isListening && (
              <span className="flex items-center text-emerald-400 text-xs">
                <Mic className="w-3 h-3 mr-1 animate-pulse" />
                Listening
              </span>
            )}
            {isSpeaking && (
              <span className="flex items-center text-blue-400 text-xs">
                <Volume2 className="w-3 h-3 mr-1 animate-pulse" />
                Speaking
              </span>
            )}
            {isLoadingAudio && (
              <Loader2 className="w-3 h-3 animate-spin text-zinc-400" />
            )}
            {isProcessing && (
              <span className="flex items-center text-yellow-400 text-xs">
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                Thinking
              </span>
            )}
          </div>
        )}
        
        {/* Transcript display */}
        {transcript && isInCall && (
          <span className="text-zinc-300 text-xs max-w-[200px] truncate">
            "{transcript}"
          </span>
        )}
        
        {/* Settings panel */}
        {showSettings && renderSettings()}
      </div>
    );
  }
  
  // Full mode
  return (
    <div className={`relative bg-zinc-900 rounded-xl p-6 border border-zinc-800 ${className}`}>
      <div className="text-center mb-6">
        <h2 className="text-xl font-semibold text-white mb-2">Voice Companion</h2>
        <p className="text-zinc-400 text-sm">
          Talk with {currentVoice.name} - {currentVoice.description}
        </p>
        <p className="text-emerald-400 text-xs mt-1">
          Powered by OpenAI natural voices
        </p>
      </div>
      
      {/* Error display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-500/50 rounded-lg text-red-400 text-sm text-center">
          {error}
        </div>
      )}
      
      {/* Transcript display */}
      {transcript && isInCall && (
        <div className="mb-4 p-3 bg-zinc-800 rounded-lg">
          <p className="text-zinc-300 text-sm">"{transcript}"</p>
        </div>
      )}
      
      {/* Main controls */}
      <div className="flex justify-center gap-4 mb-6">
        {!isInCall ? (
          <Button
            onClick={startCall}
            size="lg"
            className="bg-emerald-600 hover:bg-emerald-700 text-white rounded-full w-16 h-16"
          >
            <Phone className="w-6 h-6" />
          </Button>
        ) : (
          <>
            <Button
              onClick={toggleMute}
              size="lg"
              variant={isMuted ? 'destructive' : 'outline'}
              className="rounded-full w-14 h-14"
            >
              {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </Button>
            
            <Button
              onClick={endCall}
              size="lg"
              variant="destructive"
              className="rounded-full w-16 h-16"
            >
              <PhoneOff className="w-6 h-6" />
            </Button>
            
            <Button
              onClick={() => setShowSettings(!showSettings)}
              size="lg"
              variant="outline"
              className="rounded-full w-14 h-14"
            >
              <Settings className="w-5 h-5" />
            </Button>
          </>
        )}
      </div>
      
      {/* Status */}
      <div className="text-center text-sm">
        {isInCall ? (
          <div className="flex items-center justify-center gap-4">
            {isListening && (
              <span className="flex items-center text-emerald-400">
                <Mic className="w-4 h-4 mr-1 animate-pulse" />
                Listening...
              </span>
            )}
            {isSpeaking && (
              <span className="flex items-center text-blue-400">
                <Volume2 className="w-4 h-4 mr-1 animate-pulse" />
                Speaking...
              </span>
            )}
            {isLoadingAudio && (
              <span className="flex items-center text-zinc-400">
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                Loading audio...
              </span>
            )}
            {isProcessing && (
              <span className="flex items-center text-yellow-400">
                <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                Thinking...
              </span>
            )}
            {!isListening && !isSpeaking && !isLoadingAudio && !isProcessing && (
              <span className="text-zinc-400">In call with {currentVoice.name}</span>
            )}
          </div>
        ) : (
          <span className="text-zinc-500">Tap to start voice call</span>
        )}
      </div>
      
      {/* Settings panel */}
      {showSettings && renderSettings()}
    </div>
  );
}

export default VoiceChat;
