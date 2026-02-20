import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Mic, MicOff, Volume2, Check, AlertTriangle } from 'lucide-react';
import { toast } from 'sonner';

interface VoiceCheckInProps {
  onCheckInComplete: (status: 'okay' | 'not_okay' | 'emergency') => void;
  isOpen: boolean;
  onClose: () => void;
}

// Phrases that indicate user is okay
const OKAY_PHRASES = [
  "i'm okay", "im okay", "i am okay",
  "i'm fine", "im fine", "i am fine",
  "i'm good", "im good", "i am good",
  "i'm alright", "im alright", "i am alright",
  "doing okay", "doing fine", "doing good",
  "all good", "all is well",
  "yes", "yeah", "yep", "yup",
  "okay", "ok", "fine", "good"
];

// Phrases that indicate user needs help
const HELP_PHRASES = [
  "help", "help me", "i need help",
  "not okay", "not ok", "not good", "not fine",
  "i'm not okay", "im not okay", "i am not okay",
  "struggling", "i'm struggling",
  "bad", "i feel bad", "feeling bad",
  "crisis", "emergency", "danger",
  "scared", "afraid", "terrified",
  "hurt", "hurting", "in pain",
  "no", "nope"
];

// Emergency phrases that trigger immediate alert
const EMERGENCY_PHRASES = [
  "call 911", "call police", "call ambulance",
  "emergency", "danger", "unsafe",
  "he's here", "she's here", "they're here",
  "help me now", "come now", "hurry",
  "being hurt", "hitting me", "attacking"
];

export function VoiceCheckIn({ onCheckInComplete, isOpen, onClose }: VoiceCheckInProps) {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState<'listening' | 'processing' | 'confirmed' | 'alert'>('listening');
  const [speechSupported, setSpeechSupported] = useState(true);
  const recognitionRef = useRef<any>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Check for speech recognition support
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
    }
  }, []);

  // Play audio feedback
  const playSound = useCallback((type: 'start' | 'success' | 'alert') => {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    if (type === 'start') {
      oscillator.frequency.value = 440;
      gainNode.gain.value = 0.3;
      oscillator.start();
      setTimeout(() => {
        oscillator.frequency.value = 554;
      }, 100);
      setTimeout(() => {
        oscillator.stop();
        audioContext.close();
      }, 200);
    } else if (type === 'success') {
      oscillator.frequency.value = 523;
      gainNode.gain.value = 0.3;
      oscillator.start();
      setTimeout(() => oscillator.frequency.value = 659, 100);
      setTimeout(() => oscillator.frequency.value = 784, 200);
      setTimeout(() => {
        oscillator.stop();
        audioContext.close();
      }, 400);
    } else if (type === 'alert') {
      oscillator.frequency.value = 880;
      gainNode.gain.value = 0.4;
      oscillator.start();
      setTimeout(() => oscillator.frequency.value = 440, 200);
      setTimeout(() => oscillator.frequency.value = 880, 400);
      setTimeout(() => {
        oscillator.stop();
        audioContext.close();
      }, 600);
    }
  }, []);

  // Analyze transcript for intent
  const analyzeTranscript = useCallback((text: string) => {
    const lowerText = text.toLowerCase().trim();
    
    // Check for emergency phrases first
    for (const phrase of EMERGENCY_PHRASES) {
      if (lowerText.includes(phrase)) {
        return 'emergency';
      }
    }
    
    // Check for help phrases
    for (const phrase of HELP_PHRASES) {
      if (lowerText.includes(phrase)) {
        return 'not_okay';
      }
    }
    
    // Check for okay phrases
    for (const phrase of OKAY_PHRASES) {
      if (lowerText.includes(phrase)) {
        return 'okay';
      }
    }
    
    return null;
  }, []);

  // Start listening
  const startListening = useCallback(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    recognitionRef.current = new SpeechRecognition();
    recognitionRef.current.continuous = true;
    recognitionRef.current.interimResults = true;
    recognitionRef.current.lang = 'en-US';

    recognitionRef.current.onstart = () => {
      setIsListening(true);
      setStatus('listening');
      playSound('start');
    };

    recognitionRef.current.onresult = (event: any) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }

      const fullTranscript = finalTranscript || interimTranscript;
      setTranscript(fullTranscript);

      // Analyze the transcript
      const result = analyzeTranscript(fullTranscript);
      if (result) {
        setStatus('processing');
        recognitionRef.current?.stop();
        
        setTimeout(() => {
          if (result === 'okay') {
            setStatus('confirmed');
            playSound('success');
            toast.success('Check-in confirmed. Stay safe! 💚');
          } else {
            setStatus('alert');
            playSound('alert');
            if (result === 'emergency') {
              toast.error('Emergency detected. Alerting contacts...');
            } else {
              toast.warning('We hear you. Showing resources...');
            }
          }
          
          setTimeout(() => {
            onCheckInComplete(result);
            onClose();
          }, 2000);
        }, 500);
      }
    };

    recognitionRef.current.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
      if (event.error === 'not-allowed') {
        toast.error('Microphone access denied. Please enable it in settings.');
      }
    };

    recognitionRef.current.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current.start();

    // Auto-timeout after 30 seconds
    timeoutRef.current = setTimeout(() => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
        toast.info('Voice check-in timed out. Please try again or use manual check-in.');
        onClose();
      }
    }, 30000);
  }, [analyzeTranscript, onCheckInComplete, onClose, playSound]);

  // Stop listening
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsListening(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);

  // Auto-start when opened
  useEffect(() => {
    if (isOpen && speechSupported) {
      startListening();
    }
    return () => {
      stopListening();
    };
  }, [isOpen, speechSupported, startListening, stopListening]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-md bg-slate-900 border-slate-700">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl text-white flex items-center justify-center gap-2">
            <Volume2 className="w-6 h-6 text-emerald-400" />
            Voice Check-In
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {!speechSupported ? (
            <div className="text-center space-y-4">
              <AlertTriangle className="w-16 h-16 text-yellow-500 mx-auto" />
              <p className="text-slate-300">
                Voice recognition is not supported in your browser.
                Please use manual check-in instead.
              </p>
              <Button onClick={onClose} variant="outline">
                Close
              </Button>
            </div>
          ) : (
            <>
              {/* Microphone visualization */}
              <div className="flex justify-center">
                <div className={`relative p-8 rounded-full ${
                  status === 'listening' ? 'bg-emerald-500/20 animate-pulse' :
                  status === 'confirmed' ? 'bg-green-500/20' :
                  status === 'alert' ? 'bg-red-500/20' :
                  'bg-slate-700/50'
                }`}>
                  {status === 'confirmed' ? (
                    <Check className="w-16 h-16 text-green-400" />
                  ) : status === 'alert' ? (
                    <AlertTriangle className="w-16 h-16 text-red-400" />
                  ) : isListening ? (
                    <Mic className="w-16 h-16 text-emerald-400" />
                  ) : (
                    <MicOff className="w-16 h-16 text-slate-400" />
                  )}
                  
                  {/* Pulse rings when listening */}
                  {isListening && (
                    <>
                      <div className="absolute inset-0 rounded-full border-2 border-emerald-400/50 animate-ping" />
                      <div className="absolute inset-0 rounded-full border border-emerald-400/30 animate-pulse" style={{ animationDelay: '0.5s' }} />
                    </>
                  )}
                </div>
              </div>

              {/* Status text */}
              <div className="text-center space-y-2">
                {status === 'listening' && (
                  <>
                    <p className="text-lg text-white">Listening...</p>
                    <p className="text-sm text-slate-400">
                      Say "I'm okay" or "I need help"
                    </p>
                  </>
                )}
                {status === 'processing' && (
                  <p className="text-lg text-yellow-400">Processing...</p>
                )}
                {status === 'confirmed' && (
                  <p className="text-lg text-green-400">Check-in confirmed! ✓</p>
                )}
                {status === 'alert' && (
                  <p className="text-lg text-red-400">Alerting your contacts...</p>
                )}
              </div>

              {/* Transcript display */}
              {transcript && (
                <div className="bg-slate-800 rounded-lg p-3">
                  <p className="text-xs text-slate-500 mb-1">Heard:</p>
                  <p className="text-slate-300 italic">"{transcript}"</p>
                </div>
              )}

              {/* Example phrases */}
              <div className="bg-slate-800/50 rounded-lg p-4 space-y-2">
                <p className="text-xs text-slate-500 font-medium">Example phrases:</p>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="text-emerald-400">"I'm okay"</div>
                  <div className="text-emerald-400">"I'm fine"</div>
                  <div className="text-yellow-400">"I need help"</div>
                  <div className="text-yellow-400">"Not okay"</div>
                </div>
              </div>

              {/* Controls */}
              <div className="flex gap-3">
                <Button
                  onClick={isListening ? stopListening : startListening}
                  className={`flex-1 ${isListening ? 'bg-red-600 hover:bg-red-700' : 'bg-emerald-600 hover:bg-emerald-700'}`}
                >
                  {isListening ? (
                    <>
                      <MicOff className="w-4 h-4 mr-2" />
                      Stop
                    </>
                  ) : (
                    <>
                      <Mic className="w-4 h-4 mr-2" />
                      Start
                    </>
                  )}
                </Button>
                <Button onClick={onClose} variant="outline" className="flex-1">
                  Cancel
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default VoiceCheckIn;
