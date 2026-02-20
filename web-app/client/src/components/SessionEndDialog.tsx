import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Shield, Heart, Trash2, CheckCircle, AlertTriangle, Phone } from "lucide-react";

interface SessionEndDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirmEnd: () => void;
  messageCount: number;
}

type Stage = "stability" | "confirm" | "wiping" | "complete";

export function SessionEndDialog({ isOpen, onClose, onConfirmEnd, messageCount }: SessionEndDialogProps) {
  const [stage, setStage] = useState<Stage>("stability");
  const [wipeProgress, setWipeProgress] = useState(0);
  const [isStable, setIsStable] = useState<boolean | null>(null);

  const handleStabilityResponse = (stable: boolean) => {
    setIsStable(stable);
    if (stable) {
      setStage("confirm");
    }
  };

  const handleWipeData = async () => {
    setStage("wiping");
    
    // Simulate data wipe with progress
    const totalSteps = 100;
    const stepDelay = 30; // 3 seconds total
    
    for (let i = 0; i <= totalSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, stepDelay));
      setWipeProgress(i);
      
      // Clear actual data at specific points
      if (i === 25) {
        // Clear session storage
        sessionStorage.removeItem("reunity_chat_messages");
        sessionStorage.removeItem("reunity_session_id");
      }
      if (i === 50) {
        // Clear any cached data
        localStorage.removeItem("reunity_temp_messages");
      }
      if (i === 75) {
        // Clear conversation state
        localStorage.removeItem("reunity_conversation_state");
      }
      if (i === 100) {
        // Final cleanup
        localStorage.removeItem("reunity_last_session");
      }
    }
    
    setStage("complete");
  };

  const handleKeepData = () => {
    onConfirmEnd();
    onClose();
  };

  const handleComplete = () => {
    onConfirmEnd();
    onClose();
    // Reset for next time
    setStage("stability");
    setWipeProgress(0);
    setIsStable(null);
  };

  const resetAndClose = () => {
    setStage("stability");
    setWipeProgress(0);
    setIsStable(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0f0f12] border border-white/10 rounded-2xl max-w-lg w-full overflow-hidden">
        
        {/* Stability Check Stage */}
        {stage === "stability" && (
          <>
            <div className="p-6 border-b border-white/10">
              <div className="flex items-center gap-3 mb-2">
                <Heart className="w-8 h-8 text-emerald-400" />
                <h2 className="text-2xl font-bold text-white">Before You Go</h2>
              </div>
              <p className="text-white/60">We want to make sure you're okay</p>
            </div>

            <div className="p-6 space-y-6">
              <div className="text-center">
                <p className="text-lg text-white mb-6">
                  Are you feeling stable and ready to end this session?
                </p>
                
                <div className="flex gap-4 justify-center">
                  <Button
                    onClick={() => handleStabilityResponse(true)}
                    className="bg-emerald-600 hover:bg-emerald-500 text-white px-8 py-6 text-lg rounded-xl"
                  >
                    <CheckCircle className="w-5 h-5 mr-2" />
                    Yes, I'm stable
                  </Button>
                  <Button
                    onClick={() => handleStabilityResponse(false)}
                    variant="outline"
                    className="border-amber-500/50 text-amber-400 hover:bg-amber-500/10 px-8 py-6 text-lg rounded-xl"
                  >
                    Not quite yet
                  </Button>
                </div>
              </div>

              {isStable === false && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mt-6">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-6 h-6 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="font-semibold text-amber-400 mb-2">That's okay. Take your time.</h3>
                      <p className="text-sm text-white/80 mb-4">
                        There's no rush. You can continue chatting, or if you need immediate support:
                      </p>
                      <div className="space-y-2">
                        <a href="tel:988" className="flex items-center gap-2 text-amber-400 hover:text-amber-300">
                          <Phone className="w-4 h-4" />
                          <span><strong>988</strong> - Suicide & Crisis Lifeline</span>
                        </a>
                      </div>
                      <Button
                        onClick={resetAndClose}
                        className="mt-4 bg-amber-600 hover:bg-amber-500 text-white"
                      >
                        Continue Session
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Confirm Data Wipe Stage */}
        {stage === "confirm" && (
          <>
            <div className="p-6 border-b border-white/10">
              <div className="flex items-center gap-3 mb-2">
                <Shield className="w-8 h-8 text-emerald-400" />
                <h2 className="text-2xl font-bold text-white">Your Privacy</h2>
              </div>
              <p className="text-white/60">Would you like to delete your chat data?</p>
            </div>

            <div className="p-6 space-y-6">
              <div className="bg-gray-800/50 rounded-xl p-4">
                <p className="text-white/80 mb-2">This session contained:</p>
                <p className="text-2xl font-bold text-emerald-400">{messageCount} messages</p>
              </div>

              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
                <p className="text-sm text-white/80">
                  <strong className="text-emerald-400">Your conversations are already confidential.</strong> We don't store chat data on our servers. However, you can wipe any local session data for extra peace of mind.
                </p>
              </div>

              <div className="flex gap-4">
                <Button
                  onClick={handleWipeData}
                  className="flex-1 bg-red-600 hover:bg-red-500 text-white py-6 text-lg rounded-xl"
                >
                  <Trash2 className="w-5 h-5 mr-2" />
                  Wipe All Data
                </Button>
                <Button
                  onClick={handleKeepData}
                  variant="outline"
                  className="flex-1 border-white/20 text-white hover:bg-white/10 py-6 text-lg rounded-xl"
                >
                  Just End Session
                </Button>
              </div>
            </div>
          </>
        )}

        {/* Wiping Progress Stage */}
        {stage === "wiping" && (
          <>
            <div className="p-6 border-b border-white/10">
              <div className="flex items-center gap-3 mb-2">
                <Trash2 className="w-8 h-8 text-red-400 animate-pulse" />
                <h2 className="text-2xl font-bold text-white">Wiping Data</h2>
              </div>
              <p className="text-white/60">Securely erasing your session data...</p>
            </div>

            <div className="p-8 space-y-6">
              {/* Progress Circle */}
              <div className="flex justify-center">
                <div className="relative w-40 h-40">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-gray-800"
                    />
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={440}
                      strokeDashoffset={440 - (440 * wipeProgress) / 100}
                      className="text-red-500 transition-all duration-100"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-4xl font-bold text-white">{wipeProgress}%</span>
                  </div>
                </div>
              </div>

              {/* Progress Steps */}
              <div className="space-y-2 text-sm">
                <div className={`flex items-center gap-2 ${wipeProgress >= 25 ? 'text-emerald-400' : 'text-white/40'}`}>
                  <CheckCircle className={`w-4 h-4 ${wipeProgress >= 25 ? 'opacity-100' : 'opacity-30'}`} />
                  <span>Clearing session messages...</span>
                </div>
                <div className={`flex items-center gap-2 ${wipeProgress >= 50 ? 'text-emerald-400' : 'text-white/40'}`}>
                  <CheckCircle className={`w-4 h-4 ${wipeProgress >= 50 ? 'opacity-100' : 'opacity-30'}`} />
                  <span>Removing cached data...</span>
                </div>
                <div className={`flex items-center gap-2 ${wipeProgress >= 75 ? 'text-emerald-400' : 'text-white/40'}`}>
                  <CheckCircle className={`w-4 h-4 ${wipeProgress >= 75 ? 'opacity-100' : 'opacity-30'}`} />
                  <span>Clearing conversation state...</span>
                </div>
                <div className={`flex items-center gap-2 ${wipeProgress >= 100 ? 'text-emerald-400' : 'text-white/40'}`}>
                  <CheckCircle className={`w-4 h-4 ${wipeProgress >= 100 ? 'opacity-100' : 'opacity-30'}`} />
                  <span>Final cleanup complete</span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Complete Stage */}
        {stage === "complete" && (
          <>
            <div className="p-6 border-b border-white/10">
              <div className="flex items-center gap-3 mb-2">
                <CheckCircle className="w-8 h-8 text-emerald-400" />
                <h2 className="text-2xl font-bold text-white">Data Wiped</h2>
              </div>
              <p className="text-white/60">Your session data has been securely erased</p>
            </div>

            <div className="p-6 space-y-6">
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-6 text-center">
                <CheckCircle className="w-16 h-16 text-emerald-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">100% Complete</h3>
                <p className="text-white/60">
                  All local session data has been permanently deleted. Your privacy is protected.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-xl p-4">
                <p className="text-sm text-white/60 text-center">
                  Remember: ReUnity is here whenever you need support. Take care of yourself. 💚
                </p>
              </div>

              <Button
                onClick={handleComplete}
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white py-6 text-lg rounded-xl"
              >
                Close Session
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default SessionEndDialog;
