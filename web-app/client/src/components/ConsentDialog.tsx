import { useState, useEffect } from "react";

import { Button } from "@/components/ui/button";

import { AlertTriangle, Shield, FileText, Phone, Scale } from "lucide-react";

const CONSENT_KEY = "reunity_consent_accepted";
const CONSENT_VERSION = "3.0.0"; // Compliance audit v3 - updated disclaimer text

// Required exact disclaimer text per compliance directive
const GLOBAL_DISCLAIMER = "ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.";

export function ConsentDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [disclaimerAccepted, setDisclaimerAccepted] = useState(false);
  const [ageConfirmed, setAgeConfirmed] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem(CONSENT_KEY);
    if (!consent || JSON.parse(consent).version !== CONSENT_VERSION) {
      setIsOpen(true);
    }
  }, []);

  const handleAccept = () => {
    if (termsAccepted && disclaimerAccepted && ageConfirmed) {
      localStorage.setItem(CONSENT_KEY, JSON.stringify({
        version: CONSENT_VERSION,
        acceptedAt: new Date().toISOString(),
        termsAccepted: true,
        disclaimerAccepted: true,
        ageConfirmed: true,
      }));
      setIsOpen(false);
    }
  };

  const canAccept = termsAccepted && disclaimerAccepted && ageConfirmed;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0f0f12] border border-white/10 rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-[#0f0f12] border-b border-white/10 p-6">
          <div className="flex items-center gap-3 mb-2">
            <Shield className="w-8 h-8 text-emerald-400" />
            <h2 className="text-2xl font-bold text-white">Welcome to ReUnity</h2>
          </div>
          <p className="text-white/60">Please review and accept the following before continuing</p>
        </div>

        <div className="p-6 space-y-6">
          {/* GLOBAL DISCLAIMER - Required exact text, prominently displayed */}
          <div className="bg-blue-500/10 border-2 border-blue-500/40 rounded-xl p-5">
            <div className="flex items-start gap-3">
              <Scale className="w-6 h-6 text-blue-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-bold text-blue-400 mb-3 text-lg">Important Disclaimer</h3>
                <p className="text-base text-white/90 leading-relaxed font-medium">
                  {GLOBAL_DISCLAIMER}
                </p>
              </div>
            </div>
          </div>

          {/* Crisis Notice */}
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-400 mb-2">If You Are in Crisis</h3>
                <p className="text-sm text-white/80 mb-3">
                  If you are experiencing a mental health emergency or having thoughts of suicide, please contact emergency services immediately:
                </p>
                <div className="space-y-2 text-sm">
                  <a href="tel:988" className="flex items-center gap-2 text-red-400 hover:text-red-300">
                    <Phone className="w-4 h-4" />
                    <span><strong>988</strong> - Suicide & Crisis Lifeline (24/7)</span>
                  </a>
                  <a href="tel:911" className="flex items-center gap-2 text-red-400 hover:text-red-300">
                    <Phone className="w-4 h-4" />
                    <span><strong>911</strong> - Emergency Services</span>
                  </a>
                  <a href="https://www.crisistextline.org/" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 text-red-400 hover:text-red-300">
                    <Phone className="w-4 h-4" />
                    <span><strong>Text HOME to 741741</strong> - Crisis Text Line</span>
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Important Information */}
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4">
            <h3 className="font-semibold text-amber-400 mb-3 flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Important Information
            </h3>
            <ul className="space-y-2 text-sm text-white/80">
              <li>• ReUnity is a <strong>wellness and support tool</strong>, not a medical device</li>
              <li>• The AI companion is <strong>NOT</strong> a licensed therapist, counselor, or healthcare provider</li>
              <li>• Nothing in this app constitutes medical advice, diagnosis, or treatment</li>
              <li>• ReUnity does <strong>NOT</strong> provide crisis services — call 911 for emergencies</li>
              <li>• Your conversation data is <strong>NOT stored</strong> after sessions end</li>
              <li>• We do <strong>NOT sell</strong> your personal information to third parties</li>
            </ul>
          </div>

          {/* Data Privacy Summary */}
          <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
            <h3 className="font-semibold text-emerald-400 mb-3 flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Your Privacy
            </h3>
            <ul className="space-y-2 text-sm text-white/80">
              <li>• Conversations are processed in real-time and <strong>not permanently stored</strong></li>
              <li>• Journal entries and safety plans are <strong>encrypted and stored locally</strong> on your device</li>
              <li>• We collect only your email for authentication purposes</li>
              <li>• We do <strong>not track your location</strong> or share data with advertisers</li>
              <li>• No silent analytics tracking — you control your data</li>
            </ul>
          </div>

          {/* Consent Checkboxes */}
          <div className="space-y-4 pt-4 border-t border-white/10">
            <div 
              className="flex items-start gap-3 cursor-pointer hover:bg-white/5 p-2 rounded-lg -mx-2"
              onClick={(e) => {
                // Only toggle if not clicking a link
                if ((e.target as HTMLElement).tagName !== 'A') {
                  setTermsAccepted(!termsAccepted);
                }
              }}
            >
              <div 
                className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors cursor-pointer ${
                  termsAccepted ? 'bg-emerald-500 border-emerald-500' : 'border-white/40 bg-transparent hover:border-white/60'
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  setTermsAccepted(!termsAccepted);
                }}
              >
                {termsAccepted && <span className="text-white text-sm">✓</span>}
              </div>
              <span className="text-sm text-white/80">
                I have read and agree to the{" "}
                <a href="/terms" target="_blank" className="text-emerald-400 hover:underline" onClick={(e) => e.stopPropagation()}>Terms of Service</a>
                {" "}and{" "}
                <a href="/privacy" target="_blank" className="text-emerald-400 hover:underline" onClick={(e) => e.stopPropagation()}>Privacy Policy</a>
              </span>
            </div>

            <div 
              className="flex items-start gap-3 cursor-pointer hover:bg-white/5 p-2 rounded-lg -mx-2"
              onClick={(e) => {
                if ((e.target as HTMLElement).tagName !== 'A') {
                  setDisclaimerAccepted(!disclaimerAccepted);
                }
              }}
            >
              <div 
                className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors cursor-pointer ${
                  disclaimerAccepted ? 'bg-emerald-500 border-emerald-500' : 'border-white/40 bg-transparent hover:border-white/60'
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  setDisclaimerAccepted(!disclaimerAccepted);
                }}
              >
                {disclaimerAccepted && <span className="text-white text-sm">✓</span>}
              </div>
              <span className="text-sm text-white/80">
                I understand that ReUnity is a <strong>wellness and support tool, not a medical device</strong>, and does not provide diagnosis, treatment, or crisis services. I have read the{" "}
                <a href="/disclaimer" target="_blank" className="text-emerald-400 hover:underline" onClick={(e) => e.stopPropagation()}>Disclaimer</a>
              </span>
            </div>

            <div 
              className="flex items-start gap-3 cursor-pointer hover:bg-white/5 p-2 rounded-lg -mx-2"
              onClick={() => setAgeConfirmed(!ageConfirmed)}
            >
              <div 
                className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors cursor-pointer ${
                  ageConfirmed ? 'bg-emerald-500 border-emerald-500' : 'border-white/40 bg-transparent hover:border-white/60'
                }`}
                onClick={(e) => {
                  e.stopPropagation();
                  setAgeConfirmed(!ageConfirmed);
                }}
              >
                {ageConfirmed && <span className="text-white text-sm">✓</span>}
              </div>
              <span className="text-sm text-white/80">
                I confirm that I am <strong>17 years of age or older</strong>
              </span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-[#0f0f12] border-t border-white/10 p-6">
          <Button 
            onClick={handleAccept}
            disabled={!canAccept}
            className={`w-full py-6 text-lg font-semibold rounded-xl transition-all ${
              canAccept 
                ? "bg-emerald-600 hover:bg-emerald-500 text-white" 
                : "bg-gray-700 text-gray-400 cursor-not-allowed"
            }`}
          >
            {canAccept ? "I Understand and Accept" : "Please accept all terms to continue"}
          </Button>
          <p className="text-center text-xs text-white/40 mt-4">
            By continuing, you acknowledge that you have read and understood all terms.
          </p>
        </div>
      </div>
    </div>
  );
}

export default ConsentDialog;
