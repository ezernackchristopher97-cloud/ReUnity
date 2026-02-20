import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Fingerprint, Eye, Lock, Shield, AlertTriangle } from "lucide-react";

interface BiometricLockProps {
  onUnlock: () => void;
  title?: string;
  description?: string;
}

export function BiometricLock({ onUnlock, title = "Protected Content", description = "This content is protected for your safety" }: BiometricLockProps) {
  const [isSupported, setIsSupported] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPinFallback, setShowPinFallback] = useState(false);
  const [pin, setPin] = useState("");
  const [storedPinHash, setStoredPinHash] = useState<string | null>(null);
  const [isSettingPin, setIsSettingPin] = useState(false);
  const [confirmPin, setConfirmPin] = useState("");

  // Check if WebAuthn is supported
  useEffect(() => {
    const checkSupport = async () => {
      if (window.PublicKeyCredential) {
        try {
          const available = await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable();
          setIsSupported(available);
        } catch {
          setIsSupported(false);
        }
      }
    };
    checkSupport();

    // Check for stored PIN
    const hash = localStorage.getItem("reunity_safety_pin_hash");
    setStoredPinHash(hash);
  }, []);

  // Simple hash function for PIN (not cryptographically secure, but adequate for local use)
  const hashPin = (pin: string): string => {
    let hash = 0;
    for (let i = 0; i < pin.length; i++) {
      const char = pin.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  };

  // Biometric authentication using WebAuthn
  const authenticateWithBiometric = async () => {
    setIsAuthenticating(true);
    setError(null);

    try {
      // Check if we have a stored credential
      const credentialId = localStorage.getItem("reunity_biometric_credential");
      
      if (!credentialId) {
        // First time - create a new credential
        const challenge = new Uint8Array(32);
        crypto.getRandomValues(challenge);

        const createOptions: CredentialCreationOptions = {
          publicKey: {
            challenge,
            rp: {
              name: "ReUnity Safety Plan",
              id: window.location.hostname,
            },
            user: {
              id: new Uint8Array(16),
              name: "safety-plan-user",
              displayName: "Safety Plan User",
            },
            pubKeyCredParams: [
              { type: "public-key", alg: -7 }, // ES256
              { type: "public-key", alg: -257 }, // RS256
            ],
            authenticatorSelection: {
              authenticatorAttachment: "platform",
              userVerification: "required",
            },
            timeout: 60000,
          },
        };

        const credential = await navigator.credentials.create(createOptions) as PublicKeyCredential;
        
        // Store the credential ID
        const credId = btoa(String.fromCharCode.apply(null, Array.from(new Uint8Array(credential.rawId))));
        localStorage.setItem("reunity_biometric_credential", credId);
        
        onUnlock();
      } else {
        // Verify existing credential
        const challenge = new Uint8Array(32);
        crypto.getRandomValues(challenge);

        const getOptions: CredentialRequestOptions = {
          publicKey: {
            challenge,
            rpId: window.location.hostname,
            allowCredentials: [{
              type: "public-key",
              id: Uint8Array.from(atob(credentialId), c => c.charCodeAt(0)),
            }],
            userVerification: "required",
            timeout: 60000,
          },
        };

        await navigator.credentials.get(getOptions);
        onUnlock();
      }
    } catch (err) {
      console.error("Biometric auth error:", err);
      setError("Biometric authentication failed. Please try again or use PIN.");
      setShowPinFallback(true);
    } finally {
      setIsAuthenticating(false);
    }
  };

  // PIN authentication
  const authenticateWithPin = () => {
    if (!storedPinHash) {
      // First time - set up PIN
      if (!isSettingPin) {
        setIsSettingPin(true);
        return;
      }
      
      if (pin.length < 4) {
        setError("PIN must be at least 4 digits");
        return;
      }

      if (pin !== confirmPin) {
        setError("PINs do not match");
        return;
      }

      // Store the PIN hash
      const hash = hashPin(pin);
      localStorage.setItem("reunity_safety_pin_hash", hash);
      setStoredPinHash(hash);
      onUnlock();
    } else {
      // Verify PIN
      if (hashPin(pin) === storedPinHash) {
        onUnlock();
      } else {
        setError("Incorrect PIN");
        setPin("");
      }
    }
  };

  return (
    <div className="fixed inset-0 z-[100] bg-black/95 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0f0f12] border border-white/10 rounded-2xl max-w-md w-full p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <Shield className="w-8 h-8 text-emerald-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">{title}</h2>
          <p className="text-white/60">{description}</p>
        </div>

        {/* Security Notice */}
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 mb-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-white/80">
              Your safety plan contains sensitive information. Authentication helps protect it from unauthorized access.
            </p>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 mb-6">
            <p className="text-sm text-red-400 text-center">{error}</p>
          </div>
        )}

        {/* Biometric Option */}
        {isSupported && !showPinFallback && (
          <div className="space-y-4">
            <Button
              onClick={authenticateWithBiometric}
              disabled={isAuthenticating}
              className="w-full py-6 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl flex items-center justify-center gap-3"
            >
              {isAuthenticating ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Authenticating...
                </>
              ) : (
                <>
                  <Fingerprint className="w-6 h-6" />
                  Use Face ID / Fingerprint
                </>
              )}
            </Button>

            <div className="flex items-center gap-4">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-white/40 text-sm">or</span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

            <Button
              onClick={() => setShowPinFallback(true)}
              variant="outline"
              className="w-full py-6 border-white/20 text-white hover:bg-white/5 rounded-xl flex items-center justify-center gap-3"
            >
              <Lock className="w-5 h-5" />
              Use PIN Instead
            </Button>
          </div>
        )}

        {/* PIN Fallback */}
        {(!isSupported || showPinFallback) && (
          <div className="space-y-4">
            {!isSupported && !showPinFallback && (
              <p className="text-white/60 text-sm text-center mb-4">
                Biometric authentication is not available on this device. Please use a PIN.
              </p>
            )}

            {isSettingPin && !storedPinHash ? (
              <>
                <div>
                  <label className="text-white/60 text-sm mb-2 block">Create a PIN (at least 4 digits)</label>
                  <Input
                    type="password"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={pin}
                    onChange={(e) => setPin(e.target.value.replace(/\D/g, ""))}
                    placeholder="Enter PIN"
                    className="bg-white/5 border-white/20 text-white text-center text-2xl tracking-widest"
                    maxLength={8}
                  />
                </div>
                <div>
                  <label className="text-white/60 text-sm mb-2 block">Confirm PIN</label>
                  <Input
                    type="password"
                    inputMode="numeric"
                    pattern="[0-9]*"
                    value={confirmPin}
                    onChange={(e) => setConfirmPin(e.target.value.replace(/\D/g, ""))}
                    placeholder="Confirm PIN"
                    className="bg-white/5 border-white/20 text-white text-center text-2xl tracking-widest"
                    maxLength={8}
                  />
                </div>
              </>
            ) : (
              <div>
                <label className="text-white/60 text-sm mb-2 block">
                  {storedPinHash ? "Enter your PIN" : "Create a PIN to protect your safety plan"}
                </label>
                <Input
                  type="password"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  value={pin}
                  onChange={(e) => setPin(e.target.value.replace(/\D/g, ""))}
                  placeholder="Enter PIN"
                  className="bg-white/5 border-white/20 text-white text-center text-2xl tracking-widest"
                  maxLength={8}
                  onKeyDown={(e) => e.key === "Enter" && authenticateWithPin()}
                />
              </div>
            )}

            <Button
              onClick={authenticateWithPin}
              className="w-full py-6 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl"
            >
              {storedPinHash ? "Unlock" : isSettingPin ? "Set PIN & Continue" : "Continue"}
            </Button>

            {isSupported && showPinFallback && (
              <Button
                onClick={() => {
                  setShowPinFallback(false);
                  setError(null);
                }}
                variant="ghost"
                className="w-full text-white/60 hover:text-white"
              >
                <Fingerprint className="w-4 h-4 mr-2" />
                Try biometric again
              </Button>
            )}
          </div>
        )}

        {/* Emergency Access */}
        <div className="mt-8 pt-6 border-t border-white/10">
          <p className="text-xs text-white/40 text-center">
            If you're in immediate danger and cannot authenticate, call <strong>911</strong> or the National DV Hotline: <strong>1-800-799-7233</strong>
          </p>
        </div>
      </div>
    </div>
  );
}

export default BiometricLock;
