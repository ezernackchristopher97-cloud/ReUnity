import { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { 
  Fingerprint, 
  Scan, 
  Shield, 
  ShieldCheck, 
  ShieldAlert,
  Lock,
  Unlock,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react';

interface BiometricAuthProps {
  onAuthenticated?: () => void;
  onAuthFailed?: (error: string) => void;
  featureName?: string;
  requireAuth?: boolean;
  children?: React.ReactNode;
}

interface BiometricSettings {
  enabled: boolean;
  protectedFeatures: {
    safetyPlan: boolean;
    videoCalls: boolean;
    settings: boolean;
    exportData: boolean;
    trustedDevices: boolean;
  };
  lastAuthenticated: string | null;
  authTimeout: number; // minutes before re-auth required
}

const defaultSettings: BiometricSettings = {
  enabled: false,
  protectedFeatures: {
    safetyPlan: true,
    videoCalls: true,
    settings: false,
    exportData: true,
    trustedDevices: true,
  },
  lastAuthenticated: null,
  authTimeout: 15,
};

// Check if WebAuthn is supported
const isWebAuthnSupported = () => {
  return window.PublicKeyCredential !== undefined;
};

// Check if platform authenticator (Face ID/Touch ID) is available
const isPlatformAuthenticatorAvailable = async () => {
  if (!isWebAuthnSupported()) return false;
  try {
    return await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable();
  } catch {
    return false;
  }
};

export function useBiometricAuth() {
  const [settings, setSettings] = useState<BiometricSettings>(() => {
    const stored = localStorage.getItem('reunity_biometric_settings');
    return stored ? JSON.parse(stored) : defaultSettings;
  });
  const [isAvailable, setIsAvailable] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    isPlatformAuthenticatorAvailable().then(setIsAvailable);
  }, []);

  useEffect(() => {
    localStorage.setItem('reunity_biometric_settings', JSON.stringify(settings));
  }, [settings]);

  // Check if auth is still valid based on timeout
  useEffect(() => {
    if (settings.lastAuthenticated) {
      const lastAuth = new Date(settings.lastAuthenticated);
      const now = new Date();
      const diffMinutes = (now.getTime() - lastAuth.getTime()) / (1000 * 60);
      setIsAuthenticated(diffMinutes < settings.authTimeout);
    }
  }, [settings.lastAuthenticated, settings.authTimeout]);

  const updateSettings = useCallback((updates: Partial<BiometricSettings>) => {
    setSettings(prev => ({ ...prev, ...updates }));
  }, []);

  const updateProtectedFeature = useCallback((feature: keyof BiometricSettings['protectedFeatures'], value: boolean) => {
    setSettings(prev => ({
      ...prev,
      protectedFeatures: {
        ...prev.protectedFeatures,
        [feature]: value,
      },
    }));
  }, []);

  const authenticate = useCallback(async (): Promise<boolean> => {
    if (!isAvailable || !settings.enabled) {
      setIsAuthenticated(true);
      return true;
    }

    try {
      // Create a challenge for WebAuthn
      const challenge = new Uint8Array(32);
      crypto.getRandomValues(challenge);

      const credential = await navigator.credentials.create({
        publicKey: {
          challenge,
          rp: {
            name: 'ReUnity',
            id: window.location.hostname,
          },
          user: {
            id: new Uint8Array(16),
            name: 'reunity-user',
            displayName: 'ReUnity User',
          },
          pubKeyCredParams: [
            { type: 'public-key', alg: -7 }, // ES256
            { type: 'public-key', alg: -257 }, // RS256
          ],
          authenticatorSelection: {
            authenticatorAttachment: 'platform',
            userVerification: 'required',
          },
          timeout: 60000,
        },
      });

      if (credential) {
        setIsAuthenticated(true);
        setSettings(prev => ({
          ...prev,
          lastAuthenticated: new Date().toISOString(),
        }));
        return true;
      }
      return false;
    } catch (error) {
      console.error('Biometric auth failed:', error);
      return false;
    }
  }, [isAvailable, settings.enabled]);

  const requiresAuth = useCallback((feature: keyof BiometricSettings['protectedFeatures']): boolean => {
    if (!settings.enabled) return false;
    if (!settings.protectedFeatures[feature]) return false;
    if (isAuthenticated) return false;
    return true;
  }, [settings.enabled, settings.protectedFeatures, isAuthenticated]);

  return {
    settings,
    isAvailable,
    isAuthenticated,
    updateSettings,
    updateProtectedFeature,
    authenticate,
    requiresAuth,
  };
}

export default function BiometricAuth({
  onAuthenticated,
  onAuthFailed,
  featureName = 'this feature',
  requireAuth = true,
  children,
}: BiometricAuthProps) {
  const { settings, isAvailable, isAuthenticated, authenticate } = useBiometricAuth();
  const [showAuthDialog, setShowAuthDialog] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  useEffect(() => {
    if (requireAuth && settings.enabled && !isAuthenticated) {
      setShowAuthDialog(true);
    }
  }, [requireAuth, settings.enabled, isAuthenticated]);

  const handleAuthenticate = async () => {
    setIsAuthenticating(true);
    setAuthError(null);

    try {
      const success = await authenticate();
      if (success) {
        setShowAuthDialog(false);
        onAuthenticated?.();
      } else {
        setAuthError('Authentication failed. Please try again.');
        onAuthFailed?.('Authentication failed');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setAuthError(message);
      onAuthFailed?.(message);
    } finally {
      setIsAuthenticating(false);
    }
  };

  if (!requireAuth || !settings.enabled || isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <>
      {/* Auth Required Overlay */}
      <div className="fixed inset-0 bg-zinc-950/95 backdrop-blur-sm z-50 flex items-center justify-center">
        <div className="text-center space-y-6 max-w-md px-6">
          <div className="w-20 h-20 mx-auto bg-emerald-500/20 rounded-full flex items-center justify-center">
            <Lock className="w-10 h-10 text-emerald-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Authentication Required</h2>
            <p className="text-zinc-400">
              {featureName} is protected. Please authenticate to continue.
            </p>
          </div>
          <Button
            onClick={handleAuthenticate}
            disabled={isAuthenticating}
            className="bg-emerald-600 hover:bg-emerald-700 gap-2"
          >
            {isAuthenticating ? (
              <>
                <Scan className="w-5 h-5 animate-pulse" />
                Authenticating...
              </>
            ) : (
              <>
                <Fingerprint className="w-5 h-5" />
                Authenticate with Biometrics
              </>
            )}
          </Button>
          {authError && (
            <p className="text-red-400 text-sm">{authError}</p>
          )}
        </div>
      </div>

      {/* Auth Dialog */}
      <Dialog open={showAuthDialog} onOpenChange={setShowAuthDialog}>
        <DialogContent className="bg-zinc-900 border-zinc-800">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Shield className="w-5 h-5 text-emerald-400" />
              Biometric Authentication
            </DialogTitle>
            <DialogDescription className="text-zinc-400">
              Use Face ID or Touch ID to access {featureName}.
            </DialogDescription>
          </DialogHeader>

          <div className="py-6 flex flex-col items-center gap-4">
            <div className="w-24 h-24 bg-zinc-800 rounded-full flex items-center justify-center">
              <Fingerprint className="w-12 h-12 text-emerald-400" />
            </div>
            {authError && (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <AlertTriangle className="w-4 h-4" />
                {authError}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowAuthDialog(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleAuthenticate}
              disabled={isAuthenticating || !isAvailable}
              className="bg-emerald-600 hover:bg-emerald-700"
            >
              {isAuthenticating ? 'Authenticating...' : 'Authenticate'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

// Settings component for managing biometric protection
export function BiometricSettings() {
  const { 
    settings, 
    isAvailable, 
    updateSettings, 
    updateProtectedFeature,
    authenticate 
  } = useBiometricAuth();
  const [isEnabling, setIsEnabling] = useState(false);

  const handleToggleEnabled = async (enabled: boolean) => {
    if (enabled && isAvailable) {
      setIsEnabling(true);
      const success = await authenticate();
      setIsEnabling(false);
      if (success) {
        updateSettings({ enabled: true });
      }
    } else {
      updateSettings({ enabled: false });
    }
  };

  const features = [
    { key: 'safetyPlan' as const, label: 'Safety Plan', description: 'Protect your personalized safety plan' },
    { key: 'videoCalls' as const, label: 'Video Calls', description: 'Require auth before joining video sessions' },
    { key: 'settings' as const, label: 'Settings', description: 'Protect app settings and preferences' },
    { key: 'exportData' as const, label: 'Export Data', description: 'Require auth to export conversation history' },
    { key: 'trustedDevices' as const, label: 'Trusted Devices', description: 'Protect device pairing settings' },
  ];

  return (
    <div className="space-y-6">
      {/* Availability Status */}
      <div className={`p-4 rounded-lg border ${
        isAvailable 
          ? 'bg-emerald-500/10 border-emerald-500/30' 
          : 'bg-amber-500/10 border-amber-500/30'
      }`}>
        <div className="flex items-center gap-3">
          {isAvailable ? (
            <>
              <ShieldCheck className="w-6 h-6 text-emerald-400" />
              <div>
                <p className="font-medium text-emerald-300">Biometric Authentication Available</p>
                <p className="text-sm text-emerald-400/70">
                  Face ID or Touch ID is available on this device
                </p>
              </div>
            </>
          ) : (
            <>
              <ShieldAlert className="w-6 h-6 text-amber-400" />
              <div>
                <p className="font-medium text-amber-300">Biometric Authentication Unavailable</p>
                <p className="text-sm text-amber-400/70">
                  Your device doesn't support Face ID or Touch ID
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Enable/Disable Toggle */}
      <div className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg">
        <div className="flex items-center gap-3">
          <Fingerprint className="w-6 h-6 text-emerald-400" />
          <div>
            <Label className="text-white font-medium">Enable Biometric Protection</Label>
            <p className="text-sm text-zinc-400">
              Use Face ID or Touch ID to protect sensitive features
            </p>
          </div>
        </div>
        <Switch
          checked={settings.enabled}
          onCheckedChange={handleToggleEnabled}
          disabled={!isAvailable || isEnabling}
        />
      </div>

      {/* Protected Features */}
      {settings.enabled && (
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-zinc-300 flex items-center gap-2">
            <Lock className="w-4 h-4" />
            Protected Features
          </h4>
          <div className="space-y-2">
            {features.map((feature) => (
              <div
                key={feature.key}
                className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg"
              >
                <div>
                  <p className="text-sm text-white">{feature.label}</p>
                  <p className="text-xs text-zinc-500">{feature.description}</p>
                </div>
                <Switch
                  checked={settings.protectedFeatures[feature.key]}
                  onCheckedChange={(checked) => updateProtectedFeature(feature.key, checked)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Auth Timeout */}
      {settings.enabled && (
        <div className="p-4 bg-zinc-800/30 rounded-lg">
          <Label className="text-white font-medium">Re-authentication Timeout</Label>
          <p className="text-sm text-zinc-400 mb-3">
            How long before requiring authentication again
          </p>
          <div className="flex gap-2">
            {[5, 15, 30, 60].map((minutes) => (
              <Button
                key={minutes}
                variant={settings.authTimeout === minutes ? 'default' : 'outline'}
                size="sm"
                onClick={() => updateSettings({ authTimeout: minutes })}
                className={settings.authTimeout === minutes ? 'bg-emerald-600' : ''}
              >
                {minutes}m
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Status Indicator */}
      {settings.enabled && settings.lastAuthenticated && (
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          Last authenticated: {new Date(settings.lastAuthenticated).toLocaleString()}
        </div>
      )}
    </div>
  );
}
