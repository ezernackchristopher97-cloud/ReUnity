import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { 
  Smartphone, 
  Link2, 
  Unlink, 
  Copy, 
  Check, 
  MapPin, 
  Heart, 
  AlertTriangle,
  Shield,
  RefreshCw,
  QrCode
} from 'lucide-react';
import { toast } from 'sonner';

interface PairedDevice {
  id: string;
  name: string;
  pairedAt: string;
  lastSeen?: string;
  relationship: string;
  permissions: {
    location: boolean;
    wellness: boolean;
    crisisAlerts: boolean;
  };
}

interface PairingRequest {
  code: string;
  expiresAt: string;
  deviceName: string;
}

// Generate a random 6-character pairing code
const generatePairingCode = (): string => {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Exclude confusing chars
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
};

// Generate a unique device ID
const generateDeviceId = (): string => {
  return 'dev_' + Math.random().toString(36).substring(2, 15);
};

export function TrustedDevicePairing() {
  const [pairedDevices, setPairedDevices] = useState<PairedDevice[]>(() => {
    const saved = localStorage.getItem('reunity-paired-devices');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [showPairDialog, setShowPairDialog] = useState(false);
  const [showEnterCodeDialog, setShowEnterCodeDialog] = useState(false);
  const [pairingCode, setPairingCode] = useState<PairingRequest | null>(null);
  const [enteredCode, setEnteredCode] = useState('');
  const [deviceName, setDeviceName] = useState('');
  const [relationship, setRelationship] = useState('');
  const [copied, setCopied] = useState(false);
  const [sharingEnabled, setSharingEnabled] = useState(() => {
    return localStorage.getItem('reunity-sharing-enabled') === 'true';
  });

  // Save to localStorage
  useEffect(() => {
    localStorage.setItem('reunity-paired-devices', JSON.stringify(pairedDevices));
  }, [pairedDevices]);

  useEffect(() => {
    localStorage.setItem('reunity-sharing-enabled', String(sharingEnabled));
  }, [sharingEnabled]);

  // Generate new pairing code
  const generateNewCode = useCallback(() => {
    const code = generatePairingCode();
    const expiresAt = new Date(Date.now() + 10 * 60 * 1000).toISOString(); // 10 minutes
    
    setPairingCode({
      code,
      expiresAt,
      deviceName: navigator.userAgent.includes('Mobile') ? 'Mobile Device' : 'Desktop'
    });
  }, []);

  // Copy code to clipboard
  const copyCode = useCallback(async () => {
    if (pairingCode) {
      await navigator.clipboard.writeText(pairingCode.code);
      setCopied(true);
      toast.success('Code copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    }
  }, [pairingCode]);

  // Pair with entered code (simulated - in real app would verify with server)
  const handlePairDevice = useCallback(() => {
    if (!enteredCode || enteredCode.length !== 6) {
      toast.error('Please enter a valid 6-character code');
      return;
    }
    
    if (!deviceName.trim()) {
      toast.error('Please enter a name for this device');
      return;
    }

    // In a real app, this would verify the code with a server
    // For now, we'll simulate successful pairing
    const newDevice: PairedDevice = {
      id: generateDeviceId(),
      name: deviceName.trim(),
      pairedAt: new Date().toISOString(),
      lastSeen: new Date().toISOString(),
      relationship: relationship || 'Family Member',
      permissions: {
        location: true,
        wellness: true,
        crisisAlerts: true
      }
    };

    setPairedDevices(prev => [...prev, newDevice]);
    setShowEnterCodeDialog(false);
    setEnteredCode('');
    setDeviceName('');
    setRelationship('');
    toast.success(`Successfully paired with ${newDevice.name}`);
  }, [enteredCode, deviceName, relationship]);

  // Remove paired device
  const removePairedDevice = useCallback((deviceId: string) => {
    setPairedDevices(prev => prev.filter(d => d.id !== deviceId));
    toast.success('Device unpaired');
  }, []);

  // Update device permissions
  const updatePermissions = useCallback((deviceId: string, permission: keyof PairedDevice['permissions'], value: boolean) => {
    setPairedDevices(prev => prev.map(d => {
      if (d.id === deviceId) {
        return {
          ...d,
          permissions: {
            ...d.permissions,
            [permission]: value
          }
        };
      }
      return d;
    }));
  }, []);

  // Send emergency alert to all paired devices
  const sendEmergencyAlert = useCallback(() => {
    const alertDevices = pairedDevices.filter(d => d.permissions.crisisAlerts);
    if (alertDevices.length === 0) {
      toast.error('No devices configured to receive alerts');
      return;
    }

    // In a real app, this would send push notifications
    toast.success(`Emergency alert sent to ${alertDevices.length} device(s)`);
    
    // Simulate sending location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          console.log('Location shared:', { latitude, longitude });
          toast.info('Your location has been shared');
        },
        (error) => {
          console.error('Location error:', error);
        }
      );
    }
  }, [pairedDevices]);

  return (
    <Card className="bg-slate-900/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-lg text-white flex items-center gap-2">
          <Smartphone className="h-5 w-5 text-emerald-400" />
          Trusted Device Pairing
        </CardTitle>
        <CardDescription className="text-slate-400">
          Link your device with family members to share location and wellness data in emergencies
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Master sharing toggle */}
        <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-emerald-400" />
            <div>
              <p className="text-sm font-medium text-white">Enable Sharing</p>
              <p className="text-xs text-slate-400">Allow paired devices to receive your data</p>
            </div>
          </div>
          <Switch
            checked={sharingEnabled}
            onCheckedChange={setSharingEnabled}
          />
        </div>

        {/* Paired devices list */}
        {pairedDevices.length > 0 ? (
          <div className="space-y-3">
            <p className="text-sm text-slate-400">Paired Devices ({pairedDevices.length})</p>
            {pairedDevices.map(device => (
              <div key={device.id} className="p-3 bg-slate-800/50 rounded-lg space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                      <Smartphone className="h-5 w-5 text-emerald-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">{device.name}</p>
                      <p className="text-xs text-slate-400">{device.relationship}</p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
                    onClick={() => removePairedDevice(device.id)}
                  >
                    <Unlink className="h-4 w-4" />
                  </Button>
                </div>
                
                {/* Permissions */}
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => updatePermissions(device.id, 'location', !device.permissions.location)}
                    className={`p-2 rounded text-xs flex flex-col items-center gap-1 transition-colors ${
                      device.permissions.location 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-slate-700/50 text-slate-500'
                    }`}
                  >
                    <MapPin className="h-4 w-4" />
                    Location
                  </button>
                  <button
                    onClick={() => updatePermissions(device.id, 'wellness', !device.permissions.wellness)}
                    className={`p-2 rounded text-xs flex flex-col items-center gap-1 transition-colors ${
                      device.permissions.wellness 
                        ? 'bg-emerald-500/20 text-emerald-400' 
                        : 'bg-slate-700/50 text-slate-500'
                    }`}
                  >
                    <Heart className="h-4 w-4" />
                    Wellness
                  </button>
                  <button
                    onClick={() => updatePermissions(device.id, 'crisisAlerts', !device.permissions.crisisAlerts)}
                    className={`p-2 rounded text-xs flex flex-col items-center gap-1 transition-colors ${
                      device.permissions.crisisAlerts 
                        ? 'bg-red-500/20 text-red-400' 
                        : 'bg-slate-700/50 text-slate-500'
                    }`}
                  >
                    <AlertTriangle className="h-4 w-4" />
                    Alerts
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-6 text-slate-400">
            <Smartphone className="h-12 w-12 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No paired devices yet</p>
            <p className="text-xs">Pair with a family member's device to share safety data</p>
          </div>
        )}

        {/* Action buttons */}
        <div className="grid grid-cols-2 gap-3">
          <Button
            variant="outline"
            className="border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10"
            onClick={() => {
              generateNewCode();
              setShowPairDialog(true);
            }}
          >
            <QrCode className="h-4 w-4 mr-2" />
            Share My Code
          </Button>
          <Button
            variant="outline"
            className="border-blue-500/50 text-blue-400 hover:bg-blue-500/10"
            onClick={() => setShowEnterCodeDialog(true)}
          >
            <Link2 className="h-4 w-4 mr-2" />
            Enter Code
          </Button>
        </div>

        {/* Emergency alert button */}
        {pairedDevices.some(d => d.permissions.crisisAlerts) && sharingEnabled && (
          <Button
            className="w-full bg-red-600 hover:bg-red-700"
            onClick={sendEmergencyAlert}
          >
            <AlertTriangle className="h-4 w-4 mr-2" />
            Send Emergency Alert
          </Button>
        )}
      </CardContent>

      {/* Share Code Dialog */}
      <Dialog open={showPairDialog} onOpenChange={setShowPairDialog}>
        <DialogContent className="bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">Share Your Pairing Code</DialogTitle>
            <DialogDescription className="text-slate-400">
              Give this code to a trusted family member to pair their device with yours
            </DialogDescription>
          </DialogHeader>
          
          {pairingCode && (
            <div className="space-y-4">
              {/* Large code display */}
              <div className="bg-slate-800 rounded-lg p-6 text-center">
                <p className="text-4xl font-mono font-bold text-emerald-400 tracking-widest">
                  {pairingCode.code}
                </p>
                <p className="text-xs text-slate-500 mt-2">
                  Expires in 10 minutes
                </p>
              </div>

              {/* Copy and refresh buttons */}
              <div className="flex gap-3">
                <Button
                  className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                  onClick={copyCode}
                >
                  {copied ? (
                    <>
                      <Check className="h-4 w-4 mr-2" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-4 w-4 mr-2" />
                      Copy Code
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={generateNewCode}
                >
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>

              {/* Instructions */}
              <div className="bg-slate-800/50 rounded-lg p-3 text-sm text-slate-400">
                <p className="font-medium text-white mb-2">Instructions:</p>
                <ol className="list-decimal list-inside space-y-1">
                  <li>Share this code with your trusted contact</li>
                  <li>They open ReUnity and tap "Enter Code"</li>
                  <li>Once paired, they can receive your alerts</li>
                </ol>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Enter Code Dialog */}
      <Dialog open={showEnterCodeDialog} onOpenChange={setShowEnterCodeDialog}>
        <DialogContent className="bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">Pair with Another Device</DialogTitle>
            <DialogDescription className="text-slate-400">
              Enter the 6-character code from your family member's device
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="pairing-code" className="text-slate-300">Pairing Code</Label>
              <Input
                id="pairing-code"
                value={enteredCode}
                onChange={(e) => setEnteredCode(e.target.value.toUpperCase().slice(0, 6))}
                placeholder="ABC123"
                className="bg-slate-800 border-slate-600 text-white text-center text-2xl font-mono tracking-widest"
                maxLength={6}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="device-name" className="text-slate-300">Their Name</Label>
              <Input
                id="device-name"
                value={deviceName}
                onChange={(e) => setDeviceName(e.target.value)}
                placeholder="Mom, Dad, Sister..."
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="relationship" className="text-slate-300">Relationship</Label>
              <Input
                id="relationship"
                value={relationship}
                onChange={(e) => setRelationship(e.target.value)}
                placeholder="Parent, Sibling, Partner..."
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>

            <Button
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              onClick={handlePairDevice}
              disabled={enteredCode.length !== 6 || !deviceName.trim()}
            >
              <Link2 className="h-4 w-4 mr-2" />
              Pair Device
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

export default TrustedDevicePairing;
