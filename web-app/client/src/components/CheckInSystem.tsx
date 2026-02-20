import { useState, useEffect } from 'react';
import { Bell, BellOff, Clock, Phone, MessageSquare, Check, AlertTriangle, Settings, Mic } from 'lucide-react';
import { VoiceCheckIn } from './VoiceCheckIn';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useLanguage } from '@/contexts/LanguageContext';

interface CheckInSettings {
  enabled: boolean;
  frequency: 'daily' | 'twice_daily' | 'weekly' | 'custom';
  customHours?: number;
  emergencyContact: {
    name: string;
    phone: string;
    email?: string;
  };
  alertMessage: string;
  lastCheckIn?: string;
  nextCheckIn?: string;
}

interface CheckInHistory {
  timestamp: string;
  status: 'completed' | 'missed' | 'pending';
  method?: 'manual' | 'voice';
}

export function CheckInSystem() {
  const { t } = useLanguage();
  const [settings, setSettings] = useState<CheckInSettings>(() => {
    const saved = localStorage.getItem('reunity-checkin-settings');
    if (saved) {
      return JSON.parse(saved);
    }
    return {
      enabled: false,
      frequency: 'daily',
      emergencyContact: { name: '', phone: '', email: '' },
      alertMessage: "I haven't checked in with ReUnity. Please check on me.",
    };
  });
  
  const [history, setHistory] = useState<CheckInHistory[]>(() => {
    const saved = localStorage.getItem('reunity-checkin-history');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [showSetup, setShowSetup] = useState(false);
  const [showCheckIn, setShowCheckIn] = useState(false);
  const [showVoiceCheckIn, setShowVoiceCheckIn] = useState(false);
  const [checkInPending, setCheckInPending] = useState(false);

  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('reunity-checkin-settings', JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    localStorage.setItem('reunity-checkin-history', JSON.stringify(history));
  }, [history]);

  // Check if check-in is due
  useEffect(() => {
    if (!settings.enabled || !settings.nextCheckIn) return;

    const checkDue = () => {
      const now = new Date();
      const nextCheckIn = new Date(settings.nextCheckIn!);
      
      if (now >= nextCheckIn) {
        setCheckInPending(true);
        setShowCheckIn(true);
      }
    };

    checkDue();
    const interval = setInterval(checkDue, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, [settings.enabled, settings.nextCheckIn]);

  // Request notification permission
  useEffect(() => {
    if (settings.enabled && 'Notification' in window) {
      Notification.requestPermission();
    }
  }, [settings.enabled]);

  const calculateNextCheckIn = (frequency: string, customHours?: number): string => {
    const now = new Date();
    let hours = 24;
    
    switch (frequency) {
      case 'twice_daily':
        hours = 12;
        break;
      case 'weekly':
        hours = 168;
        break;
      case 'custom':
        hours = customHours || 24;
        break;
      default:
        hours = 24;
    }
    
    now.setHours(now.getHours() + hours);
    return now.toISOString();
  };

  const handleEnableCheckIns = () => {
    if (!settings.emergencyContact.phone) {
      alert('Please add an emergency contact first');
      return;
    }
    
    const nextCheckIn = calculateNextCheckIn(settings.frequency, settings.customHours);
    setSettings(prev => ({
      ...prev,
      enabled: true,
      nextCheckIn,
      lastCheckIn: new Date().toISOString()
    }));
    setShowSetup(false);
    
    // Show notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('ReUnity Check-Ins Enabled', {
        body: `We'll check in with you ${settings.frequency === 'daily' ? 'daily' : settings.frequency === 'twice_daily' ? 'twice daily' : 'weekly'}.`,
        icon: '/icon.png'
      });
    }
  };

  const handleCheckIn = (method: 'manual' | 'voice' = 'manual') => {
    const now = new Date().toISOString();
    const nextCheckIn = calculateNextCheckIn(settings.frequency, settings.customHours);
    
    setSettings(prev => ({
      ...prev,
      lastCheckIn: now,
      nextCheckIn
    }));
    
    setHistory(prev => [{
      timestamp: now,
      status: 'completed',
      method
    }, ...prev.slice(0, 29)]); // Keep last 30 check-ins
    
    setCheckInPending(false);
    setShowCheckIn(false);
    setShowVoiceCheckIn(false);
  };

  const handleVoiceCheckInComplete = (status: 'okay' | 'not_okay' | 'emergency') => {
    if (status === 'okay') {
      handleCheckIn('voice');
    } else if (status === 'emergency') {
      // Trigger emergency alert
      handleMissedCheckIn();
    } else {
      // Show resources for not_okay
      setShowVoiceCheckIn(false);
      setShowCheckIn(true);
    }
  };

  const handleMissedCheckIn = () => {
    // In a real app, this would send an alert to the emergency contact
    const alertUrl = `sms:${settings.emergencyContact.phone}?body=${encodeURIComponent(settings.alertMessage)}`;
    
    setHistory(prev => [{
      timestamp: new Date().toISOString(),
      status: 'missed'
    }, ...prev.slice(0, 29)]);
    
    // Open SMS with pre-filled message
    window.open(alertUrl, '_blank');
  };

  const formatTimeUntil = (dateStr: string): string => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = date.getTime() - now.getTime();
    
    if (diff < 0) return 'Now';
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 24) {
      const days = Math.floor(hours / 24);
      return `${days} day${days > 1 ? 's' : ''}`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  return (
    <>
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {settings.enabled ? (
              <Bell className="h-5 w-5 text-emerald-400" />
            ) : (
              <BellOff className="h-5 w-5 text-zinc-500" />
            )}
            {String(t('checkIn.title'))}
          </CardTitle>
          <CardDescription>
            {settings.enabled 
              ? `Next check-in: ${settings.nextCheckIn ? formatTimeUntil(settings.nextCheckIn) : 'Not set'}`
              : 'Set up scheduled wellness check-ins'
            }
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {settings.enabled ? (
            <>
              <div className="flex items-center justify-between p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-emerald-400" />
                  <span className="text-sm">
                    {settings.frequency === 'daily' && 'Daily check-ins'}
                    {settings.frequency === 'twice_daily' && 'Twice daily check-ins'}
                    {settings.frequency === 'weekly' && 'Weekly check-ins'}
                    {settings.frequency === 'custom' && `Every ${settings.customHours}h check-ins`}
                  </span>
                </div>
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowSetup(true)}
                >
                  <Settings className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg">
                <div>
                  <p className="text-sm font-medium">{settings.emergencyContact.name}</p>
                  <p className="text-xs text-zinc-400">{settings.emergencyContact.phone}</p>
                </div>
                <Phone className="h-4 w-4 text-zinc-500" />
              </div>

              {checkInPending && (
                <>
                  <Button 
                    className="w-full bg-emerald-600 hover:bg-emerald-700"
                    onClick={() => handleCheckIn('manual')}
                  >
                    <Check className="h-4 w-4 mr-2" />
                    {String(t('checkIn.confirmButton'))}
                  </Button>
                  <Button 
                    className="w-full bg-blue-600 hover:bg-blue-700"
                    onClick={() => setShowVoiceCheckIn(true)}
                  >
                    <Mic className="h-4 w-4 mr-2" />
                    Voice Check-In
                  </Button>
                </>
              )}

              <Button 
                variant="outline" 
                className="w-full"
                onClick={() => setSettings(prev => ({ ...prev, enabled: false }))}
              >
                Disable Check-Ins
              </Button>
            </>
          ) : (
            <Button 
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              onClick={() => setShowSetup(true)}
            >
              <Bell className="h-4 w-4 mr-2" />
              Set Up Check-Ins
            </Button>
          )}

          {history.length > 0 && (
            <div className="pt-4 border-t border-zinc-800">
              <p className="text-sm font-medium mb-2">Recent Check-Ins</p>
              <div className="space-y-2 max-h-32 overflow-y-auto">
                {history.slice(0, 5).map((item, i) => (
                  <div key={i} className="flex items-center justify-between text-sm">
                    <span className="text-zinc-400">
                      {new Date(item.timestamp).toLocaleDateString()}
                    </span>
                    <span className={item.status === 'completed' ? 'text-emerald-400' : 'text-red-400'}>
                      {item.status === 'completed' ? '✓ Completed' : '✗ Missed'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Setup Dialog */}
      <Dialog open={showSetup} onOpenChange={setShowSetup}>
        <DialogContent className="bg-zinc-900 border-zinc-800 max-w-md">
          <DialogHeader>
            <DialogTitle>{String(t('checkIn.setupTitle'))}</DialogTitle>
            <DialogDescription>
              We'll remind you to check in and alert your emergency contact if you don't respond.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>{String(t('checkIn.frequency'))}</Label>
              <Select 
                value={settings.frequency}
                onValueChange={(value: 'daily' | 'twice_daily' | 'weekly' | 'custom') => 
                  setSettings(prev => ({ ...prev, frequency: value }))
                }
              >
                <SelectTrigger className="bg-zinc-800 border-zinc-700">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="daily">{String(t('checkIn.daily'))}</SelectItem>
                  <SelectItem value="twice_daily">{String(t('checkIn.twiceDaily'))}</SelectItem>
                  <SelectItem value="weekly">{String(t('checkIn.weekly'))}</SelectItem>
                  <SelectItem value="custom">{String(t('checkIn.custom'))}</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {settings.frequency === 'custom' && (
              <div className="space-y-2">
                <Label>Hours between check-ins</Label>
                <Input
                  type="number"
                  min="1"
                  max="168"
                  value={settings.customHours || 24}
                  onChange={(e) => setSettings(prev => ({ 
                    ...prev, 
                    customHours: parseInt(e.target.value) || 24 
                  }))}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
            )}

            <div className="space-y-2">
              <Label>{String(t('checkIn.emergencyContact'))}</Label>
              <Input
                placeholder="Contact name"
                value={settings.emergencyContact.name}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  emergencyContact: { ...prev.emergencyContact, name: e.target.value }
                }))}
                className="bg-zinc-800 border-zinc-700"
              />
              <Input
                placeholder="Phone number"
                type="tel"
                value={settings.emergencyContact.phone}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  emergencyContact: { ...prev.emergencyContact, phone: e.target.value }
                }))}
                className="bg-zinc-800 border-zinc-700"
              />
            </div>

            <div className="space-y-2">
              <Label>{String(t('checkIn.alertMessage'))}</Label>
              <Input
                value={settings.alertMessage}
                onChange={(e) => setSettings(prev => ({ ...prev, alertMessage: e.target.value }))}
                className="bg-zinc-800 border-zinc-700"
                placeholder={String(t('checkIn.defaultAlert'))}
              />
            </div>

            <Button 
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              onClick={handleEnableCheckIns}
              disabled={!settings.emergencyContact.phone}
            >
              Enable Check-Ins
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Check-In Reminder Dialog */}
      <Dialog open={showCheckIn} onOpenChange={setShowCheckIn}>
        <DialogContent className="bg-zinc-900 border-zinc-800 max-w-sm">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-400" />
              {String(t('checkIn.title'))}
            </DialogTitle>
            <DialogDescription>
              {String(t('checkIn.missedMessage'))}
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-3">
            <Button 
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              onClick={() => handleCheckIn('manual')}
            >
              <Check className="h-4 w-4 mr-2" />
              {String(t('checkIn.confirmButton'))}
            </Button>
            
            <Button 
              className="w-full bg-blue-600 hover:bg-blue-700"
              onClick={() => {
                setShowCheckIn(false);
                setShowVoiceCheckIn(true);
              }}
            >
              <Mic className="h-4 w-4 mr-2" />
              Voice Check-In
            </Button>
            
            <Button 
              variant="outline"
              className="w-full border-red-500/50 text-red-400 hover:bg-red-500/10"
              onClick={handleMissedCheckIn}
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              Alert {settings.emergencyContact.name || 'Emergency Contact'}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Voice Check-In */}
      <VoiceCheckIn
        isOpen={showVoiceCheckIn}
        onClose={() => setShowVoiceCheckIn(false)}
        onCheckInComplete={handleVoiceCheckInComplete}
      />
    </>
  );
}
