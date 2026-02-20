import { useState, useEffect } from 'react';
import { Link } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Settings as SettingsIcon, 
  Accessibility, 
  Smartphone, 
  Globe, 
  Lock, 
  Shield,
  ArrowLeft,
  Download,
  Trash2,
  AlertTriangle,
  Bell,
  Watch,
  Video,
  Fingerprint,
  Activity
} from 'lucide-react';
import AccessibilitySettings from '@/components/AccessibilitySettings';
import TrustedDevicePairing from '@/components/TrustedDevicePairing';
import OfflineCrisisCard from '@/components/OfflineCrisisCard';
import PushNotificationManager from '@/components/PushNotificationManager';
import WearableIntegration from '@/components/WearableIntegration';
import { LanguageSelector } from '@/components/LanguageSelector';
import { BiometricSettings } from '@/components/BiometricAuth';
import MoodPrediction from '@/components/MoodPrediction';
import { trpc } from '@/lib/trpc';
import { useAuth } from '@/contexts/AuthContext';

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Heart, Mic, Loader2, Save } from 'lucide-react';


type SettingsSection = 'general' | 'preferences' | 'accessibility' | 'devices' | 'language' | 'privacy' | 'crisis' | 'notifications' | 'wearables' | 'biometric' | 'mood';

// Belief systems
const BELIEF_SYSTEMS = [
  { id: 'none', name: 'No preference / Secular' },
  { id: 'christianity', name: 'Christianity' },
  { id: 'islam', name: 'Islam' },
  { id: 'judaism', name: 'Judaism' },
  { id: 'hinduism', name: 'Hinduism' },
  { id: 'buddhism', name: 'Buddhism' },
  { id: 'sikhism', name: 'Sikhism' },
  { id: 'paganism', name: 'Paganism' },
  { id: 'atheism', name: 'Atheism' },
  { id: 'agnosticism', name: 'Agnosticism' },
  { id: 'existentialism', name: 'Existentialism' },
  { id: 'stoicism', name: 'Stoicism' },
  { id: 'nihilism', name: 'Nihilism' },
  { id: 'absurdism', name: 'Absurdism' },
  { id: 'solipsism', name: 'Solipsism' },
];

// Voice personas
const VOICE_PERSONAS = [
  { id: 'gentle_woman', name: 'Gentle Woman' },
  { id: 'gentle_man', name: 'Gentle Man' },
  { id: 'neutral', name: 'Neutral' },
  { id: 'warm_elder', name: 'Warm Elder' },
  { id: 'calm_friend', name: 'Calm Friend' },
];

export default function Settings() {
  const [activeSection, setActiveSection] = useState<SettingsSection>('general');
  const { user } = useAuth();
  
  
  // Preferences state
  const [beliefSystem, setBeliefSystem] = useState('none');
  const [voicePersona, setVoicePersona] = useState('neutral');
  const [autoPlayTTS, setAutoPlayTTS] = useState(false);
  const [isSavingPrefs, setIsSavingPrefs] = useState(false);
  
  const { data: preferences } = trpc.preferences.get.useQuery(undefined, { enabled: !!user });
  const updatePrefsMutation = trpc.preferences.update.useMutation({
    onSuccess: () => alert('Preferences saved!'),
    onError: (e) => alert('Error: ' + e.message)
  });
  
  useEffect(() => {
    if (preferences) {
      setBeliefSystem(preferences.beliefSystem || 'none');
      setVoicePersona(preferences.voicePersona || 'neutral');
      setAutoPlayTTS(preferences.autoPlayTTS || false);
    }
  }, [preferences]);
  
  const savePreferences = async () => {
    setIsSavingPrefs(true);
    await updatePrefsMutation.mutateAsync({ beliefSystem: beliefSystem === 'none' ? undefined : beliefSystem, voicePersona, autoPlayTTS });
    setIsSavingPrefs(false);
  };
  

  const exportData = async () => {
    try {
      const keys = Object.keys(localStorage).filter(k => k.startsWith('reunity') || k.startsWith('reop'));
      const data: Record<string, unknown> = {};
      keys.forEach(key => {
        const value = localStorage.getItem(key);
        if (value) {
          try { data[key] = JSON.parse(value); } catch { data[key] = value; }
        }
      });
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `reunity-data-${new Date().toISOString().split('T')[0]}.json`;
      a.click();
      URL.revokeObjectURL(url);
      alert('Data exported successfully!');
    } catch (e) {
      alert('Export failed. Please try again.');
    }
  };

  const clearAllData = () => {
    if (confirm('Are you sure you want to delete all app data? This cannot be undone.')) {
      const keys = Object.keys(localStorage).filter(k => k.startsWith('reunity') || k.startsWith('reop'));
      keys.forEach(key => localStorage.removeItem(key));
      alert('All app data has been deleted.');
    }
  };

  const sections = [
    { id: 'general' as const, label: 'General', icon: SettingsIcon },
    { id: 'preferences' as const, label: 'Preferences', icon: Heart },
    { id: 'notifications' as const, label: 'Notifications', icon: Bell },
    { id: 'wearables' as const, label: 'Wearables', icon: Watch },
    { id: 'mood' as const, label: 'Mood Prediction', icon: Activity },
    { id: 'biometric' as const, label: 'Biometric Lock', icon: Fingerprint },
    { id: 'accessibility' as const, label: 'Accessibility', icon: Accessibility },
    { id: 'devices' as const, label: 'Trusted Devices', icon: Smartphone },
    { id: 'language' as const, label: 'Language', icon: Globe },
    { id: 'privacy' as const, label: 'Privacy & Data', icon: Lock },
    { id: 'crisis' as const, label: 'Crisis Resources', icon: Shield },
  ];

  const renderContent = () => {
    switch (activeSection) {
      case 'preferences':
        return (
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader><CardTitle className="flex items-center gap-2"><Heart className="w-5 h-5 text-emerald-400" />Preferences</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Belief System / Philosophy</Label>
                <Select value={beliefSystem} onValueChange={setBeliefSystem}>
                  <SelectTrigger className="bg-zinc-800 border-zinc-700"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-zinc-800 border-zinc-700 max-h-60">
                    {BELIEF_SYSTEMS.map(b => <SelectItem key={b.id} value={b.id}>{b.name}</SelectItem>)}
                  </SelectContent>
                </Select>
                <p className="text-xs text-zinc-500">ReUnity will incorporate wisdom from your tradition.</p>
              </div>
              <div className="space-y-2">
                <Label>Voice Persona</Label>
                <Select value={voicePersona} onValueChange={setVoicePersona}>
                  <SelectTrigger className="bg-zinc-800 border-zinc-700"><SelectValue /></SelectTrigger>
                  <SelectContent className="bg-zinc-800 border-zinc-700">
                    {VOICE_PERSONAS.map(v => <SelectItem key={v.id} value={v.id}>{v.name}</SelectItem>)}
                  </SelectContent>
                </Select>
                <p className="text-xs text-zinc-500">Choose who you want to talk to. All gender identities welcome.</p>
              </div>
              <div className="flex items-center justify-between">
                <div><Label>Auto-play Voice</Label><p className="text-xs text-zinc-500">Read AI responses aloud</p></div>
                <Switch checked={autoPlayTTS} onCheckedChange={setAutoPlayTTS} />
              </div>
              <Button onClick={savePreferences} disabled={isSavingPrefs} className="w-full bg-emerald-600 hover:bg-emerald-500">
                {isSavingPrefs ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
                Save Preferences
              </Button>
            </CardContent>
          </Card>
        );
      case 'accessibility':
        return <AccessibilitySettings />;
      case 'devices':
        return <TrustedDevicePairing />;
      case 'language':
        return (
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader><CardTitle className="flex items-center gap-2"><Globe className="w-5 h-5 text-blue-400" />Language Settings</CardTitle></CardHeader>
            <CardContent><p className="text-sm text-zinc-400 mb-4">Choose your preferred language for crisis resources</p><LanguageSelector /></CardContent>
          </Card>
        );
      case 'privacy':
        return (
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader><CardTitle className="flex items-center gap-2"><Lock className="w-5 h-5 text-red-400" />Privacy & Data</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <h4 className="text-sm font-medium text-blue-300 mb-2">Your Privacy Matters</h4>
                <p className="text-xs text-zinc-400">ReUnity stores data locally on your device. Your conversations and personal information are not sent to external servers without your consent.</p>
              </div>
              <Button variant="outline" className="w-full justify-start gap-3 border-zinc-700" onClick={exportData}>
                <Download className="w-4 h-4 text-emerald-400" />Export My Data
              </Button>
              <Button variant="outline" className="w-full justify-start gap-3 border-red-500/30 text-red-400 hover:bg-red-500/10" onClick={clearAllData}>
                <Trash2 className="w-4 h-4" />Clear All Data
              </Button>
            </CardContent>
          </Card>
        );
      case 'crisis':
        return <OfflineCrisisCard />;
      case 'notifications':
        return <PushNotificationManager />;
      case 'wearables':
        return <WearableIntegration />;
      case 'biometric':
        return (
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Fingerprint className="w-5 h-5 text-emerald-400" />
                Biometric Protection
              </CardTitle>
            </CardHeader>
            <CardContent>
              <BiometricSettings />
            </CardContent>
          </Card>
        );
      case 'mood':
        return <MoodPrediction />;
      default:
        return (
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader><CardTitle className="flex items-center gap-2"><SettingsIcon className="w-5 h-5 text-emerald-400" />General Settings</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 rounded-lg bg-zinc-800/50">
                <h4 className="text-sm font-medium mb-2">About ReUnity</h4>
                <p className="text-xs text-zinc-400">Version 1.0.0</p>
                <p className="text-xs text-zinc-500 mt-2">A trauma-informed AI companion for mental health support.</p>
              </div>
              <div className="p-4 rounded-lg bg-zinc-800/50">
                <h4 className="text-sm font-medium mb-2">Safety Features</h4>
                <ul className="text-xs text-zinc-400 space-y-1">
                  <li>• Panic button with decoy mode</li>
                  <li>• Voice-activated check-ins</li>
                  <li>• Trusted device pairing</li>
                  <li>• Biometric lock for safety plans</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950">
      <header className="border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container flex items-center justify-between h-16 px-4">
          <div className="flex items-center gap-4">
            <Link href="/dashboard">
              <Button variant="ghost" size="sm" className="gap-2"><ArrowLeft className="w-4 h-4" />Back</Button>
            </Link>
            <h1 className="text-lg font-semibold">Settings</h1>
          </div>
        </div>
      </header>
      <div className="container py-6 px-4">
        <div className="flex gap-6">
          <nav className="w-48 shrink-0 space-y-1">
            {sections.map(({ id, label, icon: Icon }) => (
              <button key={id} onClick={() => setActiveSection(id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === id ? 'bg-emerald-500/20 text-emerald-400' : 'text-zinc-400 hover:bg-zinc-800 hover:text-white'}`}>
                <Icon className="w-4 h-4" />{label}
              </button>
            ))}
          </nav>
          <main className="flex-1 max-w-2xl">{renderContent()}</main>
        </div>
      </div>
      {/* Global Disclaimer */}
      <div className="border-t border-zinc-800 bg-zinc-950/90 py-6 mt-8">
        <div className="container px-4">
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
            <p className="text-amber-200 text-sm font-medium mb-2">Important Disclaimer</p>
            <p className="text-zinc-300 text-sm">ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.</p>
          </div>
          <div className="flex gap-4 mt-4 text-sm">
            <Link href="/privacy" className="text-emerald-400 hover:text-emerald-300">Privacy Policy</Link>
            <Link href="/terms" className="text-emerald-400 hover:text-emerald-300">Terms of Service</Link>
            <Link href="/disclaimer" className="text-emerald-400 hover:text-emerald-300">Full Disclaimer</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
