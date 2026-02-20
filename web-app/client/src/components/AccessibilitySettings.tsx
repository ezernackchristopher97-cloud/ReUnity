import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { 
  Accessibility, 
  Type, 
  Eye, 
  Zap, 
  MousePointer, 
  Volume2,
  RotateCcw,
  Sun,
  Moon
} from 'lucide-react';

interface AccessibilityPreferences {
  fontSize: 'small' | 'medium' | 'large' | 'xlarge';
  highContrast: boolean;
  reduceMotion: boolean;
  screenReaderOptimized: boolean;
  largeClickTargets: boolean;
  autoReadContent: boolean;
  darkMode: boolean;
}

const defaultPreferences: AccessibilityPreferences = {
  fontSize: 'medium',
  highContrast: false,
  reduceMotion: false,
  screenReaderOptimized: false,
  largeClickTargets: false,
  autoReadContent: false,
  darkMode: true,
};

const STORAGE_KEY = 'reunity_accessibility';

export default function AccessibilitySettings() {
  const [preferences, setPreferences] = useState<AccessibilityPreferences>(defaultPreferences);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        setPreferences(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load accessibility preferences:', e);
      }
    }
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    const fontSizes = { small: '14px', medium: '16px', large: '18px', xlarge: '20px' };
    root.style.setProperty('--base-font-size', fontSizes[preferences.fontSize]);
    
    if (preferences.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
    
    if (preferences.reduceMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }
    
    if (preferences.largeClickTargets) {
      root.classList.add('large-targets');
    } else {
      root.classList.remove('large-targets');
    }
  }, [preferences]);

  const savePreferences = (newPrefs: AccessibilityPreferences) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newPrefs));
    setPreferences(newPrefs);
  };

  const updatePreference = <K extends keyof AccessibilityPreferences>(
    key: K,
    value: AccessibilityPreferences[K]
  ) => {
    const newPrefs = { ...preferences, [key]: value };
    savePreferences(newPrefs);
  };

  const resetToDefaults = () => {
    savePreferences(defaultPreferences);
  };

  const fontSizes = ['small', 'medium', 'large', 'xlarge'] as const;
  const fontSizeLabels = { small: 'S', medium: 'M', large: 'L', xlarge: 'XL' };

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-blue-500/20">
              <Accessibility className="w-5 h-5 text-blue-400" />
            </div>
            <CardTitle className="text-lg">Accessibility Settings</CardTitle>
          </div>
          <Button variant="ghost" size="sm" onClick={resetToDefaults} className="text-zinc-400 hover:text-white">
            <RotateCcw className="w-4 h-4 mr-2" />Reset
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Type className="w-4 h-4" /><span>Text Size</span>
          </div>
          <div className="flex gap-2">
            {fontSizes.map((size) => (
              <Button key={size} variant={preferences.fontSize === size ? 'default' : 'outline'} size="sm"
                onClick={() => updatePreference('fontSize', size)}
                className={`flex-1 ${preferences.fontSize === size ? 'bg-blue-600 hover:bg-blue-700' : 'bg-transparent border-zinc-700 hover:bg-zinc-800'}`}>
                {fontSizeLabels[size]}
              </Button>
            ))}
          </div>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div className="flex items-center gap-3">
              <Eye className="w-5 h-5 text-yellow-400" />
              <div><Label className="text-sm font-medium">High Contrast</Label><p className="text-xs text-zinc-500">Increase color contrast</p></div>
            </div>
            <Switch checked={preferences.highContrast} onCheckedChange={(checked) => updatePreference('highContrast', checked)} />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div className="flex items-center gap-3">
              <Zap className="w-5 h-5 text-purple-400" />
              <div><Label className="text-sm font-medium">Reduce Motion</Label><p className="text-xs text-zinc-500">Minimize animations</p></div>
            </div>
            <Switch checked={preferences.reduceMotion} onCheckedChange={(checked) => updatePreference('reduceMotion', checked)} />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div className="flex items-center gap-3">
              <MousePointer className="w-5 h-5 text-emerald-400" />
              <div><Label className="text-sm font-medium">Large Touch Targets</Label><p className="text-xs text-zinc-500">Bigger buttons</p></div>
            </div>
            <Switch checked={preferences.largeClickTargets} onCheckedChange={(checked) => updatePreference('largeClickTargets', checked)} />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div className="flex items-center gap-3">
              <Volume2 className="w-5 h-5 text-blue-400" />
              <div><Label className="text-sm font-medium">Screen Reader Optimized</Label><p className="text-xs text-zinc-500">Enhanced labels</p></div>
            </div>
            <Switch checked={preferences.screenReaderOptimized} onCheckedChange={(checked) => updatePreference('screenReaderOptimized', checked)} />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div className="flex items-center gap-3">
              <Volume2 className="w-5 h-5 text-pink-400" />
              <div><Label className="text-sm font-medium">Auto-Read Responses</Label><p className="text-xs text-zinc-500">Read AI responses aloud</p></div>
            </div>
            <Switch checked={preferences.autoReadContent} onCheckedChange={(checked) => updatePreference('autoReadContent', checked)} />
          </div>
        </div>
        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <h4 className="text-sm font-medium text-blue-300 mb-2">Accessibility Tips</h4>
          <ul className="text-xs text-zinc-400 space-y-1">
            <li>• Shake device 3 times for panic mode (mobile)</li>
            <li>• Voice commands available during check-ins</li>
            <li>• Use Tab key for keyboard navigation</li>
            <li>• Safe word triggers instant panic mode</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
