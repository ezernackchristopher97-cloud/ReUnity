import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  WifiOff, 
  Phone, 
  Heart, 
  Wind, 
  Hand,
  RefreshCw,
  AlertTriangle,
  Shield
} from 'lucide-react';

interface CachedResource {
  name: string;
  phone: string;
  description: string;
}

interface GroundingTechnique {
  name: string;
  steps: string[];
  icon: React.ReactNode;
}

const CRISIS_RESOURCES: CachedResource[] = [
  { name: '988 Suicide & Crisis Lifeline', phone: '988', description: '24/7 crisis support' },
  { name: 'National DV Hotline', phone: '1-800-799-7233', description: 'Domestic violence help' },
  { name: 'Crisis Text Line', phone: 'Text HOME to 741741', description: 'Text-based support' },
  { name: 'SAMHSA Helpline', phone: '1-800-662-4357', description: 'Substance abuse help' },
  { name: 'Trevor Project', phone: '1-866-488-7386', description: 'LGBTQ+ youth crisis' },
  { name: 'Emergency Services', phone: '911', description: 'Immediate danger' },
];

const GROUNDING_TECHNIQUES: GroundingTechnique[] = [
  {
    name: '5-4-3-2-1 Grounding',
    icon: <Hand className="w-5 h-5 text-emerald-400" />,
    steps: [
      'Name 5 things you can SEE',
      'Name 4 things you can TOUCH',
      'Name 3 things you can HEAR',
      'Name 2 things you can SMELL',
      'Name 1 thing you can TASTE',
    ],
  },
  {
    name: 'Box Breathing',
    icon: <Wind className="w-5 h-5 text-blue-400" />,
    steps: [
      'Breathe IN for 4 seconds',
      'HOLD for 4 seconds',
      'Breathe OUT for 4 seconds',
      'HOLD for 4 seconds',
      'Repeat 4 times',
    ],
  },
  {
    name: 'Grounding Statements',
    icon: <Heart className="w-5 h-5 text-pink-400" />,
    steps: [
      'My name is...',
      'I am safe right now',
      'Today is [day/date]',
      'I am in [location]',
      'This feeling will pass',
    ],
  },
];

export default function OfflineCrisisCard() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [selectedTechnique, setSelectedTechnique] = useState<GroundingTechnique | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const startTechnique = (technique: GroundingTechnique) => {
    setSelectedTechnique(technique);
    setCurrentStep(0);
  };

  const nextStep = () => {
    if (selectedTechnique && currentStep < selectedTechnique.steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setSelectedTechnique(null);
      setCurrentStep(0);
    }
  };

  const callNumber = (phone: string) => {
    if (phone.startsWith('Text')) {
      window.location.href = `sms:741741?body=HOME`;
    } else {
      window.location.href = `tel:${phone.replace(/[^0-9]/g, '')}`;
    }
  };

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${isOnline ? 'bg-emerald-500/20' : 'bg-amber-500/20'}`}>
              {isOnline ? (
                <Shield className="w-5 h-5 text-emerald-400" />
              ) : (
                <WifiOff className="w-5 h-5 text-amber-400" />
              )}
            </div>
            <div>
              <CardTitle className="text-lg">
                {isOnline ? 'Crisis Resources' : 'Offline Crisis Card'}
              </CardTitle>
              <p className="text-xs text-zinc-500">
                {isOnline ? 'Connected - Full resources available' : 'Cached resources available offline'}
              </p>
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Active Grounding Technique */}
        {selectedTechnique && (
          <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <div className="flex items-center gap-2 mb-3">
              {selectedTechnique.icon}
              <span className="font-medium text-emerald-300">{selectedTechnique.name}</span>
            </div>
            <div className="text-center py-6">
              <p className="text-2xl font-medium text-white mb-2">
                {selectedTechnique.steps[currentStep]}
              </p>
              <p className="text-sm text-zinc-400">
                Step {currentStep + 1} of {selectedTechnique.steps.length}
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                className="flex-1 border-zinc-700"
                onClick={() => setSelectedTechnique(null)}
              >
                Cancel
              </Button>
              <Button
                className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                onClick={nextStep}
              >
                {currentStep < selectedTechnique.steps.length - 1 ? 'Next' : 'Done'}
              </Button>
            </div>
          </div>
        )}

        {/* Grounding Techniques */}
        {!selectedTechnique && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-zinc-400">Quick Grounding</h4>
            <div className="grid gap-2">
              {GROUNDING_TECHNIQUES.map((technique) => (
                <Button
                  key={technique.name}
                  variant="outline"
                  className="justify-start gap-3 h-auto py-3 border-zinc-700 hover:bg-zinc-800"
                  onClick={() => startTechnique(technique)}
                >
                  {technique.icon}
                  <span>{technique.name}</span>
                </Button>
              ))}
            </div>
          </div>
        )}

        {/* Crisis Hotlines */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-zinc-400 flex items-center gap-2">
            <Phone className="w-4 h-4" />
            Crisis Hotlines
          </h4>
          <div className="space-y-2">
            {CRISIS_RESOURCES.map((resource) => (
              <button
                key={resource.phone}
                onClick={() => callNumber(resource.phone)}
                className="w-full p-3 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 transition-colors text-left"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-white">{resource.name}</p>
                    <p className="text-xs text-zinc-500">{resource.description}</p>
                  </div>
                  <span className="text-emerald-400 font-mono text-sm">{resource.phone}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Offline Notice */}
        {!isOnline && (
          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5" />
              <div>
                <p className="text-sm text-amber-300">You're offline</p>
                <p className="text-xs text-zinc-400">
                  These resources are cached and available without internet. 
                  Phone calls may still work depending on your connection.
                </p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
