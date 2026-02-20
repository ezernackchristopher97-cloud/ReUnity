import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { 
  Moon, 
  Sun, 
  Clock, 
  TrendingUp, 
  TrendingDown,
  Minus,
  Calendar,
  Brain,
  AlertTriangle,
  CheckCircle,
  Zap,
  Coffee,
  Wind,
  Activity
} from 'lucide-react';

interface SleepEntry {
  id: string;
  date: string;
  bedtime: string;
  wakeTime: string;
  duration: number; // in hours
  quality: number; // 1-100
  factors: string[];
  dreams: 'none' | 'pleasant' | 'neutral' | 'nightmares';
  wakeUps: number;
  entropyImpact: number; // -20 to +20
  notes: string;
}

interface SleepTrackingProps {
  compact?: boolean;
  onEntropyUpdate?: (impact: number) => void;
}

export default function SleepTracking({ compact = false, onEntropyUpdate }: SleepTrackingProps) {
  const [activeTab, setActiveTab] = useState<'log' | 'trends' | 'insights'>('log');
  const [sleepEntries, setSleepEntries] = useState<SleepEntry[]>([]);
  const [newEntry, setNewEntry] = useState({
    bedtime: '22:00',
    wakeTime: '07:00',
    quality: 70,
    factors: [] as string[],
    dreams: 'neutral' as SleepEntry['dreams'],
    wakeUps: 0,
    notes: '',
  });

  const sleepFactors = [
    { id: 'caffeine', label: 'Caffeine', icon: Coffee, negative: true },
    { id: 'exercise', label: 'Exercise', icon: Activity, negative: false },
    { id: 'stress', label: 'High Stress', icon: AlertTriangle, negative: true },
    { id: 'meditation', label: 'Meditation', icon: Wind, negative: false },
    { id: 'screens', label: 'Late Screens', icon: Zap, negative: true },
    { id: 'routine', label: 'Consistent Routine', icon: Clock, negative: false },
  ];

  // Calculate sleep duration
  const calculateDuration = (bedtime: string, wakeTime: string): number => {
    const [bedH, bedM] = bedtime.split(':').map(Number);
    const [wakeH, wakeM] = wakeTime.split(':').map(Number);
    
    let bedMinutes = bedH * 60 + bedM;
    let wakeMinutes = wakeH * 60 + wakeM;
    
    if (wakeMinutes < bedMinutes) {
      wakeMinutes += 24 * 60; // Next day
    }
    
    return (wakeMinutes - bedMinutes) / 60;
  };

  // Calculate entropy impact based on sleep quality
  const calculateEntropyImpact = (entry: typeof newEntry): number => {
    let impact = 0;
    const duration = calculateDuration(entry.bedtime, entry.wakeTime);
    
    // Quality impact (-10 to +10)
    impact += (entry.quality - 50) / 5;
    
    // Duration impact
    if (duration < 6) impact += 5; // Too little sleep increases entropy
    else if (duration > 9) impact += 2; // Too much can also be concerning
    else if (duration >= 7 && duration <= 8) impact -= 5; // Optimal
    
    // Wake-ups impact
    impact += entry.wakeUps * 2;
    
    // Dreams impact
    if (entry.dreams === 'nightmares') impact += 5;
    else if (entry.dreams === 'pleasant') impact -= 2;
    
    // Factors impact
    entry.factors.forEach(f => {
      const factor = sleepFactors.find(sf => sf.id === f);
      if (factor) {
        impact += factor.negative ? 2 : -2;
      }
    });
    
    return Math.max(-20, Math.min(20, Math.round(impact)));
  };

  // Mock historical data
  useEffect(() => {
    const mockEntries: SleepEntry[] = [
      {
        id: '1',
        date: new Date(Date.now() - 86400000).toISOString().split('T')[0],
        bedtime: '23:30',
        wakeTime: '07:00',
        duration: 7.5,
        quality: 75,
        factors: ['meditation', 'routine'],
        dreams: 'pleasant',
        wakeUps: 1,
        entropyImpact: -5,
        notes: 'Felt rested',
      },
      {
        id: '2',
        date: new Date(Date.now() - 172800000).toISOString().split('T')[0],
        bedtime: '01:00',
        wakeTime: '06:30',
        duration: 5.5,
        quality: 45,
        factors: ['caffeine', 'screens', 'stress'],
        dreams: 'nightmares',
        wakeUps: 3,
        entropyImpact: 12,
        notes: 'Couldn\'t fall asleep',
      },
      {
        id: '3',
        date: new Date(Date.now() - 259200000).toISOString().split('T')[0],
        bedtime: '22:00',
        wakeTime: '06:00',
        duration: 8,
        quality: 85,
        factors: ['exercise', 'meditation', 'routine'],
        dreams: 'pleasant',
        wakeUps: 0,
        entropyImpact: -8,
        notes: 'Great night!',
      },
    ];
    setSleepEntries(mockEntries);
  }, []);

  const toggleFactor = (factorId: string) => {
    setNewEntry(prev => ({
      ...prev,
      factors: prev.factors.includes(factorId)
        ? prev.factors.filter(f => f !== factorId)
        : [...prev.factors, factorId],
    }));
  };

  const logSleep = () => {
    const duration = calculateDuration(newEntry.bedtime, newEntry.wakeTime);
    const entropyImpact = calculateEntropyImpact(newEntry);
    
    const entry: SleepEntry = {
      id: Date.now().toString(),
      date: new Date().toISOString().split('T')[0],
      bedtime: newEntry.bedtime,
      wakeTime: newEntry.wakeTime,
      duration,
      quality: newEntry.quality,
      factors: newEntry.factors,
      dreams: newEntry.dreams,
      wakeUps: newEntry.wakeUps,
      entropyImpact,
      notes: newEntry.notes,
    };
    
    setSleepEntries(prev => [entry, ...prev]);
    onEntropyUpdate?.(entropyImpact);
    
    // Reset form
    setNewEntry({
      bedtime: '22:00',
      wakeTime: '07:00',
      quality: 70,
      factors: [],
      dreams: 'neutral',
      wakeUps: 0,
      notes: '',
    });
  };

  // Calculate averages
  const avgQuality = sleepEntries.length > 0
    ? Math.round(sleepEntries.reduce((sum, e) => sum + e.quality, 0) / sleepEntries.length)
    : 0;
  const avgDuration = sleepEntries.length > 0
    ? (sleepEntries.reduce((sum, e) => sum + e.duration, 0) / sleepEntries.length).toFixed(1)
    : '0';
  const avgEntropyImpact = sleepEntries.length > 0
    ? Math.round(sleepEntries.reduce((sum, e) => sum + e.entropyImpact, 0) / sleepEntries.length)
    : 0;

  const getQualityColor = (quality: number) => {
    if (quality >= 80) return 'text-green-400';
    if (quality >= 60) return 'text-yellow-400';
    if (quality >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getEntropyColor = (impact: number) => {
    if (impact <= -5) return 'text-green-400';
    if (impact <= 0) return 'text-blue-400';
    if (impact <= 5) return 'text-yellow-400';
    return 'text-red-400';
  };

  if (compact) {
    return (
      <Card className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/20">
        <CardContent className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-indigo-500/20 rounded-lg">
              <Moon className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Sleep Tracking</h3>
              <p className="text-xs text-zinc-400">Quality affects entropy</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <p className={`text-lg font-bold ${getQualityColor(avgQuality)}`}>{avgQuality}%</p>
              <p className="text-xs text-zinc-500">Avg Quality</p>
            </div>
            <div>
              <p className="text-lg font-bold text-indigo-400">{avgDuration}h</p>
              <p className="text-xs text-zinc-500">Avg Duration</p>
            </div>
            <div>
              <p className={`text-lg font-bold ${getEntropyColor(avgEntropyImpact)}`}>
                {avgEntropyImpact > 0 ? '+' : ''}{avgEntropyImpact}
              </p>
              <p className="text-xs text-zinc-500">Entropy</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/20 rounded-lg">
            <Moon className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Sleep Tracking</h2>
            <p className="text-sm text-zinc-400">Monitor sleep quality and its impact on your wellbeing</p>
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-3 gap-3">
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="p-4 text-center">
            <p className={`text-2xl font-bold ${getQualityColor(avgQuality)}`}>{avgQuality}%</p>
            <p className="text-xs text-zinc-500">Avg Quality (7 days)</p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold text-indigo-400">{avgDuration}h</p>
            <p className="text-xs text-zinc-500">Avg Duration</p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="p-4 text-center">
            <div className="flex items-center justify-center gap-1">
              {avgEntropyImpact < 0 ? (
                <TrendingDown className="w-5 h-5 text-green-400" />
              ) : avgEntropyImpact > 0 ? (
                <TrendingUp className="w-5 h-5 text-red-400" />
              ) : (
                <Minus className="w-5 h-5 text-zinc-400" />
              )}
              <p className={`text-2xl font-bold ${getEntropyColor(avgEntropyImpact)}`}>
                {avgEntropyImpact > 0 ? '+' : ''}{avgEntropyImpact}
              </p>
            </div>
            <p className="text-xs text-zinc-500">Entropy Impact</p>
          </CardContent>
        </Card>
      </div>

      {/* Navigation Tabs */}
      <div className="flex gap-2 bg-zinc-900/50 p-1 rounded-lg">
        {[
          { id: 'log', label: 'Log Sleep', icon: Moon },
          { id: 'trends', label: 'Trends', icon: TrendingUp },
          { id: 'insights', label: 'Insights', icon: Brain },
        ].map(tab => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={activeTab === tab.id ? 'bg-indigo-600' : ''}
          >
            <tab.icon className="w-4 h-4 mr-2" />
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Log Sleep Tab */}
      {activeTab === 'log' && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-lg">Log Last Night's Sleep</CardTitle>
            <CardDescription>
              Track your sleep to understand its impact on your mental state
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Time inputs */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-zinc-400 mb-1 block flex items-center gap-2">
                  <Moon className="w-4 h-4" /> Bedtime
                </label>
                <Input
                  type="time"
                  value={newEntry.bedtime}
                  onChange={e => setNewEntry(prev => ({ ...prev, bedtime: e.target.value }))}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
              <div>
                <label className="text-sm text-zinc-400 mb-1 block flex items-center gap-2">
                  <Sun className="w-4 h-4" /> Wake Time
                </label>
                <Input
                  type="time"
                  value={newEntry.wakeTime}
                  onChange={e => setNewEntry(prev => ({ ...prev, wakeTime: e.target.value }))}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
            </div>

            {/* Duration display */}
            <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-indigo-400">
                {calculateDuration(newEntry.bedtime, newEntry.wakeTime).toFixed(1)} hours
              </p>
              <p className="text-xs text-zinc-400">Total Sleep Duration</p>
            </div>

            {/* Quality slider */}
            <div>
              <label className="text-sm text-zinc-400 mb-2 block">
                Sleep Quality: <span className={getQualityColor(newEntry.quality)}>{newEntry.quality}%</span>
              </label>
              <Slider
                value={[newEntry.quality]}
                onValueChange={([value]) => setNewEntry(prev => ({ ...prev, quality: value }))}
                max={100}
                step={5}
                className="py-2"
              />
              <div className="flex justify-between text-xs text-zinc-500">
                <span>Poor</span>
                <span>Fair</span>
                <span>Good</span>
                <span>Excellent</span>
              </div>
            </div>

            {/* Wake-ups */}
            <div>
              <label className="text-sm text-zinc-400 mb-2 block">
                Night Wake-ups: {newEntry.wakeUps}
              </label>
              <div className="flex gap-2">
                {[0, 1, 2, 3, 4, 5].map(num => (
                  <Button
                    key={num}
                    variant={newEntry.wakeUps === num ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setNewEntry(prev => ({ ...prev, wakeUps: num }))}
                    className={newEntry.wakeUps === num ? 'bg-indigo-600' : 'border-zinc-700'}
                  >
                    {num}{num === 5 ? '+' : ''}
                  </Button>
                ))}
              </div>
            </div>

            {/* Dreams */}
            <div>
              <label className="text-sm text-zinc-400 mb-2 block">Dreams</label>
              <div className="flex gap-2 flex-wrap">
                {[
                  { id: 'none', label: 'None', emoji: '😶' },
                  { id: 'pleasant', label: 'Pleasant', emoji: '😊' },
                  { id: 'neutral', label: 'Neutral', emoji: '😐' },
                  { id: 'nightmares', label: 'Nightmares', emoji: '😰' },
                ].map(dream => (
                  <Button
                    key={dream.id}
                    variant={newEntry.dreams === dream.id ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setNewEntry(prev => ({ ...prev, dreams: dream.id as SleepEntry['dreams'] }))}
                    className={newEntry.dreams === dream.id ? 'bg-indigo-600' : 'border-zinc-700'}
                  >
                    {dream.emoji} {dream.label}
                  </Button>
                ))}
              </div>
            </div>

            {/* Factors */}
            <div>
              <label className="text-sm text-zinc-400 mb-2 block">Factors Affecting Sleep</label>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {sleepFactors.map(factor => (
                  <Button
                    key={factor.id}
                    variant={newEntry.factors.includes(factor.id) ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => toggleFactor(factor.id)}
                    className={`justify-start ${
                      newEntry.factors.includes(factor.id)
                        ? factor.negative ? 'bg-red-600' : 'bg-green-600'
                        : 'border-zinc-700'
                    }`}
                  >
                    <factor.icon className="w-4 h-4 mr-2" />
                    {factor.label}
                  </Button>
                ))}
              </div>
            </div>

            {/* Entropy Preview */}
            <div className={`rounded-lg p-3 border ${
              calculateEntropyImpact(newEntry) <= 0 
                ? 'bg-green-500/10 border-green-500/20' 
                : 'bg-red-500/10 border-red-500/20'
            }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-zinc-400" />
                  <span className="text-sm text-zinc-300">Estimated Entropy Impact</span>
                </div>
                <span className={`text-lg font-bold ${getEntropyColor(calculateEntropyImpact(newEntry))}`}>
                  {calculateEntropyImpact(newEntry) > 0 ? '+' : ''}{calculateEntropyImpact(newEntry)}
                </span>
              </div>
              <p className="text-xs text-zinc-500 mt-1">
                {calculateEntropyImpact(newEntry) <= -5 
                  ? 'Great sleep! This will help stabilize your mental state.'
                  : calculateEntropyImpact(newEntry) <= 0
                  ? 'Decent sleep. Neutral impact on your wellbeing.'
                  : calculateEntropyImpact(newEntry) <= 5
                  ? 'Sleep could be better. Consider improving sleep hygiene.'
                  : 'Poor sleep detected. This may increase stress and anxiety.'}
              </p>
            </div>

            <Button onClick={logSleep} className="w-full bg-indigo-600 hover:bg-indigo-700">
              <Moon className="w-4 h-4 mr-2" />
              Log Sleep Entry
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Trends Tab */}
      {activeTab === 'trends' && (
        <div className="space-y-3">
          {sleepEntries.map(entry => (
            <Card key={entry.id} className="bg-zinc-900/50 border-zinc-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-zinc-400" />
                    <span className="text-sm text-zinc-300">
                      {new Date(entry.date).toLocaleDateString('en-US', {
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                  <Badge className={`${
                    entry.entropyImpact <= 0 ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
                  }`}>
                    {entry.entropyImpact > 0 ? '+' : ''}{entry.entropyImpact} entropy
                  </Badge>
                </div>

                <div className="grid grid-cols-4 gap-2 text-center mb-2">
                  <div>
                    <p className="text-lg font-bold text-indigo-400">{entry.duration.toFixed(1)}h</p>
                    <p className="text-xs text-zinc-500">Duration</p>
                  </div>
                  <div>
                    <p className={`text-lg font-bold ${getQualityColor(entry.quality)}`}>{entry.quality}%</p>
                    <p className="text-xs text-zinc-500">Quality</p>
                  </div>
                  <div>
                    <p className="text-lg font-bold text-zinc-300">{entry.wakeUps}</p>
                    <p className="text-xs text-zinc-500">Wake-ups</p>
                  </div>
                  <div>
                    <p className="text-lg">
                      {entry.dreams === 'pleasant' ? '😊' : entry.dreams === 'nightmares' ? '😰' : entry.dreams === 'none' ? '😶' : '😐'}
                    </p>
                    <p className="text-xs text-zinc-500">Dreams</p>
                  </div>
                </div>

                <div className="flex items-center gap-1 text-xs text-zinc-500">
                  <Clock className="w-3 h-3" />
                  {entry.bedtime} → {entry.wakeTime}
                </div>

                {entry.notes && (
                  <p className="text-xs text-zinc-400 mt-2 italic">"{entry.notes}"</p>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Insights Tab */}
      {activeTab === 'insights' && (
        <div className="space-y-3">
          <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-green-300">What's Working</h3>
                  <p className="text-sm text-zinc-300 mt-1">
                    Your best sleep nights include meditation and consistent routines. 
                    Keep up these habits for optimal mental wellness.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border-yellow-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-yellow-300">Areas to Improve</h3>
                  <p className="text-sm text-zinc-300 mt-1">
                    Late caffeine and screen time before bed correlate with higher entropy scores. 
                    Try avoiding screens 1 hour before sleep.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Brain className="w-5 h-5 text-indigo-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-indigo-300">Sleep-Entropy Connection</h3>
                  <p className="text-sm text-zinc-300 mt-1">
                    Your data shows a strong correlation between sleep quality and next-day entropy levels. 
                    Each 10% improvement in sleep quality reduces entropy by approximately 2 points.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardContent className="p-4">
              <h3 className="font-semibold text-white mb-3">Optimal Sleep Profile</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-zinc-400">Ideal Bedtime</span>
                  <span className="text-indigo-400">10:00 PM - 11:00 PM</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-400">Ideal Duration</span>
                  <span className="text-indigo-400">7-8 hours</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-400">Target Quality</span>
                  <span className="text-indigo-400">75%+</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-400">Max Wake-ups</span>
                  <span className="text-indigo-400">1-2</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
