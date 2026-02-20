import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Moon, Sun, TrendingUp, TrendingDown, Minus, Clock, Zap } from 'lucide-react';
import { toast } from 'sonner';

interface SleepEntry {
  date: string;
  bedtime: string;
  wakeTime: string;
  quality: number; // 1-5
  notes?: string;
  duration: number; // minutes
}

export function SleepTracker() {
  const [entries, setEntries] = useState<SleepEntry[]>([]);
  const [isLogging, setIsLogging] = useState(false);
  const [newEntry, setNewEntry] = useState({
    bedtime: '22:00',
    wakeTime: '06:00',
    quality: 3,
    notes: '',
  });

  useEffect(() => {
    const saved = localStorage.getItem('reunity_sleep_entries');
    if (saved) {
      setEntries(JSON.parse(saved));
    }
  }, []);

  const saveEntries = (newEntries: SleepEntry[]) => {
    setEntries(newEntries);
    localStorage.setItem('reunity_sleep_entries', JSON.stringify(newEntries));
  };

  const calculateDuration = (bedtime: string, wakeTime: string): number => {
    const [bedHour, bedMin] = bedtime.split(':').map(Number);
    const [wakeHour, wakeMin] = wakeTime.split(':').map(Number);
    
    let bedMinutes = bedHour * 60 + bedMin;
    let wakeMinutes = wakeHour * 60 + wakeMin;
    
    // Handle overnight sleep
    if (wakeMinutes < bedMinutes) {
      wakeMinutes += 24 * 60;
    }
    
    return wakeMinutes - bedMinutes;
  };

  const formatDuration = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const logSleep = () => {
    const today = new Date().toISOString().split('T')[0];
    const duration = calculateDuration(newEntry.bedtime, newEntry.wakeTime);
    
    const entry: SleepEntry = {
      date: today,
      bedtime: newEntry.bedtime,
      wakeTime: newEntry.wakeTime,
      quality: newEntry.quality,
      notes: newEntry.notes,
      duration,
    };

    // Remove existing entry for today if any
    const filtered = entries.filter(e => e.date !== today);
    saveEntries([...filtered, entry]);
    
    setIsLogging(false);
    setNewEntry({ bedtime: '22:00', wakeTime: '06:00', quality: 3, notes: '' });
    toast.success('Sleep logged successfully');
  };

  const getWeekStats = () => {
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    const weekEntries = entries.filter(e => new Date(e.date) >= weekAgo);
    
    if (weekEntries.length === 0) {
      return { avgDuration: 0, avgQuality: 0, trend: 0, count: 0 };
    }

    const avgDuration = weekEntries.reduce((sum, e) => sum + e.duration, 0) / weekEntries.length;
    const avgQuality = weekEntries.reduce((sum, e) => sum + e.quality, 0) / weekEntries.length;
    
    // Calculate trend (compare first half to second half of week)
    const firstHalf = weekEntries.slice(0, Math.ceil(weekEntries.length / 2));
    const secondHalf = weekEntries.slice(Math.ceil(weekEntries.length / 2));
    
    const firstAvg = firstHalf.length > 0 ? firstHalf.reduce((sum, e) => sum + e.quality, 0) / firstHalf.length : 0;
    const secondAvg = secondHalf.length > 0 ? secondHalf.reduce((sum, e) => sum + e.quality, 0) / secondHalf.length : 0;
    
    return {
      avgDuration,
      avgQuality,
      trend: secondAvg - firstAvg,
      count: weekEntries.length,
    };
  };

  const stats = getWeekStats();
  const todayEntry = entries.find(e => e.date === new Date().toISOString().split('T')[0]);

  const qualityLabels = ['', 'Poor', 'Fair', 'Okay', 'Good', 'Great'];
  const qualityColors = ['', 'text-red-400', 'text-orange-400', 'text-yellow-400', 'text-emerald-400', 'text-emerald-500'];

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Moon className="w-5 h-5 text-indigo-400" />
            Sleep Tracker
          </CardTitle>
          {!isLogging && !todayEntry && (
            <Button variant="outline" size="sm" onClick={() => setIsLogging(true)}>
              Log Sleep
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isLogging ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-zinc-400 flex items-center gap-2">
                  <Moon className="w-4 h-4" /> Bedtime
                </label>
                <Input
                  type="time"
                  value={newEntry.bedtime}
                  onChange={e => setNewEntry({ ...newEntry, bedtime: e.target.value })}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm text-zinc-400 flex items-center gap-2">
                  <Sun className="w-4 h-4" /> Wake Time
                </label>
                <Input
                  type="time"
                  value={newEntry.wakeTime}
                  onChange={e => setNewEntry({ ...newEntry, wakeTime: e.target.value })}
                  className="mt-1"
                />
              </div>
            </div>

            <div>
              <label className="text-sm text-zinc-400">Sleep Quality</label>
              <div className="flex gap-2 mt-2">
                {[1, 2, 3, 4, 5].map(q => (
                  <button
                    key={q}
                    onClick={() => setNewEntry({ ...newEntry, quality: q })}
                    className={`
                      flex-1 py-2 rounded-lg text-sm font-medium transition-all
                      ${newEntry.quality === q 
                        ? 'bg-indigo-500 text-white' 
                        : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}
                    `}
                  >
                    {qualityLabels[q]}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-sm text-zinc-400">Notes (optional)</label>
              <Input
                value={newEntry.notes}
                onChange={e => setNewEntry({ ...newEntry, notes: e.target.value })}
                placeholder="e.g., Woke up once, had vivid dreams"
                className="mt-1"
              />
            </div>

            <div className="flex gap-2">
              <Button onClick={logSleep} className="flex-1">
                Save
              </Button>
              <Button variant="outline" onClick={() => setIsLogging(false)}>
                Cancel
              </Button>
            </div>
          </div>
        ) : todayEntry ? (
          <div className="space-y-4">
            <div className="bg-indigo-900/20 rounded-xl p-4 border border-indigo-800/30">
              <p className="text-sm text-zinc-400 mb-2">Last Night's Sleep</p>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">{formatDuration(todayEntry.duration)}</p>
                    <p className="text-xs text-zinc-500">Duration</p>
                  </div>
                  <div className="text-center">
                    <p className={`text-2xl font-bold ${qualityColors[todayEntry.quality]}`}>
                      {qualityLabels[todayEntry.quality]}
                    </p>
                    <p className="text-xs text-zinc-500">Quality</p>
                  </div>
                </div>
                <div className="text-right text-sm text-zinc-400">
                  <p>{todayEntry.bedtime} → {todayEntry.wakeTime}</p>
                </div>
              </div>
              {todayEntry.notes && (
                <p className="text-sm text-zinc-400 mt-2 pt-2 border-t border-zinc-700/50">
                  {todayEntry.notes}
                </p>
              )}
            </div>

            {/* Week stats */}
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-xs text-zinc-500">Avg Duration</p>
                <p className="text-lg font-semibold text-white">
                  {stats.avgDuration > 0 ? formatDuration(Math.round(stats.avgDuration)) : '-'}
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Avg Quality</p>
                <p className={`text-lg font-semibold ${qualityColors[Math.round(stats.avgQuality)] || 'text-zinc-400'}`}>
                  {stats.avgQuality > 0 ? stats.avgQuality.toFixed(1) : '-'}
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500">Trend</p>
                <div className="flex items-center justify-center gap-1">
                  {stats.trend > 0.2 ? (
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                  ) : stats.trend < -0.2 ? (
                    <TrendingDown className="w-4 h-4 text-red-400" />
                  ) : (
                    <Minus className="w-4 h-4 text-zinc-400" />
                  )}
                  <span className={`text-lg font-semibold ${
                    stats.trend > 0.2 ? 'text-emerald-400' : 
                    stats.trend < -0.2 ? 'text-red-400' : 'text-zinc-400'
                  }`}>
                    {stats.trend > 0 ? '+' : ''}{stats.trend.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            {/* Sleep tips based on data */}
            {stats.avgDuration > 0 && stats.avgDuration < 420 && (
              <div className="bg-amber-900/20 rounded-lg p-3 border border-amber-800/30">
                <div className="flex items-start gap-2">
                  <Zap className="w-4 h-4 text-amber-400 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-amber-400">Sleep Tip</p>
                    <p className="text-xs text-zinc-400">
                      You're averaging less than 7 hours. Try going to bed 30 minutes earlier tonight.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8 text-zinc-500">
            <Moon className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No sleep logged today</p>
            <p className="text-sm">Track your sleep to see patterns</p>
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={() => setIsLogging(true)}
            >
              Log Last Night's Sleep
            </Button>
          </div>
        )}

        {/* Recent entries */}
        {entries.length > 0 && !isLogging && (
          <div className="mt-4 pt-4 border-t border-zinc-800">
            <p className="text-sm text-zinc-400 mb-2">Recent Nights</p>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {entries
                .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
                .slice(0, 5)
                .map(entry => (
                  <div
                    key={entry.date}
                    className="flex items-center justify-between text-sm p-2 bg-zinc-800/50 rounded-lg"
                  >
                    <span className="text-zinc-400">
                      {new Date(entry.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                    </span>
                    <div className="flex items-center gap-3">
                      <span className="text-white">{formatDuration(entry.duration)}</span>
                      <span className={qualityColors[entry.quality]}>{qualityLabels[entry.quality]}</span>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default SleepTracker;
