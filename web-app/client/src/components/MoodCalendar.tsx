import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ChevronLeft, ChevronRight, Calendar, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MoodEntry {
  date: string;
  mood: number; // 1-5 scale
  notes?: string;
  entropy?: number;
}

interface MoodCalendarProps {
  entries?: MoodEntry[];
  onDateClick?: (date: string) => void;
}

const MOOD_COLORS = {
  1: 'bg-red-500', // Very low
  2: 'bg-orange-500', // Low
  3: 'bg-yellow-500', // Neutral
  4: 'bg-emerald-400', // Good
  5: 'bg-emerald-600', // Great
};

const MOOD_LABELS = {
  1: 'Crisis',
  2: 'Struggling',
  3: 'Okay',
  4: 'Good',
  5: 'Great',
};

export function MoodCalendar({ entries = [], onDateClick }: MoodCalendarProps) {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [moodData, setMoodData] = useState<Record<string, MoodEntry>>({});

  useEffect(() => {
    // Load mood data from localStorage
    const stored = localStorage.getItem('reunity_mood_calendar');
    if (stored) {
      setMoodData(JSON.parse(stored));
    }
    
    // Merge with provided entries
    if (entries.length > 0) {
      const merged = { ...moodData };
      entries.forEach(entry => {
        merged[entry.date] = entry;
      });
      setMoodData(merged);
    }
  }, [entries]);

  const getDaysInMonth = (date: Date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDay = firstDay.getDay();
    
    return { daysInMonth, startingDay };
  };

  const { daysInMonth, startingDay } = getDaysInMonth(currentDate);

  const navigateMonth = (direction: number) => {
    setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + direction, 1));
  };

  const formatDateKey = (day: number) => {
    const year = currentDate.getFullYear();
    const month = String(currentDate.getMonth() + 1).padStart(2, '0');
    const dayStr = String(day).padStart(2, '0');
    return `${year}-${month}-${dayStr}`;
  };

  const getMoodForDay = (day: number): MoodEntry | undefined => {
    const dateKey = formatDateKey(day);
    return moodData[dateKey];
  };

  const calculateMonthStats = () => {
    let total = 0;
    let count = 0;
    let trend = 0;
    const firstHalf: number[] = [];
    const secondHalf: number[] = [];

    for (let day = 1; day <= daysInMonth; day++) {
      const mood = getMoodForDay(day);
      if (mood) {
        total += mood.mood;
        count++;
        if (day <= 15) {
          firstHalf.push(mood.mood);
        } else {
          secondHalf.push(mood.mood);
        }
      }
    }

    const avgFirst = firstHalf.length > 0 ? firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length : 0;
    const avgSecond = secondHalf.length > 0 ? secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length : 0;
    
    if (avgFirst > 0 && avgSecond > 0) {
      trend = avgSecond - avgFirst;
    }

    return {
      average: count > 0 ? total / count : 0,
      count,
      trend,
    };
  };

  const stats = calculateMonthStats();

  const handleDayClick = (day: number) => {
    const dateKey = formatDateKey(day);
    if (onDateClick) {
      onDateClick(dateKey);
    }
  };

  const monthName = currentDate.toLocaleString('default', { month: 'long', year: 'numeric' });

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Calendar className="w-5 h-5 text-emerald-400" />
            Mood Calendar
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigateMonth(-1)}
              className="h-8 w-8"
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <span className="text-sm font-medium min-w-[140px] text-center">{monthName}</span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigateMonth(1)}
              className="h-8 w-8"
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Day headers */}
        <div className="grid grid-cols-7 gap-1 mb-2">
          {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
            <div key={day} className="text-center text-xs text-zinc-500 py-1">
              {day}
            </div>
          ))}
        </div>

        {/* Calendar grid */}
        <div className="grid grid-cols-7 gap-1">
          {/* Empty cells for days before month starts */}
          {Array.from({ length: startingDay }).map((_, i) => (
            <div key={`empty-${i}`} className="aspect-square" />
          ))}

          {/* Days of the month */}
          {Array.from({ length: daysInMonth }).map((_, i) => {
            const day = i + 1;
            const mood = getMoodForDay(day);
            const isToday = new Date().toDateString() === new Date(currentDate.getFullYear(), currentDate.getMonth(), day).toDateString();

            return (
              <button
                key={day}
                onClick={() => handleDayClick(day)}
                className={`
                  aspect-square rounded-lg flex items-center justify-center text-xs font-medium
                  transition-all hover:scale-105 relative
                  ${isToday ? 'ring-2 ring-emerald-400' : ''}
                  ${mood ? MOOD_COLORS[mood.mood as keyof typeof MOOD_COLORS] : 'bg-zinc-800 hover:bg-zinc-700'}
                  ${mood ? 'text-white' : 'text-zinc-400'}
                `}
              >
                {day}
                {mood && mood.entropy && mood.entropy > 0.7 && (
                  <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-400 rounded-full" />
                )}
              </button>
            );
          })}
        </div>

        {/* Month stats */}
        <div className="mt-4 pt-4 border-t border-zinc-800">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-xs text-zinc-500">Average Mood</p>
              <p className="text-lg font-semibold text-emerald-400">
                {stats.average > 0 ? stats.average.toFixed(1) : '-'}
              </p>
            </div>
            <div>
              <p className="text-xs text-zinc-500">Days Tracked</p>
              <p className="text-lg font-semibold text-white">{stats.count}</p>
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
        </div>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-2 justify-center">
          {Object.entries(MOOD_LABELS).map(([level, label]) => (
            <div key={level} className="flex items-center gap-1">
              <div className={`w-3 h-3 rounded ${MOOD_COLORS[parseInt(level) as keyof typeof MOOD_COLORS]}`} />
              <span className="text-xs text-zinc-400">{label}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export default MoodCalendar;
