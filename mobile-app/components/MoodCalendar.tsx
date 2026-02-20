import React, { useState, useMemo } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';

interface MoodEntry {
  date: string;
  mood: number;
  note?: string;
}

const moodColors: Record<number, string> = {
  1: '#ef4444',
  2: '#f97316',
  3: '#eab308',
  4: '#22c55e',
  5: '#10b981',
};

const moodLabels: Record<number, string> = {
  1: 'Very Low',
  2: 'Low',
  3: 'Neutral',
  4: 'Good',
  5: 'Great',
};

// Generate mock data
const generateMockData = (): MoodEntry[] => {
  const data: MoodEntry[] = [];
  const today = new Date();
  for (let i = 60; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    if (Math.random() > 0.15) {
      const baseMood = 3 + Math.sin(i / 7) * 1.5;
      const mood = Math.max(1, Math.min(5, Math.round(baseMood + (Math.random() - 0.5) * 2)));
      data.push({
        date: date.toISOString().split('T')[0],
        mood,
        note: mood <= 2 ? 'Difficult day' : mood >= 4 ? 'Feeling good' : undefined,
      });
    }
  }
  return data;
};

export default function MoodCalendar() {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [moodData] = useState<MoodEntry[]>(generateMockData);

  const moodMap = useMemo(() => {
    const map = new Map<string, MoodEntry>();
    moodData.forEach(entry => map.set(entry.date, entry));
    return map;
  }, [moodData]);

  const calendarDays = useMemo(() => {
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const startPadding = firstDay.getDay();
    const days: (Date | null)[] = [];

    for (let i = 0; i < startPadding; i++) {
      days.push(null);
    }

    for (let i = 1; i <= lastDay.getDate(); i++) {
      days.push(new Date(year, month, i));
    }

    return days;
  }, [currentDate]);

  const monthStats = useMemo(() => {
    const year = currentDate.getFullYear();
    const month = currentDate.getMonth();
    const monthEntries = moodData.filter(entry => {
      const entryDate = new Date(entry.date);
      return entryDate.getFullYear() === year && entryDate.getMonth() === month;
    });

    if (monthEntries.length === 0) {
      return { average: 0, goodDays: 0, lowDays: 0 };
    }

    const average = monthEntries.reduce((sum, e) => sum + e.mood, 0) / monthEntries.length;
    const goodDays = monthEntries.filter(e => e.mood >= 4).length;
    const lowDays = monthEntries.filter(e => e.mood <= 2).length;

    return { average, goodDays, lowDays };
  }, [currentDate, moodData]);

  const navigateMonth = (direction: number) => {
    const newDate = new Date(currentDate);
    newDate.setMonth(newDate.getMonth() + direction);
    setCurrentDate(newDate);
  };

  const selectedEntry = selectedDate ? moodMap.get(selectedDate) : null;

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>Mood Calendar</Text>
      <Text style={styles.subtitle}>Track your emotional patterns</Text>

      {/* Stats */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>{monthStats.average.toFixed(1)}</Text>
          <Text style={styles.statLabel}>Avg Mood</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#22c55e' }]}>{monthStats.goodDays}</Text>
          <Text style={styles.statLabel}>Good Days</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#ef4444' }]}>{monthStats.lowDays}</Text>
          <Text style={styles.statLabel}>Low Days</Text>
        </View>
      </View>

      {/* Month Navigation */}
      <View style={styles.monthNav}>
        <TouchableOpacity onPress={() => navigateMonth(-1)} style={styles.navButton}>
          <Text style={styles.navButtonText}>←</Text>
        </TouchableOpacity>
        <Text style={styles.monthTitle}>
          {currentDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
        </Text>
        <TouchableOpacity onPress={() => navigateMonth(1)} style={styles.navButton}>
          <Text style={styles.navButtonText}>→</Text>
        </TouchableOpacity>
      </View>

      {/* Day Headers */}
      <View style={styles.dayHeaders}>
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
          <Text key={day} style={styles.dayHeader}>{day}</Text>
        ))}
      </View>

      {/* Calendar Grid */}
      <View style={styles.calendarGrid}>
        {calendarDays.map((date, i) => {
          if (!date) {
            return <View key={i} style={styles.emptyDay} />;
          }

          const dateStr = date.toISOString().split('T')[0];
          const entry = moodMap.get(dateStr);
          const isToday = dateStr === new Date().toISOString().split('T')[0];
          const isSelected = dateStr === selectedDate;

          return (
            <TouchableOpacity
              key={i}
              onPress={() => setSelectedDate(dateStr)}
              style={[
                styles.dayCell,
                { backgroundColor: entry ? moodColors[entry.mood] : '#333' },
                isToday && styles.todayCell,
                isSelected && styles.selectedCell,
              ]}
            >
              <Text style={[styles.dayText, { color: entry ? '#fff' : '#888' }]}>
                {date.getDate()}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Selected Day Details */}
      {selectedEntry && (
        <View style={styles.detailCard}>
          <Text style={styles.detailDate}>
            {new Date(selectedDate!).toLocaleDateString('en-US', { 
              weekday: 'long', month: 'long', day: 'numeric' 
            })}
          </Text>
          <View style={[styles.moodBadge, { backgroundColor: moodColors[selectedEntry.mood] }]}>
            <Text style={styles.moodBadgeText}>{moodLabels[selectedEntry.mood]}</Text>
          </View>
          {selectedEntry.note && (
            <Text style={styles.noteText}>"{selectedEntry.note}"</Text>
          )}
        </View>
      )}

      {/* Legend */}
      <View style={styles.legend}>
        {Object.entries(moodLabels).map(([level, label]) => (
          <View key={level} style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: moodColors[parseInt(level)] }]} />
            <Text style={styles.legendText}>{label}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    padding: 20,
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 20,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  monthNav: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  navButton: {
    padding: 12,
  },
  navButtonText: {
    fontSize: 20,
    color: '#fff',
  },
  monthTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  dayHeaders: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  dayHeader: {
    flex: 1,
    textAlign: 'center',
    fontSize: 12,
    color: '#888',
  },
  calendarGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  emptyDay: {
    width: '14.28%',
    aspectRatio: 1,
  },
  dayCell: {
    width: '14.28%',
    aspectRatio: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
    marginBottom: 4,
  },
  todayCell: {
    borderWidth: 2,
    borderColor: '#fff',
  },
  selectedCell: {
    borderWidth: 2,
    borderColor: '#8b5cf6',
  },
  dayText: {
    fontSize: 14,
    fontWeight: '500',
  },
  detailCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginTop: 20,
  },
  detailDate: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 12,
  },
  moodBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    marginBottom: 12,
  },
  moodBadgeText: {
    color: '#fff',
    fontWeight: '600',
  },
  noteText: {
    fontSize: 14,
    color: '#888',
    fontStyle: 'italic',
  },
  legend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 20,
    paddingBottom: 40,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 4,
  },
  legendText: {
    fontSize: 12,
    color: '#888',
  },
});
