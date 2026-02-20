import React, { useState, useMemo } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface CrisisEvent {
  id: string;
  date: Date;
  severity: 'low' | 'moderate' | 'high' | 'crisis';
  entropyScore: number;
  triggers: string[];
  duration: number;
  timeOfDay: string;
}

const generateMockEvents = (): CrisisEvent[] => {
  const events: CrisisEvent[] = [];
  const triggers = ['Work stress', 'Family conflict', 'Sleep deprivation', 'Social isolation', 'Financial worry'];
  
  for (let i = 0; i < 20; i++) {
    if (Math.random() < 0.5) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const hour = Math.floor(Math.random() * 24);
      date.setHours(hour);
      
      const severity = Math.random() < 0.1 ? 'crisis' : 
                      Math.random() < 0.3 ? 'high' :
                      Math.random() < 0.6 ? 'moderate' : 'low';
      
      events.push({
        id: `event-${i}`,
        date,
        severity,
        entropyScore: severity === 'crisis' ? 85 + Math.random() * 15 :
                      severity === 'high' ? 65 + Math.random() * 20 :
                      severity === 'moderate' ? 40 + Math.random() * 25 :
                      20 + Math.random() * 20,
        triggers: [triggers[Math.floor(Math.random() * triggers.length)]],
        duration: Math.floor(15 + Math.random() * 120),
        timeOfDay: hour < 6 ? 'night' : hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening',
      });
    }
  }
  
  return events.sort((a, b) => b.date.getTime() - a.date.getTime());
};

export default function CrisisInterventionTimeline() {
  const [events] = useState<CrisisEvent[]>(generateMockEvents);
  const [timeRange, setTimeRange] = useState<'week' | 'month'>('month');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);

  const filteredEvents = useMemo(() => {
    const now = new Date();
    const cutoff = new Date();
    cutoff.setDate(now.getDate() - (timeRange === 'week' ? 7 : 30));
    return events.filter(e => e.date >= cutoff);
  }, [events, timeRange]);

  const stats = useMemo(() => {
    const crisisCount = filteredEvents.filter(e => e.severity === 'crisis').length;
    const highCount = filteredEvents.filter(e => e.severity === 'high').length;
    const avgEntropy = filteredEvents.length > 0
      ? Math.round(filteredEvents.reduce((sum, e) => sum + e.entropyScore, 0) / filteredEvents.length)
      : 0;
    return { crisisCount, highCount, avgEntropy, total: filteredEvents.length };
  }, [filteredEvents]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return '#22C55E';
      case 'moderate': return '#EAB308';
      case 'high': return '#F97316';
      case 'crisis': return '#EF4444';
      default: return '#6B7280';
    }
  };

  const getTimeAgo = (date: Date) => {
    const days = Math.floor((Date.now() - date.getTime()) / (1000 * 60 * 60 * 24));
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    return `${days} days ago`;
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="analytics" size={24} color="#3B82F6" />
        <Text style={styles.title}>Crisis Timeline</Text>
      </View>
      <Text style={styles.subtitle}>Track patterns and identify triggers</Text>

      {/* Time Range Toggle */}
      <View style={styles.toggleContainer}>
        <TouchableOpacity
          style={[styles.toggleButton, timeRange === 'week' && styles.toggleActive]}
          onPress={() => setTimeRange('week')}
        >
          <Text style={[styles.toggleText, timeRange === 'week' && styles.toggleTextActive]}>7 Days</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.toggleButton, timeRange === 'month' && styles.toggleActive]}
          onPress={() => setTimeRange('month')}
        >
          <Text style={[styles.toggleText, timeRange === 'month' && styles.toggleTextActive]}>30 Days</Text>
        </TouchableOpacity>
      </View>

      {/* Stats */}
      <View style={styles.statsContainer}>
        <View style={styles.statCard}>
          <Text style={styles.statValue}>{stats.total}</Text>
          <Text style={styles.statLabel}>Events</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#EF4444' }]}>{stats.crisisCount}</Text>
          <Text style={styles.statLabel}>Crisis</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={[styles.statValue, { color: '#A855F7' }]}>{stats.avgEntropy}</Text>
          <Text style={styles.statLabel}>Avg Entropy</Text>
        </View>
      </View>

      {/* Timeline */}
      <View style={styles.timeline}>
        {filteredEvents.map((event) => (
          <TouchableOpacity
            key={event.id}
            style={[styles.eventCard, { borderLeftColor: getSeverityColor(event.severity) }]}
            onPress={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
          >
            <View style={styles.eventHeader}>
              <View style={styles.eventInfo}>
                <Text style={styles.eventDate}>{getTimeAgo(event.date)}</Text>
                <Text style={styles.eventTime}>{event.timeOfDay}</Text>
              </View>
              <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(event.severity) + '30' }]}>
                <Text style={[styles.severityText, { color: getSeverityColor(event.severity) }]}>
                  {event.severity}
                </Text>
              </View>
            </View>
            
            {expandedEvent === event.id && (
              <View style={styles.eventDetails}>
                <View style={styles.detailRow}>
                  <Ionicons name="flash" size={16} color="#EAB308" />
                  <Text style={styles.detailText}>Trigger: {event.triggers.join(', ')}</Text>
                </View>
                <View style={styles.detailRow}>
                  <Ionicons name="time" size={16} color="#3B82F6" />
                  <Text style={styles.detailText}>Duration: {event.duration} minutes</Text>
                </View>
                <View style={styles.detailRow}>
                  <Ionicons name="pulse" size={16} color="#A855F7" />
                  <Text style={styles.detailText}>Entropy: {Math.round(event.entropyScore)}</Text>
                </View>
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {filteredEvents.length === 0 && (
        <View style={styles.emptyState}>
          <Ionicons name="analytics-outline" size={48} color="#4B5563" />
          <Text style={styles.emptyText}>No events in this period</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', padding: 16 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  title: { fontSize: 20, fontWeight: 'bold', color: '#F8FAFC' },
  subtitle: { fontSize: 14, color: '#94A3B8', marginTop: 4, marginBottom: 16 },
  toggleContainer: { flexDirection: 'row', gap: 8, marginBottom: 16 },
  toggleButton: { flex: 1, paddingVertical: 8, alignItems: 'center', backgroundColor: '#1E293B', borderRadius: 8 },
  toggleActive: { backgroundColor: '#3B82F6' },
  toggleText: { color: '#94A3B8', fontWeight: '600' },
  toggleTextActive: { color: '#FFFFFF' },
  statsContainer: { flexDirection: 'row', gap: 8, marginBottom: 20 },
  statCard: { flex: 1, backgroundColor: '#1E293B', padding: 12, borderRadius: 8, alignItems: 'center' },
  statValue: { fontSize: 24, fontWeight: 'bold', color: '#3B82F6' },
  statLabel: { fontSize: 12, color: '#94A3B8', marginTop: 4 },
  timeline: { gap: 8 },
  eventCard: { backgroundColor: '#1E293B', borderRadius: 8, padding: 12, borderLeftWidth: 3 },
  eventHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  eventInfo: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  eventDate: { fontSize: 14, color: '#F8FAFC', fontWeight: '500' },
  eventTime: { fontSize: 12, color: '#94A3B8', textTransform: 'capitalize' },
  severityBadge: { paddingVertical: 4, paddingHorizontal: 8, borderRadius: 4 },
  severityText: { fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },
  eventDetails: { marginTop: 12, paddingTop: 12, borderTopWidth: 1, borderTopColor: '#334155', gap: 8 },
  detailRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  detailText: { fontSize: 14, color: '#CBD5E1' },
  emptyState: { alignItems: 'center', paddingVertical: 40 },
  emptyText: { fontSize: 14, color: '#6B7280', marginTop: 8 },
});
