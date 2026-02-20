import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView, ActivityIndicator } from 'react-native';

interface TherapistNote {
  id: string;
  therapistName: string;
  sessionDate: string;
  type: 'session' | 'progress' | 'support_plan' | 'crisis';
  title: string;
  content: string;
  isSharedWithClient: boolean;
  includeInReport: boolean;
  tags: string[];
}

const mockNotes: TherapistNote[] = [
  {
    id: '1',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2026-01-20',
    type: 'session',
    title: 'Weekly Check-in Session',
    content: 'Client showed improved coping strategies. Discussed grounding techniques and their application during anxiety episodes.',
    isSharedWithClient: true,
    includeInReport: true,
    tags: ['anxiety', 'coping', 'progress'],
  },
  {
    id: '2',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2026-01-13',
    type: 'progress',
    title: 'Monthly Progress Review',
    content: 'Significant improvement in mood stability over the past month. Entropy scores trending downward.',
    isSharedWithClient: true,
    includeInReport: true,
    tags: ['progress', 'mood'],
  },
  {
    id: '3',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2026-01-06',
    type: 'support_plan',
    title: 'Updated Wellness Goals',
    content: 'Goals for Q1 2026: Reduce anxiety episodes, establish consistent sleep schedule, build support network.',
    isSharedWithClient: true,
    includeInReport: true,
    tags: ['goals', 'wellness'],
  },
];

const typeColors: Record<string, string> = {
  session: '#10b981',
  progress: '#6366f1',
  support_plan: '#8b5cf6',
  crisis: '#ef4444',
};

const typeLabels: Record<string, string> = {
  session: 'Session Note',
  progress: 'Progress Review',
  support_plan: 'Support Plan',
  crisis: 'Crisis Note',
};

export default function TherapistNotesSync() {
  const [notes, setNotes] = useState<TherapistNote[]>(mockNotes);
  const [selectedNote, setSelectedNote] = useState<TherapistNote | null>(null);
  const [syncing, setSyncing] = useState(false);

  const handleSync = async () => {
    setSyncing(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setSyncing(false);
  };

  const toggleIncludeInReport = (noteId: string) => {
    setNotes(prev =>
      prev.map(note =>
        note.id === noteId ? { ...note, includeInReport: !note.includeInReport } : note
      )
    );
  };

  const notesForReport = notes.filter(n => n.includeInReport && n.isSharedWithClient);

  if (selectedNote) {
    return (
      <ScrollView style={styles.container}>
        <TouchableOpacity onPress={() => setSelectedNote(null)} style={styles.backButton}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>

        <Text style={styles.noteTitle}>{selectedNote.title}</Text>
        
        <View style={styles.metaRow}>
          <View style={[styles.typeBadge, { backgroundColor: typeColors[selectedNote.type] + '30' }]}>
            <Text style={[styles.typeBadgeText, { color: typeColors[selectedNote.type] }]}>
              {typeLabels[selectedNote.type]}
            </Text>
          </View>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Therapist:</Text>
          <Text style={styles.infoValue}>{selectedNote.therapistName}</Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Date:</Text>
          <Text style={styles.infoValue}>
            {new Date(selectedNote.sessionDate).toLocaleDateString()}
          </Text>
        </View>

        <View style={styles.contentBox}>
          <Text style={styles.contentText}>{selectedNote.content}</Text>
        </View>

        <View style={styles.tagsRow}>
          {selectedNote.tags.map(tag => (
            <View key={tag} style={styles.tag}>
              <Text style={styles.tagText}>#{tag}</Text>
            </View>
          ))}
        </View>

        <TouchableOpacity
          style={[
            styles.reportButton,
            selectedNote.includeInReport && styles.reportButtonActive,
          ]}
          onPress={() => toggleIncludeInReport(selectedNote.id)}
          disabled={!selectedNote.isSharedWithClient}
        >
          <Text style={styles.reportButtonText}>
            {selectedNote.includeInReport ? '✓ Included in Report' : 'Add to Report'}
          </Text>
        </TouchableOpacity>
      </ScrollView>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>Therapist Notes</Text>
      <Text style={styles.subtitle}>Session notes synced to your wellness reports</Text>

      {/* Sync Button */}
      <TouchableOpacity style={styles.syncButton} onPress={handleSync} disabled={syncing}>
        {syncing ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.syncButtonText}>↻ Sync Now</Text>
        )}
      </TouchableOpacity>

      {/* Report Summary */}
      <View style={styles.summaryCard}>
        <Text style={styles.summaryTitle}>Wellness Report</Text>
        <Text style={styles.summaryText}>
          {notesForReport.length} notes will be included in your next export
        </Text>
      </View>

      {/* Notes List */}
      {notes.filter(n => n.isSharedWithClient).map(note => (
        <TouchableOpacity
          key={note.id}
          style={styles.noteCard}
          onPress={() => setSelectedNote(note)}
        >
          <View style={styles.noteHeader}>
            <View style={[styles.typeBadge, { backgroundColor: typeColors[note.type] + '30' }]}>
              <Text style={[styles.typeBadgeText, { color: typeColors[note.type] }]}>
                {typeLabels[note.type]}
              </Text>
            </View>
            {note.includeInReport && (
              <View style={styles.reportIndicator}>
                <Text style={styles.reportIndicatorText}>📋</Text>
              </View>
            )}
          </View>
          <Text style={styles.noteCardTitle}>{note.title}</Text>
          <Text style={styles.noteCardMeta}>
            {note.therapistName} • {new Date(note.sessionDate).toLocaleDateString()}
          </Text>
        </TouchableOpacity>
      ))}
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
  syncButton: {
    backgroundColor: '#6366f1',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 16,
  },
  syncButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  summaryCard: {
    backgroundColor: '#6366f120',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#6366f140',
  },
  summaryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  summaryText: {
    fontSize: 14,
    color: '#888',
  },
  noteCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  noteHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  typeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  typeBadgeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  reportIndicator: {
    width: 24,
    height: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  reportIndicatorText: {
    fontSize: 14,
  },
  noteCardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  noteCardMeta: {
    fontSize: 12,
    color: '#888',
  },
  backButton: {
    marginBottom: 20,
  },
  backButtonText: {
    color: '#6366f1',
    fontSize: 16,
  },
  noteTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  metaRow: {
    marginBottom: 16,
  },
  infoRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  infoLabel: {
    fontSize: 14,
    color: '#888',
    width: 80,
  },
  infoValue: {
    fontSize: 14,
    color: '#fff',
    flex: 1,
  },
  contentBox: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    marginBottom: 16,
  },
  contentText: {
    fontSize: 14,
    color: '#ccc',
    lineHeight: 22,
  },
  tagsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 20,
  },
  tag: {
    backgroundColor: '#333',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
  },
  tagText: {
    fontSize: 12,
    color: '#888',
  },
  reportButton: {
    backgroundColor: '#333',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 40,
  },
  reportButtonActive: {
    backgroundColor: '#6366f1',
  },
  reportButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
});
