import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Modal, TextInput, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface GroupSession {
  id: string;
  title: string;
  description: string;
  therapistName: string;
  type: 'support' | 'psychoeducation' | 'skills' | 'process';
  topic: string;
  maxParticipants: number;
  currentParticipants: number;
  scheduledDate: string;
  scheduledTime: string;
  duration: number;
  status: 'scheduled' | 'in-progress' | 'completed';
}

const SESSION_TYPES = {
  support: { label: 'Support Group', color: '#3b82f6' },
  psychoeducation: { label: 'Psychoeducation', color: '#8b5cf6' },
  skills: { label: 'Skills Training', color: '#22c55e' },
  process: { label: 'Process Group', color: '#f97316' },
};

export default function GroupTherapySessions() {
  const [sessions, setSessions] = useState<GroupSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<GroupSession | null>(null);

  useEffect(() => {
    // Demo sessions
    setSessions([
      {
        id: '1',
        title: 'Anxiety Support Circle',
        description: 'A safe space to share experiences and learn coping strategies.',
        therapistName: 'Dr. Sarah Chen',
        type: 'support',
        topic: 'Anxiety Management',
        maxParticipants: 10,
        currentParticipants: 6,
        scheduledDate: new Date(Date.now() + 86400000).toISOString().split('T')[0],
        scheduledTime: '18:00',
        duration: 90,
        status: 'scheduled',
      },
      {
        id: '2',
        title: 'DBT Skills Workshop',
        description: 'Learn and practice DBT skills for emotional regulation.',
        therapistName: 'Dr. Michael Torres',
        type: 'skills',
        topic: 'Stress Management',
        maxParticipants: 12,
        currentParticipants: 8,
        scheduledDate: new Date(Date.now() + 172800000).toISOString().split('T')[0],
        scheduledTime: '14:00',
        duration: 120,
        status: 'scheduled',
      },
    ]);
  }, []);

  const joinSession = (sessionId: string) => {
    Alert.alert(
      'Join Session',
      'You will be notified when the session is about to start.',
      [{ text: 'OK' }]
    );
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="people" size={28} color="#10b981" />
        <Text style={styles.title}>Group Sessions</Text>
      </View>
      <Text style={styles.subtitle}>Join supportive group therapy sessions</Text>

      {sessions.map((session) => (
        <TouchableOpacity
          key={session.id}
          style={styles.sessionCard}
          onPress={() => setSelectedSession(session)}
        >
          <View style={styles.sessionHeader}>
            <View style={[styles.typeBadge, { backgroundColor: SESSION_TYPES[session.type].color + '30' }]}>
              <Text style={[styles.typeText, { color: SESSION_TYPES[session.type].color }]}>
                {SESSION_TYPES[session.type].label}
              </Text>
            </View>
          </View>
          
          <Text style={styles.sessionTitle}>{session.title}</Text>
          <Text style={styles.sessionDescription}>{session.description}</Text>
          
          <View style={styles.sessionMeta}>
            <View style={styles.metaItem}>
              <Ionicons name="calendar-outline" size={16} color="#9ca3af" />
              <Text style={styles.metaText}>{formatDate(session.scheduledDate)}</Text>
            </View>
            <View style={styles.metaItem}>
              <Ionicons name="time-outline" size={16} color="#9ca3af" />
              <Text style={styles.metaText}>{session.scheduledTime}</Text>
            </View>
          </View>

          <View style={styles.sessionFooter}>
            <View style={styles.participantInfo}>
              <Ionicons name="people-outline" size={16} color="#9ca3af" />
              <Text style={styles.participantText}>
                {session.currentParticipants}/{session.maxParticipants}
              </Text>
            </View>
            <TouchableOpacity
              style={styles.joinButton}
              onPress={() => joinSession(session.id)}
            >
              <Text style={styles.joinButtonText}>Join</Text>
            </TouchableOpacity>
          </View>
        </TouchableOpacity>
      ))}

      <Modal
        visible={!!selectedSession}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setSelectedSession(null)}
      >
        {selectedSession && (
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setSelectedSession(null)}
              >
                <Ionicons name="close" size={24} color="#fff" />
              </TouchableOpacity>
              
              <View style={[styles.typeBadge, { backgroundColor: SESSION_TYPES[selectedSession.type].color + '30', alignSelf: 'flex-start' }]}>
                <Text style={[styles.typeText, { color: SESSION_TYPES[selectedSession.type].color }]}>
                  {SESSION_TYPES[selectedSession.type].label}
                </Text>
              </View>
              
              <Text style={styles.modalTitle}>{selectedSession.title}</Text>
              <Text style={styles.modalDescription}>{selectedSession.description}</Text>
              
              <View style={styles.modalMeta}>
                <View style={styles.metaRow}>
                  <Ionicons name="person" size={20} color="#10b981" />
                  <Text style={styles.modalMetaText}>{selectedSession.therapistName}</Text>
                </View>
                <View style={styles.metaRow}>
                  <Ionicons name="calendar" size={20} color="#10b981" />
                  <Text style={styles.modalMetaText}>
                    {formatDate(selectedSession.scheduledDate)} at {selectedSession.scheduledTime}
                  </Text>
                </View>
                <View style={styles.metaRow}>
                  <Ionicons name="time" size={20} color="#10b981" />
                  <Text style={styles.modalMetaText}>{selectedSession.duration} minutes</Text>
                </View>
              </View>

              <TouchableOpacity
                style={styles.modalJoinButton}
                onPress={() => {
                  joinSession(selectedSession.id);
                  setSelectedSession(null);
                }}
              >
                <Text style={styles.modalJoinButtonText}>Join Session</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0c',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  subtitle: {
    fontSize: 14,
    color: '#9ca3af',
    marginBottom: 24,
  },
  sessionCard: {
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  sessionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  typeBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  typeText: {
    fontSize: 12,
    fontWeight: '600',
  },
  sessionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  sessionDescription: {
    fontSize: 14,
    color: '#9ca3af',
    marginBottom: 12,
  },
  sessionMeta: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 12,
  },
  metaItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  metaText: {
    fontSize: 12,
    color: '#9ca3af',
  },
  sessionFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#27272a',
  },
  participantInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  participantText: {
    fontSize: 14,
    color: '#9ca3af',
  },
  joinButton: {
    backgroundColor: '#10b981',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 8,
  },
  joinButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#18181b',
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 24,
    paddingBottom: 40,
  },
  closeButton: {
    position: 'absolute',
    top: 16,
    right: 16,
    zIndex: 1,
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 16,
    marginBottom: 8,
  },
  modalDescription: {
    fontSize: 16,
    color: '#9ca3af',
    marginBottom: 24,
  },
  modalMeta: {
    gap: 12,
    marginBottom: 24,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  modalMetaText: {
    fontSize: 16,
    color: '#fff',
  },
  modalJoinButton: {
    backgroundColor: '#10b981',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  modalJoinButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
});
