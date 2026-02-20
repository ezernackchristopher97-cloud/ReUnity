import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface PeerProfile {
  id: string;
  anonymousName: string;
  experiences: string[];
  bio: string;
  isOnline: boolean;
  matchScore: number;
}

export default function PeerSupportMatching() {
  const [activeTab, setActiveTab] = useState<'find' | 'matches' | 'chat'>('find');
  const [selectedExperiences, setSelectedExperiences] = useState<string[]>([]);
  const [matches] = useState<PeerProfile[]>([
    { id: '1', anonymousName: 'HopefulHeart', experiences: ['anxiety', 'depression'], bio: 'Been through dark times, now helping others.', isOnline: true, matchScore: 92 },
    { id: '2', anonymousName: 'GentleWarrior', experiences: ['ptsd', 'anxiety'], bio: 'Survivor and listener.', isOnline: true, matchScore: 87 },
    { id: '3', anonymousName: 'QuietStrength', experiences: ['grief', 'caregiver'], bio: 'Walking beside you.', isOnline: false, matchScore: 78 },
  ]);

  const experienceCategories = [
    { id: 'anxiety', label: 'Anxiety', icon: '😰' },
    { id: 'depression', label: 'Depression', icon: '😔' },
    { id: 'ptsd', label: 'PTSD/Trauma', icon: '💔' },
    { id: 'grief', label: 'Grief & Loss', icon: '🕊️' },
    { id: 'relationship', label: 'Relationships', icon: '💑' },
    { id: 'family', label: 'Family', icon: '👨‍👩‍👧' },
  ];

  const toggleExperience = (id: string) => {
    setSelectedExperiences(prev =>
      prev.includes(id) ? prev.filter(e => e !== id) : [...prev, id]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.iconContainer}>
          <Ionicons name="people" size={24} color="#a855f7" />
        </View>
        <View>
          <Text style={styles.title}>Peer Support</Text>
          <Text style={styles.subtitle}>Connect anonymously with others who understand</Text>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {[
          { id: 'find', label: 'Find', icon: 'search' },
          { id: 'matches', label: 'Matches', icon: 'person-add' },
          { id: 'chat', label: 'Chat', icon: 'chatbubbles' },
        ].map(tab => (
          <TouchableOpacity
            key={tab.id}
            style={[styles.tab, activeTab === tab.id && styles.activeTab]}
            onPress={() => setActiveTab(tab.id as typeof activeTab)}
          >
            <Ionicons name={tab.icon as any} size={18} color={activeTab === tab.id ? '#fff' : '#a1a1aa'} />
            <Text style={[styles.tabText, activeTab === tab.id && styles.activeTabText]}>{tab.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Find Tab */}
      {activeTab === 'find' && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Select experiences to connect over:</Text>
          <View style={styles.experienceGrid}>
            {experienceCategories.map(exp => (
              <TouchableOpacity
                key={exp.id}
                style={[styles.experienceButton, selectedExperiences.includes(exp.id) && styles.selectedExperience]}
                onPress={() => toggleExperience(exp.id)}
              >
                <Text style={styles.experienceEmoji}>{exp.icon}</Text>
                <Text style={[styles.experienceLabel, selectedExperiences.includes(exp.id) && styles.selectedLabel]}>
                  {exp.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
          <TouchableOpacity
            style={[styles.searchButton, selectedExperiences.length === 0 && styles.disabledButton]}
            disabled={selectedExperiences.length === 0}
          >
            <Ionicons name="search" size={20} color="#fff" />
            <Text style={styles.searchButtonText}>Find Peer Supporters</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Matches Tab */}
      {activeTab === 'matches' && (
        <View style={styles.section}>
          {matches.map(peer => (
            <View key={peer.id} style={styles.matchCard}>
              <View style={styles.matchHeader}>
                <View style={styles.avatar}>
                  <Text style={styles.avatarText}>{peer.anonymousName.charAt(0)}</Text>
                  {peer.isOnline && <View style={styles.onlineIndicator} />}
                </View>
                <View style={styles.matchInfo}>
                  <Text style={styles.matchName}>{peer.anonymousName}</Text>
                  <Text style={styles.matchBio}>{peer.bio}</Text>
                </View>
                <View style={styles.matchScore}>
                  <Text style={styles.scoreText}>{peer.matchScore}%</Text>
                  <Text style={styles.scoreLabel}>Match</Text>
                </View>
              </View>
              <TouchableOpacity style={styles.chatButton}>
                <Ionicons name="chatbubble" size={16} color="#fff" />
                <Text style={styles.chatButtonText}>Start Conversation</Text>
              </TouchableOpacity>
            </View>
          ))}
        </View>
      )}

      {/* Chat Tab */}
      {activeTab === 'chat' && (
        <View style={styles.emptyState}>
          <Ionicons name="chatbubbles-outline" size={48} color="#52525b" />
          <Text style={styles.emptyTitle}>No Active Conversations</Text>
          <Text style={styles.emptySubtitle}>Find a peer supporter to start chatting</Text>
        </View>
      )}

      {/* Guidelines */}
      <View style={styles.guidelines}>
        <Ionicons name="shield-checkmark" size={20} color="#eab308" />
        <Text style={styles.guidelinesText}>
          All conversations are anonymous and moderated for safety.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#09090b' },
  header: { flexDirection: 'row', alignItems: 'center', padding: 20, gap: 12 },
  iconContainer: { width: 48, height: 48, borderRadius: 12, backgroundColor: 'rgba(168, 85, 247, 0.2)', justifyContent: 'center', alignItems: 'center' },
  title: { fontSize: 20, fontWeight: 'bold', color: '#fff' },
  subtitle: { fontSize: 14, color: '#a1a1aa' },
  tabs: { flexDirection: 'row', marginHorizontal: 16, backgroundColor: 'rgba(39, 39, 42, 0.5)', borderRadius: 8, padding: 4 },
  tab: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 8, gap: 6, borderRadius: 6 },
  activeTab: { backgroundColor: '#a855f7' },
  tabText: { fontSize: 14, color: '#a1a1aa' },
  activeTabText: { color: '#fff', fontWeight: '600' },
  section: { padding: 16 },
  sectionTitle: { fontSize: 14, color: '#a1a1aa', marginBottom: 12 },
  experienceGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  experienceButton: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, borderWidth: 1, borderColor: '#3f3f46', gap: 6 },
  selectedExperience: { backgroundColor: '#a855f7', borderColor: '#a855f7' },
  experienceEmoji: { fontSize: 16 },
  experienceLabel: { fontSize: 14, color: '#a1a1aa' },
  selectedLabel: { color: '#fff' },
  searchButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', backgroundColor: '#a855f7', paddingVertical: 14, borderRadius: 8, marginTop: 16, gap: 8 },
  disabledButton: { opacity: 0.5 },
  searchButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },
  matchCard: { backgroundColor: 'rgba(39, 39, 42, 0.5)', borderRadius: 12, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: '#27272a' },
  matchHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  avatar: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#a855f7', justifyContent: 'center', alignItems: 'center' },
  avatarText: { fontSize: 18, fontWeight: 'bold', color: '#fff' },
  onlineIndicator: { position: 'absolute', bottom: 0, right: 0, width: 12, height: 12, borderRadius: 6, backgroundColor: '#22c55e', borderWidth: 2, borderColor: '#09090b' },
  matchInfo: { flex: 1, marginLeft: 12 },
  matchName: { fontSize: 16, fontWeight: '600', color: '#fff' },
  matchBio: { fontSize: 12, color: '#a1a1aa', marginTop: 2 },
  matchScore: { alignItems: 'center' },
  scoreText: { fontSize: 18, fontWeight: 'bold', color: '#a855f7' },
  scoreLabel: { fontSize: 10, color: '#a1a1aa' },
  chatButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', backgroundColor: '#a855f7', paddingVertical: 10, borderRadius: 8, gap: 6 },
  chatButtonText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  emptyState: { alignItems: 'center', paddingVertical: 48 },
  emptyTitle: { fontSize: 18, fontWeight: '600', color: '#52525b', marginTop: 12 },
  emptySubtitle: { fontSize: 14, color: '#3f3f46', marginTop: 4 },
  guidelines: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(234, 179, 8, 0.1)', margin: 16, padding: 12, borderRadius: 8, gap: 8 },
  guidelinesText: { flex: 1, fontSize: 12, color: '#eab308' },
});
