import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput, Animated } from 'react-native';
import { Audio } from 'expo-av';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface MeditationSession {
  id: string;
  title: string;
  description: string;
  duration: number;
  category: string;
  difficulty: string;
  instructor: string;
  isFavorite: boolean;
  playCount: number;
}

const MEDITATION_LIBRARY: MeditationSession[] = [
  { id: '1', title: 'Calm Breathing for Anxiety', description: 'A gentle breathing exercise to calm your nervous system', duration: 5, category: 'anxiety', difficulty: 'beginner', instructor: 'Dr. Sarah Chen', isFavorite: false, playCount: 0 },
  { id: '2', title: 'Body Scan for Grounding', description: 'Progressive body scan to reconnect with your physical self', duration: 15, category: 'grounding', difficulty: 'beginner', instructor: 'Dr. Michael Torres', isFavorite: false, playCount: 0 },
  { id: '3', title: 'Sleep Preparation Journey', description: 'A calming visualization for restful sleep', duration: 20, category: 'sleep', difficulty: 'beginner', instructor: 'Emma Williams', isFavorite: false, playCount: 0 },
  { id: '4', title: 'Self-Compassion Practice', description: 'Cultivate kindness toward yourself', duration: 12, category: 'self-compassion', difficulty: 'intermediate', instructor: 'Dr. Sarah Chen', isFavorite: false, playCount: 0 },
  { id: '5', title: 'Stress Release Visualization', description: 'Release tension through guided imagery', duration: 10, category: 'stress', difficulty: 'beginner', instructor: 'Dr. Michael Torres', isFavorite: false, playCount: 0 },
  { id: '6', title: 'Safe Place Visualization', description: 'Create your inner safe place for comfort', duration: 15, category: 'trauma', difficulty: 'intermediate', instructor: 'Dr. Lisa Park', isFavorite: false, playCount: 0 },
  { id: '7', title: 'Morning Intention Setting', description: 'Start your day with clarity and positive intentions', duration: 8, category: 'general', difficulty: 'beginner', instructor: 'Emma Williams', isFavorite: false, playCount: 0 },
  { id: '8', title: 'Depression Lift Practice', description: 'Gentle movement and visualization to lift low mood', duration: 18, category: 'depression', difficulty: 'intermediate', instructor: 'Dr. Sarah Chen', isFavorite: false, playCount: 0 },
  { id: '9', title: '5-4-3-2-1 Grounding', description: 'Quick sensory grounding for dissociation or panic', duration: 3, category: 'grounding', difficulty: 'beginner', instructor: 'Dr. Michael Torres', isFavorite: false, playCount: 0 },
  { id: '10', title: 'Deep Sleep Meditation', description: 'Extended practice for deep, restorative sleep', duration: 45, category: 'sleep', difficulty: 'beginner', instructor: 'Emma Williams', isFavorite: false, playCount: 0 },
];

const CATEGORY_COLORS: Record<string, string> = {
  anxiety: '#f59e0b',
  depression: '#eab308',
  sleep: '#6366f1',
  stress: '#3b82f6',
  grounding: '#22c55e',
  'self-compassion': '#ec4899',
  trauma: '#a855f7',
  general: '#06b6d4',
};

export default function GuidedMeditationLibrary() {
  const [sessions, setSessions] = useState<MeditationSession[]>(MEDITATION_LIBRARY);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [currentSession, setCurrentSession] = useState<MeditationSession | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const progressInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const saved = await AsyncStorage.getItem('reunity-meditation-sessions');
      if (saved) {
        const parsed = JSON.parse(saved);
        setSessions(MEDITATION_LIBRARY.map(lib => ({
          ...lib,
          isFavorite: parsed.find((p: MeditationSession) => p.id === lib.id)?.isFavorite || false,
          playCount: parsed.find((p: MeditationSession) => p.id === lib.id)?.playCount || 0,
        })));
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const saveSessions = async (newSessions: MeditationSession[]) => {
    try {
      await AsyncStorage.setItem('reunity-meditation-sessions', JSON.stringify(newSessions));
      setSessions(newSessions);
    } catch (error) {
      console.error('Error saving sessions:', error);
    }
  };

  const filteredSessions = sessions.filter(session => {
    const matchesSearch = session.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         session.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || session.category === selectedCategory;
    const matchesFavorites = !showFavoritesOnly || session.isFavorite;
    return matchesSearch && matchesCategory && matchesFavorites;
  });

  const playSession = (session: MeditationSession) => {
    if (progressInterval.current) {
      clearInterval(progressInterval.current);
    }
    
    setCurrentSession(session);
    setIsPlaying(true);
    setProgress(0);
    progressAnim.setValue(0);

    // Update play count
    const updatedSessions = sessions.map(s =>
      s.id === session.id ? { ...s, playCount: s.playCount + 1 } : s
    );
    saveSessions(updatedSessions);

    // Simulate playback
    progressInterval.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          if (progressInterval.current) clearInterval(progressInterval.current);
          setIsPlaying(false);
          return 100;
        }
        return prev + (100 / (session.duration * 60));
      });
    }, 1000);

    Animated.timing(progressAnim, {
      toValue: 1,
      duration: session.duration * 60 * 1000,
      useNativeDriver: false,
    }).start();
  };

  const togglePlayPause = () => {
    if (isPlaying) {
      if (progressInterval.current) clearInterval(progressInterval.current);
      progressAnim.stopAnimation();
    } else if (currentSession) {
      progressInterval.current = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            if (progressInterval.current) clearInterval(progressInterval.current);
            setIsPlaying(false);
            return 100;
          }
          return prev + (100 / (currentSession.duration * 60));
        });
      }, 1000);
    }
    setIsPlaying(!isPlaying);
  };

  const toggleFavorite = (sessionId: string) => {
    const updatedSessions = sessions.map(s =>
      s.id === sessionId ? { ...s, isFavorite: !s.isFavorite } : s
    );
    saveSessions(updatedSessions);
  };

  const categories = ['all', 'anxiety', 'depression', 'sleep', 'stress', 'grounding', 'self-compassion', 'trauma', 'general'];

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🧘 Guided Meditation</Text>
        <TouchableOpacity
          style={[styles.favButton, showFavoritesOnly && styles.favButtonActive]}
          onPress={() => setShowFavoritesOnly(!showFavoritesOnly)}
        >
          <Text style={styles.favButtonText}>❤️ Favorites</Text>
        </TouchableOpacity>
      </View>

      {/* Search */}
      <TextInput
        style={styles.searchInput}
        placeholder="Search meditations..."
        placeholderTextColor="#71717a"
        value={searchQuery}
        onChangeText={setSearchQuery}
      />

      {/* Categories */}
      <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.categoriesScroll}>
        {categories.map(cat => (
          <TouchableOpacity
            key={cat}
            style={[
              styles.categoryChip,
              selectedCategory === cat && styles.categoryChipActive,
              cat !== 'all' && { borderColor: CATEGORY_COLORS[cat] },
            ]}
            onPress={() => setSelectedCategory(cat)}
          >
            <Text style={[
              styles.categoryChipText,
              selectedCategory === cat && styles.categoryChipTextActive,
            ]}>
              {cat === 'all' ? 'All' : cat.charAt(0).toUpperCase() + cat.slice(1).replace('-', ' ')}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Now Playing */}
      {currentSession && (
        <View style={styles.nowPlaying}>
          <View style={styles.nowPlayingHeader}>
            <View style={[styles.categoryDot, { backgroundColor: CATEGORY_COLORS[currentSession.category] }]} />
            <View style={styles.nowPlayingInfo}>
              <Text style={styles.nowPlayingTitle}>{currentSession.title}</Text>
              <Text style={styles.nowPlayingInstructor}>{currentSession.instructor}</Text>
            </View>
            <TouchableOpacity style={styles.playPauseButton} onPress={togglePlayPause}>
              <Text style={styles.playPauseText}>{isPlaying ? '⏸️' : '▶️'}</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${progress}%` }]} />
          </View>
          <View style={styles.progressLabels}>
            <Text style={styles.progressTime}>{Math.floor((progress / 100) * currentSession.duration)} min</Text>
            <Text style={styles.progressTime}>{currentSession.duration} min</Text>
          </View>
        </View>
      )}

      {/* Session List */}
      {filteredSessions.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyIcon}>🧘</Text>
          <Text style={styles.emptyText}>No meditations found</Text>
        </View>
      ) : (
        filteredSessions.map(session => (
          <TouchableOpacity
            key={session.id}
            style={[
              styles.sessionCard,
              currentSession?.id === session.id && isPlaying && styles.sessionCardActive,
            ]}
            onPress={() => playSession(session)}
          >
            <View style={[styles.categoryIndicator, { backgroundColor: CATEGORY_COLORS[session.category] }]} />
            <View style={styles.sessionInfo}>
              <Text style={styles.sessionTitle}>{session.title}</Text>
              <Text style={styles.sessionDescription} numberOfLines={1}>{session.description}</Text>
              <View style={styles.sessionMeta}>
                <Text style={styles.sessionDuration}>⏱️ {session.duration} min</Text>
                <Text style={styles.sessionDifficulty}>{session.difficulty}</Text>
              </View>
            </View>
            <TouchableOpacity
              style={styles.favoriteButton}
              onPress={() => toggleFavorite(session.id)}
            >
              <Text style={styles.favoriteIcon}>{session.isFavorite ? '❤️' : '🤍'}</Text>
            </TouchableOpacity>
          </TouchableOpacity>
        ))
      )}

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>Powered by ReUnity</Text>
        <Text style={styles.footerLink}>entropy-physics-ai.com</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0a0a0a', padding: 16 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#fff' },
  favButton: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16, borderWidth: 1, borderColor: '#3f3f46' },
  favButtonActive: { backgroundColor: '#ec4899', borderColor: '#ec4899' },
  favButtonText: { color: '#fff', fontSize: 14 },
  searchInput: { backgroundColor: '#18181b', borderRadius: 12, padding: 12, color: '#fff', marginBottom: 16, borderWidth: 1, borderColor: '#27272a' },
  categoriesScroll: { marginBottom: 16 },
  categoryChip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, borderWidth: 1, borderColor: '#3f3f46', marginRight: 8 },
  categoryChipActive: { backgroundColor: '#7c3aed', borderColor: '#7c3aed' },
  categoryChipText: { color: '#a1a1aa', fontSize: 14 },
  categoryChipTextActive: { color: '#fff' },
  nowPlaying: { backgroundColor: '#134e4a', borderRadius: 16, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: '#14b8a6' },
  nowPlayingHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  categoryDot: { width: 12, height: 12, borderRadius: 6, marginRight: 12 },
  nowPlayingInfo: { flex: 1 },
  nowPlayingTitle: { color: '#fff', fontSize: 16, fontWeight: '600' },
  nowPlayingInstructor: { color: '#5eead4', fontSize: 14 },
  playPauseButton: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#14b8a6', justifyContent: 'center', alignItems: 'center' },
  playPauseText: { fontSize: 20 },
  progressBar: { height: 4, backgroundColor: '#0d3d3d', borderRadius: 2, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: '#14b8a6' },
  progressLabels: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 4 },
  progressTime: { color: '#5eead4', fontSize: 12 },
  emptyState: { alignItems: 'center', paddingVertical: 48 },
  emptyIcon: { fontSize: 48, marginBottom: 16 },
  emptyText: { color: '#71717a', fontSize: 16 },
  sessionCard: { flexDirection: 'row', backgroundColor: '#18181b', borderRadius: 12, padding: 12, marginBottom: 12, borderWidth: 1, borderColor: '#27272a' },
  sessionCardActive: { borderColor: '#14b8a6' },
  categoryIndicator: { width: 4, borderRadius: 2, marginRight: 12 },
  sessionInfo: { flex: 1 },
  sessionTitle: { color: '#fff', fontSize: 16, fontWeight: '500', marginBottom: 4 },
  sessionDescription: { color: '#71717a', fontSize: 14, marginBottom: 8 },
  sessionMeta: { flexDirection: 'row', gap: 12 },
  sessionDuration: { color: '#a1a1aa', fontSize: 12 },
  sessionDifficulty: { color: '#a1a1aa', fontSize: 12, textTransform: 'capitalize' },
  favoriteButton: { justifyContent: 'center', paddingLeft: 12 },
  favoriteIcon: { fontSize: 20 },
  footer: { alignItems: 'center', paddingVertical: 24 },
  footerText: { color: '#52525b', fontSize: 12 },
  footerLink: { color: '#10b981', fontSize: 12, marginTop: 4 },
});
