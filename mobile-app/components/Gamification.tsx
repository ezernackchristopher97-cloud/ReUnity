import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Modal } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface Streak {
  id: string;
  name: string;
  icon: string;
  currentStreak: number;
  longestStreak: number;
  isActiveToday: boolean;
  color: string;
}

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  isUnlocked: boolean;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  xpReward: number;
}

const RARITY_COLORS = {
  common: '#71717a',
  rare: '#3b82f6',
  epic: '#8b5cf6',
  legendary: '#f59e0b',
};

export default function Gamification() {
  const [level, setLevel] = useState(5);
  const [totalXP, setTotalXP] = useState(1250);
  const [streaks, setStreaks] = useState<Streak[]>([]);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [showAchievement, setShowAchievement] = useState<Achievement | null>(null);

  useEffect(() => {
    setStreaks([
      { id: 'checkin', name: 'Check-In', icon: 'checkmark-circle', currentStreak: 7, longestStreak: 14, isActiveToday: true, color: '#10b981' },
      { id: 'journal', name: 'Journal', icon: 'book', currentStreak: 3, longestStreak: 10, isActiveToday: true, color: '#3b82f6' },
      { id: 'meditation', name: 'Mindfulness', icon: 'leaf', currentStreak: 5, longestStreak: 21, isActiveToday: false, color: '#8b5cf6' },
      { id: 'selfcare', name: 'Self-Care', icon: 'heart', currentStreak: 2, longestStreak: 7, isActiveToday: true, color: '#ec4899' },
    ]);

    setAchievements([
      { id: '1', name: 'First Steps', description: 'Complete your first check-in', icon: 'star', isUnlocked: true, rarity: 'common', xpReward: 50 },
      { id: '2', name: 'Week Warrior', description: '7-day check-in streak', icon: 'flame', isUnlocked: true, rarity: 'rare', xpReward: 200 },
      { id: '3', name: 'Month Master', description: '30-day check-in streak', icon: 'trophy', isUnlocked: false, rarity: 'epic', xpReward: 1000 },
      { id: '4', name: 'Century Club', description: '100-day check-in streak', icon: 'medal', isUnlocked: false, rarity: 'legendary', xpReward: 5000 },
      { id: '5', name: 'Dear Diary', description: 'Write your first journal', icon: 'book', isUnlocked: true, rarity: 'common', xpReward: 50 },
      { id: '6', name: 'Crisis Survivor', description: 'Navigate a crisis with support', icon: 'shield', isUnlocked: true, rarity: 'epic', xpReward: 500 },
    ]);
  }, []);

  const xpProgress = ((totalXP % 500) / 500) * 100;

  return (
    <ScrollView style={styles.container}>
      {/* Level Card */}
      <View style={styles.levelCard}>
        <View style={styles.levelCircle}>
          <Text style={styles.levelNumber}>{level}</Text>
        </View>
        <View style={styles.levelInfo}>
          <Text style={styles.levelTitle}>Level {level}</Text>
          <Text style={styles.xpText}>{totalXP.toLocaleString()} XP</Text>
          <View style={styles.progressBar}>
            <View style={[styles.progressFill, { width: `${xpProgress}%` }]} />
          </View>
          <Text style={styles.progressText}>{500 - (totalXP % 500)} XP to Level {level + 1}</Text>
        </View>
      </View>

      {/* Streaks */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="flame" size={24} color="#f97316" />
          <Text style={styles.sectionTitle}>Active Streaks</Text>
        </View>
        <View style={styles.streaksGrid}>
          {streaks.map((streak) => (
            <TouchableOpacity
              key={streak.id}
              style={[styles.streakCard, streak.isActiveToday && styles.streakActive]}
            >
              <View style={[styles.streakIcon, { backgroundColor: streak.color + '30' }]}>
                <Ionicons name={streak.icon as any} size={24} color={streak.color} />
              </View>
              <Text style={styles.streakName}>{streak.name}</Text>
              <View style={styles.streakCount}>
                <Ionicons name="flame" size={16} color={streak.currentStreak > 0 ? '#f97316' : '#52525b'} />
                <Text style={styles.streakNumber}>{streak.currentStreak}</Text>
              </View>
              <Text style={styles.streakBest}>Best: {streak.longestStreak}</Text>
              {streak.isActiveToday ? (
                <View style={styles.doneBadge}>
                  <Text style={styles.doneBadgeText}>Done!</Text>
                </View>
              ) : (
                <View style={styles.pendingBadge}>
                  <Text style={styles.pendingBadgeText}>Tap to complete</Text>
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Achievements */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="trophy" size={24} color="#f59e0b" />
          <Text style={styles.sectionTitle}>
            Achievements ({achievements.filter(a => a.isUnlocked).length}/{achievements.length})
          </Text>
        </View>
        <View style={styles.achievementsGrid}>
          {achievements.map((achievement) => (
            <TouchableOpacity
              key={achievement.id}
              style={[
                styles.achievementCard,
                !achievement.isUnlocked && styles.achievementLocked,
                { borderColor: RARITY_COLORS[achievement.rarity] + '50' }
              ]}
              onPress={() => achievement.isUnlocked && setShowAchievement(achievement)}
            >
              <View style={[
                styles.achievementIcon,
                { backgroundColor: achievement.isUnlocked ? RARITY_COLORS[achievement.rarity] + '30' : '#27272a' }
              ]}>
                {achievement.isUnlocked ? (
                  <Ionicons name={achievement.icon as any} size={28} color={RARITY_COLORS[achievement.rarity]} />
                ) : (
                  <Ionicons name="lock-closed" size={28} color="#52525b" />
                )}
              </View>
              <View style={[styles.rarityBadge, { backgroundColor: RARITY_COLORS[achievement.rarity] + '30' }]}>
                <Text style={[styles.rarityText, { color: RARITY_COLORS[achievement.rarity] }]}>
                  {achievement.rarity}
                </Text>
              </View>
              <Text style={[styles.achievementName, !achievement.isUnlocked && styles.textMuted]}>
                {achievement.name}
              </Text>
              <Text style={styles.achievementXP}>+{achievement.xpReward} XP</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Achievement Modal */}
      <Modal
        visible={!!showAchievement}
        animationType="fade"
        transparent={true}
        onRequestClose={() => setShowAchievement(null)}
      >
        {showAchievement && (
          <TouchableOpacity
            style={styles.modalOverlay}
            activeOpacity={1}
            onPress={() => setShowAchievement(null)}
          >
            <View style={[styles.modalContent, { borderColor: RARITY_COLORS[showAchievement.rarity] }]}>
              <View style={[styles.modalIcon, { backgroundColor: RARITY_COLORS[showAchievement.rarity] }]}>
                <Ionicons name={showAchievement.icon as any} size={40} color="#fff" />
              </View>
              <View style={[styles.rarityBadge, { backgroundColor: RARITY_COLORS[showAchievement.rarity] + '30' }]}>
                <Text style={[styles.rarityText, { color: RARITY_COLORS[showAchievement.rarity] }]}>
                  {showAchievement.rarity.toUpperCase()}
                </Text>
              </View>
              <Text style={styles.modalTitle}>{showAchievement.name}</Text>
              <Text style={styles.modalDescription}>{showAchievement.description}</Text>
              <View style={styles.modalXP}>
                <Ionicons name="sparkles" size={20} color="#f59e0b" />
                <Text style={styles.modalXPText}>+{showAchievement.xpReward} XP</Text>
              </View>
            </View>
          </TouchableOpacity>
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
  levelCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#18181b',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  levelCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#f59e0b',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 20,
  },
  levelNumber: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
  },
  levelInfo: {
    flex: 1,
  },
  levelTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  xpText: {
    fontSize: 16,
    color: '#f59e0b',
    marginBottom: 8,
  },
  progressBar: {
    height: 8,
    backgroundColor: '#27272a',
    borderRadius: 4,
    marginBottom: 4,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#10b981',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 12,
    color: '#9ca3af',
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  streaksGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  streakCard: {
    width: '47%',
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#27272a',
  },
  streakActive: {
    borderColor: '#10b981',
  },
  streakIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  streakName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#fff',
    marginBottom: 4,
  },
  streakCount: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  streakNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  streakBest: {
    fontSize: 12,
    color: '#71717a',
    marginTop: 4,
  },
  doneBadge: {
    backgroundColor: '#10b98130',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    marginTop: 8,
  },
  doneBadgeText: {
    fontSize: 12,
    color: '#10b981',
    fontWeight: '500',
  },
  pendingBadge: {
    backgroundColor: '#27272a',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    marginTop: 8,
  },
  pendingBadgeText: {
    fontSize: 10,
    color: '#71717a',
  },
  achievementsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  achievementCard: {
    width: '47%',
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
  },
  achievementLocked: {
    opacity: 0.5,
  },
  achievementIcon: {
    width: 56,
    height: 56,
    borderRadius: 28,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  rarityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 2,
    borderRadius: 10,
    marginBottom: 8,
  },
  rarityText: {
    fontSize: 10,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  achievementName: {
    fontSize: 14,
    fontWeight: '500',
    color: '#fff',
    textAlign: 'center',
  },
  achievementXP: {
    fontSize: 12,
    color: '#f59e0b',
    marginTop: 4,
  },
  textMuted: {
    color: '#71717a',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: '#18181b',
    borderRadius: 24,
    padding: 32,
    alignItems: 'center',
    width: '80%',
    borderWidth: 2,
  },
  modalIcon: {
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  modalDescription: {
    fontSize: 16,
    color: '#9ca3af',
    textAlign: 'center',
    marginBottom: 16,
  },
  modalXP: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  modalXPText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#f59e0b',
  },
});
