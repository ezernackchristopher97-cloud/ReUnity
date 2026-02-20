import React, { useState, useEffect, useMemo } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  TextInput, 
  ScrollView, 
  Alert,
  Modal,
  Linking
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';

interface JournalEntry {
  id: string;
  date: string;
  content: string;
  sentiment: SentimentResult;
  mood?: 'great' | 'good' | 'okay' | 'bad' | 'terrible';
  tags: string[];
  isPrivate: boolean;
  createdAt: number;
  updatedAt: number;
}

interface SentimentResult {
  score: number;
  magnitude: number;
  label: 'positive' | 'negative' | 'neutral' | 'mixed';
  keywords: string[];
  concerns: string[];
}

interface JournalWithSentimentProps {
  onSentimentUpdate?: (entries: JournalEntry[]) => void;
  onCrisisDetected?: (entry: JournalEntry) => void;
}

const CRISIS_KEYWORDS = [
  'suicide', 'kill myself', 'end it all', 'want to die', 'no point',
  'self harm', 'hurt myself', 'cutting', 'overdose', 'hopeless',
  'give up', 'cant go on', 'worthless', 'burden', 'better off without me'
];

const POSITIVE_KEYWORDS = [
  'happy', 'grateful', 'thankful', 'joy', 'excited', 'hopeful',
  'peaceful', 'calm', 'loved', 'supported', 'proud', 'accomplished'
];

const NEGATIVE_KEYWORDS = [
  'sad', 'anxious', 'worried', 'stressed', 'overwhelmed', 'tired',
  'exhausted', 'angry', 'frustrated', 'lonely', 'isolated', 'scared'
];

const MOOD_OPTIONS = [
  { value: 'great', icon: 'sunny', label: 'Great', color: '#f59e0b' },
  { value: 'good', icon: 'happy', label: 'Good', color: '#10b981' },
  { value: 'okay', icon: 'remove', label: 'Okay', color: '#3b82f6' },
  { value: 'bad', icon: 'cloud', label: 'Bad', color: '#71717a' },
  { value: 'terrible', icon: 'rainy', label: 'Terrible', color: '#ef4444' },
] as const;

const PROMPTS = [
  "What are you grateful for today?",
  "How are you feeling right now?",
  "What's been on your mind lately?",
  "Describe a moment that made you smile today.",
  "What challenges did you face today?",
  "What would make tomorrow better?",
];

function analyzeSentiment(text: string): SentimentResult {
  const lowerText = text.toLowerCase();
  const words = lowerText.split(/\s+/);
  
  let positiveCount = 0;
  let negativeCount = 0;
  const foundKeywords: string[] = [];
  const concerns: string[] = [];
  
  for (const keyword of CRISIS_KEYWORDS) {
    if (lowerText.includes(keyword)) {
      concerns.push(keyword);
    }
  }
  
  for (const word of words) {
    if (POSITIVE_KEYWORDS.some(kw => word.includes(kw))) {
      positiveCount++;
      foundKeywords.push(word);
    }
    if (NEGATIVE_KEYWORDS.some(kw => word.includes(kw))) {
      negativeCount++;
      foundKeywords.push(word);
    }
  }
  
  const total = positiveCount + negativeCount;
  let score = 0;
  let label: SentimentResult['label'] = 'neutral';
  
  if (total > 0) {
    score = (positiveCount - negativeCount) / total;
    
    if (concerns.length > 0) {
      score = Math.min(score, -0.5);
      label = 'negative';
    } else if (score > 0.3) {
      label = 'positive';
    } else if (score < -0.3) {
      label = 'negative';
    } else if (positiveCount > 0 && negativeCount > 0) {
      label = 'mixed';
    }
  }
  
  const magnitude = Math.min(1, (positiveCount + negativeCount) / 10);
  
  return {
    score,
    magnitude,
    label,
    keywords: Array.from(new Set(foundKeywords)).slice(0, 5),
    concerns,
  };
}

export default function JournalWithSentiment({ onSentimentUpdate, onCrisisDetected }: JournalWithSentimentProps) {
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [isWriting, setIsWriting] = useState(false);
  const [content, setContent] = useState('');
  const [selectedMood, setSelectedMood] = useState<JournalEntry['mood']>();
  const [showCrisisAlert, setShowCrisisAlert] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState(PROMPTS[0]);

  useEffect(() => {
    loadEntries();
  }, []);

  useEffect(() => {
    const idx = Math.floor(Math.random() * PROMPTS.length);
    setCurrentPrompt(PROMPTS[idx]);
  }, [isWriting]);

  const liveSentiment = useMemo(() => {
    if (content.length < 10) return null;
    return analyzeSentiment(content);
  }, [content]);

  useEffect(() => {
    if (liveSentiment?.concerns && liveSentiment.concerns.length > 0) {
      setShowCrisisAlert(true);
    }
  }, [liveSentiment]);

  const loadEntries = async () => {
    try {
      const stored = await AsyncStorage.getItem('reunity_journal_entries');
      if (stored) {
        setEntries(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load entries:', error);
    }
  };

  const saveEntry = async () => {
    if (!content.trim()) return;
    
    const sentiment = analyzeSentiment(content);
    const now = Date.now();
    const today = new Date().toISOString().split('T')[0];
    
    const newEntry: JournalEntry = {
      id: now.toString(),
      date: today,
      content,
      sentiment,
      mood: selectedMood,
      tags: [],
      isPrivate: true,
      createdAt: now,
      updatedAt: now,
    };
    
    const updatedEntries = [newEntry, ...entries];
    
    try {
      await AsyncStorage.setItem('reunity_journal_entries', JSON.stringify(updatedEntries));
      setEntries(updatedEntries);
      onSentimentUpdate?.(updatedEntries);
      
      if (sentiment.concerns.length > 0) {
        onCrisisDetected?.(newEntry);
      }
      
      resetForm();
    } catch (error) {
      Alert.alert('Error', 'Failed to save entry');
    }
  };

  const deleteEntry = (id: string) => {
    Alert.alert(
      'Delete Entry',
      'Are you sure you want to delete this journal entry?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: async () => {
            const updatedEntries = entries.filter(e => e.id !== id);
            await AsyncStorage.setItem('reunity_journal_entries', JSON.stringify(updatedEntries));
            setEntries(updatedEntries);
            onSentimentUpdate?.(updatedEntries);
          }
        },
      ]
    );
  };

  const resetForm = () => {
    setContent('');
    setSelectedMood(undefined);
    setIsWriting(false);
    setShowCrisisAlert(false);
  };

  const weeklyStats = useMemo(() => {
    const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
    const weekEntries = entries.filter(e => e.createdAt > weekAgo);
    
    const avgSentiment = weekEntries.length > 0
      ? weekEntries.reduce((sum, e) => sum + e.sentiment.score, 0) / weekEntries.length
      : 0;
    
    return { avgSentiment, entryCount: weekEntries.length };
  }, [entries]);

  const todayStr = new Date().toISOString().split('T')[0];
  const hasEntryToday = entries.some(e => e.date === todayStr);

  return (
    <ScrollView style={styles.container}>
      {/* Crisis Alert Modal */}
      <Modal
        visible={showCrisisAlert}
        transparent
        animationType="fade"
        onRequestClose={() => setShowCrisisAlert(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.alertIcon}>
              <Ionicons name="heart" size={32} color="#ef4444" />
            </View>
            
            <Text style={styles.modalTitle}>We're Here For You</Text>
            <Text style={styles.modalSubtitle}>
              It sounds like you're going through a difficult time
            </Text>
            
            <Text style={styles.modalText}>
              If you're having thoughts of self-harm or suicide, please reach out for support.
            </Text>

            <TouchableOpacity
              style={styles.crisisButton}
              onPress={() => {
                Linking.openURL('tel:988');
                setShowCrisisAlert(false);
              }}
            >
              <Ionicons name="call" size={20} color="#fff" />
              <Text style={styles.crisisButtonText}>Call 988 Crisis Lifeline</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.textButton}
              onPress={() => {
                Linking.openURL('sms:741741?body=HELLO');
                setShowCrisisAlert(false);
              }}
            >
              <Ionicons name="chatbubble" size={20} color="#3b82f6" />
              <Text style={styles.textButtonText}>Text HOME to 741741</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.dismissButton}
              onPress={() => setShowCrisisAlert(false)}
            >
              <Text style={styles.dismissText}>Continue Writing</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Weekly Summary */}
      <View style={styles.summaryCard}>
        <View style={styles.summaryHeader}>
          <Ionicons name="sparkles" size={20} color="#10b981" />
          <Text style={styles.summaryTitle}>This Week's Journey</Text>
        </View>
        <View style={styles.summaryStats}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{weeklyStats.entryCount}</Text>
            <Text style={styles.statLabel}>Entries</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={[
              styles.statValue,
              weeklyStats.avgSentiment > 0.2 ? styles.positive :
              weeklyStats.avgSentiment < -0.2 ? styles.negative : styles.neutral
            ]}>
              {weeklyStats.avgSentiment > 0 ? '+' : ''}{(weeklyStats.avgSentiment * 100).toFixed(0)}%
            </Text>
            <Text style={styles.statLabel}>Avg Mood</Text>
          </View>
        </View>
      </View>

      {/* New Entry Button or Writing Area */}
      {!isWriting ? (
        <View style={styles.newEntryCard}>
          {!hasEntryToday ? (
            <>
              <Ionicons name="book" size={48} color="rgba(16, 185, 129, 0.5)" />
              <Text style={styles.emptyText}>You haven't journaled today</Text>
              <TouchableOpacity
                style={styles.writeButton}
                onPress={() => setIsWriting(true)}
              >
                <Ionicons name="add" size={20} color="#fff" />
                <Text style={styles.writeButtonText}>Write Today's Entry</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Ionicons name="heart" size={48} color="#10b981" />
              <Text style={styles.successText}>Great job journaling today!</Text>
              <TouchableOpacity
                style={styles.addMoreButton}
                onPress={() => setIsWriting(true)}
              >
                <Ionicons name="add" size={20} color="#10b981" />
                <Text style={styles.addMoreText}>Add Another Entry</Text>
              </TouchableOpacity>
            </>
          )}
        </View>
      ) : (
        <View style={styles.writingCard}>
          <View style={styles.writingHeader}>
            <View style={styles.writingTitleRow}>
              <Ionicons name="book" size={20} color="#10b981" />
              <Text style={styles.writingTitle}>New Journal Entry</Text>
            </View>
            <TouchableOpacity onPress={resetForm}>
              <Ionicons name="close" size={24} color="#71717a" />
            </TouchableOpacity>
          </View>
          
          <Text style={styles.prompt}>"{currentPrompt}"</Text>

          {/* Mood Selection */}
          <Text style={styles.label}>How are you feeling?</Text>
          <View style={styles.moodRow}>
            {MOOD_OPTIONS.map(option => (
              <TouchableOpacity
                key={option.value}
                style={[
                  styles.moodButton,
                  selectedMood === option.value && styles.moodButtonActive
                ]}
                onPress={() => setSelectedMood(option.value)}
              >
                <Ionicons 
                  name={option.icon as any} 
                  size={20} 
                  color={selectedMood === option.value ? '#fff' : option.color} 
                />
                <Text style={[
                  styles.moodLabel,
                  selectedMood === option.value && styles.moodLabelActive
                ]}>
                  {option.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Content */}
          <TextInput
            style={styles.textArea}
            placeholder="Write your thoughts..."
            placeholderTextColor="#71717a"
            multiline
            value={content}
            onChangeText={setContent}
          />

          {/* Live Sentiment */}
          {liveSentiment && (
            <View style={styles.sentimentIndicator}>
              <Text style={styles.sentimentLabel}>Sentiment:</Text>
              <View style={[
                styles.sentimentBadge,
                liveSentiment.label === 'positive' && styles.sentimentPositive,
                liveSentiment.label === 'negative' && styles.sentimentNegative,
                liveSentiment.label === 'mixed' && styles.sentimentMixed,
              ]}>
                <Text style={styles.sentimentText}>{liveSentiment.label}</Text>
              </View>
            </View>
          )}

          {/* Save Button */}
          <TouchableOpacity style={styles.saveButton} onPress={saveEntry}>
            <Ionicons name="save" size={20} color="#fff" />
            <Text style={styles.saveButtonText}>Save Entry</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Past Entries */}
      <View style={styles.historySection}>
        <View style={styles.historyHeader}>
          <Ionicons name="calendar" size={20} color="#10b981" />
          <Text style={styles.historyTitle}>Recent Entries</Text>
        </View>

        {entries.slice(0, 10).map(entry => {
          const moodOption = MOOD_OPTIONS.find(o => o.value === entry.mood);
          
          return (
            <View key={entry.id} style={styles.entryItem}>
              <View style={styles.entryHeader}>
                <View style={styles.entryMeta}>
                  {moodOption && (
                    <Ionicons 
                      name={moodOption.icon as any} 
                      size={16} 
                      color={moodOption.color} 
                    />
                  )}
                  <Text style={styles.entryDate}>
                    {new Date(entry.createdAt).toLocaleDateString()}
                  </Text>
                  <View style={[
                    styles.entrySentiment,
                    entry.sentiment.label === 'positive' && styles.sentimentPositive,
                    entry.sentiment.label === 'negative' && styles.sentimentNegative,
                  ]}>
                    <Text style={styles.entrySentimentText}>{entry.sentiment.label}</Text>
                  </View>
                </View>
                <TouchableOpacity onPress={() => deleteEntry(entry.id)}>
                  <Ionicons name="trash" size={16} color="#ef4444" />
                </TouchableOpacity>
              </View>
              <Text style={styles.entryContent} numberOfLines={3}>
                {entry.content}
              </Text>
            </View>
          );
        })}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
  summaryCard: {
    margin: 16,
    padding: 16,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(16, 185, 129, 0.2)',
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  summaryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6ee7b7',
    marginLeft: 8,
  },
  summaryStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
  },
  statLabel: {
    fontSize: 12,
    color: '#a1a1aa',
  },
  positive: { color: '#10b981' },
  negative: { color: '#ef4444' },
  neutral: { color: '#d4d4d8' },
  newEntryCard: {
    margin: 16,
    padding: 32,
    backgroundColor: '#18181b',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#27272a',
    alignItems: 'center',
  },
  emptyText: {
    color: '#71717a',
    marginTop: 12,
    marginBottom: 16,
  },
  successText: {
    color: '#d4d4d8',
    marginTop: 12,
    marginBottom: 16,
  },
  writeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#10b981',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  writeButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  addMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#10b981',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  addMoreText: {
    color: '#10b981',
    fontWeight: '600',
  },
  writingCard: {
    margin: 16,
    padding: 16,
    backgroundColor: '#18181b',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  writingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  writingTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  writingTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  prompt: {
    fontStyle: 'italic',
    color: '#71717a',
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    color: '#a1a1aa',
    marginBottom: 8,
  },
  moodRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
  },
  moodButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3f3f46',
    gap: 4,
  },
  moodButtonActive: {
    backgroundColor: '#3f3f46',
    borderColor: '#52525b',
  },
  moodLabel: {
    fontSize: 12,
    color: '#a1a1aa',
  },
  moodLabelActive: {
    color: '#fff',
  },
  textArea: {
    backgroundColor: '#27272a',
    borderWidth: 1,
    borderColor: '#3f3f46',
    borderRadius: 8,
    padding: 12,
    color: '#fff',
    height: 150,
    textAlignVertical: 'top',
    marginBottom: 12,
  },
  sentimentIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  sentimentLabel: {
    fontSize: 12,
    color: '#71717a',
  },
  sentimentBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
    backgroundColor: '#3f3f46',
  },
  sentimentPositive: {
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
  },
  sentimentNegative: {
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
  },
  sentimentMixed: {
    backgroundColor: 'rgba(245, 158, 11, 0.2)',
  },
  sentimentText: {
    fontSize: 12,
    color: '#d4d4d8',
  },
  saveButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#10b981',
    padding: 14,
    borderRadius: 8,
    gap: 8,
  },
  saveButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  historySection: {
    margin: 16,
    padding: 16,
    backgroundColor: '#18181b',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  historyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  historyTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  entryItem: {
    backgroundColor: '#27272a',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  entryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  entryMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  entryDate: {
    fontSize: 12,
    color: '#71717a',
  },
  entrySentiment: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
    backgroundColor: '#3f3f46',
  },
  entrySentimentText: {
    fontSize: 10,
    color: '#a1a1aa',
  },
  entryContent: {
    color: '#d4d4d8',
    fontSize: 14,
    lineHeight: 20,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    backgroundColor: '#18181b',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.5)',
  },
  alertIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#fff',
    textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#a1a1aa',
    textAlign: 'center',
    marginBottom: 16,
  },
  modalText: {
    fontSize: 14,
    color: '#d4d4d8',
    textAlign: 'center',
    marginBottom: 24,
  },
  crisisButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#ef4444',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  crisisButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  textButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: '#3b82f6',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  textButtonText: {
    color: '#3b82f6',
    fontWeight: '600',
    fontSize: 16,
  },
  dismissButton: {
    padding: 12,
    alignItems: 'center',
  },
  dismissText: {
    color: '#71717a',
    fontSize: 14,
  },
});
