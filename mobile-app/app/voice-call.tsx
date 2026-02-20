import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, TouchableOpacity, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import VoiceChat from '../components/VoiceChat';

export default function VoiceCallScreen() {
  const router = useRouter();

  const handleSendMessage = async (text: string): Promise<string> => {
    // In a real implementation, this would call the ReUnity AI backend
    // For now, return a supportive response
    const responses = [
      "I hear you, and I'm here with you. Take a moment to breathe. You're doing well by reaching out.",
      "Thank you for sharing that with me. Your feelings are valid. Let's take this one step at a time.",
      "I'm listening. It sounds like you're going through something difficult. You don't have to face this alone.",
      "That takes courage to express. I want you to know that whatever you're feeling right now, it will pass. I'm here.",
      "I understand. Sometimes just speaking our thoughts aloud can help. I'm here to listen without judgment.",
    ];
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    return responses[Math.floor(Math.random() * responses.length)];
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity 
          style={styles.backButton}
          onPress={() => router.back()}
        >
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Voice Call</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        {/* Voice Chat Component */}
        <VoiceChat 
          onSendMessage={handleSendMessage}
        />

        {/* Info Section */}
        <View style={styles.infoSection}>
          <View style={styles.infoCard}>
            <Ionicons name="information-circle" size={20} color="#10b981" />
            <Text style={styles.infoText}>
              ReUnity uses a gentle, calming voice to respond to you. Speak naturally about how you're feeling.
            </Text>
          </View>

          <View style={styles.infoCard}>
            <Ionicons name="shield-checkmark" size={20} color="#10b981" />
            <Text style={styles.infoText}>
              Your voice conversations are private and secure. We're here to support you.
            </Text>
          </View>

          <View style={styles.infoCard}>
            <Ionicons name="heart" size={20} color="#10b981" />
            <Text style={styles.infoText}>
              If you're in crisis, say "I need help" or "I'm not safe" and ReUnity will provide immediate support resources.
            </Text>
          </View>
        </View>

        {/* Quick Phrases */}
        <View style={styles.quickPhrases}>
          <Text style={styles.sectionTitle}>Things you can say:</Text>
          <View style={styles.phraseList}>
            <View style={styles.phraseItem}>
              <Text style={styles.phraseText}>"I'm feeling anxious today"</Text>
            </View>
            <View style={styles.phraseItem}>
              <Text style={styles.phraseText}>"I need someone to talk to"</Text>
            </View>
            <View style={styles.phraseItem}>
              <Text style={styles.phraseText}>"Help me calm down"</Text>
            </View>
            <View style={styles.phraseItem}>
              <Text style={styles.phraseText}>"I'm okay, just checking in"</Text>
            </View>
          </View>
        </View>
      </ScrollView>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Powered by ReUnity AI • entropy-physics-ai.com
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0c',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.1)',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  placeholder: {
    width: 40,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    gap: 24,
  },
  infoSection: {
    gap: 12,
  },
  infoCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(16, 185, 129, 0.2)',
    borderRadius: 12,
    padding: 16,
  },
  infoText: {
    flex: 1,
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
    lineHeight: 20,
  },
  quickPhrases: {
    gap: 12,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  phraseList: {
    gap: 8,
  },
  phraseItem: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  phraseText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    fontStyle: 'italic',
  },
  footer: {
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.4)',
  },
});
