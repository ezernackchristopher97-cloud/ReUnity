import React, { useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, SafeAreaView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import LanguageSelector from '../components/LanguageSelector';
import BeliefSystemSelector from '../components/BeliefSystemSelector';

export default function PreferencesScreen() {
  const [selectedLanguage, setSelectedLanguage] = useState<string>('');
  const [selectedBelief, setSelectedBelief] = useState<string>('');
  const [voicePersona, setVoicePersona] = useState<string>('gentle-woman');

  const voiceOptions = [
    { id: 'gentle-woman', name: 'Gentle Woman', description: 'A soft, nurturing feminine voice' },
    { id: 'gentle-man', name: 'Gentle Man', description: 'A calm, reassuring masculine voice' },
    { id: 'neutral', name: 'Neutral Voice', description: 'A balanced, gender-neutral tone' },
    { id: 'warm-elder', name: 'Warm Elder', description: 'A wise, comforting elder voice' },
    { id: 'calm-friend', name: 'Calm Friend', description: 'A friendly, supportive companion' },
  ];

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Personalize ReUnity</Text>
          <Text style={styles.subtitle}>
            Make ReUnity feel like home. All settings are optional and can be changed anytime.
          </Text>
        </View>

        {/* Language Section */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="language" size={20} color="#10b981" />
            <Text style={styles.sectionTitle}>Language</Text>
          </View>
          <Text style={styles.sectionDescription}>
            We support 30+ languages including Native American, South Asian, Middle Eastern, 
            and many more. ReUnity will greet you and offer comfort in your language.
          </Text>
          <LanguageSelector 
            selectedLanguage={selectedLanguage}
            onSelectLanguage={setSelectedLanguage}
          />
        </View>

        {/* Belief System Section */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="sparkles" size={20} color="#8b5cf6" />
            <Text style={styles.sectionTitle}>Spiritual/Philosophical Background</Text>
          </View>
          <Text style={styles.sectionDescription}>
            All beliefs are honored here - religious, spiritual, philosophical, or secular. 
            ReUnity will offer comfort and coping strategies aligned with your worldview.
          </Text>
          <BeliefSystemSelector 
            selectedBelief={selectedBelief}
            onSelectBelief={setSelectedBelief}
          />
        </View>

        {/* Voice Persona Section */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="mic" size={20} color="#3b82f6" />
            <Text style={styles.sectionTitle}>Voice Companion</Text>
          </View>
          <Text style={styles.sectionDescription}>
            Choose the voice that feels most comfortable for you during voice conversations.
            Everyone is welcome here.
          </Text>
          
          <View style={styles.voiceOptions}>
            {voiceOptions.map(voice => (
              <TouchableOpacity
                key={voice.id}
                style={[
                  styles.voiceOption,
                  voicePersona === voice.id && styles.voiceOptionSelected
                ]}
                onPress={() => setVoicePersona(voice.id)}
              >
                <View style={[
                  styles.voiceIcon,
                  voicePersona === voice.id && styles.voiceIconSelected
                ]}>
                  <Ionicons 
                    name="person" 
                    size={18} 
                    color={voicePersona === voice.id ? '#3b82f6' : '#6b7280'} 
                  />
                </View>
                <View style={styles.voiceInfo}>
                  <Text style={[
                    styles.voiceName,
                    voicePersona === voice.id && styles.voiceNameSelected
                  ]}>
                    {voice.name}
                  </Text>
                  <Text style={styles.voiceDescription}>{voice.description}</Text>
                </View>
                {voicePersona === voice.id && (
                  <Ionicons name="checkmark-circle" size={22} color="#3b82f6" />
                )}
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Inclusivity Notice */}
        <View style={styles.inclusivityCard}>
          <Ionicons name="heart" size={24} color="#ec4899" />
          <Text style={styles.inclusivityTitle}>Everyone Belongs Here</Text>
          <Text style={styles.inclusivityText}>
            ReUnity welcomes all identities, backgrounds, beliefs, and languages. 
            Your mental health matters, and you deserve support that honors who you are.
          </Text>
        </View>

        <View style={styles.spacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0f0f1a',
  },
  container: {
    flex: 1,
    padding: 16,
  },
  header: {
    marginBottom: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 15,
    color: '#9ca3af',
    lineHeight: 22,
  },
  section: {
    marginBottom: 28,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: '#fff',
  },
  sectionDescription: {
    fontSize: 13,
    color: '#6b7280',
    lineHeight: 20,
    marginBottom: 12,
  },
  voiceOptions: {
    gap: 8,
  },
  voiceOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 14,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
  },
  voiceOptionSelected: {
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  voiceIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(107, 114, 128, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  voiceIconSelected: {
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
  },
  voiceInfo: {
    flex: 1,
  },
  voiceName: {
    fontSize: 15,
    fontWeight: '500',
    color: '#fff',
    marginBottom: 2,
  },
  voiceNameSelected: {
    color: '#3b82f6',
  },
  voiceDescription: {
    fontSize: 12,
    color: '#6b7280',
  },
  inclusivityCard: {
    backgroundColor: 'rgba(236, 72, 153, 0.1)',
    borderRadius: 16,
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(236, 72, 153, 0.2)',
  },
  inclusivityTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ec4899',
    marginTop: 10,
    marginBottom: 8,
  },
  inclusivityText: {
    fontSize: 14,
    color: '#d1d5db',
    textAlign: 'center',
    lineHeight: 22,
  },
  spacer: {
    height: 40,
  },
});
