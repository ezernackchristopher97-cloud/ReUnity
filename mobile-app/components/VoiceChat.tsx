import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, Modal, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';

interface VoiceChatProps {
  onTranscript?: (text: string) => void;
  onSendMessage?: (text: string) => Promise<string>;
  compact?: boolean;
}

// Voice persona options - inclusive of all gender identities
type VoicePersona = 'gentle-woman' | 'gentle-man' | 'neutral' | 'warm-elder' | 'calm-friend';

interface VoiceConfig {
  name: string;
  description: string;
  voiceNames: string[]; // Preferred voice names to search for
  pitch: number;
  rate: number;
}

const VOICE_PERSONAS: Record<VoicePersona, VoiceConfig> = {
  'gentle-woman': {
    name: 'Gentle Woman',
    description: 'A soft, nurturing feminine voice',
    voiceNames: ['samantha', 'karen', 'victoria', 'moira', 'fiona', 'female', 'woman'],
    pitch: 1.1,
    rate: 0.9,
  },
  'gentle-man': {
    name: 'Gentle Man',
    description: 'A calm, reassuring masculine voice',
    voiceNames: ['daniel', 'alex', 'tom', 'oliver', 'james', 'david', 'male', 'man'],
    pitch: 0.9,
    rate: 0.9,
  },
  'neutral': {
    name: 'Neutral Voice',
    description: 'A balanced, gender-neutral tone',
    voiceNames: ['alex', 'samantha', 'karen', 'google'],
    pitch: 1.0,
    rate: 0.95,
  },
  'warm-elder': {
    name: 'Warm Elder',
    description: 'A wise, comforting elder voice',
    voiceNames: ['moira', 'fiona', 'karen', 'samantha', 'female'],
    pitch: 0.95,
    rate: 0.85,
  },
  'calm-friend': {
    name: 'Calm Friend',
    description: 'A friendly, supportive companion',
    voiceNames: ['samantha', 'alex', 'karen', 'daniel'],
    pitch: 1.0,
    rate: 0.92,
  },
};

// Greetings for each persona
const GREETINGS: Record<VoicePersona, string> = {
  'gentle-woman': "Hello, I'm here with you. Take your time, and speak whenever you're ready. I'm listening.",
  'gentle-man': "Hey there. I'm right here with you. Take all the time you need. I'm listening.",
  'neutral': "Hello. I'm here to listen and support you. Speak whenever you feel ready.",
  'warm-elder': "Hello, dear one. I'm here with you now. There's no rush. Share what's on your heart when you're ready.",
  'calm-friend': "Hey, I'm here for you. No pressure at all. Just talk when you feel like it.",
};

export default function VoiceChat({ onTranscript, onSendMessage, compact = false }: VoiceChatProps) {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCallActive, setIsCallActive] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPersona, setSelectedPersona] = useState<VoicePersona>('gentle-woman');
  const [showSettings, setShowSettings] = useState(false);
  const [availableVoices, setAvailableVoices] = useState<Speech.Voice[]>([]);
  
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Load available voices
  useEffect(() => {
    const loadVoices = async () => {
      try {
        const voices = await Speech.getAvailableVoicesAsync();
        setAvailableVoices(voices);
      } catch (err) {
        console.error('Failed to load voices:', err);
      }
    };
    loadVoices();
  }, []);

  // Pulse animation for active states
  useEffect(() => {
    if (isListening || isSpeaking) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.2,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [isListening, isSpeaking, pulseAnim]);

  // Find best voice for persona
  const findBestVoice = (): string | undefined => {
    if (availableVoices.length === 0) return undefined;
    
    const persona = VOICE_PERSONAS[selectedPersona];
    
    // Try to find a voice matching persona preferences
    for (const preferredName of persona.voiceNames) {
      const match = availableVoices.find(v => 
        v.name.toLowerCase().includes(preferredName.toLowerCase()) ||
        v.identifier.toLowerCase().includes(preferredName.toLowerCase())
      );
      if (match) return match.identifier;
    }
    
    // Fallback to first English voice
    const englishVoice = availableVoices.find(v => 
      v.language.startsWith('en')
    );
    return englishVoice?.identifier;
  };

  // Text-to-speech with selected voice persona
  const speakText = async (text: string) => {
    if (!voiceEnabled || !text) return;

    setIsSpeaking(true);
    setIsListening(false);

    try {
      const persona = VOICE_PERSONAS[selectedPersona];
      const voiceId = findBestVoice();

      await Speech.speak(text, {
        language: 'en-US',
        pitch: persona.pitch,
        rate: persona.rate,
        voice: voiceId,
        onDone: () => {
          setIsSpeaking(false);
          // Resume listening after speaking
          if (isCallActive) {
            setIsListening(true);
          }
        },
        onError: () => {
          setIsSpeaking(false);
        },
      });
    } catch (err) {
      console.error('TTS error:', err);
      setIsSpeaking(false);
    }
  };

  // Handle sending voice message to AI
  const handleSendVoiceMessage = async (text: string) => {
    if (!text.trim() || !onSendMessage) return;

    setIsProcessing(true);
    setTranscript('');

    try {
      const response = await onSendMessage(text);
      if (response && voiceEnabled) {
        await speakText(response);
      }
    } catch (err) {
      console.error('Error sending voice message:', err);
      setError('Failed to get response. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Start voice call
  const startCall = () => {
    setIsCallActive(true);
    setError(null);
    setTranscript('');
    setIsListening(true);

    // Greeting message based on persona
    speakText(GREETINGS[selectedPersona]);
  };

  // End voice call
  const endCall = () => {
    setIsCallActive(false);
    setIsListening(false);
    setIsSpeaking(false);
    setTranscript('');
    Speech.stop();
  };

  // Toggle microphone
  const toggleMic = () => {
    setIsListening(!isListening);
  };

  // Stop current speech
  const stopSpeaking = () => {
    Speech.stop();
    setIsSpeaking(false);
  };

  // Test voice
  const testVoice = () => {
    speakText("Hello, this is how I sound. I'm here to support you.");
  };

  // Voice Settings Modal
  const VoiceSettingsModal = () => (
    <Modal
      visible={showSettings}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowSettings(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Choose Your Companion's Voice</Text>
            <TouchableOpacity onPress={() => setShowSettings(false)}>
              <Ionicons name="close" size={24} color="#fff" />
            </TouchableOpacity>
          </View>

          <Text style={styles.modalSubtitle}>
            Everyone is welcome here. Choose the voice that feels most comfortable for you.
          </Text>

          <ScrollView style={styles.personaList}>
            {(Object.entries(VOICE_PERSONAS) as [VoicePersona, VoiceConfig][]).map(([key, config]) => (
              <TouchableOpacity
                key={key}
                style={[
                  styles.personaOption,
                  selectedPersona === key && styles.personaOptionSelected
                ]}
                onPress={() => setSelectedPersona(key)}
              >
                <View style={[
                  styles.personaIcon,
                  selectedPersona === key && styles.personaIconSelected
                ]}>
                  <Ionicons 
                    name="person" 
                    size={20} 
                    color={selectedPersona === key ? '#10b981' : '#6b7280'} 
                  />
                </View>
                <View style={styles.personaInfo}>
                  <Text style={[
                    styles.personaName,
                    selectedPersona === key && styles.personaNameSelected
                  ]}>
                    {config.name}
                  </Text>
                  <Text style={styles.personaDescription}>{config.description}</Text>
                </View>
                {selectedPersona === key && (
                  <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                )}
              </TouchableOpacity>
            ))}
          </ScrollView>

          <TouchableOpacity style={styles.testButton} onPress={testVoice}>
            <Ionicons name="volume-high" size={20} color="#fff" />
            <Text style={styles.testButtonText}>Test Voice</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );

  if (compact) {
    return (
      <View style={styles.compactContainer}>
        <VoiceSettingsModal />
        {!isCallActive ? (
          <View style={styles.compactControls}>
            <TouchableOpacity
              style={styles.startCallButton}
              onPress={startCall}
            >
              <Ionicons name="call" size={20} color="#fff" />
              <Text style={styles.buttonText}>Voice Call</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.settingsButton}
              onPress={() => setShowSettings(true)}
            >
              <Ionicons name="settings-outline" size={20} color="#fff" />
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.compactControls}>
            <TouchableOpacity
              style={styles.endCallButton}
              onPress={endCall}
            >
              <Ionicons name="call" size={20} color="#fff" />
              <Text style={styles.buttonText}>End</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.micButton, isListening && styles.micButtonActive]}
              onPress={toggleMic}
            >
              <Ionicons 
                name={isListening ? "mic" : "mic-off"} 
                size={20} 
                color="#fff" 
              />
            </TouchableOpacity>
            {isSpeaking && (
              <TouchableOpacity
                style={styles.stopButton}
                onPress={stopSpeaking}
              >
                <Ionicons name="volume-mute" size={20} color="#fff" />
              </TouchableOpacity>
            )}
            {isProcessing && (
              <View style={styles.processingIndicator}>
                <Ionicons name="hourglass" size={20} color="#10b981" />
              </View>
            )}
          </View>
        )}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <VoiceSettingsModal />
      
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Ionicons name="call" size={20} color="#10b981" />
          <Text style={styles.title}>Voice Conversation</Text>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity 
            style={styles.headerButton}
            onPress={() => setShowSettings(true)}
          >
            <Ionicons name="settings-outline" size={20} color="#6b7280" />
          </TouchableOpacity>
          <TouchableOpacity onPress={() => setVoiceEnabled(!voiceEnabled)}>
            <Ionicons 
              name={voiceEnabled ? "volume-high" : "volume-mute"} 
              size={20} 
              color={voiceEnabled ? "#10b981" : "#6b7280"} 
            />
          </TouchableOpacity>
        </View>
      </View>

      {/* Current persona indicator */}
      <View style={styles.personaIndicator}>
        <Ionicons name="person" size={14} color="#10b981" />
        <Text style={styles.personaIndicatorText}>
          {VOICE_PERSONAS[selectedPersona].name}
        </Text>
        <TouchableOpacity onPress={() => setShowSettings(true)}>
          <Text style={styles.changeText}>Change</Text>
        </TouchableOpacity>
      </View>

      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <View style={styles.content}>
        {!isCallActive ? (
          <View style={styles.idleState}>
            <View style={styles.phoneIconContainer}>
              <Ionicons name="call" size={40} color="#10b981" />
            </View>
            <Text style={styles.idleTitle}>Talk to ReUnity with your voice</Text>
            <Text style={styles.idleSubtitle}>
              {VOICE_PERSONAS[selectedPersona].description}
            </Text>
            <TouchableOpacity
              style={styles.startCallButtonLarge}
              onPress={startCall}
            >
              <Ionicons name="call" size={24} color="#fff" />
              <Text style={styles.startCallText}>Start Voice Call</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.activeState}>
            {/* Active Call UI */}
            <Animated.View 
              style={[
                styles.statusCircle,
                isSpeaking && styles.statusCircleSpeaking,
                isListening && !isSpeaking && styles.statusCircleListening,
                { transform: [{ scale: pulseAnim }] }
              ]}
            >
              {isSpeaking ? (
                <Ionicons name="volume-high" size={48} color="#10b981" />
              ) : isListening ? (
                <Ionicons name="mic" size={48} color="#3b82f6" />
              ) : isProcessing ? (
                <Ionicons name="hourglass" size={48} color="#f59e0b" />
              ) : (
                <Ionicons name="mic-off" size={48} color="#6b7280" />
              )}
            </Animated.View>

            {/* Status Text */}
            <Text style={styles.statusText}>
              {isSpeaking && `${VOICE_PERSONAS[selectedPersona].name} is speaking...`}
              {isListening && !isSpeaking && "Listening to you..."}
              {isProcessing && "Processing your message..."}
              {!isListening && !isSpeaking && !isProcessing && "Microphone paused"}
            </Text>

            {/* Transcript Display */}
            {transcript ? (
              <View style={styles.transcriptContainer}>
                <Text style={styles.transcriptLabel}>You said:</Text>
                <Text style={styles.transcriptText}>{transcript}</Text>
              </View>
            ) : null}

            {/* Call Controls */}
            <View style={styles.callControls}>
              <TouchableOpacity
                style={[styles.controlButton, isListening && styles.controlButtonActive]}
                onPress={toggleMic}
                disabled={isSpeaking}
              >
                <Ionicons 
                  name={isListening ? "mic" : "mic-off"} 
                  size={24} 
                  color="#fff" 
                />
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.endCallButtonLarge}
                onPress={endCall}
              >
                <Ionicons name="call" size={24} color="#fff" />
              </TouchableOpacity>

              {isSpeaking && (
                <TouchableOpacity
                  style={styles.controlButton}
                  onPress={stopSpeaking}
                >
                  <Ionicons name="volume-mute" size={24} color="#fff" />
                </TouchableOpacity>
              )}
            </View>
          </View>
        )}
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Speak naturally - ReUnity will respond with {VOICE_PERSONAS[selectedPersona].name.toLowerCase()}
        </Text>
        <Text style={styles.footerText}>
          Say "I'm okay" or describe how you're feeling
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    borderRadius: 12,
    padding: 16,
  },
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  compactControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  headerButton: {
    padding: 4,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  personaIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 8,
    marginBottom: 12,
  },
  personaIndicatorText: {
    fontSize: 12,
    color: '#10b981',
    flex: 1,
  },
  changeText: {
    fontSize: 12,
    color: '#6b7280',
    textDecorationLine: 'underline',
  },
  errorBanner: {
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    padding: 8,
    borderRadius: 8,
    marginBottom: 12,
  },
  errorText: {
    color: '#ef4444',
    fontSize: 12,
    textAlign: 'center',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  idleState: {
    alignItems: 'center',
    gap: 16,
  },
  phoneIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  idleTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    textAlign: 'center',
  },
  idleSubtitle: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
  },
  startCallButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  startCallButtonLarge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 12,
    marginTop: 8,
  },
  startCallText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  settingsButton: {
    backgroundColor: 'rgba(107, 114, 128, 0.3)',
    padding: 10,
    borderRadius: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
  activeState: {
    alignItems: 'center',
    gap: 20,
    width: '100%',
  },
  statusCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: 'rgba(107, 114, 128, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'rgba(107, 114, 128, 0.3)',
  },
  statusCircleSpeaking: {
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
    borderColor: 'rgba(16, 185, 129, 0.5)',
  },
  statusCircleListening: {
    backgroundColor: 'rgba(59, 130, 246, 0.2)',
    borderColor: 'rgba(59, 130, 246, 0.5)',
  },
  statusText: {
    fontSize: 14,
    color: '#9ca3af',
    textAlign: 'center',
  },
  transcriptContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    padding: 12,
    borderRadius: 8,
    width: '100%',
  },
  transcriptLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  transcriptText: {
    fontSize: 14,
    color: '#fff',
  },
  callControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    marginTop: 16,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(107, 114, 128, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  controlButtonActive: {
    backgroundColor: '#10b981',
  },
  endCallButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#ef4444',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
  },
  endCallButtonLarge: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#ef4444',
    justifyContent: 'center',
    alignItems: 'center',
  },
  micButton: {
    backgroundColor: 'rgba(107, 114, 128, 0.3)',
    padding: 10,
    borderRadius: 8,
  },
  micButtonActive: {
    backgroundColor: '#10b981',
  },
  stopButton: {
    backgroundColor: 'rgba(107, 114, 128, 0.3)',
    padding: 10,
    borderRadius: 8,
  },
  processingIndicator: {
    padding: 10,
  },
  footer: {
    alignItems: 'center',
    gap: 4,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  footerText: {
    fontSize: 12,
    color: '#6b7280',
    textAlign: 'center',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1a1a2e',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  modalSubtitle: {
    fontSize: 13,
    color: '#9ca3af',
    marginBottom: 16,
  },
  personaList: {
    maxHeight: 350,
  },
  personaOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 12,
    backgroundColor: 'rgba(107, 114, 128, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(107, 114, 128, 0.2)',
    marginBottom: 8,
  },
  personaOptionSelected: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderColor: 'rgba(16, 185, 129, 0.3)',
  },
  personaIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(107, 114, 128, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  personaIconSelected: {
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
  },
  personaInfo: {
    flex: 1,
  },
  personaName: {
    fontSize: 15,
    fontWeight: '500',
    color: '#fff',
    marginBottom: 2,
  },
  personaNameSelected: {
    color: '#10b981',
  },
  personaDescription: {
    fontSize: 12,
    color: '#9ca3af',
  },
  testButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: 'rgba(107, 114, 128, 0.3)',
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  testButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '500',
  },
});
