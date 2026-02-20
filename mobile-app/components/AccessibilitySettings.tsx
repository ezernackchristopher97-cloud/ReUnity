import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  Vibration,
  AccessibilityInfo,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';

interface AccessibilityPreferences {
  fontSize: 'small' | 'medium' | 'large' | 'xlarge';
  highContrast: boolean;
  reduceMotion: boolean;
  screenReaderOptimized: boolean;
  largeClickTargets: boolean;
  hapticFeedback: boolean;
  autoReadContent: boolean;
}

const defaultPreferences: AccessibilityPreferences = {
  fontSize: 'medium',
  highContrast: false,
  reduceMotion: false,
  screenReaderOptimized: false,
  largeClickTargets: false,
  hapticFeedback: true,
  autoReadContent: false,
};

const STORAGE_KEY = 'reunity_accessibility';

export default function AccessibilitySettings() {
  const [preferences, setPreferences] = useState<AccessibilityPreferences>(defaultPreferences);

  useEffect(() => {
    loadPreferences();
  }, []);

  const loadPreferences = async () => {
    try {
      const saved = await AsyncStorage.getItem(STORAGE_KEY);
      if (saved) {
        setPreferences(JSON.parse(saved));
      }
    } catch (e) {
      console.error('Failed to load accessibility preferences:', e);
    }
  };

  const savePreferences = async (newPrefs: AccessibilityPreferences) => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(newPrefs));
      setPreferences(newPrefs);
      
      if (newPrefs.hapticFeedback) {
        Vibration.vibrate(10);
      }
    } catch (e) {
      console.error('Failed to save accessibility preferences:', e);
    }
  };

  const updatePreference = <K extends keyof AccessibilityPreferences>(
    key: K,
    value: AccessibilityPreferences[K]
  ) => {
    const newPrefs = { ...preferences, [key]: value };
    savePreferences(newPrefs);
  };

  const resetToDefaults = () => {
    savePreferences(defaultPreferences);
  };

  const fontSizes = ['small', 'medium', 'large', 'xlarge'] as const;
  const fontSizeLabels = {
    small: 'S',
    medium: 'M',
    large: 'L',
    xlarge: 'XL',
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerRow}>
          <Ionicons name="accessibility" size={24} color="#60a5fa" />
          <Text style={styles.title}>Accessibility</Text>
        </View>
        <TouchableOpacity onPress={resetToDefaults} style={styles.resetButton}>
          <Ionicons name="refresh" size={16} color="#9ca3af" />
          <Text style={styles.resetText}>Reset</Text>
        </TouchableOpacity>
      </View>

      {/* Font Size */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Ionicons name="text" size={20} color="#9ca3af" />
          <Text style={styles.sectionTitle}>Text Size</Text>
        </View>
        <View style={styles.fontSizeRow}>
          {fontSizes.map((size) => (
            <TouchableOpacity
              key={size}
              onPress={() => updatePreference('fontSize', size)}
              style={[
                styles.fontSizeButton,
                preferences.fontSize === size && styles.fontSizeButtonActive,
              ]}
              accessibilityLabel={`${size} text size`}
              accessibilityState={{ selected: preferences.fontSize === size }}
            >
              <Text
                style={[
                  styles.fontSizeText,
                  preferences.fontSize === size && styles.fontSizeTextActive,
                ]}
              >
                {fontSizeLabels[size]}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Toggle Options */}
      <View style={styles.toggleSection}>
        {/* High Contrast */}
        <View style={styles.toggleRow}>
          <View style={styles.toggleInfo}>
            <Ionicons name="contrast" size={20} color="#facc15" />
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>High Contrast</Text>
              <Text style={styles.toggleDesc}>Increase color contrast</Text>
            </View>
          </View>
          <Switch
            value={preferences.highContrast}
            onValueChange={(value) => updatePreference('highContrast', value)}
            trackColor={{ false: '#3f3f46', true: '#ca8a04' }}
            thumbColor={preferences.highContrast ? '#facc15' : '#71717a'}
          />
        </View>

        {/* Reduce Motion */}
        <View style={styles.toggleRow}>
          <View style={styles.toggleInfo}>
            <Ionicons name="eye-off" size={20} color="#a855f7" />
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>Reduce Motion</Text>
              <Text style={styles.toggleDesc}>Minimize animations</Text>
            </View>
          </View>
          <Switch
            value={preferences.reduceMotion}
            onValueChange={(value) => updatePreference('reduceMotion', value)}
            trackColor={{ false: '#3f3f46', true: '#7c3aed' }}
            thumbColor={preferences.reduceMotion ? '#a855f7' : '#71717a'}
          />
        </View>

        {/* Large Touch Targets */}
        <View style={styles.toggleRow}>
          <View style={styles.toggleInfo}>
            <Ionicons name="finger-print" size={20} color="#10b981" />
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>Large Touch Targets</Text>
              <Text style={styles.toggleDesc}>Bigger buttons and links</Text>
            </View>
          </View>
          <Switch
            value={preferences.largeClickTargets}
            onValueChange={(value) => updatePreference('largeClickTargets', value)}
            trackColor={{ false: '#3f3f46', true: '#059669' }}
            thumbColor={preferences.largeClickTargets ? '#10b981' : '#71717a'}
          />
        </View>

        {/* Screen Reader Mode */}
        <View style={styles.toggleRow}>
          <View style={styles.toggleInfo}>
            <Ionicons name="volume-high" size={20} color="#3b82f6" />
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>Screen Reader Mode</Text>
              <Text style={styles.toggleDesc}>Optimize for VoiceOver/TalkBack</Text>
            </View>
          </View>
          <Switch
            value={preferences.screenReaderOptimized}
            onValueChange={(value) => updatePreference('screenReaderOptimized', value)}
            trackColor={{ false: '#3f3f46', true: '#2563eb' }}
            thumbColor={preferences.screenReaderOptimized ? '#3b82f6' : '#71717a'}
          />
        </View>

        {/* Auto-Read Responses */}
        <View style={styles.toggleRow}>
          <View style={styles.toggleInfo}>
            <Ionicons name="chatbubble-ellipses" size={20} color="#ec4899" />
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>Auto-Read Responses</Text>
              <Text style={styles.toggleDesc}>Read AI responses aloud</Text>
            </View>
          </View>
          <Switch
            value={preferences.autoReadContent}
            onValueChange={(value) => updatePreference('autoReadContent', value)}
            trackColor={{ false: '#3f3f46', true: '#db2777' }}
            thumbColor={preferences.autoReadContent ? '#ec4899' : '#71717a'}
          />
        </View>

        {/* Haptic Feedback */}
        <View style={styles.toggleRow}>
          <View style={styles.toggleInfo}>
            <Ionicons name="phone-portrait" size={20} color="#f97316" />
            <View style={styles.toggleText}>
              <Text style={styles.toggleTitle}>Haptic Feedback</Text>
              <Text style={styles.toggleDesc}>Vibration on interactions</Text>
            </View>
          </View>
          <Switch
            value={preferences.hapticFeedback}
            onValueChange={(value) => updatePreference('hapticFeedback', value)}
            trackColor={{ false: '#3f3f46', true: '#ea580c' }}
            thumbColor={preferences.hapticFeedback ? '#f97316' : '#71717a'}
          />
        </View>
      </View>

      {/* Tips */}
      <View style={styles.tipsContainer}>
        <Text style={styles.tipsTitle}>Accessibility Tips</Text>
        <Text style={styles.tipText}>• Shake device 3 times for panic mode</Text>
        <Text style={styles.tipText}>• Voice commands available during check-ins</Text>
        <Text style={styles.tipText}>• Use VoiceOver/TalkBack for full navigation</Text>
        <Text style={styles.tipText}>• Safe word triggers instant panic mode</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#18181b',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#ffffff',
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    padding: 8,
  },
  resetText: {
    color: '#9ca3af',
    fontSize: 14,
  },
  section: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  sectionTitle: {
    color: '#9ca3af',
    fontSize: 14,
  },
  fontSizeRow: {
    flexDirection: 'row',
    gap: 8,
  },
  fontSizeButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#27272a',
    alignItems: 'center',
  },
  fontSizeButtonActive: {
    backgroundColor: '#2563eb',
  },
  fontSizeText: {
    color: '#9ca3af',
    fontSize: 16,
    fontWeight: '600',
  },
  fontSizeTextActive: {
    color: '#ffffff',
  },
  toggleSection: {
    padding: 16,
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 12,
    backgroundColor: '#27272a50',
    borderRadius: 12,
    marginBottom: 8,
  },
  toggleInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  toggleText: {
    flex: 1,
  },
  toggleTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
  },
  toggleDesc: {
    color: '#71717a',
    fontSize: 12,
    marginTop: 2,
  },
  tipsContainer: {
    margin: 16,
    padding: 16,
    backgroundColor: '#1e3a5f30',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#3b82f630',
  },
  tipsTitle: {
    color: '#93c5fd',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
  },
  tipText: {
    color: '#9ca3af',
    fontSize: 12,
    marginBottom: 4,
  },
});
