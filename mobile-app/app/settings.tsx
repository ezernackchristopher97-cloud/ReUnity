import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  Share,
  Linking,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';
import AccessibilitySettings from '../components/AccessibilitySettings';
import TrustedDevicePairing from '../components/TrustedDevicePairing';
import LanguageSelector from '../components/LanguageSelector';

type SettingsSection = 'general' | 'accessibility' | 'devices' | 'language' | 'privacy' | 'crisis';

export default function Settings() {
  const [activeSection, setActiveSection] = useState<SettingsSection>('general');

  const clearAllData = async () => {
    Alert.alert(
      'Clear All Data',
      'Are you sure you want to delete all app data? This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            try {
              const keys = await AsyncStorage.getAllKeys();
              const reunityKeys = keys.filter(k => k.startsWith('reunity') || k.startsWith('reop'));
              await AsyncStorage.multiRemove(reunityKeys);
              Alert.alert('Success', 'All data cleared');
            } catch (e) {
              Alert.alert('Error', 'Failed to clear data');
            }
          },
        },
      ]
    );
  };

  const exportData = async () => {
    try {
      const keys = await AsyncStorage.getAllKeys();
      const reunityKeys = keys.filter(k => k.startsWith('reunity') || k.startsWith('reop'));
      const data: Record<string, any> = {};
      
      for (const key of reunityKeys) {
        const value = await AsyncStorage.getItem(key);
        if (value) {
          try {
            data[key] = JSON.parse(value);
          } catch {
            data[key] = value;
          }
        }
      }

      await Share.share({
        message: JSON.stringify(data, null, 2),
        title: 'ReUnity Data Export',
      });
    } catch (e) {
      Alert.alert('Error', 'Failed to export data');
    }
  };

  const sections: { id: SettingsSection; label: string; icon: string }[] = [
    { id: 'general', label: 'General', icon: 'settings' },
    { id: 'accessibility', label: 'Accessibility', icon: 'accessibility' },
    { id: 'devices', label: 'Trusted Devices', icon: 'phone-portrait' },
    { id: 'language', label: 'Language', icon: 'globe' },
    { id: 'privacy', label: 'Privacy & Data', icon: 'lock-closed' },
    { id: 'crisis', label: 'Crisis Resources', icon: 'shield' },
  ];

  const renderContent = () => {
    switch (activeSection) {
      case 'accessibility':
        return <AccessibilitySettings />;
      
      case 'devices':
        return <TrustedDevicePairing />;
      
      case 'language':
        return (
          <View style={styles.sectionContent}>
            <View style={styles.sectionHeader}>
              <Ionicons name="globe" size={24} color="#3b82f6" />
              <Text style={styles.sectionTitle}>Language Settings</Text>
            </View>
            <Text style={styles.sectionDescription}>
              Choose your preferred language for crisis resources
            </Text>
            <LanguageSelector />
          </View>
        );
      
      case 'privacy':
        return (
          <View style={styles.sectionContent}>
            <View style={styles.sectionHeader}>
              <Ionicons name="lock-closed" size={24} color="#ef4444" />
              <Text style={styles.sectionTitle}>Privacy & Data</Text>
            </View>
            
            <View style={styles.infoBox}>
              <Text style={styles.infoTitle}>Your Privacy Matters</Text>
              <Text style={styles.infoText}>
                ReUnity stores data locally on your device. Your conversations and 
                personal information are not sent to external servers.
              </Text>
            </View>

            <TouchableOpacity style={styles.actionRow} onPress={exportData}>
              <View style={styles.actionInfo}>
                <Ionicons name="download" size={20} color="#10b981" />
                <View>
                  <Text style={styles.actionTitle}>Export My Data</Text>
                  <Text style={styles.actionDesc}>Download all your data</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#71717a" />
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.actionRow, styles.dangerRow]} 
              onPress={clearAllData}
            >
              <View style={styles.actionInfo}>
                <Ionicons name="trash" size={20} color="#ef4444" />
                <View>
                  <Text style={[styles.actionTitle, { color: '#ef4444' }]}>Clear All Data</Text>
                  <Text style={styles.actionDesc}>Permanently delete all app data</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#71717a" />
            </TouchableOpacity>
          </View>
        );
      
      case 'crisis':
        return (
          <View style={styles.sectionContent}>
            <View style={styles.sectionHeader}>
              <Ionicons name="shield" size={24} color="#10b981" />
              <Text style={styles.sectionTitle}>Safety Features</Text>
            </View>

            <View style={styles.featureCard}>
              <Text style={styles.featureTitle}>Panic Button</Text>
              <Text style={styles.featureDesc}>
                Shake device 3 times or use the gear icon to activate decoy mode
              </Text>
            </View>

            <View style={styles.featureCard}>
              <Text style={styles.featureTitle}>Voice Check-In</Text>
              <Text style={styles.featureDesc}>
                Respond to check-ins with voice commands like "I'm okay" or "I need help"
              </Text>
            </View>

            <View style={styles.featureCard}>
              <Text style={styles.featureTitle}>Safe Word</Text>
              <Text style={styles.featureDesc}>
                Set a custom word that triggers panic mode when typed anywhere
              </Text>
            </View>

            <View style={styles.featureCard}>
              <Text style={styles.featureTitle}>Biometric Lock</Text>
              <Text style={styles.featureDesc}>
                Protect your safety plans with Face ID, fingerprint, or PIN
              </Text>
            </View>

            <View style={styles.crisisNumbers}>
              <Text style={styles.crisisTitle}>Crisis Hotlines</Text>
              <View style={styles.crisisItem}>
                <Text style={styles.crisisName}>988 Suicide & Crisis Lifeline</Text>
                <Text style={styles.crisisPhone}>988</Text>
              </View>
              <View style={styles.crisisItem}>
                <Text style={styles.crisisName}>National DV Hotline</Text>
                <Text style={styles.crisisPhone}>1-800-799-7233</Text>
              </View>
              <View style={styles.crisisItem}>
                <Text style={styles.crisisName}>Crisis Text Line</Text>
                <Text style={styles.crisisPhone}>Text HOME to 741741</Text>
              </View>
            </View>
          </View>
        );
      
      default:
        return (
          <View style={styles.sectionContent}>
            {/* Required Global Disclaimer */}
            <View style={[styles.infoBox, { backgroundColor: '#3b82f615', borderColor: '#3b82f640', marginBottom: 16 }]}>
              <Text style={[styles.infoText, { color: '#93c5fd', fontSize: 12 }]}>
                ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.
              </Text>
            </View>

            <View style={styles.sectionHeader}>
              <Ionicons name="settings" size={24} color="#10b981" />
              <Text style={styles.sectionTitle}>General Settings</Text>
            </View>

            <TouchableOpacity style={styles.actionRow}>
              <View style={styles.actionInfo}>
                <Ionicons name="notifications" size={20} color="#3b82f6" />
                <View>
                  <Text style={styles.actionTitle}>Notifications</Text>
                  <Text style={styles.actionDesc}>Check-in reminders</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#71717a" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.actionRow}>
              <View style={styles.actionInfo}>
                <Ionicons name="help-circle" size={20} color="#a855f7" />
                <View>
                  <Text style={styles.actionTitle}>Help & Support</Text>
                  <Text style={styles.actionDesc}>Get help using ReUnity</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#71717a" />
            </TouchableOpacity>

            <TouchableOpacity style={styles.actionRow}>
              <View style={styles.actionInfo}>
                <Ionicons name="information-circle" size={20} color="#f59e0b" />
                <View>
                  <Text style={styles.actionTitle}>About</Text>
                  <Text style={styles.actionDesc}>Version 1.0.0</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color="#71717a" />
            </TouchableOpacity>

            <TouchableOpacity 
              style={styles.actionRow}
              onPress={() => Linking.openURL('https://entropy-physics-ai.com/')}
            >
              <View style={styles.actionInfo}>
                <Ionicons name="globe" size={20} color="#10b981" />
                <View>
                  <Text style={styles.actionTitle}>Entropy Physics AI</Text>
                  <Text style={styles.actionDesc}>Visit our main website</Text>
                </View>
              </View>
              <Ionicons name="open-outline" size={20} color="#71717a" />
            </TouchableOpacity>
          </View>
        );
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#ffffff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Settings</Text>
        <View style={{ width: 40 }} />
      </View>

      <View style={styles.content}>
        {/* Sidebar */}
        <View style={styles.sidebar}>
          {sections.map(section => (
            <TouchableOpacity
              key={section.id}
              style={[
                styles.sidebarItem,
                activeSection === section.id && styles.sidebarItemActive,
              ]}
              onPress={() => setActiveSection(section.id)}
            >
              <Ionicons
                name={section.icon as any}
                size={20}
                color={activeSection === section.id ? '#10b981' : '#71717a'}
              />
              <Text
                style={[
                  styles.sidebarText,
                  activeSection === section.id && styles.sidebarTextActive,
                ]}
              >
                {section.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Main Content */}
        <ScrollView style={styles.mainContent}>
          {renderContent()}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f11',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  backButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
  },
  content: {
    flex: 1,
    flexDirection: 'row',
  },
  sidebar: {
    width: 80,
    backgroundColor: '#18181b',
    paddingVertical: 8,
    borderRightWidth: 1,
    borderRightColor: '#27272a',
  },
  sidebarItem: {
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 8,
  },
  sidebarItemActive: {
    backgroundColor: '#10b98120',
    borderRightWidth: 2,
    borderRightColor: '#10b981',
  },
  sidebarText: {
    fontSize: 10,
    color: '#71717a',
    marginTop: 4,
    textAlign: 'center',
  },
  sidebarTextActive: {
    color: '#10b981',
  },
  mainContent: {
    flex: 1,
  },
  sectionContent: {
    padding: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
  },
  sectionDescription: {
    color: '#9ca3af',
    fontSize: 14,
    marginBottom: 16,
  },
  infoBox: {
    backgroundColor: '#3b82f620',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#3b82f630',
  },
  infoTitle: {
    color: '#93c5fd',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 8,
  },
  infoText: {
    color: '#9ca3af',
    fontSize: 13,
    lineHeight: 18,
  },
  actionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  dangerRow: {
    backgroundColor: '#ef444420',
    borderWidth: 1,
    borderColor: '#ef444430',
  },
  actionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  actionTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
  },
  actionDesc: {
    color: '#71717a',
    fontSize: 12,
    marginTop: 2,
  },
  featureCard: {
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  featureTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 4,
  },
  featureDesc: {
    color: '#9ca3af',
    fontSize: 13,
  },
  crisisNumbers: {
    marginTop: 16,
  },
  crisisTitle: {
    color: '#9ca3af',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  crisisItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#27272a',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  crisisName: {
    color: '#ffffff',
    fontSize: 14,
  },
  crisisPhone: {
    color: '#10b981',
    fontSize: 14,
    fontFamily: 'monospace',
  },
});
