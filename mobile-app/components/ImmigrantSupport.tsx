import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { 
  getGroundingForSituation, 
  getReassurance, 
  getMediaLiteracyTips,
  getSystemsAnalysis,
  GroundingTechnique,
  ReassuranceMessage
} from '../lib/immigrant-support';

interface ImmigrantSupportProps {
  onClose?: () => void;
}

export default function ImmigrantSupport({ onClose }: ImmigrantSupportProps) {
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [expandedGrounding, setExpandedGrounding] = useState<string | null>(null);

  const topics = [
    { id: 'policy-fears', label: 'Worried about policies', icon: 'document-text' },
    { id: 'family-separation', label: 'Family concerns', icon: 'people' },
    { id: 'workplace', label: 'Work worries', icon: 'briefcase' },
    { id: 'community', label: 'Community safety', icon: 'home' },
    { id: 'media-anxiety', label: 'News anxiety', icon: 'newspaper' },
  ];

  const groundingTechniques = getGroundingForSituation(selectedTopic || 'general');
  const reassurance = getReassurance(selectedTopic === 'policy-fears' ? 'policy-fear' : 
                                     selectedTopic === 'media-anxiety' ? 'media-overwhelm' : 
                                     'general');
  const mediaLiteracyTips = getMediaLiteracyTips();
  const systemsAnalysis = getSystemsAnalysis('policy-implementation');

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Ionicons name="heart" size={24} color="#10b981" />
          <Text style={styles.title}>You Are Welcome Here</Text>
        </View>
        {onClose && (
          <TouchableOpacity onPress={onClose}>
            <Ionicons name="close" size={24} color="#6b7280" />
          </TouchableOpacity>
        )}
      </View>

      <Text style={styles.subtitle}>
        Whatever you're feeling right now is valid. Let's take a breath together.
      </Text>

      {/* Topic Selection */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>What's on your mind?</Text>
        <View style={styles.topicGrid}>
          {topics.map(topic => (
            <TouchableOpacity
              key={topic.id}
              style={[
                styles.topicButton,
                selectedTopic === topic.id && styles.topicButtonSelected
              ]}
              onPress={() => setSelectedTopic(topic.id)}
            >
              <Ionicons 
                name={topic.icon as any} 
                size={20} 
                color={selectedTopic === topic.id ? '#10b981' : '#9ca3af'} 
              />
              <Text style={[
                styles.topicText,
                selectedTopic === topic.id && styles.topicTextSelected
              ]}>
                {topic.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Reassurance Message */}
      {reassurance && (
        <View style={styles.reassuranceCard}>
          <View style={styles.reassuranceHeader}>
            <Ionicons name="sunny" size={20} color="#f59e0b" />
            <Text style={styles.reassuranceTitle}>A Gentle Reminder</Text>
          </View>
          <Text style={styles.reassuranceText}>{reassurance.message}</Text>
          <View style={styles.perspectiveBox}>
            <Text style={styles.perspectiveLabel}>Perspective:</Text>
            <Text style={styles.perspectiveText}>{reassurance.perspective}</Text>
          </View>
        </View>
      )}

      {/* Grounding Techniques */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Grounding Exercises</Text>
        <Text style={styles.sectionSubtitle}>
          These can help calm your nervous system right now
        </Text>
        
        {groundingTechniques.slice(0, 3).map((technique: GroundingTechnique) => (
          <TouchableOpacity
            key={technique.id}
            style={styles.groundingCard}
            onPress={() => setExpandedGrounding(
              expandedGrounding === technique.id ? null : technique.id
            )}
          >
            <View style={styles.groundingHeader}>
              <Text style={styles.groundingName}>{technique.name}</Text>
              <Ionicons 
                name={expandedGrounding === technique.id ? 'chevron-up' : 'chevron-down'} 
                size={20} 
                color="#6b7280" 
              />
            </View>
            <Text style={styles.groundingDescription}>{technique.description}</Text>
            
            {expandedGrounding === technique.id && (
              <View style={styles.groundingSteps}>
                {technique.steps.map((step, index) => (
                  <View key={index} style={styles.stepRow}>
                    <View style={styles.stepNumber}>
                      <Text style={styles.stepNumberText}>{index + 1}</Text>
                    </View>
                    <Text style={styles.stepText}>{step}</Text>
                  </View>
                ))}
              </View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {/* Systems Analysis - How Things Actually Work */}
      {systemsAnalysis && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>How Systems Actually Work</Text>
          <View style={styles.systemsCard}>
            <Text style={styles.systemsReality}>{systemsAnalysis.reality}</Text>
            <View style={styles.factsList}>
              {systemsAnalysis.facts.slice(0, 4).map((fact, index) => (
                <View key={index} style={styles.factRow}>
                  <Ionicons name="checkmark-circle" size={16} color="#10b981" />
                  <Text style={styles.factText}>{fact}</Text>
                </View>
              ))}
            </View>
          </View>
        </View>
      )}

      {/* Media Literacy Tips */}
      {selectedTopic === 'media-anxiety' && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Media Literacy</Text>
          <Text style={styles.sectionSubtitle}>
            Helpful reminders when consuming news
          </Text>
          
          {mediaLiteracyTips.slice(0, 5).map((tip, index) => (
            <View key={index} style={styles.tipCard}>
              <Ionicons name="information-circle" size={20} color="#3b82f6" />
              <Text style={styles.tipText}>{tip}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Community Resources */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>You're Not Alone</Text>
        <View style={styles.communityCard}>
          <Text style={styles.communityText}>
            Communities across the country actively support their immigrant neighbors. 
            Local organizations, faith communities, and neighbors are here for you.
          </Text>
          <View style={styles.resourceRow}>
            <Ionicons name="call" size={16} color="#10b981" />
            <Text style={styles.resourceText}>
              National Immigrant Women's Advocacy Project: 1-202-274-4457
            </Text>
          </View>
          <View style={styles.resourceRow}>
            <Ionicons name="call" size={16} color="#10b981" />
            <Text style={styles.resourceText}>
              United We Dream: 1-844-363-1423
            </Text>
          </View>
        </View>
      </View>

      {/* Closing Affirmation */}
      <View style={styles.affirmationCard}>
        <Text style={styles.affirmationText}>
          "Your presence here matters. Your story matters. You belong."
        </Text>
      </View>

      <View style={styles.spacer} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
  },
  subtitle: {
    fontSize: 15,
    color: '#9ca3af',
    marginBottom: 20,
    lineHeight: 22,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  sectionSubtitle: {
    fontSize: 13,
    color: '#6b7280',
    marginBottom: 12,
  },
  topicGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  topicButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 10,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  topicButtonSelected: {
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    borderColor: 'rgba(16, 185, 129, 0.4)',
  },
  topicText: {
    fontSize: 13,
    color: '#9ca3af',
  },
  topicTextSelected: {
    color: '#10b981',
  },
  reassuranceCard: {
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.2)',
  },
  reassuranceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  reassuranceTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#f59e0b',
  },
  reassuranceText: {
    fontSize: 15,
    color: '#fff',
    lineHeight: 22,
    marginBottom: 12,
  },
  perspectiveBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    borderRadius: 8,
    padding: 12,
  },
  perspectiveLabel: {
    fontSize: 11,
    color: '#f59e0b',
    fontWeight: '600',
    marginBottom: 4,
  },
  perspectiveText: {
    fontSize: 13,
    color: '#d1d5db',
    lineHeight: 20,
  },
  groundingCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
  },
  groundingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  groundingName: {
    fontSize: 15,
    fontWeight: '600',
    color: '#10b981',
  },
  groundingDescription: {
    fontSize: 13,
    color: '#9ca3af',
    lineHeight: 20,
  },
  groundingSteps: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  stepNumber: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: 'rgba(16, 185, 129, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  stepNumberText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#10b981',
  },
  stepText: {
    flex: 1,
    fontSize: 14,
    color: '#d1d5db',
    lineHeight: 20,
  },
  systemsCard: {
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.2)',
  },
  systemsReality: {
    fontSize: 14,
    color: '#fff',
    lineHeight: 22,
    marginBottom: 12,
  },
  factsList: {
    gap: 8,
  },
  factRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
  },
  factText: {
    flex: 1,
    fontSize: 13,
    color: '#d1d5db',
    lineHeight: 20,
  },
  tipCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
    borderRadius: 10,
    padding: 12,
    marginBottom: 8,
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    color: '#d1d5db',
    lineHeight: 20,
  },
  communityCard: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 12,
    padding: 14,
  },
  communityText: {
    fontSize: 14,
    color: '#d1d5db',
    lineHeight: 22,
    marginBottom: 12,
  },
  resourceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 8,
  },
  resourceText: {
    fontSize: 13,
    color: '#10b981',
  },
  affirmationCard: {
    backgroundColor: 'rgba(139, 92, 246, 0.15)',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
  },
  affirmationText: {
    fontSize: 16,
    color: '#c4b5fd',
    fontStyle: 'italic',
    textAlign: 'center',
    lineHeight: 24,
  },
  spacer: {
    height: 40,
  },
});
