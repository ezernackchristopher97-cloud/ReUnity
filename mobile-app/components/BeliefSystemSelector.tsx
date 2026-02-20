import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { 
  getAllBeliefSystems, 
  getBeliefSystem, 
  getComfortingPhrase, 
  getCopingStrategies,
  BeliefSystem 
} from '../lib/belief-systems';

interface BeliefSystemSelectorProps {
  selectedBelief?: string;
  onSelectBelief: (beliefId: string) => void;
  compact?: boolean;
}

export default function BeliefSystemSelector({ 
  selectedBelief, 
  onSelectBelief, 
  compact = false 
}: BeliefSystemSelectorProps) {
  const [showModal, setShowModal] = useState(false);
  
  const allBeliefs = getAllBeliefSystems();
  const selectedBeliefSystem = selectedBelief ? getBeliefSystem(selectedBelief) : null;

  // Group beliefs by category
  const groupedBeliefs: Record<string, BeliefSystem[]> = {
    'Abrahamic Religions': allBeliefs.filter(b => 
      ['christianity', 'islam', 'judaism'].includes(b.id)
    ),
    'Eastern Religions': allBeliefs.filter(b => 
      ['buddhism', 'hinduism', 'sikhism', 'taoism', 'shinto', 'jainism'].includes(b.id)
    ),
    'Philosophical Frameworks': allBeliefs.filter(b => 
      ['existentialism', 'stoicism', 'nihilism', 'absurdism', 'solipsism'].includes(b.id)
    ),
    'Spiritual Traditions': allBeliefs.filter(b => 
      ['paganism', 'wicca', 'shamanism', 'animism', 'new-age'].includes(b.id)
    ),
    'Secular Perspectives': allBeliefs.filter(b => 
      ['atheism', 'agnosticism', 'secular-humanism', 'universalism'].includes(b.id)
    ),
  };

  const handleSelect = (beliefId: string) => {
    onSelectBelief(beliefId);
    setShowModal(false);
  };

  const BeliefModalContent = () => (
    <View style={styles.modalOverlay}>
      <View style={styles.modalContent}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Your Spiritual Path</Text>
          <TouchableOpacity onPress={() => setShowModal(false)}>
            <Ionicons name="close" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        <Text style={styles.modalSubtitle}>
          All beliefs are honored here. Choose what resonates with you, or skip this step.
        </Text>

        <ScrollView style={styles.beliefList}>
          {/* Option to skip */}
          <TouchableOpacity
            style={[
              styles.beliefOption,
              !selectedBelief && styles.beliefOptionSelected
            ]}
            onPress={() => handleSelect('')}
          >
            <View style={styles.beliefInfo}>
              <Text style={[
                styles.beliefName,
                !selectedBelief && styles.beliefNameSelected
              ]}>
                No preference / Universal
              </Text>
              <Text style={styles.beliefDescription}>
                Use inclusive, universal language
              </Text>
            </View>
            {!selectedBelief && (
              <Ionicons name="checkmark-circle" size={24} color="#10b981" />
            )}
          </TouchableOpacity>

          {Object.entries(groupedBeliefs).map(([category, beliefs]) => {
            if (beliefs.length === 0) return null;
            return (
              <View key={category} style={styles.categoryGroup}>
                <Text style={styles.categoryTitle}>{category}</Text>
                {beliefs.map(belief => (
                  <TouchableOpacity
                    key={belief.id}
                    style={[
                      styles.beliefOption,
                      selectedBelief === belief.id && styles.beliefOptionSelected
                    ]}
                    onPress={() => handleSelect(belief.id)}
                  >
                    <View style={styles.beliefInfo}>
                      <Text style={[
                        styles.beliefName,
                        selectedBelief === belief.id && styles.beliefNameSelected
                      ]}>
                        {belief.name}
                      </Text>
                      <Text style={styles.beliefDescription} numberOfLines={2}>
                        {belief.coreBeliefs[0]}
                      </Text>
                    </View>
                    {selectedBelief === belief.id && (
                      <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            );
          })}
        </ScrollView>

        {/* Preview of selected belief */}
        {selectedBeliefSystem && (
          <View style={styles.previewSection}>
            <Text style={styles.previewTitle}>
              Comfort from {selectedBeliefSystem.name}:
            </Text>
            <Text style={styles.previewText}>
              {getComfortingPhrase(selectedBeliefSystem.id)}
            </Text>
          </View>
        )}
      </View>
    </View>
  );

  if (compact) {
    return (
      <>
        <TouchableOpacity 
          style={styles.compactButton}
          onPress={() => setShowModal(true)}
        >
          <Ionicons name="sparkles" size={20} color="#8b5cf6" />
          <Text style={styles.compactText}>
            {selectedBeliefSystem ? selectedBeliefSystem.name : 'Beliefs'}
          </Text>
        </TouchableOpacity>

        <Modal
          visible={showModal}
          animationType="slide"
          transparent={true}
          onRequestClose={() => setShowModal(false)}
        >
          <BeliefModalContent />
        </Modal>
      </>
    );
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity 
        style={styles.selectorButton}
        onPress={() => setShowModal(true)}
      >
        <Ionicons name="sparkles" size={24} color="#8b5cf6" />
        <View style={styles.selectorInfo}>
          <Text style={styles.selectorLabel}>Spiritual/Philosophical Background</Text>
          <Text style={styles.selectorValue}>
            {selectedBeliefSystem ? selectedBeliefSystem.name : 'Universal (no preference)'}
          </Text>
        </View>
        <Ionicons name="chevron-forward" size={20} color="#6b7280" />
      </TouchableOpacity>

      {/* Show coping strategies if belief is selected */}
      {selectedBeliefSystem && (
        <View style={styles.copingSection}>
          <Text style={styles.copingTitle}>
            Coping strategies from {selectedBeliefSystem.name}:
          </Text>
          {getCopingStrategies(selectedBeliefSystem.id).slice(0, 3).map((strategy, index) => (
            <View key={index} style={styles.copingItem}>
              <Ionicons name="leaf" size={14} color="#8b5cf6" />
              <Text style={styles.copingText}>{strategy}</Text>
            </View>
          ))}
        </View>
      )}

      <Modal
        visible={showModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowModal(false)}
      >
        <BeliefModalContent />
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: 8,
  },
  compactButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    padding: 8,
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
    borderRadius: 8,
  },
  compactText: {
    color: '#8b5cf6',
    fontSize: 14,
  },
  selectorButton: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  selectorInfo: {
    flex: 1,
    marginLeft: 12,
  },
  selectorLabel: {
    fontSize: 12,
    color: '#6b7280',
  },
  selectorValue: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '500',
  },
  copingSection: {
    marginTop: 12,
    padding: 12,
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
    borderRadius: 10,
  },
  copingTitle: {
    fontSize: 12,
    color: '#8b5cf6',
    fontWeight: '600',
    marginBottom: 8,
  },
  copingItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginBottom: 6,
  },
  copingText: {
    flex: 1,
    fontSize: 13,
    color: '#d1d5db',
    lineHeight: 18,
  },
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
    maxHeight: '85%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#9ca3af',
    marginBottom: 16,
    lineHeight: 20,
  },
  beliefList: {
    maxHeight: 400,
  },
  categoryGroup: {
    marginBottom: 16,
  },
  categoryTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#8b5cf6',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    paddingLeft: 4,
  },
  beliefOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 10,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  beliefOptionSelected: {
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
    borderColor: 'rgba(139, 92, 246, 0.3)',
  },
  beliefInfo: {
    flex: 1,
  },
  beliefName: {
    fontSize: 15,
    color: '#fff',
    fontWeight: '500',
    marginBottom: 2,
  },
  beliefNameSelected: {
    color: '#8b5cf6',
  },
  beliefDescription: {
    fontSize: 12,
    color: '#6b7280',
    lineHeight: 16,
  },
  previewSection: {
    marginTop: 16,
    padding: 12,
    backgroundColor: 'rgba(139, 92, 246, 0.1)',
    borderRadius: 10,
  },
  previewTitle: {
    fontSize: 12,
    color: '#8b5cf6',
    marginBottom: 4,
  },
  previewText: {
    fontSize: 14,
    color: '#fff',
    fontStyle: 'italic',
  },
});
