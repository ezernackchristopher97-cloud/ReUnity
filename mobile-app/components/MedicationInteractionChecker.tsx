import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface Medication {
  id: string;
  name: string;
  category: string;
}

interface Interaction {
  medications: [string, string];
  severity: 'mild' | 'moderate' | 'severe';
  description: string;
  recommendation: string;
}

const MEDICATION_DATABASE: Record<string, { category: string; aliases: string[] }> = {
  'sertraline': { category: 'SSRI', aliases: ['zoloft'] },
  'fluoxetine': { category: 'SSRI', aliases: ['prozac'] },
  'escitalopram': { category: 'SSRI', aliases: ['lexapro'] },
  'lithium': { category: 'Mood Stabilizer', aliases: ['lithobid'] },
  'lamotrigine': { category: 'Mood Stabilizer', aliases: ['lamictal'] },
  'quetiapine': { category: 'Antipsychotic', aliases: ['seroquel'] },
  'alprazolam': { category: 'Benzodiazepine', aliases: ['xanax'] },
  'lorazepam': { category: 'Benzodiazepine', aliases: ['ativan'] },
  'bupropion': { category: 'Antidepressant', aliases: ['wellbutrin'] },
  'trazodone': { category: 'Sleep Aid', aliases: ['desyrel'] },
};

const INTERACTIONS: Interaction[] = [
  { medications: ['sertraline', 'lithium'], severity: 'moderate', description: 'Increased serotonin effects', recommendation: 'Monitor for tremor, confusion' },
  { medications: ['alprazolam', 'quetiapine'], severity: 'moderate', description: 'Enhanced sedation', recommendation: 'Use lower doses' },
  { medications: ['lithium', 'ibuprofen'], severity: 'moderate', description: 'Increased lithium levels', recommendation: 'Use acetaminophen instead' },
];

export default function MedicationInteractionChecker() {
  const [medications, setMedications] = useState<Medication[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [interactions, setInteractions] = useState<Interaction[]>([]);

  const searchMedications = (term: string) => {
    setSearchTerm(term);
    if (term.length < 2) {
      setSearchResults([]);
      return;
    }
    
    const lowerTerm = term.toLowerCase();
    const results: string[] = [];
    
    for (const [name, data] of Object.entries(MEDICATION_DATABASE)) {
      if (name.includes(lowerTerm) || data.aliases.some(a => a.includes(lowerTerm))) {
        results.push(name);
      }
    }
    
    setSearchResults(results.slice(0, 5));
  };

  const addMedication = (name: string) => {
    const medData = MEDICATION_DATABASE[name];
    if (!medData || medications.some(m => m.name === name)) return;
    
    const newMed: Medication = {
      id: Date.now().toString(),
      name,
      category: medData.category,
    };
    
    const newMedications = [...medications, newMed];
    setMedications(newMedications);
    setSearchTerm('');
    setSearchResults([]);
    checkInteractions(newMedications);
  };

  const removeMedication = (id: string) => {
    const newMedications = medications.filter(m => m.id !== id);
    setMedications(newMedications);
    checkInteractions(newMedications);
  };

  const checkInteractions = (meds: Medication[]) => {
    const foundInteractions: Interaction[] = [];
    
    for (let i = 0; i < meds.length; i++) {
      for (let j = i + 1; j < meds.length; j++) {
        const med1 = meds[i].name;
        const med2 = meds[j].name;
        
        for (const interaction of INTERACTIONS) {
          if (
            (interaction.medications[0] === med1 && interaction.medications[1] === med2) ||
            (interaction.medications[0] === med2 && interaction.medications[1] === med1)
          ) {
            foundInteractions.push(interaction);
          }
        }
      }
    }
    
    setInteractions(foundInteractions);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'mild': return '#EAB308';
      case 'moderate': return '#F97316';
      case 'severe': return '#EF4444';
      default: return '#6B7280';
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="medical" size={24} color="#A855F7" />
        <Text style={styles.title}>Medication Checker</Text>
      </View>
      <Text style={styles.subtitle}>Track medications and check for interactions</Text>

      {/* Search */}
      <View style={styles.searchContainer}>
        <Ionicons name="search" size={20} color="#6B7280" style={styles.searchIcon} />
        <TextInput
          style={styles.searchInput}
          placeholder="Search medications..."
          placeholderTextColor="#6B7280"
          value={searchTerm}
          onChangeText={searchMedications}
        />
      </View>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <View style={styles.resultsContainer}>
          {searchResults.map((name) => (
            <TouchableOpacity
              key={name}
              style={styles.resultItem}
              onPress={() => addMedication(name)}
            >
              <Text style={styles.resultName}>{name}</Text>
              <Text style={styles.resultCategory}>{MEDICATION_DATABASE[name].category}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      {/* Current Medications */}
      {medications.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Your Medications</Text>
          <View style={styles.medicationList}>
            {medications.map((med) => (
              <View key={med.id} style={styles.medicationBadge}>
                <Text style={styles.medicationName}>{med.name}</Text>
                <TouchableOpacity onPress={() => removeMedication(med.id)}>
                  <Ionicons name="close" size={16} color="#9CA3AF" />
                </TouchableOpacity>
              </View>
            ))}
          </View>
        </View>
      )}

      {/* Interactions */}
      {interactions.length > 0 && (
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: '#EF4444' }]}>⚠️ Interactions Detected</Text>
          {interactions.map((interaction, i) => (
            <View key={i} style={[styles.interactionCard, { borderColor: getSeverityColor(interaction.severity) }]}>
              <View style={styles.interactionHeader}>
                <Text style={styles.interactionMeds}>
                  {interaction.medications[0]} + {interaction.medications[1]}
                </Text>
                <View style={[styles.severityBadge, { backgroundColor: getSeverityColor(interaction.severity) + '30' }]}>
                  <Text style={[styles.severityText, { color: getSeverityColor(interaction.severity) }]}>
                    {interaction.severity}
                  </Text>
                </View>
              </View>
              <Text style={styles.interactionDesc}>{interaction.description}</Text>
              <Text style={styles.interactionRec}>💡 {interaction.recommendation}</Text>
            </View>
          ))}
        </View>
      )}

      {medications.length > 0 && interactions.length === 0 && (
        <View style={styles.safeCard}>
          <Ionicons name="checkmark-circle" size={24} color="#10B981" />
          <Text style={styles.safeText}>No interactions detected</Text>
        </View>
      )}

      {/* Disclaimer */}
      <View style={styles.disclaimer}>
        <Text style={styles.disclaimerText}>
          ReUnity is a wellness and support tool. It is not a medical device and does not provide diagnosis, treatment, or crisis services. If you are in immediate danger, call 911 or your local emergency services.
        </Text>
        <Text style={[styles.disclaimerText, { marginTop: 8 }]}>
          This tool provides general information only. Always consult your doctor or pharmacist.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', padding: 16 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  title: { fontSize: 20, fontWeight: 'bold', color: '#F8FAFC' },
  subtitle: { fontSize: 14, color: '#94A3B8', marginTop: 4, marginBottom: 16 },
  searchContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#1E293B', borderRadius: 8, paddingHorizontal: 12 },
  searchIcon: { marginRight: 8 },
  searchInput: { flex: 1, height: 44, color: '#F8FAFC', fontSize: 16 },
  resultsContainer: { backgroundColor: '#1E293B', borderRadius: 8, marginTop: 8 },
  resultItem: { padding: 12, borderBottomWidth: 1, borderBottomColor: '#334155' },
  resultName: { fontSize: 16, color: '#F8FAFC', textTransform: 'capitalize' },
  resultCategory: { fontSize: 12, color: '#94A3B8' },
  section: { marginTop: 20 },
  sectionTitle: { fontSize: 16, fontWeight: '600', color: '#F8FAFC', marginBottom: 12 },
  medicationList: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  medicationBadge: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#334155', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 16 },
  medicationName: { color: '#F8FAFC', textTransform: 'capitalize' },
  interactionCard: { backgroundColor: '#1E293B', borderRadius: 8, padding: 12, marginBottom: 8, borderLeftWidth: 3 },
  interactionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  interactionMeds: { fontSize: 14, fontWeight: '600', color: '#F8FAFC', textTransform: 'capitalize' },
  severityBadge: { paddingVertical: 2, paddingHorizontal: 8, borderRadius: 4 },
  severityText: { fontSize: 12, fontWeight: '600', textTransform: 'capitalize' },
  interactionDesc: { fontSize: 14, color: '#CBD5E1', marginBottom: 4 },
  interactionRec: { fontSize: 12, color: '#10B981' },
  safeCard: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: '#10B98120', padding: 12, borderRadius: 8, marginTop: 16 },
  safeText: { color: '#10B981', fontSize: 14 },
  disclaimer: { backgroundColor: '#1E293B', padding: 12, borderRadius: 8, marginTop: 20 },
  disclaimerText: { fontSize: 12, color: '#94A3B8', textAlign: 'center' },
});
