import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Modal, ScrollView, TextInput } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { getAllLanguages, searchLanguages, getGreeting, getComfortingPhrase, Language } from '../lib/languages';

interface LanguageSelectorProps {
  selectedLanguage?: string;
  onSelectLanguage: (languageId: string) => void;
  compact?: boolean;
}

export default function LanguageSelector({ 
  selectedLanguage, 
  onSelectLanguage, 
  compact = false 
}: LanguageSelectorProps) {
  const [showModal, setShowModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  const allLanguages = getAllLanguages();
  const filteredLanguages = searchQuery 
    ? searchLanguages(searchQuery)
    : allLanguages;

  const selectedLang = selectedLanguage 
    ? allLanguages.find(l => l.id === selectedLanguage)
    : null;

  const handleSelect = (langId: string) => {
    onSelectLanguage(langId);
    setShowModal(false);
    setSearchQuery('');
  };

  // Group languages by region
  const groupedLanguages: Record<string, Language[]> = {
    'Native American': filteredLanguages.filter(l => 
      ['navajo', 'cherokee', 'lakota', 'ojibwe', 'apache'].includes(l.id)
    ),
    'South Asian': filteredLanguages.filter(l => 
      ['hindi', 'urdu', 'punjabi', 'bengali', 'tamil', 'telugu', 'gujarati'].includes(l.id)
    ),
    'Middle Eastern': filteredLanguages.filter(l => 
      ['arabic', 'farsi', 'turkish', 'hebrew'].includes(l.id)
    ),
    'East Asian': filteredLanguages.filter(l => 
      ['mandarin', 'cantonese', 'japanese', 'korean', 'vietnamese', 'tagalog'].includes(l.id)
    ),
    'African': filteredLanguages.filter(l => 
      ['swahili', 'amharic', 'somali'].includes(l.id)
    ),
    'European': filteredLanguages.filter(l => 
      ['spanish', 'portuguese', 'french', 'german', 'italian', 'polish', 'russian', 'ukrainian'].includes(l.id)
    ),
  };

  if (compact) {
    return (
      <>
        <TouchableOpacity 
          style={styles.compactButton}
          onPress={() => setShowModal(true)}
        >
          <Ionicons name="language" size={20} color="#10b981" />
          <Text style={styles.compactText}>
            {selectedLang ? selectedLang.nativeName : 'Language'}
          </Text>
        </TouchableOpacity>

        <Modal
          visible={showModal}
          animationType="slide"
          transparent={true}
          onRequestClose={() => setShowModal(false)}
        >
          <LanguageModalContent />
        </Modal>
      </>
    );
  }

  const LanguageModalContent = () => (
    <View style={styles.modalOverlay}>
      <View style={styles.modalContent}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Select Your Language</Text>
          <TouchableOpacity onPress={() => setShowModal(false)}>
            <Ionicons name="close" size={24} color="#fff" />
          </TouchableOpacity>
        </View>

        <Text style={styles.modalSubtitle}>
          All languages and communities are welcome here
        </Text>

        <View style={styles.searchContainer}>
          <Ionicons name="search" size={20} color="#6b7280" />
          <TextInput
            style={styles.searchInput}
            placeholder="Search languages..."
            placeholderTextColor="#6b7280"
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
        </View>

        <ScrollView style={styles.languageList}>
          {Object.entries(groupedLanguages).map(([region, languages]) => {
            if (languages.length === 0) return null;
            return (
              <View key={region} style={styles.regionGroup}>
                <Text style={styles.regionTitle}>{region}</Text>
                {languages.map(lang => (
                  <TouchableOpacity
                    key={lang.id}
                    style={[
                      styles.languageOption,
                      selectedLanguage === lang.id && styles.languageOptionSelected
                    ]}
                    onPress={() => handleSelect(lang.id)}
                  >
                    <View style={styles.languageInfo}>
                      <Text style={[
                        styles.languageName,
                        selectedLanguage === lang.id && styles.languageNameSelected
                      ]}>
                        {lang.nativeName}
                      </Text>
                      <Text style={styles.languageEnglish}>{lang.name}</Text>
                      {lang.communities && (
                        <Text style={styles.communities}>
                          {lang.communities.slice(0, 2).join(', ')}
                        </Text>
                      )}
                    </View>
                    {selectedLanguage === lang.id && (
                      <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                    )}
                  </TouchableOpacity>
                ))}
              </View>
            );
          })}
        </ScrollView>

        {selectedLang && (
          <View style={styles.previewSection}>
            <Text style={styles.previewTitle}>Greeting in {selectedLang.name}:</Text>
            <Text style={styles.previewText}>{getGreeting(selectedLang.id)}</Text>
          </View>
        )}
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <TouchableOpacity 
        style={styles.selectorButton}
        onPress={() => setShowModal(true)}
      >
        <Ionicons name="language" size={24} color="#10b981" />
        <View style={styles.selectorInfo}>
          <Text style={styles.selectorLabel}>Language</Text>
          <Text style={styles.selectorValue}>
            {selectedLang ? selectedLang.nativeName : 'Select your language'}
          </Text>
        </View>
        <Ionicons name="chevron-forward" size={20} color="#6b7280" />
      </TouchableOpacity>

      <Modal
        visible={showModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowModal(false)}
      >
        <LanguageModalContent />
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
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 8,
  },
  compactText: {
    color: '#10b981',
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
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 10,
    paddingHorizontal: 12,
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 8,
    color: '#fff',
    fontSize: 16,
  },
  languageList: {
    maxHeight: 400,
  },
  regionGroup: {
    marginBottom: 16,
  },
  regionTitle: {
    fontSize: 12,
    fontWeight: '600',
    color: '#10b981',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 8,
    paddingLeft: 4,
  },
  languageOption: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 10,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  languageOptionSelected: {
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderColor: 'rgba(16, 185, 129, 0.3)',
  },
  languageInfo: {
    flex: 1,
  },
  languageName: {
    fontSize: 16,
    color: '#fff',
    fontWeight: '500',
  },
  languageNameSelected: {
    color: '#10b981',
  },
  languageEnglish: {
    fontSize: 13,
    color: '#9ca3af',
  },
  communities: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 2,
  },
  previewSection: {
    marginTop: 16,
    padding: 12,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 10,
  },
  previewTitle: {
    fontSize: 12,
    color: '#10b981',
    marginBottom: 4,
  },
  previewText: {
    fontSize: 16,
    color: '#fff',
  },
});
