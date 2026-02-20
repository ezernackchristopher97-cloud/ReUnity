import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import MedicationInteractionChecker from '../components/MedicationInteractionChecker';

export default function MedicationsScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <MedicationInteractionChecker />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
});
