import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import MedicationReminders from '../components/MedicationReminders';

export default function RemindersScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <MedicationReminders />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
