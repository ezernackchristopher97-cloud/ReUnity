import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import TherapistNotesSync from '../components/TherapistNotesSync';

export default function NotesScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <TherapistNotesSync />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
