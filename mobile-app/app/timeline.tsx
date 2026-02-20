import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import CrisisInterventionTimeline from '../components/CrisisInterventionTimeline';

export default function TimelineScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <CrisisInterventionTimeline />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
});
