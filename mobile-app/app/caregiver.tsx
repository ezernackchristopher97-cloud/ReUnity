import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import CaregiverDashboard from '../components/CaregiverDashboard';

export default function CaregiverScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <CaregiverDashboard />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0c',
  },
});
