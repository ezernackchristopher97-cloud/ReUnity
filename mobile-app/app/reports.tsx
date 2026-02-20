import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import WellnessReportExport from '../components/WellnessReportExport';

export default function ReportsScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <WellnessReportExport />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
