import React from 'react';
import { SafeAreaView, StyleSheet, StatusBar } from 'react-native';
import JournalWithSentiment from '../components/JournalWithSentiment';

export default function JournalScreen() {
  const handleCrisisDetected = (entry: any) => {
    console.log('Crisis detected in journal entry:', entry.id);
    // Could trigger emergency contact alert here
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#09090b" />
      <JournalWithSentiment 
        onCrisisDetected={handleCrisisDetected}
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
});
