import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import GuidedMeditationLibrary from '../components/GuidedMeditationLibrary';

export default function MeditationScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <GuidedMeditationLibrary />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
