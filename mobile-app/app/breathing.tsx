import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import BreathingExercises from '../components/BreathingExercises';

export default function BreathingScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <BreathingExercises />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
