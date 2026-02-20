import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import Gamification from '../components/Gamification';

export default function AchievementsScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <Gamification />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0c',
  },
});
