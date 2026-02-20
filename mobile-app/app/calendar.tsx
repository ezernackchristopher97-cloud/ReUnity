import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import MoodCalendar from '../components/MoodCalendar';

export default function CalendarScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <MoodCalendar />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
  },
});
