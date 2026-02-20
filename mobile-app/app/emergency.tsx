import React from 'react';
import { SafeAreaView, StyleSheet, StatusBar } from 'react-native';
import EmergencyContacts from '../components/EmergencyContacts';

export default function EmergencyScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#09090b" />
      <EmergencyContacts />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
});
