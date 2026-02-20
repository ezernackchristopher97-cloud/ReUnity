import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import GroupTherapySessions from '../components/GroupTherapySessions';

export default function GroupsScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <GroupTherapySessions />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0c',
  },
});
