import React from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import CommunitySupportGroups from '../components/CommunitySupportGroups';

export default function CommunityScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <CommunitySupportGroups />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
});
