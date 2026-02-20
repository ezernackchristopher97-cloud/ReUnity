import React from 'react';
import { View, StyleSheet, SafeAreaView, TouchableOpacity, Text } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import FamilyGroupChat from '../components/FamilyGroupChat';

export default function FamilyScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container}>
      <FamilyGroupChat />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#09090b' },
});
