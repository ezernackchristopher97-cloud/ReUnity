import React, { useState } from 'react';
import { SafeAreaView, StyleSheet, StatusBar, Modal, View, Text, TouchableOpacity, Linking } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import MoodPrediction from '../components/MoodPrediction';

export default function MoodPredictionScreen() {
  const [showHighRiskAlert, setShowHighRiskAlert] = useState(false);

  const handleHighRiskDetected = () => {
    setShowHighRiskAlert(true);
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#09090b" />
      <MoodPrediction onHighRiskDetected={handleHighRiskDetected} />

      {/* High Risk Alert Modal */}
      <Modal
        visible={showHighRiskAlert}
        transparent
        animationType="fade"
        onRequestClose={() => setShowHighRiskAlert(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.alertIcon}>
              <Ionicons name="heart" size={32} color="#ef4444" />
            </View>
            
            <Text style={styles.modalTitle}>We're Here For You</Text>
            <Text style={styles.modalSubtitle}>
              Your wellness data suggests you may need support
            </Text>
            
            <Text style={styles.modalText}>
              Would you like to reach out to someone? One tap to call your emergency contact or crisis line.
            </Text>

            <TouchableOpacity
              style={styles.crisisButton}
              onPress={() => {
                Linking.openURL('tel:988');
                setShowHighRiskAlert(false);
              }}
            >
              <Ionicons name="call" size={20} color="#fff" />
              <Text style={styles.crisisButtonText}>Call 988 Crisis Lifeline</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.textButton}
              onPress={() => {
                Linking.openURL('sms:741741?body=HELLO');
                setShowHighRiskAlert(false);
              }}
            >
              <Ionicons name="chatbubble" size={20} color="#3b82f6" />
              <Text style={styles.textButtonText}>Text HOME to 741741</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.dismissButton}
              onPress={() => setShowHighRiskAlert(false)}
            >
              <Text style={styles.dismissText}>I'm okay for now</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    backgroundColor: '#18181b',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.5)',
  },
  alertIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#fff',
    textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: 14,
    color: '#a1a1aa',
    textAlign: 'center',
    marginBottom: 16,
  },
  modalText: {
    fontSize: 14,
    color: '#d4d4d8',
    textAlign: 'center',
    marginBottom: 24,
  },
  crisisButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#ef4444',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  crisisButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  textButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: '#3b82f6',
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
  },
  textButtonText: {
    color: '#3b82f6',
    fontWeight: '600',
    fontSize: 16,
  },
  dismissButton: {
    padding: 12,
    alignItems: 'center',
  },
  dismissText: {
    color: '#71717a',
    fontSize: 14,
  },
});
