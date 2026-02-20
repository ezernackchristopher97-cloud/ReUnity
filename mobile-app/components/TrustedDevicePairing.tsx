import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
  TextInput,
  Alert,
  Modal,
  Share,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';

interface TrustedDevice {
  id: string;
  name: string;
  relationship: string;
  pairedAt: string;
  lastSeen: string;
  permissions: {
    location: boolean;
    wellness: boolean;
    crisisAlerts: boolean;
  };
}

interface PairingCode {
  code: string;
  expiresAt: number;
}

const STORAGE_KEY = 'reunity_trusted_devices';

export default function TrustedDevicePairing() {
  const [devices, setDevices] = useState<TrustedDevice[]>([]);
  const [pairingCode, setPairingCode] = useState<PairingCode | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEnterCodeModal, setShowEnterCodeModal] = useState(false);
  const [enteredCode, setEnteredCode] = useState('');
  const [deviceName, setDeviceName] = useState('');
  const [relationship, setRelationship] = useState('');

  useEffect(() => {
    loadDevices();
  }, []);

  const loadDevices = async () => {
    try {
      const saved = await AsyncStorage.getItem(STORAGE_KEY);
      if (saved) {
        setDevices(JSON.parse(saved));
      }
    } catch (e) {
      console.error('Failed to load devices:', e);
    }
  };

  const saveDevices = async (newDevices: TrustedDevice[]) => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(newDevices));
      setDevices(newDevices);
    } catch (e) {
      console.error('Failed to save devices:', e);
    }
  };

  const generatePairingCode = (): string => {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  };

  const createPairingCode = () => {
    const code = generatePairingCode();
    const expiresAt = Date.now() + 10 * 60 * 1000; // 10 minutes
    setPairingCode({ code, expiresAt });
    setShowAddModal(true);
  };

  const sharePairingCode = async () => {
    if (!pairingCode) return;
    
    try {
      await Share.share({
        message: `Connect with me on ReUnity!\n\nPairing Code: ${pairingCode.code}\n\nThis code expires in 10 minutes.\n\nDownload ReUnity to connect.`,
        title: 'ReUnity Pairing Code',
      });
    } catch (e) {
      console.error('Failed to share:', e);
    }
  };

  const handleEnterCode = () => {
    if (enteredCode.length !== 6) {
      Alert.alert('Invalid Code', 'Please enter a 6-character pairing code.');
      return;
    }

    // Simulate pairing (in real app, this would verify with server)
    const newDevice: TrustedDevice = {
      id: `dev_${Date.now()}`,
      name: deviceName || 'Family Member',
      relationship: relationship || 'Family',
      pairedAt: new Date().toISOString(),
      lastSeen: new Date().toISOString(),
      permissions: {
        location: true,
        wellness: true,
        crisisAlerts: true,
      },
    };

    saveDevices([...devices, newDevice]);
    setShowEnterCodeModal(false);
    setEnteredCode('');
    setDeviceName('');
    setRelationship('');
    Alert.alert('Success', 'Device paired successfully!');
  };

  const updatePermission = (
    deviceId: string,
    permission: keyof TrustedDevice['permissions'],
    value: boolean
  ) => {
    const updated = devices.map((d) =>
      d.id === deviceId
        ? { ...d, permissions: { ...d.permissions, [permission]: value } }
        : d
    );
    saveDevices(updated);
  };

  const removeDevice = (deviceId: string) => {
    Alert.alert(
      'Remove Device',
      'Are you sure you want to remove this trusted device?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: () => {
            const updated = devices.filter((d) => d.id !== deviceId);
            saveDevices(updated);
          },
        },
      ]
    );
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <View style={styles.headerRow}>
          <Ionicons name="link" size={24} color="#10b981" />
          <Text style={styles.title}>Trusted Devices</Text>
        </View>
        <Text style={styles.subtitle}>
          Share your location and wellness data with trusted family members
        </Text>
      </View>

      {/* Info Banner */}
      <View style={styles.infoBanner}>
        <Ionicons name="shield-checkmark" size={20} color="#10b981" />
        <Text style={styles.infoText}>
          Paired devices can see your wellness status and receive crisis alerts.
          You control what data is shared.
        </Text>
      </View>

      {/* Add Device Buttons */}
      <View style={styles.addButtons}>
        <TouchableOpacity
          style={styles.addButton}
          onPress={createPairingCode}
        >
          <Ionicons name="qr-code" size={24} color="#10b981" />
          <Text style={styles.addButtonText}>Generate Code</Text>
          <Text style={styles.addButtonSubtext}>Share with family</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.addButton}
          onPress={() => setShowEnterCodeModal(true)}
        >
          <Ionicons name="keypad" size={24} color="#3b82f6" />
          <Text style={styles.addButtonText}>Enter Code</Text>
          <Text style={styles.addButtonSubtext}>Connect to family</Text>
        </TouchableOpacity>
      </View>

      {/* Devices List */}
      {devices.length > 0 && (
        <View style={styles.devicesList}>
          <Text style={styles.sectionTitle}>Paired Devices</Text>
          {devices.map((device) => (
            <View key={device.id} style={styles.deviceCard}>
              <View style={styles.deviceHeader}>
                <View style={styles.deviceInfo}>
                  <View style={styles.deviceAvatar}>
                    <Text style={styles.deviceAvatarText}>
                      {device.name.charAt(0).toUpperCase()}
                    </Text>
                  </View>
                  <View>
                    <Text style={styles.deviceName}>{device.name}</Text>
                    <Text style={styles.deviceRelationship}>
                      {device.relationship} • Paired {formatDate(device.pairedAt)}
                    </Text>
                  </View>
                </View>
                <TouchableOpacity
                  onPress={() => removeDevice(device.id)}
                  style={styles.removeButton}
                >
                  <Ionicons name="trash-outline" size={18} color="#ef4444" />
                </TouchableOpacity>
              </View>

              <View style={styles.permissionsSection}>
                <Text style={styles.permissionsTitle}>Shared Data</Text>
                
                <View style={styles.permissionRow}>
                  <View style={styles.permissionInfo}>
                    <Ionicons name="location" size={16} color="#3b82f6" />
                    <Text style={styles.permissionText}>Location</Text>
                  </View>
                  <Switch
                    value={device.permissions.location}
                    onValueChange={(v) => updatePermission(device.id, 'location', v)}
                    trackColor={{ false: '#3f3f46', true: '#2563eb' }}
                    thumbColor={device.permissions.location ? '#3b82f6' : '#71717a'}
                  />
                </View>

                <View style={styles.permissionRow}>
                  <View style={styles.permissionInfo}>
                    <Ionicons name="heart" size={16} color="#ec4899" />
                    <Text style={styles.permissionText}>Wellness Status</Text>
                  </View>
                  <Switch
                    value={device.permissions.wellness}
                    onValueChange={(v) => updatePermission(device.id, 'wellness', v)}
                    trackColor={{ false: '#3f3f46', true: '#db2777' }}
                    thumbColor={device.permissions.wellness ? '#ec4899' : '#71717a'}
                  />
                </View>

                <View style={styles.permissionRow}>
                  <View style={styles.permissionInfo}>
                    <Ionicons name="warning" size={16} color="#f97316" />
                    <Text style={styles.permissionText}>Crisis Alerts</Text>
                  </View>
                  <Switch
                    value={device.permissions.crisisAlerts}
                    onValueChange={(v) => updatePermission(device.id, 'crisisAlerts', v)}
                    trackColor={{ false: '#3f3f46', true: '#ea580c' }}
                    thumbColor={device.permissions.crisisAlerts ? '#f97316' : '#71717a'}
                  />
                </View>
              </View>
            </View>
          ))}
        </View>
      )}

      {/* Empty State */}
      {devices.length === 0 && (
        <View style={styles.emptyState}>
          <Ionicons name="people" size={48} color="#3f3f46" />
          <Text style={styles.emptyTitle}>No Trusted Devices</Text>
          <Text style={styles.emptyText}>
            Connect with family members to share your wellness status and receive
            support during difficult times.
          </Text>
        </View>
      )}

      {/* Generate Code Modal */}
      <Modal
        visible={showAddModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowAddModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Your Pairing Code</Text>
              <TouchableOpacity onPress={() => setShowAddModal(false)}>
                <Ionicons name="close" size={24} color="#9ca3af" />
              </TouchableOpacity>
            </View>

            {pairingCode && (
              <>
                <View style={styles.codeContainer}>
                  <Text style={styles.codeText}>{pairingCode.code}</Text>
                </View>
                <Text style={styles.codeExpiry}>
                  Expires in 10 minutes
                </Text>

                <TouchableOpacity
                  style={styles.shareButton}
                  onPress={sharePairingCode}
                >
                  <Ionicons name="share" size={20} color="#ffffff" />
                  <Text style={styles.shareButtonText}>Share Code</Text>
                </TouchableOpacity>

                <Text style={styles.instructions}>
                  Share this code with a family member. They can enter it in their
                  ReUnity app to connect with you.
                </Text>
              </>
            )}
          </View>
        </View>
      </Modal>

      {/* Enter Code Modal */}
      <Modal
        visible={showEnterCodeModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowEnterCodeModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Enter Pairing Code</Text>
              <TouchableOpacity onPress={() => setShowEnterCodeModal(false)}>
                <Ionicons name="close" size={24} color="#9ca3af" />
              </TouchableOpacity>
            </View>

            <TextInput
              style={styles.codeInput}
              value={enteredCode}
              onChangeText={(text) => setEnteredCode(text.toUpperCase())}
              placeholder="XXXXXX"
              placeholderTextColor="#71717a"
              maxLength={6}
              autoCapitalize="characters"
            />

            <TextInput
              style={styles.input}
              value={deviceName}
              onChangeText={setDeviceName}
              placeholder="Device name (e.g., Mom's Phone)"
              placeholderTextColor="#71717a"
            />

            <TextInput
              style={styles.input}
              value={relationship}
              onChangeText={setRelationship}
              placeholder="Relationship (e.g., Mother)"
              placeholderTextColor="#71717a"
            />

            <TouchableOpacity
              style={[
                styles.pairButton,
                enteredCode.length !== 6 && styles.pairButtonDisabled,
              ]}
              onPress={handleEnterCode}
              disabled={enteredCode.length !== 6}
            >
              <Ionicons name="link" size={20} color="#ffffff" />
              <Text style={styles.pairButtonText}>Pair Device</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#18181b',
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    color: '#ffffff',
  },
  subtitle: {
    color: '#9ca3af',
    fontSize: 14,
  },
  infoBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 12,
    margin: 16,
    padding: 16,
    backgroundColor: '#10b98120',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#10b98130',
  },
  infoText: {
    flex: 1,
    color: '#6ee7b7',
    fontSize: 13,
    lineHeight: 18,
  },
  addButtons: {
    flexDirection: 'row',
    gap: 12,
    padding: 16,
  },
  addButton: {
    flex: 1,
    padding: 16,
    backgroundColor: '#27272a',
    borderRadius: 12,
    alignItems: 'center',
    gap: 8,
  },
  addButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  addButtonSubtext: {
    color: '#71717a',
    fontSize: 12,
  },
  devicesList: {
    padding: 16,
  },
  sectionTitle: {
    color: '#9ca3af',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 12,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  deviceCard: {
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  deviceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  deviceInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  deviceAvatar: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#10b98130',
    alignItems: 'center',
    justifyContent: 'center',
  },
  deviceAvatarText: {
    color: '#10b981',
    fontSize: 18,
    fontWeight: '600',
  },
  deviceName: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '500',
  },
  deviceRelationship: {
    color: '#71717a',
    fontSize: 12,
    marginTop: 2,
  },
  removeButton: {
    padding: 8,
  },
  permissionsSection: {
    borderTopWidth: 1,
    borderTopColor: '#3f3f46',
    paddingTop: 12,
  },
  permissionsTitle: {
    color: '#9ca3af',
    fontSize: 12,
    marginBottom: 8,
  },
  permissionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  permissionInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  permissionText: {
    color: '#ffffff',
    fontSize: 14,
  },
  emptyState: {
    alignItems: 'center',
    padding: 32,
    marginTop: 32,
  },
  emptyTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyText: {
    color: '#71717a',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    width: '100%',
    backgroundColor: '#27272a',
    borderRadius: 16,
    padding: 24,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  modalTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
  codeContainer: {
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 24,
    alignItems: 'center',
    marginBottom: 8,
  },
  codeText: {
    color: '#10b981',
    fontSize: 36,
    fontWeight: '700',
    letterSpacing: 8,
    fontFamily: 'monospace',
  },
  codeExpiry: {
    color: '#71717a',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 24,
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  shareButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  instructions: {
    color: '#9ca3af',
    fontSize: 13,
    textAlign: 'center',
    lineHeight: 18,
  },
  codeInput: {
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 20,
    fontSize: 24,
    fontWeight: '700',
    color: '#ffffff',
    textAlign: 'center',
    letterSpacing: 8,
    marginBottom: 16,
  },
  input: {
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#ffffff',
    marginBottom: 12,
  },
  pairButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#3b82f6',
    borderRadius: 12,
    padding: 16,
    marginTop: 8,
  },
  pairButtonDisabled: {
    backgroundColor: '#3f3f46',
  },
  pairButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
});
