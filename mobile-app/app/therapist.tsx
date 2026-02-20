import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  Modal,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

interface Client {
  id: number;
  name: string;
  email: string;
  status: 'active' | 'paused' | 'pending';
  avgEntropy: number;
  entropyTrend: 'improving' | 'stable' | 'declining';
  recentAlerts: number;
  lastActivity: string;
}

interface Alert {
  id: number;
  clientId: number;
  clientName: string;
  type: 'crisis' | 'high_entropy' | 'missed_checkin' | 'progress';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  createdAt: string;
  isViewed: boolean;
}

// Mock data
const mockClients: Client[] = [
  {
    id: 1,
    name: 'Sarah M.',
    email: 's***@email.com',
    status: 'active',
    avgEntropy: 0.45,
    entropyTrend: 'improving',
    recentAlerts: 0,
    lastActivity: '2 hours ago',
  },
  {
    id: 2,
    name: 'Michael R.',
    email: 'm***@email.com',
    status: 'active',
    avgEntropy: 0.72,
    entropyTrend: 'declining',
    recentAlerts: 2,
    lastActivity: '5 hours ago',
  },
  {
    id: 3,
    name: 'Jennifer L.',
    email: 'j***@email.com',
    status: 'pending',
    avgEntropy: 0,
    entropyTrend: 'stable',
    recentAlerts: 0,
    lastActivity: 'Never',
  },
];

const mockAlerts: Alert[] = [
  {
    id: 1,
    clientId: 2,
    clientName: 'Michael R.',
    type: 'high_entropy',
    severity: 'high',
    title: 'Elevated Entropy',
    description: 'Entropy score increased to 0.82 over 3 days',
    createdAt: '2 hours ago',
    isViewed: false,
  },
  {
    id: 2,
    clientId: 2,
    clientName: 'Michael R.',
    type: 'missed_checkin',
    severity: 'medium',
    title: 'Missed Check-In',
    description: 'Client missed scheduled check-in yesterday',
    createdAt: '1 day ago',
    isViewed: true,
  },
  {
    id: 3,
    clientId: 1,
    clientName: 'Sarah M.',
    type: 'progress',
    severity: 'low',
    title: 'Positive Progress',
    description: 'Entropy decreased by 15% this week',
    createdAt: '3 days ago',
    isViewed: true,
  },
];

export default function TherapistPortal() {
  const [activeTab, setActiveTab] = useState<'clients' | 'alerts'>('clients');
  const [clients, setClients] = useState(mockClients);
  const [alerts, setAlerts] = useState(mockAlerts);
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [selectedClient, setSelectedClient] = useState<Client | null>(null);
  const [showClientModal, setShowClientModal] = useState(false);

  const unreadAlerts = alerts.filter(a => !a.isViewed).length;

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return 'trending-down';
      case 'declining': return 'trending-up';
      default: return 'remove';
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'improving': return '#10b981';
      case 'declining': return '#ef4444';
      default: return '#f59e0b';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#ef4444';
      case 'high': return '#f97316';
      case 'medium': return '#f59e0b';
      default: return '#3b82f6';
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'crisis': return 'warning';
      case 'high_entropy': return 'pulse';
      case 'missed_checkin': return 'time';
      case 'progress': return 'checkmark-circle';
      default: return 'notifications';
    }
  };

  const handleInvite = () => {
    if (!inviteEmail) return;
    
    const newClient: Client = {
      id: clients.length + 1,
      name: 'New Client',
      email: inviteEmail.slice(0, 1) + '***@' + inviteEmail.split('@')[1],
      status: 'pending',
      avgEntropy: 0,
      entropyTrend: 'stable',
      recentAlerts: 0,
      lastActivity: 'Never',
    };
    
    setClients([...clients, newClient]);
    setShowInviteModal(false);
    setInviteEmail('');
    Alert.alert('Success', 'Invitation sent to client');
  };

  const markAlertViewed = (alertId: number) => {
    setAlerts(prev => prev.map(a => 
      a.id === alertId ? { ...a, isViewed: true } : a
    ));
  };

  const viewClient = (client: Client) => {
    setSelectedClient(client);
    setShowClientModal(true);
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#ffffff" />
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle}>Therapist Portal</Text>
          <View style={styles.badge}>
            <Ionicons name="shield-checkmark" size={12} color="#10b981" />
            <Text style={styles.badgeText}>Licensed</Text>
          </View>
        </View>
        <TouchableOpacity 
          style={styles.alertButton}
          onPress={() => setActiveTab('alerts')}
        >
          <Ionicons name="notifications" size={24} color="#ffffff" />
          {unreadAlerts > 0 && (
            <View style={styles.alertBadge}>
              <Text style={styles.alertBadgeText}>{unreadAlerts}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'clients' && styles.tabActive]}
          onPress={() => setActiveTab('clients')}
        >
          <Ionicons 
            name="people" 
            size={20} 
            color={activeTab === 'clients' ? '#10b981' : '#71717a'} 
          />
          <Text style={[styles.tabText, activeTab === 'clients' && styles.tabTextActive]}>
            Clients
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'alerts' && styles.tabActive]}
          onPress={() => setActiveTab('alerts')}
        >
          <Ionicons 
            name="notifications" 
            size={20} 
            color={activeTab === 'alerts' ? '#10b981' : '#71717a'} 
          />
          <Text style={[styles.tabText, activeTab === 'alerts' && styles.tabTextActive]}>
            Alerts
          </Text>
          {unreadAlerts > 0 && (
            <View style={styles.tabBadge}>
              <Text style={styles.tabBadgeText}>{unreadAlerts}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        {activeTab === 'clients' && (
          <>
            {/* Stats */}
            <View style={styles.statsRow}>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>{clients.filter(c => c.status === 'active').length}</Text>
                <Text style={styles.statLabel}>Active</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>{clients.filter(c => c.status === 'pending').length}</Text>
                <Text style={styles.statLabel}>Pending</Text>
              </View>
              <View style={[styles.statCard, { backgroundColor: '#ef444420' }]}>
                <Text style={[styles.statValue, { color: '#ef4444' }]}>
                  {alerts.filter(a => a.severity === 'high' || a.severity === 'critical').length}
                </Text>
                <Text style={styles.statLabel}>Alerts</Text>
              </View>
            </View>

            {/* Invite Button */}
            <TouchableOpacity
              style={styles.inviteButton}
              onPress={() => setShowInviteModal(true)}
            >
              <Ionicons name="person-add" size={20} color="#ffffff" />
              <Text style={styles.inviteButtonText}>Invite Client</Text>
            </TouchableOpacity>

            {/* Clients List */}
            {clients.map(client => (
              <TouchableOpacity
                key={client.id}
                style={styles.clientCard}
                onPress={() => viewClient(client)}
              >
                <View style={styles.clientHeader}>
                  <View style={styles.clientAvatar}>
                    <Text style={styles.clientAvatarText}>
                      {client.name.charAt(0)}
                    </Text>
                  </View>
                  <View style={styles.clientInfo}>
                    <Text style={styles.clientName}>{client.name}</Text>
                    <Text style={styles.clientEmail}>{client.email}</Text>
                  </View>
                  <View style={[
                    styles.statusBadge,
                    client.status === 'active' && styles.statusActive,
                    client.status === 'pending' && styles.statusPending,
                  ]}>
                    <Text style={styles.statusText}>{client.status}</Text>
                  </View>
                </View>

                {client.status === 'active' && (
                  <View style={styles.clientStats}>
                    <View style={styles.clientStat}>
                      <Text style={styles.clientStatLabel}>Entropy</Text>
                      <View style={styles.entropyRow}>
                        <Text style={[
                          styles.clientStatValue,
                          { color: client.avgEntropy > 0.7 ? '#ef4444' : 
                                   client.avgEntropy > 0.5 ? '#f59e0b' : '#10b981' }
                        ]}>
                          {(client.avgEntropy * 100).toFixed(0)}%
                        </Text>
                        <Ionicons 
                          name={getTrendIcon(client.entropyTrend)} 
                          size={16} 
                          color={getTrendColor(client.entropyTrend)} 
                        />
                      </View>
                    </View>
                    <View style={styles.clientStat}>
                      <Text style={styles.clientStatLabel}>Last Active</Text>
                      <Text style={styles.clientStatValue}>{client.lastActivity}</Text>
                    </View>
                    {client.recentAlerts > 0 && (
                      <View style={styles.alertIndicator}>
                        <Ionicons name="warning" size={14} color="#ef4444" />
                        <Text style={styles.alertIndicatorText}>{client.recentAlerts}</Text>
                      </View>
                    )}
                  </View>
                )}
              </TouchableOpacity>
            ))}
          </>
        )}

        {activeTab === 'alerts' && (
          <>
            {alerts.map(alert => (
              <TouchableOpacity
                key={alert.id}
                style={[
                  styles.alertCard,
                  !alert.isViewed && styles.alertCardUnread,
                ]}
                onPress={() => markAlertViewed(alert.id)}
              >
                <View style={[
                  styles.alertIconContainer,
                  { backgroundColor: `${getSeverityColor(alert.severity)}20` }
                ]}>
                  <Ionicons 
                    name={getAlertIcon(alert.type)} 
                    size={24} 
                    color={getSeverityColor(alert.severity)} 
                  />
                </View>
                <View style={styles.alertContent}>
                  <View style={styles.alertHeader}>
                    <Text style={styles.alertTitle}>{alert.title}</Text>
                    {!alert.isViewed && (
                      <View style={styles.newBadge}>
                        <Text style={styles.newBadgeText}>New</Text>
                      </View>
                    )}
                  </View>
                  <Text style={styles.alertDescription}>{alert.description}</Text>
                  <View style={styles.alertFooter}>
                    <Text style={styles.alertClient}>{alert.clientName}</Text>
                    <Text style={styles.alertTime}>{alert.createdAt}</Text>
                  </View>
                </View>
              </TouchableOpacity>
            ))}
          </>
        )}
      </ScrollView>

      {/* Invite Modal */}
      <Modal
        visible={showInviteModal}
        transparent
        animationType="fade"
        onRequestClose={() => setShowInviteModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Invite Client</Text>
              <TouchableOpacity onPress={() => setShowInviteModal(false)}>
                <Ionicons name="close" size={24} color="#9ca3af" />
              </TouchableOpacity>
            </View>
            
            <Text style={styles.modalDescription}>
              Send an invitation to connect with a client. They will need to consent to share their data.
            </Text>

            <TextInput
              style={styles.input}
              value={inviteEmail}
              onChangeText={setInviteEmail}
              placeholder="client@email.com"
              placeholderTextColor="#71717a"
              keyboardType="email-address"
              autoCapitalize="none"
            />

            <TouchableOpacity
              style={[styles.sendButton, !inviteEmail && styles.sendButtonDisabled]}
              onPress={handleInvite}
              disabled={!inviteEmail}
            >
              <Ionicons name="send" size={20} color="#ffffff" />
              <Text style={styles.sendButtonText}>Send Invitation</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Client Detail Modal */}
      <Modal
        visible={showClientModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowClientModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { maxHeight: '80%' }]}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{selectedClient?.name}</Text>
              <TouchableOpacity onPress={() => setShowClientModal(false)}>
                <Ionicons name="close" size={24} color="#9ca3af" />
              </TouchableOpacity>
            </View>

            {selectedClient?.status === 'active' && (
              <ScrollView>
                {/* Entropy Chart Placeholder */}
                <View style={styles.chartContainer}>
                  <Text style={styles.chartTitle}>7-Day Entropy Trend</Text>
                  <View style={styles.chartBars}>
                    {[0.55, 0.60, 0.58, 0.65, 0.68, 0.72, 0.72].map((val, i) => (
                      <View key={i} style={styles.chartBarContainer}>
                        <View 
                          style={[
                            styles.chartBar,
                            { 
                              height: `${val * 100}%`,
                              backgroundColor: val > 0.7 ? '#ef4444' : 
                                             val > 0.5 ? '#f59e0b' : '#10b981'
                            }
                          ]} 
                        />
                        <Text style={styles.chartLabel}>{20 + i}</Text>
                      </View>
                    ))}
                  </View>
                </View>

                {/* Stats */}
                <View style={styles.detailStats}>
                  <View style={styles.detailStat}>
                    <Text style={styles.detailStatValue}>13</Text>
                    <Text style={styles.detailStatLabel}>Journal Entries</Text>
                  </View>
                  <View style={styles.detailStat}>
                    <Text style={[styles.detailStatValue, { color: '#10b981' }]}>10</Text>
                    <Text style={styles.detailStatLabel}>Check-Ins</Text>
                  </View>
                  <View style={styles.detailStat}>
                    <Text style={[styles.detailStatValue, { color: '#ef4444' }]}>1</Text>
                    <Text style={styles.detailStatLabel}>Crisis Events</Text>
                  </View>
                </View>

                {/* Consent Info */}
                <View style={styles.consentSection}>
                  <Text style={styles.consentTitle}>Data Sharing Consent</Text>
                  <View style={styles.consentItem}>
                    <Text style={styles.consentLabel}>Entropy Data</Text>
                    <View style={styles.consentBadge}>
                      <Text style={styles.consentBadgeText}>Consented</Text>
                    </View>
                  </View>
                  <View style={styles.consentItem}>
                    <Text style={styles.consentLabel}>Journal Summaries</Text>
                    <View style={styles.consentBadge}>
                      <Text style={styles.consentBadgeText}>Consented</Text>
                    </View>
                  </View>
                  <View style={styles.consentItem}>
                    <Text style={styles.consentLabel}>Crisis Alerts</Text>
                    <View style={styles.consentBadge}>
                      <Text style={styles.consentBadgeText}>Enabled</Text>
                    </View>
                  </View>
                </View>

                {/* Actions */}
                <View style={styles.actionButtons}>
                  <TouchableOpacity style={styles.actionButton}>
                    <Ionicons name="chatbubble" size={20} color="#ffffff" />
                    <Text style={styles.actionButtonText}>Message</Text>
                  </TouchableOpacity>
                  <TouchableOpacity style={[styles.actionButton, styles.actionButtonSecondary]}>
                    <Ionicons name="document-text" size={20} color="#10b981" />
                    <Text style={[styles.actionButtonText, { color: '#10b981' }]}>Notes</Text>
                  </TouchableOpacity>
                </View>
              </ScrollView>
            )}

            {selectedClient?.status === 'pending' && (
              <View style={styles.pendingState}>
                <Ionicons name="time" size={48} color="#f59e0b" />
                <Text style={styles.pendingTitle}>Awaiting Consent</Text>
                <Text style={styles.pendingText}>
                  Invitation sent to {selectedClient.email}
                </Text>
              </View>
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0f11',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  backButton: {
    padding: 8,
  },
  headerCenter: {
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#ffffff',
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 4,
  },
  badgeText: {
    fontSize: 12,
    color: '#10b981',
  },
  alertButton: {
    padding: 8,
    position: 'relative',
  },
  alertBadge: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: '#ef4444',
    borderRadius: 10,
    width: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alertBadgeText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  tabs: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderBottomColor: '#27272a',
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 12,
  },
  tabActive: {
    borderBottomWidth: 2,
    borderBottomColor: '#10b981',
  },
  tabText: {
    fontSize: 14,
    color: '#71717a',
  },
  tabTextActive: {
    color: '#10b981',
    fontWeight: '500',
  },
  tabBadge: {
    backgroundColor: '#ef4444',
    borderRadius: 10,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  tabBadgeText: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: '600',
  },
  content: {
    flex: 1,
    padding: 16,
  },
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#ffffff',
  },
  statLabel: {
    fontSize: 12,
    color: '#71717a',
    marginTop: 4,
  },
  inviteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
  },
  inviteButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  clientCard: {
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  clientHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  clientAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: '#10b98130',
    alignItems: 'center',
    justifyContent: 'center',
  },
  clientAvatarText: {
    color: '#10b981',
    fontSize: 18,
    fontWeight: '600',
  },
  clientInfo: {
    flex: 1,
    marginLeft: 12,
  },
  clientName: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '500',
  },
  clientEmail: {
    color: '#71717a',
    fontSize: 12,
    marginTop: 2,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: '#3f3f46',
  },
  statusActive: {
    backgroundColor: '#10b98130',
  },
  statusPending: {
    backgroundColor: '#f59e0b30',
  },
  statusText: {
    fontSize: 12,
    color: '#ffffff',
    textTransform: 'capitalize',
  },
  clientStats: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#3f3f46',
  },
  clientStat: {
    flex: 1,
  },
  clientStatLabel: {
    fontSize: 11,
    color: '#71717a',
  },
  clientStatValue: {
    fontSize: 14,
    color: '#ffffff',
    fontWeight: '500',
    marginTop: 2,
  },
  entropyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  alertIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#ef444430',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  alertIndicatorText: {
    color: '#ef4444',
    fontSize: 12,
    fontWeight: '600',
  },
  alertCard: {
    flexDirection: 'row',
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  alertCardUnread: {
    borderWidth: 1,
    borderColor: '#3b82f650',
  },
  alertIconContainer: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  alertContent: {
    flex: 1,
  },
  alertHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  alertTitle: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '500',
  },
  newBadge: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  newBadgeText: {
    color: '#ffffff',
    fontSize: 10,
    fontWeight: '600',
  },
  alertDescription: {
    color: '#9ca3af',
    fontSize: 13,
    marginTop: 4,
  },
  alertFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  alertClient: {
    color: '#71717a',
    fontSize: 12,
  },
  alertTime: {
    color: '#71717a',
    fontSize: 12,
    marginLeft: 8,
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
    marginBottom: 16,
  },
  modalTitle: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '600',
  },
  modalDescription: {
    color: '#9ca3af',
    fontSize: 14,
    marginBottom: 16,
  },
  input: {
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    color: '#ffffff',
    marginBottom: 16,
  },
  sendButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    borderRadius: 12,
    padding: 16,
  },
  sendButtonDisabled: {
    backgroundColor: '#3f3f46',
  },
  sendButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  chartContainer: {
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  chartTitle: {
    color: '#9ca3af',
    fontSize: 12,
    marginBottom: 12,
  },
  chartBars: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 100,
    gap: 8,
  },
  chartBarContainer: {
    flex: 1,
    alignItems: 'center',
    height: '100%',
    justifyContent: 'flex-end',
  },
  chartBar: {
    width: '100%',
    borderRadius: 4,
  },
  chartLabel: {
    color: '#71717a',
    fontSize: 10,
    marginTop: 4,
  },
  detailStats: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  detailStat: {
    flex: 1,
    backgroundColor: '#18181b',
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
  },
  detailStatValue: {
    fontSize: 20,
    fontWeight: '700',
    color: '#ffffff',
  },
  detailStatLabel: {
    fontSize: 11,
    color: '#71717a',
    marginTop: 4,
  },
  consentSection: {
    marginBottom: 16,
  },
  consentTitle: {
    color: '#9ca3af',
    fontSize: 12,
    marginBottom: 8,
  },
  consentItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#18181b',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  consentLabel: {
    color: '#ffffff',
    fontSize: 14,
  },
  consentBadge: {
    backgroundColor: '#10b98130',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  consentBadgeText: {
    color: '#10b981',
    fontSize: 12,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#10b981',
    borderRadius: 12,
    padding: 14,
  },
  actionButtonSecondary: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#10b981',
  },
  actionButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  pendingState: {
    alignItems: 'center',
    padding: 32,
  },
  pendingTitle: {
    color: '#f59e0b',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
  },
  pendingText: {
    color: '#71717a',
    fontSize: 14,
    marginTop: 8,
  },
});
