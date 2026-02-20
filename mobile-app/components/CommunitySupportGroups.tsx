import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, Linking } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface SupportGroup {
  id: string;
  name: string;
  topic: string;
  description: string;
  memberCount: number;
  isModerated: boolean;
  isMember: boolean;
}

const MOCK_GROUPS: SupportGroup[] = [
  { id: '1', name: 'Anxiety Warriors', topic: 'Anxiety', description: 'A safe space to share experiences with anxiety and learn coping strategies.', memberCount: 1247, isModerated: true, isMember: true },
  { id: '2', name: 'Depression Support Circle', topic: 'Depression', description: 'Understanding depression together. Share your journey, find hope.', memberCount: 2103, isModerated: true, isMember: false },
  { id: '3', name: 'PTSD & Trauma Healing', topic: 'PTSD/Trauma', description: 'For survivors of trauma. A moderated, trigger-warned space.', memberCount: 856, isModerated: true, isMember: true },
  { id: '4', name: 'BPD Understanding', topic: 'BPD', description: 'Living with BPD. DBT skills, emotional regulation, and peer support.', memberCount: 634, isModerated: true, isMember: false },
  { id: '5', name: 'Grief & Loss', topic: 'Grief', description: 'Processing loss together. All grief is valid here.', memberCount: 1089, isModerated: true, isMember: false },
];

export default function CommunitySupportGroups() {
  const [groups, setGroups] = useState<SupportGroup[]>(MOCK_GROUPS);
  const [searchTerm, setSearchTerm] = useState('');
  const [activeTab, setActiveTab] = useState<'discover' | 'my-groups'>('discover');

  const filteredGroups = groups.filter(g =>
    g.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    g.topic.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const myGroups = groups.filter(g => g.isMember);

  const joinGroup = (groupId: string) => {
    setGroups(groups.map(g =>
      g.id === groupId ? { ...g, isMember: true, memberCount: g.memberCount + 1 } : g
    ));
  };

  const leaveGroup = (groupId: string) => {
    setGroups(groups.map(g =>
      g.id === groupId ? { ...g, isMember: false, memberCount: g.memberCount - 1 } : g
    ));
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="people" size={24} color="#6366F1" />
        <Text style={styles.title}>Community Groups</Text>
      </View>
      <Text style={styles.subtitle}>Connect with others who understand</Text>

      {/* Tabs */}
      <View style={styles.tabContainer}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'discover' && styles.tabActive]}
          onPress={() => setActiveTab('discover')}
        >
          <Text style={[styles.tabText, activeTab === 'discover' && styles.tabTextActive]}>Discover</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'my-groups' && styles.tabActive]}
          onPress={() => setActiveTab('my-groups')}
        >
          <Text style={[styles.tabText, activeTab === 'my-groups' && styles.tabTextActive]}>
            My Groups ({myGroups.length})
          </Text>
        </TouchableOpacity>
      </View>

      {activeTab === 'discover' && (
        <>
          {/* Search */}
          <View style={styles.searchContainer}>
            <Ionicons name="search" size={20} color="#6B7280" style={styles.searchIcon} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search groups..."
              placeholderTextColor="#6B7280"
              value={searchTerm}
              onChangeText={setSearchTerm}
            />
          </View>

          {/* Groups List */}
          <View style={styles.groupsList}>
            {filteredGroups.map((group) => (
              <View key={group.id} style={styles.groupCard}>
                <View style={styles.groupHeader}>
                  <View style={styles.groupInfo}>
                    <Text style={styles.groupName}>{group.name}</Text>
                    {group.isModerated && (
                      <View style={styles.moderatedBadge}>
                        <Ionicons name="shield-checkmark" size={12} color="#10B981" />
                        <Text style={styles.moderatedText}>Moderated</Text>
                      </View>
                    )}
                  </View>
                  <View style={styles.memberCount}>
                    <Ionicons name="people" size={14} color="#94A3B8" />
                    <Text style={styles.memberText}>{group.memberCount.toLocaleString()}</Text>
                  </View>
                </View>
                <Text style={styles.groupDescription}>{group.description}</Text>
                <View style={styles.groupFooter}>
                  <View style={styles.topicBadge}>
                    <Text style={styles.topicText}>{group.topic}</Text>
                  </View>
                  {group.isMember ? (
                    <TouchableOpacity
                      style={styles.leaveButton}
                      onPress={() => leaveGroup(group.id)}
                    >
                      <Text style={styles.leaveButtonText}>Leave</Text>
                    </TouchableOpacity>
                  ) : (
                    <TouchableOpacity
                      style={styles.joinButton}
                      onPress={() => joinGroup(group.id)}
                    >
                      <Ionicons name="add" size={16} color="#FFFFFF" />
                      <Text style={styles.joinButtonText}>Join</Text>
                    </TouchableOpacity>
                  )}
                </View>
              </View>
            ))}
          </View>
        </>
      )}

      {activeTab === 'my-groups' && (
        <View style={styles.groupsList}>
          {myGroups.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="people-outline" size={48} color="#4B5563" />
              <Text style={styles.emptyText}>You haven't joined any groups yet</Text>
              <TouchableOpacity
                style={styles.discoverButton}
                onPress={() => setActiveTab('discover')}
              >
                <Text style={styles.discoverButtonText}>Discover Groups</Text>
              </TouchableOpacity>
            </View>
          ) : (
            myGroups.map((group) => (
              <TouchableOpacity key={group.id} style={styles.myGroupCard}>
                <View style={styles.myGroupInfo}>
                  <Text style={styles.myGroupName}>{group.name}</Text>
                  <Text style={styles.myGroupMembers}>{group.memberCount.toLocaleString()} members</Text>
                </View>
                <Ionicons name="chatbubbles" size={20} color="#6366F1" />
              </TouchableOpacity>
            ))
          )}
        </View>
      )}

      {/* Safety Notice */}
      <View style={styles.safetyNotice}>
        <Ionicons name="warning" size={16} color="#EAB308" />
        <Text style={styles.safetyText}>
          Community support is not a substitute for professional help. If in crisis, call 988.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A', padding: 16 },
  header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  title: { fontSize: 20, fontWeight: 'bold', color: '#F8FAFC' },
  subtitle: { fontSize: 14, color: '#94A3B8', marginTop: 4, marginBottom: 16 },
  tabContainer: { flexDirection: 'row', gap: 8, marginBottom: 16 },
  tab: { flex: 1, paddingVertical: 10, alignItems: 'center', backgroundColor: '#1E293B', borderRadius: 8 },
  tabActive: { backgroundColor: '#6366F1' },
  tabText: { color: '#94A3B8', fontWeight: '600' },
  tabTextActive: { color: '#FFFFFF' },
  searchContainer: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#1E293B', borderRadius: 8, paddingHorizontal: 12, marginBottom: 16 },
  searchIcon: { marginRight: 8 },
  searchInput: { flex: 1, height: 44, color: '#F8FAFC', fontSize: 16 },
  groupsList: { gap: 12 },
  groupCard: { backgroundColor: '#1E293B', borderRadius: 12, padding: 16 },
  groupHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 },
  groupInfo: { flex: 1 },
  groupName: { fontSize: 16, fontWeight: '600', color: '#F8FAFC' },
  moderatedBadge: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 4 },
  moderatedText: { fontSize: 12, color: '#10B981' },
  memberCount: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  memberText: { fontSize: 12, color: '#94A3B8' },
  groupDescription: { fontSize: 14, color: '#CBD5E1', marginBottom: 12 },
  groupFooter: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  topicBadge: { backgroundColor: '#334155', paddingVertical: 4, paddingHorizontal: 8, borderRadius: 4 },
  topicText: { fontSize: 12, color: '#94A3B8' },
  joinButton: { flexDirection: 'row', alignItems: 'center', gap: 4, backgroundColor: '#6366F1', paddingVertical: 6, paddingHorizontal: 12, borderRadius: 6 },
  joinButtonText: { color: '#FFFFFF', fontWeight: '600' },
  leaveButton: { paddingVertical: 6, paddingHorizontal: 12, borderRadius: 6, borderWidth: 1, borderColor: '#6366F1' },
  leaveButtonText: { color: '#6366F1', fontWeight: '600' },
  emptyState: { alignItems: 'center', paddingVertical: 40 },
  emptyText: { fontSize: 14, color: '#6B7280', marginTop: 8, marginBottom: 16 },
  discoverButton: { backgroundColor: '#6366F1', paddingVertical: 10, paddingHorizontal: 20, borderRadius: 8 },
  discoverButtonText: { color: '#FFFFFF', fontWeight: '600' },
  myGroupCard: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#1E293B', borderRadius: 12, padding: 16 },
  myGroupInfo: {},
  myGroupName: { fontSize: 16, fontWeight: '600', color: '#F8FAFC' },
  myGroupMembers: { fontSize: 12, color: '#94A3B8', marginTop: 2 },
  safetyNotice: { flexDirection: 'row', alignItems: 'flex-start', gap: 8, backgroundColor: '#EAB30820', padding: 12, borderRadius: 8, marginTop: 20 },
  safetyText: { flex: 1, fontSize: 12, color: '#EAB308' },
});
