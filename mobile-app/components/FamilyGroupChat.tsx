import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, TextInput, KeyboardAvoidingView, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

interface FamilyMember {
  id: string;
  name: string;
  relationship: string;
  avatar: string;
  isOnline: boolean;
}

interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  content: string;
  timestamp: string;
  type: 'text' | 'checkin' | 'support';
}

export default function FamilyGroupChat() {
  const [activeTab, setActiveTab] = useState<'chat' | 'members' | 'alerts'>('chat');
  const [newMessage, setNewMessage] = useState('');
  const scrollViewRef = useRef<ScrollView>(null);

  const familyMembers: FamilyMember[] = [
    { id: '1', name: 'You', relationship: 'Self', avatar: 'Y', isOnline: true },
    { id: '2', name: 'Mom', relationship: 'Mother', avatar: 'M', isOnline: true },
    { id: '3', name: 'Dad', relationship: 'Father', avatar: 'D', isOnline: false },
    { id: '4', name: 'Sarah', relationship: 'Sister', avatar: 'S', isOnline: true },
  ];

  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: '1', senderId: '2', senderName: 'Mom', content: 'Good morning everyone! Hope you all have a wonderful day 💕', timestamp: new Date(Date.now() - 3600000).toISOString(), type: 'text' },
    { id: '2', senderId: '4', senderName: 'Sarah', content: 'Just completed my morning check-in. Feeling good!', timestamp: new Date(Date.now() - 1800000).toISOString(), type: 'checkin' },
    { id: '3', senderId: '1', senderName: 'You', content: 'Thanks for the support yesterday, everyone. It really helped.', timestamp: new Date(Date.now() - 900000).toISOString(), type: 'text' },
  ]);

  const quickResponses = [
    { emoji: '💚', text: "I'm doing okay" },
    { emoji: '🤗', text: 'Sending love' },
    { emoji: '📞', text: 'Can we talk?' },
    { emoji: '💪', text: 'Tough but managing' },
  ];

  const sendMessage = (content?: string) => {
    const text = content || newMessage;
    if (!text.trim()) return;

    const message: ChatMessage = {
      id: Date.now().toString(),
      senderId: '1',
      senderName: 'You',
      content: text,
      timestamp: new Date().toISOString(),
      type: 'text',
    };
    setMessages(prev => [...prev, message]);
    setNewMessage('');
    setTimeout(() => scrollViewRef.current?.scrollToEnd({ animated: true }), 100);
  };

  const sendCheckIn = () => {
    const message: ChatMessage = {
      id: Date.now().toString(),
      senderId: '1',
      senderName: 'You',
      content: "I've completed my daily check-in",
      timestamp: new Date().toISOString(),
      type: 'checkin',
    };
    setMessages(prev => [...prev, message]);
  };

  const sendSupportRequest = () => {
    const message: ChatMessage = {
      id: Date.now().toString(),
      senderId: '1',
      senderName: 'You',
      content: 'I could use some support right now 💙',
      timestamp: new Date().toISOString(),
      type: 'support',
    };
    setMessages(prev => [...prev, message]);
  };

  const onlineCount = familyMembers.filter(m => m.isOnline).length;

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <View style={styles.iconContainer}>
            <Ionicons name="people" size={24} color="#ec4899" />
          </View>
          <View>
            <Text style={styles.title}>Family Support Circle</Text>
            <Text style={styles.subtitle}>{familyMembers.length} members • {onlineCount} online</Text>
          </View>
        </View>
        <View style={styles.headerActions}>
          <TouchableOpacity style={styles.headerButton}>
            <Ionicons name="call" size={20} color="#a1a1aa" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.headerButton}>
            <Ionicons name="videocam" size={20} color="#a1a1aa" />
          </TouchableOpacity>
        </View>
      </View>

      {/* Tabs */}
      <View style={styles.tabs}>
        {[
          { id: 'chat', label: 'Chat', icon: 'chatbubbles' },
          { id: 'members', label: 'Members', icon: 'people' },
          { id: 'alerts', label: 'Alerts', icon: 'notifications' },
        ].map(tab => (
          <TouchableOpacity
            key={tab.id}
            style={[styles.tab, activeTab === tab.id && styles.activeTab]}
            onPress={() => setActiveTab(tab.id as typeof activeTab)}
          >
            <Ionicons name={tab.icon as any} size={18} color={activeTab === tab.id ? '#fff' : '#a1a1aa'} />
            <Text style={[styles.tabText, activeTab === tab.id && styles.activeTabText]}>{tab.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Chat Tab */}
      {activeTab === 'chat' && (
        <View style={styles.chatContainer}>
          <ScrollView 
            ref={scrollViewRef}
            style={styles.messagesContainer}
            contentContainerStyle={styles.messagesContent}
          >
            {messages.map(msg => {
              const isOwn = msg.senderId === '1';
              const sender = familyMembers.find(m => m.id === msg.senderId);

              return (
                <View key={msg.id} style={[styles.messageRow, isOwn && styles.ownMessageRow]}>
                  {!isOwn && (
                    <View style={styles.messageAvatar}>
                      <Text style={styles.avatarText}>{sender?.avatar}</Text>
                    </View>
                  )}
                  <View style={styles.messageContent}>
                    {!isOwn && <Text style={styles.senderName}>{msg.senderName}</Text>}
                    <View style={[
                      styles.messageBubble,
                      isOwn && styles.ownBubble,
                      msg.type === 'checkin' && styles.checkinBubble,
                      msg.type === 'support' && styles.supportBubble,
                    ]}>
                      {msg.type === 'checkin' && (
                        <View style={styles.messageTypeLabel}>
                          <Ionicons name="checkmark-circle" size={14} color="#22c55e" />
                          <Text style={styles.messageTypeLabelText}>Check-in</Text>
                        </View>
                      )}
                      {msg.type === 'support' && (
                        <View style={styles.messageTypeLabel}>
                          <Ionicons name="heart" size={14} color="#3b82f6" />
                          <Text style={[styles.messageTypeLabelText, { color: '#3b82f6' }]}>Support Request</Text>
                        </View>
                      )}
                      <Text style={styles.messageText}>{msg.content}</Text>
                      <Text style={styles.messageTime}>
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </Text>
                    </View>
                  </View>
                </View>
              );
            })}
          </ScrollView>

          {/* Quick Actions */}
          <View style={styles.quickActions}>
            <TouchableOpacity style={styles.quickActionButton} onPress={sendCheckIn}>
              <Ionicons name="checkmark-circle" size={16} color="#22c55e" />
              <Text style={styles.quickActionText}>Check-in</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickActionButton} onPress={sendSupportRequest}>
              <Ionicons name="heart" size={16} color="#3b82f6" />
              <Text style={styles.quickActionText}>Need Support</Text>
            </TouchableOpacity>
          </View>

          {/* Quick Responses */}
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.quickResponses}>
            {quickResponses.map((response, i) => (
              <TouchableOpacity
                key={i}
                style={styles.quickResponse}
                onPress={() => sendMessage(response.text)}
              >
                <Text style={styles.quickResponseEmoji}>{response.emoji}</Text>
                <Text style={styles.quickResponseText}>{response.text}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>

          {/* Input */}
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              value={newMessage}
              onChangeText={setNewMessage}
              placeholder="Message your family..."
              placeholderTextColor="#52525b"
              onSubmitEditing={() => sendMessage()}
            />
            <TouchableOpacity style={styles.sendButton} onPress={() => sendMessage()}>
              <Ionicons name="send" size={20} color="#fff" />
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Members Tab */}
      {activeTab === 'members' && (
        <ScrollView style={styles.membersContainer}>
          <TouchableOpacity style={styles.inviteButton}>
            <Ionicons name="person-add" size={20} color="#fff" />
            <Text style={styles.inviteButtonText}>Invite Family Member</Text>
          </TouchableOpacity>

          {familyMembers.map(member => (
            <View key={member.id} style={styles.memberCard}>
              <View style={styles.memberAvatar}>
                <Text style={styles.memberAvatarText}>{member.avatar}</Text>
                {member.isOnline && <View style={styles.onlineIndicator} />}
              </View>
              <View style={styles.memberInfo}>
                <Text style={styles.memberName}>{member.name}</Text>
                <Text style={styles.memberRelationship}>{member.relationship}</Text>
              </View>
              <TouchableOpacity>
                <Ionicons name="ellipsis-vertical" size={20} color="#a1a1aa" />
              </TouchableOpacity>
            </View>
          ))}
        </ScrollView>
      )}

      {/* Alerts Tab */}
      {activeTab === 'alerts' && (
        <ScrollView style={styles.alertsContainer}>
          <View style={styles.alertCard}>
            <Ionicons name="warning" size={20} color="#eab308" />
            <View style={styles.alertContent}>
              <Text style={styles.alertTitle}>Recent Alert</Text>
              <Text style={styles.alertText}>
                Sarah completed her check-in after a 36-hour gap. Current mood: Improving.
              </Text>
              <Text style={styles.alertTime}>2 hours ago</Text>
            </View>
          </View>

          <Text style={styles.alertSettingsTitle}>Alert Settings</Text>
          {[
            { label: 'Crisis alerts', enabled: true },
            { label: 'Missed check-ins', enabled: true },
            { label: 'Mood decline', enabled: true },
            { label: 'Support requests', enabled: true },
          ].map((setting, i) => (
            <View key={i} style={styles.alertSetting}>
              <Text style={styles.alertSettingLabel}>{setting.label}</Text>
              <View style={[styles.toggle, setting.enabled && styles.toggleActive]}>
                <View style={[styles.toggleKnob, setting.enabled && styles.toggleKnobActive]} />
              </View>
            </View>
          ))}
        </ScrollView>
      )}
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#09090b' },
  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', padding: 16, borderBottomWidth: 1, borderBottomColor: '#27272a' },
  headerLeft: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  iconContainer: { width: 44, height: 44, borderRadius: 12, backgroundColor: 'rgba(236, 72, 153, 0.2)', justifyContent: 'center', alignItems: 'center' },
  title: { fontSize: 18, fontWeight: 'bold', color: '#fff' },
  subtitle: { fontSize: 12, color: '#a1a1aa' },
  headerActions: { flexDirection: 'row', gap: 8 },
  headerButton: { width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(39, 39, 42, 0.5)', justifyContent: 'center', alignItems: 'center' },
  tabs: { flexDirection: 'row', marginHorizontal: 16, marginVertical: 12, backgroundColor: 'rgba(39, 39, 42, 0.5)', borderRadius: 8, padding: 4 },
  tab: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 8, gap: 6, borderRadius: 6 },
  activeTab: { backgroundColor: '#ec4899' },
  tabText: { fontSize: 14, color: '#a1a1aa' },
  activeTabText: { color: '#fff', fontWeight: '600' },
  chatContainer: { flex: 1 },
  messagesContainer: { flex: 1 },
  messagesContent: { padding: 16 },
  messageRow: { flexDirection: 'row', marginBottom: 12, alignItems: 'flex-end' },
  ownMessageRow: { justifyContent: 'flex-end' },
  messageAvatar: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#ec4899', justifyContent: 'center', alignItems: 'center', marginRight: 8 },
  avatarText: { fontSize: 14, fontWeight: 'bold', color: '#fff' },
  messageContent: { maxWidth: '75%' },
  senderName: { fontSize: 12, color: '#a1a1aa', marginBottom: 4 },
  messageBubble: { backgroundColor: '#27272a', borderRadius: 16, padding: 12 },
  ownBubble: { backgroundColor: '#ec4899' },
  checkinBubble: { backgroundColor: 'rgba(34, 197, 94, 0.2)', borderWidth: 1, borderColor: 'rgba(34, 197, 94, 0.3)' },
  supportBubble: { backgroundColor: 'rgba(59, 130, 246, 0.2)', borderWidth: 1, borderColor: 'rgba(59, 130, 246, 0.3)' },
  messageTypeLabel: { flexDirection: 'row', alignItems: 'center', gap: 4, marginBottom: 4 },
  messageTypeLabelText: { fontSize: 10, color: '#22c55e', fontWeight: '600' },
  messageText: { fontSize: 14, color: '#fff' },
  messageTime: { fontSize: 10, color: 'rgba(255, 255, 255, 0.6)', marginTop: 4 },
  quickActions: { flexDirection: 'row', paddingHorizontal: 16, gap: 8 },
  quickActionButton: { flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 8, borderRadius: 8, borderWidth: 1, borderColor: '#3f3f46', gap: 6 },
  quickActionText: { fontSize: 12, color: '#a1a1aa' },
  quickResponses: { paddingHorizontal: 16, paddingVertical: 8 },
  quickResponse: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 12, paddingVertical: 8, borderRadius: 16, borderWidth: 1, borderColor: '#3f3f46', marginRight: 8, gap: 6 },
  quickResponseEmoji: { fontSize: 14 },
  quickResponseText: { fontSize: 12, color: '#a1a1aa' },
  inputContainer: { flexDirection: 'row', padding: 16, gap: 8, borderTopWidth: 1, borderTopColor: '#27272a' },
  input: { flex: 1, backgroundColor: '#27272a', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 10, color: '#fff', fontSize: 14 },
  sendButton: { width: 40, height: 40, borderRadius: 20, backgroundColor: '#ec4899', justifyContent: 'center', alignItems: 'center' },
  membersContainer: { flex: 1, padding: 16 },
  inviteButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', backgroundColor: '#ec4899', paddingVertical: 12, borderRadius: 8, marginBottom: 16, gap: 8 },
  inviteButtonText: { color: '#fff', fontSize: 14, fontWeight: '600' },
  memberCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(39, 39, 42, 0.5)', borderRadius: 12, padding: 12, marginBottom: 8 },
  memberAvatar: { width: 44, height: 44, borderRadius: 22, backgroundColor: '#ec4899', justifyContent: 'center', alignItems: 'center' },
  memberAvatarText: { fontSize: 16, fontWeight: 'bold', color: '#fff' },
  onlineIndicator: { position: 'absolute', bottom: 0, right: 0, width: 12, height: 12, borderRadius: 6, backgroundColor: '#22c55e', borderWidth: 2, borderColor: '#09090b' },
  memberInfo: { flex: 1, marginLeft: 12 },
  memberName: { fontSize: 16, fontWeight: '600', color: '#fff' },
  memberRelationship: { fontSize: 12, color: '#a1a1aa' },
  alertsContainer: { flex: 1, padding: 16 },
  alertCard: { flexDirection: 'row', backgroundColor: 'rgba(234, 179, 8, 0.1)', borderRadius: 12, padding: 16, marginBottom: 16, gap: 12 },
  alertContent: { flex: 1 },
  alertTitle: { fontSize: 14, fontWeight: '600', color: '#eab308', marginBottom: 4 },
  alertText: { fontSize: 13, color: '#a1a1aa', lineHeight: 18 },
  alertTime: { fontSize: 11, color: '#71717a', marginTop: 8 },
  alertSettingsTitle: { fontSize: 16, fontWeight: '600', color: '#fff', marginBottom: 12 },
  alertSetting: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'rgba(39, 39, 42, 0.5)', borderRadius: 8, padding: 12, marginBottom: 8 },
  alertSettingLabel: { fontSize: 14, color: '#fff' },
  toggle: { width: 44, height: 24, borderRadius: 12, backgroundColor: '#3f3f46', padding: 2 },
  toggleActive: { backgroundColor: '#ec4899' },
  toggleKnob: { width: 20, height: 20, borderRadius: 10, backgroundColor: '#fff' },
  toggleKnobActive: { transform: [{ translateX: 20 }] },
});
