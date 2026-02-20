import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  Users, 
  MessageCircle, 
  Send,
  Heart,
  Shield,
  Bell,
  Settings,
  UserPlus,
  Phone,
  Video,
  Image,
  Smile,
  AlertTriangle,
  CheckCircle,
  Clock,
  X,
  MoreVertical,
  Pin,
  Star
} from 'lucide-react';

interface FamilyMember {
  id: string;
  name: string;
  relationship: string;
  avatar: string;
  isOnline: boolean;
  lastSeen?: string;
  role: 'admin' | 'member';
}

interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  content: string;
  timestamp: string;
  type: 'text' | 'alert' | 'checkin' | 'support' | 'image';
  isPinned?: boolean;
  reactions?: { emoji: string; count: number; users: string[] }[];
}

interface FamilyGroupChatProps {
  compact?: boolean;
}

export default function FamilyGroupChat({ compact = false }: FamilyGroupChatProps) {
  const [activeTab, setActiveTab] = useState<'chat' | 'members' | 'alerts' | 'settings'>('chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [familyMembers, setFamilyMembers] = useState<FamilyMember[]>([]);
  const [showQuickResponses, setShowQuickResponses] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const quickResponses = [
    { emoji: '💚', text: "I'm doing okay today" },
    { emoji: '🤗', text: "Sending love and support" },
    { emoji: '📞', text: "Can we talk soon?" },
    { emoji: '🙏', text: "Thank you for checking in" },
    { emoji: '💪', text: "Having a tough day but managing" },
    { emoji: '☀️', text: "Feeling better today!" },
  ];

  // Mock data
  useEffect(() => {
    const mockMembers: FamilyMember[] = [
      { id: '1', name: 'You', relationship: 'Self', avatar: 'Y', isOnline: true, role: 'admin' },
      { id: '2', name: 'Mom', relationship: 'Mother', avatar: 'M', isOnline: true, role: 'member' },
      { id: '3', name: 'Dad', relationship: 'Father', avatar: 'D', isOnline: false, lastSeen: '2 hours ago', role: 'member' },
      { id: '4', name: 'Sarah', relationship: 'Sister', avatar: 'S', isOnline: true, role: 'member' },
      { id: '5', name: 'Dr. Chen', relationship: 'Therapist', avatar: 'C', isOnline: false, lastSeen: '1 day ago', role: 'member' },
    ];
    setFamilyMembers(mockMembers);

    const mockMessages: ChatMessage[] = [
      {
        id: '1',
        senderId: '2',
        senderName: 'Mom',
        content: "Good morning everyone! Hope you all have a wonderful day 💕",
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        type: 'text',
        reactions: [{ emoji: '❤️', count: 3, users: ['1', '3', '4'] }],
      },
      {
        id: '2',
        senderId: '4',
        senderName: 'Sarah',
        content: "Just completed my morning check-in. Feeling good!",
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        type: 'checkin',
      },
      {
        id: '3',
        senderId: '1',
        senderName: 'You',
        content: "Thanks for the support yesterday, everyone. It really helped.",
        timestamp: new Date(Date.now() - 900000).toISOString(),
        type: 'text',
        reactions: [{ emoji: '💚', count: 4, users: ['2', '3', '4', '5'] }],
      },
      {
        id: '4',
        senderId: '5',
        senderName: 'Dr. Chen',
        content: "Remember: Our next family session is Thursday at 4pm. Looking forward to seeing everyone's progress!",
        timestamp: new Date(Date.now() - 600000).toISOString(),
        type: 'text',
        isPinned: true,
      },
    ];
    setMessages(mockMessages);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = (content?: string) => {
    const messageContent = content || newMessage;
    if (!messageContent.trim()) return;

    const message: ChatMessage = {
      id: Date.now().toString(),
      senderId: '1',
      senderName: 'You',
      content: messageContent,
      timestamp: new Date().toISOString(),
      type: 'text',
    };

    setMessages(prev => [...prev, message]);
    setNewMessage('');
    setShowQuickResponses(false);
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
      content: "I could use some support right now 💙",
      timestamp: new Date().toISOString(),
      type: 'support',
    };
    setMessages(prev => [...prev, message]);
  };

  const onlineCount = familyMembers.filter(m => m.isOnline).length;

  if (compact) {
    return (
      <Card className="bg-gradient-to-br from-pink-500/10 to-rose-500/10 border-pink-500/20">
        <CardContent className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-pink-500/20 rounded-lg">
              <Users className="w-5 h-5 text-pink-400" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Family Chat</h3>
              <p className="text-xs text-zinc-400">Coordinated support</p>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex -space-x-2">
              {familyMembers.slice(0, 4).map(member => (
                <div
                  key={member.id}
                  className="relative w-8 h-8 rounded-full bg-gradient-to-br from-pink-400 to-rose-400 border-2 border-zinc-900 flex items-center justify-center text-xs font-bold"
                >
                  {member.avatar}
                  {member.isOnline && (
                    <span className="absolute bottom-0 right-0 w-2 h-2 bg-green-500 rounded-full border border-zinc-900" />
                  )}
                </div>
              ))}
            </div>
            <Badge variant="outline" className="border-green-500/50 text-green-400">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-1" />
              {onlineCount} Online
            </Badge>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-pink-500/20 rounded-lg">
            <Users className="w-6 h-6 text-pink-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Family Support Circle</h2>
            <p className="text-sm text-zinc-400">{familyMembers.length} members • {onlineCount} online</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="border-zinc-700">
            <Phone className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="sm" className="border-zinc-700">
            <Video className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex gap-2 bg-zinc-900/50 p-1 rounded-lg">
        {[
          { id: 'chat', label: 'Chat', icon: MessageCircle },
          { id: 'members', label: 'Members', icon: Users },
          { id: 'alerts', label: 'Alerts', icon: Bell },
          { id: 'settings', label: 'Settings', icon: Settings },
        ].map(tab => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={activeTab === tab.id ? 'bg-pink-600' : ''}
          >
            <tab.icon className="w-4 h-4 mr-2" />
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Chat Tab */}
      {activeTab === 'chat' && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="p-0">
            {/* Pinned Message */}
            {messages.find(m => m.isPinned) && (
              <div className="bg-yellow-500/10 border-b border-yellow-500/20 p-3 flex items-center gap-2">
                <Pin className="w-4 h-4 text-yellow-400" />
                <p className="text-sm text-yellow-200 flex-1">
                  {messages.find(m => m.isPinned)?.content}
                </p>
              </div>
            )}

            {/* Messages */}
            <div className="h-80 overflow-y-auto p-4 space-y-4">
              {messages.map(msg => {
                const sender = familyMembers.find(m => m.id === msg.senderId);
                const isOwn = msg.senderId === '1';

                return (
                  <div
                    key={msg.id}
                    className={`flex ${isOwn ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`flex gap-2 max-w-[80%] ${isOwn ? 'flex-row-reverse' : ''}`}>
                      {!isOwn && (
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-400 to-rose-400 flex items-center justify-center text-xs font-bold flex-shrink-0">
                          {sender?.avatar || '?'}
                        </div>
                      )}
                      <div>
                        {!isOwn && (
                          <p className="text-xs text-zinc-400 mb-1">{msg.senderName}</p>
                        )}
                        <div
                          className={`p-3 rounded-lg ${
                            msg.type === 'checkin'
                              ? 'bg-green-500/20 border border-green-500/30'
                              : msg.type === 'support'
                              ? 'bg-blue-500/20 border border-blue-500/30'
                              : msg.type === 'alert'
                              ? 'bg-red-500/20 border border-red-500/30'
                              : isOwn
                              ? 'bg-pink-600'
                              : 'bg-zinc-800'
                          }`}
                        >
                          {msg.type === 'checkin' && (
                            <div className="flex items-center gap-2 mb-1">
                              <CheckCircle className="w-4 h-4 text-green-400" />
                              <span className="text-xs text-green-400 font-medium">Check-in</span>
                            </div>
                          )}
                          {msg.type === 'support' && (
                            <div className="flex items-center gap-2 mb-1">
                              <Heart className="w-4 h-4 text-blue-400" />
                              <span className="text-xs text-blue-400 font-medium">Support Request</span>
                            </div>
                          )}
                          <p className="text-sm text-white">{msg.content}</p>
                          <p className="text-xs opacity-60 mt-1">
                            {new Date(msg.timestamp).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </p>
                        </div>
                        {msg.reactions && msg.reactions.length > 0 && (
                          <div className="flex gap-1 mt-1">
                            {msg.reactions.map((reaction, i) => (
                              <span
                                key={i}
                                className="text-xs bg-zinc-800 px-2 py-0.5 rounded-full"
                              >
                                {reaction.emoji} {reaction.count}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </div>

            {/* Quick Responses */}
            {showQuickResponses && (
              <div className="border-t border-zinc-800 p-3 bg-zinc-900/50">
                <div className="grid grid-cols-2 gap-2">
                  {quickResponses.map((response, i) => (
                    <Button
                      key={i}
                      variant="outline"
                      size="sm"
                      onClick={() => sendMessage(response.text)}
                      className="justify-start border-zinc-700 text-left"
                    >
                      <span className="mr-2">{response.emoji}</span>
                      <span className="truncate">{response.text}</span>
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Input */}
            <div className="p-4 border-t border-zinc-800">
              <div className="flex gap-2 mb-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={sendCheckIn}
                  className="border-green-500/50 text-green-400 hover:bg-green-500/10"
                >
                  <CheckCircle className="w-4 h-4 mr-1" />
                  Check-in
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={sendSupportRequest}
                  className="border-blue-500/50 text-blue-400 hover:bg-blue-500/10"
                >
                  <Heart className="w-4 h-4 mr-1" />
                  Need Support
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowQuickResponses(!showQuickResponses)}
                  className="border-zinc-700"
                >
                  <Smile className="w-4 h-4" />
                </Button>
              </div>
              <div className="flex gap-2">
                <Input
                  value={newMessage}
                  onChange={e => setNewMessage(e.target.value)}
                  placeholder="Message your family..."
                  className="bg-zinc-800 border-zinc-700"
                  onKeyPress={e => e.key === 'Enter' && sendMessage()}
                />
                <Button onClick={() => sendMessage()} className="bg-pink-600 hover:bg-pink-700">
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Members Tab */}
      {activeTab === 'members' && (
        <div className="space-y-3">
          <Button className="w-full bg-pink-600 hover:bg-pink-700">
            <UserPlus className="w-4 h-4 mr-2" />
            Invite Family Member
          </Button>

          {familyMembers.map(member => (
            <Card key={member.id} className="bg-zinc-900/50 border-zinc-800">
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-pink-400 to-rose-400 flex items-center justify-center text-lg font-bold">
                        {member.avatar}
                      </div>
                      {member.isOnline && (
                        <span className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-zinc-900" />
                      )}
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">{member.name}</h3>
                      <p className="text-xs text-zinc-400">{member.relationship}</p>
                      {!member.isOnline && member.lastSeen && (
                        <p className="text-xs text-zinc-500">Last seen {member.lastSeen}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {member.role === 'admin' && (
                      <Badge className="bg-yellow-500/20 text-yellow-300">
                        <Star className="w-3 h-3 mr-1" />
                        Admin
                      </Badge>
                    )}
                    <Button variant="ghost" size="sm">
                      <MoreVertical className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Alerts Tab */}
      {activeTab === 'alerts' && (
        <div className="space-y-3">
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Alert Settings</CardTitle>
              <CardDescription>Configure when family members receive alerts</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {[
                { label: 'Crisis alerts', desc: 'When high risk is detected', enabled: true },
                { label: 'Missed check-ins', desc: 'After 24 hours without check-in', enabled: true },
                { label: 'Mood decline', desc: 'When mood trends downward', enabled: true },
                { label: 'Support requests', desc: 'When someone asks for support', enabled: true },
                { label: 'Daily summaries', desc: 'Daily wellness overview', enabled: false },
              ].map((alert, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg">
                  <div>
                    <p className="text-sm text-white">{alert.label}</p>
                    <p className="text-xs text-zinc-400">{alert.desc}</p>
                  </div>
                  <div className={`w-10 h-6 rounded-full p-1 cursor-pointer transition-colors ${
                    alert.enabled ? 'bg-pink-600' : 'bg-zinc-700'
                  }`}>
                    <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                      alert.enabled ? 'translate-x-4' : ''
                    }`} />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card className="bg-yellow-500/10 border-yellow-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-yellow-300">Recent Alert</h3>
                  <p className="text-sm text-zinc-300 mt-1">
                    Sarah completed her check-in after a 36-hour gap. Current mood: Improving.
                  </p>
                  <p className="text-xs text-zinc-500 mt-2">2 hours ago</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Settings Tab */}
      {activeTab === 'settings' && (
        <div className="space-y-3">
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Group Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <label className="text-sm text-zinc-400 mb-1 block">Group Name</label>
                <Input
                  defaultValue="Family Support Circle"
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>

              <div className="p-3 bg-zinc-800/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-white">Privacy Mode</span>
                  <Badge className="bg-green-500/20 text-green-300">Enabled</Badge>
                </div>
                <p className="text-xs text-zinc-400">
                  Messages are end-to-end encrypted. Only group members can see content.
                </p>
              </div>

              <div className="p-3 bg-zinc-800/50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-white">Therapist Access</span>
                  <Badge className="bg-blue-500/20 text-blue-300">Limited</Badge>
                </div>
                <p className="text-xs text-zinc-400">
                  Dr. Chen can view messages but cannot share outside the group.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-red-500/10 border-red-500/20">
            <CardContent className="p-4">
              <Button variant="outline" className="w-full border-red-500/50 text-red-400 hover:bg-red-500/10">
                Leave Group
              </Button>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
