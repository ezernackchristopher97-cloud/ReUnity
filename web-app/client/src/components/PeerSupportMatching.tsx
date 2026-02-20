import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { 
  Users, 
  MessageCircle, 
  Shield, 
  Heart, 
  Search, 
  Send,
  AlertTriangle,
  CheckCircle,
  Clock,
  UserPlus,
  X,
  Flag,
  Info
} from 'lucide-react';

interface PeerProfile {
  id: string;
  anonymousName: string;
  experiences: string[];
  supportingOthers: string[];
  bio: string;
  isOnline: boolean;
  matchScore: number;
  joinedDate: string;
  conversationCount: number;
}

interface ChatMessage {
  id: string;
  senderId: string;
  content: string;
  timestamp: string;
  isOwn: boolean;
}

interface PeerSupportMatchingProps {
  compact?: boolean;
}

export default function PeerSupportMatching({ compact = false }: PeerSupportMatchingProps) {
  const [activeTab, setActiveTab] = useState<'find' | 'matches' | 'chat' | 'profile'>('find');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedExperiences, setSelectedExperiences] = useState<string[]>([]);
  const [matches, setMatches] = useState<PeerProfile[]>([]);
  const [activeChat, setActiveChat] = useState<PeerProfile | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [showGuidelines, setShowGuidelines] = useState(false);
  const [userProfile, setUserProfile] = useState({
    anonymousName: '',
    experiences: [] as string[],
    supportingOthers: [] as string[],
    bio: '',
  });

  const experienceCategories = [
    { id: 'anxiety', label: 'Anxiety', icon: '😰' },
    { id: 'depression', label: 'Depression', icon: '😔' },
    { id: 'ptsd', label: 'PTSD/Trauma', icon: '💔' },
    { id: 'grief', label: 'Grief & Loss', icon: '🕊️' },
    { id: 'relationship', label: 'Relationship Issues', icon: '💑' },
    { id: 'family', label: 'Family Challenges', icon: '👨‍👩‍👧' },
    { id: 'addiction', label: 'Recovery Journey', icon: '🌱' },
    { id: 'eating', label: 'Eating Concerns', icon: '🍃' },
    { id: 'identity', label: 'Identity & Self', icon: '🦋' },
    { id: 'lgbtq', label: 'LGBTQ+ Support', icon: '🏳️‍🌈' },
    { id: 'chronic', label: 'Chronic Illness', icon: '💪' },
    { id: 'caregiver', label: 'Caregiver Stress', icon: '🤝' },
  ];

  // Mock data for demonstration
  useEffect(() => {
    const mockMatches: PeerProfile[] = [
      {
        id: '1',
        anonymousName: 'HopefulHeart',
        experiences: ['anxiety', 'depression'],
        supportingOthers: ['anxiety', 'grief'],
        bio: 'Been through dark times, now helping others find their light.',
        isOnline: true,
        matchScore: 92,
        joinedDate: '2025-10-15',
        conversationCount: 47,
      },
      {
        id: '2',
        anonymousName: 'GentleWarrior',
        experiences: ['ptsd', 'anxiety'],
        supportingOthers: ['ptsd', 'relationship'],
        bio: 'Survivor and listener. Your story matters.',
        isOnline: true,
        matchScore: 87,
        joinedDate: '2025-08-22',
        conversationCount: 123,
      },
      {
        id: '3',
        anonymousName: 'QuietStrength',
        experiences: ['grief', 'caregiver'],
        supportingOthers: ['grief', 'chronic'],
        bio: 'Walking beside you through the hard days.',
        isOnline: false,
        matchScore: 78,
        joinedDate: '2025-11-03',
        conversationCount: 31,
      },
    ];
    setMatches(mockMatches);
  }, []);

  const handleSearch = () => {
    setIsSearching(true);
    setTimeout(() => {
      setIsSearching(false);
      setActiveTab('matches');
    }, 1500);
  };

  const toggleExperience = (expId: string) => {
    setSelectedExperiences(prev =>
      prev.includes(expId)
        ? prev.filter(e => e !== expId)
        : [...prev, expId]
    );
  };

  const startChat = (peer: PeerProfile) => {
    setActiveChat(peer);
    setMessages([
      {
        id: '1',
        senderId: peer.id,
        content: `Hi there! I'm ${peer.anonymousName}. I'm here to listen and support you. How are you feeling today?`,
        timestamp: new Date(Date.now() - 60000).toISOString(),
        isOwn: false,
      },
    ]);
    setActiveTab('chat');
  };

  const sendMessage = () => {
    if (!newMessage.trim() || !activeChat) return;

    const message: ChatMessage = {
      id: Date.now().toString(),
      senderId: 'self',
      content: newMessage,
      timestamp: new Date().toISOString(),
      isOwn: true,
    };

    setMessages(prev => [...prev, message]);
    setNewMessage('');

    // Simulate peer response
    setTimeout(() => {
      const responses = [
        "Thank you for sharing that with me. It takes courage to open up.",
        "I hear you. That sounds really difficult.",
        "You're not alone in this. I've been through something similar.",
        "How long have you been feeling this way?",
        "What helps you cope when things get tough?",
      ];
      const response: ChatMessage = {
        id: (Date.now() + 1).toString(),
        senderId: activeChat.id,
        content: responses[Math.floor(Math.random() * responses.length)],
        timestamp: new Date().toISOString(),
        isOwn: false,
      };
      setMessages(prev => [...prev, response]);
    }, 2000);
  };

  if (compact) {
    return (
      <Card className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/20">
        <CardContent className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Users className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Peer Support</h3>
              <p className="text-xs text-zinc-400">Connect anonymously</p>
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex -space-x-2">
              {[1, 2, 3].map(i => (
                <div
                  key={i}
                  className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 border-2 border-zinc-900 flex items-center justify-center text-xs font-bold"
                >
                  {['H', 'G', 'Q'][i - 1]}
                </div>
              ))}
            </div>
            <Badge variant="outline" className="border-green-500/50 text-green-400">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse" />
              12 Online
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
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Users className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Peer Support</h2>
            <p className="text-sm text-zinc-400">Connect with others who understand</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowGuidelines(true)}
          className="border-zinc-700"
        >
          <Info className="w-4 h-4 mr-2" />
          Guidelines
        </Button>
      </div>

      {/* Guidelines Modal */}
      {showGuidelines && (
        <Card className="bg-zinc-900 border-yellow-500/30">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-yellow-400 flex items-center gap-2">
                <Shield className="w-5 h-5" />
                Community Guidelines
              </CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setShowGuidelines(false)}>
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
              <p className="text-zinc-300">Be respectful and supportive. We're all here to help each other.</p>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
              <p className="text-zinc-300">Keep conversations anonymous. Don't share personal identifying information.</p>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
              <p className="text-zinc-300">Listen without judgment. Everyone's experience is valid.</p>
            </div>
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
              <p className="text-zinc-300">Peer support is not therapy. For crisis situations, use professional resources.</p>
            </div>
            <div className="flex items-start gap-2">
              <Flag className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
              <p className="text-zinc-300">Report any inappropriate behavior. Safety is our priority.</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Navigation Tabs */}
      <div className="flex gap-2 bg-zinc-900/50 p-1 rounded-lg">
        {[
          { id: 'find', label: 'Find Peers', icon: Search },
          { id: 'matches', label: 'Matches', icon: UserPlus },
          { id: 'chat', label: 'Chat', icon: MessageCircle },
          { id: 'profile', label: 'My Profile', icon: Heart },
        ].map(tab => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setActiveTab(tab.id as typeof activeTab)}
            className={activeTab === tab.id ? 'bg-purple-600' : ''}
          >
            <tab.icon className="w-4 h-4 mr-2" />
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Find Peers Tab */}
      {activeTab === 'find' && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-lg">Find Your Support Match</CardTitle>
            <CardDescription>
              Select experiences you'd like to connect over
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
              {experienceCategories.map(exp => (
                <Button
                  key={exp.id}
                  variant={selectedExperiences.includes(exp.id) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleExperience(exp.id)}
                  className={`justify-start ${
                    selectedExperiences.includes(exp.id)
                      ? 'bg-purple-600 hover:bg-purple-700'
                      : 'border-zinc-700 hover:border-purple-500'
                  }`}
                >
                  <span className="mr-2">{exp.icon}</span>
                  {exp.label}
                </Button>
              ))}
            </div>

            <Button
              onClick={handleSearch}
              disabled={selectedExperiences.length === 0 || isSearching}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
            >
              {isSearching ? (
                <>
                  <Clock className="w-4 h-4 mr-2 animate-spin" />
                  Finding matches...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Find Peer Supporters
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Matches Tab */}
      {activeTab === 'matches' && (
        <div className="space-y-3">
          {matches.map(peer => (
            <Card key={peer.id} className="bg-zinc-900/50 border-zinc-800 hover:border-purple-500/50 transition-colors">
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 flex items-center justify-center text-lg font-bold">
                        {peer.anonymousName.charAt(0)}
                      </div>
                      {peer.isOnline && (
                        <span className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 rounded-full border-2 border-zinc-900" />
                      )}
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">{peer.anonymousName}</h3>
                      <p className="text-xs text-zinc-400">{peer.conversationCount} conversations</p>
                    </div>
                  </div>
                  <Badge className="bg-purple-500/20 text-purple-300">
                    {peer.matchScore}% Match
                  </Badge>
                </div>

                <p className="text-sm text-zinc-300 mt-3 mb-3">{peer.bio}</p>

                <div className="flex flex-wrap gap-1 mb-3">
                  {peer.experiences.map(exp => {
                    const category = experienceCategories.find(c => c.id === exp);
                    return category ? (
                      <Badge key={exp} variant="outline" className="border-zinc-700 text-xs">
                        {category.icon} {category.label}
                      </Badge>
                    ) : null;
                  })}
                </div>

                <Button
                  onClick={() => startChat(peer)}
                  className="w-full bg-purple-600 hover:bg-purple-700"
                  size="sm"
                >
                  <MessageCircle className="w-4 h-4 mr-2" />
                  Start Conversation
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Chat Tab */}
      {activeTab === 'chat' && activeChat && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader className="pb-2 border-b border-zinc-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 flex items-center justify-center font-bold">
                    {activeChat.anonymousName.charAt(0)}
                  </div>
                  {activeChat.isOnline && (
                    <span className="absolute bottom-0 right-0 w-2.5 h-2.5 bg-green-500 rounded-full border-2 border-zinc-900" />
                  )}
                </div>
                <div>
                  <h3 className="font-semibold text-white">{activeChat.anonymousName}</h3>
                  <p className="text-xs text-zinc-400">
                    {activeChat.isOnline ? 'Online' : 'Offline'}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setActiveChat(null)}
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {/* Messages */}
            <div className="h-64 overflow-y-auto p-4 space-y-3">
              {messages.map(msg => (
                <div
                  key={msg.id}
                  className={`flex ${msg.isOwn ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      msg.isOwn
                        ? 'bg-purple-600 text-white'
                        : 'bg-zinc-800 text-zinc-200'
                    }`}
                  >
                    <p className="text-sm">{msg.content}</p>
                    <p className="text-xs opacity-60 mt-1">
                      {new Date(msg.timestamp).toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Input */}
            <div className="p-4 border-t border-zinc-800">
              <div className="flex gap-2">
                <Input
                  value={newMessage}
                  onChange={e => setNewMessage(e.target.value)}
                  placeholder="Type your message..."
                  className="bg-zinc-800 border-zinc-700"
                  onKeyPress={e => e.key === 'Enter' && sendMessage()}
                />
                <Button onClick={sendMessage} className="bg-purple-600 hover:bg-purple-700">
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === 'chat' && !activeChat && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardContent className="p-8 text-center">
            <MessageCircle className="w-12 h-12 text-zinc-600 mx-auto mb-3" />
            <h3 className="text-lg font-semibold text-zinc-400">No Active Conversation</h3>
            <p className="text-sm text-zinc-500 mb-4">
              Find a peer supporter to start chatting
            </p>
            <Button
              onClick={() => setActiveTab('matches')}
              variant="outline"
              className="border-purple-500 text-purple-400"
            >
              View Matches
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Profile Tab */}
      {activeTab === 'profile' && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-lg">Your Anonymous Profile</CardTitle>
            <CardDescription>
              This is how other peers will see you
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm text-zinc-400 mb-1 block">Anonymous Name</label>
              <Input
                value={userProfile.anonymousName}
                onChange={e => setUserProfile(prev => ({ ...prev, anonymousName: e.target.value }))}
                placeholder="e.g., HopefulHeart"
                className="bg-zinc-800 border-zinc-700"
              />
            </div>

            <div>
              <label className="text-sm text-zinc-400 mb-1 block">Your Bio</label>
              <Textarea
                value={userProfile.bio}
                onChange={e => setUserProfile(prev => ({ ...prev, bio: e.target.value }))}
                placeholder="Share a bit about yourself and your journey..."
                className="bg-zinc-800 border-zinc-700"
                rows={3}
              />
            </div>

            <div>
              <label className="text-sm text-zinc-400 mb-2 block">
                Experiences I'm comfortable sharing about:
              </label>
              <div className="flex flex-wrap gap-2">
                {experienceCategories.slice(0, 6).map(exp => (
                  <Badge
                    key={exp.id}
                    variant={userProfile.experiences.includes(exp.id) ? 'default' : 'outline'}
                    className={`cursor-pointer ${
                      userProfile.experiences.includes(exp.id)
                        ? 'bg-purple-600'
                        : 'border-zinc-700'
                    }`}
                    onClick={() =>
                      setUserProfile(prev => ({
                        ...prev,
                        experiences: prev.experiences.includes(exp.id)
                          ? prev.experiences.filter(e => e !== exp.id)
                          : [...prev.experiences, exp.id],
                      }))
                    }
                  >
                    {exp.icon} {exp.label}
                  </Badge>
                ))}
              </div>
            </div>

            <Button className="w-full bg-purple-600 hover:bg-purple-700">
              Save Profile
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
