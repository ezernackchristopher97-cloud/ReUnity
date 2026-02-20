import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Users, 
  Video, 
  Calendar, 
  Clock, 
  Plus, 
  MessageSquare, 
  Shield, 
  UserPlus,
  Settings,
  Play,
  Pause,
  Mic,
  MicOff,
  VideoIcon,
  VideoOff,
  Hand,
  Send,
  X
} from 'lucide-react';

interface GroupSession {
  id: string;
  title: string;
  description: string;
  therapistId: string;
  therapistName: string;
  type: 'support' | 'psychoeducation' | 'skills' | 'process';
  topic: string;
  maxParticipants: number;
  currentParticipants: number;
  scheduledDate: string;
  scheduledTime: string;
  duration: number; // minutes
  status: 'scheduled' | 'in-progress' | 'completed' | 'cancelled';
  isRecurring: boolean;
  recurringPattern?: 'weekly' | 'biweekly' | 'monthly';
  participants: Participant[];
  waitlist: string[];
  guidelines: string[];
}

interface Participant {
  id: string;
  name: string;
  joinedAt: number;
  isMuted: boolean;
  isVideoOn: boolean;
  hasHandRaised: boolean;
}

interface ChatMessage {
  id: string;
  participantId: string;
  participantName: string;
  content: string;
  timestamp: number;
  isTherapist: boolean;
}

interface GroupTherapySessionsProps {
  isTherapist?: boolean;
  userId?: string;
  userName?: string;
}

const SESSION_TYPES = {
  support: { label: 'Support Group', color: 'bg-blue-500/20 text-blue-400', description: 'Peer support and shared experiences' },
  psychoeducation: { label: 'Psychoeducation', color: 'bg-purple-500/20 text-purple-400', description: 'Learning about mental health topics' },
  skills: { label: 'Skills Training', color: 'bg-green-500/20 text-green-400', description: 'DBT, CBT, and coping skills' },
  process: { label: 'Process Group', color: 'bg-orange-500/20 text-orange-400', description: 'Interpersonal exploration and growth' },
};

const TOPICS = [
  'Anxiety Management',
  'Depression Support',
  'Trauma Recovery',
  'Grief & Loss',
  'Relationship Skills',
  'Stress Management',
  'Self-Esteem Building',
  'Mindfulness Practice',
  'Anger Management',
  'Addiction Recovery',
  'LGBTQ+ Support',
  'Caregiver Support',
];

export default function GroupTherapySessions({ isTherapist = false, userId = 'user1', userName = 'User' }: GroupTherapySessionsProps) {
  const [sessions, setSessions] = useState<GroupSession[]>([]);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [activeSession, setActiveSession] = useState<GroupSession | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isMuted, setIsMuted] = useState(false);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [hasHandRaised, setHasHandRaised] = useState(false);
  
  // New session form state
  const [newSession, setNewSession] = useState({
    title: '',
    description: '',
    type: 'support' as GroupSession['type'],
    topic: '',
    maxParticipants: 8,
    scheduledDate: '',
    scheduledTime: '',
    duration: 60,
    isRecurring: false,
    recurringPattern: 'weekly' as 'weekly' | 'biweekly' | 'monthly',
  });

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = () => {
    const stored = localStorage.getItem('reunity_group_sessions');
    if (stored) {
      setSessions(JSON.parse(stored));
    } else {
      // Demo sessions
      const demoSessions: GroupSession[] = [
        {
          id: '1',
          title: 'Anxiety Support Circle',
          description: 'A safe space to share experiences and learn coping strategies for anxiety.',
          therapistId: 'therapist1',
          therapistName: 'Dr. Sarah Chen',
          type: 'support',
          topic: 'Anxiety Management',
          maxParticipants: 10,
          currentParticipants: 6,
          scheduledDate: new Date(Date.now() + 86400000).toISOString().split('T')[0],
          scheduledTime: '18:00',
          duration: 90,
          status: 'scheduled',
          isRecurring: true,
          recurringPattern: 'weekly',
          participants: [],
          waitlist: [],
          guidelines: [
            'Maintain confidentiality - what\'s shared here stays here',
            'Speak from your own experience using "I" statements',
            'Be respectful and supportive of others',
            'Raise your hand to speak during discussions',
          ],
        },
        {
          id: '2',
          title: 'DBT Skills Workshop',
          description: 'Learn and practice Dialectical Behavior Therapy skills for emotional regulation.',
          therapistId: 'therapist2',
          therapistName: 'Dr. Michael Torres',
          type: 'skills',
          topic: 'Stress Management',
          maxParticipants: 12,
          currentParticipants: 8,
          scheduledDate: new Date(Date.now() + 172800000).toISOString().split('T')[0],
          scheduledTime: '14:00',
          duration: 120,
          status: 'scheduled',
          isRecurring: true,
          recurringPattern: 'biweekly',
          participants: [],
          waitlist: [],
          guidelines: [
            'Come prepared with your DBT workbook',
            'Practice skills between sessions',
            'Share your experiences with homework assignments',
          ],
        },
        {
          id: '3',
          title: 'Trauma Recovery Group',
          description: 'A trauma-informed space for healing and connection.',
          therapistId: 'therapist1',
          therapistName: 'Dr. Sarah Chen',
          type: 'process',
          topic: 'Trauma Recovery',
          maxParticipants: 8,
          currentParticipants: 5,
          scheduledDate: new Date(Date.now() + 259200000).toISOString().split('T')[0],
          scheduledTime: '10:00',
          duration: 90,
          status: 'scheduled',
          isRecurring: true,
          recurringPattern: 'weekly',
          participants: [],
          waitlist: [],
          guidelines: [
            'You control what and how much you share',
            'Grounding techniques will be available throughout',
            'Take breaks as needed',
            'No pressure to speak - listening is also participation',
          ],
        },
      ];
      setSessions(demoSessions);
      localStorage.setItem('reunity_group_sessions', JSON.stringify(demoSessions));
    }
  };

  const saveSessions = (updatedSessions: GroupSession[]) => {
    setSessions(updatedSessions);
    localStorage.setItem('reunity_group_sessions', JSON.stringify(updatedSessions));
  };

  const createSession = () => {
    if (!newSession.title || !newSession.scheduledDate || !newSession.scheduledTime) return;

    const session: GroupSession = {
      id: Date.now().toString(),
      ...newSession,
      therapistId: userId,
      therapistName: userName,
      currentParticipants: 0,
      status: 'scheduled',
      participants: [],
      waitlist: [],
      guidelines: [
        'Maintain confidentiality',
        'Be respectful and supportive',
        'Raise your hand to speak',
      ],
    };

    saveSessions([...sessions, session]);
    setShowCreateDialog(false);
    setNewSession({
      title: '',
      description: '',
      type: 'support',
      topic: '',
      maxParticipants: 8,
      scheduledDate: '',
      scheduledTime: '',
      duration: 60,
      isRecurring: false,
      recurringPattern: 'weekly',
    });
  };

  const joinSession = (sessionId: string) => {
    const updated = sessions.map(s => {
      if (s.id === sessionId && s.currentParticipants < s.maxParticipants) {
        return {
          ...s,
          currentParticipants: s.currentParticipants + 1,
          participants: [...s.participants, {
            id: userId,
            name: userName,
            joinedAt: Date.now(),
            isMuted: false,
            isVideoOn: true,
            hasHandRaised: false,
          }],
        };
      }
      return s;
    });
    saveSessions(updated);
  };

  const startSession = (session: GroupSession) => {
    setActiveSession(session);
    setChatMessages([
      {
        id: '1',
        participantId: session.therapistId,
        participantName: session.therapistName,
        content: `Welcome to ${session.title}! Please review our group guidelines and feel free to introduce yourself when you're ready.`,
        timestamp: Date.now(),
        isTherapist: true,
      },
    ]);
  };

  const sendChatMessage = () => {
    if (!chatInput.trim() || !activeSession) return;

    const message: ChatMessage = {
      id: Date.now().toString(),
      participantId: userId,
      participantName: userName,
      content: chatInput,
      timestamp: Date.now(),
      isTherapist: isTherapist,
    };

    setChatMessages([...chatMessages, message]);
    setChatInput('');
  };

  const toggleHandRaise = () => {
    setHasHandRaised(!hasHandRaised);
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'short',
      day: 'numeric',
    });
  };

  // Active Session View
  if (activeSession) {
    return (
      <div className="fixed inset-0 bg-zinc-950 z-50 flex flex-col">
        {/* Header */}
        <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Badge className={SESSION_TYPES[activeSession.type].color}>
              {SESSION_TYPES[activeSession.type].label}
            </Badge>
            <h2 className="font-semibold text-white">{activeSession.title}</h2>
            <span className="text-sm text-zinc-400">
              {activeSession.currentParticipants} participants
            </span>
          </div>
          <Button variant="destructive" size="sm" onClick={() => setActiveSession(null)}>
            <X className="w-4 h-4 mr-2" />
            Leave Session
          </Button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex">
          {/* Video Grid */}
          <div className="flex-1 p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 auto-rows-fr">
            {/* Therapist Video */}
            <div className="relative bg-zinc-800 rounded-lg aspect-video flex items-center justify-center border-2 border-emerald-500/50">
              <div className="text-center">
                <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center mx-auto mb-2">
                  <Users className="w-8 h-8 text-emerald-400" />
                </div>
                <p className="text-white font-medium">{activeSession.therapistName}</p>
                <Badge className="mt-1 bg-emerald-500/20 text-emerald-400">Host</Badge>
              </div>
              <div className="absolute bottom-2 left-2 flex gap-1">
                <div className="bg-zinc-900/80 p-1 rounded">
                  <Mic className="w-4 h-4 text-white" />
                </div>
              </div>
            </div>

            {/* Participant Videos */}
            {Array.from({ length: Math.min(activeSession.currentParticipants, 7) }).map((_, idx) => (
              <div key={idx} className="relative bg-zinc-800 rounded-lg aspect-video flex items-center justify-center">
                <div className="text-center">
                  <div className="w-12 h-12 rounded-full bg-zinc-700 flex items-center justify-center mx-auto mb-2">
                    <Users className="w-6 h-6 text-zinc-400" />
                  </div>
                  <p className="text-white text-sm">Participant {idx + 1}</p>
                </div>
              </div>
            ))}

            {/* Your Video */}
            <div className="relative bg-zinc-800 rounded-lg aspect-video flex items-center justify-center border-2 border-blue-500/50">
              <div className="text-center">
                <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center mx-auto mb-2">
                  <Users className="w-6 h-6 text-blue-400" />
                </div>
                <p className="text-white text-sm">You</p>
              </div>
              {hasHandRaised && (
                <div className="absolute top-2 right-2 bg-yellow-500 p-1 rounded">
                  <Hand className="w-4 h-4 text-black" />
                </div>
              )}
              <div className="absolute bottom-2 left-2 flex gap-1">
                <div className={`p-1 rounded ${isMuted ? 'bg-red-500' : 'bg-zinc-900/80'}`}>
                  {isMuted ? <MicOff className="w-4 h-4 text-white" /> : <Mic className="w-4 h-4 text-white" />}
                </div>
                <div className={`p-1 rounded ${!isVideoOn ? 'bg-red-500' : 'bg-zinc-900/80'}`}>
                  {isVideoOn ? <VideoIcon className="w-4 h-4 text-white" /> : <VideoOff className="w-4 h-4 text-white" />}
                </div>
              </div>
            </div>
          </div>

          {/* Chat Sidebar */}
          <div className="w-80 bg-zinc-900 border-l border-zinc-800 flex flex-col">
            <div className="p-3 border-b border-zinc-800">
              <h3 className="font-medium text-white flex items-center gap-2">
                <MessageSquare className="w-4 h-4" />
                Group Chat
              </h3>
            </div>
            <ScrollArea className="flex-1 p-3">
              <div className="space-y-3">
                {chatMessages.map((msg) => (
                  <div key={msg.id} className={`${msg.isTherapist ? 'bg-emerald-500/10' : 'bg-zinc-800'} rounded-lg p-2`}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-sm font-medium ${msg.isTherapist ? 'text-emerald-400' : 'text-white'}`}>
                        {msg.participantName}
                      </span>
                      {msg.isTherapist && <Badge className="text-xs bg-emerald-500/20 text-emerald-400">Host</Badge>}
                      <span className="text-xs text-zinc-500">
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <p className="text-sm text-zinc-300">{msg.content}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
            <div className="p-3 border-t border-zinc-800">
              <div className="flex gap-2">
                <Input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type a message..."
                  className="flex-1 bg-zinc-800 border-zinc-700"
                  onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                />
                <Button size="icon" onClick={sendChatMessage}>
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="bg-zinc-900 border-t border-zinc-800 px-4 py-3 flex items-center justify-center gap-4">
          <Button
            variant={isMuted ? 'destructive' : 'secondary'}
            size="lg"
            onClick={() => setIsMuted(!isMuted)}
          >
            {isMuted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
          </Button>
          <Button
            variant={!isVideoOn ? 'destructive' : 'secondary'}
            size="lg"
            onClick={() => setIsVideoOn(!isVideoOn)}
          >
            {isVideoOn ? <VideoIcon className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
          </Button>
          <Button
            variant={hasHandRaised ? 'default' : 'secondary'}
            size="lg"
            onClick={toggleHandRaise}
            className={hasHandRaised ? 'bg-yellow-500 hover:bg-yellow-600' : ''}
          >
            <Hand className="w-5 h-5" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Users className="w-6 h-6 text-emerald-400" />
            Group Therapy Sessions
          </h2>
          <p className="text-zinc-400 mt-1">
            {isTherapist ? 'Manage and host group therapy sessions' : 'Join supportive group sessions led by licensed therapists'}
          </p>
        </div>
        {isTherapist && (
          <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
            <DialogTrigger asChild>
              <Button className="bg-emerald-600 hover:bg-emerald-700">
                <Plus className="w-4 h-4 mr-2" />
                Create Session
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-zinc-900 border-zinc-800 max-w-lg">
              <DialogHeader>
                <DialogTitle className="text-white">Create Group Session</DialogTitle>
                <DialogDescription>Set up a new group therapy session for your clients.</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 mt-4">
                <div>
                  <Label>Session Title</Label>
                  <Input
                    value={newSession.title}
                    onChange={(e) => setNewSession({ ...newSession, title: e.target.value })}
                    placeholder="e.g., Anxiety Support Circle"
                    className="bg-zinc-800 border-zinc-700"
                  />
                </div>
                <div>
                  <Label>Description</Label>
                  <Textarea
                    value={newSession.description}
                    onChange={(e) => setNewSession({ ...newSession, description: e.target.value })}
                    placeholder="Describe what participants can expect..."
                    className="bg-zinc-800 border-zinc-700"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Session Type</Label>
                    <Select
                      value={newSession.type}
                      onValueChange={(v) => setNewSession({ ...newSession, type: v as GroupSession['type'] })}
                    >
                      <SelectTrigger className="bg-zinc-800 border-zinc-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(SESSION_TYPES).map(([key, { label }]) => (
                          <SelectItem key={key} value={key}>{label}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Topic</Label>
                    <Select
                      value={newSession.topic}
                      onValueChange={(v) => setNewSession({ ...newSession, topic: v })}
                    >
                      <SelectTrigger className="bg-zinc-800 border-zinc-700">
                        <SelectValue placeholder="Select topic" />
                      </SelectTrigger>
                      <SelectContent>
                        {TOPICS.map((topic) => (
                          <SelectItem key={topic} value={topic}>{topic}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <Label>Date</Label>
                    <Input
                      type="date"
                      value={newSession.scheduledDate}
                      onChange={(e) => setNewSession({ ...newSession, scheduledDate: e.target.value })}
                      className="bg-zinc-800 border-zinc-700"
                    />
                  </div>
                  <div>
                    <Label>Time</Label>
                    <Input
                      type="time"
                      value={newSession.scheduledTime}
                      onChange={(e) => setNewSession({ ...newSession, scheduledTime: e.target.value })}
                      className="bg-zinc-800 border-zinc-700"
                    />
                  </div>
                  <div>
                    <Label>Duration (min)</Label>
                    <Input
                      type="number"
                      value={newSession.duration}
                      onChange={(e) => setNewSession({ ...newSession, duration: parseInt(e.target.value) })}
                      className="bg-zinc-800 border-zinc-700"
                    />
                  </div>
                </div>
                <div>
                  <Label>Max Participants</Label>
                  <Input
                    type="number"
                    value={newSession.maxParticipants}
                    onChange={(e) => setNewSession({ ...newSession, maxParticipants: parseInt(e.target.value) })}
                    className="bg-zinc-800 border-zinc-700"
                    min={2}
                    max={20}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="recurring"
                    checked={newSession.isRecurring}
                    onChange={(e) => setNewSession({ ...newSession, isRecurring: e.target.checked })}
                    className="rounded"
                  />
                  <Label htmlFor="recurring">Recurring session</Label>
                  {newSession.isRecurring && (
                    <Select
                      value={newSession.recurringPattern}
                      onValueChange={(v) => setNewSession({ ...newSession, recurringPattern: v as 'weekly' | 'biweekly' | 'monthly' })}
                    >
                      <SelectTrigger className="w-32 bg-zinc-800 border-zinc-700">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="weekly">Weekly</SelectItem>
                        <SelectItem value="biweekly">Biweekly</SelectItem>
                        <SelectItem value="monthly">Monthly</SelectItem>
                      </SelectContent>
                    </Select>
                  )}
                </div>
                <Button onClick={createSession} className="w-full bg-emerald-600 hover:bg-emerald-700">
                  Create Session
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        )}
      </div>

      {/* Session Types Legend */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(SESSION_TYPES).map(([key, { label, color, description }]) => (
          <div key={key} className="flex items-center gap-2 bg-zinc-800/50 rounded-lg px-3 py-2">
            <Badge className={color}>{label}</Badge>
            <span className="text-xs text-zinc-400">{description}</span>
          </div>
        ))}
      </div>

      {/* Sessions Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sessions.filter(s => s.status === 'scheduled').map((session) => (
          <Card key={session.id} className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <Badge className={SESSION_TYPES[session.type].color}>
                  {SESSION_TYPES[session.type].label}
                </Badge>
                {session.isRecurring && (
                  <Badge variant="outline" className="text-xs">
                    {session.recurringPattern}
                  </Badge>
                )}
              </div>
              <CardTitle className="text-lg text-white mt-2">{session.title}</CardTitle>
              <CardDescription className="text-zinc-400">{session.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-4 text-sm text-zinc-400">
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  {formatDate(session.scheduledDate)}
                </div>
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  {session.scheduledTime}
                </div>
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <Shield className="w-4 h-4 text-emerald-400" />
                <span className="text-zinc-300">{session.therapistName}</span>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-zinc-400">
                  <Users className="w-4 h-4" />
                  <span>{session.currentParticipants}/{session.maxParticipants} participants</span>
                </div>
                <span className="text-sm text-zinc-500">{session.duration} min</span>
              </div>

              {/* Progress bar for capacity */}
              <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-emerald-500 transition-all"
                  style={{ width: `${(session.currentParticipants / session.maxParticipants) * 100}%` }}
                />
              </div>

              <div className="flex gap-2">
                {isTherapist && session.therapistId === userId ? (
                  <Button 
                    className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                    onClick={() => startSession(session)}
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Start Session
                  </Button>
                ) : session.currentParticipants < session.maxParticipants ? (
                  <Button 
                    className="flex-1 bg-blue-600 hover:bg-blue-700"
                    onClick={() => joinSession(session.id)}
                  >
                    <UserPlus className="w-4 h-4 mr-2" />
                    Join Session
                  </Button>
                ) : (
                  <Button variant="outline" className="flex-1" disabled>
                    Session Full
                  </Button>
                )}
                <Button variant="outline" size="icon">
                  <Settings className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {sessions.filter(s => s.status === 'scheduled').length === 0 && (
        <div className="text-center py-12 bg-zinc-900/50 rounded-lg border border-zinc-800">
          <Users className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-white mb-2">No Upcoming Sessions</h3>
          <p className="text-zinc-400">
            {isTherapist 
              ? 'Create your first group therapy session to get started.'
              : 'Check back later for available group sessions.'}
          </p>
        </div>
      )}
    </div>
  );
}
