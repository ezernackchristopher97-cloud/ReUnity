import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Users, 
  MessageCircle, 
  Phone, 
  Heart, 
  Coffee, 
  Sun,
  Star,
  Clock,
  Check,
  X,
  ChevronRight,
  Sparkles,
  UserPlus
} from 'lucide-react';

// Connection prompt types based on isolation patterns
const connectionPrompts = {
  gentle: [
    {
      id: 'text_friend',
      title: 'Send a quick text',
      description: 'Reach out to someone you trust with a simple "thinking of you"',
      icon: MessageCircle,
      action: 'Text a friend',
      difficulty: 'easy'
    },
    {
      id: 'reply_message',
      title: 'Reply to a message',
      description: 'Is there a message you\'ve been meaning to respond to?',
      icon: MessageCircle,
      action: 'Check messages',
      difficulty: 'easy'
    },
    {
      id: 'share_something',
      title: 'Share something small',
      description: 'Send a meme, article, or photo to someone who might enjoy it',
      icon: Star,
      action: 'Share with someone',
      difficulty: 'easy'
    }
  ],
  moderate: [
    {
      id: 'voice_call',
      title: 'Make a short call',
      description: 'A 5-minute call can make a big difference. Who would you like to hear?',
      icon: Phone,
      action: 'Call someone',
      difficulty: 'medium'
    },
    {
      id: 'schedule_hangout',
      title: 'Plan something to look forward to',
      description: 'Schedule a coffee date, walk, or video chat for later this week',
      icon: Coffee,
      action: 'Make plans',
      difficulty: 'medium'
    },
    {
      id: 'join_online',
      title: 'Join an online community',
      description: 'Participate in a forum, group chat, or online event',
      icon: Users,
      action: 'Find a community',
      difficulty: 'medium'
    }
  ],
  meaningful: [
    {
      id: 'express_gratitude',
      title: 'Express gratitude',
      description: 'Tell someone specific why you appreciate them',
      icon: Heart,
      action: 'Share appreciation',
      difficulty: 'meaningful'
    },
    {
      id: 'ask_for_help',
      title: 'Ask for support',
      description: 'It\'s okay to reach out when you need help. Who can you trust?',
      icon: UserPlus,
      action: 'Reach out',
      difficulty: 'meaningful'
    },
    {
      id: 'reconnect',
      title: 'Reconnect with someone',
      description: 'Is there someone you\'ve lost touch with who you miss?',
      icon: Users,
      action: 'Reconnect',
      difficulty: 'meaningful'
    }
  ]
};

// Isolation indicators to detect
const isolationIndicators = [
  'alone', 'lonely', 'isolated', 'no one', 'nobody', 'by myself',
  'no friends', 'no one cares', 'disconnected', 'withdrawn',
  'haven\'t talked', 'haven\'t seen anyone', 'staying in',
  'avoiding people', 'don\'t want to see', 'pushing away'
];

// Encouraging messages for completing prompts
const encouragingMessages = [
  "That took courage. You did something meaningful today.",
  "Connection is a gift you give yourself. Well done.",
  "Small steps lead to big changes. You're doing great.",
  "Every reach-out matters. You're building your support network.",
  "You chose connection over isolation. That's powerful.",
  "Human connection heals. Thank you for taking that step."
];

interface ConnectionLog {
  promptId: string;
  timestamp: string;
  completed: boolean;
  notes?: string;
}

interface SocialConnectionPromptsProps {
  recentMessages?: string[];
  daysWithoutSocialActivity?: number;
  onPromptCompleted?: (promptId: string) => void;
}

export function SocialConnectionPrompts({
  recentMessages = [],
  daysWithoutSocialActivity = 0,
  onPromptCompleted
}: SocialConnectionPromptsProps) {
  const [connectionLog, setConnectionLog] = useState<ConnectionLog[]>([]);
  const [currentPrompt, setCurrentPrompt] = useState<typeof connectionPrompts.gentle[0] | null>(null);
  const [showEncouragement, setShowEncouragement] = useState(false);
  const [encouragementMessage, setEncouragementMessage] = useState('');
  const [dismissedToday, setDismissedToday] = useState(false);
  const [isolationScore, setIsolationScore] = useState(0);
  
  useEffect(() => {
    // Load connection log
    const saved = localStorage.getItem('reunity_connection_log');
    if (saved) {
      setConnectionLog(JSON.parse(saved));
    }
    
    // Check if dismissed today
    const dismissedDate = localStorage.getItem('reunity_connection_dismissed');
    if (dismissedDate === new Date().toDateString()) {
      setDismissedToday(true);
    }
    
    // Calculate isolation score
    calculateIsolationScore();
  }, [recentMessages, daysWithoutSocialActivity]);
  
  const calculateIsolationScore = () => {
    let score = 0;
    
    // Check recent messages for isolation indicators
    const allText = recentMessages.join(' ').toLowerCase();
    isolationIndicators.forEach(indicator => {
      if (allText.includes(indicator)) {
        score += 2;
      }
    });
    
    // Add score based on days without social activity
    score += Math.min(daysWithoutSocialActivity * 1.5, 10);
    
    // Check connection log - reduce score if recent connections
    const recentConnections = connectionLog.filter(log => {
      const logDate = new Date(log.timestamp);
      const daysDiff = (Date.now() - logDate.getTime()) / (1000 * 60 * 60 * 24);
      return daysDiff < 3 && log.completed;
    });
    score -= recentConnections.length * 2;
    
    setIsolationScore(Math.max(0, Math.min(score, 10)));
    
    // Select appropriate prompt based on score
    selectPrompt(Math.max(0, Math.min(score, 10)));
  };
  
  const selectPrompt = (score: number) => {
    // Don't show if dismissed today or low isolation score
    if (dismissedToday || score < 3) {
      setCurrentPrompt(null);
      return;
    }
    
    let promptPool: typeof connectionPrompts.gentle;
    
    if (score >= 7) {
      // High isolation - suggest gentle, easy prompts
      promptPool = connectionPrompts.gentle;
    } else if (score >= 5) {
      // Moderate isolation - suggest moderate prompts
      promptPool = connectionPrompts.moderate;
    } else {
      // Lower isolation - suggest meaningful prompts
      promptPool = connectionPrompts.meaningful;
    }
    
    // Filter out recently completed prompts
    const recentlyCompleted = connectionLog
      .filter(log => {
        const logDate = new Date(log.timestamp);
        const daysDiff = (Date.now() - logDate.getTime()) / (1000 * 60 * 60 * 24);
        return daysDiff < 2 && log.completed;
      })
      .map(log => log.promptId);
    
    const availablePrompts = promptPool.filter(p => !recentlyCompleted.includes(p.id));
    
    if (availablePrompts.length > 0) {
      const randomIndex = Math.floor(Math.random() * availablePrompts.length);
      setCurrentPrompt(availablePrompts[randomIndex]);
    } else {
      // All prompts completed recently, show first one anyway
      setCurrentPrompt(promptPool[0]);
    }
  };
  
  const completePrompt = () => {
    if (!currentPrompt) return;
    
    const newLog: ConnectionLog = {
      promptId: currentPrompt.id,
      timestamp: new Date().toISOString(),
      completed: true
    };
    
    const updatedLog = [...connectionLog, newLog];
    setConnectionLog(updatedLog);
    localStorage.setItem('reunity_connection_log', JSON.stringify(updatedLog));
    
    // Show encouragement
    const randomMessage = encouragingMessages[Math.floor(Math.random() * encouragingMessages.length)];
    setEncouragementMessage(randomMessage);
    setShowEncouragement(true);
    
    // Callback
    if (onPromptCompleted) {
      onPromptCompleted(currentPrompt.id);
    }
    
    // Hide prompt after delay
    setTimeout(() => {
      setShowEncouragement(false);
      setCurrentPrompt(null);
    }, 4000);
  };
  
  const dismissPrompt = () => {
    setDismissedToday(true);
    localStorage.setItem('reunity_connection_dismissed', new Date().toDateString());
    setCurrentPrompt(null);
  };
  
  const remindLater = () => {
    setCurrentPrompt(null);
    // Will show again on next calculation
  };
  
  // Get connection streak
  const getConnectionStreak = () => {
    let streak = 0;
    const sortedLog = [...connectionLog]
      .filter(log => log.completed)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    
    if (sortedLog.length === 0) return 0;
    
    let currentDate = new Date();
    currentDate.setHours(0, 0, 0, 0);
    
    for (const log of sortedLog) {
      const logDate = new Date(log.timestamp);
      logDate.setHours(0, 0, 0, 0);
      
      const daysDiff = Math.floor((currentDate.getTime() - logDate.getTime()) / (1000 * 60 * 60 * 24));
      
      if (daysDiff <= 1) {
        streak++;
        currentDate = logDate;
      } else {
        break;
      }
    }
    
    return streak;
  };
  
  // Don't render if no prompt or dismissed
  if (!currentPrompt && !showEncouragement) {
    // Show mini widget if there's a streak
    const streak = getConnectionStreak();
    if (streak > 0) {
      return (
        <div className="flex items-center gap-2 p-2 rounded-lg bg-teal-900/20 border border-teal-700/30">
          <Users className="w-4 h-4 text-teal-400" />
          <span className="text-sm text-teal-300">
            {streak} day connection streak
          </span>
          <Sparkles className="w-3 h-3 text-teal-400" />
        </div>
      );
    }
    return null;
  }
  
  if (showEncouragement) {
    return (
      <Card className="bg-gradient-to-br from-teal-900/40 to-emerald-900/40 border-teal-500/50">
        <CardContent className="py-6">
          <div className="flex flex-col items-center text-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-teal-500/30 flex items-center justify-center">
              <Check className="w-6 h-6 text-teal-400" />
            </div>
            <p className="text-teal-200 text-lg">
              {encouragementMessage}
            </p>
            <div className="flex items-center gap-2 text-teal-400">
              <Sparkles className="w-4 h-4" />
              <span className="text-sm">Connection logged</span>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  const PromptIcon = currentPrompt?.icon || Users;
  
  return (
    <Card className="bg-gradient-to-br from-teal-900/30 to-cyan-900/30 border-teal-700/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-teal-200">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5" />
            <span>Connection Prompt</span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={dismissPrompt}
            className="text-gray-400 hover:text-gray-300 h-6 w-6 p-0"
            title="Don't show today"
          >
            <X className="w-4 h-4" />
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-full bg-teal-500/20 flex items-center justify-center flex-shrink-0">
            <PromptIcon className="w-5 h-5 text-teal-400" />
          </div>
          <div className="flex-1">
            <h4 className="text-teal-100 font-medium">
              {currentPrompt?.title}
            </h4>
            <p className="text-sm text-teal-300/80 mt-1">
              {currentPrompt?.description}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            onClick={completePrompt}
            className="flex-1 bg-teal-600 hover:bg-teal-700 text-white"
          >
            {currentPrompt?.action}
            <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
          <Button
            variant="outline"
            onClick={remindLater}
            className="border-teal-700 text-teal-300 hover:bg-teal-900/50"
          >
            <Clock className="w-4 h-4" />
          </Button>
        </div>
        
        <p className="text-xs text-teal-400/60 text-center">
          Human connection is healing. Even small gestures matter.
        </p>
      </CardContent>
    </Card>
  );
}

export default SocialConnectionPrompts;
