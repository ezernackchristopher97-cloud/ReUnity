import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Award, Star, Flame, Heart, Shield, Brain, Sun, Moon, Zap, Target, Trophy, Lock } from 'lucide-react';
import { toast } from 'sonner';

interface Badge {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  requirement: number;
  category: 'streak' | 'milestone' | 'skill' | 'community';
  color: string;
}

interface UserProgress {
  journalEntries: number;
  checkIns: number;
  groundingExercises: number;
  chatSessions: number;
  consecutiveDays: number;
  moodImprovements: number;
  resourcesViewed: number;
  peerConnections: number;
}

const BADGES: Badge[] = [
  // Streak badges
  { id: 'first-step', name: 'First Step', description: 'Complete your first check-in', icon: <Star className="w-6 h-6" />, requirement: 1, category: 'streak', color: 'from-yellow-400 to-orange-500' },
  { id: 'week-warrior', name: 'Week Warrior', description: '7 consecutive days of check-ins', icon: <Flame className="w-6 h-6" />, requirement: 7, category: 'streak', color: 'from-orange-500 to-red-500' },
  { id: 'month-master', name: 'Month Master', description: '30 consecutive days of check-ins', icon: <Trophy className="w-6 h-6" />, requirement: 30, category: 'streak', color: 'from-purple-500 to-pink-500' },
  
  // Milestone badges
  { id: 'journal-starter', name: 'Journal Starter', description: 'Write 5 journal entries', icon: <Heart className="w-6 h-6" />, requirement: 5, category: 'milestone', color: 'from-pink-400 to-rose-500' },
  { id: 'journal-pro', name: 'Journal Pro', description: 'Write 25 journal entries', icon: <Heart className="w-6 h-6" />, requirement: 25, category: 'milestone', color: 'from-rose-500 to-red-600' },
  { id: 'chat-explorer', name: 'Chat Explorer', description: 'Have 10 chat sessions', icon: <Brain className="w-6 h-6" />, requirement: 10, category: 'milestone', color: 'from-blue-400 to-indigo-500' },
  
  // Skill badges
  { id: 'grounding-guru', name: 'Grounding Guru', description: 'Complete 20 grounding exercises', icon: <Shield className="w-6 h-6" />, requirement: 20, category: 'skill', color: 'from-emerald-400 to-teal-500' },
  { id: 'breathing-master', name: 'Breathing Master', description: 'Complete 50 breathing exercises', icon: <Sun className="w-6 h-6" />, requirement: 50, category: 'skill', color: 'from-cyan-400 to-blue-500' },
  { id: 'mood-tracker', name: 'Mood Tracker', description: 'Log mood for 14 days', icon: <Moon className="w-6 h-6" />, requirement: 14, category: 'skill', color: 'from-indigo-400 to-purple-500' },
  
  // Community badges
  { id: 'resource-reader', name: 'Resource Reader', description: 'View 10 resources', icon: <Zap className="w-6 h-6" />, requirement: 10, category: 'community', color: 'from-amber-400 to-yellow-500' },
  { id: 'peer-connector', name: 'Peer Connector', description: 'Connect with 3 peers', icon: <Target className="w-6 h-6" />, requirement: 3, category: 'community', color: 'from-green-400 to-emerald-500' },
  { id: 'improvement-champion', name: 'Improvement Champion', description: 'Show mood improvement 5 times', icon: <Award className="w-6 h-6" />, requirement: 5, category: 'community', color: 'from-violet-400 to-purple-600' },
];

export function ProgressBadges() {
  const [progress, setProgress] = useState<UserProgress>({
    journalEntries: 0,
    checkIns: 0,
    groundingExercises: 0,
    chatSessions: 0,
    consecutiveDays: 0,
    moodImprovements: 0,
    resourcesViewed: 0,
    peerConnections: 0,
  });
  const [unlockedBadges, setUnlockedBadges] = useState<string[]>([]);
  const [selectedBadge, setSelectedBadge] = useState<Badge | null>(null);

  useEffect(() => {
    // Load progress from localStorage
    const savedProgress = localStorage.getItem('reunity_user_progress');
    const savedBadges = localStorage.getItem('reunity_unlocked_badges');
    
    if (savedProgress) {
      setProgress(JSON.parse(savedProgress));
    }
    
    if (savedBadges) {
      setUnlockedBadges(JSON.parse(savedBadges));
    }
  }, []);

  const getBadgeProgress = (badge: Badge): number => {
    switch (badge.id) {
      case 'first-step':
      case 'week-warrior':
      case 'month-master':
        return Math.min(progress.consecutiveDays / badge.requirement * 100, 100);
      case 'journal-starter':
      case 'journal-pro':
        return Math.min(progress.journalEntries / badge.requirement * 100, 100);
      case 'chat-explorer':
        return Math.min(progress.chatSessions / badge.requirement * 100, 100);
      case 'grounding-guru':
      case 'breathing-master':
        return Math.min(progress.groundingExercises / badge.requirement * 100, 100);
      case 'mood-tracker':
        return Math.min(progress.checkIns / badge.requirement * 100, 100);
      case 'resource-reader':
        return Math.min(progress.resourcesViewed / badge.requirement * 100, 100);
      case 'peer-connector':
        return Math.min(progress.peerConnections / badge.requirement * 100, 100);
      case 'improvement-champion':
        return Math.min(progress.moodImprovements / badge.requirement * 100, 100);
      default:
        return 0;
    }
  };

  const isUnlocked = (badge: Badge): boolean => {
    return unlockedBadges.includes(badge.id);
  };

  const unlockBadge = (badge: Badge) => {
    if (!isUnlocked(badge) && getBadgeProgress(badge) >= 100) {
      const newUnlocked = [...unlockedBadges, badge.id];
      setUnlockedBadges(newUnlocked);
      localStorage.setItem('reunity_unlocked_badges', JSON.stringify(newUnlocked));
      toast.success(`🎉 Badge Unlocked: ${badge.name}!`);
    }
  };

  // Check for new unlocks
  useEffect(() => {
    BADGES.forEach(badge => {
      if (getBadgeProgress(badge) >= 100 && !isUnlocked(badge)) {
        unlockBadge(badge);
      }
    });
  }, [progress]);

  const totalBadges = BADGES.length;
  const earnedBadges = unlockedBadges.length;
  const overallProgress = (earnedBadges / totalBadges) * 100;

  const categories = ['streak', 'milestone', 'skill', 'community'] as const;
  const categoryLabels = {
    streak: '🔥 Streak',
    milestone: '🎯 Milestones',
    skill: '💪 Skills',
    community: '🤝 Community',
  };

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Award className="w-5 h-5 text-emerald-400" />
            Progress & Achievements
          </CardTitle>
          <div className="text-sm text-zinc-400">
            {earnedBadges}/{totalBadges} badges
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Overall progress */}
        <div className="mb-6">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-zinc-400">Overall Progress</span>
            <span className="text-emerald-400">{Math.round(overallProgress)}%</span>
          </div>
          <Progress value={overallProgress} className="h-2" />
        </div>

        {/* Badge grid by category */}
        {categories.map(category => (
          <div key={category} className="mb-6">
            <h3 className="text-sm font-medium text-zinc-300 mb-3">{categoryLabels[category]}</h3>
            <div className="grid grid-cols-3 gap-3">
              {BADGES.filter(b => b.category === category).map(badge => {
                const unlocked = isUnlocked(badge);
                const badgeProgress = getBadgeProgress(badge);
                
                return (
                  <button
                    key={badge.id}
                    onClick={() => setSelectedBadge(badge)}
                    className={`
                      relative p-3 rounded-xl transition-all
                      ${unlocked 
                        ? `bg-gradient-to-br ${badge.color} shadow-lg` 
                        : 'bg-zinc-800/50 hover:bg-zinc-800'}
                    `}
                  >
                    <div className={`
                      flex flex-col items-center gap-1
                      ${unlocked ? 'text-white' : 'text-zinc-500'}
                    `}>
                      {unlocked ? badge.icon : <Lock className="w-6 h-6" />}
                      <span className="text-xs font-medium text-center leading-tight">
                        {badge.name}
                      </span>
                    </div>
                    {!unlocked && badgeProgress > 0 && (
                      <div className="absolute bottom-1 left-1 right-1">
                        <Progress value={badgeProgress} className="h-1" />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}

        {/* Selected badge detail */}
        {selectedBadge && (
          <div className="mt-4 p-4 bg-zinc-800/50 rounded-xl">
            <div className="flex items-center gap-3 mb-2">
              <div className={`
                p-2 rounded-lg
                ${isUnlocked(selectedBadge) 
                  ? `bg-gradient-to-br ${selectedBadge.color}` 
                  : 'bg-zinc-700'}
              `}>
                {isUnlocked(selectedBadge) ? selectedBadge.icon : <Lock className="w-6 h-6 text-zinc-400" />}
              </div>
              <div>
                <h4 className="font-medium text-white">{selectedBadge.name}</h4>
                <p className="text-sm text-zinc-400">{selectedBadge.description}</p>
              </div>
            </div>
            <div className="mt-3">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-zinc-400">Progress</span>
                <span className={isUnlocked(selectedBadge) ? 'text-emerald-400' : 'text-zinc-400'}>
                  {Math.round(getBadgeProgress(selectedBadge))}%
                </span>
              </div>
              <Progress value={getBadgeProgress(selectedBadge)} className="h-2" />
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="mt-2 w-full"
              onClick={() => setSelectedBadge(null)}
            >
              Close
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Helper function to update progress (call this from other components)
export function updateProgress(key: keyof UserProgress, increment: number = 1) {
  const savedProgress = localStorage.getItem('reunity_user_progress');
  const progress: UserProgress = savedProgress ? JSON.parse(savedProgress) : {
    journalEntries: 0,
    checkIns: 0,
    groundingExercises: 0,
    chatSessions: 0,
    consecutiveDays: 0,
    moodImprovements: 0,
    resourcesViewed: 0,
    peerConnections: 0,
  };
  
  progress[key] += increment;
  localStorage.setItem('reunity_user_progress', JSON.stringify(progress));
  
  // Dispatch event for components to update
  window.dispatchEvent(new CustomEvent('reunity-progress-update', { detail: progress }));
}

export default ProgressBadges;
