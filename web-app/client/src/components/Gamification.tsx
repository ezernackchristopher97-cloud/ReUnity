import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Flame, 
  Trophy, 
  Star, 
  Heart, 
  BookOpen, 
  Wind, 
  Calendar, 
  Target,
  Award,
  Zap,
  Shield,
  Sparkles,
  Crown,
  Medal,
  Gift,
  CheckCircle2,
  Lock
} from 'lucide-react';

interface Streak {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  currentStreak: number;
  longestStreak: number;
  lastCompletedDate: string | null;
  isActiveToday: boolean;
  color: string;
}

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  category: 'streak' | 'milestone' | 'special' | 'community';
  requirement: string;
  progress: number;
  maxProgress: number;
  isUnlocked: boolean;
  unlockedDate?: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  xpReward: number;
}

interface GamificationProps {
  userId?: string;
  compact?: boolean;
}

const RARITY_COLORS = {
  common: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30',
  rare: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  epic: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  legendary: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
};

const RARITY_GLOW = {
  common: '',
  rare: 'shadow-blue-500/20',
  epic: 'shadow-purple-500/20',
  legendary: 'shadow-amber-500/20 animate-pulse',
};

export default function Gamification({ userId = 'user1', compact = false }: GamificationProps) {
  const [streaks, setStreaks] = useState<Streak[]>([]);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [totalXP, setTotalXP] = useState(0);
  const [level, setLevel] = useState(1);
  const [showCelebration, setShowCelebration] = useState(false);
  const [newAchievement, setNewAchievement] = useState<Achievement | null>(null);

  useEffect(() => {
    loadGamificationData();
  }, []);

  const loadGamificationData = () => {
    const storedStreaks = localStorage.getItem(`reunity_streaks_${userId}`);
    const storedAchievements = localStorage.getItem(`reunity_achievements_${userId}`);
    const storedXP = localStorage.getItem(`reunity_xp_${userId}`);

    if (storedStreaks) {
      setStreaks(JSON.parse(storedStreaks));
    } else {
      initializeStreaks();
    }

    if (storedAchievements) {
      setAchievements(JSON.parse(storedAchievements));
    } else {
      initializeAchievements();
    }

    if (storedXP) {
      const xp = parseInt(storedXP);
      setTotalXP(xp);
      setLevel(calculateLevel(xp));
    }
  };

  const calculateLevel = (xp: number): number => {
    // Level formula: each level requires progressively more XP
    return Math.floor(Math.sqrt(xp / 100)) + 1;
  };

  const getXPForNextLevel = (currentLevel: number): number => {
    return Math.pow(currentLevel, 2) * 100;
  };

  const initializeStreaks = () => {
    const defaultStreaks: Streak[] = [
      {
        id: 'checkin',
        name: 'Daily Check-In',
        description: 'Complete your daily wellness check-in',
        icon: <CheckCircle2 className="w-5 h-5" />,
        currentStreak: 7,
        longestStreak: 14,
        lastCompletedDate: new Date().toISOString().split('T')[0],
        isActiveToday: true,
        color: 'text-emerald-400',
      },
      {
        id: 'journal',
        name: 'Journaling',
        description: 'Write in your journal',
        icon: <BookOpen className="w-5 h-5" />,
        currentStreak: 3,
        longestStreak: 10,
        lastCompletedDate: new Date().toISOString().split('T')[0],
        isActiveToday: true,
        color: 'text-blue-400',
      },
      {
        id: 'meditation',
        name: 'Mindfulness',
        description: 'Practice grounding or meditation',
        icon: <Wind className="w-5 h-5" />,
        currentStreak: 5,
        longestStreak: 21,
        lastCompletedDate: new Date(Date.now() - 86400000).toISOString().split('T')[0],
        isActiveToday: false,
        color: 'text-purple-400',
      },
      {
        id: 'selfcare',
        name: 'Self-Care',
        description: 'Complete a self-care activity',
        icon: <Heart className="w-5 h-5" />,
        currentStreak: 2,
        longestStreak: 7,
        lastCompletedDate: new Date().toISOString().split('T')[0],
        isActiveToday: true,
        color: 'text-pink-400',
      },
    ];
    setStreaks(defaultStreaks);
    localStorage.setItem(`reunity_streaks_${userId}`, JSON.stringify(defaultStreaks));
  };

  const initializeAchievements = () => {
    const defaultAchievements: Achievement[] = [
      // Streak Achievements
      {
        id: 'first_checkin',
        name: 'First Steps',
        description: 'Complete your first daily check-in',
        icon: <Star className="w-6 h-6" />,
        category: 'streak',
        requirement: 'Complete 1 check-in',
        progress: 1,
        maxProgress: 1,
        isUnlocked: true,
        unlockedDate: new Date(Date.now() - 604800000).toISOString(),
        rarity: 'common',
        xpReward: 50,
      },
      {
        id: 'week_warrior',
        name: 'Week Warrior',
        description: 'Maintain a 7-day check-in streak',
        icon: <Flame className="w-6 h-6" />,
        category: 'streak',
        requirement: '7-day streak',
        progress: 7,
        maxProgress: 7,
        isUnlocked: true,
        unlockedDate: new Date().toISOString(),
        rarity: 'rare',
        xpReward: 200,
      },
      {
        id: 'month_master',
        name: 'Month Master',
        description: 'Maintain a 30-day check-in streak',
        icon: <Crown className="w-6 h-6" />,
        category: 'streak',
        requirement: '30-day streak',
        progress: 7,
        maxProgress: 30,
        isUnlocked: false,
        rarity: 'epic',
        xpReward: 1000,
      },
      {
        id: 'century_club',
        name: 'Century Club',
        description: 'Maintain a 100-day check-in streak',
        icon: <Trophy className="w-6 h-6" />,
        category: 'streak',
        requirement: '100-day streak',
        progress: 7,
        maxProgress: 100,
        isUnlocked: false,
        rarity: 'legendary',
        xpReward: 5000,
      },
      // Milestone Achievements
      {
        id: 'journal_starter',
        name: 'Dear Diary',
        description: 'Write your first journal entry',
        icon: <BookOpen className="w-6 h-6" />,
        category: 'milestone',
        requirement: 'Write 1 journal entry',
        progress: 1,
        maxProgress: 1,
        isUnlocked: true,
        unlockedDate: new Date(Date.now() - 432000000).toISOString(),
        rarity: 'common',
        xpReward: 50,
      },
      {
        id: 'storyteller',
        name: 'Storyteller',
        description: 'Write 50 journal entries',
        icon: <Award className="w-6 h-6" />,
        category: 'milestone',
        requirement: 'Write 50 entries',
        progress: 12,
        maxProgress: 50,
        isUnlocked: false,
        rarity: 'rare',
        xpReward: 500,
      },
      {
        id: 'zen_master',
        name: 'Zen Master',
        description: 'Complete 100 grounding exercises',
        icon: <Wind className="w-6 h-6" />,
        category: 'milestone',
        requirement: '100 grounding exercises',
        progress: 23,
        maxProgress: 100,
        isUnlocked: false,
        rarity: 'epic',
        xpReward: 1000,
      },
      // Special Achievements
      {
        id: 'night_owl',
        name: 'Night Owl',
        description: 'Complete a check-in after midnight',
        icon: <Sparkles className="w-6 h-6" />,
        category: 'special',
        requirement: 'Check-in after midnight',
        progress: 1,
        maxProgress: 1,
        isUnlocked: true,
        unlockedDate: new Date(Date.now() - 259200000).toISOString(),
        rarity: 'rare',
        xpReward: 100,
      },
      {
        id: 'comeback_kid',
        name: 'Comeback Kid',
        description: 'Return after a 7+ day break and check in',
        icon: <Zap className="w-6 h-6" />,
        category: 'special',
        requirement: 'Return after 7+ day break',
        progress: 0,
        maxProgress: 1,
        isUnlocked: false,
        rarity: 'rare',
        xpReward: 150,
      },
      {
        id: 'crisis_survivor',
        name: 'Crisis Survivor',
        description: 'Successfully navigate a crisis moment with support',
        icon: <Shield className="w-6 h-6" />,
        category: 'special',
        requirement: 'Navigate crisis with support',
        progress: 1,
        maxProgress: 1,
        isUnlocked: true,
        unlockedDate: new Date(Date.now() - 172800000).toISOString(),
        rarity: 'epic',
        xpReward: 500,
      },
      // Community Achievements
      {
        id: 'group_joiner',
        name: 'Stronger Together',
        description: 'Join your first group therapy session',
        icon: <Heart className="w-6 h-6" />,
        category: 'community',
        requirement: 'Join 1 group session',
        progress: 0,
        maxProgress: 1,
        isUnlocked: false,
        rarity: 'common',
        xpReward: 100,
      },
      {
        id: 'support_network',
        name: 'Support Network',
        description: 'Add 3 trusted contacts',
        icon: <Medal className="w-6 h-6" />,
        category: 'community',
        requirement: 'Add 3 trusted contacts',
        progress: 2,
        maxProgress: 3,
        isUnlocked: false,
        rarity: 'rare',
        xpReward: 200,
      },
    ];
    setAchievements(defaultAchievements);
    localStorage.setItem(`reunity_achievements_${userId}`, JSON.stringify(defaultAchievements));
    
    // Calculate initial XP from unlocked achievements
    const initialXP = defaultAchievements
      .filter(a => a.isUnlocked)
      .reduce((sum, a) => sum + a.xpReward, 0);
    setTotalXP(initialXP);
    setLevel(calculateLevel(initialXP));
    localStorage.setItem(`reunity_xp_${userId}`, initialXP.toString());
  };

  const completeStreak = (streakId: string) => {
    const today = new Date().toISOString().split('T')[0];
    const updated = streaks.map(s => {
      if (s.id === streakId && !s.isActiveToday) {
        const newStreak = s.currentStreak + 1;
        return {
          ...s,
          currentStreak: newStreak,
          longestStreak: Math.max(s.longestStreak, newStreak),
          lastCompletedDate: today,
          isActiveToday: true,
        };
      }
      return s;
    });
    setStreaks(updated);
    localStorage.setItem(`reunity_streaks_${userId}`, JSON.stringify(updated));
    
    // Add XP for completing streak
    addXP(25);
    
    // Check for streak achievements
    checkStreakAchievements(updated);
  };

  const addXP = (amount: number) => {
    const newXP = totalXP + amount;
    const newLevel = calculateLevel(newXP);
    
    if (newLevel > level) {
      setShowCelebration(true);
      setTimeout(() => setShowCelebration(false), 3000);
    }
    
    setTotalXP(newXP);
    setLevel(newLevel);
    localStorage.setItem(`reunity_xp_${userId}`, newXP.toString());
  };

  const checkStreakAchievements = (currentStreaks: Streak[]) => {
    const checkInStreak = currentStreaks.find(s => s.id === 'checkin');
    if (!checkInStreak) return;

    const updatedAchievements = achievements.map(a => {
      if (a.category === 'streak' && !a.isUnlocked) {
        if (a.id === 'week_warrior' && checkInStreak.currentStreak >= 7) {
          setNewAchievement(a);
          addXP(a.xpReward);
          return { ...a, isUnlocked: true, unlockedDate: new Date().toISOString(), progress: 7 };
        }
        if (a.id === 'month_master' && checkInStreak.currentStreak >= 30) {
          setNewAchievement(a);
          addXP(a.xpReward);
          return { ...a, isUnlocked: true, unlockedDate: new Date().toISOString(), progress: 30 };
        }
        if (a.id === 'century_club' && checkInStreak.currentStreak >= 100) {
          setNewAchievement(a);
          addXP(a.xpReward);
          return { ...a, isUnlocked: true, unlockedDate: new Date().toISOString(), progress: 100 };
        }
        // Update progress
        return { ...a, progress: Math.min(checkInStreak.currentStreak, a.maxProgress) };
      }
      return a;
    });

    setAchievements(updatedAchievements);
    localStorage.setItem(`reunity_achievements_${userId}`, JSON.stringify(updatedAchievements));
  };

  const xpForCurrentLevel = getXPForNextLevel(level - 1);
  const xpForNextLevel = getXPForNextLevel(level);
  const xpProgress = ((totalXP - xpForCurrentLevel) / (xpForNextLevel - xpForCurrentLevel)) * 100;

  const unlockedAchievements = achievements.filter(a => a.isUnlocked);
  const lockedAchievements = achievements.filter(a => !a.isUnlocked);

  if (compact) {
    return (
      <Card className="bg-zinc-900 border-zinc-800">
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
                <span className="text-lg font-bold text-white">{level}</span>
              </div>
              <div>
                <p className="text-sm text-zinc-400">Level {level}</p>
                <p className="text-lg font-bold text-white">{totalXP.toLocaleString()} XP</p>
              </div>
            </div>
            <div className="flex gap-1">
              {streaks.slice(0, 3).map(s => (
                <div 
                  key={s.id} 
                  className={`w-8 h-8 rounded-full flex items-center justify-center ${s.isActiveToday ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}
                >
                  <span className={s.color}>{s.icon}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-zinc-400">
              <span>Progress to Level {level + 1}</span>
              <span>{Math.round(xpProgress)}%</span>
            </div>
            <Progress value={xpProgress} className="h-2" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Achievement Celebration Modal */}
      {newAchievement && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center" onClick={() => setNewAchievement(null)}>
          <div className={`bg-zinc-900 border ${RARITY_COLORS[newAchievement.rarity]} rounded-2xl p-8 text-center max-w-sm mx-4 animate-bounce-in shadow-2xl ${RARITY_GLOW[newAchievement.rarity]}`}>
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center">
              {newAchievement.icon}
            </div>
            <Badge className={RARITY_COLORS[newAchievement.rarity] + ' mb-2'}>
              {newAchievement.rarity.toUpperCase()}
            </Badge>
            <h2 className="text-2xl font-bold text-white mb-2">{newAchievement.name}</h2>
            <p className="text-zinc-400 mb-4">{newAchievement.description}</p>
            <div className="flex items-center justify-center gap-2 text-amber-400">
              <Sparkles className="w-5 h-5" />
              <span className="font-bold">+{newAchievement.xpReward} XP</span>
            </div>
          </div>
        </div>
      )}

      {/* Level Card */}
      <Card className="bg-gradient-to-br from-zinc-900 to-zinc-800 border-zinc-700 overflow-hidden">
        <CardContent className="p-6">
          <div className="flex items-center gap-6">
            <div className="relative">
              <div className="w-20 h-20 rounded-full bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center shadow-lg shadow-amber-500/30">
                <span className="text-3xl font-bold text-white">{level}</span>
              </div>
              {showCelebration && (
                <div className="absolute -top-2 -right-2 animate-bounce">
                  <Sparkles className="w-8 h-8 text-amber-400" />
                </div>
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-1">
                <h2 className="text-2xl font-bold text-white">Level {level}</h2>
                <Badge className="bg-amber-500/20 text-amber-400">
                  {totalXP.toLocaleString()} XP
                </Badge>
              </div>
              <p className="text-zinc-400 text-sm mb-3">
                {xpForNextLevel - totalXP} XP to Level {level + 1}
              </p>
              <Progress value={xpProgress} className="h-3" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Streaks Section */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Flame className="w-5 h-5 text-orange-400" />
          Active Streaks
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {streaks.map((streak) => (
            <Card 
              key={streak.id} 
              className={`bg-zinc-900 border-zinc-800 cursor-pointer transition-all hover:scale-105 ${streak.isActiveToday ? 'ring-2 ring-emerald-500/50' : ''}`}
              onClick={() => !streak.isActiveToday && completeStreak(streak.id)}
            >
              <CardContent className="p-4 text-center">
                <div className={`w-12 h-12 rounded-full mx-auto mb-3 flex items-center justify-center ${streak.isActiveToday ? 'bg-emerald-500/20' : 'bg-zinc-800'}`}>
                  <span className={streak.color}>{streak.icon}</span>
                </div>
                <p className="text-sm font-medium text-white mb-1">{streak.name}</p>
                <div className="flex items-center justify-center gap-1">
                  <Flame className={`w-4 h-4 ${streak.currentStreak > 0 ? 'text-orange-400' : 'text-zinc-600'}`} />
                  <span className="text-2xl font-bold text-white">{streak.currentStreak}</span>
                </div>
                <p className="text-xs text-zinc-500 mt-1">Best: {streak.longestStreak} days</p>
                {streak.isActiveToday ? (
                  <Badge className="mt-2 bg-emerald-500/20 text-emerald-400">Done today!</Badge>
                ) : (
                  <Badge variant="outline" className="mt-2 text-zinc-400">Tap to complete</Badge>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Achievements Section */}
      <div>
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Trophy className="w-5 h-5 text-amber-400" />
          Achievements ({unlockedAchievements.length}/{achievements.length})
        </h3>
        
        {/* Unlocked Achievements */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
          {unlockedAchievements.map((achievement) => (
            <Card 
              key={achievement.id} 
              className={`bg-zinc-900 border ${RARITY_COLORS[achievement.rarity]} shadow-lg ${RARITY_GLOW[achievement.rarity]}`}
            >
              <CardContent className="p-4 text-center">
                <div className={`w-14 h-14 rounded-full mx-auto mb-3 flex items-center justify-center bg-gradient-to-br ${
                  achievement.rarity === 'legendary' ? 'from-amber-500 to-orange-600' :
                  achievement.rarity === 'epic' ? 'from-purple-500 to-pink-600' :
                  achievement.rarity === 'rare' ? 'from-blue-500 to-cyan-600' :
                  'from-zinc-500 to-zinc-600'
                }`}>
                  {achievement.icon}
                </div>
                <Badge className={RARITY_COLORS[achievement.rarity] + ' mb-2'}>
                  {achievement.rarity}
                </Badge>
                <p className="text-sm font-medium text-white">{achievement.name}</p>
                <p className="text-xs text-zinc-400 mt-1">{achievement.description}</p>
                <p className="text-xs text-amber-400 mt-2">+{achievement.xpReward} XP</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Locked Achievements */}
        <h4 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
          <Lock className="w-4 h-4" />
          Locked Achievements
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {lockedAchievements.map((achievement) => (
            <Card key={achievement.id} className="bg-zinc-900/50 border-zinc-800 opacity-60">
              <CardContent className="p-4 text-center">
                <div className="w-14 h-14 rounded-full mx-auto mb-3 flex items-center justify-center bg-zinc-800">
                  <Lock className="w-6 h-6 text-zinc-600" />
                </div>
                <Badge variant="outline" className="mb-2 text-zinc-500">
                  {achievement.rarity}
                </Badge>
                <p className="text-sm font-medium text-zinc-400">{achievement.name}</p>
                <p className="text-xs text-zinc-500 mt-1">{achievement.requirement}</p>
                <div className="mt-3">
                  <Progress value={(achievement.progress / achievement.maxProgress) * 100} className="h-1.5" />
                  <p className="text-xs text-zinc-500 mt-1">
                    {achievement.progress}/{achievement.maxProgress}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
