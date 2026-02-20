import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Play, Pause, Heart, Clock, Search, Filter, Volume2, VolumeX, SkipBack, SkipForward, Headphones, Leaf, Moon, Sun, Wind, Waves, Brain, Sparkles } from 'lucide-react';

interface MeditationSession {
  id: string;
  title: string;
  description: string;
  duration: number; // in minutes
  category: 'anxiety' | 'depression' | 'sleep' | 'stress' | 'grounding' | 'self-compassion' | 'trauma' | 'general';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  instructor: string;
  audioUrl: string;
  imageUrl: string;
  isFavorite: boolean;
  playCount: number;
  tags: string[];
}

const MEDITATION_LIBRARY: MeditationSession[] = [
  {
    id: '1',
    title: 'Calm Breathing for Anxiety',
    description: 'A gentle breathing exercise to calm your nervous system and reduce anxiety symptoms.',
    duration: 5,
    category: 'anxiety',
    difficulty: 'beginner',
    instructor: 'Dr. Sarah Chen',
    audioUrl: '/meditations/calm-breathing.mp3',
    imageUrl: '/meditations/calm-breathing.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['breathing', 'quick', 'panic relief'],
  },
  {
    id: '2',
    title: 'Body Scan for Grounding',
    description: 'Progressive body scan to reconnect with your physical self and ground in the present moment.',
    duration: 15,
    category: 'grounding',
    difficulty: 'beginner',
    instructor: 'Dr. Michael Torres',
    audioUrl: '/meditations/body-scan.mp3',
    imageUrl: '/meditations/body-scan.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['grounding', 'dissociation', 'body awareness'],
  },
  {
    id: '3',
    title: 'Sleep Preparation Journey',
    description: 'A calming visualization to prepare your mind and body for restful sleep.',
    duration: 20,
    category: 'sleep',
    difficulty: 'beginner',
    instructor: 'Emma Williams',
    audioUrl: '/meditations/sleep-journey.mp3',
    imageUrl: '/meditations/sleep-journey.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['sleep', 'insomnia', 'relaxation'],
  },
  {
    id: '4',
    title: 'Self-Compassion Practice',
    description: 'Cultivate kindness toward yourself with this loving-kindness meditation.',
    duration: 12,
    category: 'self-compassion',
    difficulty: 'intermediate',
    instructor: 'Dr. Sarah Chen',
    audioUrl: '/meditations/self-compassion.mp3',
    imageUrl: '/meditations/self-compassion.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['self-love', 'inner critic', 'healing'],
  },
  {
    id: '5',
    title: 'Stress Release Visualization',
    description: 'Release tension and stress through guided imagery and progressive relaxation.',
    duration: 10,
    category: 'stress',
    difficulty: 'beginner',
    instructor: 'Dr. Michael Torres',
    audioUrl: '/meditations/stress-release.mp3',
    imageUrl: '/meditations/stress-release.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['stress', 'tension', 'work'],
  },
  {
    id: '6',
    title: 'Safe Place Visualization',
    description: 'Create and visit your inner safe place for comfort during difficult moments.',
    duration: 15,
    category: 'trauma',
    difficulty: 'intermediate',
    instructor: 'Dr. Lisa Park',
    audioUrl: '/meditations/safe-place.mp3',
    imageUrl: '/meditations/safe-place.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['trauma', 'safety', 'PTSD'],
  },
  {
    id: '7',
    title: 'Morning Intention Setting',
    description: 'Start your day with clarity and positive intentions through this morning practice.',
    duration: 8,
    category: 'general',
    difficulty: 'beginner',
    instructor: 'Emma Williams',
    audioUrl: '/meditations/morning-intention.mp3',
    imageUrl: '/meditations/morning-intention.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['morning', 'routine', 'mindfulness'],
  },
  {
    id: '8',
    title: 'Depression Lift Practice',
    description: 'Gentle movement and visualization to help lift low mood and increase energy.',
    duration: 18,
    category: 'depression',
    difficulty: 'intermediate',
    instructor: 'Dr. Sarah Chen',
    audioUrl: '/meditations/depression-lift.mp3',
    imageUrl: '/meditations/depression-lift.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['depression', 'energy', 'motivation'],
  },
  {
    id: '9',
    title: '5-4-3-2-1 Grounding',
    description: 'Quick sensory grounding technique for moments of dissociation or panic.',
    duration: 3,
    category: 'grounding',
    difficulty: 'beginner',
    instructor: 'Dr. Michael Torres',
    audioUrl: '/meditations/54321-grounding.mp3',
    imageUrl: '/meditations/54321-grounding.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['grounding', 'quick', 'emergency'],
  },
  {
    id: '10',
    title: 'Deep Sleep Meditation',
    description: 'Extended practice for deep, restorative sleep with delta wave sounds.',
    duration: 45,
    category: 'sleep',
    difficulty: 'beginner',
    instructor: 'Emma Williams',
    audioUrl: '/meditations/deep-sleep.mp3',
    imageUrl: '/meditations/deep-sleep.jpg',
    isFavorite: false,
    playCount: 0,
    tags: ['sleep', 'deep', 'overnight'],
  },
];

const CATEGORY_ICONS: Record<string, typeof Leaf> = {
  anxiety: Wind,
  depression: Sun,
  sleep: Moon,
  stress: Waves,
  grounding: Leaf,
  'self-compassion': Heart,
  trauma: Brain,
  general: Sparkles,
};

const CATEGORY_COLORS: Record<string, string> = {
  anxiety: 'text-amber-400 bg-amber-500/20',
  depression: 'text-yellow-400 bg-yellow-500/20',
  sleep: 'text-indigo-400 bg-indigo-500/20',
  stress: 'text-blue-400 bg-blue-500/20',
  grounding: 'text-green-400 bg-green-500/20',
  'self-compassion': 'text-pink-400 bg-pink-500/20',
  trauma: 'text-purple-400 bg-purple-500/20',
  general: 'text-cyan-400 bg-cyan-500/20',
};

export function GuidedMeditationLibrary() {
  const [sessions, setSessions] = useState<MeditationSession[]>(() => {
    const saved = localStorage.getItem('reunity-meditation-sessions');
    if (saved) {
      const parsed = JSON.parse(saved);
      return MEDITATION_LIBRARY.map(lib => ({
        ...lib,
        isFavorite: parsed.find((p: MeditationSession) => p.id === lib.id)?.isFavorite || false,
        playCount: parsed.find((p: MeditationSession) => p.id === lib.id)?.playCount || 0,
      }));
    }
    return MEDITATION_LIBRARY;
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedDuration, setSelectedDuration] = useState<string>('all');
  const [currentSession, setCurrentSession] = useState<MeditationSession | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    localStorage.setItem('reunity-meditation-sessions', JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    return () => {
      if (progressInterval.current) {
        clearInterval(progressInterval.current);
      }
    };
  }, []);

  const filteredSessions = sessions.filter(session => {
    const matchesSearch = session.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         session.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         session.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || session.category === selectedCategory;
    const matchesDuration = selectedDuration === 'all' ||
                          (selectedDuration === 'short' && session.duration <= 5) ||
                          (selectedDuration === 'medium' && session.duration > 5 && session.duration <= 15) ||
                          (selectedDuration === 'long' && session.duration > 15);
    const matchesFavorites = !showFavoritesOnly || session.isFavorite;
    return matchesSearch && matchesCategory && matchesDuration && matchesFavorites;
  });

  const playSession = (session: MeditationSession) => {
    setCurrentSession(session);
    setIsPlaying(true);
    setProgress(0);

    // Update play count
    setSessions(sessions.map(s =>
      s.id === session.id ? { ...s, playCount: s.playCount + 1 } : s
    ));

    // Simulate audio playback with progress
    if (progressInterval.current) {
      clearInterval(progressInterval.current);
    }
    progressInterval.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          if (progressInterval.current) clearInterval(progressInterval.current);
          setIsPlaying(false);
          return 100;
        }
        return prev + (100 / (session.duration * 60)); // Update every second
      });
    }, 1000);
  };

  const togglePlayPause = () => {
    if (isPlaying) {
      if (progressInterval.current) clearInterval(progressInterval.current);
    } else if (currentSession) {
      progressInterval.current = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            if (progressInterval.current) clearInterval(progressInterval.current);
            setIsPlaying(false);
            return 100;
          }
          return prev + (100 / (currentSession.duration * 60));
        });
      }, 1000);
    }
    setIsPlaying(!isPlaying);
  };

  const toggleFavorite = (sessionId: string) => {
    setSessions(sessions.map(s =>
      s.id === sessionId ? { ...s, isFavorite: !s.isFavorite } : s
    ));
  };

  const formatTime = (minutes: number) => {
    if (minutes < 60) return `${minutes} min`;
    const hrs = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hrs}h ${mins}m`;
  };

  const categories = ['all', 'anxiety', 'depression', 'sleep', 'stress', 'grounding', 'self-compassion', 'trauma', 'general'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-teal-500/20 rounded-lg">
            <Headphones className="w-6 h-6 text-teal-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white">Guided Meditation Library</h2>
            <p className="text-sm text-zinc-400">{sessions.length} sessions for your wellness journey</p>
          </div>
        </div>
        <Button
          variant={showFavoritesOnly ? 'default' : 'outline'}
          onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
          className={showFavoritesOnly ? 'bg-pink-600 hover:bg-pink-700' : 'border-zinc-700'}
        >
          <Heart className={`w-4 h-4 mr-2 ${showFavoritesOnly ? 'fill-current' : ''}`} />
          Favorites
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col md:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <Input
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search meditations..."
            className="pl-10 bg-zinc-800 border-zinc-700"
          />
        </div>
        <div className="flex gap-2">
          <select
            value={selectedCategory}
            onChange={e => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-md text-zinc-300 text-sm"
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>
                {cat === 'all' ? 'All Categories' : cat.charAt(0).toUpperCase() + cat.slice(1).replace('-', ' ')}
              </option>
            ))}
          </select>
          <select
            value={selectedDuration}
            onChange={e => setSelectedDuration(e.target.value)}
            className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-md text-zinc-300 text-sm"
          >
            <option value="all">Any Duration</option>
            <option value="short">Quick (≤5 min)</option>
            <option value="medium">Medium (5-15 min)</option>
            <option value="long">Long (15+ min)</option>
          </select>
        </div>
      </div>

      {/* Now Playing */}
      {currentSession && (
        <Card className="bg-gradient-to-r from-teal-900/30 to-cyan-900/30 border-teal-800/50">
          <CardContent className="p-4">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-lg ${CATEGORY_COLORS[currentSession.category]}`}>
                {(() => {
                  const Icon = CATEGORY_ICONS[currentSession.category];
                  return <Icon className="w-6 h-6" />;
                })()}
              </div>
              <div className="flex-1">
                <h3 className="text-white font-medium">{currentSession.title}</h3>
                <p className="text-sm text-zinc-400">{currentSession.instructor}</p>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" className="text-zinc-400 hover:text-white">
                  <SkipBack className="w-5 h-5" />
                </Button>
                <Button
                  onClick={togglePlayPause}
                  className="w-12 h-12 rounded-full bg-teal-600 hover:bg-teal-700"
                >
                  {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-1" />}
                </Button>
                <Button variant="ghost" size="icon" className="text-zinc-400 hover:text-white">
                  <SkipForward className="w-5 h-5" />
                </Button>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setIsMuted(!isMuted)}
                  className="text-zinc-400 hover:text-white"
                >
                  {isMuted ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                </Button>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={isMuted ? 0 : volume}
                  onChange={e => {
                    setVolume(parseFloat(e.target.value));
                    setIsMuted(false);
                  }}
                  className="w-20 accent-teal-500"
                />
              </div>
            </div>
            <div className="mt-4">
              <div className="h-1 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-teal-500 transition-all duration-1000"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between mt-1 text-xs text-zinc-500">
                <span>{Math.floor((progress / 100) * currentSession.duration)} min</span>
                <span>{currentSession.duration} min</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Session Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredSessions.length === 0 ? (
          <div className="col-span-full text-center py-12">
            <Headphones className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
            <p className="text-zinc-400">No meditations found</p>
            <p className="text-sm text-zinc-500 mt-1">Try adjusting your filters</p>
          </div>
        ) : (
          filteredSessions.map(session => {
            const Icon = CATEGORY_ICONS[session.category];
            const isCurrentlyPlaying = currentSession?.id === session.id && isPlaying;
            return (
              <Card
                key={session.id}
                className={`bg-zinc-900/50 border-zinc-800 hover:border-zinc-700 transition-all cursor-pointer ${
                  isCurrentlyPlaying ? 'ring-2 ring-teal-500' : ''
                }`}
                onClick={() => playSession(session)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className={`p-2 rounded-lg ${CATEGORY_COLORS[session.category]}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={e => {
                        e.stopPropagation();
                        toggleFavorite(session.id);
                      }}
                      className={session.isFavorite ? 'text-pink-400' : 'text-zinc-500'}
                    >
                      <Heart className={`w-5 h-5 ${session.isFavorite ? 'fill-current' : ''}`} />
                    </Button>
                  </div>
                  <h3 className="text-white font-medium mb-1">{session.title}</h3>
                  <p className="text-sm text-zinc-400 line-clamp-2 mb-3">{session.description}</p>
                  <div className="flex items-center justify-between text-xs text-zinc-500">
                    <div className="flex items-center gap-3">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatTime(session.duration)}
                      </span>
                      <span className="capitalize">{session.difficulty}</span>
                    </div>
                    {isCurrentlyPlaying && (
                      <div className="flex items-center gap-1 text-teal-400">
                        <div className="w-2 h-2 bg-teal-400 rounded-full animate-pulse" />
                        Playing
                      </div>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1 mt-3">
                    {session.tags.slice(0, 3).map(tag => (
                      <span key={tag} className="px-2 py-0.5 bg-zinc-800 text-zinc-400 text-xs rounded">
                        {tag}
                      </span>
                    ))}
                  </div>
                </CardContent>
              </Card>
            );
          })
        )}
      </div>

      {/* Recently Played */}
      {sessions.some(s => s.playCount > 0) && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-white text-base">Recently Played</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-3 overflow-x-auto pb-2">
              {sessions
                .filter(s => s.playCount > 0)
                .sort((a, b) => b.playCount - a.playCount)
                .slice(0, 5)
                .map(session => {
                  const Icon = CATEGORY_ICONS[session.category];
                  return (
                    <button
                      key={session.id}
                      onClick={() => playSession(session)}
                      className="flex-shrink-0 flex items-center gap-3 p-2 bg-zinc-800/50 rounded-lg hover:bg-zinc-800 transition-colors"
                    >
                      <div className={`p-2 rounded ${CATEGORY_COLORS[session.category]}`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <div className="text-left">
                        <p className="text-sm text-white">{session.title}</p>
                        <p className="text-xs text-zinc-500">{session.playCount} plays</p>
                      </div>
                    </button>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default GuidedMeditationLibrary;
