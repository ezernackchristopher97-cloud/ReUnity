import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  BookOpen, 
  Plus, 
  Calendar, 
  TrendingUp, 
  TrendingDown,
  Minus,
  Smile,
  Frown,
  Meh,
  Heart,
  Cloud,
  Sun,
  CloudRain,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Save,
  Trash2,
  Edit2,
  X,
  AlertTriangle,
  Lock,
  Eye,
  EyeOff
} from 'lucide-react';

interface JournalEntry {
  id: string;
  date: string;
  content: string;
  sentiment: SentimentResult;
  mood?: 'great' | 'good' | 'okay' | 'bad' | 'terrible';
  tags: string[];
  isPrivate: boolean;
  createdAt: number;
  updatedAt: number;
}

interface SentimentResult {
  score: number; // -1 to 1
  magnitude: number; // 0 to 1
  label: 'positive' | 'negative' | 'neutral' | 'mixed';
  keywords: string[];
  concerns: string[];
}

interface JournalWithSentimentProps {
  onSentimentUpdate?: (entries: JournalEntry[]) => void;
  onCrisisDetected?: (entry: JournalEntry) => void;
}

// Crisis keywords to watch for
const CRISIS_KEYWORDS = [
  'suicide', 'kill myself', 'end it all', 'want to die', 'no point',
  'self harm', 'hurt myself', 'cutting', 'overdose', 'hopeless',
  'give up', 'cant go on', 'worthless', 'burden', 'better off without me'
];

// Positive keywords
const POSITIVE_KEYWORDS = [
  'happy', 'grateful', 'thankful', 'joy', 'excited', 'hopeful',
  'peaceful', 'calm', 'loved', 'supported', 'proud', 'accomplished',
  'better', 'improving', 'progress', 'breakthrough', 'healing'
];

// Negative keywords
const NEGATIVE_KEYWORDS = [
  'sad', 'anxious', 'worried', 'stressed', 'overwhelmed', 'tired',
  'exhausted', 'angry', 'frustrated', 'lonely', 'isolated', 'scared',
  'panic', 'depressed', 'numb', 'empty', 'lost'
];

// Simple sentiment analysis function
function analyzeSentiment(text: string): SentimentResult {
  const lowerText = text.toLowerCase();
  const words = lowerText.split(/\s+/);
  
  let positiveCount = 0;
  let negativeCount = 0;
  const foundKeywords: string[] = [];
  const concerns: string[] = [];
  
  // Check for crisis keywords first
  for (const keyword of CRISIS_KEYWORDS) {
    if (lowerText.includes(keyword)) {
      concerns.push(keyword);
    }
  }
  
  // Count positive and negative words
  for (const word of words) {
    if (POSITIVE_KEYWORDS.some(kw => word.includes(kw))) {
      positiveCount++;
      foundKeywords.push(word);
    }
    if (NEGATIVE_KEYWORDS.some(kw => word.includes(kw))) {
      negativeCount++;
      foundKeywords.push(word);
    }
  }
  
  // Calculate score
  const total = positiveCount + negativeCount;
  let score = 0;
  let label: SentimentResult['label'] = 'neutral';
  
  if (total > 0) {
    score = (positiveCount - negativeCount) / total;
    
    if (concerns.length > 0) {
      score = Math.min(score, -0.5);
      label = 'negative';
    } else if (score > 0.3) {
      label = 'positive';
    } else if (score < -0.3) {
      label = 'negative';
    } else if (positiveCount > 0 && negativeCount > 0) {
      label = 'mixed';
    }
  }
  
  // Magnitude based on word count and emotional intensity
  const magnitude = Math.min(1, (positiveCount + negativeCount) / 10);
  
  return {
    score,
    magnitude,
    label,
    keywords: Array.from(new Set(foundKeywords)).slice(0, 5),
    concerns,
  };
}

// Mood suggestions based on entry content
const MOOD_OPTIONS = [
  { value: 'great', icon: Sun, label: 'Great', color: 'text-amber-400' },
  { value: 'good', icon: Smile, label: 'Good', color: 'text-emerald-400' },
  { value: 'okay', icon: Meh, label: 'Okay', color: 'text-blue-400' },
  { value: 'bad', icon: Cloud, label: 'Bad', color: 'text-zinc-400' },
  { value: 'terrible', icon: CloudRain, label: 'Terrible', color: 'text-red-400' },
] as const;

// Journal prompts
const PROMPTS = [
  "What are you grateful for today?",
  "How are you feeling right now?",
  "What's been on your mind lately?",
  "Describe a moment that made you smile today.",
  "What challenges did you face today?",
  "What would make tomorrow better?",
  "What's something you're looking forward to?",
  "How did you take care of yourself today?",
];

export default function JournalWithSentiment({ onSentimentUpdate, onCrisisDetected }: JournalWithSentimentProps) {
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [isWriting, setIsWriting] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [content, setContent] = useState('');
  const [selectedMood, setSelectedMood] = useState<JournalEntry['mood']>();
  const [isPrivate, setIsPrivate] = useState(true);
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [viewDate, setViewDate] = useState(new Date());
  const [showCrisisAlert, setShowCrisisAlert] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState(PROMPTS[0]);

  // Load entries from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('reunity_journal_entries');
    if (stored) {
      setEntries(JSON.parse(stored));
    }
  }, []);

  // Rotate prompt
  useEffect(() => {
    const idx = Math.floor(Math.random() * PROMPTS.length);
    setCurrentPrompt(PROMPTS[idx]);
  }, [isWriting]);

  // Live sentiment analysis
  const liveSentiment = useMemo(() => {
    if (content.length < 10) return null;
    return analyzeSentiment(content);
  }, [content]);

  // Check for crisis keywords in real-time
  useEffect(() => {
    if (liveSentiment?.concerns && liveSentiment.concerns.length > 0) {
      setShowCrisisAlert(true);
    }
  }, [liveSentiment]);

  // Calculate sentiment trends
  const sentimentTrend = useMemo(() => {
    if (entries.length < 2) return 'stable';
    
    const recent = entries.slice(0, 7);
    const older = entries.slice(7, 14);
    
    if (older.length === 0) return 'stable';
    
    const recentAvg = recent.reduce((sum, e) => sum + e.sentiment.score, 0) / recent.length;
    const olderAvg = older.reduce((sum, e) => sum + e.sentiment.score, 0) / older.length;
    
    if (recentAvg - olderAvg > 0.2) return 'improving';
    if (recentAvg - olderAvg < -0.2) return 'declining';
    return 'stable';
  }, [entries]);

  const saveEntry = () => {
    if (!content.trim()) return;
    
    const sentiment = analyzeSentiment(content);
    const now = Date.now();
    const today = new Date().toISOString().split('T')[0];
    
    const newEntry: JournalEntry = {
      id: editingId || now.toString(),
      date: today,
      content,
      sentiment,
      mood: selectedMood,
      tags,
      isPrivate,
      createdAt: editingId ? entries.find(e => e.id === editingId)?.createdAt || now : now,
      updatedAt: now,
    };
    
    let updatedEntries: JournalEntry[];
    
    if (editingId) {
      updatedEntries = entries.map(e => e.id === editingId ? newEntry : e);
    } else {
      updatedEntries = [newEntry, ...entries];
    }
    
    setEntries(updatedEntries);
    localStorage.setItem('reunity_journal_entries', JSON.stringify(updatedEntries));
    
    // Notify parent of sentiment update
    onSentimentUpdate?.(updatedEntries);
    
    // Check for crisis
    if (sentiment.concerns.length > 0) {
      onCrisisDetected?.(newEntry);
    }
    
    resetForm();
  };

  const deleteEntry = (id: string) => {
    const updatedEntries = entries.filter(e => e.id !== id);
    setEntries(updatedEntries);
    localStorage.setItem('reunity_journal_entries', JSON.stringify(updatedEntries));
    onSentimentUpdate?.(updatedEntries);
  };

  const startEdit = (entry: JournalEntry) => {
    setEditingId(entry.id);
    setContent(entry.content);
    setSelectedMood(entry.mood);
    setTags(entry.tags);
    setIsPrivate(entry.isPrivate);
    setIsWriting(true);
  };

  const resetForm = () => {
    setContent('');
    setSelectedMood(undefined);
    setTags([]);
    setTagInput('');
    setIsPrivate(true);
    setIsWriting(false);
    setEditingId(null);
    setShowCrisisAlert(false);
  };

  const addTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags([...tags, tagInput.trim()]);
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setTags(tags.filter(t => t !== tag));
  };

  const navigateDate = (direction: 'prev' | 'next') => {
    const newDate = new Date(viewDate);
    newDate.setDate(viewDate.getDate() + (direction === 'next' ? 1 : -1));
    setViewDate(newDate);
  };

  const entriesForDate = entries.filter(e => e.date === viewDate.toISOString().split('T')[0]);
  const todayStr = new Date().toISOString().split('T')[0];
  const hasEntryToday = entries.some(e => e.date === todayStr);

  // Calculate weekly stats
  const weeklyStats = useMemo(() => {
    const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
    const weekEntries = entries.filter(e => e.createdAt > weekAgo);
    
    const avgSentiment = weekEntries.length > 0
      ? weekEntries.reduce((sum, e) => sum + e.sentiment.score, 0) / weekEntries.length
      : 0;
    
    const moodCounts = weekEntries.reduce((acc, e) => {
      if (e.mood) acc[e.mood] = (acc[e.mood] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return { avgSentiment, moodCounts, entryCount: weekEntries.length };
  }, [entries]);

  return (
    <div className="space-y-6">
      {/* Crisis Alert */}
      {showCrisisAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-red-500/50 rounded-xl p-6 max-w-md mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">We're Here For You</h3>
                <p className="text-sm text-zinc-400">It sounds like you're going through a difficult time</p>
              </div>
            </div>
            
            <p className="text-zinc-300 mb-4">
              If you're having thoughts of self-harm or suicide, please reach out for support. You don't have to face this alone.
            </p>
            
            <div className="space-y-2">
              <Button
                className="w-full bg-red-600 hover:bg-red-700"
                onClick={() => window.location.href = 'tel:988'}
              >
                Call 988 Crisis Lifeline
              </Button>
              <Button
                variant="outline"
                className="w-full border-zinc-700"
                onClick={() => window.location.href = 'sms:741741?body=HELLO'}
              >
                Text HOME to 741741
              </Button>
              <Button
                variant="ghost"
                className="w-full text-zinc-500"
                onClick={() => setShowCrisisAlert(false)}
              >
                Continue Writing
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Weekly Summary */}
      <Card className="bg-gradient-to-br from-emerald-900/30 to-teal-900/30 border-emerald-700/30">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-emerald-400" />
              <CardTitle className="text-lg text-emerald-300">This Week's Journey</CardTitle>
            </div>
            <div className="flex items-center gap-1">
              {sentimentTrend === 'improving' && <TrendingUp className="w-4 h-4 text-emerald-400" />}
              {sentimentTrend === 'declining' && <TrendingDown className="w-4 h-4 text-red-400" />}
              {sentimentTrend === 'stable' && <Minus className="w-4 h-4 text-zinc-400" />}
              <span className="text-sm text-zinc-400 capitalize">{sentimentTrend}</span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-white">{weeklyStats.entryCount}</p>
              <p className="text-xs text-zinc-400">Entries</p>
            </div>
            <div className="text-center">
              <p className={`text-2xl font-bold ${
                weeklyStats.avgSentiment > 0.2 ? 'text-emerald-400' :
                weeklyStats.avgSentiment < -0.2 ? 'text-red-400' : 'text-zinc-300'
              }`}>
                {weeklyStats.avgSentiment > 0 ? '+' : ''}{(weeklyStats.avgSentiment * 100).toFixed(0)}%
              </p>
              <p className="text-xs text-zinc-400">Avg Mood</p>
            </div>
            <div className="text-center">
              <div className="flex justify-center gap-1">
                {Object.entries(weeklyStats.moodCounts).slice(0, 3).map(([mood, count]) => {
                  const option = MOOD_OPTIONS.find(o => o.value === mood);
                  if (!option) return null;
                  const Icon = option.icon;
                  return <Icon key={mood} className={`w-5 h-5 ${option.color}`} />;
                })}
              </div>
              <p className="text-xs text-zinc-400">Top Moods</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* New Entry Button or Writing Area */}
      {!isWriting ? (
        <Card className="bg-zinc-900/80 border-zinc-800">
          <CardContent className="py-6">
            <div className="text-center">
              {!hasEntryToday ? (
                <>
                  <BookOpen className="w-12 h-12 mx-auto mb-3 text-emerald-400 opacity-50" />
                  <p className="text-zinc-400 mb-4">You haven't journaled today</p>
                  <Button
                    onClick={() => setIsWriting(true)}
                    className="bg-emerald-600 hover:bg-emerald-700"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Write Today's Entry
                  </Button>
                </>
              ) : (
                <>
                  <Heart className="w-12 h-12 mx-auto mb-3 text-emerald-400" />
                  <p className="text-zinc-300 mb-4">Great job journaling today!</p>
                  <Button
                    onClick={() => setIsWriting(true)}
                    variant="outline"
                    className="border-emerald-600 text-emerald-400"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add Another Entry
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="bg-zinc-900/80 border-zinc-800">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-emerald-400" />
                <CardTitle className="text-lg text-white">
                  {editingId ? 'Edit Entry' : 'New Journal Entry'}
                </CardTitle>
              </div>
              <Button variant="ghost" size="sm" onClick={resetForm} className="text-zinc-400">
                <X className="w-4 h-4" />
              </Button>
            </div>
            <CardDescription className="text-zinc-400 italic">
              "{currentPrompt}"
            </CardDescription>
          </CardHeader>
          
          <CardContent className="space-y-4">
            {/* Mood Selection */}
            <div>
              <label className="text-sm text-zinc-400 mb-2 block">How are you feeling?</label>
              <div className="flex gap-2">
                {MOOD_OPTIONS.map(option => {
                  const Icon = option.icon;
                  return (
                    <Button
                      key={option.value}
                      variant={selectedMood === option.value ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setSelectedMood(option.value)}
                      className={selectedMood === option.value ? 'bg-zinc-700' : 'border-zinc-700'}
                    >
                      <Icon className={`w-4 h-4 mr-1 ${option.color}`} />
                      {option.label}
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Content */}
            <div>
              <textarea
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Write your thoughts..."
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg p-4 text-white placeholder:text-zinc-500 resize-none h-40 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
              />
              
              {/* Live Sentiment Indicator */}
              {liveSentiment && (
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs text-zinc-500">Sentiment:</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    liveSentiment.label === 'positive' ? 'bg-emerald-500/20 text-emerald-400' :
                    liveSentiment.label === 'negative' ? 'bg-red-500/20 text-red-400' :
                    liveSentiment.label === 'mixed' ? 'bg-amber-500/20 text-amber-400' :
                    'bg-zinc-700 text-zinc-400'
                  }`}>
                    {liveSentiment.label}
                  </span>
                  {liveSentiment.keywords.length > 0 && (
                    <span className="text-xs text-zinc-500">
                      Keywords: {liveSentiment.keywords.join(', ')}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Tags */}
            <div>
              <label className="text-sm text-zinc-400 mb-2 block">Tags</label>
              <div className="flex flex-wrap gap-2 mb-2">
                {tags.map(tag => (
                  <span
                    key={tag}
                    className="text-xs px-2 py-1 rounded-full bg-zinc-700 text-zinc-300 flex items-center gap-1"
                  >
                    {tag}
                    <button onClick={() => removeTag(tag)} className="hover:text-red-400">
                      <X className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex gap-2">
                <input
                  value={tagInput}
                  onChange={(e) => setTagInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                  placeholder="Add a tag..."
                  className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1 text-sm text-white"
                />
                <Button size="sm" variant="outline" onClick={addTag} className="border-zinc-700">
                  Add
                </Button>
              </div>
            </div>

            {/* Privacy Toggle */}
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-sm text-zinc-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={isPrivate}
                  onChange={(e) => setIsPrivate(e.target.checked)}
                  className="rounded border-zinc-600"
                />
                <Lock className="w-4 h-4" />
                Keep this entry private
              </label>
            </div>

            {/* Save Button */}
            <Button onClick={saveEntry} className="w-full bg-emerald-600 hover:bg-emerald-700">
              <Save className="w-4 h-4 mr-2" />
              {editingId ? 'Update Entry' : 'Save Entry'}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Past Entries */}
      <Card className="bg-zinc-900/80 border-zinc-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-emerald-400" />
              <CardTitle className="text-lg text-white">Journal History</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => navigateDate('prev')} className="text-zinc-400">
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <span className="text-sm text-zinc-300">
                {viewDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
              </span>
              <Button variant="ghost" size="sm" onClick={() => navigateDate('next')} className="text-zinc-400">
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        
        <CardContent>
          {entriesForDate.length === 0 ? (
            <div className="text-center py-8 text-zinc-500">
              <BookOpen className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No entries for this date</p>
            </div>
          ) : (
            <div className="space-y-4">
              {entriesForDate.map(entry => {
                const moodOption = MOOD_OPTIONS.find(o => o.value === entry.mood);
                const MoodIcon = moodOption?.icon || Meh;
                
                return (
                  <div key={entry.id} className="bg-zinc-800/50 rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <MoodIcon className={`w-5 h-5 ${moodOption?.color || 'text-zinc-400'}`} />
                        <span className="text-sm text-zinc-400">
                          {new Date(entry.createdAt).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}
                        </span>
                        {entry.isPrivate && <Lock className="w-3 h-3 text-zinc-500" />}
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          entry.sentiment.label === 'positive' ? 'bg-emerald-500/20 text-emerald-400' :
                          entry.sentiment.label === 'negative' ? 'bg-red-500/20 text-red-400' :
                          entry.sentiment.label === 'mixed' ? 'bg-amber-500/20 text-amber-400' :
                          'bg-zinc-700 text-zinc-400'
                        }`}>
                          {entry.sentiment.label}
                        </span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => startEdit(entry)}
                          className="text-zinc-400 hover:text-white h-8 w-8"
                        >
                          <Edit2 className="w-4 h-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => deleteEntry(entry.id)}
                          className="text-red-400 hover:bg-red-500/20 h-8 w-8"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    
                    <p className="text-zinc-300 whitespace-pre-wrap">{entry.content}</p>
                    
                    {entry.tags.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-3">
                        {entry.tags.map(tag => (
                          <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-zinc-700 text-zinc-400">
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
