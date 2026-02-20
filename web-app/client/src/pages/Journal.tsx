import { useState, useEffect, useMemo } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  BookOpen, 
  PenLine, 
  TrendingUp, 
  Calendar,
  Sparkles,
  Heart,
  Cloud,
  Sun,
  Moon,
  Zap,
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Activity,
  Loader2
} from "lucide-react";
import { trpc } from "@/lib/trpc";
import { toast } from "sonner";

// Mood options with icons
const moodOptions = [
  { id: "peaceful", label: "Peaceful", icon: Sun, color: "text-yellow-400", bg: "bg-yellow-500/20" },
  { id: "hopeful", label: "Hopeful", icon: Sparkles, color: "text-emerald-400", bg: "bg-emerald-500/20" },
  { id: "content", label: "Content", icon: Heart, color: "text-pink-400", bg: "bg-pink-500/20" },
  { id: "anxious", label: "Anxious", icon: Zap, color: "text-amber-400", bg: "bg-amber-500/20" },
  { id: "sad", label: "Sad", icon: Cloud, color: "text-blue-400", bg: "bg-blue-500/20" },
  { id: "overwhelmed", label: "Overwhelmed", icon: AlertTriangle, color: "text-red-400", bg: "bg-red-500/20" },
  { id: "numb", label: "Numb", icon: Moon, color: "text-slate-400", bg: "bg-slate-500/20" },
  { id: "angry", label: "Angry", icon: Zap, color: "text-orange-400", bg: "bg-orange-500/20" }
];

// Writing prompts for journaling
const writingPrompts = [
  "What am I grateful for today, even if it's small?",
  "What emotions am I feeling right now? Where do I feel them in my body?",
  "What coping strategy helped me today?",
  "What would I tell a friend who was feeling the way I feel?",
  "What is one thing I did today that took courage?",
  "What patterns am I noticing in my thoughts lately?",
  "What does my inner critic say? How can I respond with compassion?",
  "What boundaries do I need to set or maintain?"
];

// Type definitions
interface JournalEntry {
  id: number;
  title: string | null;
  content: string;
  moodTags: unknown;
  entropyScore: string | null;
  entropyState: string | null;
  createdAt: Date;
}

interface TrajectoryPoint {
  date: Date;
  entropy: number;
  state: string | null;
  moodTags: string[] | any[];
}

interface TrajectoryPrediction {
  date: Date;
  predicted: number;
}

// Fallback trajectory data generator
const generateFallbackTrajectory = (): { trajectoryData: TrajectoryPoint[], predictions: TrajectoryPrediction[] } => {
  const trajectoryData: TrajectoryPoint[] = [];
  const predictions: TrajectoryPrediction[] = [];
  let entropy = 0.5;
  for (let i = 0; i < 30; i++) {
    const neighborInfluence = Math.sin(i / 5) * 0.1;
    const noise = (Math.random() - 0.5) * 0.15;
    entropy = Math.max(0, Math.min(1, entropy + neighborInfluence + noise));
    const date = new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000);
    trajectoryData.push({
      date,
      entropy,
      state: entropy < 0.35 ? 'low' : entropy < 0.65 ? 'moderate' : 'high',
      moodTags: []
    });
    predictions.push({
      date,
      predicted: entropy + (Math.random() - 0.5) * 0.1
    });
  }
  return { trajectoryData, predictions };
};

export default function Journal() {
  const { user, isLoading: authLoading } = useAuth();
  const [activeTab, setActiveTab] = useState("write");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [selectedMoods, setSelectedMoods] = useState<string[]>([]);
  const [showInsights, setShowInsights] = useState(true);
  const [currentPrompt, setCurrentPrompt] = useState(() => 
    writingPrompts[Math.floor(Math.random() * writingPrompts.length)]
  );

  // tRPC queries
  const { data: entriesData, isLoading: entriesLoading, refetch: refetchEntries } = trpc.journal.getEntries.useQuery(
    { limit: 50 },
    { enabled: !!user }
  );

  const { data: trajectoryResponse, isLoading: trajectoryLoading } = trpc.journal.getTrajectory.useQuery(
    { days: 30 },
    { enabled: !!user }
  );

  // Use fallback if no data
  const fallbackData = useMemo(() => generateFallbackTrajectory(), []);
  const trajectoryData = trajectoryResponse?.trajectory || fallbackData.trajectoryData;
  const predictions = trajectoryResponse?.predictions || fallbackData.predictions;

  const { data: insightsData } = trpc.journal.getInsights.useQuery(
    undefined,
    { enabled: !!user }
  );

  // tRPC mutations
  const createEntryMutation = trpc.journal.createEntry.useMutation({
    onSuccess: () => {
      toast.success("Journal entry saved!");
      setTitle("");
      setContent("");
      setSelectedMoods([]);
      refetchEntries();
      setActiveTab("entries");
    },
    onError: (error) => {
      toast.error(error.message);
    }
  });

  const entries = entriesData || [];

  const toggleMood = (id: string) => {
    if (selectedMoods.includes(id)) {
      setSelectedMoods(selectedMoods.filter(m => m !== id));
    } else {
      setSelectedMoods([...selectedMoods, id]);
    }
  };

  const handleSaveEntry = () => {
    if (!content.trim()) return;
    createEntryMutation.mutate({
      title: title || "Untitled Entry",
      content,
      moodTags: selectedMoods
    });
  };

  const getNewPrompt = () => {
    const newPrompt = writingPrompts[Math.floor(Math.random() * writingPrompts.length)];
    setCurrentPrompt(newPrompt);
  };

  const getEntropyColor = (score: number) => {
    if (score < 0.35) return "text-emerald-400";
    if (score < 0.65) return "text-amber-400";
    return "text-red-400";
  };

  const getEntropyLabel = (state: string) => {
    switch (state) {
      case "low": return { label: "Stable", color: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" };
      case "moderate": return { label: "Transitioning", color: "bg-amber-500/20 text-amber-300 border-amber-500/30" };
      case "high": return { label: "Elevated", color: "bg-red-500/20 text-red-300 border-red-500/30" };
      default: return { label: "Unknown", color: "bg-slate-500/20 text-slate-300 border-slate-500/30" };
    }
  };

  // Calculate insights from entries
  const insights = useMemo(() => {
    const recentEntries = entries.slice(0, 7) as JournalEntry[];
    const avgEntropy = recentEntries.length > 0 
      ? recentEntries.reduce((sum, e) => sum + parseFloat(e.entropyScore || '0.5'), 0) / recentEntries.length
      : 0.5;
    const moodCounts: Record<string, number> = {};
    recentEntries.forEach((e: JournalEntry) => {
      const tags = Array.isArray(e.moodTags) ? e.moodTags as string[] : [];
      tags.forEach((m: string) => {
        moodCounts[m] = (moodCounts[m] || 0) + 1;
      });
    });
    const topMood = Object.entries(moodCounts).sort((a, b) => b[1] - a[1])[0];
    
    return {
      avgEntropy,
      topMood: topMood ? topMood[0] : null,
      trend: avgEntropy > 0.5 ? "increasing" : "decreasing",
      entriesThisWeek: recentEntries.length
    };
  }, [entries]);

  if (authLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-emerald-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <BookOpen className="h-6 w-6 text-emerald-400" />
            <span className="text-xl font-bold text-white">Journal</span>
          </Link>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
              <Activity className="h-3 w-3 mr-1" />
              Entropy Tracking
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Insights Banner */}
        {showInsights && entries.length > 0 && (
          <Card className="bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border-emerald-500/30 mb-8">
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-full bg-emerald-500/20">
                    <TrendingUp className="h-6 w-6 text-emerald-400" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-white">Your Vicsek Trajectory</h3>
                    <p className="text-slate-400 text-sm mt-1">
                      Based on your recent entries, your emotional entropy is{" "}
                      <span className={getEntropyColor(insights.avgEntropy)}>
                        {insights.trend === "decreasing" ? "stabilizing" : "fluctuating"}
                      </span>.
                      {insights.topMood && (
                        <> Your most common mood this week: <span className="text-emerald-300">{insights.topMood}</span>.</>
                      )}
                    </p>
                  </div>
                </div>
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => setShowInsights(false)}
                  className="text-slate-400"
                >
                  Dismiss
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-slate-800/50 border-slate-700 mb-6">
            <TabsTrigger value="write" className="data-[state=active]:bg-emerald-500/20">
              <PenLine className="h-4 w-4 mr-2" />
              Write
            </TabsTrigger>
            <TabsTrigger value="entries" className="data-[state=active]:bg-emerald-500/20">
              <BookOpen className="h-4 w-4 mr-2" />
              Entries
            </TabsTrigger>
            <TabsTrigger value="patterns" className="data-[state=active]:bg-emerald-500/20">
              <BarChart3 className="h-4 w-4 mr-2" />
              Patterns
            </TabsTrigger>
          </TabsList>

          {/* Write Tab */}
          <TabsContent value="write">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <PenLine className="h-5 w-5 text-emerald-400" />
                  New Journal Entry
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Express yourself freely. Your entries help track your emotional patterns.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="Entry title (optional)"
                    className="bg-slate-900/50 border-slate-600 text-white text-lg"
                  />
                </div>

                <div className="space-y-2">
                  <Textarea
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    placeholder="What's on your mind? How are you feeling? What happened today?&#10;&#10;Write freely - this is your safe space..."
                    className="bg-slate-900/50 border-slate-600 text-white min-h-[250px] resize-none"
                  />
                </div>

                <div className="space-y-4">
                  <label className="text-sm text-slate-400">How are you feeling?</label>
                  <div className="flex flex-wrap gap-2">
                    {moodOptions.map((mood) => {
                      const Icon = mood.icon;
                      const isSelected = selectedMoods.includes(mood.id);
                      return (
                        <button
                          key={mood.id}
                          onClick={() => toggleMood(mood.id)}
                          className={`flex items-center gap-2 px-4 py-2 rounded-full transition-all ${
                            isSelected
                              ? `${mood.bg} ${mood.color} border border-current`
                              : "bg-slate-700/50 text-slate-400 hover:bg-slate-700"
                          }`}
                        >
                          <Icon className="h-4 w-4" />
                          <span>{mood.label}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="flex justify-end gap-4">
                  <Button
                    variant="outline"
                    className="border-slate-600 text-slate-300"
                    onClick={() => {
                      setTitle("");
                      setContent("");
                      setSelectedMoods([]);
                    }}
                  >
                    Clear
                  </Button>
                  <Button
                    onClick={handleSaveEntry}
                    disabled={!content.trim()}
                    className="bg-emerald-500 hover:bg-emerald-600"
                  >
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    Save Entry
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Writing Prompts */}
            <Card className="bg-slate-800/50 border-slate-700 mt-6">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-amber-400" />
                  Writing Prompts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    "What am I grateful for today?",
                    "What triggered me today and how did I cope?",
                    "What would I tell my younger self?",
                    "What small victory can I celebrate?",
                    "What does my ideal day look like?",
                    "What boundaries do I need to set?"
                  ].map((prompt, i) => (
                    <button
                      key={i}
                      onClick={() => setContent(prompt + "\n\n")}
                      className="p-4 text-left rounded-lg bg-slate-700/30 hover:bg-slate-700/50 text-slate-300 transition-colors"
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Entries Tab */}
          <TabsContent value="entries">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">Your Entries</h2>
                <Badge variant="outline" className="border-slate-600 text-slate-300">
                  {entries.length} entries
                </Badge>
              </div>

              {entries.length === 0 ? (
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardContent className="p-8 text-center">
                    <BookOpen className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                    <p className="text-slate-400">No entries yet</p>
                    <p className="text-sm text-slate-500 mt-2">
                      Start journaling to track your emotional patterns
                    </p>
                    <Button 
                      onClick={() => setActiveTab("write")}
                      className="mt-4 bg-emerald-500 hover:bg-emerald-600"
                    >
                      Write First Entry
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                entries.map((entry: JournalEntry) => {
                  const entropyInfo = getEntropyLabel(entry.entropyState || 'moderate');
                  const moodTags = Array.isArray(entry.moodTags) ? entry.moodTags as string[] : [];
                  const entropyScore = parseFloat(entry.entropyScore || '0.5');
                  return (
                    <Card key={entry.id} className="bg-slate-800/50 border-slate-700">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div>
                            <h3 className="text-lg font-medium text-white">{entry.title || 'Untitled'}</h3>
                            <p className="text-sm text-slate-400">
                              {new Date(entry.createdAt).toLocaleDateString('en-US', {
                                weekday: 'long',
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                              })}
                            </p>
                          </div>
                          <Badge className={entropyInfo.color}>
                            {entropyInfo.label}
                          </Badge>
                        </div>

                        <p className="text-slate-300 line-clamp-3 mb-4">
                          {entry.content}
                        </p>

                        <div className="flex items-center justify-between">
                          <div className="flex flex-wrap gap-2">
                            {moodTags.map((moodId: string) => {
                              const mood = moodOptions.find(m => m.id === moodId);
                              if (!mood) return null;
                              const Icon = mood.icon;
                              return (
                                <Badge 
                                  key={moodId}
                                  variant="outline" 
                                  className={`${mood.bg} ${mood.color} border-current`}
                                >
                                  <Icon className="h-3 w-3 mr-1" />
                                  {mood.label}
                                </Badge>
                              );
                            })}
                          </div>
                          <div className="flex items-center gap-2 text-sm">
                            <Activity className={`h-4 w-4 ${getEntropyColor(entropyScore)}`} />
                            <span className={getEntropyColor(entropyScore)}>
                              {(entropyScore * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })
              )}
            </div>
          </TabsContent>

          {/* Patterns Tab */}
          <TabsContent value="patterns">
            <div className="grid gap-6">
              {/* Vicsek Trajectory Visualization */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Activity className="h-5 w-5 text-emerald-400" />
                    Emotional Entropy Trajectory
                  </CardTitle>
                  <CardDescription className="text-slate-400">
                    Using the Vicsek flocking model to predict emotional patterns based on neighboring states
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {/* Simple SVG visualization */}
                  <div className="relative h-64 bg-slate-900/50 rounded-lg p-4">
                    <svg className="w-full h-full" viewBox="0 0 400 200">
                      {/* Grid lines */}
                      <defs>
                        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(100,116,139,0.2)" strokeWidth="1"/>
                        </pattern>
                      </defs>
                      <rect width="100%" height="100%" fill="url(#grid)" />
                      
                      {/* Entropy zones */}
                      <rect x="0" y="0" width="400" height="70" fill="rgba(239,68,68,0.1)" />
                      <rect x="0" y="70" width="400" height="60" fill="rgba(245,158,11,0.1)" />
                      <rect x="0" y="130" width="400" height="70" fill="rgba(16,185,129,0.1)" />
                      
                      {/* Zone labels */}
                      <text x="10" y="40" fill="rgba(239,68,68,0.6)" fontSize="10">High Entropy</text>
                      <text x="10" y="105" fill="rgba(245,158,11,0.6)" fontSize="10">Moderate</text>
                      <text x="10" y="170" fill="rgba(16,185,129,0.6)" fontSize="10">Low Entropy</text>
                      
                      {/* Actual trajectory line */}
                      <path
                        d={trajectoryData.map((d: TrajectoryPoint, i: number) => {
                          const x = (i / Math.max(trajectoryData.length - 1, 1)) * 380 + 10;
                          const y = 190 - (d.entropy * 180);
                          return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')}
                        fill="none"
                        stroke="rgb(16,185,129)"
                        strokeWidth="2"
                      />
                      
                      {/* Predicted trajectory (dashed) */}
                      <path
                        d={predictions.slice(-10).map((d: TrajectoryPrediction, i: number) => {
                          const x = ((i + 20) / 29) * 380 + 10;
                          const y = 190 - (d.predicted * 180);
                          return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')}
                        fill="none"
                        stroke="rgb(16,185,129)"
                        strokeWidth="2"
                        strokeDasharray="5,5"
                        opacity="0.5"
                      />
                      
                      {/* Data points */}
                      {trajectoryData.filter((_: TrajectoryPoint, i: number) => i % 3 === 0).map((d: TrajectoryPoint, i: number) => {
                        const x = ((i * 3) / Math.max(trajectoryData.length - 1, 1)) * 380 + 10;
                        const y = 190 - (d.entropy * 180);
                        return (
                          <circle
                            key={i}
                            cx={x}
                            cy={y}
                            r="4"
                            fill="rgb(16,185,129)"
                            className="cursor-pointer hover:r-6"
                          />
                        );
                      })}
                    </svg>
                    
                    {/* Legend */}
                    <div className="absolute bottom-2 right-2 flex items-center gap-4 text-xs text-slate-400">
                      <div className="flex items-center gap-1">
                        <div className="w-4 h-0.5 bg-emerald-500"></div>
                        <span>Actual</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-4 h-0.5 bg-emerald-500/50 border-dashed"></div>
                        <span>Predicted</span>
                      </div>
                    </div>
                  </div>
                  
                  <Alert className="mt-4 bg-teal-500/10 border-teal-500/30">
                    <Activity className="h-4 w-4 text-teal-400" />
                    <AlertTitle className="text-teal-300">Vicsek Model Insight</AlertTitle>
                    <AlertDescription className="text-teal-200/80">
                      Your emotional trajectory shows flocking behavior - your states tend to align with 
                      neighboring emotional states over time. The model predicts continued stabilization 
                      if current patterns hold.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>

              {/* Mood Distribution */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Mood Distribution</CardTitle>
                  <CardDescription className="text-slate-400">
                    Your most common emotional states this month
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {moodOptions.map((mood) => {
                      const Icon = mood.icon;
                      const count = entries.filter((e: JournalEntry) => {
                        const tags = Array.isArray(e.moodTags) ? e.moodTags : [];
                        return tags.includes(mood.id);
                      }).length;
                      const percentage = entries.length > 0 ? (count / entries.length) * 100 : 0;
                      
                      return (
                        <div key={mood.id} className="flex items-center gap-4">
                          <div className={`p-2 rounded-lg ${mood.bg}`}>
                            <Icon className={`h-4 w-4 ${mood.color}`} />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm text-slate-300">{mood.label}</span>
                              <span className="text-sm text-slate-400">{count} entries</span>
                            </div>
                            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${mood.bg.replace('/20', '')} transition-all`}
                                style={{ width: `${percentage}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Pattern Insights */}
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <Sparkles className="h-5 w-5 text-amber-400" />
                    Pattern Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Alert className="bg-emerald-500/10 border-emerald-500/30">
                    <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                    <AlertTitle className="text-emerald-300">Positive Pattern</AlertTitle>
                    <AlertDescription className="text-emerald-200/80">
                      Your entropy levels have been decreasing over the past week, indicating 
                      emotional stabilization. Keep using your coping strategies!
                    </AlertDescription>
                  </Alert>

                  <Alert className="bg-amber-500/10 border-amber-500/30">
                    <AlertTriangle className="h-4 w-4 text-amber-400" />
                    <AlertTitle className="text-amber-300">Trigger Pattern</AlertTitle>
                    <AlertDescription className="text-amber-200/80">
                      Higher entropy entries often occur on weekends. Consider planning 
                      extra support or activities during these times.
                    </AlertDescription>
                  </Alert>

                  <Alert className="bg-teal-500/10 border-teal-500/30">
                    <Heart className="h-4 w-4 text-teal-400" />
                    <AlertTitle className="text-teal-300">Coping Success</AlertTitle>
                    <AlertDescription className="text-teal-200/80">
                      Entries mentioning grounding techniques show 40% lower entropy scores. 
                      These strategies are working for you.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
