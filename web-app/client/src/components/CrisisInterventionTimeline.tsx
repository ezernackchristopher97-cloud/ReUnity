import { useState, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Activity, 
  AlertTriangle, 
  Calendar, 
  ChevronDown, 
  ChevronUp, 
  Clock, 
  Heart, 
  Lightbulb, 
  Moon, 
  Shield, 
  Sun, 
  TrendingDown, 
  TrendingUp,
  Zap
} from "lucide-react";

interface CrisisEvent {
  id: string;
  date: Date;
  severity: "low" | "moderate" | "high" | "crisis";
  entropyScore: number;
  triggers: string[];
  symptoms: string[];
  interventionUsed: string | null;
  duration: number; // minutes
  resolution: string;
  timeOfDay: "morning" | "afternoon" | "evening" | "night";
  dayOfWeek: number;
}

interface Pattern {
  type: "time" | "trigger" | "day" | "seasonal";
  description: string;
  frequency: number;
  recommendation: string;
}

// Mock historical data - in production this would come from the database
const generateMockEvents = (): CrisisEvent[] => {
  const events: CrisisEvent[] = [];
  const triggers = [
    "Work stress", "Family conflict", "Sleep deprivation", "Social isolation",
    "Financial worry", "Health anxiety", "Relationship tension", "Trauma reminder",
    "Overwhelming tasks", "Sensory overload"
  ];
  const symptoms = [
    "Racing thoughts", "Panic", "Dissociation", "Emotional numbness",
    "Crying", "Anger", "Hopelessness", "Physical tension", "Intrusive thoughts"
  ];
  const interventions = [
    "5-4-3-2-1 Grounding", "Box Breathing", "Called support person",
    "Used safety plan", "Journaling", "Physical exercise", "Meditation"
  ];
  const resolutions = [
    "Symptoms subsided naturally", "Grounding technique helped",
    "Support from loved one", "Professional intervention", "Self-care activities"
  ];
  
  // Generate 30 days of data
  for (let i = 0; i < 30; i++) {
    // 40% chance of an event each day
    if (Math.random() < 0.4) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const hour = Math.floor(Math.random() * 24);
      date.setHours(hour);
      
      const severity = Math.random() < 0.1 ? "crisis" : 
                      Math.random() < 0.3 ? "high" :
                      Math.random() < 0.6 ? "moderate" : "low";
      
      events.push({
        id: `event-${i}`,
        date,
        severity,
        entropyScore: severity === "crisis" ? 85 + Math.random() * 15 :
                      severity === "high" ? 65 + Math.random() * 20 :
                      severity === "moderate" ? 40 + Math.random() * 25 :
                      20 + Math.random() * 20,
        triggers: [triggers[Math.floor(Math.random() * triggers.length)]],
        symptoms: [
          symptoms[Math.floor(Math.random() * symptoms.length)],
          symptoms[Math.floor(Math.random() * symptoms.length)]
        ].filter((v, i, a) => a.indexOf(v) === i),
        interventionUsed: Math.random() > 0.3 ? interventions[Math.floor(Math.random() * interventions.length)] : null,
        duration: Math.floor(15 + Math.random() * 120),
        resolution: resolutions[Math.floor(Math.random() * resolutions.length)],
        timeOfDay: hour < 6 ? "night" : hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening",
        dayOfWeek: date.getDay()
      });
    }
  }
  
  return events.sort((a, b) => b.date.getTime() - a.date.getTime());
};

export function CrisisInterventionTimeline({ compact = false }: { compact?: boolean }) {
  const [events] = useState<CrisisEvent[]>(generateMockEvents);
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<"week" | "month" | "all">("month");

  const filteredEvents = useMemo(() => {
    const now = new Date();
    const cutoff = new Date();
    
    if (timeRange === "week") {
      cutoff.setDate(now.getDate() - 7);
    } else if (timeRange === "month") {
      cutoff.setDate(now.getDate() - 30);
    } else {
      cutoff.setFullYear(now.getFullYear() - 1);
    }
    
    return events.filter(e => e.date >= cutoff);
  }, [events, timeRange]);

  const patterns = useMemo((): Pattern[] => {
    const patterns: Pattern[] = [];
    
    // Time of day analysis
    const timeCount: Record<string, number> = { morning: 0, afternoon: 0, evening: 0, night: 0 };
    filteredEvents.forEach(e => timeCount[e.timeOfDay]++);
    const maxTime = Object.entries(timeCount).sort((a, b) => b[1] - a[1])[0];
    if (maxTime[1] > filteredEvents.length * 0.3) {
      patterns.push({
        type: "time",
        description: `Most episodes occur during ${maxTime[0]} hours`,
        frequency: Math.round((maxTime[1] / filteredEvents.length) * 100),
        recommendation: maxTime[0] === "night" 
          ? "Consider improving sleep hygiene and having grounding tools by your bed"
          : maxTime[0] === "morning"
          ? "Try a calming morning routine before starting your day"
          : maxTime[0] === "evening"
          ? "Build in decompression time after work/daily activities"
          : "Schedule regular breaks and check-ins during the day"
      });
    }
    
    // Day of week analysis
    const dayCount: Record<number, number> = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    filteredEvents.forEach(e => dayCount[e.dayOfWeek]++);
    const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const maxDay = Object.entries(dayCount).sort((a, b) => b[1] - a[1])[0];
    if (parseInt(maxDay[1].toString()) > filteredEvents.length * 0.25) {
      patterns.push({
        type: "day",
        description: `${dayNames[parseInt(maxDay[0])]}s tend to be more challenging`,
        frequency: Math.round((parseInt(maxDay[1].toString()) / filteredEvents.length) * 100),
        recommendation: `Plan extra self-care and support for ${dayNames[parseInt(maxDay[0])]}s`
      });
    }
    
    // Trigger analysis
    const triggerCount: Record<string, number> = {};
    filteredEvents.forEach(e => {
      e.triggers.forEach(t => {
        triggerCount[t] = (triggerCount[t] || 0) + 1;
      });
    });
    const topTriggers = Object.entries(triggerCount).sort((a, b) => b[1] - a[1]).slice(0, 2);
    topTriggers.forEach(([trigger, count]) => {
      if (count > 2) {
        patterns.push({
          type: "trigger",
          description: `"${trigger}" is a recurring trigger`,
          frequency: count,
          recommendation: `Develop specific coping strategies for ${trigger.toLowerCase()}`
        });
      }
    });
    
    return patterns;
  }, [filteredEvents]);

  const stats = useMemo(() => {
    const crisisCount = filteredEvents.filter(e => e.severity === "crisis").length;
    const highCount = filteredEvents.filter(e => e.severity === "high").length;
    const avgDuration = filteredEvents.length > 0 
      ? Math.round(filteredEvents.reduce((sum, e) => sum + e.duration, 0) / filteredEvents.length)
      : 0;
    const avgEntropy = filteredEvents.length > 0
      ? Math.round(filteredEvents.reduce((sum, e) => sum + e.entropyScore, 0) / filteredEvents.length)
      : 0;
    const interventionRate = filteredEvents.length > 0
      ? Math.round((filteredEvents.filter(e => e.interventionUsed).length / filteredEvents.length) * 100)
      : 0;
    
    return { crisisCount, highCount, avgDuration, avgEntropy, interventionRate, total: filteredEvents.length };
  }, [filteredEvents]);

  const getSeverityColor = (severity: CrisisEvent["severity"]) => {
    switch (severity) {
      case "low": return "bg-green-500/20 text-green-400 border-green-500/30";
      case "moderate": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "high": return "bg-orange-500/20 text-orange-400 border-orange-500/30";
      case "crisis": return "bg-red-500/20 text-red-400 border-red-500/30";
    }
  };

  const getTimeIcon = (time: CrisisEvent["timeOfDay"]) => {
    switch (time) {
      case "morning": return <Sun className="h-3 w-3" />;
      case "afternoon": return <Sun className="h-3 w-3" />;
      case "evening": return <Moon className="h-3 w-3" />;
      case "night": return <Moon className="h-3 w-3" />;
    }
  };

  if (compact) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-4 w-4 text-blue-400" />
            Crisis Timeline
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between text-xs">
            <div className="text-slate-400">
              {stats.total} events tracked
            </div>
            {stats.crisisCount > 0 && (
              <Badge variant="outline" className="text-xs bg-red-500/20 text-red-400 border-red-500/30">
                {stats.crisisCount} crisis
              </Badge>
            )}
          </div>
          <div className="mt-2 flex gap-1">
            {filteredEvents.slice(0, 14).map((event, i) => (
              <div
                key={i}
                className={`h-6 w-2 rounded-sm ${
                  event.severity === "crisis" ? "bg-red-500" :
                  event.severity === "high" ? "bg-orange-500" :
                  event.severity === "moderate" ? "bg-yellow-500" :
                  "bg-green-500"
                }`}
                title={`${event.date.toLocaleDateString()} - ${event.severity}`}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-blue-400" />
                Crisis Intervention Timeline
              </CardTitle>
              <CardDescription>
                Track patterns and identify triggers to prevent future crises
              </CardDescription>
            </div>
            <div className="flex gap-1">
              {(["week", "month", "all"] as const).map((range) => (
                <Button
                  key={range}
                  variant={timeRange === range ? "default" : "outline"}
                  size="sm"
                  onClick={() => setTimeRange(range)}
                  className="text-xs"
                >
                  {range === "week" ? "7 Days" : range === "month" ? "30 Days" : "All"}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="timeline" className="space-y-4">
            <TabsList className="bg-slate-900/50">
              <TabsTrigger value="timeline">Timeline</TabsTrigger>
              <TabsTrigger value="patterns">Patterns</TabsTrigger>
              <TabsTrigger value="stats">Statistics</TabsTrigger>
            </TabsList>

            <TabsContent value="timeline" className="space-y-3">
              {filteredEvents.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No events recorded in this time period</p>
                  <p className="text-xs mt-1">Events are automatically tracked from your check-ins and mood data</p>
                </div>
              ) : (
                filteredEvents.map((event) => (
                  <div
                    key={event.id}
                    className={`rounded-lg border p-3 ${getSeverityColor(event.severity)}`}
                  >
                    <div 
                      className="flex items-center justify-between cursor-pointer"
                      onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1 text-xs">
                          <Calendar className="h-3 w-3" />
                          {event.date.toLocaleDateString()}
                        </div>
                        <div className="flex items-center gap-1 text-xs">
                          {getTimeIcon(event.timeOfDay)}
                          {event.timeOfDay}
                        </div>
                        <Badge variant="outline" className={getSeverityColor(event.severity)}>
                          {event.severity}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs">Entropy: {Math.round(event.entropyScore)}</span>
                        {expandedEvent === event.id ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </div>
                    </div>
                    
                    {expandedEvent === event.id && (
                      <div className="mt-3 pt-3 border-t border-current/20 space-y-2 text-sm">
                        <div className="flex items-start gap-2">
                          <Zap className="h-4 w-4 mt-0.5 text-yellow-400" />
                          <div>
                            <strong>Triggers:</strong> {event.triggers.join(", ")}
                          </div>
                        </div>
                        <div className="flex items-start gap-2">
                          <AlertTriangle className="h-4 w-4 mt-0.5 text-orange-400" />
                          <div>
                            <strong>Symptoms:</strong> {event.symptoms.join(", ")}
                          </div>
                        </div>
                        <div className="flex items-start gap-2">
                          <Clock className="h-4 w-4 mt-0.5 text-blue-400" />
                          <div>
                            <strong>Duration:</strong> {event.duration} minutes
                          </div>
                        </div>
                        {event.interventionUsed && (
                          <div className="flex items-start gap-2">
                            <Shield className="h-4 w-4 mt-0.5 text-emerald-400" />
                            <div>
                              <strong>Intervention:</strong> {event.interventionUsed}
                            </div>
                          </div>
                        )}
                        <div className="flex items-start gap-2">
                          <Heart className="h-4 w-4 mt-0.5 text-pink-400" />
                          <div>
                            <strong>Resolution:</strong> {event.resolution}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </TabsContent>

            <TabsContent value="patterns" className="space-y-3">
              {patterns.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <Lightbulb className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>Not enough data to identify patterns yet</p>
                  <p className="text-xs mt-1">Continue tracking to discover your unique patterns</p>
                </div>
              ) : (
                patterns.map((pattern, i) => (
                  <div key={i} className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-lg bg-blue-500/20">
                        <Lightbulb className="h-5 w-5 text-blue-400" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">{pattern.description}</div>
                        <div className="text-xs text-slate-400 mt-1">
                          {pattern.type === "trigger" 
                            ? `Occurred ${pattern.frequency} times`
                            : `${pattern.frequency}% of episodes`
                          }
                        </div>
                        <div className="mt-2 p-2 bg-emerald-500/10 border border-emerald-500/30 rounded text-xs text-emerald-400">
                          <strong>Prevention Strategy:</strong> {pattern.recommendation}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </TabsContent>

            <TabsContent value="stats" className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-blue-400">{stats.total}</div>
                  <div className="text-xs text-slate-400">Total Events</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-red-400">{stats.crisisCount}</div>
                  <div className="text-xs text-slate-400">Crisis Episodes</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-orange-400">{stats.highCount}</div>
                  <div className="text-xs text-slate-400">High Severity</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-purple-400">{stats.avgEntropy}</div>
                  <div className="text-xs text-slate-400">Avg Entropy</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-yellow-400">{stats.avgDuration}m</div>
                  <div className="text-xs text-slate-400">Avg Duration</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-emerald-400">{stats.interventionRate}%</div>
                  <div className="text-xs text-slate-400">Used Intervention</div>
                </div>
              </div>

              {/* Trend indicator */}
              <div className="bg-slate-900/50 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">Overall Trend</div>
                  {stats.crisisCount === 0 ? (
                    <div className="flex items-center gap-2 text-emerald-400">
                      <TrendingDown className="h-4 w-4" />
                      <span className="text-sm">Improving</span>
                    </div>
                  ) : stats.crisisCount > 2 ? (
                    <div className="flex items-center gap-2 text-red-400">
                      <TrendingUp className="h-4 w-4" />
                      <span className="text-sm">Needs attention</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-yellow-400">
                      <Activity className="h-4 w-4" />
                      <span className="text-sm">Stable</span>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
