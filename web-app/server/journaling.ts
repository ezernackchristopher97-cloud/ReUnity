/**
 * Journaling System with Entropy Tracking
 * 
 * Allows users to track their emotional patterns over time with
 * entropy analysis and Vicsek trajectory visualization.
 */

import { EntropyState, ConditionCategory } from "./reunity";
import { analyzeTrajectory, VicsekPrediction } from "./vicsek";

export interface JournalEntry {
  id: string;
  odId: string;  // User's open ID
  createdAt: Date;
  updatedAt: Date;
  
  // Content
  content: string;
  title?: string;
  
  // Mood/state tags (user-selected)
  moodTags: MoodTag[];
  customTags: string[];
  
  // AI-analyzed metrics
  entropyScore: number;
  entropyState: EntropyState;
  detectedConditions: ConditionCategory[];
  detectedStates: string[];
  
  // Patterns detected
  triggersIdentified: string[];
  copingUsed: string[];
  
  // Privacy
  isPrivate: boolean;
  isEncrypted: boolean;
}

export type MoodTag = 
  | "happy"
  | "sad"
  | "anxious"
  | "angry"
  | "calm"
  | "hopeful"
  | "hopeless"
  | "numb"
  | "overwhelmed"
  | "peaceful"
  | "scared"
  | "frustrated"
  | "grateful"
  | "lonely"
  | "loved"
  | "confused"
  | "dissociated"
  | "grounded"
  | "triggered"
  | "recovering";

export interface JournalAnalytics {
  // Time-based metrics
  entropyOverTime: { date: Date; entropy: number; state: EntropyState }[];
  moodDistribution: Record<MoodTag, number>;
  
  // Pattern analysis
  commonTriggers: { trigger: string; frequency: number }[];
  effectiveCoping: { technique: string; successRate: number }[];
  
  // Vicsek trajectory
  trajectoryPrediction: VicsekPrediction;
  trajectoryHistory: { date: Date; trajectory: string; urgency: string }[];
  
  // Insights
  insights: JournalInsight[];
  
  // Progress metrics
  averageEntropy: number;
  entropyTrend: "improving" | "stable" | "declining";
  daysJournaled: number;
  currentStreak: number;
  longestStreak: number;
}

export interface JournalInsight {
  id: string;
  type: "pattern" | "progress" | "warning" | "suggestion";
  title: string;
  description: string;
  confidence: number;
  relatedEntries: string[];
  createdAt: Date;
}

export interface JournalPrompt {
  id: string;
  category: "reflection" | "gratitude" | "processing" | "grounding" | "growth";
  prompt: string;
  followUp?: string;
  forStates?: EntropyState[];
  forConditions?: ConditionCategory[];
}

// Mood tag metadata for UI
export const moodTagInfo: Record<MoodTag, { emoji: string; color: string; description: string }> = {
  happy: { emoji: "😊", color: "#4ade80", description: "Feeling good, content" },
  sad: { emoji: "😢", color: "#60a5fa", description: "Feeling down, blue" },
  anxious: { emoji: "😰", color: "#fbbf24", description: "Worried, nervous, on edge" },
  angry: { emoji: "😠", color: "#f87171", description: "Frustrated, irritated, mad" },
  calm: { emoji: "😌", color: "#a78bfa", description: "Peaceful, relaxed" },
  hopeful: { emoji: "🌟", color: "#fcd34d", description: "Optimistic about the future" },
  hopeless: { emoji: "😞", color: "#6b7280", description: "Feeling like nothing will change" },
  numb: { emoji: "😶", color: "#9ca3af", description: "Not feeling much of anything" },
  overwhelmed: { emoji: "😵", color: "#fb923c", description: "Too much to handle" },
  peaceful: { emoji: "🕊️", color: "#86efac", description: "Inner calm, serenity" },
  scared: { emoji: "😨", color: "#fca5a5", description: "Afraid, fearful" },
  frustrated: { emoji: "😤", color: "#fdba74", description: "Stuck, blocked, annoyed" },
  grateful: { emoji: "🙏", color: "#c4b5fd", description: "Appreciating what you have" },
  lonely: { emoji: "😔", color: "#93c5fd", description: "Feeling isolated, alone" },
  loved: { emoji: "💕", color: "#f9a8d4", description: "Feeling connected, cared for" },
  confused: { emoji: "😕", color: "#d1d5db", description: "Uncertain, unclear" },
  dissociated: { emoji: "🌫️", color: "#e5e7eb", description: "Disconnected, foggy, unreal" },
  grounded: { emoji: "🌳", color: "#34d399", description: "Present, connected to body" },
  triggered: { emoji: "⚡", color: "#ef4444", description: "Activated by something" },
  recovering: { emoji: "🌱", color: "#10b981", description: "Coming back from difficulty" }
};

// Journal prompts library
export const journalPrompts: JournalPrompt[] = [
  // Reflection prompts
  {
    id: "reflect_1",
    category: "reflection",
    prompt: "What emotions have been most present for you today?",
    followUp: "Can you trace where these feelings might be coming from?"
  },
  {
    id: "reflect_2",
    category: "reflection",
    prompt: "What was the hardest moment of your day? What made it hard?",
    followUp: "How did you get through it?"
  },
  {
    id: "reflect_3",
    category: "reflection",
    prompt: "If your feelings right now could speak, what would they say?",
    followUp: "What do they need from you?"
  },
  {
    id: "reflect_4",
    category: "reflection",
    prompt: "What patterns do you notice in how you've been feeling lately?"
  },
  {
    id: "reflect_5",
    category: "reflection",
    prompt: "What would you tell a friend who was feeling the way you feel right now?"
  },
  
  // Gratitude prompts
  {
    id: "gratitude_1",
    category: "gratitude",
    prompt: "Name three small things that brought you comfort today.",
    followUp: "Why did these things help?"
  },
  {
    id: "gratitude_2",
    category: "gratitude",
    prompt: "Who is someone who has shown you kindness recently?",
    followUp: "How did their kindness affect you?"
  },
  {
    id: "gratitude_3",
    category: "gratitude",
    prompt: "What is one thing your body did for you today that you can appreciate?"
  },
  {
    id: "gratitude_4",
    category: "gratitude",
    prompt: "What is something you're looking forward to, even if it's small?"
  },
  
  // Processing prompts
  {
    id: "process_1",
    category: "processing",
    prompt: "What is weighing on you right now?",
    followUp: "What would it feel like to set it down, even temporarily?",
    forStates: [EntropyState.HIGH, EntropyState.MODERATE]
  },
  {
    id: "process_2",
    category: "processing",
    prompt: "Is there something you need to say that you haven't been able to say?",
    followUp: "What stops you from saying it?"
  },
  {
    id: "process_3",
    category: "processing",
    prompt: "What are you avoiding right now? What would happen if you faced it?",
    forStates: [EntropyState.MODERATE, EntropyState.LOW]
  },
  {
    id: "process_4",
    category: "processing",
    prompt: "Write about a memory that keeps coming back to you.",
    followUp: "What do you think it's trying to tell you?",
    forConditions: [ConditionCategory.TRAUMA_PTSD]
  },
  
  // Grounding prompts
  {
    id: "ground_1",
    category: "grounding",
    prompt: "Describe your surroundings right now using all five senses.",
    forStates: [EntropyState.CRISIS, EntropyState.HIGH]
  },
  {
    id: "ground_2",
    category: "grounding",
    prompt: "What does safety feel like in your body? Where do you feel it?",
    forStates: [EntropyState.HIGH, EntropyState.MODERATE]
  },
  {
    id: "ground_3",
    category: "grounding",
    prompt: "Name five things you can see, four you can touch, three you can hear.",
    forStates: [EntropyState.CRISIS, EntropyState.HIGH],
    forConditions: [ConditionCategory.DISSOCIATIVE]
  },
  {
    id: "ground_4",
    category: "grounding",
    prompt: "What is one thing that is true right now, in this moment?",
    forStates: [EntropyState.CRISIS, EntropyState.HIGH]
  },
  
  // Growth prompts
  {
    id: "growth_1",
    category: "growth",
    prompt: "What is one thing you've learned about yourself recently?",
    forStates: [EntropyState.STABLE, EntropyState.LOW]
  },
  {
    id: "growth_2",
    category: "growth",
    prompt: "What is a boundary you've set or want to set?",
    followUp: "What makes this boundary important to you?"
  },
  {
    id: "growth_3",
    category: "growth",
    prompt: "How have you grown from a difficult experience?",
    forStates: [EntropyState.STABLE, EntropyState.LOW]
  },
  {
    id: "growth_4",
    category: "growth",
    prompt: "What does healing look like for you? What small step could you take toward it?"
  },
  {
    id: "growth_5",
    category: "growth",
    prompt: "What would you like to tell your past self? What would you like to tell your future self?"
  }
];

/**
 * Get a prompt appropriate for the user's current state
 */
export function getPromptForState(
  state: EntropyState,
  conditions: ConditionCategory[] = [],
  category?: JournalPrompt["category"]
): JournalPrompt {
  let candidates = journalPrompts;
  
  // Filter by category if specified
  if (category) {
    candidates = candidates.filter(p => p.category === category);
  }
  
  // Filter by state if specified
  const stateFiltered = candidates.filter(p => 
    !p.forStates || p.forStates.includes(state)
  );
  
  if (stateFiltered.length > 0) {
    candidates = stateFiltered;
  }
  
  // Prefer condition-specific prompts if available
  if (conditions.length > 0) {
    const conditionFiltered = candidates.filter(p =>
      p.forConditions && p.forConditions.some(c => conditions.includes(c))
    );
    if (conditionFiltered.length > 0) {
      candidates = conditionFiltered;
    }
  }
  
  // Return random prompt from candidates
  return candidates[Math.floor(Math.random() * candidates.length)];
}

/**
 * Analyze journal entry for patterns and triggers
 */
export function analyzeJournalEntry(
  content: string,
  moodTags: MoodTag[]
): {
  triggersIdentified: string[];
  copingUsed: string[];
  themes: string[];
} {
  const text = content.toLowerCase();
  
  // Common trigger patterns
  const triggerPatterns: Record<string, string[]> = {
    "conflict": ["argument", "fight", "yelled", "screamed", "disagreement"],
    "rejection": ["rejected", "ignored", "left out", "abandoned", "ghosted"],
    "failure": ["failed", "messed up", "mistake", "wrong", "not good enough"],
    "loss": ["lost", "gone", "miss", "died", "ended"],
    "stress": ["overwhelmed", "too much", "deadline", "pressure", "stressed"],
    "isolation": ["alone", "lonely", "no one", "isolated", "by myself"],
    "memory": ["reminded", "flashback", "memory", "remembered", "triggered"],
    "body": ["tired", "exhausted", "sick", "pain", "hungry", "didn't sleep"],
    "relationship": ["partner", "friend", "family", "mom", "dad", "ex"],
    "work": ["work", "job", "boss", "coworker", "meeting"]
  };
  
  // Coping strategies mentioned
  const copingPatterns: Record<string, string[]> = {
    "breathing": ["breathed", "breathing", "deep breath", "box breathing"],
    "grounding": ["grounded", "grounding", "5-4-3-2-1", "senses"],
    "movement": ["walked", "exercise", "yoga", "stretched", "ran"],
    "connection": ["talked to", "called", "texted", "reached out"],
    "creativity": ["wrote", "drew", "painted", "played music", "created"],
    "self-care": ["bath", "shower", "ate", "slept", "rest"],
    "distraction": ["watched", "read", "played", "listened to"],
    "mindfulness": ["meditated", "mindful", "present", "noticed"],
    "journaling": ["writing this", "journaling", "getting it out"],
    "therapy": ["therapist", "therapy", "session", "counselor"]
  };
  
  const triggersIdentified: string[] = [];
  const copingUsed: string[] = [];
  const themes: string[] = [];
  
  // Detect triggers
  for (const [trigger, patterns] of Object.entries(triggerPatterns)) {
    if (patterns.some(p => text.includes(p))) {
      triggersIdentified.push(trigger);
    }
  }
  
  // Detect coping strategies
  for (const [coping, patterns] of Object.entries(copingPatterns)) {
    if (patterns.some(p => text.includes(p))) {
      copingUsed.push(coping);
    }
  }
  
  // Detect themes from mood tags
  if (moodTags.includes("anxious") || moodTags.includes("scared")) {
    themes.push("anxiety");
  }
  if (moodTags.includes("sad") || moodTags.includes("hopeless")) {
    themes.push("depression");
  }
  if (moodTags.includes("dissociated")) {
    themes.push("dissociation");
  }
  if (moodTags.includes("triggered")) {
    themes.push("trauma_activation");
  }
  if (moodTags.includes("grounded") || moodTags.includes("calm") || moodTags.includes("peaceful")) {
    themes.push("regulation");
  }
  
  return { triggersIdentified, copingUsed, themes };
}

/**
 * Calculate analytics from journal entries
 */
export function calculateAnalytics(entries: JournalEntry[]): JournalAnalytics {
  if (entries.length === 0) {
    return {
      entropyOverTime: [],
      moodDistribution: {} as Record<MoodTag, number>,
      commonTriggers: [],
      effectiveCoping: [],
      trajectoryPrediction: {
        predictedTrajectory: "stable",
        urgency: "low",
        confidence: 0,
        alignmentStrength: 0,
        noiseLevel: 0,
        recommendedIntervention: ""
      },
      trajectoryHistory: [],
      insights: [],
      averageEntropy: 0,
      entropyTrend: "stable",
      daysJournaled: 0,
      currentStreak: 0,
      longestStreak: 0
    };
  }
  
  // Sort entries by date
  const sortedEntries = [...entries].sort((a, b) => 
    new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
  );
  
  // Entropy over time
  const entropyOverTime = sortedEntries.map(e => ({
    date: new Date(e.createdAt),
    entropy: e.entropyScore,
    state: e.entropyState
  }));
  
  // Mood distribution
  const moodDistribution: Record<MoodTag, number> = {} as Record<MoodTag, number>;
  for (const entry of entries) {
    for (const mood of entry.moodTags) {
      moodDistribution[mood] = (moodDistribution[mood] || 0) + 1;
    }
  }
  
  // Common triggers
  const triggerCounts: Record<string, number> = {};
  for (const entry of entries) {
    for (const trigger of entry.triggersIdentified) {
      triggerCounts[trigger] = (triggerCounts[trigger] || 0) + 1;
    }
  }
  const commonTriggers = Object.entries(triggerCounts)
    .map(([trigger, frequency]) => ({ trigger, frequency }))
    .sort((a, b) => b.frequency - a.frequency)
    .slice(0, 10);
  
  // Effective coping (coping strategies used when entropy improved)
  const copingEffectiveness: Record<string, { used: number; helped: number }> = {};
  for (let i = 1; i < sortedEntries.length; i++) {
    const prev = sortedEntries[i - 1];
    const curr = sortedEntries[i];
    const improved = curr.entropyScore < prev.entropyScore;
    
    for (const coping of prev.copingUsed) {
      if (!copingEffectiveness[coping]) {
        copingEffectiveness[coping] = { used: 0, helped: 0 };
      }
      copingEffectiveness[coping].used++;
      if (improved) {
        copingEffectiveness[coping].helped++;
      }
    }
  }
  const effectiveCoping = Object.entries(copingEffectiveness)
    .map(([technique, stats]) => ({
      technique,
      successRate: stats.used > 0 ? stats.helped / stats.used : 0
    }))
    .filter(c => copingEffectiveness[c.technique].used >= 2)  // Minimum usage threshold
    .sort((a, b) => b.successRate - a.successRate);
  
  // Vicsek trajectory prediction
  const recentStates = sortedEntries.slice(-10).flatMap(e => e.detectedStates);
  const recentEntropy = sortedEntries.slice(-5).reduce((sum, e) => sum + e.entropyScore, 0) / 
    Math.min(5, sortedEntries.length);
  const trajectoryPrediction = analyzeTrajectory(recentStates, recentEntropy, []);
  
  // Average entropy
  const averageEntropy = entries.reduce((sum, e) => sum + e.entropyScore, 0) / entries.length;
  
  // Entropy trend (compare last 5 to previous 5)
  let entropyTrend: "improving" | "stable" | "declining" = "stable";
  if (sortedEntries.length >= 10) {
    const recent5 = sortedEntries.slice(-5).reduce((sum, e) => sum + e.entropyScore, 0) / 5;
    const previous5 = sortedEntries.slice(-10, -5).reduce((sum, e) => sum + e.entropyScore, 0) / 5;
    if (recent5 < previous5 - 0.1) {
      entropyTrend = "improving";
    } else if (recent5 > previous5 + 0.1) {
      entropyTrend = "declining";
    }
  }
  
  // Calculate streaks
  const uniqueDays = new Set(entries.map(e => 
    new Date(e.createdAt).toISOString().split('T')[0]
  ));
  const daysJournaled = uniqueDays.size;
  
  // Current streak
  let currentStreak = 0;
  const today = new Date();
  for (let i = 0; i < 365; i++) {
    const checkDate = new Date(today);
    checkDate.setDate(checkDate.getDate() - i);
    const dateStr = checkDate.toISOString().split('T')[0];
    if (uniqueDays.has(dateStr)) {
      currentStreak++;
    } else if (i > 0) {  // Allow today to be missing
      break;
    }
  }
  
  // Longest streak (simplified calculation)
  const sortedDays = Array.from(uniqueDays).sort();
  let longestStreak = 1;
  let tempStreak = 1;
  for (let i = 1; i < sortedDays.length; i++) {
    const prev = new Date(sortedDays[i - 1]);
    const curr = new Date(sortedDays[i]);
    const diffDays = (curr.getTime() - prev.getTime()) / (1000 * 60 * 60 * 24);
    if (diffDays === 1) {
      tempStreak++;
      longestStreak = Math.max(longestStreak, tempStreak);
    } else {
      tempStreak = 1;
    }
  }
  
  // Generate insights
  const insights: JournalInsight[] = [];
  
  // Trigger insight
  if (commonTriggers.length > 0) {
    insights.push({
      id: "trigger_insight",
      type: "pattern",
      title: `Common Trigger: ${commonTriggers[0].trigger}`,
      description: `"${commonTriggers[0].trigger}" appears in ${commonTriggers[0].frequency} of your entries. Recognizing this pattern can help you prepare.`,
      confidence: Math.min(0.9, commonTriggers[0].frequency / entries.length),
      relatedEntries: entries.filter(e => e.triggersIdentified.includes(commonTriggers[0].trigger)).map(e => e.id),
      createdAt: new Date()
    });
  }
  
  // Coping insight
  if (effectiveCoping.length > 0 && effectiveCoping[0].successRate > 0.5) {
    insights.push({
      id: "coping_insight",
      type: "progress",
      title: `Effective Coping: ${effectiveCoping[0].technique}`,
      description: `${effectiveCoping[0].technique} has been helpful ${Math.round(effectiveCoping[0].successRate * 100)}% of the time. Keep using what works!`,
      confidence: effectiveCoping[0].successRate,
      relatedEntries: [],
      createdAt: new Date()
    });
  }
  
  // Trend insight
  if (entropyTrend === "improving") {
    insights.push({
      id: "trend_insight",
      type: "progress",
      title: "Positive Trend",
      description: "Your emotional regulation has been improving over recent entries. Your hard work is paying off.",
      confidence: 0.7,
      relatedEntries: [],
      createdAt: new Date()
    });
  } else if (entropyTrend === "declining") {
    insights.push({
      id: "trend_insight",
      type: "warning",
      title: "Increasing Distress",
      description: "Your recent entries show higher distress levels. This might be a good time to reach out for support.",
      confidence: 0.7,
      relatedEntries: [],
      createdAt: new Date()
    });
  }
  
  return {
    entropyOverTime,
    moodDistribution,
    commonTriggers,
    effectiveCoping,
    trajectoryPrediction,
    trajectoryHistory: [],  // Would be populated from stored predictions
    insights,
    averageEntropy,
    entropyTrend,
    daysJournaled,
    currentStreak,
    longestStreak
  };
}

/**
 * Format analytics for display
 */
export function formatAnalyticsSummary(analytics: JournalAnalytics): string {
  let summary = "📊 YOUR JOURNAL INSIGHTS\n\n";
  
  summary += `📅 Journaling Stats:\n`;
  summary += `   Days journaled: ${analytics.daysJournaled}\n`;
  summary += `   Current streak: ${analytics.currentStreak} days\n`;
  summary += `   Longest streak: ${analytics.longestStreak} days\n\n`;
  
  summary += `📈 Emotional Trend: ${analytics.entropyTrend}\n`;
  summary += `   Average regulation: ${((1 - analytics.averageEntropy) * 100).toFixed(0)}%\n\n`;
  
  if (analytics.commonTriggers.length > 0) {
    summary += `⚡ Common Triggers:\n`;
    for (const t of analytics.commonTriggers.slice(0, 3)) {
      summary += `   - ${t.trigger} (${t.frequency} times)\n`;
    }
    summary += "\n";
  }
  
  if (analytics.effectiveCoping.length > 0) {
    summary += `💪 What's Helping:\n`;
    for (const c of analytics.effectiveCoping.slice(0, 3)) {
      summary += `   - ${c.technique} (${Math.round(c.successRate * 100)}% effective)\n`;
    }
    summary += "\n";
  }
  
  summary += `🔮 Trajectory: ${analytics.trajectoryPrediction.predictedTrajectory}\n`;
  summary += `   Urgency: ${analytics.trajectoryPrediction.urgency}\n`;
  
  return summary;
}
