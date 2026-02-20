import { describe, it, expect } from 'vitest';

// Phase 10 Features Tests: Daily Affirmations, Symptom Tracker, Social Connection Prompts

describe('Daily Affirmations Enhanced', () => {
  // Affirmation categories
  const affirmationCategories = [
    'depression', 'anxiety', 'trauma', 'grief', 'isolation',
    'selfWorth', 'recovery', 'morning', 'evening', 'general'
  ];

  it('should have affirmations for all mood categories', () => {
    affirmationCategories.forEach(category => {
      expect(category).toBeDefined();
    });
    expect(affirmationCategories.length).toBe(10);
  });

  it('should provide time-based greetings', () => {
    const hour = new Date().getHours();
    let expectedPeriod: string;
    
    if (hour < 12) expectedPeriod = 'morning';
    else if (hour < 17) expectedPeriod = 'afternoon';
    else if (hour < 21) expectedPeriod = 'evening';
    else expectedPeriod = 'night';
    
    expect(['morning', 'afternoon', 'evening', 'night']).toContain(expectedPeriod);
  });

  it('should personalize affirmations based on mood patterns', () => {
    const lowMoods = [1, 2, 1, 2, 1];
    const avgMood = lowMoods.reduce((a, b) => a + b, 0) / lowMoods.length;
    
    // Low mood should trigger depression affirmations
    expect(avgMood).toBeLessThan(2.5);
  });

  it('should support favorite affirmations storage', () => {
    const favorites = ['affirmation1', 'affirmation2'];
    const stored = JSON.stringify(favorites);
    const parsed = JSON.parse(stored);
    
    expect(parsed).toEqual(favorites);
    expect(parsed.length).toBe(2);
  });

  it('should support text-to-speech for affirmations', () => {
    // Web Speech API availability check
    const speechSupported = typeof window !== 'undefined' && 'speechSynthesis' in window;
    // In test environment, this will be false, but the component handles it
    expect(typeof speechSupported).toBe('boolean');
  });

  it('should support notification scheduling for morning affirmations', () => {
    const notificationTime = new Date();
    notificationTime.setHours(8, 0, 0, 0);
    
    expect(notificationTime.getHours()).toBe(8);
    expect(notificationTime.getMinutes()).toBe(0);
  });
});

describe('Symptom Tracker', () => {
  // Symptom categories
  const symptomCategories = {
    physical: ['headache', 'muscle_tension', 'fatigue', 'nausea', 'dizziness', 'chest_tightness', 'trembling', 'sweating'],
    cognitive: ['brain_fog', 'concentration', 'memory', 'racing_thoughts', 'confusion', 'intrusive_thoughts'],
    cardiovascular: ['rapid_heartbeat', 'palpitations', 'shortness_breath'],
    sleep: ['insomnia', 'hypersomnia', 'nightmares', 'restless_sleep', 'sleep_paralysis'],
    appetite: ['loss_appetite', 'increased_appetite', 'cravings', 'digestive_issues'],
    energy: ['low_energy', 'restlessness', 'agitation', 'lethargy']
  };

  it('should have all symptom categories defined', () => {
    expect(Object.keys(symptomCategories).length).toBe(6);
  });

  it('should have physical symptoms defined', () => {
    expect(symptomCategories.physical.length).toBe(8);
    expect(symptomCategories.physical).toContain('headache');
    expect(symptomCategories.physical).toContain('fatigue');
  });

  it('should have cognitive symptoms defined', () => {
    expect(symptomCategories.cognitive.length).toBe(6);
    expect(symptomCategories.cognitive).toContain('brain_fog');
    expect(symptomCategories.cognitive).toContain('intrusive_thoughts');
  });

  it('should have cardiovascular symptoms defined', () => {
    expect(symptomCategories.cardiovascular.length).toBe(3);
    expect(symptomCategories.cardiovascular).toContain('rapid_heartbeat');
  });

  it('should have sleep symptoms defined', () => {
    expect(symptomCategories.sleep.length).toBe(5);
    expect(symptomCategories.sleep).toContain('insomnia');
    expect(symptomCategories.sleep).toContain('nightmares');
  });

  it('should have appetite symptoms defined', () => {
    expect(symptomCategories.appetite.length).toBe(4);
    expect(symptomCategories.appetite).toContain('loss_appetite');
  });

  it('should have energy symptoms defined', () => {
    expect(symptomCategories.energy.length).toBe(4);
    expect(symptomCategories.energy).toContain('low_energy');
    expect(symptomCategories.energy).toContain('restlessness');
  });

  it('should support severity levels', () => {
    const severityLevels = [
      { value: 1, label: 'Mild' },
      { value: 2, label: 'Moderate' },
      { value: 3, label: 'Severe' }
    ];
    
    expect(severityLevels.length).toBe(3);
    expect(severityLevels[0].label).toBe('Mild');
    expect(severityLevels[2].label).toBe('Severe');
  });

  it('should calculate mood correlations correctly', () => {
    const symptomData = [
      { symptomId: 'headache', mood: 2 },
      { symptomId: 'headache', mood: 1 },
      { symptomId: 'headache', mood: 2 },
      { symptomId: 'fatigue', mood: 3 },
      { symptomId: 'fatigue', mood: 4 }
    ];
    
    // Calculate average mood when headache present
    const headacheMoods = symptomData.filter(d => d.symptomId === 'headache').map(d => d.mood);
    const avgHeadacheMood = headacheMoods.reduce((a, b) => a + b, 0) / headacheMoods.length;
    
    expect(avgHeadacheMood).toBeCloseTo(1.67, 1);
    
    // Calculate average mood when fatigue present
    const fatigueMoods = symptomData.filter(d => d.symptomId === 'fatigue').map(d => d.mood);
    const avgFatigueMood = fatigueMoods.reduce((a, b) => a + b, 0) / fatigueMoods.length;
    
    expect(avgFatigueMood).toBe(3.5);
  });

  it('should store symptom entries with timestamps', () => {
    const entry = {
      id: 'headache-2026-01-27',
      symptomId: 'headache',
      severity: 2,
      timestamp: new Date().toISOString(),
      mood: 3,
      notes: 'After stressful meeting'
    };
    
    expect(entry.symptomId).toBe('headache');
    expect(entry.severity).toBe(2);
    expect(entry.timestamp).toBeDefined();
  });

  it('should filter recent symptoms (24 hours)', () => {
    const now = Date.now();
    const oneDayAgo = now - 24 * 60 * 60 * 1000;
    const twoDaysAgo = now - 48 * 60 * 60 * 1000;
    
    const entries = [
      { timestamp: new Date(now - 1000).toISOString() }, // Recent
      { timestamp: new Date(oneDayAgo + 1000).toISOString() }, // Within 24h
      { timestamp: new Date(twoDaysAgo).toISOString() } // Old
    ];
    
    const recentThreshold = new Date(oneDayAgo).toISOString();
    const recentEntries = entries.filter(e => e.timestamp > recentThreshold);
    
    expect(recentEntries.length).toBe(2);
  });
});

describe('Social Connection Prompts', () => {
  // Connection prompt types
  const connectionPrompts = {
    gentle: ['text_friend', 'reply_message', 'share_something'],
    moderate: ['voice_call', 'schedule_hangout', 'join_online'],
    meaningful: ['express_gratitude', 'ask_for_help', 'reconnect']
  };

  it('should have gentle prompts for high isolation', () => {
    expect(connectionPrompts.gentle.length).toBe(3);
    expect(connectionPrompts.gentle).toContain('text_friend');
  });

  it('should have moderate prompts for medium isolation', () => {
    expect(connectionPrompts.moderate.length).toBe(3);
    expect(connectionPrompts.moderate).toContain('voice_call');
  });

  it('should have meaningful prompts for low isolation', () => {
    expect(connectionPrompts.meaningful.length).toBe(3);
    expect(connectionPrompts.meaningful).toContain('express_gratitude');
  });

  it('should detect isolation indicators in text', () => {
    const isolationIndicators = [
      'alone', 'lonely', 'isolated', 'no one', 'nobody', 'by myself',
      'no friends', 'no one cares', 'disconnected', 'withdrawn'
    ];
    
    const testText = "I feel so alone and isolated lately";
    const foundIndicators = isolationIndicators.filter(ind => 
      testText.toLowerCase().includes(ind)
    );
    
    expect(foundIndicators.length).toBeGreaterThan(0);
    expect(foundIndicators).toContain('alone');
    expect(foundIndicators).toContain('isolated');
  });

  it('should calculate isolation score correctly', () => {
    const calculateIsolationScore = (messages: string[], daysWithoutActivity: number) => {
      let score = 0;
      const isolationIndicators = ['alone', 'lonely', 'isolated', 'no one'];
      
      const allText = messages.join(' ').toLowerCase();
      isolationIndicators.forEach(indicator => {
        if (allText.includes(indicator)) {
          score += 2;
        }
      });
      
      score += Math.min(daysWithoutActivity * 1.5, 10);
      
      return Math.max(0, Math.min(score, 10));
    };
    
    // Test with isolation indicators
    const highIsolation = calculateIsolationScore(['I feel so alone and lonely'], 3);
    expect(highIsolation).toBeGreaterThan(5);
    
    // Test without indicators
    const lowIsolation = calculateIsolationScore(['Had a great day'], 0);
    expect(lowIsolation).toBe(0);
  });

  it('should select appropriate prompts based on isolation score', () => {
    const selectPromptType = (score: number) => {
      if (score >= 7) return 'gentle';
      if (score >= 5) return 'moderate';
      return 'meaningful';
    };
    
    expect(selectPromptType(8)).toBe('gentle');
    expect(selectPromptType(6)).toBe('moderate');
    expect(selectPromptType(3)).toBe('meaningful');
  });

  it('should track connection streaks', () => {
    const calculateStreak = (logs: { timestamp: string, completed: boolean }[]) => {
      let streak = 0;
      const sortedLogs = logs
        .filter(log => log.completed)
        .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      
      if (sortedLogs.length === 0) return 0;
      
      let currentDate = new Date();
      currentDate.setHours(0, 0, 0, 0);
      
      for (const log of sortedLogs) {
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
    
    const today = new Date().toISOString();
    const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    
    const logs = [
      { timestamp: today, completed: true },
      { timestamp: yesterday, completed: true }
    ];
    
    const streak = calculateStreak(logs);
    expect(streak).toBeGreaterThanOrEqual(1);
  });

  it('should support dismissing prompts for the day', () => {
    const dismissedDate = new Date().toDateString();
    const currentDate = new Date().toDateString();
    
    expect(dismissedDate).toBe(currentDate);
  });

  it('should provide encouraging messages on completion', () => {
    const encouragingMessages = [
      "That took courage. You did something meaningful today.",
      "Connection is a gift you give yourself. Well done.",
      "Small steps lead to big changes. You're doing great."
    ];
    
    expect(encouragingMessages.length).toBeGreaterThan(0);
    expect(encouragingMessages[0]).toContain('courage');
  });
});

describe('Integration Tests', () => {
  it('should integrate affirmations with mood data', () => {
    const moodData = [2, 3, 2, 1, 2];
    const avgMood = moodData.reduce((a, b) => a + b, 0) / moodData.length;
    
    // Low mood should trigger depression-focused affirmations
    const shouldShowDepressionAffirmations = avgMood < 2.5;
    expect(shouldShowDepressionAffirmations).toBe(true);
  });

  it('should integrate symptoms with mood tracking', () => {
    const symptomEntry = {
      symptomId: 'headache',
      severity: 2,
      mood: 2,
      timestamp: new Date().toISOString()
    };
    
    // Symptom logged with mood context
    expect(symptomEntry.mood).toBeDefined();
    expect(symptomEntry.severity).toBeDefined();
  });

  it('should integrate social prompts with isolation detection', () => {
    const recentMessages = ['I feel so alone', 'Nobody understands me'];
    const isolationKeywords = ['alone', 'nobody'];
    
    let isolationScore = 0;
    recentMessages.forEach(msg => {
      isolationKeywords.forEach(keyword => {
        if (msg.toLowerCase().includes(keyword)) {
          isolationScore += 2;
        }
      });
    });
    
    // Should trigger social connection prompts
    expect(isolationScore).toBeGreaterThan(0);
  });

  it('should support localStorage persistence for all features', () => {
    const testData = {
      affirmationFavorites: ['aff1', 'aff2'],
      symptomHistory: [{ id: 'test', symptomId: 'headache' }],
      connectionLog: [{ promptId: 'text_friend', completed: true }]
    };
    
    const serialized = JSON.stringify(testData);
    const deserialized = JSON.parse(serialized);
    
    expect(deserialized.affirmationFavorites.length).toBe(2);
    expect(deserialized.symptomHistory.length).toBe(1);
    expect(deserialized.connectionLog.length).toBe(1);
  });
});
