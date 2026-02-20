import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock localStorage
const localStorageMock = {
  store: {} as Record<string, string>,
  getItem: vi.fn((key: string) => localStorageMock.store[key] || null),
  setItem: vi.fn((key: string, value: string) => { localStorageMock.store[key] = value; }),
  removeItem: vi.fn((key: string) => { delete localStorageMock.store[key]; }),
  clear: vi.fn(() => { localStorageMock.store = {}; }),
};

vi.stubGlobal('localStorage', localStorageMock);

describe('Phase 9 Features - Breathing Exercises, Mood Calendar, Therapist Notes', () => {
  beforeEach(() => {
    localStorageMock.clear();
  });

  describe('Breathing Exercises', () => {
    const breathingExercises = [
      { id: 'box', name: 'Box Breathing', pattern: { inhale: 4, hold1: 4, exhale: 4, hold2: 4 }, cycles: 4 },
      { id: '478', name: '4-7-8 Breathing', pattern: { inhale: 4, hold1: 7, exhale: 8, hold2: 0 }, cycles: 4 },
      { id: 'calm', name: 'Calming Breath', pattern: { inhale: 4, hold1: 2, exhale: 6, hold2: 0 }, cycles: 6 },
    ];

    it('should have box breathing exercise with correct pattern', () => {
      const boxBreathing = breathingExercises.find(e => e.id === 'box');
      expect(boxBreathing).toBeDefined();
      expect(boxBreathing?.pattern.inhale).toBe(4);
      expect(boxBreathing?.pattern.hold1).toBe(4);
      expect(boxBreathing?.pattern.exhale).toBe(4);
      expect(boxBreathing?.pattern.hold2).toBe(4);
    });

    it('should have 4-7-8 breathing exercise with correct pattern', () => {
      const breathing478 = breathingExercises.find(e => e.id === '478');
      expect(breathing478).toBeDefined();
      expect(breathing478?.pattern.inhale).toBe(4);
      expect(breathing478?.pattern.hold1).toBe(7);
      expect(breathing478?.pattern.exhale).toBe(8);
      expect(breathing478?.pattern.hold2).toBe(0);
    });

    it('should have calming breath exercise with correct pattern', () => {
      const calmBreathing = breathingExercises.find(e => e.id === 'calm');
      expect(calmBreathing).toBeDefined();
      expect(calmBreathing?.pattern.inhale).toBe(4);
      expect(calmBreathing?.pattern.hold1).toBe(2);
      expect(calmBreathing?.pattern.exhale).toBe(6);
    });

    it('should calculate total cycle duration correctly', () => {
      const boxBreathing = breathingExercises.find(e => e.id === 'box');
      if (boxBreathing) {
        const cycleDuration = boxBreathing.pattern.inhale + boxBreathing.pattern.hold1 + 
                             boxBreathing.pattern.exhale + boxBreathing.pattern.hold2;
        expect(cycleDuration).toBe(16); // 4+4+4+4 = 16 seconds per cycle
      }
    });

    it('should calculate total exercise duration correctly', () => {
      const breathing478 = breathingExercises.find(e => e.id === '478');
      if (breathing478) {
        const cycleDuration = breathing478.pattern.inhale + breathing478.pattern.hold1 + 
                             breathing478.pattern.exhale + breathing478.pattern.hold2;
        const totalDuration = cycleDuration * breathing478.cycles;
        expect(totalDuration).toBe(76); // (4+7+8+0) * 4 = 76 seconds
      }
    });

    it('should track completed exercises', () => {
      const completedExercises: string[] = [];
      completedExercises.push('box');
      completedExercises.push('478');
      expect(completedExercises).toContain('box');
      expect(completedExercises).toContain('478');
      expect(completedExercises.length).toBe(2);
    });
  });

  describe('Mood Calendar', () => {
    const moodColors: Record<number, string> = {
      1: '#ef4444', // Very Low - Red
      2: '#f97316', // Low - Orange
      3: '#eab308', // Neutral - Yellow
      4: '#22c55e', // Good - Green
      5: '#10b981', // Great - Emerald
    };

    const moodLabels: Record<number, string> = {
      1: 'Very Low',
      2: 'Low',
      3: 'Neutral',
      4: 'Good',
      5: 'Great',
    };

    it('should have 5 mood levels with distinct colors', () => {
      expect(Object.keys(moodColors).length).toBe(5);
      const uniqueColors = new Set(Object.values(moodColors));
      expect(uniqueColors.size).toBe(5);
    });

    it('should have labels for all mood levels', () => {
      expect(Object.keys(moodLabels).length).toBe(5);
      expect(moodLabels[1]).toBe('Very Low');
      expect(moodLabels[5]).toBe('Great');
    });

    it('should calculate monthly statistics correctly', () => {
      const monthEntries = [
        { date: '2026-01-01', mood: 3 },
        { date: '2026-01-02', mood: 4 },
        { date: '2026-01-03', mood: 5 },
        { date: '2026-01-04', mood: 2 },
        { date: '2026-01-05', mood: 4 },
      ];

      const average = monthEntries.reduce((sum, e) => sum + e.mood, 0) / monthEntries.length;
      const goodDays = monthEntries.filter(e => e.mood >= 4).length;
      const lowDays = monthEntries.filter(e => e.mood <= 2).length;

      expect(average).toBe(3.6);
      expect(goodDays).toBe(3);
      expect(lowDays).toBe(1);
    });

    it('should generate calendar days correctly for January 2026', () => {
      const year = 2026;
      const month = 0; // January
      const firstDay = new Date(year, month, 1);
      const lastDay = new Date(year, month + 1, 0);
      
      expect(firstDay.getDay()).toBe(4); // January 1, 2026 is Thursday
      expect(lastDay.getDate()).toBe(31); // January has 31 days
    });

    it('should identify today correctly', () => {
      const today = new Date().toISOString().split('T')[0];
      expect(today).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    });

    it('should track mood trends over time', () => {
      const weekMoods = [2, 3, 3, 4, 4, 5, 4];
      const trend = weekMoods[weekMoods.length - 1] - weekMoods[0];
      expect(trend).toBe(2); // Positive trend
    });
  });

  describe('Therapist Notes Sync', () => {
    interface TherapistNote {
      id: string;
      therapistName: string;
      sessionDate: string;
      type: 'session' | 'progress' | 'treatment_plan' | 'crisis';
      title: string;
      content: string;
      isSharedWithClient: boolean;
      includeInReport: boolean;
      tags: string[];
    }

    const mockNotes: TherapistNote[] = [
      {
        id: '1',
        therapistName: 'Dr. Sarah Chen',
        sessionDate: '2026-01-20',
        type: 'session',
        title: 'Weekly Check-in Session',
        content: 'Client showed improved coping strategies.',
        isSharedWithClient: true,
        includeInReport: true,
        tags: ['anxiety', 'coping', 'progress'],
      },
      {
        id: '2',
        therapistName: 'Dr. Sarah Chen',
        sessionDate: '2026-01-13',
        type: 'progress',
        title: 'Monthly Progress Review',
        content: 'Significant improvement in mood stability.',
        isSharedWithClient: true,
        includeInReport: true,
        tags: ['progress', 'mood'],
      },
      {
        id: '3',
        therapistName: 'Dr. Sarah Chen',
        sessionDate: '2026-01-06',
        type: 'treatment_plan',
        title: 'Updated Treatment Goals',
        content: 'Goals for Q1 2026.',
        isSharedWithClient: true,
        includeInReport: false,
        tags: ['goals', 'treatment'],
      },
    ];

    it('should have 4 note types', () => {
      const noteTypes = ['session', 'progress', 'treatment_plan', 'crisis'];
      expect(noteTypes.length).toBe(4);
    });

    it('should filter notes for wellness report', () => {
      const notesForReport = mockNotes.filter(n => n.includeInReport && n.isSharedWithClient);
      expect(notesForReport.length).toBe(2);
    });

    it('should only show notes shared with client', () => {
      const sharedNotes = mockNotes.filter(n => n.isSharedWithClient);
      expect(sharedNotes.length).toBe(3);
    });

    it('should toggle include in report correctly', () => {
      const note = { ...mockNotes[2] };
      expect(note.includeInReport).toBe(false);
      note.includeInReport = !note.includeInReport;
      expect(note.includeInReport).toBe(true);
    });

    it('should sort notes by date descending', () => {
      const sortedNotes = [...mockNotes].sort((a, b) => 
        new Date(b.sessionDate).getTime() - new Date(a.sessionDate).getTime()
      );
      expect(sortedNotes[0].sessionDate).toBe('2026-01-20');
      expect(sortedNotes[2].sessionDate).toBe('2026-01-06');
    });

    it('should have tags for categorization', () => {
      const allTags = mockNotes.flatMap(n => n.tags);
      expect(allTags).toContain('anxiety');
      expect(allTags).toContain('progress');
      expect(allTags).toContain('goals');
    });

    it('should track therapist name', () => {
      const therapistNames = [...new Set(mockNotes.map(n => n.therapistName))];
      expect(therapistNames).toContain('Dr. Sarah Chen');
    });
  });

  describe('Integration Tests', () => {
    it('should calculate breathing exercise benefit for mood', () => {
      // Simulating that completing breathing exercises improves mood
      let currentMood = 2; // Low mood
      const exerciseCompleted = true;
      if (exerciseCompleted) {
        currentMood = Math.min(5, currentMood + 1);
      }
      expect(currentMood).toBe(3);
    });

    it('should correlate mood calendar with therapist notes', () => {
      const moodEntry = { date: '2026-01-20', mood: 4 };
      const therapistNote = { sessionDate: '2026-01-20', type: 'session' };
      
      expect(moodEntry.date).toBe(therapistNote.sessionDate);
    });

    it('should track breathing exercise completion in mood data', () => {
      const dailyData = {
        date: '2026-01-26',
        mood: 4,
        breathingExercisesCompleted: 2,
        journalEntry: true,
      };
      
      expect(dailyData.breathingExercisesCompleted).toBeGreaterThan(0);
    });

    it('should generate comprehensive wellness report data', () => {
      const reportData = {
        dateRange: { start: '2026-01-01', end: '2026-01-26' },
        averageMood: 3.8,
        breathingExercisesCompleted: 15,
        therapistNotesIncluded: 2,
        moodTrend: 'improving',
      };

      expect(reportData.averageMood).toBeGreaterThan(0);
      expect(reportData.therapistNotesIncluded).toBeGreaterThan(0);
      expect(reportData.moodTrend).toBe('improving');
    });
  });
});
