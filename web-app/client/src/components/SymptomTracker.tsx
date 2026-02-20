import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Activity, 
  Brain, 
  Heart, 
  Thermometer, 
  Moon, 
  Utensils,
  Zap,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Plus,
  X,
  Check
} from 'lucide-react';

// Physical symptom categories with their icons and descriptions
const symptomCategories = {
  physical: {
    label: 'Physical',
    icon: Activity,
    symptoms: [
      { id: 'headache', name: 'Headache', description: 'Head pain or pressure' },
      { id: 'muscle_tension', name: 'Muscle Tension', description: 'Tight or sore muscles' },
      { id: 'fatigue', name: 'Fatigue', description: 'Feeling tired or exhausted' },
      { id: 'nausea', name: 'Nausea', description: 'Stomach upset or queasiness' },
      { id: 'dizziness', name: 'Dizziness', description: 'Lightheaded or unsteady' },
      { id: 'chest_tightness', name: 'Chest Tightness', description: 'Pressure or tightness in chest' },
      { id: 'trembling', name: 'Trembling', description: 'Shaking or tremors' },
      { id: 'sweating', name: 'Sweating', description: 'Excessive perspiration' }
    ]
  },
  cognitive: {
    label: 'Cognitive',
    icon: Brain,
    symptoms: [
      { id: 'brain_fog', name: 'Brain Fog', description: 'Difficulty thinking clearly' },
      { id: 'concentration', name: 'Poor Concentration', description: 'Hard to focus' },
      { id: 'memory', name: 'Memory Issues', description: 'Forgetfulness' },
      { id: 'racing_thoughts', name: 'Racing Thoughts', description: 'Thoughts moving too fast' },
      { id: 'confusion', name: 'Confusion', description: 'Feeling disoriented' },
      { id: 'intrusive_thoughts', name: 'Intrusive Thoughts', description: 'Unwanted thoughts' }
    ]
  },
  cardiovascular: {
    label: 'Heart',
    icon: Heart,
    symptoms: [
      { id: 'rapid_heartbeat', name: 'Rapid Heartbeat', description: 'Heart racing or pounding' },
      { id: 'palpitations', name: 'Palpitations', description: 'Irregular heartbeat' },
      { id: 'shortness_breath', name: 'Shortness of Breath', description: 'Difficulty breathing' }
    ]
  },
  sleep: {
    label: 'Sleep',
    icon: Moon,
    symptoms: [
      { id: 'insomnia', name: 'Insomnia', description: 'Difficulty falling asleep' },
      { id: 'hypersomnia', name: 'Hypersomnia', description: 'Sleeping too much' },
      { id: 'nightmares', name: 'Nightmares', description: 'Disturbing dreams' },
      { id: 'restless_sleep', name: 'Restless Sleep', description: 'Waking frequently' },
      { id: 'sleep_paralysis', name: 'Sleep Paralysis', description: 'Unable to move when waking' }
    ]
  },
  appetite: {
    label: 'Appetite',
    icon: Utensils,
    symptoms: [
      { id: 'loss_appetite', name: 'Loss of Appetite', description: 'Not wanting to eat' },
      { id: 'increased_appetite', name: 'Increased Appetite', description: 'Eating more than usual' },
      { id: 'cravings', name: 'Cravings', description: 'Strong urges for specific foods' },
      { id: 'digestive_issues', name: 'Digestive Issues', description: 'Stomach problems' }
    ]
  },
  energy: {
    label: 'Energy',
    icon: Zap,
    symptoms: [
      { id: 'low_energy', name: 'Low Energy', description: 'Feeling drained' },
      { id: 'restlessness', name: 'Restlessness', description: 'Unable to sit still' },
      { id: 'agitation', name: 'Agitation', description: 'Feeling on edge' },
      { id: 'lethargy', name: 'Lethargy', description: 'Sluggish and slow' }
    ]
  }
};

// Severity levels
const severityLevels = [
  { value: 1, label: 'Mild', color: 'bg-yellow-500' },
  { value: 2, label: 'Moderate', color: 'bg-orange-500' },
  { value: 3, label: 'Severe', color: 'bg-red-500' }
];

interface SymptomEntry {
  id: string;
  symptomId: string;
  severity: number;
  timestamp: string;
  mood?: number;
  notes?: string;
}

interface SymptomCorrelation {
  symptomId: string;
  symptomName: string;
  avgMoodWhenPresent: number;
  avgMoodWhenAbsent: number;
  correlation: 'positive' | 'negative' | 'neutral';
  occurrences: number;
}

export function SymptomTracker() {
  const [selectedSymptoms, setSelectedSymptoms] = useState<Map<string, number>>(new Map());
  const [symptomHistory, setSymptomHistory] = useState<SymptomEntry[]>([]);
  const [correlations, setCorrelations] = useState<SymptomCorrelation[]>([]);
  const [currentMood, setCurrentMood] = useState<number>(3);
  const [notes, setNotes] = useState('');
  const [activeTab, setActiveTab] = useState('log');
  const [showSuccess, setShowSuccess] = useState(false);
  
  useEffect(() => {
    // Load history from localStorage
    const saved = localStorage.getItem('reunity_symptom_history');
    if (saved) {
      const history = JSON.parse(saved);
      setSymptomHistory(history);
      calculateCorrelations(history);
    }
  }, []);
  
  const calculateCorrelations = (history: SymptomEntry[]) => {
    // Group entries by symptom
    const symptomData: Record<string, { moods: number[], count: number }> = {};
    const allMoods: number[] = [];
    
    history.forEach(entry => {
      if (entry.mood) {
        allMoods.push(entry.mood);
        if (!symptomData[entry.symptomId]) {
          symptomData[entry.symptomId] = { moods: [], count: 0 };
        }
        symptomData[entry.symptomId].moods.push(entry.mood);
        symptomData[entry.symptomId].count++;
      }
    });
    
    const avgOverallMood = allMoods.length > 0 
      ? allMoods.reduce((a, b) => a + b, 0) / allMoods.length 
      : 3;
    
    // Calculate correlations
    const correlationResults: SymptomCorrelation[] = [];
    
    Object.entries(symptomData).forEach(([symptomId, data]) => {
      if (data.count >= 3) { // Need at least 3 occurrences for meaningful correlation
        const avgMoodWhenPresent = data.moods.reduce((a, b) => a + b, 0) / data.moods.length;
        const avgMoodWhenAbsent = avgOverallMood; // Simplified
        
        // Find symptom name
        let symptomName = symptomId;
        Object.values(symptomCategories).forEach(cat => {
          const found = cat.symptoms.find(s => s.id === symptomId);
          if (found) symptomName = found.name;
        });
        
        const diff = avgMoodWhenPresent - avgMoodWhenAbsent;
        let correlation: 'positive' | 'negative' | 'neutral' = 'neutral';
        if (diff < -0.3) correlation = 'negative';
        else if (diff > 0.3) correlation = 'positive';
        
        correlationResults.push({
          symptomId,
          symptomName,
          avgMoodWhenPresent,
          avgMoodWhenAbsent,
          correlation,
          occurrences: data.count
        });
      }
    });
    
    // Sort by strongest correlation
    correlationResults.sort((a, b) => 
      Math.abs(a.avgMoodWhenPresent - a.avgMoodWhenAbsent) - 
      Math.abs(b.avgMoodWhenPresent - b.avgMoodWhenAbsent)
    ).reverse();
    
    setCorrelations(correlationResults);
  };
  
  const toggleSymptom = (symptomId: string, severity: number) => {
    const newSelected = new Map(selectedSymptoms);
    if (newSelected.has(symptomId) && newSelected.get(symptomId) === severity) {
      newSelected.delete(symptomId);
    } else {
      newSelected.set(symptomId, severity);
    }
    setSelectedSymptoms(newSelected);
  };
  
  const logSymptoms = () => {
    if (selectedSymptoms.size === 0) return;
    
    const timestamp = new Date().toISOString();
    const newEntries: SymptomEntry[] = [];
    
    selectedSymptoms.forEach((severity, symptomId) => {
      newEntries.push({
        id: `${symptomId}-${timestamp}`,
        symptomId,
        severity,
        timestamp,
        mood: currentMood,
        notes: notes || undefined
      });
    });
    
    const updatedHistory = [...symptomHistory, ...newEntries];
    setSymptomHistory(updatedHistory);
    localStorage.setItem('reunity_symptom_history', JSON.stringify(updatedHistory));
    
    // Recalculate correlations
    calculateCorrelations(updatedHistory);
    
    // Reset form
    setSelectedSymptoms(new Map());
    setNotes('');
    setShowSuccess(true);
    setTimeout(() => setShowSuccess(false), 3000);
  };
  
  const getRecentSymptoms = () => {
    const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    return symptomHistory.filter(entry => entry.timestamp > oneDayAgo);
  };
  
  const getMoodTrendIcon = (correlation: 'positive' | 'negative' | 'neutral') => {
    switch (correlation) {
      case 'negative': return <TrendingDown className="w-4 h-4 text-red-400" />;
      case 'positive': return <TrendingUp className="w-4 h-4 text-green-400" />;
      default: return <Minus className="w-4 h-4 text-gray-400" />;
    }
  };
  
  return (
    <Card className="bg-gradient-to-br from-purple-900/30 to-indigo-900/30 border-purple-700/50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-purple-200">
          <Thermometer className="w-5 h-5" />
          Symptom Tracker
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="log">Log</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="insights">Insights</TabsTrigger>
          </TabsList>
          
          <TabsContent value="log" className="space-y-4">
            {/* Mood Selection */}
            <div className="space-y-2">
              <label className="text-sm text-purple-300">Current Mood</label>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5].map(mood => (
                  <Button
                    key={mood}
                    variant={currentMood === mood ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setCurrentMood(mood)}
                    className={currentMood === mood 
                      ? 'bg-purple-600 hover:bg-purple-700' 
                      : 'border-purple-700 text-purple-300'
                    }
                  >
                    {mood === 1 ? '😢' : mood === 2 ? '😔' : mood === 3 ? '😐' : mood === 4 ? '🙂' : '😊'}
                  </Button>
                ))}
              </div>
            </div>
            
            {/* Symptom Categories */}
            <div className="space-y-3 max-h-[400px] overflow-y-auto">
              {Object.entries(symptomCategories).map(([key, category]) => {
                const CategoryIcon = category.icon;
                return (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center gap-2 text-purple-300">
                      <CategoryIcon className="w-4 h-4" />
                      <span className="text-sm font-medium">{category.label}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {category.symptoms.map(symptom => {
                        const isSelected = selectedSymptoms.has(symptom.id);
                        const severity = selectedSymptoms.get(symptom.id) || 0;
                        return (
                          <div 
                            key={symptom.id}
                            className={`p-2 rounded-lg border transition-all ${
                              isSelected 
                                ? 'border-purple-500 bg-purple-900/50' 
                                : 'border-purple-700/30 bg-purple-900/20 hover:border-purple-600'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-sm text-purple-200">{symptom.name}</span>
                              {isSelected && (
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => {
                                    const newSelected = new Map(selectedSymptoms);
                                    newSelected.delete(symptom.id);
                                    setSelectedSymptoms(newSelected);
                                  }}
                                  className="h-5 w-5 p-0 text-purple-400"
                                >
                                  <X className="w-3 h-3" />
                                </Button>
                              )}
                            </div>
                            <div className="flex gap-1">
                              {severityLevels.map(level => (
                                <Button
                                  key={level.value}
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => toggleSymptom(symptom.id, level.value)}
                                  className={`h-6 px-2 text-xs ${
                                    severity === level.value 
                                      ? `${level.color} text-white` 
                                      : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600/50'
                                  }`}
                                >
                                  {level.label}
                                </Button>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
            
            {/* Notes */}
            <div className="space-y-2">
              <label className="text-sm text-purple-300">Notes (optional)</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Any additional context..."
                className="w-full p-2 rounded-lg bg-purple-900/30 border border-purple-700/50 text-purple-100 placeholder-purple-400/50 text-sm resize-none"
                rows={2}
              />
            </div>
            
            {/* Submit */}
            <Button
              onClick={logSymptoms}
              disabled={selectedSymptoms.size === 0}
              className="w-full bg-purple-600 hover:bg-purple-700 disabled:opacity-50"
            >
              {showSuccess ? (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Logged Successfully
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-2" />
                  Log {selectedSymptoms.size} Symptom{selectedSymptoms.size !== 1 ? 's' : ''}
                </>
              )}
            </Button>
          </TabsContent>
          
          <TabsContent value="history" className="space-y-3">
            {getRecentSymptoms().length === 0 ? (
              <p className="text-center text-purple-400 py-4">
                No symptoms logged in the last 24 hours
              </p>
            ) : (
              <div className="space-y-2 max-h-[400px] overflow-y-auto">
                {getRecentSymptoms().reverse().map(entry => {
                  let symptomName = entry.symptomId;
                  Object.values(symptomCategories).forEach(cat => {
                    const found = cat.symptoms.find(s => s.id === entry.symptomId);
                    if (found) symptomName = found.name;
                  });
                  
                  const severity = severityLevels.find(s => s.value === entry.severity);
                  
                  return (
                    <div 
                      key={entry.id}
                      className="p-3 rounded-lg bg-purple-900/30 border border-purple-700/30"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-purple-200">{symptomName}</span>
                        <span className={`px-2 py-0.5 rounded text-xs text-white ${severity?.color || 'bg-gray-500'}`}>
                          {severity?.label || 'Unknown'}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 mt-1 text-xs text-purple-400">
                        <span>
                          {new Date(entry.timestamp).toLocaleString()}
                        </span>
                        {entry.mood && (
                          <span>Mood: {entry.mood}/5</span>
                        )}
                      </div>
                      {entry.notes && (
                        <p className="mt-1 text-xs text-purple-300 italic">
                          {entry.notes}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="insights" className="space-y-4">
            {correlations.length === 0 ? (
              <div className="text-center py-4">
                <AlertCircle className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                <p className="text-purple-300">
                  Log symptoms for at least 3 days to see mood correlations
                </p>
              </div>
            ) : (
              <>
                <p className="text-sm text-purple-300">
                  Symptoms that correlate with your mood patterns:
                </p>
                <div className="space-y-2">
                  {correlations.slice(0, 5).map(corr => (
                    <div 
                      key={corr.symptomId}
                      className="p-3 rounded-lg bg-purple-900/30 border border-purple-700/30"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-purple-200">{corr.symptomName}</span>
                        {getMoodTrendIcon(corr.correlation)}
                      </div>
                      <div className="mt-1 text-xs text-purple-400">
                        <span>
                          {corr.correlation === 'negative' 
                            ? 'Associated with lower mood' 
                            : corr.correlation === 'positive'
                            ? 'Associated with better mood'
                            : 'No clear mood impact'
                          }
                        </span>
                        <span className="ml-2">
                          ({corr.occurrences} occurrences)
                        </span>
                      </div>
                      <div className="mt-2 flex items-center gap-2 text-xs">
                        <span className="text-purple-400">Avg mood when present:</span>
                        <span className="text-purple-200">{corr.avgMoodWhenPresent.toFixed(1)}/5</span>
                      </div>
                    </div>
                  ))}
                </div>
                
                {correlations.some(c => c.correlation === 'negative') && (
                  <div className="p-3 rounded-lg bg-amber-900/30 border border-amber-700/50">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5" />
                      <div>
                        <p className="text-sm text-amber-200">
                          Tracking tip
                        </p>
                        <p className="text-xs text-amber-300 mt-1">
                          Some symptoms correlate with lower mood. Consider discussing these patterns with a healthcare provider.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default SymptomTracker;
