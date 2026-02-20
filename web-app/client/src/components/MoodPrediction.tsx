import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Activity, 
  Heart, 
  Moon,
  Zap,
  Shield,
  Bell,
  BellOff,
  ChevronRight,
  Info
} from 'lucide-react';

interface HRVDataPoint {
  timestamp: number;
  hrv: number;
  heartRate: number;
  sleepQuality?: number;
  stressLevel?: number;
}

interface PredictionResult {
  riskLevel: 'low' | 'moderate' | 'elevated' | 'high';
  confidence: number;
  predictedTimeframe: string;
  factors: Array<{
    name: string;
    impact: 'positive' | 'negative' | 'neutral';
    value: number;
    description: string;
  }>;
  recommendations: string[];
  trend: 'improving' | 'stable' | 'declining';
}

interface MoodPredictionProps {
  onAlertTriggered?: (prediction: PredictionResult) => void;
  onHighRiskDetected?: (riskLevel: string) => void;
}

// Simulated HRV data for demonstration
const generateMockHRVData = (): HRVDataPoint[] => {
  const data: HRVDataPoint[] = [];
  const now = Date.now();
  
  for (let i = 0; i < 7; i++) {
    const dayOffset = (6 - i) * 24 * 60 * 60 * 1000;
    // Generate 4 readings per day
    for (let j = 0; j < 4; j++) {
      const hourOffset = j * 6 * 60 * 60 * 1000;
      const baseHRV = 45 + Math.random() * 30;
      const trend = i < 3 ? -2 : 2; // Declining then improving
      
      data.push({
        timestamp: now - dayOffset + hourOffset,
        hrv: baseHRV + trend * (6 - i) + (Math.random() - 0.5) * 10,
        heartRate: 65 + Math.random() * 20,
        sleepQuality: 60 + Math.random() * 30,
        stressLevel: 30 + Math.random() * 40,
      });
    }
  }
  
  return data;
};

// HRV-based mood prediction algorithm
const predictMood = (data: HRVDataPoint[]): PredictionResult => {
  if (data.length < 4) {
    return {
      riskLevel: 'low',
      confidence: 0,
      predictedTimeframe: 'Insufficient data',
      factors: [],
      recommendations: ['Continue wearing your device to collect more data'],
      trend: 'stable',
    };
  }

  // Calculate recent vs historical averages
  const recentData = data.slice(-8); // Last 2 days
  const historicalData = data.slice(0, -8);
  
  const recentAvgHRV = recentData.reduce((sum, d) => sum + d.hrv, 0) / recentData.length;
  const historicalAvgHRV = historicalData.length > 0 
    ? historicalData.reduce((sum, d) => sum + d.hrv, 0) / historicalData.length 
    : recentAvgHRV;
  
  const recentAvgHR = recentData.reduce((sum, d) => sum + d.heartRate, 0) / recentData.length;
  const recentAvgSleep = recentData.reduce((sum, d) => sum + (d.sleepQuality || 70), 0) / recentData.length;
  const recentAvgStress = recentData.reduce((sum, d) => sum + (d.stressLevel || 40), 0) / recentData.length;
  
  // Calculate HRV trend (positive = improving)
  const hrvChange = recentAvgHRV - historicalAvgHRV;
  const hrvChangePercent = (hrvChange / historicalAvgHRV) * 100;
  
  // Determine trend
  let trend: 'improving' | 'stable' | 'declining';
  if (hrvChangePercent > 5) trend = 'improving';
  else if (hrvChangePercent < -5) trend = 'declining';
  else trend = 'stable';
  
  // Calculate risk factors
  const factors: PredictionResult['factors'] = [];
  
  // HRV factor
  factors.push({
    name: 'Heart Rate Variability',
    impact: recentAvgHRV > 50 ? 'positive' : recentAvgHRV > 35 ? 'neutral' : 'negative',
    value: Math.round(recentAvgHRV),
    description: recentAvgHRV > 50 
      ? 'Good autonomic nervous system balance' 
      : recentAvgHRV > 35 
        ? 'Moderate stress response' 
        : 'Elevated stress indicators',
  });
  
  // Sleep factor
  factors.push({
    name: 'Sleep Quality',
    impact: recentAvgSleep > 70 ? 'positive' : recentAvgSleep > 50 ? 'neutral' : 'negative',
    value: Math.round(recentAvgSleep),
    description: recentAvgSleep > 70 
      ? 'Restorative sleep patterns' 
      : recentAvgSleep > 50 
        ? 'Adequate but could improve' 
        : 'Poor sleep affecting recovery',
  });
  
  // Resting heart rate factor
  factors.push({
    name: 'Resting Heart Rate',
    impact: recentAvgHR < 70 ? 'positive' : recentAvgHR < 85 ? 'neutral' : 'negative',
    value: Math.round(recentAvgHR),
    description: recentAvgHR < 70 
      ? 'Healthy cardiovascular state' 
      : recentAvgHR < 85 
        ? 'Normal range' 
        : 'Elevated - may indicate stress',
  });
  
  // Stress level factor
  factors.push({
    name: 'Stress Level',
    impact: recentAvgStress < 40 ? 'positive' : recentAvgStress < 60 ? 'neutral' : 'negative',
    value: Math.round(recentAvgStress),
    description: recentAvgStress < 40 
      ? 'Well-managed stress' 
      : recentAvgStress < 60 
        ? 'Moderate stress levels' 
        : 'High stress - intervention recommended',
  });
  
  // Calculate overall risk level
  const negativeFactors = factors.filter(f => f.impact === 'negative').length;
  const positiveFactors = factors.filter(f => f.impact === 'positive').length;
  
  let riskLevel: PredictionResult['riskLevel'];
  let confidence: number;
  
  if (negativeFactors >= 3 || (negativeFactors >= 2 && trend === 'declining')) {
    riskLevel = 'high';
    confidence = 75 + Math.random() * 15;
  } else if (negativeFactors >= 2 || (negativeFactors >= 1 && trend === 'declining')) {
    riskLevel = 'elevated';
    confidence = 65 + Math.random() * 20;
  } else if (negativeFactors >= 1 || trend === 'declining') {
    riskLevel = 'moderate';
    confidence = 55 + Math.random() * 25;
  } else {
    riskLevel = 'low';
    confidence = 70 + Math.random() * 20;
  }
  
  // Generate recommendations based on factors
  const recommendations: string[] = [];
  
  if (recentAvgHRV < 40) {
    recommendations.push('Practice deep breathing exercises to improve HRV');
    recommendations.push('Consider a grounding technique when feeling overwhelmed');
  }
  if (recentAvgSleep < 60) {
    recommendations.push('Prioritize sleep hygiene - consistent bedtime routine');
    recommendations.push('Limit screen time 1 hour before bed');
  }
  if (recentAvgStress > 50) {
    recommendations.push('Schedule short breaks throughout the day');
    recommendations.push('Try the 5-4-3-2-1 grounding technique');
  }
  if (trend === 'declining') {
    recommendations.push('Reach out to your support network');
    recommendations.push('Consider scheduling a check-in with your therapist');
  }
  
  if (recommendations.length === 0) {
    recommendations.push('Continue your current wellness practices');
    recommendations.push('Your metrics look healthy - keep it up!');
  }
  
  // Predicted timeframe
  const timeframes: Record<string, string> = {
    high: 'Next 12-24 hours',
    elevated: 'Next 24-48 hours',
    moderate: 'Next 2-3 days',
    low: 'No immediate concerns',
  };
  
  return {
    riskLevel,
    confidence: Math.round(confidence),
    predictedTimeframe: timeframes[riskLevel],
    factors,
    recommendations,
    trend,
  };
};

export default function MoodPrediction({ onAlertTriggered, onHighRiskDetected }: MoodPredictionProps) {
  const [hrvData, setHrvData] = useState<HRVDataPoint[]>([]);
  const [alertsEnabled, setAlertsEnabled] = useState(true);
  const [showDetails, setShowDetails] = useState(false);
  
  // Load HRV data from localStorage or generate mock data
  useEffect(() => {
    const stored = localStorage.getItem('reunity_hrv_history');
    if (stored) {
      setHrvData(JSON.parse(stored));
    } else {
      const mockData = generateMockHRVData();
      setHrvData(mockData);
      localStorage.setItem('reunity_hrv_history', JSON.stringify(mockData));
    }
    
    // Load alert preference
    const alertPref = localStorage.getItem('reunity_mood_alerts');
    if (alertPref !== null) {
      setAlertsEnabled(JSON.parse(alertPref));
    }
  }, []);
  
  const prediction = useMemo(() => predictMood(hrvData), [hrvData]);
  
  // Trigger alert callback when risk is elevated
  useEffect(() => {
    if (alertsEnabled && (prediction.riskLevel === 'high' || prediction.riskLevel === 'elevated')) {
      onAlertTriggered?.(prediction);
    }
    // Trigger emergency contact dialog for high risk
    if (prediction.riskLevel === 'high') {
      onHighRiskDetected?.(prediction.riskLevel);
    }
  }, [prediction, alertsEnabled, onAlertTriggered, onHighRiskDetected]);
  
  const toggleAlerts = () => {
    const newValue = !alertsEnabled;
    setAlertsEnabled(newValue);
    localStorage.setItem('reunity_mood_alerts', JSON.stringify(newValue));
  };
  
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-emerald-400';
      case 'moderate': return 'text-yellow-400';
      case 'elevated': return 'text-orange-400';
      case 'high': return 'text-red-400';
      default: return 'text-zinc-400';
    }
  };
  
  const getRiskBgColor = (level: string) => {
    switch (level) {
      case 'low': return 'bg-emerald-500/20 border-emerald-500/30';
      case 'moderate': return 'bg-yellow-500/20 border-yellow-500/30';
      case 'elevated': return 'bg-orange-500/20 border-orange-500/30';
      case 'high': return 'bg-red-500/20 border-red-500/30';
      default: return 'bg-zinc-500/20 border-zinc-500/30';
    }
  };
  
  const getTrendIcon = () => {
    switch (prediction.trend) {
      case 'improving': return <TrendingUp className="w-5 h-5 text-emerald-400" />;
      case 'declining': return <TrendingDown className="w-5 h-5 text-red-400" />;
      default: return <Activity className="w-5 h-5 text-zinc-400" />;
    }
  };
  
  const getImpactIcon = (impact: string) => {
    switch (impact) {
      case 'positive': return <div className="w-2 h-2 rounded-full bg-emerald-400" />;
      case 'negative': return <div className="w-2 h-2 rounded-full bg-red-400" />;
      default: return <div className="w-2 h-2 rounded-full bg-zinc-400" />;
    }
  };

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-purple-400" />
            <CardTitle className="text-lg text-white">Mood Prediction</CardTitle>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleAlerts}
            className={alertsEnabled ? 'text-emerald-400' : 'text-zinc-500'}
          >
            {alertsEnabled ? <Bell className="w-5 h-5" /> : <BellOff className="w-5 h-5" />}
          </Button>
        </div>
        <CardDescription className="text-zinc-400">
          AI-powered wellness prediction based on your HRV trends
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Risk Level Display */}
        <div className={`p-4 rounded-lg border ${getRiskBgColor(prediction.riskLevel)}`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {prediction.riskLevel === 'high' || prediction.riskLevel === 'elevated' ? (
                <AlertTriangle className={`w-5 h-5 ${getRiskColor(prediction.riskLevel)}`} />
              ) : (
                <Shield className={`w-5 h-5 ${getRiskColor(prediction.riskLevel)}`} />
              )}
              <span className={`font-semibold capitalize ${getRiskColor(prediction.riskLevel)}`}>
                {prediction.riskLevel} Risk
              </span>
            </div>
            <div className="flex items-center gap-2">
              {getTrendIcon()}
              <span className="text-sm text-zinc-400 capitalize">{prediction.trend}</span>
            </div>
          </div>
          
          <div className="flex items-center gap-2 mb-2">
            <Progress 
              value={prediction.confidence} 
              className="h-2 flex-1"
            />
            <span className="text-xs text-zinc-400">{prediction.confidence}% confidence</span>
          </div>
          
          <p className="text-sm text-zinc-300">
            <span className="text-zinc-500">Timeframe:</span> {prediction.predictedTimeframe}
          </p>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-2 gap-3">
          {prediction.factors.slice(0, 2).map((factor) => (
            <div key={factor.name} className="p-3 bg-zinc-800/50 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                {factor.name === 'Heart Rate Variability' && <Heart className="w-4 h-4 text-red-400" />}
                {factor.name === 'Sleep Quality' && <Moon className="w-4 h-4 text-blue-400" />}
                {factor.name === 'Resting Heart Rate' && <Activity className="w-4 h-4 text-pink-400" />}
                {factor.name === 'Stress Level' && <Zap className="w-4 h-4 text-yellow-400" />}
                <span className="text-xs text-zinc-400">{factor.name}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-lg font-semibold text-white">
                  {factor.value}{factor.name.includes('Rate') ? ' bpm' : factor.name.includes('HRV') ? ' ms' : '%'}
                </span>
                {getImpactIcon(factor.impact)}
              </div>
            </div>
          ))}
        </div>
        
        {/* Expand/Collapse Details */}
        <Button
          variant="ghost"
          className="w-full justify-between text-zinc-400 hover:text-white"
          onClick={() => setShowDetails(!showDetails)}
        >
          <span className="flex items-center gap-2">
            <Info className="w-4 h-4" />
            {showDetails ? 'Hide Details' : 'Show Details & Recommendations'}
          </span>
          <ChevronRight className={`w-4 h-4 transition-transform ${showDetails ? 'rotate-90' : ''}`} />
        </Button>
        
        {showDetails && (
          <div className="space-y-4 pt-2">
            {/* All Factors */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-zinc-300">Contributing Factors</h4>
              {prediction.factors.map((factor) => (
                <div key={factor.name} className="p-3 bg-zinc-800/30 rounded-lg">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-white">{factor.name}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-zinc-300">
                        {factor.value}{factor.name.includes('Rate') ? ' bpm' : factor.name.includes('HRV') ? ' ms' : '%'}
                      </span>
                      {getImpactIcon(factor.impact)}
                    </div>
                  </div>
                  <p className="text-xs text-zinc-500">{factor.description}</p>
                </div>
              ))}
            </div>
            
            {/* Recommendations */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-zinc-300">Recommendations</h4>
              <div className="space-y-2">
                {prediction.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start gap-2 p-2 bg-emerald-500/10 rounded-lg">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 mt-1.5" />
                    <span className="text-sm text-emerald-200">{rec}</span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Data Info */}
            <div className="text-xs text-zinc-500 flex items-center gap-1">
              <Info className="w-3 h-3" />
              Based on {hrvData.length} data points over the last 7 days
            </div>
          </div>
        )}
        
        {/* High Risk Alert */}
        {prediction.riskLevel === 'high' && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5" />
              <div>
                <p className="font-medium text-red-300">Proactive Alert</p>
                <p className="text-sm text-red-200/80 mt-1">
                  Your biometric data suggests elevated stress. Consider reaching out to your 
                  support network or using a grounding technique. If you're in crisis, 
                  please contact 988.
                </p>
                <div className="flex gap-2 mt-3">
                  <Button size="sm" variant="outline" className="border-red-500/50 text-red-400">
                    View Grounding Techniques
                  </Button>
                  <Button size="sm" className="bg-red-600 hover:bg-red-700">
                    Contact Support
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Export prediction function for use in other components
export { predictMood, type PredictionResult, type HRVDataPoint };
