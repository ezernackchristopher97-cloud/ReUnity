import { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Wind, Play, Pause, RotateCcw, Check } from 'lucide-react';

type BreathingTechnique = 'box' | '478' | 'calm' | 'energize';

interface TechniqueConfig {
  name: string;
  description: string;
  phases: { name: string; duration: number; instruction: string }[];
  color: string;
  benefits: string[];
}

const techniques: Record<BreathingTechnique, TechniqueConfig> = {
  box: {
    name: 'Box Breathing',
    description: 'Equal duration inhale, hold, exhale, hold. Used by Navy SEALs for stress management.',
    phases: [
      { name: 'Inhale', duration: 4, instruction: 'Breathe in slowly through your nose' },
      { name: 'Hold', duration: 4, instruction: 'Hold your breath gently' },
      { name: 'Exhale', duration: 4, instruction: 'Breathe out slowly through your mouth' },
      { name: 'Hold', duration: 4, instruction: 'Hold before the next breath' },
    ],
    color: '#10b981',
    benefits: ['Reduces stress', 'Improves focus', 'Calms anxiety', 'Enhances clarity'],
  },
  '478': {
    name: '4-7-8 Breathing',
    description: 'Dr. Andrew Weil\'s relaxation technique. Excellent for sleep and anxiety.',
    phases: [
      { name: 'Inhale', duration: 4, instruction: 'Breathe in quietly through your nose' },
      { name: 'Hold', duration: 7, instruction: 'Hold your breath' },
      { name: 'Exhale', duration: 8, instruction: 'Exhale completely through your mouth' },
    ],
    color: '#6366f1',
    benefits: ['Promotes sleep', 'Reduces anxiety', 'Manages cravings', 'Controls anger'],
  },
  calm: {
    name: 'Calming Breath',
    description: 'Extended exhale activates parasympathetic nervous system for deep relaxation.',
    phases: [
      { name: 'Inhale', duration: 4, instruction: 'Breathe in through your nose' },
      { name: 'Exhale', duration: 6, instruction: 'Slowly exhale, longer than inhale' },
    ],
    color: '#8b5cf6',
    benefits: ['Deep relaxation', 'Slows heart rate', 'Reduces tension', 'Promotes calm'],
  },
  energize: {
    name: 'Energizing Breath',
    description: 'Quick, rhythmic breathing to increase alertness and energy.',
    phases: [
      { name: 'Inhale', duration: 2, instruction: 'Quick breath in through nose' },
      { name: 'Exhale', duration: 2, instruction: 'Quick breath out through mouth' },
    ],
    color: '#f59e0b',
    benefits: ['Increases energy', 'Boosts alertness', 'Clears mind', 'Improves mood'],
  },
};

export function BreathingExercises({ compact = false }: { compact?: boolean }) {
  const [selectedTechnique, setSelectedTechnique] = useState<BreathingTechnique>('box');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentPhaseIndex, setCurrentPhaseIndex] = useState(0);
  const [phaseProgress, setPhaseProgress] = useState(0);
  const [cyclesCompleted, setCyclesCompleted] = useState(0);
  const [totalCycles, setTotalCycles] = useState(4);
  const [showCompletion, setShowCompletion] = useState(false);
  const animationRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);

  const technique = techniques[selectedTechnique];
  const currentPhase = technique.phases[currentPhaseIndex];
  const totalPhaseDuration = currentPhase.duration * 1000;

  useEffect(() => {
    if (!isPlaying) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    startTimeRef.current = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTimeRef.current;
      const progress = Math.min(elapsed / totalPhaseDuration, 1);
      setPhaseProgress(progress);

      if (progress >= 1) {
        const nextPhaseIndex = (currentPhaseIndex + 1) % technique.phases.length;
        
        if (nextPhaseIndex === 0) {
          const newCycles = cyclesCompleted + 1;
          setCyclesCompleted(newCycles);
          
          if (newCycles >= totalCycles) {
            setIsPlaying(false);
            setShowCompletion(true);
            setTimeout(() => setShowCompletion(false), 3000);
            return;
          }
        }
        
        setCurrentPhaseIndex(nextPhaseIndex);
        startTimeRef.current = currentTime;
        setPhaseProgress(0);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, currentPhaseIndex, totalPhaseDuration, cyclesCompleted, totalCycles, technique.phases.length]);

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentPhaseIndex(0);
    setPhaseProgress(0);
    setCyclesCompleted(0);
    setShowCompletion(false);
  };

  const handleTechniqueChange = (tech: BreathingTechnique) => {
    handleReset();
    setSelectedTechnique(tech);
  };

  const getCircleScale = () => {
    const phase = currentPhase.name.toLowerCase();
    if (phase === 'inhale') {
      return 0.5 + (phaseProgress * 0.5);
    } else if (phase === 'exhale') {
      return 1 - (phaseProgress * 0.5);
    }
    return phase === 'hold' && currentPhaseIndex > 0 ? 1 : 0.5;
  };

  const circleScale = getCircleScale();

  if (compact) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div 
              className="w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300"
              style={{ 
                backgroundColor: `${technique.color}20`,
                transform: `scale(${isPlaying ? circleScale : 1})`,
              }}
            >
              <Wind className="w-6 h-6" style={{ color: technique.color }} />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-white">Breathing Exercises</p>
              <p className="text-xs text-slate-400">
                {isPlaying ? `${currentPhase.name} - ${Math.ceil(currentPhase.duration * (1 - phaseProgress))}s` : 'Tap to start'}
              </p>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setIsPlaying(!isPlaying)}
              className="text-emerald-400"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
          <Wind className="w-5 h-5 text-emerald-400" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Breathing Exercises</h2>
          <p className="text-sm text-slate-400">Guided breathing for calm and focus</p>
        </div>
      </div>

      {/* Technique Selection */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {(Object.keys(techniques) as BreathingTechnique[]).map((tech) => (
          <button
            key={tech}
            onClick={() => handleTechniqueChange(tech)}
            className={`p-3 rounded-lg border transition-all ${
              selectedTechnique === tech
                ? 'border-emerald-500 bg-emerald-500/10'
                : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
            }`}
          >
            <p className="text-sm font-medium text-white">{techniques[tech].name}</p>
          </button>
        ))}
      </div>

      {/* Animation Circle */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="p-8">
          <div className="flex flex-col items-center">
            {/* Main breathing circle */}
            <div className="relative w-64 h-64 flex items-center justify-center">
              {/* Outer ring */}
              <div 
                className="absolute inset-0 rounded-full border-4 opacity-20"
                style={{ borderColor: technique.color }}
              />
              
              {/* Progress ring */}
              <svg className="absolute inset-0 w-full h-full -rotate-90">
                <circle
                  cx="128"
                  cy="128"
                  r="124"
                  fill="none"
                  stroke={technique.color}
                  strokeWidth="4"
                  strokeDasharray={`${phaseProgress * 779} 779`}
                  className="transition-all duration-100"
                  opacity="0.5"
                />
              </svg>

              {/* Animated breathing circle */}
              <div
                className="rounded-full transition-all duration-300 ease-in-out flex items-center justify-center"
                style={{
                  width: `${circleScale * 200}px`,
                  height: `${circleScale * 200}px`,
                  backgroundColor: `${technique.color}30`,
                  boxShadow: isPlaying ? `0 0 60px ${technique.color}40` : 'none',
                }}
              >
                <div
                  className="rounded-full flex items-center justify-center"
                  style={{
                    width: `${circleScale * 150}px`,
                    height: `${circleScale * 150}px`,
                    backgroundColor: `${technique.color}50`,
                  }}
                >
                  {showCompletion ? (
                    <Check className="w-16 h-16 text-white" />
                  ) : (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-white">{currentPhase.name}</p>
                      <p className="text-3xl font-mono text-white/80">
                        {Math.ceil(currentPhase.duration * (1 - phaseProgress))}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Instruction */}
            <p className="mt-6 text-lg text-slate-300 text-center">
              {showCompletion ? 'Great job! Exercise complete.' : currentPhase.instruction}
            </p>

            {/* Cycle counter */}
            <p className="mt-2 text-sm text-slate-400">
              Cycle {cyclesCompleted + 1} of {totalCycles}
            </p>

            {/* Controls */}
            <div className="flex items-center gap-4 mt-6">
              <Button
                variant="outline"
                size="lg"
                onClick={handleReset}
                className="border-slate-600"
              >
                <RotateCcw className="w-5 h-5 mr-2" />
                Reset
              </Button>
              <Button
                size="lg"
                onClick={() => setIsPlaying(!isPlaying)}
                style={{ backgroundColor: technique.color }}
                className="text-white px-8"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-5 h-5 mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" />
                    Start
                  </>
                )}
              </Button>
            </div>

            {/* Cycle selector */}
            <div className="flex items-center gap-2 mt-4">
              <span className="text-sm text-slate-400">Cycles:</span>
              {[2, 4, 6, 8].map((num) => (
                <button
                  key={num}
                  onClick={() => setTotalCycles(num)}
                  className={`w-8 h-8 rounded-full text-sm ${
                    totalCycles === num
                      ? 'bg-emerald-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {num}
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technique Info */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white">{technique.name}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-slate-300">{technique.description}</p>
          
          <div>
            <p className="text-sm font-medium text-slate-400 mb-2">Pattern:</p>
            <div className="flex flex-wrap gap-2">
              {technique.phases.map((phase, i) => (
                <span
                  key={i}
                  className="px-3 py-1 rounded-full text-sm"
                  style={{ 
                    backgroundColor: `${technique.color}20`,
                    color: technique.color,
                  }}
                >
                  {phase.name}: {phase.duration}s
                </span>
              ))}
            </div>
          </div>

          <div>
            <p className="text-sm font-medium text-slate-400 mb-2">Benefits:</p>
            <div className="flex flex-wrap gap-2">
              {technique.benefits.map((benefit, i) => (
                <span
                  key={i}
                  className="px-3 py-1 rounded-full bg-slate-700 text-slate-300 text-sm"
                >
                  {benefit}
                </span>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
