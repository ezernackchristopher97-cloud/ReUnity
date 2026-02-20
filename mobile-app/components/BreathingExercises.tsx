import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, Dimensions } from 'react-native';

interface BreathingExercise {
  id: string;
  name: string;
  description: string;
  pattern: { inhale: number; hold1: number; exhale: number; hold2: number };
  cycles: number;
  benefits: string[];
}

const exercises: BreathingExercise[] = [
  {
    id: 'box',
    name: 'Box Breathing',
    description: 'Equal parts inhale, hold, exhale, hold. Used by Navy SEALs.',
    pattern: { inhale: 4, hold1: 4, exhale: 4, hold2: 4 },
    cycles: 4,
    benefits: ['Reduces stress', 'Improves focus', 'Calms nervous system'],
  },
  {
    id: '478',
    name: '4-7-8 Breathing',
    description: 'Relaxing breath technique for sleep and anxiety.',
    pattern: { inhale: 4, hold1: 7, exhale: 8, hold2: 0 },
    cycles: 4,
    benefits: ['Promotes sleep', 'Reduces anxiety', 'Lowers heart rate'],
  },
  {
    id: 'calm',
    name: 'Calming Breath',
    description: 'Simple technique for immediate stress relief.',
    pattern: { inhale: 4, hold1: 2, exhale: 6, hold2: 0 },
    cycles: 6,
    benefits: ['Quick relief', 'Easy to remember', 'Activates relaxation'],
  },
];

export default function BreathingExercises() {
  const [selectedExercise, setSelectedExercise] = useState<BreathingExercise | null>(null);
  const [isActive, setIsActive] = useState(false);
  const [phase, setPhase] = useState<'inhale' | 'hold1' | 'exhale' | 'hold2'>('inhale');
  const [currentCycle, setCurrentCycle] = useState(1);
  const [countdown, setCountdown] = useState(0);
  const scaleAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (!isActive || !selectedExercise) return;

    const pattern = selectedExercise.pattern;
    let timer: NodeJS.Timeout;

    const runPhase = (currentPhase: typeof phase, duration: number) => {
      setPhase(currentPhase);
      setCountdown(duration);

      // Animate circle
      const targetScale = currentPhase === 'inhale' ? 1.5 : currentPhase === 'exhale' ? 1 : scaleAnim._value;
      Animated.timing(scaleAnim, {
        toValue: targetScale,
        duration: duration * 1000,
        useNativeDriver: true,
      }).start();

      let remaining = duration;
      timer = setInterval(() => {
        remaining--;
        setCountdown(remaining);
        if (remaining <= 0) {
          clearInterval(timer);
        }
      }, 1000);
    };

    const sequence = async () => {
      if (pattern.inhale > 0) {
        runPhase('inhale', pattern.inhale);
        await new Promise(r => setTimeout(r, pattern.inhale * 1000));
      }
      if (pattern.hold1 > 0) {
        runPhase('hold1', pattern.hold1);
        await new Promise(r => setTimeout(r, pattern.hold1 * 1000));
      }
      if (pattern.exhale > 0) {
        runPhase('exhale', pattern.exhale);
        await new Promise(r => setTimeout(r, pattern.exhale * 1000));
      }
      if (pattern.hold2 > 0) {
        runPhase('hold2', pattern.hold2);
        await new Promise(r => setTimeout(r, pattern.hold2 * 1000));
      }

      if (currentCycle < selectedExercise.cycles) {
        setCurrentCycle(c => c + 1);
      } else {
        setIsActive(false);
        setCurrentCycle(1);
      }
    };

    sequence();

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isActive, currentCycle, selectedExercise]);

  const startExercise = (exercise: BreathingExercise) => {
    setSelectedExercise(exercise);
    setIsActive(true);
    setCurrentCycle(1);
    scaleAnim.setValue(1);
  };

  const stopExercise = () => {
    setIsActive(false);
    setCurrentCycle(1);
    scaleAnim.setValue(1);
  };

  const getPhaseLabel = () => {
    switch (phase) {
      case 'inhale': return 'Breathe In';
      case 'hold1': return 'Hold';
      case 'exhale': return 'Breathe Out';
      case 'hold2': return 'Hold';
    }
  };

  if (isActive && selectedExercise) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>{selectedExercise.name}</Text>
        <Text style={styles.cycleText}>Cycle {currentCycle} of {selectedExercise.cycles}</Text>

        <View style={styles.circleContainer}>
          <Animated.View style={[styles.breathCircle, { transform: [{ scale: scaleAnim }] }]}>
            <Text style={styles.phaseLabel}>{getPhaseLabel()}</Text>
            <Text style={styles.countdown}>{countdown}</Text>
          </Animated.View>
        </View>

        <TouchableOpacity style={styles.stopButton} onPress={stopExercise}>
          <Text style={styles.stopButtonText}>Stop</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Breathing Exercises</Text>
      <Text style={styles.subtitle}>Guided breathing for calm and focus</Text>

      {exercises.map(exercise => (
        <TouchableOpacity
          key={exercise.id}
          style={styles.exerciseCard}
          onPress={() => startExercise(exercise)}
        >
          <Text style={styles.exerciseName}>{exercise.name}</Text>
          <Text style={styles.exerciseDesc}>{exercise.description}</Text>
          <View style={styles.patternRow}>
            <Text style={styles.patternText}>
              {exercise.pattern.inhale}s in • {exercise.pattern.hold1}s hold • {exercise.pattern.exhale}s out
              {exercise.pattern.hold2 > 0 ? ` • ${exercise.pattern.hold2}s hold` : ''}
            </Text>
          </View>
          <View style={styles.benefitsRow}>
            {exercise.benefits.map((benefit, i) => (
              <View key={i} style={styles.benefitTag}>
                <Text style={styles.benefitText}>{benefit}</Text>
              </View>
            ))}
          </View>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    padding: 20,
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 24,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
  },
  cycleText: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginTop: 8,
  },
  circleContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  breathCircle: {
    width: width * 0.6,
    height: width * 0.6,
    borderRadius: width * 0.3,
    backgroundColor: '#10b981',
    justifyContent: 'center',
    alignItems: 'center',
  },
  phaseLabel: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  countdown: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  stopButton: {
    backgroundColor: '#ef4444',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
  },
  stopButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  exerciseCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  exerciseName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 4,
  },
  exerciseDesc: {
    fontSize: 14,
    color: '#888',
    marginBottom: 12,
  },
  patternRow: {
    marginBottom: 12,
  },
  patternText: {
    fontSize: 12,
    color: '#10b981',
  },
  benefitsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  benefitTag: {
    backgroundColor: '#10b98120',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  benefitText: {
    fontSize: 12,
    color: '#10b981',
  },
});
