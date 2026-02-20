/**
 * Vicsek Flocking Model for Emotional Trajectory Prediction
 * 
 * Based on the Vicsek model of collective motion, this module predicts
 * emotional trajectory based on neighboring factors (past states, context,
 * detected patterns). Like birds in a flock, emotional states tend to
 * align with their "neighbors" - recent history and environmental factors.
 * 
 * Key concepts:
 * - Each emotional state is a "particle" with direction (trajectory)
 * - Particles align with neighbors within a radius (context window)
 * - Noise represents unpredictability/external factors
 * - Emergent behavior predicts likely next state
 */

export interface EmotionalParticle {
  state: string;
  intensity: number;  // 0-1
  direction: number;  // -1 (deteriorating) to 1 (improving)
  timestamp: number;
}

export interface VicsekPrediction {
  predictedTrajectory: "improving" | "stable" | "deteriorating" | "crisis_imminent";
  confidence: number;
  alignmentStrength: number;  // How strongly states are aligning
  noiseLevel: number;  // Unpredictability in the system
  recommendedIntervention: string;
  urgency: "low" | "medium" | "high" | "critical";
}

export interface FlockState {
  particles: EmotionalParticle[];
  averageDirection: number;
  orderParameter: number;  // 0 = chaos, 1 = perfect alignment
}

/**
 * Vicsek Model Implementation for Emotional State Prediction
 */
export class VicsekEmotionalModel {
  private particles: EmotionalParticle[] = [];
  private readonly maxParticles = 20;  // Rolling window of states
  private readonly alignmentRadius = 0.3;  // How far back to look for alignment
  private readonly noiseAmplitude = 0.1;  // Base noise level
  
  // State severity mapping (higher = more severe)
  private readonly stateSeverity: Record<string, number> = {
    // Crisis states
    "suicidal_ideation": 1.0,
    "active_suicidal_plan": 1.0,
    "self_harm_urge": 0.95,
    "psychotic_episode": 0.95,
    
    // High distress
    "panic": 0.85,
    "flashback": 0.85,
    "dissociation": 0.8,
    "depersonalization": 0.8,
    "derealization": 0.8,
    "splitting": 0.8,
    
    // Moderate distress
    "overwhelm": 0.7,
    "trauma_activation": 0.7,
    "hypervigilance": 0.65,
    "anxiety": 0.6,
    "depression": 0.6,
    "shame": 0.6,
    "anger": 0.55,
    
    // Lower distress
    "loneliness": 0.5,
    "emotional_numbness": 0.45,
    "identity_confusion": 0.4,
    
    // Neutral/positive
    "stable": 0.2,
    "grounded": 0.1,
    "calm": 0.05
  };
  
  /**
   * Add a new emotional state observation
   */
  addObservation(states: string[], entropy: number): void {
    // Calculate average severity of current states
    let totalSeverity = 0;
    let count = 0;
    
    for (const state of states) {
      const severity = this.stateSeverity[state] ?? 0.5;
      totalSeverity += severity;
      count++;
    }
    
    const avgSeverity = count > 0 ? totalSeverity / count : 0.5;
    
    // Calculate direction based on comparison with recent history
    let direction = 0;
    if (this.particles.length > 0) {
      const recentAvg = this.particles.slice(-3).reduce((sum, p) => sum + p.intensity, 0) / 
                        Math.min(3, this.particles.length);
      direction = recentAvg - avgSeverity;  // Positive = improving, negative = deteriorating
    }
    
    // Add particle
    this.particles.push({
      state: states[0] || "unknown",
      intensity: avgSeverity,
      direction: Math.max(-1, Math.min(1, direction)),
      timestamp: Date.now()
    });
    
    // Trim to max particles
    if (this.particles.length > this.maxParticles) {
      this.particles = this.particles.slice(-this.maxParticles);
    }
  }
  
  /**
   * Calculate the Vicsek order parameter
   * 0 = complete disorder (chaotic emotional state)
   * 1 = perfect alignment (consistent trajectory)
   */
  calculateOrderParameter(): number {
    if (this.particles.length < 2) return 0.5;
    
    // Sum of direction vectors
    let sumX = 0;
    let sumY = 0;
    
    for (const particle of this.particles) {
      // Convert direction to unit vector
      const angle = particle.direction * Math.PI / 2;  // -90 to 90 degrees
      sumX += Math.cos(angle);
      sumY += Math.sin(angle);
    }
    
    // Order parameter is magnitude of average direction
    const magnitude = Math.sqrt(sumX * sumX + sumY * sumY) / this.particles.length;
    return magnitude;
  }
  
  /**
   * Calculate average direction of the flock
   */
  calculateAverageDirection(): number {
    if (this.particles.length === 0) return 0;
    
    const sum = this.particles.reduce((acc, p) => acc + p.direction, 0);
    return sum / this.particles.length;
  }
  
  /**
   * Calculate noise level based on variance in directions
   */
  calculateNoiseLevel(): number {
    if (this.particles.length < 2) return this.noiseAmplitude;
    
    const avgDir = this.calculateAverageDirection();
    const variance = this.particles.reduce((acc, p) => 
      acc + Math.pow(p.direction - avgDir, 2), 0) / this.particles.length;
    
    return Math.sqrt(variance) + this.noiseAmplitude;
  }
  
  /**
   * Predict emotional trajectory using Vicsek alignment
   */
  predict(): VicsekPrediction {
    const orderParameter = this.calculateOrderParameter();
    const avgDirection = this.calculateAverageDirection();
    const noiseLevel = this.calculateNoiseLevel();
    
    // Recent intensity trend
    const recentParticles = this.particles.slice(-5);
    const recentAvgIntensity = recentParticles.length > 0 ?
      recentParticles.reduce((sum, p) => sum + p.intensity, 0) / recentParticles.length : 0.5;
    
    // Determine trajectory
    let predictedTrajectory: VicsekPrediction["predictedTrajectory"];
    let urgency: VicsekPrediction["urgency"];
    let recommendedIntervention: string;
    
    // High order parameter = consistent trajectory
    if (orderParameter > 0.7) {
      if (avgDirection > 0.2) {
        predictedTrajectory = "improving";
        urgency = "low";
        recommendedIntervention = "Continue current approach. Reinforce positive trajectory.";
      } else if (avgDirection < -0.2) {
        if (recentAvgIntensity > 0.8) {
          predictedTrajectory = "crisis_imminent";
          urgency = "critical";
          recommendedIntervention = "IMMEDIATE INTERVENTION REQUIRED. Trajectory shows consistent deterioration toward crisis.";
        } else {
          predictedTrajectory = "deteriorating";
          urgency = "high";
          recommendedIntervention = "Escalate support. Pattern shows consistent decline.";
        }
      } else {
        predictedTrajectory = "stable";
        urgency = "medium";
        recommendedIntervention = "Monitor closely. Stable but not improving.";
      }
    } else {
      // Low order parameter = chaotic/unpredictable
      if (recentAvgIntensity > 0.7) {
        predictedTrajectory = "deteriorating";
        urgency = "high";
        recommendedIntervention = "High distress with unpredictable pattern. Prioritize stabilization.";
      } else {
        predictedTrajectory = "stable";
        urgency = "medium";
        recommendedIntervention = "Variable emotional state. Focus on grounding and consistency.";
      }
    }
    
    // Confidence based on data quality
    const confidence = Math.min(1, this.particles.length / 10) * orderParameter;
    
    return {
      predictedTrajectory,
      confidence,
      alignmentStrength: orderParameter,
      noiseLevel,
      recommendedIntervention,
      urgency
    };
  }
  
  /**
   * Get current flock state for analysis
   */
  getFlockState(): FlockState {
    return {
      particles: [...this.particles],
      averageDirection: this.calculateAverageDirection(),
      orderParameter: this.calculateOrderParameter()
    };
  }
  
  /**
   * Reset the model (new session)
   */
  reset(): void {
    this.particles = [];
  }
}

// Global instance for session continuity
export const globalVicsekModel = new VicsekEmotionalModel();

/**
 * Analyze emotional trajectory and get intervention recommendation
 */
export function analyzeTrajectory(
  currentStates: string[],
  entropy: number,
  history: string[] = []
): VicsekPrediction {
  // Add current observation
  globalVicsekModel.addObservation(currentStates, entropy);
  
  // Get prediction
  return globalVicsekModel.predict();
}

/**
 * Format trajectory prediction for LLM context
 */
export function formatTrajectoryForPrompt(prediction: VicsekPrediction): string {
  if (prediction.urgency === "low" && prediction.predictedTrajectory === "improving") {
    return "";  // Don't clutter prompt when things are going well
  }
  
  let output = "\n\n[TRAJECTORY ANALYSIS - BACKEND ONLY, DO NOT MENTION TO USER]\n";
  output += `Predicted trajectory: ${prediction.predictedTrajectory.toUpperCase()}\n`;
  output += `Urgency: ${prediction.urgency.toUpperCase()}\n`;
  output += `Confidence: ${(prediction.confidence * 100).toFixed(0)}%\n`;
  output += `Recommendation: ${prediction.recommendedIntervention}\n`;
  
  if (prediction.urgency === "critical") {
    output += "\n⚠️ CRITICAL: Consistent deterioration pattern detected. Prioritize immediate safety.\n";
  }
  
  return output;
}
