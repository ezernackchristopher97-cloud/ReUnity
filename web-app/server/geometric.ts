/**
 * Geometric Computing Module for ReUnity
 * 
 * Implements quaternion-based semantic encoding and toroidal memory embedding
 * for enhanced emotional understanding and more human-like responses.
 * 
 * Quaternion Encoding Process:
 * 1. Takes user words and converts semantics into 4D geometric space
 * 2. Uses algebraic rotation rules (Hamilton's rules: i²=j²=k²=ijk=-1)
 * 3. Assigns meaning through rotation counts
 * 4. Converts to binary for direct interpretation
 * 5. Feeds into entropy calculations for response calibration
 * 
 * This enables the AI to interpret meaning and emotion directly,
 * producing more human-like, empathetic responses.
 * 
 * Created by Christopher Ezernack, REOP Solutions
 */

// =============================================================================
// SECTION 1: QUATERNION SEMANTIC ENCODING
// =============================================================================

/**
 * Quaternion class for 4D semantic encoding
 * Uses Hamilton's algebraic rules: i²=j²=k²=ijk=-1
 * 
 * Components represent:
 * - w: Real component (overall semantic weight)
 * - x (i): Emotional valence dimension
 * - y (j): Urgency/intensity dimension  
 * - z (k): Complexity/abstraction dimension
 */
export class Quaternion {
  constructor(
    public w: number = 1,  // Real component
    public x: number = 0,  // i component (emotional)
    public y: number = 0,  // j component (urgency)
    public z: number = 0   // k component (complexity)
  ) {}

  /**
   * Normalize to unit quaternion (rotation representation)
   */
  normalize(): Quaternion {
    const mag = Math.sqrt(this.w * this.w + this.x * this.x + this.y * this.y + this.z * this.z);
    if (mag === 0) return new Quaternion(1, 0, 0, 0);
    return new Quaternion(this.w / mag, this.x / mag, this.y / mag, this.z / mag);
  }

  /**
   * Quaternion multiplication (non-commutative - order matters for chain logic)
   * Implements Hamilton's rules: ij=k, jk=i, ki=j, ji=-k, kj=-i, ik=-j
   */
  multiply(q: Quaternion): Quaternion {
    return new Quaternion(
      this.w * q.w - this.x * q.x - this.y * q.y - this.z * q.z,
      this.w * q.x + this.x * q.w + this.y * q.z - this.z * q.y,
      this.w * q.y - this.x * q.z + this.y * q.w + this.z * q.x,
      this.w * q.z + this.x * q.y - this.y * q.x + this.z * q.w
    );
  }

  /**
   * Conjugate (inverse rotation)
   */
  conjugate(): Quaternion {
    return new Quaternion(this.w, -this.x, -this.y, -this.z);
  }

  /**
   * Dot product for similarity comparison
   */
  dot(q: Quaternion): number {
    return this.w * q.w + this.x * q.x + this.y * q.y + this.z * q.z;
  }

  /**
   * Get rotation angle (radians) - represents semantic "distance" from neutral
   */
  getRotationAngle(): number {
    const normalized = this.normalize();
    return 2 * Math.acos(Math.min(1, Math.abs(normalized.w)));
  }

  /**
   * Get rotation count (number of quarter-turns from neutral)
   * Used for assigning meaning levels
   */
  getRotationCount(): number {
    const angle = this.getRotationAngle();
    return Math.round(angle / (Math.PI / 2)); // Quarter-turn units
  }

  /**
   * Convert to binary encoding for direct machine interpretation
   * Each component encoded as 16-bit signed integer
   */
  toBinary(): string {
    const encode = (v: number): string => {
      // Clamp to [-1, 1] and scale to 16-bit range
      const clamped = Math.max(-1, Math.min(1, v));
      const scaled = Math.round((clamped + 1) * 32767);
      return scaled.toString(2).padStart(16, '0');
    };
    return encode(this.w) + encode(this.x) + encode(this.y) + encode(this.z);
  }

  /**
   * Get binary meaning value (sum of binary 1s indicates intensity)
   */
  getBinaryMeaning(): number {
    const binary = this.toBinary();
    let count = 0;
    for (const char of binary) {
      if (char === '1') count++;
    }
    return count / 64; // Normalize to 0-1 range
  }

  /**
   * Compute semantic hash for quick comparison
   */
  semanticHash(): string {
    const hash = (v: number) => Math.round((v + 1) * 127).toString(16).padStart(2, '0');
    return hash(this.w) + hash(this.x) + hash(this.y) + hash(this.z);
  }

  /**
   * Distance to another quaternion (semantic difference)
   */
  distance(q: Quaternion): number {
    const dot = Math.abs(this.dot(q));
    return Math.acos(Math.min(1, dot)) * 2;
  }
}

// =============================================================================
// SECTION 2: WORD-TO-QUATERNION ENCODING
// =============================================================================

/**
 * Emotional word mappings for valence encoding (x component)
 */
const EMOTIONAL_WORDS: Record<string, number> = {
  // Positive emotions (positive x values)
  'happy': 0.8, 'joy': 0.9, 'love': 0.85, 'hope': 0.7, 'peace': 0.75,
  'calm': 0.6, 'safe': 0.7, 'good': 0.5, 'better': 0.55, 'okay': 0.3,
  'grateful': 0.75, 'relieved': 0.65, 'content': 0.6, 'proud': 0.7,
  'excited': 0.75, 'confident': 0.65, 'hopeful': 0.7, 'supported': 0.6,
  
  // Negative emotions (negative x values)
  'sad': -0.6, 'depressed': -0.8, 'anxious': -0.7, 'scared': -0.75,
  'afraid': -0.7, 'terrified': -0.9, 'panic': -0.85, 'hurt': -0.65,
  'pain': -0.7, 'angry': -0.6, 'rage': -0.85, 'hate': -0.8,
  'hopeless': -0.9, 'worthless': -0.85, 'empty': -0.75, 'numb': -0.6,
  'alone': -0.7, 'abandoned': -0.8, 'rejected': -0.75, 'betrayed': -0.8,
  'guilty': -0.65, 'ashamed': -0.7, 'disgusted': -0.6, 'frustrated': -0.55,
  'overwhelmed': -0.75, 'exhausted': -0.6, 'trapped': -0.8, 'lost': -0.65
};

/**
 * Urgency word mappings (y component)
 */
const URGENCY_WORDS: Record<string, number> = {
  // High urgency (positive y)
  'now': 0.8, 'immediately': 0.9, 'urgent': 0.85, 'emergency': 0.95,
  'crisis': 0.9, 'help': 0.7, 'please': 0.5, 'need': 0.6,
  'cant': 0.7, "can't": 0.7, 'dying': 0.95, 'killing': 0.95,
  'suicide': 0.95, 'suicidal': 0.95, 'end': 0.6, 'stop': 0.5,
  
  // Low urgency (negative y)
  'sometimes': -0.3, 'occasionally': -0.4, 'maybe': -0.3, 'wondering': -0.2,
  'thinking': -0.1, 'curious': -0.2, 'eventually': -0.5, 'someday': -0.6
};

/**
 * Complexity word mappings (z component)
 */
const COMPLEXITY_WORDS: Record<string, number> = {
  // High complexity (positive z)
  'dissociating': 0.8, 'dissociation': 0.8, 'depersonalization': 0.85,
  'derealization': 0.85, 'flashback': 0.75, 'triggered': 0.7,
  'splitting': 0.8, 'identity': 0.7, 'fragmented': 0.75, 'parts': 0.6,
  'system': 0.65, 'alter': 0.7, 'switch': 0.65, 'trauma': 0.7,
  'ptsd': 0.75, 'complex': 0.6, 'pattern': 0.5, 'relationship': 0.4,
  
  // Low complexity (negative z)
  'simple': -0.5, 'basic': -0.4, 'just': -0.3, 'only': -0.2
};

/**
 * Encode a single word into a quaternion
 */
function encodeWord(word: string): Quaternion {
  const lower = word.toLowerCase();
  
  // Get component values from word mappings
  const emotional = EMOTIONAL_WORDS[lower] || 0;
  const urgency = URGENCY_WORDS[lower] || 0;
  const complexity = COMPLEXITY_WORDS[lower] || 0;
  
  // Character-based encoding for unknown words
  let charWeight = 0;
  for (let i = 0; i < lower.length; i++) {
    charWeight += (lower.charCodeAt(i) - 97) * Math.pow(0.1, i + 1);
  }
  charWeight = Math.tanh(charWeight); // Normalize to [-1, 1]
  
  // W component: word length normalized + char weight
  const w = Math.tanh((lower.length - 5) * 0.1 + charWeight * 0.3);
  
  // If word has known emotional/urgency/complexity, use those
  // Otherwise, derive from character patterns
  const x = emotional !== 0 ? emotional : charWeight * 0.3;
  const y = urgency !== 0 ? urgency : 0;
  const z = complexity !== 0 ? complexity : 0;
  
  return new Quaternion(w, x, y, z);
}

/**
 * Encode full text into a quaternion using chain multiplication
 * Order matters due to non-commutativity - captures word sequence meaning
 */
export function encodeSemantics(text: string): Quaternion {
  if (!text || text.trim().length === 0) {
    return new Quaternion(1, 0, 0, 0);
  }

  // Tokenize
  const words = text.toLowerCase()
    .replace(/[^\w\s'-]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 0);

  if (words.length === 0) {
    return new Quaternion(1, 0, 0, 0);
  }

  // Chain multiply word quaternions (order-dependent)
  let result = encodeWord(words[0]);
  
  for (let i = 1; i < words.length; i++) {
    const wordQ = encodeWord(words[i]);
    // Multiply and normalize to prevent magnitude explosion
    result = result.multiply(wordQ).normalize();
  }

  return result.normalize();
}

/**
 * Get rotation-based meaning from text
 * Returns rotation count and binary meaning value
 */
export function getSemanticMeaning(text: string): {
  rotationCount: number;
  binaryMeaning: number;
  emotionalValence: number;
  urgencyLevel: number;
  complexityLevel: number;
} {
  const q = encodeSemantics(text);
  
  return {
    rotationCount: q.getRotationCount(),
    binaryMeaning: q.getBinaryMeaning(),
    emotionalValence: q.x,  // -1 (negative) to +1 (positive)
    urgencyLevel: q.y,      // -1 (low) to +1 (high)
    complexityLevel: q.z    // -1 (simple) to +1 (complex)
  };
}

// =============================================================================
// SECTION 3: TOROIDAL MEMORY EMBEDDING
// =============================================================================

/**
 * Position on torus surface
 */
export interface TorusPosition {
  theta: number;  // Major angle (0 to 2π)
  phi: number;    // Minor angle (0 to 2π)
}

/**
 * Toroidal memory with bounded capacity
 * Memory wraps around like a torus - old memories cycle out naturally
 */
export class ToroidalMemory {
  private states: Quaternion[] = [];
  private positions: TorusPosition[] = [];
  private maxStates: number;

  constructor(maxStates: number = 1000) {
    this.maxStates = maxStates;
  }

  /**
   * Embed quaternion onto torus surface
   */
  embed(q: Quaternion): TorusPosition {
    const theta = Math.atan2(q.y, q.x) + Math.PI;
    const phi = Math.atan2(q.z, q.w) + Math.PI;
    return { theta, phi };
  }

  /**
   * Push new state (cycles when full)
   */
  pushState(q: Quaternion): void {
    const position = this.embed(q);
    
    if (this.states.length >= this.maxStates) {
      this.states.shift();
      this.positions.shift();
    }
    
    this.states.push(q);
    this.positions.push(position);
  }

  /**
   * Find nearest states to query
   */
  findNearest(q: Quaternion, k: number = 5): Array<{ state: Quaternion; distance: number }> {
    const distances = this.states.map((state, i) => ({
      state,
      distance: q.distance(state)
    }));
    
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, k);
  }

  /**
   * Get memory statistics
   */
  getStats(): { statesStored: number; maxStates: number; utilizationPercent: number } {
    return {
      statesStored: this.states.length,
      maxStates: this.maxStates,
      utilizationPercent: (this.states.length / this.maxStates) * 100
    };
  }

  /**
   * Clear memory
   */
  clear(): void {
    this.states = [];
    this.positions = [];
  }
}

// Global torus instance
export const globalTorus = new ToroidalMemory(1000);

// =============================================================================
// SECTION 4: REGIME SELECTION
// =============================================================================

export type RegimeType = 'emotional' | 'analytical' | 'creative' | 'crisis' | 'grounding';

/**
 * Select processing regime based on semantic analysis
 */
export function selectRegime(text: string, semantics: Quaternion): RegimeType {
  const lower = text.toLowerCase();
  const meaning = getSemanticMeaning(text);
  
  // Crisis regime (highest priority)
  if (meaning.urgencyLevel > 0.7 || 
      lower.includes('suicide') || lower.includes('kill myself') ||
      lower.includes('want to die') || lower.includes('end my life')) {
    return 'crisis';
  }
  
  // Grounding regime
  if (meaning.complexityLevel > 0.5 ||
      lower.includes('dissociat') || lower.includes('flashback') ||
      lower.includes('panic') || lower.includes('triggered')) {
    return 'grounding';
  }
  
  // Emotional regime (default for negative valence)
  if (meaning.emotionalValence < -0.3) {
    return 'emotional';
  }
  
  // Analytical regime
  if (lower.includes('why') || lower.includes('how') || 
      lower.includes('explain') || lower.includes('understand')) {
    return 'analytical';
  }
  
  // Creative regime
  if (lower.includes('imagine') || lower.includes('visualize') ||
      lower.includes('picture') || lower.includes('dream')) {
    return 'creative';
  }
  
  return 'emotional';
}

// =============================================================================
// SECTION 5: L1/L2 FILTERS
// =============================================================================

export interface FilterResult {
  passed: boolean;
  score: number;
  reasons: string[];
}

/**
 * L1 Coherence Filter - validates input structure
 */
export function l1CoherenceFilter(text: string, semantics: Quaternion): FilterResult {
  const reasons: string[] = [];
  let score = 1.0;

  // Check minimum length
  if (text.trim().length < 2) {
    reasons.push('Input too short');
    score -= 0.5;
  }

  // Check for excessive repetition
  const words = text.toLowerCase().split(/\s+/);
  const uniqueWords = new Set(words);
  if (uniqueWords.size / words.length < 0.3 && words.length > 5) {
    reasons.push('Excessive repetition');
    score -= 0.2;
  }

  return { passed: score > 0.5, score: Math.max(0, score), reasons };
}

/**
 * L2 Stability Filter - validates semantic stability
 */
export function l2StabilityFilter(semantics: Quaternion): FilterResult {
  const reasons: string[] = [];
  let score = 1.0;

  // Check quaternion is normalized
  const mag = Math.sqrt(
    semantics.w * semantics.w + semantics.x * semantics.x + 
    semantics.y * semantics.y + semantics.z * semantics.z
  );
  if (Math.abs(mag - 1.0) > 0.1) {
    reasons.push('Semantic instability');
    score -= 0.2;
  }

  return { passed: score > 0.6, score: Math.max(0, score), reasons };
}

// =============================================================================
// SECTION 6: ABSURDITY GAP
// =============================================================================

/**
 * Compute absurdity gap (off-topic/testing detection)
 */
export function computeAbsurdityGap(text: string, semantics: Quaternion): number {
  const lower = text.toLowerCase();
  let score = 0;

  // Testing/jailbreak patterns
  const testPatterns = [
    'ignore previous', 'ignore all', 'new instructions', 'pretend you are',
    'act as if', 'roleplay as', 'you are now', 'forget everything',
    'system prompt', 'bypass', 'override', 'jailbreak'
  ];
  for (const pattern of testPatterns) {
    if (lower.includes(pattern)) score += 0.4;
  }

  // Off-topic patterns
  const offTopicPatterns = [
    'write code', 'programming', 'recipe', 'weather forecast',
    'stock price', 'sports score', 'celebrity'
  ];
  for (const pattern of offTopicPatterns) {
    if (lower.includes(pattern)) score += 0.2;
  }

  return Math.min(1, score);
}

// =============================================================================
// SECTION 7: CONSENSUS SCORING
// =============================================================================

export interface ConsensusScores {
  builderScore: number;
  criticScore: number;
  verifierScore: number;
  finalConfidence: number;
}

/**
 * Compute consensus scores for response quality
 */
export function computeConsensus(response: string, regime: RegimeType): ConsensusScores {
  const lower = response.toLowerCase();
  
  // Builder: response completeness
  const builderScore = Math.min(1, response.length / 500) * 0.8 + 
                       (response.includes('.') ? 0.1 : 0) +
                       (response.includes('?') ? 0.1 : 0);

  // Critic: appropriateness for regime
  let criticScore = 0.7;
  if (regime === 'crisis' && (lower.includes('988') || lower.includes('crisis'))) {
    criticScore = 0.9;
  } else if (regime === 'grounding' && (lower.includes('breath') || lower.includes('ground'))) {
    criticScore = 0.85;
  } else if (regime === 'emotional' && (lower.includes('feel') || lower.includes('hear you'))) {
    criticScore = 0.85;
  }

  // Verifier: average
  const verifierScore = (builderScore + criticScore) / 2;
  
  // Final confidence
  const finalConfidence = builderScore * 0.3 + criticScore * 0.4 + verifierScore * 0.3;

  return {
    builderScore: Math.round(builderScore * 100) / 100,
    criticScore: Math.round(criticScore * 100) / 100,
    verifierScore: Math.round(verifierScore * 100) / 100,
    finalConfidence: Math.round(finalConfidence * 100) / 100
  };
}

// =============================================================================
// SECTION 8: ENTROPY CONTRIBUTION
// =============================================================================

/**
 * Calculate entropy contribution from quaternion encoding
 * This feeds directly into the main entropy calculations
 */
export function calculateEntropyContribution(semantics: Quaternion): {
  emotionalEntropy: number;
  urgencyEntropy: number;
  complexityEntropy: number;
  totalContribution: number;
} {
  // Convert quaternion components to entropy contributions
  // Negative emotional valence increases entropy
  const emotionalEntropy = Math.max(0, -semantics.x) * 0.4;
  
  // High urgency increases entropy
  const urgencyEntropy = Math.max(0, semantics.y) * 0.3;
  
  // High complexity increases entropy
  const complexityEntropy = Math.max(0, semantics.z) * 0.2;
  
  // Binary meaning adds subtle contribution
  const binaryContribution = semantics.getBinaryMeaning() * 0.1;
  
  const totalContribution = emotionalEntropy + urgencyEntropy + complexityEntropy + binaryContribution;
  
  return {
    emotionalEntropy,
    urgencyEntropy,
    complexityEntropy,
    totalContribution: Math.min(1, totalContribution)
  };
}

// =============================================================================
// SECTION 9: MAIN PROCESSING FUNCTION
// =============================================================================

export interface GeometricProcessingResult {
  semantics: Quaternion;
  meaning: {
    rotationCount: number;
    binaryMeaning: number;
    emotionalValence: number;
    urgencyLevel: number;
    complexityLevel: number;
  };
  regime: RegimeType;
  l1Filter: FilterResult;
  l2Filter: FilterResult;
  absurdityScore: number;
  entropyContribution: {
    emotionalEntropy: number;
    urgencyEntropy: number;
    complexityEntropy: number;
    totalContribution: number;
  };
  torusPosition: TorusPosition;
}

/**
 * Main geometric processing function
 * Processes input through full quaternion pipeline and returns entropy contribution
 */
export function processGeometric(text: string): GeometricProcessingResult {
  // Step 1: Encode text to quaternion
  const semantics = encodeSemantics(text);
  
  // Step 2: Get semantic meaning
  const meaning = getSemanticMeaning(text);
  
  // Step 3: Select regime
  const regime = selectRegime(text, semantics);
  
  // Step 4: L1 Filter
  const l1Filter = l1CoherenceFilter(text, semantics);
  
  // Step 5: L2 Filter
  const l2Filter = l2StabilityFilter(semantics);
  
  // Step 6: Absurdity gap
  const absurdityScore = computeAbsurdityGap(text, semantics);
  
  // Step 7: Calculate entropy contribution
  const entropyContribution = calculateEntropyContribution(semantics);
  
  // Step 8: Embed on torus and store
  const torusPosition = globalTorus.embed(semantics);
  globalTorus.pushState(semantics);
  
  // Log for debugging (backend only)
  console.log(`[Geometric] Regime: ${regime}, Entropy contribution: ${entropyContribution.totalContribution.toFixed(3)}, Rotations: ${meaning.rotationCount}`);
  
  return {
    semantics,
    meaning,
    regime,
    l1Filter,
    l2Filter,
    absurdityScore,
    entropyContribution,
    torusPosition
  };
}

/**
 * Add consensus scores after response generation
 */
export function addConsensus(
  result: GeometricProcessingResult, 
  response: string
): GeometricProcessingResult & { consensus: ConsensusScores } {
  return {
    ...result,
    consensus: computeConsensus(response, result.regime)
  };
}
