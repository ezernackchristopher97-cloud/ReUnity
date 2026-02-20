import groundingTechniquesData from "../shared/grounding-techniques.json";
import mentalHealthData from "../shared/mental-health-interventions.json";

interface Technique {
  id: string;
  name: string;
  category: string;
  intensity: string;
  duration: string;
  forStates: string[];
  contraindications: string[];
  instructions: string;
  scienceBasis: string;
}

interface ConditionInfo {
  name: string;
  coreExperiences: string[];
  validationStatements: string[];
  interventionPriority: string[];
  contraindicated: string[];
  keyPhrases: string[];
}

interface CrisisProtocol {
  priority: string;
  immediateActions: string[];
  techniques: string[];
  resources: string[];
  languageGuidance: string;
}

// Load all techniques into a flat map for quick lookup
const allTechniques: Map<string, Technique> = new Map();
const techniquesByCategory: Map<string, Technique[]> = new Map();

// Initialize technique maps
function initializeTechniques() {
  const categories = groundingTechniquesData.techniques as Record<string, Technique[]>;
  for (const [category, techniques] of Object.entries(categories)) {
    techniquesByCategory.set(category, techniques);
    for (const technique of techniques) {
      allTechniques.set(technique.id, technique);
    }
  }
}
initializeTechniques();

// State to technique mapping
const stateMapping = groundingTechniquesData.stateToTechniqueMapping as Record<string, string[]>;

// Mental health conditions
const conditions = mentalHealthData.conditions as Record<string, ConditionInfo>;
const crisisProtocols = mentalHealthData.crisisProtocols as Record<string, CrisisProtocol>;

/**
 * Detect mental health conditions from message content
 */
export function detectConditions(message: string, history: string[]): string[] {
  const fullText = [message, ...history].join(" ").toLowerCase();
  const detected: string[] = [];
  
  for (const [conditionId, condition] of Object.entries(conditions)) {
    for (const phrase of condition.keyPhrases) {
      if (fullText.includes(phrase.toLowerCase())) {
        if (!detected.includes(conditionId)) {
          detected.push(conditionId);
        }
        break;
      }
    }
  }
  
  return detected;
}

/**
 * Detect emotional/mental states from message
 */
export function detectStates(message: string, entropy: number, regime: string): string[] {
  const text = message.toLowerCase();
  const states: string[] = [];
  
  // State detection patterns
  const statePatterns: Record<string, string[]> = {
    dissociation: ["dissociat", "not real", "floating", "watching myself", "outside my body", "foggy", "numb", "detached"],
    panic: ["panic", "can't breathe", "heart racing", "going to die", "losing control"],
    flashback: ["flashback", "feels like it's happening", "can't escape", "back there", "reliving"],
    anxiety: ["anxious", "worried", "nervous", "on edge", "can't relax", "stressed"],
    depression: ["hopeless", "worthless", "empty", "no point", "tired", "can't get up", "don't care"],
    anger: ["angry", "furious", "rage", "want to hurt", "pissed", "hate"],
    shame: ["ashamed", "disgusting", "worthless", "hate myself", "embarrassed", "pathetic"],
    trauma_activation: ["triggered", "trauma", "abuse", "assault", "violated"],
    identity_confusion: ["don't know who I am", "identity", "different people", "not myself"],
    emotional_numbness: ["numb", "can't feel", "empty", "nothing", "flat"],
    overwhelm: ["overwhelmed", "too much", "can't handle", "drowning", "falling apart"],
    loneliness: ["alone", "lonely", "no one", "isolated", "abandoned"],
    hypervigilance: ["can't relax", "always watching", "on guard", "jumpy", "startled"],
    derealization: ["not real", "dream", "fake", "simulation", "unreal"],
    depersonalization: ["not me", "watching myself", "robot", "alien", "disconnected from body"],
    suicidal_ideation: ["want to die", "kill myself", "end it", "suicide", "not worth living", "better off dead"],
    self_harm_urge: ["cut", "hurt myself", "burn", "self harm", "pain helps"],
    did_osdd: ["parts", "alters", "system", "switching", "lost time", "we ", " us "],
    bpd_crisis: ["everyone leaves", "splitting", "favorite person", "empty", "identity"],
    ptsd_activation: ["flashback", "nightmare", "triggered", "hypervigilant", "startle"]
  };
  
  for (const [state, patterns] of Object.entries(statePatterns)) {
    for (const pattern of patterns) {
      if (text.includes(pattern)) {
        if (!states.includes(state)) {
          states.push(state);
        }
        break;
      }
    }
  }
  
  // Add states based on entropy and regime
  if (entropy > 0.7) {
    if (!states.includes("overwhelm")) states.push("overwhelm");
  }
  if (regime === "crisis") {
    if (!states.includes("panic")) states.push("panic");
  }
  
  return states;
}

/**
 * Select appropriate techniques based on detected states and conditions
 */
export function selectTechniques(
  states: string[],
  conditionIds: string[],
  intensity: "low" | "medium" | "high" = "medium",
  contraindications: string[] = []
): Technique[] {
  const selectedIds = new Set<string>();
  const selected: Technique[] = [];
  
  // First, get techniques from detected conditions
  for (const conditionId of conditionIds) {
    const condition = conditions[conditionId];
    if (condition) {
      for (const techId of condition.interventionPriority.slice(0, 2)) {
        if (!selectedIds.has(techId)) {
          const tech = allTechniques.get(techId);
          if (tech && !tech.contraindications.some(c => contraindications.includes(c))) {
            selectedIds.add(techId);
            selected.push(tech);
          }
        }
      }
    }
  }
  
  // Then, get techniques from detected states
  for (const state of states) {
    const techIds = stateMapping[state];
    if (techIds) {
      for (const techId of techIds.slice(0, 2)) {
        if (!selectedIds.has(techId)) {
          const tech = allTechniques.get(techId);
          if (tech && !tech.contraindications.some(c => contraindications.includes(c))) {
            selectedIds.add(techId);
            selected.push(tech);
          }
        }
      }
    }
  }
  
  // Filter by intensity preference
  const intensityOrder = ["low", "medium", "high"];
  const maxIntensityIndex = intensityOrder.indexOf(intensity);
  
  const filtered = selected.filter(t => {
    const techIntensity = intensityOrder.indexOf(t.intensity);
    return techIntensity <= maxIntensityIndex + 1; // Allow one level above
  });
  
  // Return top 3-4 most relevant techniques
  return filtered.slice(0, 4);
}

/**
 * Get validation statements for detected conditions
 */
export function getValidationStatements(conditionIds: string[]): string[] {
  const statements: string[] = [];
  
  for (const conditionId of conditionIds) {
    const condition = conditions[conditionId];
    if (condition && condition.validationStatements.length > 0) {
      // Pick 1-2 random validation statements per condition
      const shuffled = [...condition.validationStatements].sort(() => Math.random() - 0.5);
      statements.push(...shuffled.slice(0, 2));
    }
  }
  
  return statements.slice(0, 4); // Max 4 statements
}

/**
 * Get crisis protocol if applicable
 */
export function getCrisisProtocol(states: string[]): CrisisProtocol | null {
  // Check for active suicidal plan FIRST (CRITICAL priority)
  const activePlanIndicators = ["active_plan", "written_note", "suicide_note", "goodbye", "given_away", "have_means", "tonight", "set_date", "made_plan"];
  if (states.some(s => activePlanIndicators.some(indicator => s.includes(indicator))) || states.includes("active_suicidal_plan")) {
    return crisisProtocols["active_suicidal_plan"] || crisisProtocols["suicidal_ideation"];
  }
  
  // Check for crisis states in priority order
  const crisisPriority = ["suicidal_ideation", "self_harm_active", "psychotic_episode", "panic_attack", "dissociative_episode"];
  
  for (const crisisType of crisisPriority) {
    if (states.includes(crisisType) || states.includes(crisisType.replace("_active", "_urge"))) {
      return crisisProtocols[crisisType] || null;
    }
  }
  
  // Check for panic
  if (states.includes("panic")) {
    return crisisProtocols["panic_attack"];
  }
  
  // Check for dissociation
  if (states.includes("dissociation") || states.includes("derealization") || states.includes("depersonalization")) {
    return crisisProtocols["dissociative_episode"];
  }
  
  return null;
}

/**
 * Format techniques for LLM context
 */
export function formatTechniquesForPrompt(techniques: Technique[]): string {
  if (techniques.length === 0) return "";
  
  let output = "\\n\\nSELECTED GROUNDING TECHNIQUES (use these specifically, not generic 5-4-3-2-1):\\n";
  
  for (const tech of techniques) {
    output += `\\n### ${tech.name} (${tech.category}, ${tech.intensity} intensity, ${tech.duration})\\n`;
    output += `Instructions: ${tech.instructions}\\n`;
    output += `Why it works: ${tech.scienceBasis}\\n`;
  }
  
  return output;
}

/**
 * Format validation statements for LLM context
 */
export function formatValidationForPrompt(statements: string[]): string {
  if (statements.length === 0) return "";
  
  let output = "\\n\\nVALIDATION STATEMENTS TO INCORPORATE:\\n";
  for (const statement of statements) {
    output += `• ${statement}\\n`;
  }
  
  return output;
}

/**
 * Format crisis protocol for LLM context
 */
export function formatCrisisProtocol(protocol: CrisisProtocol | null): string {
  if (!protocol) return "";
  
  let output = `\\n\\n⚠️ CRISIS PROTOCOL ACTIVE (Priority: ${protocol.priority})\\n`;
  output += `Language Guidance: ${protocol.languageGuidance}\\n`;
  output += `Immediate Actions: ${protocol.immediateActions.join(", ")}\\n`;
  
  return output;
}

/**
 * Main function to get tailored intervention guidance
 */
export function getTailoredIntervention(
  message: string,
  history: string[],
  entropy: number,
  regime: string
): {
  techniques: Technique[];
  validationStatements: string[];
  crisisProtocol: CrisisProtocol | null;
  detectedConditions: string[];
  detectedStates: string[];
  promptGuidance: string;
} {
  // Detect conditions and states
  const detectedConditions = detectConditions(message, history);
  const detectedStates = detectStates(message, entropy, regime);
  
  // Get crisis protocol if needed
  const crisisProtocol = getCrisisProtocol(detectedStates);
  
  // Select techniques
  const intensity = crisisProtocol?.priority === "CRITICAL" ? "high" :
                   crisisProtocol?.priority === "HIGHEST" ? "high" : 
                   crisisProtocol?.priority === "HIGH" ? "medium" : "low";
  const techniques = selectTechniques(detectedStates, detectedConditions, intensity as "low" | "medium" | "high");
  
  // Get validation statements
  const validationStatements = getValidationStatements(detectedConditions);
  
  // Build prompt guidance
  let promptGuidance = "";
  promptGuidance += formatCrisisProtocol(crisisProtocol);
  promptGuidance += formatValidationForPrompt(validationStatements);
  promptGuidance += formatTechniquesForPrompt(techniques);
  
  if (detectedConditions.length > 0) {
    promptGuidance += `\\n\\nDETECTED CONDITIONS: ${detectedConditions.map(c => conditions[c]?.name || c).join(", ")}`;
  }
  if (detectedStates.length > 0) {
    promptGuidance += `\\nDETECTED STATES: ${detectedStates.join(", ")}`;
  }
  
  promptGuidance += `\\n\\nIMPORTANT: Use the SPECIFIC techniques listed above. Do NOT default to generic 5-4-3-2-1 unless it's specifically selected. Tailor your response to the detected conditions and states.`;
  
  return {
    techniques,
    validationStatements,
    crisisProtocol,
    detectedConditions,
    detectedStates,
    promptGuidance
  };
}

// Export condition info for reference
export function getConditionInfo(conditionId: string): ConditionInfo | null {
  return conditions[conditionId] || null;
}

// Export all technique categories
export function getAllCategories(): string[] {
  return Array.from(techniquesByCategory.keys());
}

// Export techniques by category
export function getTechniquesByCategory(category: string): Technique[] {
  return techniquesByCategory.get(category) || [];
}
