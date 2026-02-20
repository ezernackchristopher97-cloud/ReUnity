// ============================================================================
// REUNITY CONTEXT AWARENESS SYSTEM
// Loads environmental, cultural, and community context data for enhanced responses
// ============================================================================

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load context awareness data from JSON file at startup
const contextPath = join(__dirname, '..', 'shared', 'context-awareness.json');
let contextData: any;
try {
  contextData = JSON.parse(readFileSync(contextPath, 'utf-8'));
} catch (e) {
  console.error('Failed to load context-awareness.json:', e);
  contextData = { environmentalContext: {}, culturalContext: {}, communityContext: {}, socioeconomicContext: {} };
}

// ============================================================================
// TYPES
// ============================================================================

export interface EnvironmentProfile {
  type: 'urban' | 'suburban' | 'rural' | 'remote';
  characteristics: string[];
  challenges: string[];
  adaptedSupport: {
    grounding: string;
    resources: string;
    crisis: string;
    specialResources?: Array<{ name: string; phone?: string; website?: string; description: string }>;
  };
}

export interface CulturalProfile {
  culture: string;
  considerations: string[];
  adaptedSupport: {
    approach: string;
    resources: Array<{ name: string; phone?: string; website?: string; description: string }>;
    language?: string;
  };
}

export interface CommunityProfile {
  community: string;
  considerations: string[];
  adaptedSupport: {
    approach: string;
    resources: Array<{ name: string; phone?: string; website?: string; description: string }>;
  };
}

export interface ContextAnalysis {
  environment: EnvironmentProfile | null;
  cultural: CulturalProfile[];
  community: CommunityProfile[];
  socioeconomic: string[];
  contextualGuidance: string;
  additionalResources: Array<{ name: string; phone?: string; website?: string; description: string }>;
}

// ============================================================================
// ENVIRONMENT DETECTION
// ============================================================================

export function detectEnvironment(text: string): EnvironmentProfile | null {
  const lowerText = text.toLowerCase();
  
  // Check each environment type
  for (const [envType, envData] of Object.entries(contextData.environmentalContext)) {
    const data = envData as any;
    if (data.indicators) {
      for (const indicator of data.indicators) {
        if (lowerText.includes(indicator.toLowerCase())) {
          return {
            type: envType as 'urban' | 'suburban' | 'rural' | 'remote',
            characteristics: data.characteristics || [],
            challenges: data.challenges || [],
            adaptedSupport: data.adaptedSupport || {}
          };
        }
      }
    }
  }
  
  return null;
}

// ============================================================================
// CULTURAL CONTEXT DETECTION
// ============================================================================

export function detectCulturalContext(text: string): CulturalProfile[] {
  const lowerText = text.toLowerCase();
  const detected: CulturalProfile[] = [];
  
  for (const [culture, cultureData] of Object.entries(contextData.culturalContext)) {
    if (culture === 'general') continue;
    
    const data = cultureData as any;
    if (data.indicators) {
      for (const indicator of data.indicators) {
        if (lowerText.includes(indicator.toLowerCase())) {
          detected.push({
            culture,
            considerations: data.considerations || [],
            adaptedSupport: data.adaptedSupport || { approach: '', resources: [] }
          });
          break; // Only add each culture once
        }
      }
    }
  }
  
  return detected;
}

// ============================================================================
// COMMUNITY CONTEXT DETECTION
// ============================================================================

export function detectCommunityContext(text: string): CommunityProfile[] {
  const lowerText = text.toLowerCase();
  const detected: CommunityProfile[] = [];
  
  for (const [community, communityData] of Object.entries(contextData.communityContext)) {
    const data = communityData as any;
    
    // Handle nested religious communities
    if (community === 'religious') {
      for (const [religion, religionData] of Object.entries(data)) {
        const relData = religionData as any;
        if (relData.indicators) {
          for (const indicator of relData.indicators) {
            if (lowerText.includes(indicator.toLowerCase())) {
              detected.push({
                community: `religious_${religion}`,
                considerations: relData.considerations || [],
                adaptedSupport: relData.adaptedSupport || { approach: '', resources: [] }
              });
              break;
            }
          }
        }
      }
    } else if (data.indicators) {
      for (const indicator of data.indicators) {
        if (lowerText.includes(indicator.toLowerCase())) {
          detected.push({
            community,
            considerations: data.considerations || [],
            adaptedSupport: data.adaptedSupport || { approach: '', resources: [] }
          });
          break;
        }
      }
    }
  }
  
  return detected;
}

// ============================================================================
// SOCIOECONOMIC CONTEXT DETECTION
// ============================================================================

export function detectSocioeconomicContext(text: string): string[] {
  const lowerText = text.toLowerCase();
  const detected: string[] = [];
  
  for (const [context, contextDataItem] of Object.entries(contextData.socioeconomicContext)) {
    const data = contextDataItem as any;
    if (data.indicators) {
      for (const indicator of data.indicators) {
        if (lowerText.includes(indicator.toLowerCase())) {
          detected.push(context);
          break;
        }
      }
    }
  }
  
  return detected;
}

// ============================================================================
// COMPREHENSIVE CONTEXT ANALYSIS
// ============================================================================

export function analyzeContext(text: string, conversationHistory: string[] = []): ContextAnalysis {
  // Combine current message with recent history for better context
  const fullContext = [...conversationHistory.slice(-5), text].join(' ');
  
  const environment = detectEnvironment(fullContext);
  const cultural = detectCulturalContext(fullContext);
  const community = detectCommunityContext(fullContext);
  const socioeconomic = detectSocioeconomicContext(fullContext);
  
  // Gather all additional resources
  const additionalResources: Array<{ name: string; phone?: string; website?: string; description: string }> = [];
  
  // Add environment-specific resources
  if (environment?.adaptedSupport?.specialResources) {
    additionalResources.push(...environment.adaptedSupport.specialResources);
  }
  
  // Add cultural resources
  for (const c of cultural) {
    if (c.adaptedSupport?.resources) {
      additionalResources.push(...c.adaptedSupport.resources);
    }
  }
  
  // Add community resources
  for (const c of community) {
    if (c.adaptedSupport?.resources) {
      additionalResources.push(...c.adaptedSupport.resources);
    }
  }
  
  // Add socioeconomic resources
  for (const s of socioeconomic) {
    const seData = contextData.socioeconomicContext[s];
    if (seData?.adaptedSupport?.resources) {
      additionalResources.push(...seData.adaptedSupport.resources);
    }
  }
  
  // Generate contextual guidance for the AI
  const contextualGuidance = generateContextualGuidance(environment, cultural, community, socioeconomic);
  
  return {
    environment,
    cultural,
    community,
    socioeconomic,
    contextualGuidance,
    additionalResources
  };
}

// ============================================================================
// GENERATE CONTEXTUAL GUIDANCE FOR AI RESPONSE
// ============================================================================

function generateContextualGuidance(
  environment: EnvironmentProfile | null,
  cultural: CulturalProfile[],
  community: CommunityProfile[],
  socioeconomic: string[]
): string {
  const guidance: string[] = [];
  
  // Environment guidance
  if (environment) {
    guidance.push(`ENVIRONMENT: User appears to be in ${environment.type} setting.`);
    if (environment.adaptedSupport?.grounding) {
      guidance.push(`Grounding adaptation: ${environment.adaptedSupport.grounding}`);
    }
    if (environment.challenges?.length > 0) {
      guidance.push(`Be aware of challenges: ${environment.challenges.slice(0, 3).join(', ')}`);
    }
  }
  
  // Cultural guidance
  for (const c of cultural) {
    guidance.push(`CULTURAL CONTEXT: ${c.culture} background detected.`);
    if (c.adaptedSupport?.approach) {
      guidance.push(`Approach: ${c.adaptedSupport.approach}`);
    }
    if (c.considerations?.length > 0) {
      guidance.push(`Consider: ${c.considerations.slice(0, 2).join('; ')}`);
    }
  }
  
  // Community guidance
  for (const c of community) {
    guidance.push(`COMMUNITY: ${c.community} community context.`);
    if (c.adaptedSupport?.approach) {
      guidance.push(`Approach: ${c.adaptedSupport.approach}`);
    }
  }
  
  // Socioeconomic guidance
  if (socioeconomic.includes('poverty')) {
    guidance.push('SOCIOECONOMIC: Financial stress detected. Prioritize free/low-cost resources. Address basic needs.');
  }
  if (socioeconomic.includes('housingSecurity')) {
    guidance.push('HOUSING: Housing instability detected. This is a major stressor - acknowledge and provide housing resources.');
  }
  
  // General cultural principles
  if (cultural.length > 0 || community.length > 0) {
    const general = contextData.culturalContext?.general?.principles || [];
    if (general.length > 0) {
      guidance.push(`GENERAL PRINCIPLES: ${general.slice(0, 2).join('; ')}`);
    }
  }
  
  return guidance.join('\n');
}

// ============================================================================
// FORMAT CONTEXT-AWARE RESOURCES FOR RESPONSE
// ============================================================================

export function formatContextResources(analysis: ContextAnalysis): string {
  if (analysis.additionalResources.length === 0) return '';
  
  // Deduplicate by name
  const seen = new Set<string>();
  const unique = analysis.additionalResources.filter(r => {
    if (seen.has(r.name)) return false;
    seen.add(r.name);
    return true;
  });
  
  if (unique.length === 0) return '';
  
  const lines = unique.slice(0, 4).map(r => {
    let line = `• ${r.name}`;
    if (r.phone) line += `: ${r.phone}`;
    else if (r.website) line += `: ${r.website}`;
    return line;
  });
  
  return `**Community-Specific Support:**\n${lines.join('\n')}`;
}

// ============================================================================
// GET ADAPTED GROUNDING SUGGESTION
// ============================================================================

export function getAdaptedGrounding(analysis: ContextAnalysis, baseGrounding: string): string {
  // If environment detected, adapt grounding suggestion
  if (analysis.environment?.adaptedSupport?.grounding) {
    return `${baseGrounding}\n\n*Adapted for your setting: ${analysis.environment.adaptedSupport.grounding}*`;
  }
  return baseGrounding;
}
