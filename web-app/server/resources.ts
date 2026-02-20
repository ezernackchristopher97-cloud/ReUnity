// ============================================================================
// REUNITY RESOURCE SYSTEM
// Loads resource data from shared/resources.json for easy maintenance
// ============================================================================

import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load resources from JSON file at startup
const resourcesPath = join(__dirname, '..', 'shared', 'resources.json');
let resourceData: any;
try {
  resourceData = JSON.parse(readFileSync(resourcesPath, 'utf-8'));
} catch (e) {
  console.error('Failed to load resources.json:', e);
  resourceData = { national: {}, states: {}, conditionSpecific: {}, ruralResources: {} };
}

// ============================================================================
// TYPES
// ============================================================================

export interface Resource {
  name: string;
  phone?: string;
  text?: string;
  chat?: string;
  website?: string;
  description: string;
  hours?: string;
  languages?: string[];
  specializations?: string[];
  priority?: number;
}

export interface StateResource {
  name: string;
  crisis: string;
  mentalHealth: string;
  ruralNote?: string;
}

export interface ConditionResources {
  resources: Resource[];
  grounding: string[];
}

export interface ResourceSelection {
  crisis: Resource[];
  condition: Resource[];
  state: StateResource | null;
  rural: any | null;
  categories: string[];
}

// ============================================================================
// EXPORTED DATA (from JSON)
// ============================================================================

export const nationalResources = resourceData.national;
export const stateResources: Record<string, StateResource> = resourceData.states;
export const conditionResources: Record<string, ConditionResources> = resourceData.conditionSpecific;
export const ruralResources = resourceData.ruralResources;

// ============================================================================
// STATE DETECTION
// ============================================================================

const stateNameToAbbr: Record<string, string> = {
  'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
  'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
  'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
  'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
  'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
  'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
  'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
  'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
  'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
  'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
  'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
  'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
  'wisconsin': 'WI', 'wyoming': 'WY', 'district of columbia': 'DC', 'dc': 'DC'
};

export function detectStateFromText(text: string): string | null {
  const lowerText = text.toLowerCase();
  
  // Check for state names
  for (const [name, abbr] of Object.entries(stateNameToAbbr)) {
    if (lowerText.includes(name)) {
      return abbr;
    }
  }
  
  // Check for state abbreviations (with word boundaries)
  const abbrPattern = /\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)\b/i;
  const match = text.match(abbrPattern);
  if (match) {
    return match[1].toUpperCase();
  }
  
  return null;
}

// ============================================================================
// CONTEXT DETECTION
// ============================================================================

export function detectRuralContext(text: string): boolean {
  const ruralIndicators = [
    'rural', 'small town', 'country', 'farm', 'ranch', 'middle of nowhere',
    'hours away', 'no therapist', 'no counselor', 'nearest', 'drive',
    'isolated', 'remote', 'population', 'miles away', 'hour drive',
    '2 hours', '3 hours', 'far from'
  ];
  const lowerText = text.toLowerCase();
  return ruralIndicators.some(indicator => lowerText.includes(indicator));
}

// ============================================================================
// RESOURCE SELECTION
// ============================================================================

export function selectResources(
  state: string | null,
  conditions: string[],
  isRural: boolean,
  isCrisis: boolean
): ResourceSelection {
  const result: ResourceSelection = {
    crisis: [],
    condition: [],
    state: null,
    rural: null,
    categories: []
  };
  
  // Always include crisis resources if in crisis
  if (isCrisis && nationalResources.crisis) {
    result.crisis = nationalResources.crisis;
    result.categories.push('crisis');
  }
  
  // Add state-specific resources
  if (state && stateResources[state]) {
    result.state = stateResources[state];
    result.categories.push('state');
  }
  
  // Add condition-specific resources
  for (const condition of conditions) {
    if (conditionResources[condition]) {
      result.condition.push(...conditionResources[condition].resources);
      result.categories.push(condition);
    }
  }
  
  // Add domestic violence resources if abuse detected
  if ((conditions.includes('abuse') || conditions.includes('domesticViolence')) && nationalResources.domesticViolence) {
    result.condition.push(...nationalResources.domesticViolence);
    result.categories.push('domesticViolence');
  }
  
  // Add substance use resources if detected
  if ((conditions.includes('substanceUse') || conditions.includes('addiction')) && nationalResources.substanceUse) {
    result.condition.push(...nationalResources.substanceUse);
    result.categories.push('substanceUse');
  }
  
  // Add eating disorder resources if detected
  if (conditions.includes('eatingDisorder') && nationalResources.eatingDisorders) {
    result.condition.push(...nationalResources.eatingDisorders);
    result.categories.push('eatingDisorder');
  }
  
  // Add LGBTQ+ resources if detected
  if (conditions.includes('lgbtq') && nationalResources.lgbtq) {
    result.condition.push(...nationalResources.lgbtq);
    result.categories.push('lgbtq');
  }
  
  // Add veteran resources if detected
  if (conditions.includes('veteran') && nationalResources.veterans) {
    result.condition.push(...nationalResources.veterans);
    result.categories.push('veteran');
  }
  
  // Add rural resources if rural context
  if (isRural && ruralResources) {
    result.rural = ruralResources;
    result.categories.push('rural');
  }
  
  // Deduplicate condition resources
  const seen = new Set<string>();
  result.condition = result.condition.filter(r => {
    if (seen.has(r.name)) return false;
    seen.add(r.name);
    return true;
  });
  
  // Sort by priority
  result.crisis.sort((a, b) => (a.priority || 99) - (b.priority || 99));
  result.condition.sort((a, b) => (a.priority || 99) - (b.priority || 99));
  
  return result;
}

// ============================================================================
// FORMAT FOR AI RESPONSE
// ============================================================================

export function formatResourcesForResponse(selection: ResourceSelection): string {
  const parts: string[] = [];
  
  if (selection.crisis.length > 0) {
    const crisisLines = selection.crisis.slice(0, 3).map(r => {
      let line = `• ${r.name}`;
      if (r.phone) line += `: ${r.phone}`;
      if (r.text) line += ` (Text: ${r.text})`;
      return line;
    });
    parts.push(`**Immediate Support:**\n${crisisLines.join('\n')}`);
  }
  
  if (selection.state) {
    parts.push(`**${selection.state.name} Resources:**\n• Crisis: ${selection.state.crisis}\n• ${selection.state.mentalHealth}`);
    if (selection.state.ruralNote) {
      parts.push(`*Note: ${selection.state.ruralNote}*`);
    }
  }
  
  if (selection.condition.length > 0) {
    const conditionLines = selection.condition.slice(0, 4).map(r => {
      let line = `• ${r.name}`;
      if (r.phone) line += `: ${r.phone}`;
      else if (r.website) line += `: ${r.website}`;
      return line;
    });
    parts.push(`**Specialized Support:**\n${conditionLines.join('\n')}`);
  }
  
  if (selection.rural) {
    const telehealthLines = selection.rural.telehealth?.slice(0, 2).map((r: Resource) => 
      `• ${r.name}: ${r.website}`
    ) || [];
    if (telehealthLines.length > 0) {
      parts.push(`**Telehealth Options (No Travel Required):**\n${telehealthLines.join('\n')}`);
    }
  }
  
  return parts.join('\n\n');
}

// ============================================================================
// GROUNDING FOR CONDITION
// ============================================================================

export function getGroundingForCondition(condition: string): string[] {
  if (conditionResources[condition]) {
    return conditionResources[condition].grounding;
  }
  return ['5-4-3-2-1', 'box_breathing']; // defaults
}
