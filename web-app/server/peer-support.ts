/**
 * Peer Support Matching System
 * 
 * Connects users with similar experiences for community support.
 * Includes comprehensive safety guardrails, crisis detection,
 * and moderation systems.
 */

import { EntropyState } from "./reunity";

export interface PeerProfile {
  id: string;
  odId: string;  // User's open ID
  displayName: string;  // Anonymous display name
  createdAt: Date;
  updatedAt: Date;
  
  // Experience tags for matching
  experiences: ExperienceTag[];
  
  // Support preferences
  preferences: SupportPreferences;
  
  // Safety settings
  safetySettings: SafetySettings;
  
  // Status
  isActive: boolean;
  lastActive: Date;
  isBanned: boolean;
  banReason?: string;
}

export type ExperienceTag = 
  | "anxiety"
  | "depression"
  | "ptsd"
  | "cptsd"
  | "bpd"
  | "bipolar"
  | "ocd"
  | "eating_disorder"
  | "substance_recovery"
  | "grief"
  | "domestic_violence"
  | "sexual_assault"
  | "childhood_trauma"
  | "dissociation"
  | "self_harm_recovery"
  | "suicidal_ideation_recovery"
  | "caregiver_burnout"
  | "chronic_illness"
  | "lgbtq"
  | "neurodivergent"
  | "rural_isolation"
  | "religious_trauma"
  | "relationship_abuse"
  | "parental_alienation"
  | "workplace_trauma";

export interface SupportPreferences {
  // What kind of support they want
  wantToGiveSupport: boolean;
  wantToReceiveSupport: boolean;
  
  // Communication preferences
  preferTextChat: boolean;
  preferVoice: boolean;
  preferVideo: boolean;
  
  // Availability
  timezone: string;
  availableTimes: string[];  // e.g., "weekday_evenings", "weekends"
  
  // Matching preferences
  preferSameGender: boolean;
  preferSimilarAge: boolean;
  ageRange?: { min: number; max: number };
  
  // Topics they're comfortable discussing
  comfortableTopics: ExperienceTag[];
  
  // Topics they want to avoid
  triggerTopics: string[];
}

export interface SafetySettings {
  // Crisis protocol
  emergencyContact?: string;
  allowCrisisEscalation: boolean;
  
  // Privacy
  shareLocation: boolean;  // For resource matching
  locationState?: string;
  
  // Boundaries
  maxConnectionsPerWeek: number;
  cooldownBetweenSessions: number;  // hours
  
  // Reporting
  hasReportedOthers: boolean;
  hasBeenReported: boolean;
  reportCount: number;
}

export interface PeerConnection {
  id: string;
  requesterId: string;
  responderId: string;
  status: "pending" | "accepted" | "declined" | "blocked" | "ended";
  createdAt: Date;
  updatedAt: Date;
  
  // Match quality
  matchScore: number;
  sharedExperiences: ExperienceTag[];
  
  // Session tracking
  sessionCount: number;
  lastSessionAt?: Date;
  totalMinutes: number;
  
  // Safety
  flaggedForReview: boolean;
  flagReason?: string;
}

export interface PeerMessage {
  id: string;
  connectionId: string;
  senderId: string;
  content: string;
  timestamp: Date;
  
  // Safety analysis
  entropyLevel: number;
  crisisDetected: boolean;
  flagged: boolean;
  flagReason?: string;
}

export interface ModerationAction {
  id: string;
  targetUserId: string;
  reporterId?: string;
  action: "warning" | "temporary_ban" | "permanent_ban" | "review_required";
  reason: string;
  timestamp: Date;
  resolvedAt?: Date;
  resolvedBy?: string;
  resolution?: string;
}

// Safety guardrails
const CRISIS_KEYWORDS = [
  "kill myself", "suicide", "end it all", "want to die", "better off dead",
  "no reason to live", "going to do it", "goodbye", "final", "can't go on",
  "hurt myself", "self harm", "cutting", "overdose"
];

const HARMFUL_CONTENT_KEYWORDS = [
  "meet in person", "send photos", "your address", "where do you live",
  "how old are you", "send money", "venmo", "cashapp", "paypal",
  "romantic", "dating", "relationship with you"
];

const MANDATORY_RESOURCES = {
  crisis: {
    name: "988 Suicide & Crisis Lifeline",
    phone: "988",
    text: "Text 988"
  },
  selfHarm: {
    name: "Crisis Text Line",
    text: "Text HOME to 741741"
  },
  domesticViolence: {
    name: "National DV Hotline",
    phone: "1-800-799-7233"
  }
};

/**
 * Calculate match score between two profiles
 */
export function calculateMatchScore(profile1: PeerProfile, profile2: PeerProfile): number {
  let score = 0;
  const maxScore = 100;
  
  // Shared experiences (40 points max)
  const sharedExperiences = profile1.experiences.filter(e => 
    profile2.experiences.includes(e)
  );
  score += Math.min(40, sharedExperiences.length * 10);
  
  // Complementary support preferences (20 points)
  if (profile1.preferences.wantToGiveSupport && profile2.preferences.wantToReceiveSupport) {
    score += 10;
  }
  if (profile1.preferences.wantToReceiveSupport && profile2.preferences.wantToGiveSupport) {
    score += 10;
  }
  
  // Communication compatibility (15 points)
  if (profile1.preferences.preferTextChat && profile2.preferences.preferTextChat) {
    score += 5;
  }
  if (profile1.preferences.preferVoice && profile2.preferences.preferVoice) {
    score += 5;
  }
  if (profile1.preferences.preferVideo && profile2.preferences.preferVideo) {
    score += 5;
  }
  
  // Availability overlap (15 points)
  const sharedTimes = profile1.preferences.availableTimes.filter(t =>
    profile2.preferences.availableTimes.includes(t)
  );
  score += Math.min(15, sharedTimes.length * 5);
  
  // No trigger conflicts (10 points)
  const hasConflict = profile1.preferences.triggerTopics.some(t =>
    profile2.preferences.comfortableTopics.includes(t as ExperienceTag)
  ) || profile2.preferences.triggerTopics.some(t =>
    profile1.preferences.comfortableTopics.includes(t as ExperienceTag)
  );
  if (!hasConflict) {
    score += 10;
  }
  
  return Math.min(maxScore, score);
}

/**
 * Find potential matches for a user
 */
export function findMatches(
  userProfile: PeerProfile,
  allProfiles: PeerProfile[],
  limit: number = 10
): { profile: PeerProfile; score: number; sharedExperiences: ExperienceTag[] }[] {
  const matches = allProfiles
    .filter(p => 
      p.id !== userProfile.id && 
      p.isActive && 
      !p.isBanned
    )
    .map(profile => ({
      profile,
      score: calculateMatchScore(userProfile, profile),
      sharedExperiences: userProfile.experiences.filter(e => 
        profile.experiences.includes(e)
      )
    }))
    .filter(m => m.score >= 30)  // Minimum match threshold
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
  
  return matches;
}

/**
 * Analyze message for safety concerns
 */
export function analyzeMessageSafety(message: string): {
  isSafe: boolean;
  crisisDetected: boolean;
  harmfulContent: boolean;
  flagReason?: string;
  requiredResources?: typeof MANDATORY_RESOURCES[keyof typeof MANDATORY_RESOURCES];
} {
  const lowerMessage = message.toLowerCase();
  
  // Check for crisis indicators
  const crisisDetected = CRISIS_KEYWORDS.some(kw => lowerMessage.includes(kw));
  
  // Check for harmful content
  const harmfulContent = HARMFUL_CONTENT_KEYWORDS.some(kw => lowerMessage.includes(kw));
  
  let flagReason: string | undefined;
  let requiredResources: typeof MANDATORY_RESOURCES[keyof typeof MANDATORY_RESOURCES] | undefined;
  
  if (crisisDetected) {
    if (lowerMessage.includes("suicide") || lowerMessage.includes("kill myself") || lowerMessage.includes("want to die")) {
      flagReason = "Suicidal ideation detected";
      requiredResources = MANDATORY_RESOURCES.crisis;
    } else if (lowerMessage.includes("self harm") || lowerMessage.includes("cutting")) {
      flagReason = "Self-harm content detected";
      requiredResources = MANDATORY_RESOURCES.selfHarm;
    }
  }
  
  if (harmfulContent) {
    flagReason = "Potentially harmful content detected";
  }
  
  return {
    isSafe: !crisisDetected && !harmfulContent,
    crisisDetected,
    harmfulContent,
    flagReason,
    requiredResources
  };
}

/**
 * Generate anonymous display name
 */
export function generateDisplayName(): string {
  const adjectives = [
    "Gentle", "Brave", "Kind", "Calm", "Warm", "Steady", "Hopeful", "Strong",
    "Peaceful", "Resilient", "Caring", "Patient", "Wise", "Tender", "Bright"
  ];
  
  const nouns = [
    "River", "Mountain", "Star", "Moon", "Sun", "Ocean", "Forest", "Meadow",
    "Cloud", "Breeze", "Light", "Dawn", "Sky", "Rain", "Bloom"
  ];
  
  const adj = adjectives[Math.floor(Math.random() * adjectives.length)];
  const noun = nouns[Math.floor(Math.random() * nouns.length)];
  const num = Math.floor(Math.random() * 1000);
  
  return `${adj}${noun}${num}`;
}

/**
 * Get community guidelines
 */
export function getCommunityGuidelines(): string {
  return `
PEER SUPPORT COMMUNITY GUIDELINES

Welcome to the ReUnity Peer Support Community. This is a safe space for mutual support.

WHAT THIS IS:
- A place to connect with others who understand your experiences
- A space for mutual support, not professional treatment
- An anonymous community focused on healing together

WHAT THIS IS NOT:
- A replacement for professional mental health care
- A dating or social networking platform
- A place to give medical or legal advice

COMMUNITY RULES:

1. RESPECT BOUNDARIES
   - Ask before discussing sensitive topics
   - Accept "no" without question
   - Don't pressure others to share more than they're comfortable with

2. MAINTAIN ANONYMITY
   - Don't share personal identifying information
   - Don't ask for others' real names, locations, or contact info
   - Don't share photos or request photos

3. SUPPORT, DON'T FIX
   - Listen more than you advise
   - Validate feelings without trying to solve everything
   - Share your experience, not prescriptions

4. CRISIS PROTOCOL
   - If someone is in crisis, encourage professional resources
   - Share crisis hotline numbers when appropriate
   - Report concerning content to moderators

5. NO HARMFUL CONTENT
   - No graphic descriptions of self-harm methods
   - No encouragement of harmful behaviors
   - No romantic or sexual content

6. REPORT CONCERNS
   - Use the report button for guideline violations
   - Moderators review all reports within 24 hours
   - False reports may result in account action

REMEMBER:
- You are not alone
- Your experiences are valid
- Healing is possible
- Help is available

Crisis Resources:
- 988 Suicide & Crisis Lifeline: Call or text 988
- Crisis Text Line: Text HOME to 741741
- National DV Hotline: 1-800-799-7233
`;
}

/**
 * Create welcome message for new peer connection
 */
export function createWelcomeMessage(sharedExperiences: ExperienceTag[]): string {
  const experienceLabels: Record<ExperienceTag, string> = {
    anxiety: "anxiety",
    depression: "depression",
    ptsd: "PTSD",
    cptsd: "complex trauma",
    bpd: "BPD",
    bipolar: "bipolar",
    ocd: "OCD",
    eating_disorder: "eating disorder recovery",
    substance_recovery: "substance recovery",
    grief: "grief",
    domestic_violence: "domestic violence",
    sexual_assault: "sexual assault recovery",
    childhood_trauma: "childhood trauma",
    dissociation: "dissociation",
    self_harm_recovery: "self-harm recovery",
    suicidal_ideation_recovery: "suicidal ideation recovery",
    caregiver_burnout: "caregiver burnout",
    chronic_illness: "chronic illness",
    lgbtq: "LGBTQ+ experiences",
    neurodivergent: "neurodivergence",
    rural_isolation: "rural isolation",
    religious_trauma: "religious trauma",
    relationship_abuse: "relationship abuse",
    parental_alienation: "parental alienation",
    workplace_trauma: "workplace trauma"
  };
  
  const shared = sharedExperiences.map(e => experienceLabels[e]).join(", ");
  
  return `Welcome to your peer support connection!

You've been matched because you both have experience with: ${shared}

GUIDELINES FOR THIS CONVERSATION:
- Be kind and respectful
- Listen as much as you share
- Respect each other's boundaries
- You can end the conversation at any time

REMEMBER:
- This is peer support, not professional therapy
- If either of you is in crisis, please reach out to 988
- You're both here because you understand what it's like

Take your time getting to know each other. There's no pressure.`;
}

/**
 * Get experience tag descriptions for UI
 */
export function getExperienceDescriptions(): Record<ExperienceTag, string> {
  return {
    anxiety: "Living with anxiety disorders (GAD, social anxiety, panic disorder)",
    depression: "Experiencing depression or depressive episodes",
    ptsd: "Post-traumatic stress from specific events",
    cptsd: "Complex trauma from prolonged or repeated experiences",
    bpd: "Borderline personality disorder experiences",
    bipolar: "Bipolar disorder and mood cycling",
    ocd: "Obsessive-compulsive disorder and intrusive thoughts",
    eating_disorder: "Recovery from eating disorders",
    substance_recovery: "Recovery from substance use",
    grief: "Processing grief and loss",
    domestic_violence: "Surviving domestic violence or abuse",
    sexual_assault: "Surviving sexual assault or abuse",
    childhood_trauma: "Processing childhood trauma or neglect",
    dissociation: "Experiencing dissociation or dissociative disorders",
    self_harm_recovery: "Recovery from self-harm behaviors",
    suicidal_ideation_recovery: "Recovery from suicidal thoughts",
    caregiver_burnout: "Burnout from caregiving responsibilities",
    chronic_illness: "Living with chronic illness or pain",
    lgbtq: "LGBTQ+ identity and related experiences",
    neurodivergent: "Autism, ADHD, or other neurodivergence",
    rural_isolation: "Isolation from living in rural areas",
    religious_trauma: "Trauma from religious experiences or leaving religion",
    relationship_abuse: "Surviving emotionally abusive relationships",
    parental_alienation: "Experiencing parental alienation",
    workplace_trauma: "Trauma from workplace harassment or abuse"
  };
}
