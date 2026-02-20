/**
 * BPD Splitting Grounding Module
 * 
 * Specialized module for helping people with BPD during splitting episodes.
 * Uses entropy-based regulation to help restore integrated thinking.
 * 
 * Splitting = black-and-white thinking where people/situations are seen as
 * all good or all bad. This module helps restore nuance and integration.
 */

export interface SplittingAnalysis {
  isSplitting: boolean;
  splittingIntensity: number;  // 0-1
  splittingTarget: "self" | "other" | "situation" | "world" | null;
  polarization: "idealization" | "devaluation" | null;
  entropyContribution: number;
  groundingProtocol: SplittingGroundingProtocol;
}

export interface SplittingGroundingProtocol {
  name: string;
  steps: string[];
  dialecticalStatements: string[];
  integrationPrompts: string[];
  validationFirst: string;
  urgency: "low" | "medium" | "high";
}

// Splitting detection patterns
const splittingIndicators = {
  // Black-and-white language
  absoluteTerms: [
    "always", "never", "everyone", "no one", "completely", "totally",
    "perfect", "horrible", "best", "worst", "hate", "love", "all", "nothing"
  ],
  
  // Idealization patterns
  idealization: [
    "perfect", "amazing", "best ever", "only one who", "saved me",
    "soulmate", "meant to be", "finally found", "everything i need",
    "can do no wrong", "angel", "saint"
  ],
  
  // Devaluation patterns
  devaluation: [
    "hate", "worst", "evil", "monster", "toxic", "abuser", "narcissist",
    "dead to me", "nothing to me", "wish they were dead", "ruined my life",
    "never cared", "always lied", "fake", "manipulator"
  ],
  
  // Self-splitting
  selfSplitting: [
    "i'm worthless", "i'm disgusting", "i'm the worst", "i'm perfect",
    "i'm a monster", "i'm evil", "i'm nothing", "i'm everything",
    "i hate myself", "i'm amazing", "i'm garbage", "i'm trash"
  ],
  
  // Rapid shift indicators
  rapidShift: [
    "but now", "suddenly", "changed", "different person", "not who i thought",
    "realized", "finally see", "true colors", "mask off"
  ]
};

// Grounding protocols for different splitting types
const groundingProtocols: Record<string, SplittingGroundingProtocol> = {
  self_devaluation: {
    name: "Self-Integration Grounding",
    steps: [
      "Notice the intensity of the self-criticism. Rate it 1-10.",
      "Place one hand on your heart. Feel your heartbeat.",
      "Say aloud: 'I am having the thought that I am [the criticism].'",
      "Now say: 'I am a person who sometimes [the behavior], AND I am also a person who [positive quality].'",
      "Name three things you did today, no matter how small.",
      "Remind yourself: 'I am not my worst moment. I am not my best moment. I am all of my moments.'"
    ],
    dialecticalStatements: [
      "I can be struggling AND still be worthy of compassion.",
      "I made a mistake AND I am still a whole person.",
      "I feel like I'm the worst AND this feeling will pass.",
      "I'm in pain AND I can get through this."
    ],
    integrationPrompts: [
      "What would you say to a friend who felt this way about themselves?",
      "Can you remember a time when you felt differently about yourself?",
      "What is one small thing about yourself that isn't all bad or all good?"
    ],
    validationFirst: "The pain you're feeling about yourself is real. Self-criticism this intense is exhausting. You're not broken for feeling this way.",
    urgency: "high"
  },
  
  other_devaluation: {
    name: "Relationship Nuance Grounding",
    steps: [
      "Notice the intensity of your feelings about this person. Rate it 1-10.",
      "Take three slow breaths. Feel your feet on the ground.",
      "Say: 'Right now, I am feeling [emotion] about [person].'",
      "Ask yourself: 'Is this how I've always felt, or is this feeling from right now?'",
      "Try to name ONE neutral thing about this person (not good or bad, just neutral).",
      "Remind yourself: 'People are complicated. This person has hurt me AND may have other qualities too.'"
    ],
    dialecticalStatements: [
      "This person hurt me AND they may not be entirely evil.",
      "I can be angry at someone AND still recognize they're human.",
      "This relationship is painful AND it may have had some good moments too.",
      "I can protect myself AND not need to see them as a monster to do so."
    ],
    integrationPrompts: [
      "Was there ever a time when you felt differently about this person?",
      "What might have been going on for them (not to excuse, just to understand)?",
      "Can you hold both the hurt AND any complexity at the same time?"
    ],
    validationFirst: "Your anger and pain about this person are valid. When someone hurts us, it makes sense to see them negatively. You don't have to forgive or forget to find some nuance.",
    urgency: "medium"
  },
  
  other_idealization: {
    name: "Balanced Connection Grounding",
    steps: [
      "Notice how intensely positive you feel. Rate it 1-10.",
      "Take a breath. Feel your body in this moment.",
      "Say: 'I am feeling very positive about [person] right now.'",
      "Ask yourself: 'Is this person perfect, or am I seeing them through a filter right now?'",
      "Try to name ONE thing about this person that is neutral or could be a flaw.",
      "Remind yourself: 'Real connection includes accepting someone's imperfections.'"
    ],
    dialecticalStatements: [
      "I can deeply care about someone AND acknowledge they're not perfect.",
      "This person is wonderful to me AND they are still human.",
      "I can feel intense connection AND stay grounded in reality.",
      "Love doesn't require perfection from either of us."
    ],
    integrationPrompts: [
      "What might happen if this person disappoints you?",
      "Can you love someone without needing them to be perfect?",
      "What would a balanced view of this person look like?"
    ],
    validationFirst: "It feels amazing to connect with someone who feels so right. That feeling is real and valid. Staying grounded doesn't mean diminishing the connection.",
    urgency: "low"
  },
  
  situation_splitting: {
    name: "Situation Integration Grounding",
    steps: [
      "Notice how you're viewing this situation. Is it all good or all bad?",
      "Take a breath. Ground your feet.",
      "Say: 'Right now, this situation feels [description].'",
      "Ask: 'What is one thing about this situation that is neutral?'",
      "Ask: 'What is one thing that could be different from how I'm seeing it?'",
      "Remind yourself: 'Most situations have multiple aspects. I can see more than one.'"
    ],
    dialecticalStatements: [
      "This situation is difficult AND it may have some opportunities.",
      "Things feel terrible AND this moment will pass.",
      "This is hard AND I've gotten through hard things before.",
      "I can hate this situation AND still cope with it."
    ],
    integrationPrompts: [
      "What would someone else see in this situation?",
      "Is there any part of this that isn't entirely bad?",
      "What might you think about this situation in a week? A month?"
    ],
    validationFirst: "When things feel overwhelming, it's natural to see them in extremes. Your distress about this situation is valid.",
    urgency: "medium"
  },
  
  world_splitting: {
    name: "World View Grounding",
    steps: [
      "Notice the thought: 'Everything is [bad/hopeless/ruined].'",
      "Take a breath. Look around the room.",
      "Name three things you can see that are neutral (not good or bad).",
      "Say: 'Right now, I am having the thought that everything is [description].'",
      "Ask: 'Is there anything in my life right now that isn't part of this feeling?'",
      "Remind yourself: 'My feelings are real, AND they're not the whole picture.'"
    ],
    dialecticalStatements: [
      "The world feels dark AND there are still moments of light.",
      "Things feel hopeless AND I've felt this way before and it changed.",
      "Everything feels wrong AND some things are still okay.",
      "I can feel despair AND still take one small step."
    ],
    integrationPrompts: [
      "Is there one small thing that isn't terrible right now?",
      "What would you tell someone else who felt this way?",
      "Can you remember a time when things felt different?"
    ],
    validationFirst: "When pain is this big, it colors everything. The world feeling dark is a sign of how much you're hurting, not a sign that the world is actually all dark.",
    urgency: "high"
  }
};

/**
 * Analyze message for splitting patterns
 */
export function analyzeSplitting(message: string, entropy: number): SplittingAnalysis {
  const text = message.toLowerCase();
  
  // Count indicators
  let absoluteCount = 0;
  let idealizationCount = 0;
  let devaluationCount = 0;
  let selfSplittingCount = 0;
  let rapidShiftCount = 0;
  
  for (const term of splittingIndicators.absoluteTerms) {
    if (text.includes(term)) absoluteCount++;
  }
  
  for (const term of splittingIndicators.idealization) {
    if (text.includes(term)) idealizationCount++;
  }
  
  for (const term of splittingIndicators.devaluation) {
    if (text.includes(term)) devaluationCount++;
  }
  
  for (const term of splittingIndicators.selfSplitting) {
    if (text.includes(term)) selfSplittingCount++;
  }
  
  for (const term of splittingIndicators.rapidShift) {
    if (text.includes(term)) rapidShiftCount++;
  }
  
  // Calculate splitting intensity
  const totalIndicators = absoluteCount + idealizationCount + devaluationCount + selfSplittingCount + rapidShiftCount;
  const splittingIntensity = Math.min(1, totalIndicators / 10);
  
  // Determine if splitting is occurring
  const isSplitting = splittingIntensity > 0.3 || absoluteCount >= 3 || 
                      idealizationCount >= 2 || devaluationCount >= 2 || selfSplittingCount >= 2;
  
  // Determine target and polarization
  let splittingTarget: SplittingAnalysis["splittingTarget"] = null;
  let polarization: SplittingAnalysis["polarization"] = null;
  
  if (isSplitting) {
    // Determine target
    if (selfSplittingCount > 0) {
      splittingTarget = "self";
    } else if (text.includes("they") || text.includes("him") || text.includes("her") || 
               text.includes("my partner") || text.includes("my friend") || text.includes("my mom") ||
               text.includes("my dad") || text.includes("my boss")) {
      splittingTarget = "other";
    } else if (text.includes("everything") || text.includes("the world") || text.includes("life")) {
      splittingTarget = "world";
    } else {
      splittingTarget = "situation";
    }
    
    // Determine polarization
    if (idealizationCount > devaluationCount) {
      polarization = "idealization";
    } else if (devaluationCount > 0) {
      polarization = "devaluation";
    }
  }
  
  // Select appropriate grounding protocol
  let protocolKey = "situation_splitting";
  if (splittingTarget === "self" && polarization === "devaluation") {
    protocolKey = "self_devaluation";
  } else if (splittingTarget === "other" && polarization === "devaluation") {
    protocolKey = "other_devaluation";
  } else if (splittingTarget === "other" && polarization === "idealization") {
    protocolKey = "other_idealization";
  } else if (splittingTarget === "world") {
    protocolKey = "world_splitting";
  }
  
  // Calculate entropy contribution (splitting increases entropy)
  const entropyContribution = splittingIntensity * 0.3;
  
  return {
    isSplitting,
    splittingIntensity,
    splittingTarget,
    polarization,
    entropyContribution,
    groundingProtocol: groundingProtocols[protocolKey]
  };
}

/**
 * Format splitting analysis for LLM context
 */
export function formatSplittingForPrompt(analysis: SplittingAnalysis): string {
  if (!analysis.isSplitting) return "";
  
  const protocol = analysis.groundingProtocol;
  
  let output = "\n\n[BPD SPLITTING DETECTED - USE DIALECTICAL APPROACH]\n";
  output += `Splitting target: ${analysis.splittingTarget}\n`;
  output += `Polarization: ${analysis.polarization}\n`;
  output += `Intensity: ${(analysis.splittingIntensity * 100).toFixed(0)}%\n\n`;
  
  output += `VALIDATION FIRST: ${protocol.validationFirst}\n\n`;
  
  output += "DIALECTICAL STATEMENTS TO USE:\n";
  for (const statement of protocol.dialecticalStatements) {
    output += `- "${statement}"\n`;
  }
  
  output += "\nGROUNDING STEPS (guide user through these):\n";
  for (let i = 0; i < protocol.steps.length; i++) {
    output += `${i + 1}. ${protocol.steps[i]}\n`;
  }
  
  output += "\nINTEGRATION PROMPTS (use gently):\n";
  for (const prompt of protocol.integrationPrompts) {
    output += `- ${prompt}\n`;
  }
  
  output += "\nKEY: Do NOT challenge the splitting directly. Validate first, then gently introduce nuance. ";
  output += "Use 'AND' instead of 'BUT'. Help them hold two truths at once.\n";
  
  return output;
}
