/**
 * Rural-Specific Support Module
 * 
 * Specialized interventions for people in rural and remote areas facing:
 * - Geographic isolation
 * - Limited access to services
 * - Domestic violence in remote locations
 * - Agricultural/farming stress
 * - Lack of anonymity in small communities
 * - Cultural barriers to seeking help
 * 
 * Special focus on helping victims who may not recognize abuse patterns.
 */

export interface RuralContext {
  isRural: boolean;
  isolationLevel: "moderate" | "high" | "extreme";
  accessToServices: "limited" | "very_limited" | "none";
  domesticViolenceRisk: boolean;
  recognizesAbuse: boolean;
  safetyConstraints: string[];
  culturalFactors: string[];
}

export interface RuralIntervention {
  category: string;
  validation: string;
  psychoeducation: string[];
  safetyPlanning: string[];
  resources: RuralResource[];
  selfCareStrategies: string[];
  communityBuilding: string[];
}

export interface RuralResource {
  name: string;
  type: "hotline" | "text" | "online" | "telehealth" | "local";
  contact: string;
  note: string;
  safetyNote?: string;
}

// Rural isolation indicators
const ruralIndicators = {
  geographic: [
    "rural", "remote", "farm", "ranch", "country", "isolated", "middle of nowhere",
    "miles from", "nearest town", "no neighbors", "off grid", "mountain", "prairie",
    "no cell service", "satellite internet", "dirt road", "gravel road"
  ],
  
  accessBarriers: [
    "no therapist", "no counselor", "hours away", "can't get there", "no transportation",
    "no bus", "no uber", "can't drive", "he has the car", "she has the car",
    "no childcare", "can't leave", "stuck here", "trapped", "no way out"
  ],
  
  communityPressure: [
    "everyone knows everyone", "small town", "gossip", "reputation", "family name",
    "church", "congregation", "community", "what will people think", "can't tell anyone",
    "no privacy", "word gets around"
  ],
  
  agriculturalStress: [
    "harvest", "drought", "crop", "livestock", "cattle", "farm debt", "lost the farm",
    "weather", "prices", "market", "equipment", "loan", "foreclosure"
  ]
};

// Domestic violence indicators (especially for those who may not recognize it)
const dvIndicators = {
  // Control patterns
  control: [
    "won't let me", "doesn't let me", "not allowed to", "permission", "have to ask",
    "checks my phone", "tracks me", "follows me", "monitors", "controls the money",
    "gives me an allowance", "takes my paycheck", "hides the keys", "disabled the car"
  ],
  
  // Isolation patterns
  isolation: [
    "cut off from", "can't see my family", "friends aren't allowed", "no contact with",
    "moved me away", "took me from", "don't have anyone", "he's all i have",
    "she's all i have", "no one else"
  ],
  
  // Minimization (victim may not recognize)
  minimization: [
    "it's not that bad", "only when drinking", "doesn't mean it", "loves me really",
    "just gets angry", "just stressed", "my fault", "i provoked", "i shouldn't have",
    "i made them", "normal for couples", "every relationship", "at least doesn't hit",
    "could be worse"
  ],
  
  // Physical indicators
  physical: [
    "hit", "punch", "slap", "kick", "choke", "strangle", "threw", "pushed",
    "grabbed", "bruise", "mark", "hurt me", "scared of", "when angry"
  ],
  
  // Coercive control
  coercive: [
    "threatens", "if i leave", "kill myself if", "take the kids", "no one will believe",
    "ruin you", "destroy you", "make you pay", "you'll regret", "you'll be sorry",
    "nowhere to go", "nothing without me"
  ],
  
  // Children involved
  children: [
    "kids", "children", "baby", "pregnant", "custody", "take them away",
    "bad mother", "bad father", "unfit parent"
  ]
};

// Psychoeducation about abuse patterns (gentle, non-judgmental)
const abuseEducation = {
  whatIsAbuse: [
    "Abuse isn't just physical violence. It includes patterns of control, isolation, and fear.",
    "In healthy relationships, both people feel free to make choices, see friends and family, and have their own money.",
    "It's not your fault. Abuse is about the abuser's need for control, not anything you did.",
    "Many people in abusive situations don't recognize it at first. The patterns often start small and build over time.",
    "Feeling confused about whether it's 'really abuse' is common. Abusers often make their partners question reality."
  ],
  
  whyStay: [
    "There are many reasons people stay in difficult relationships. It doesn't mean you're weak or stupid.",
    "Fear, financial dependence, children, love, hope things will change, isolation, lack of options - these are all real barriers.",
    "Leaving is often the most dangerous time. Your caution makes sense.",
    "You know your situation best. Only you can decide what's safe and right for you."
  ],
  
  ruralChallenges: [
    "In rural areas, leaving can feel impossible. Limited resources, no anonymity, and geographic isolation are real barriers.",
    "Small communities can make it hard to seek help without everyone knowing.",
    "Being far from services doesn't mean you're without options. There are resources designed for rural situations.",
    "Your safety matters, even if help feels far away."
  ]
};

// Rural-specific resources
const ruralResources: RuralResource[] = [
  {
    name: "National Domestic Violence Hotline",
    type: "hotline",
    contact: "1-800-799-7233",
    note: "24/7, can help create safety plans for rural situations",
    safetyNote: "Call from a safe phone. Clear call history after."
  },
  {
    name: "Crisis Text Line",
    type: "text",
    contact: "Text HOME to 741741",
    note: "Text-based support when calling isn't safe or possible",
    safetyNote: "Delete text thread after. Use a safe phone."
  },
  {
    name: "National Sexual Assault Hotline",
    type: "hotline",
    contact: "1-800-656-4673",
    note: "RAINN - 24/7 support",
    safetyNote: "Available via phone or online chat"
  },
  {
    name: "Farm Aid Hotline",
    type: "hotline",
    contact: "1-800-327-6243",
    note: "Support for farming families facing stress, financial crisis",
    safetyNote: "Confidential support for agricultural stress"
  },
  {
    name: "Rural Health Information Hub",
    type: "online",
    contact: "ruralhealthinfo.org",
    note: "Find telehealth and rural health services in your area",
    safetyNote: "Use private browsing mode"
  },
  {
    name: "WomensLaw.org",
    type: "online",
    contact: "womenslaw.org",
    note: "Legal information, state-by-state resources, email hotline",
    safetyNote: "Has safety tips for using the internet safely"
  },
  {
    name: "DomesticShelters.org",
    type: "online",
    contact: "domesticshelters.org",
    note: "Find shelters, even in rural areas",
    safetyNote: "Use private browsing. Some shelters offer transportation."
  },
  {
    name: "StrongHearts Native Helpline",
    type: "hotline",
    contact: "1-844-762-8483",
    note: "For Native Americans, Alaska Natives - culturally appropriate support",
    safetyNote: "24/7, anonymous, confidential"
  }
];

// Safety planning for rural DV situations
const ruralSafetyPlanning = {
  immediate: [
    "Identify the safest room in your home (one with a lock, phone, window for escape).",
    "Keep important documents hidden but accessible (ID, birth certificates, financial records).",
    "Memorize important phone numbers in case your phone is taken.",
    "If you have a safe neighbor, even miles away, establish a code word or signal.",
    "Keep a bag packed and hidden (at a neighbor's, in your car, in a barn)."
  ],
  
  communication: [
    "If possible, get a prepaid phone and hide it somewhere safe.",
    "Use library computers or a trusted friend's phone for sensitive searches.",
    "Clear browser history, or use private/incognito mode.",
    "Be aware of tracking apps or devices on your phone or car.",
    "Create a code word with someone you trust to signal you need help."
  ],
  
  transportation: [
    "Keep gas in your car when possible.",
    "Know alternative routes out of your area.",
    "Some shelters offer transportation, even to rural areas - ask when you call.",
    "Identify trusted people who might help with transportation in an emergency."
  ],
  
  children: [
    "Teach children to call 911 and give your address.",
    "Have a safe place for children to go during an incident (neighbor, hiding spot).",
    "Don't tell children the safety plan if they might tell the abuser.",
    "Document injuries and incidents when safe to do so."
  ],
  
  financial: [
    "If possible, set aside small amounts of cash over time.",
    "Know your financial situation (accounts, debts, assets).",
    "Some DV organizations can help with emergency funds.",
    "You may be entitled to financial support even if you leave."
  ]
};

/**
 * Analyze message for rural context and DV indicators
 */
export function analyzeRuralContext(message: string, userContext?: any): RuralContext {
  const text = message.toLowerCase();
  
  // Check for rural indicators
  let ruralScore = 0;
  for (const indicator of ruralIndicators.geographic) {
    if (text.includes(indicator)) ruralScore += 2;
  }
  for (const indicator of ruralIndicators.accessBarriers) {
    if (text.includes(indicator)) ruralScore += 1;
  }
  for (const indicator of ruralIndicators.communityPressure) {
    if (text.includes(indicator)) ruralScore += 1;
  }
  
  // Check for DV indicators
  let dvScore = 0;
  let controlScore = 0;
  let minimizationScore = 0;
  let physicalScore = 0;
  let childrenInvolved = false;
  
  for (const indicator of dvIndicators.control) {
    if (text.includes(indicator)) { dvScore += 2; controlScore++; }
  }
  for (const indicator of dvIndicators.isolation) {
    if (text.includes(indicator)) { dvScore += 2; controlScore++; }
  }
  for (const indicator of dvIndicators.minimization) {
    if (text.includes(indicator)) { dvScore += 1; minimizationScore++; }
  }
  for (const indicator of dvIndicators.physical) {
    if (text.includes(indicator)) { dvScore += 3; physicalScore++; }
  }
  for (const indicator of dvIndicators.coercive) {
    if (text.includes(indicator)) { dvScore += 3; controlScore++; }
  }
  for (const indicator of dvIndicators.children) {
    if (text.includes(indicator)) { childrenInvolved = true; }
  }
  
  // Determine isolation level
  let isolationLevel: RuralContext["isolationLevel"] = "moderate";
  if (ruralScore > 6) isolationLevel = "extreme";
  else if (ruralScore > 3) isolationLevel = "high";
  
  // Determine access to services
  let accessToServices: RuralContext["accessToServices"] = "limited";
  if (text.includes("no service") || text.includes("hours away") || text.includes("no way")) {
    accessToServices = "none";
  } else if (text.includes("far") || text.includes("difficult")) {
    accessToServices = "very_limited";
  }
  
  // Determine if recognizes abuse (minimization suggests may not)
  const recognizesAbuse = minimizationScore < 2 && (physicalScore > 0 || controlScore > 2);
  
  // Safety constraints
  const safetyConstraints: string[] = [];
  if (text.includes("phone") && (text.includes("check") || text.includes("monitor"))) {
    safetyConstraints.push("phone_monitored");
  }
  if (text.includes("car") && (text.includes("won't let") || text.includes("disabled") || text.includes("keys"))) {
    safetyConstraints.push("no_transportation");
  }
  if (childrenInvolved) {
    safetyConstraints.push("children_involved");
  }
  if (text.includes("money") && (text.includes("control") || text.includes("allowance") || text.includes("takes"))) {
    safetyConstraints.push("financial_control");
  }
  
  // Cultural factors
  const culturalFactors: string[] = [];
  if (text.includes("church") || text.includes("congregation") || text.includes("pastor")) {
    culturalFactors.push("religious_community");
  }
  if (text.includes("family name") || text.includes("reputation") || text.includes("shame")) {
    culturalFactors.push("family_honor");
  }
  if (text.includes("traditional") || text.includes("old fashioned") || text.includes("supposed to")) {
    culturalFactors.push("traditional_values");
  }
  
  return {
    isRural: ruralScore > 2,
    isolationLevel,
    accessToServices,
    domesticViolenceRisk: dvScore > 4,
    recognizesAbuse,
    safetyConstraints,
    culturalFactors
  };
}

/**
 * Get tailored intervention for rural context
 */
export function getRuralIntervention(context: RuralContext, gender?: string): RuralIntervention {
  const intervention: RuralIntervention = {
    category: context.domesticViolenceRisk ? "rural_dv" : "rural_isolation",
    validation: "",
    psychoeducation: [],
    safetyPlanning: [],
    resources: [],
    selfCareStrategies: [],
    communityBuilding: []
  };
  
  // Validation
  if (context.domesticViolenceRisk) {
    if (!context.recognizesAbuse) {
      intervention.validation = "It sounds like you're in a really difficult situation. What you're describing - the control, the isolation, the fear - those are serious. You deserve to feel safe and free.";
      intervention.psychoeducation = abuseEducation.whatIsAbuse.slice(0, 3);
      intervention.psychoeducation.push(...abuseEducation.whyStay.slice(0, 2));
    } else {
      intervention.validation = "I hear you. Being in this situation, especially in a rural area where help feels so far away, is incredibly hard. Your feelings are valid, and your safety matters.";
      intervention.psychoeducation = abuseEducation.ruralChallenges;
    }
    
    // Safety planning based on constraints
    intervention.safetyPlanning = [...ruralSafetyPlanning.immediate];
    
    if (context.safetyConstraints.includes("phone_monitored")) {
      intervention.safetyPlanning.push(...ruralSafetyPlanning.communication);
    }
    if (context.safetyConstraints.includes("no_transportation")) {
      intervention.safetyPlanning.push(...ruralSafetyPlanning.transportation);
    }
    if (context.safetyConstraints.includes("children_involved")) {
      intervention.safetyPlanning.push(...ruralSafetyPlanning.children);
    }
    if (context.safetyConstraints.includes("financial_control")) {
      intervention.safetyPlanning.push(...ruralSafetyPlanning.financial);
    }
    
    // Resources
    intervention.resources = ruralResources.filter(r => 
      r.type === "text" || r.type === "hotline" || r.name.includes("Domestic") || r.name.includes("Women")
    );
    
  } else {
    // Rural isolation without DV
    intervention.validation = "Living in a rural area can feel incredibly isolating, especially when you're struggling. The distance from services and support is a real barrier, and your feelings about that are valid.";
    
    intervention.psychoeducation = [
      "Isolation can intensify difficult emotions. It's not weakness - it's a natural response to being far from connection.",
      "Many people in rural areas face similar challenges accessing mental health support. You're not alone in this.",
      "Telehealth has expanded options significantly. Many therapists now offer video sessions."
    ];
    
    intervention.resources = ruralResources.filter(r => 
      r.type === "telehealth" || r.type === "online" || r.name.includes("Farm") || r.name.includes("Rural")
    );
    
    intervention.selfCareStrategies = [
      "Create a daily routine that includes time outside, even briefly.",
      "Use video calls to maintain connections with friends and family.",
      "Consider online support groups for people in similar situations.",
      "Physical activity, even walking on your property, can help regulate mood.",
      "Journaling can provide an outlet when there's no one to talk to."
    ];
    
    intervention.communityBuilding = [
      "Look for online communities related to your interests or situation.",
      "Some churches and community centers offer support groups.",
      "Agricultural extension offices sometimes have mental health resources.",
      "Consider whether there's anyone in your area you could reach out to, even casually."
    ];
  }
  
  return intervention;
}

/**
 * Format rural intervention for LLM context
 */
export function formatRuralInterventionForPrompt(
  context: RuralContext, 
  intervention: RuralIntervention
): string {
  if (!context.isRural && !context.domesticViolenceRisk) return "";
  
  let output = "\n\n[RURAL/ISOLATION CONTEXT DETECTED]\n";
  output += `Isolation level: ${context.isolationLevel}\n`;
  output += `Access to services: ${context.accessToServices}\n`;
  
  if (context.domesticViolenceRisk) {
    output += `\n⚠️ DOMESTIC VIOLENCE INDICATORS DETECTED\n`;
    output += `Recognizes abuse: ${context.recognizesAbuse ? "Yes" : "May not fully recognize"}\n`;
    output += `Safety constraints: ${context.safetyConstraints.join(", ") || "None identified"}\n`;
    
    if (!context.recognizesAbuse) {
      output += "\nIMPORTANT: User may not recognize their situation as abuse. Be gentle. Don't label it directly.\n";
      output += "Instead, reflect what they're describing and gently provide information about healthy relationships.\n";
    }
  }
  
  output += `\nVALIDATION TO USE: "${intervention.validation}"\n`;
  
  if (intervention.psychoeducation.length > 0) {
    output += "\nPSYCHOEDUCATION (share gently, one at a time):\n";
    for (const edu of intervention.psychoeducation.slice(0, 3)) {
      output += `- ${edu}\n`;
    }
  }
  
  if (intervention.safetyPlanning.length > 0 && context.domesticViolenceRisk) {
    output += "\nSAFETY PLANNING (offer when appropriate):\n";
    for (const step of intervention.safetyPlanning.slice(0, 5)) {
      output += `- ${step}\n`;
    }
  }
  
  if (intervention.resources.length > 0) {
    output += "\nRESOURCES TO SHARE:\n";
    for (const resource of intervention.resources.slice(0, 4)) {
      output += `- ${resource.name}: ${resource.contact}`;
      if (resource.safetyNote) output += ` (${resource.safetyNote})`;
      output += "\n";
    }
  }
  
  return output;
}
