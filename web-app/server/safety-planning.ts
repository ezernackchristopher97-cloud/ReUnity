/**
 * Crisis Safety Planning Wizard
 * 
 * A step-by-step guided tool for users in domestic violence situations
 * to create a personalized, encrypted escape plan.
 * 
 * Designed with rural isolation in mind - addresses transportation,
 * distance, limited resources, and technology monitoring.
 */

export interface SafetyPlanStep {
  id: string;
  title: string;
  description: string;
  questions: SafetyQuestion[];
  ruralConsiderations?: string[];
  resources?: SafetyResource[];
  warnings?: string[];
}

export interface SafetyQuestion {
  id: string;
  question: string;
  type: "text" | "multitext" | "checklist" | "yesno" | "select";
  options?: string[];
  placeholder?: string;
  sensitive?: boolean;  // Extra encryption for this field
  helpText?: string;
}

export interface SafetyResource {
  name: string;
  description: string;
  phone?: string;
  text?: string;
  website?: string;
  ruralFriendly: boolean;
}

export interface SafetyPlan {
  id: string;
  userId: string;
  createdAt: Date;
  updatedAt: Date;
  encryptedData: string;  // All sensitive data encrypted
  completedSteps: string[];
  isComplete: boolean;
}

export interface SafetyPlanData {
  // Step 1: Safe Contacts
  safeContacts: {
    name: string;
    relationship: string;
    phone: string;
    codeWord: string;
    knowsSituation: boolean;
  }[];
  
  // Step 2: Warning Signs
  warningSigns: string[];
  dangerousTimePatterns: string[];
  
  // Step 3: Safe Locations
  safeLocations: {
    name: string;
    address: string;
    distance: string;
    hasKey: boolean;
    knowsSituation: boolean;
  }[];
  nearestShelter?: string;
  
  // Step 4: Emergency Bag
  emergencyBagLocation: string;
  emergencyBagItems: string[];
  
  // Step 5: Documents
  documentsSecured: string[];
  documentLocation: string;
  
  // Step 6: Financial Safety
  hasHiddenMoney: boolean;
  hiddenMoneyLocation?: string;
  hasSecretAccount: boolean;
  financialSteps: string[];
  
  // Step 7: Technology Safety
  phoneMonitored: boolean;
  locationTracked: boolean;
  socialMediaMonitored: boolean;
  technologySafetySteps: string[];
  
  // Step 8: Children
  hasChildren: boolean;
  childrenNames?: string[];
  schoolInfo?: string;
  custodyConsiderations?: string;
  childSafetySteps?: string[];
  
  // Step 9: Pets
  hasPets: boolean;
  petInfo?: string;
  petSafetyPlan?: string;
  
  // Step 10: Exit Strategy
  bestTimeToLeave: string;
  transportationPlan: string;
  firstDestination: string;
  backupPlan: string;
}

// The wizard steps
export const safetyPlanSteps: SafetyPlanStep[] = [
  {
    id: "safe_contacts",
    title: "Safe Contacts & Code Words",
    description: "Identify people you can trust and create secret signals to communicate danger without alerting your abuser.",
    questions: [
      {
        id: "contact_name",
        question: "Who is someone you trust completely?",
        type: "text",
        placeholder: "Name",
        helpText: "This could be a friend, family member, coworker, or neighbor"
      },
      {
        id: "contact_relationship",
        question: "What is their relationship to you?",
        type: "text",
        placeholder: "Friend, sister, coworker, etc."
      },
      {
        id: "contact_phone",
        question: "What is their phone number?",
        type: "text",
        placeholder: "Phone number",
        sensitive: true
      },
      {
        id: "code_word",
        question: "Create a code word or phrase that signals you need help",
        type: "text",
        placeholder: "e.g., 'I need to pick up milk' or 'red'",
        helpText: "Choose something that sounds normal in conversation but your contact will recognize as a distress signal"
      },
      {
        id: "knows_situation",
        question: "Does this person know about your situation?",
        type: "yesno"
      }
    ],
    ruralConsiderations: [
      "In rural areas, neighbors may be far away - consider contacts who can reach you quickly",
      "If phone service is unreliable, establish a check-in schedule",
      "Consider having a contact who can call authorities if they don't hear from you"
    ],
    resources: [
      {
        name: "National Domestic Violence Hotline",
        description: "24/7 support, safety planning, and local resources",
        phone: "1-800-799-7233",
        text: "Text START to 88788",
        ruralFriendly: true
      }
    ]
  },
  
  {
    id: "warning_signs",
    title: "Recognizing Warning Signs",
    description: "Understanding patterns helps you anticipate danger and act before situations escalate.",
    questions: [
      {
        id: "warning_signs",
        question: "What signs tell you violence may be coming?",
        type: "multitext",
        placeholder: "e.g., drinking, certain tone of voice, specific topics",
        helpText: "Think about what happens before violent episodes"
      },
      {
        id: "dangerous_times",
        question: "Are there times that are more dangerous?",
        type: "multitext",
        placeholder: "e.g., after work, weekends, holidays, payday",
        helpText: "Identifying patterns helps you plan safer times to leave"
      },
      {
        id: "escalation_pattern",
        question: "How does the abuse typically escalate?",
        type: "text",
        placeholder: "Describe the pattern you've noticed"
      }
    ],
    warnings: [
      "Trust your instincts - if something feels wrong, it probably is",
      "Escalation often happens when the abuser senses you might leave",
      "The most dangerous time is often when leaving - plan carefully"
    ]
  },
  
  {
    id: "safe_locations",
    title: "Safe Places to Go",
    description: "Know where you can go at any time, day or night, if you need to leave quickly.",
    questions: [
      {
        id: "safe_place_name",
        question: "Where could you go if you needed to leave?",
        type: "text",
        placeholder: "Friend's house, family member, shelter"
      },
      {
        id: "safe_place_address",
        question: "What is the address?",
        type: "text",
        placeholder: "Address",
        sensitive: true
      },
      {
        id: "safe_place_distance",
        question: "How far away is it?",
        type: "text",
        placeholder: "e.g., 5 miles, 2 hours"
      },
      {
        id: "has_key",
        question: "Do you have a key or can you get in anytime?",
        type: "yesno"
      },
      {
        id: "backup_location",
        question: "What is your backup location if the first isn't available?",
        type: "text",
        placeholder: "Second safe place"
      }
    ],
    ruralConsiderations: [
      "In isolated areas, distance is a major factor - know multiple routes",
      "Consider locations along your route where you could stop if needed",
      "Gas stations, hospitals, and police stations are open 24/7",
      "If you don't have transportation, identify who could pick you up"
    ],
    resources: [
      {
        name: "DomesticShelters.org",
        description: "Find shelters near you, including rural areas",
        website: "https://www.domesticshelters.org",
        ruralFriendly: true
      }
    ]
  },
  
  {
    id: "emergency_bag",
    title: "Emergency Bag",
    description: "Prepare a bag with essentials that you can grab quickly. Keep it hidden or at a trusted person's home.",
    questions: [
      {
        id: "bag_location",
        question: "Where will you keep your emergency bag?",
        type: "text",
        placeholder: "e.g., car trunk, friend's house, work locker",
        sensitive: true,
        helpText: "Choose somewhere your abuser won't find it"
      },
      {
        id: "bag_items",
        question: "Check off items you have or will gather:",
        type: "checklist",
        options: [
          "Cash (small bills)",
          "Change of clothes",
          "Toiletries",
          "Phone charger",
          "Medications",
          "Copies of important documents",
          "Keys (car, house, work)",
          "Children's items (if applicable)",
          "Pet supplies (if applicable)",
          "Comfort item",
          "Snacks and water",
          "Flashlight",
          "First aid kit"
        ]
      }
    ],
    ruralConsiderations: [
      "Include extra gas money - distances are longer",
      "Pack for weather - you may be traveling in harsh conditions",
      "Include a paper map in case phone service is unavailable",
      "Consider keeping supplies in your car if safe"
    ]
  },
  
  {
    id: "documents",
    title: "Important Documents",
    description: "Having your documents makes starting over much easier. Secure copies now if possible.",
    questions: [
      {
        id: "documents_secured",
        question: "Which documents do you have access to or copies of?",
        type: "checklist",
        options: [
          "Driver's license / ID",
          "Social Security card",
          "Birth certificate",
          "Passport",
          "Children's birth certificates",
          "Children's Social Security cards",
          "Marriage certificate",
          "Protective orders / legal documents",
          "Medical records",
          "Insurance cards",
          "Bank statements",
          "Pay stubs",
          "Lease / mortgage documents",
          "Car title / registration",
          "Photos documenting abuse",
          "Immigration documents (if applicable)"
        ]
      },
      {
        id: "document_location",
        question: "Where are these documents stored safely?",
        type: "text",
        placeholder: "e.g., safe deposit box, friend's house",
        sensitive: true
      },
      {
        id: "missing_documents",
        question: "Which documents do you still need to secure?",
        type: "multitext",
        placeholder: "List documents you need to get copies of"
      }
    ],
    warnings: [
      "If you can't get originals, copies are still helpful",
      "Take photos of documents and email them to a secure account",
      "Don't let your abuser know you're gathering documents"
    ]
  },
  
  {
    id: "financial_safety",
    title: "Financial Safety",
    description: "Financial abuse is common. Taking steps now can help you have resources when you leave.",
    questions: [
      {
        id: "financial_control",
        question: "Does your abuser control the money?",
        type: "yesno"
      },
      {
        id: "hidden_money",
        question: "Have you been able to set aside any money?",
        type: "yesno"
      },
      {
        id: "hidden_money_location",
        question: "Where is it hidden?",
        type: "text",
        placeholder: "Location",
        sensitive: true
      },
      {
        id: "secret_account",
        question: "Do you have access to a bank account your abuser doesn't know about?",
        type: "yesno"
      },
      {
        id: "financial_steps",
        question: "What financial steps can you take?",
        type: "checklist",
        options: [
          "Open a secret bank account (use a friend's address)",
          "Set aside small amounts of cash",
          "Get a P.O. Box for mail",
          "Know your credit score",
          "Document shared assets",
          "Keep records of income and expenses",
          "Apply for benefits if eligible",
          "Research emergency financial assistance"
        ]
      }
    ],
    ruralConsiderations: [
      "In small towns, be careful which bank you use - word travels",
      "Consider a bank in a nearby town where you're not known",
      "Online banking can be done privately if you clear browser history"
    ],
    resources: [
      {
        name: "NNEDV Financial Safety",
        description: "Resources for financial safety planning",
        website: "https://nnedv.org/content/financial-safety/",
        ruralFriendly: true
      }
    ]
  },
  
  {
    id: "technology_safety",
    title: "Technology Safety",
    description: "Abusers often use technology to monitor and control. Understanding this helps you stay safe.",
    questions: [
      {
        id: "phone_monitored",
        question: "Do you think your phone is being monitored?",
        type: "yesno",
        helpText: "Signs include: they know things you only said on the phone, apps you didn't install, battery draining quickly"
      },
      {
        id: "location_tracked",
        question: "Is your location being tracked?",
        type: "yesno",
        helpText: "Check for tracking apps, AirTags, or GPS devices in your car"
      },
      {
        id: "social_media_monitored",
        question: "Does your abuser monitor your social media or email?",
        type: "yesno"
      },
      {
        id: "tech_safety_steps",
        question: "Technology safety steps to consider:",
        type: "checklist",
        options: [
          "Use a safer device (library computer, friend's phone)",
          "Create new email account they don't know about",
          "Use private/incognito browsing",
          "Clear browser history after searching for help",
          "Check phone for tracking apps",
          "Check car for GPS trackers",
          "Change passwords from a safe device",
          "Turn off location sharing",
          "Be careful what you post on social media",
          "Use a code with friends for texts/calls"
        ]
      }
    ],
    warnings: [
      "If your phone is monitored, use a different device to search for help",
      "Libraries have free computers you can use privately",
      "Don't change passwords from a monitored device - it alerts them",
      "Consider getting a prepaid phone they don't know about"
    ],
    resources: [
      {
        name: "Safety Net",
        description: "Technology safety resources from NNEDV",
        website: "https://www.techsafety.org",
        ruralFriendly: true
      }
    ]
  },
  
  {
    id: "children",
    title: "Children's Safety",
    description: "If you have children, their safety is part of your plan. This section helps you prepare.",
    questions: [
      {
        id: "has_children",
        question: "Do you have children?",
        type: "yesno"
      },
      {
        id: "children_names",
        question: "Children's names and ages:",
        type: "multitext",
        placeholder: "Name, age"
      },
      {
        id: "school_info",
        question: "School/daycare information:",
        type: "text",
        placeholder: "School name, address, contact"
      },
      {
        id: "custody_concerns",
        question: "Do you have custody concerns?",
        type: "text",
        placeholder: "Describe any custody-related concerns"
      },
      {
        id: "child_safety_steps",
        question: "Steps for children's safety:",
        type: "checklist",
        options: [
          "Teach children how to call 911",
          "Create a code word children understand",
          "Identify a safe room in the house",
          "Tell children it's not their fault",
          "Pack comfort items for children",
          "Inform school/daycare of situation",
          "Get copies of children's documents",
          "Know children's schedules",
          "Plan for children's medications/needs"
        ]
      }
    ],
    warnings: [
      "Never leave children alone with the abuser if you fear for their safety",
      "Document any abuse or threats toward children",
      "Courts take child safety seriously - document everything"
    ]
  },
  
  {
    id: "pets",
    title: "Pet Safety",
    description: "Many people delay leaving because of pets. There are options to keep them safe too.",
    questions: [
      {
        id: "has_pets",
        question: "Do you have pets?",
        type: "yesno"
      },
      {
        id: "pet_info",
        question: "Pet information (type, name):",
        type: "text",
        placeholder: "e.g., Dog, Max"
      },
      {
        id: "pet_threatened",
        question: "Has your abuser threatened or harmed your pets?",
        type: "yesno"
      },
      {
        id: "pet_plan",
        question: "Plan for your pet:",
        type: "select",
        options: [
          "Take pet with me",
          "Friend/family will care for pet",
          "Use a pet-friendly shelter",
          "Use Safe Place for Pets program",
          "Board pet temporarily",
          "Still figuring this out"
        ]
      }
    ],
    ruralConsiderations: [
      "Farm animals require special planning - contact local agricultural extension",
      "Some shelters now accept pets or have partnerships with foster programs",
      "Veterinarians may know of temporary foster options"
    ],
    resources: [
      {
        name: "Safe Place for Pets",
        description: "Program helping DV survivors with pet safety",
        website: "https://redrover.org/relief/safe-place/",
        ruralFriendly: true
      }
    ]
  },
  
  {
    id: "exit_strategy",
    title: "Your Exit Strategy",
    description: "This is your plan for when you're ready to leave. Review and update it as needed.",
    questions: [
      {
        id: "best_time",
        question: "When is the safest time for you to leave?",
        type: "text",
        placeholder: "e.g., when they're at work, after they fall asleep",
        helpText: "Consider their schedule and when you'll have the most time"
      },
      {
        id: "transportation",
        question: "How will you leave?",
        type: "select",
        options: [
          "Drive my own car",
          "Someone will pick me up",
          "Public transportation",
          "Taxi/rideshare",
          "Walk to a safe location",
          "Other"
        ]
      },
      {
        id: "first_destination",
        question: "Where will you go first?",
        type: "text",
        placeholder: "Your first safe destination",
        sensitive: true
      },
      {
        id: "backup_plan",
        question: "What is your backup plan if the first doesn't work?",
        type: "text",
        placeholder: "Alternative plan"
      },
      {
        id: "who_to_call",
        question: "Who will you call when you're safe?",
        type: "text",
        placeholder: "Name and number"
      },
      {
        id: "final_checklist",
        question: "Before leaving, remember to:",
        type: "checklist",
        options: [
          "Grab emergency bag",
          "Take important documents",
          "Take phone/charger",
          "Take keys",
          "Take medications",
          "Take children (if applicable)",
          "Take pets (if applicable)",
          "Leave when it's safest",
          "Go to planned safe location",
          "Call safe contact when you arrive"
        ]
      }
    ],
    ruralConsiderations: [
      "Plan your route carefully - know where gas stations and safe stops are",
      "Have a backup route in case roads are blocked or watched",
      "Consider the weather and road conditions",
      "If you don't have a car, arrange transportation in advance",
      "Know the distance and how long it will take"
    ],
    warnings: [
      "The most dangerous time is when leaving - be very careful",
      "Don't tell your abuser you're leaving",
      "If possible, leave when they're not home",
      "Trust your instincts about timing"
    ]
  }
];

/**
 * Get a specific step by ID
 */
export function getStep(stepId: string): SafetyPlanStep | undefined {
  return safetyPlanSteps.find(s => s.id === stepId);
}

/**
 * Get all steps
 */
export function getAllSteps(): SafetyPlanStep[] {
  return safetyPlanSteps;
}

/**
 * Get step by index
 */
export function getStepByIndex(index: number): SafetyPlanStep | undefined {
  return safetyPlanSteps[index];
}

/**
 * Calculate completion percentage
 */
export function calculateCompletion(completedSteps: string[]): number {
  return (completedSteps.length / safetyPlanSteps.length) * 100;
}

/**
 * Get next incomplete step
 */
export function getNextIncompleteStep(completedSteps: string[]): SafetyPlanStep | undefined {
  return safetyPlanSteps.find(s => !completedSteps.includes(s.id));
}

/**
 * Format safety plan for export (non-sensitive summary)
 */
export function formatPlanSummary(data: Partial<SafetyPlanData>): string {
  let summary = "SAFETY PLAN SUMMARY\n";
  summary += "===================\n\n";
  
  summary += "This is your personal safety plan. Keep it somewhere safe where your abuser cannot find it.\n\n";
  
  if (data.safeContacts && data.safeContacts.length > 0) {
    summary += "SAFE CONTACTS:\n";
    for (const contact of data.safeContacts) {
      summary += `- ${contact.name} (${contact.relationship})\n`;
      summary += `  Code word: ${contact.codeWord}\n`;
    }
    summary += "\n";
  }
  
  if (data.warningSigns && data.warningSigns.length > 0) {
    summary += "WARNING SIGNS TO WATCH FOR:\n";
    for (const sign of data.warningSigns) {
      summary += `- ${sign}\n`;
    }
    summary += "\n";
  }
  
  if (data.safeLocations && data.safeLocations.length > 0) {
    summary += "SAFE PLACES:\n";
    for (const loc of data.safeLocations) {
      summary += `- ${loc.name} (${loc.distance} away)\n`;
    }
    summary += "\n";
  }
  
  summary += "EMERGENCY NUMBERS:\n";
  summary += "- National DV Hotline: 1-800-799-7233\n";
  summary += "- Text START to 88788\n";
  summary += "- 911 for emergencies\n\n";
  
  summary += "Remember: You deserve to be safe. This is not your fault.\n";
  
  return summary;
}

/**
 * Get crisis resources for immediate danger
 */
export function getImmediateDangerResources(): SafetyResource[] {
  return [
    {
      name: "911",
      description: "For immediate danger, call 911",
      phone: "911",
      ruralFriendly: true
    },
    {
      name: "National Domestic Violence Hotline",
      description: "24/7 confidential support",
      phone: "1-800-799-7233",
      text: "Text START to 88788",
      ruralFriendly: true
    },
    {
      name: "Crisis Text Line",
      description: "Text-based crisis support",
      text: "Text HOME to 741741",
      ruralFriendly: true
    }
  ];
}
